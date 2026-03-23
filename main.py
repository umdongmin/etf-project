import os
import sys

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Streamlit secrets.toml 로드 (Flask에서도 사용 가능하도록)
try:
    import toml
    secrets_path = os.path.join(os.path.dirname(__file__), '.streamlit', 'secrets.toml')
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r', encoding='utf-8') as f:
            secrets = toml.load(f)
            for key, value in secrets.items():
                if key not in os.environ:  # 환경변수가 없으면 secrets에서 로드
                    os.environ[key] = str(value)
        print("✅ Streamlit Secrets Loaded", flush=True)
except (ImportError, FileNotFoundError):
    pass

# 로그 즉시 출력 설정 (배포 시 로그 확인의 핵심)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("--- 🚀 TQQQ Alert Bot Production Starting ---", flush=True)

# [1단계] 스트림릿 모듈 속이기 (core 임포트 시 충돌 방지)
class DummyDict(dict):
    """딕셔너리처럼 동작하는 더미 session_state"""
    def get(self, key, default=None):
        return super().get(key, default)

class Dummy:
    def __init__(self):
        self.session_state = DummyDict()  # session_state는 딕셔너리

    def __getattr__(self, name):
        if name == 'session_state':
            return self.session_state
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        return lambda *args, **kwargs: Dummy()

    def __call__(self, *args, **kwargs):
        return Dummy()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Initialized")

try:
    try:
        from flask import Flask
        app = Flask(__name__)
    except ImportError:
        app = None
        print("⚠️ Flask 미설치 — dry-run 모드로만 동작합니다.", flush=True)

    from core.bot_service import run_tqqq_bot
    from core.data_updater import run_daily_update

    def handle_request():
        import datetime
        from flask import request

        # 한국 시간(KST) 구하기
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        hour = now_kst.hour
        year = now_kst.year

        # 영구 서머타임 자동 계산 (3월 둘째 일요일 ~ 11월 첫째 일요일)
        def get_dst_range(y):
            first_march = datetime.datetime(y, 3, 1)
            dst_start_day = first_march + datetime.timedelta(days=(6 - first_march.weekday() + 7) % 7 + 7)
            first_nov = datetime.datetime(y, 11, 1)
            dst_end_day = first_nov + datetime.timedelta(days=(6 - first_nov.weekday() + 7) % 7)
            return dst_start_day.replace(hour=16), dst_end_day.replace(hour=15)

        dst_start, dst_end = get_dst_range(year)
        is_dst = dst_start <= now_kst.replace(tzinfo=None) < dst_end

        # 스마트 시간 필터 (중복 알림 방지/자동 계절 대응)
        is_scheduler = "Google-Cloud-Scheduler" in request.headers.get("User-Agent", "")
        if is_scheduler:
            if is_dst and hour == 5:
                return "SKIP: Summer time, already handled at 04:40", 200
            if not is_dst and hour == 4:
                return "SKIP: Winter time, waiting for 05:40", 200

        print(f"✅ Executing Request (Hour:{hour}, DST:{is_dst}, Scheduler:{is_scheduler})", flush=True)

        # .env에서 TELEGRAM_TOKEN으로 설정된 경우 호환성 처리
        token   = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        # 데이터 업데이트 + GitHub push (항상 실행)
        run_daily_update()

        # Telegram 알림 (신호 있을 때만 발송)
        if token and chat_id:
            run_tqqq_bot(token, chat_id)
            return "SUCCESS: Daily update & report done!", 200
        else:
            return "SUCCESS: Daily update done (Telegram keys missing).", 200

    # ── Telegram Webhook: /approve, /reject, /kill, /resume, /status ────

    def telegram_webhook():
        """Telegram Bot Webhook 수신 — KIS 주문 승인/거부/긴급정지"""
        from flask import request as flask_request
        data = flask_request.get_json(silent=True)
        if not data:
            return 'OK', 200

        message = data.get('message', {})
        text = (message.get('text') or '').strip()
        incoming_chat_id = str(message.get('chat', {}).get('id', ''))

        # 인가된 사용자만 허용
        allowed_chat_id = os.environ.get('TELEGRAM_CHAT_ID', '')
        if incoming_chat_id != allowed_chat_id:
            print(f"[Telegram Webhook] Unauthorized chat_id: {incoming_chat_id}", flush=True)
            return 'OK', 200

        token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")

        def _send_reply(reply_text):
            if token and incoming_chat_id:
                try:
                    import requests as req
                    req.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={"chat_id": incoming_chat_id, "text": reply_text, "parse_mode": "Markdown"},
                        timeout=10,
                    )
                except Exception as e:
                    print(f"[Telegram Reply Error] {e}", flush=True)

        try:
            if text.startswith('/approve'):
                parts = text.split()
                if len(parts) < 2:
                    _send_reply("사용법: /approve <승인코드>")
                    return 'OK', 200
                code = parts[1].strip()
                from core.kis_trader import KISTrader
                trader = KISTrader()
                result = trader.execute_approved_order(code)
                icon = '✅' if result['success'] else '❌'
                _send_reply(f"{icon} {result['message']}")

            elif text.startswith('/reject'):
                parts = text.split()
                if len(parts) < 2:
                    _send_reply("사용법: /reject <승인코드>")
                    return 'OK', 200
                code = parts[1].strip()
                from core.kis_trader import KISTrader
                result = KISTrader.reject_order(code)
                icon = '✅' if result['success'] else '❌'
                _send_reply(f"{icon} {result['message']}")

            elif text.startswith('/kill'):
                from core.storage import AssetStorage, OrderStorage
                accounts = AssetStorage.list_accounts()
                halted = []
                for acc in accounts:
                    OrderStorage.set_trading_halt(acc['id'], True)
                    halted.append(acc['name'])
                _send_reply(f"🛑 *거래 긴급 중단*\n모든 계좌 거래 정지됨: {', '.join(halted)}")

            elif text.startswith('/resume'):
                from core.storage import AssetStorage, OrderStorage
                accounts = AssetStorage.list_accounts()
                resumed = []
                for acc in accounts:
                    OrderStorage.set_trading_halt(acc['id'], False)
                    resumed.append(acc['name'])
                _send_reply(f"▶️ *거래 재개*\n모든 계좌 거래 활성화: {', '.join(resumed)}")

            elif text.startswith('/status'):
                import datetime as dt
                mode = os.environ.get('TRADING_MODE', 'disabled')
                kill = os.environ.get('TRADING_KILL_SWITCH', 'false')
                lines = [
                    "━━━━━━━━━━━━━━━━━━━━━━",
                    f"📊 *KIS 거래 상태*",
                    f"  Mode: `{mode}`",
                    f"  Kill Switch: `{kill}`",
                ]
                # 대기 중 승인 목록
                from core.storage import OrderStorage
                pending = OrderStorage.list_pending_approvals('pending')
                if pending:
                    lines.append(f"\n🔔 *대기 중 승인: {len(pending)}건*")
                    for p in pending[:5]:
                        icon = '🟢' if p['side'] == 'buy' else '🔴'
                        lines.append(f"  {icon} {p['ticker']} {p['quantity']}주 — /approve {p['approval_code']}")
                else:
                    lines.append("\n✅ 대기 중 승인 없음")

                # KIS 잔고 조회 시도
                if mode in ('paper', 'live'):
                    try:
                        from core.kis_client import KISClient
                        client = KISClient(mode)
                        balance = client.get_balance()
                        if balance:
                            lines.append(f"\n💰 *KIS 잔고 ({mode})*")
                            for ticker, info in balance.items():
                                lines.append(f"  {ticker}: {info['qty']}주 (${info['eval_amount']:,.0f})")
                    except Exception as e:
                        lines.append(f"\n⚠️ KIS 잔고 조회 실패: {e}")

                lines.append("━━━━━━━━━━━━━━━━━━━━━━")
                _send_reply('\n'.join(lines))

        except Exception as e:
            print(f"[Telegram Webhook Error] {e}", flush=True)
            import traceback
            traceback.print_exc()
            _send_reply(f"❌ 오류 발생: {str(e)[:200]}")

        return 'OK', 200

    if app:
        app.add_url_rule('/', 'handle_request', handle_request, methods=['GET', 'POST'])
        app.add_url_rule('/telegram-webhook', 'telegram_webhook', telegram_webhook, methods=['POST'])

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port)

except Exception as e:
    print(f"!!! [FATAL BOOT ERROR] !!!\n{e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
