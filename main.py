import os
import sys
import time

# � 로그 즉시 출력 설정 (배포 시 로그 확인의 핵심)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("--- 🚀 TQQQ Alert Bot Production Starting ---", flush=True)

# 🌟 [1단계] 스트림릿 모듈 속이기 (core 임포트 시 충돌 방지)
# 프로젝트 내부에서 streamlit을 불러와도 에러가 나지 않게 "바보" 객체로 대체합니다.
class Dummy:
    def __getattr__(self, name):
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        return lambda *args, **kwargs: Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *args): pass

sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Initialized")

try:
    from flask import Flask
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def handle_request():
        import datetime
        from flask import request
        
        # 한국 시간(KST) 구하기
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        hour = now_kst.hour
        year = now_kst.year
        
        # 🌟 영구 서머타임 자동 계산 (3월 둘째 일요일 ~ 11월 첫째 일요일)
        def get_dst_range(y):
            # 3월 둘째 일요일 (KST 기준 일요일 오후 4시 전환)
            first_march = datetime.datetime(y, 3, 1)
            dst_start_day = first_march + datetime.timedelta(days=(6 - first_march.weekday() + 7) % 7 + 7)
            # 11월 첫째 일요일 (KST 기준 일요일 오후 3시 전환)
            first_nov = datetime.datetime(y, 11, 1)
            dst_end_day = first_nov + datetime.timedelta(days=(6 - first_nov.weekday() + 7) % 7)
            return dst_start_day.replace(hour=16), dst_end_day.replace(hour=15)

        dst_start, dst_end = get_dst_range(year)
        is_dst = dst_start <= now_kst.replace(tzinfo=None) < dst_end

        # 🌟 스마트 시간 필터 (중복 알림 방지/자동 계절 대응)
        is_scheduler = "Google-Cloud-Scheduler" in request.headers.get("User-Agent", "")
        if is_scheduler:
            if is_dst and hour == 5:
                return "SKIP: Summer time, already handled at 04:40", 200
            if not is_dst and hour == 4:
                return "SKIP: Winter time, waiting for 05:40", 200

        print(f"✅ Executing Request (Hour:{hour}, DST:{is_dst}, Scheduler:{is_scheduler})", flush=True)
        
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        if token and chat_id:
            run_tqqq_bot(token, chat_id)
            return "SUCCESS: Daily report sent!", 200
        else:
            return "ERROR: Missing API Keys.", 500

    def run_tqqq_bot(token, chat_id):
        """무거운 라이브러리 임포트와 실제 데이터 분석 수행"""
        try:
            print("🚀 [Worker] Loading data and running strategy...", flush=True)
            import json, datetime, requests
            # 이 시점에 core 모듈을 불러옵니다.
            from core.data import DataService
            from core.engine import StrategyEngine

            # 1. 파일 경로 (Cloud Run 환경)
            project_root = os.path.dirname(os.path.abspath(__file__))
            strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
            
            if not os.path.exists(strat_path):
                print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다 -> {strat_path}", flush=True)
                return

            with open(strat_path, 'r', encoding='utf-8') as f:
                strat = json.load(f)

            # 2. 분석 실행 (실시간 데이터 수집 및 시그널 계산)
            sim_start_date = datetime.date(2026, 1, 1)
            fetch_start_date = (sim_start_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            
            print(f"📡 분석 대상: {fetch_start_date} ~ 오늘")
            
            # 2-1. 데이터 수집 (PCCR 제거 버전 적용)
            data_raw, fg, vix, _, _, _ = DataService.fetch_live_data(start_date=fetch_start_date)
            all_tickers = list(data_raw.keys())
            current_prices = DataService.fetch_current_prices(all_tickers)
            
            # 2-2. 🌟 가상 종가 주입 (UI의 '실시간 시그널 프리뷰'와 동일한 로직)
            data = DataService.inject_virtual_close(data_raw, current_prices)

            gh, _, events = StrategyEngine.run_golden_strategy(
                data, fg, vix, 
                strat.get('leverage_asset', 'TQQQ'), 
                strat.get('base_asset', 'QQQ'),
                0.0, 
                sim_start_date, 
                datetime.date.today(), 
                strat, 
                strat.get('trade_at', '종가'), 
                salt='vFinal_Final_Live'
            )

            # 3. 알림 전송
            if not gh.empty:
                latest = gh.iloc[-1]
                signal = str(latest.get('Trade_Label', ''))
                
                # 🌟 [수정] 매수 또는 매도 신호가 있을 때만 알람 발송
                # '진행중', '중립' 등 액션이 필요 없는 날은 조용히 넘어갑니다.
                if "매수" in signal or "매도" in signal:
                    msg = f"🔔 **TQQQ 매매 시그널 발생!**\n\n📌 신호: `{signal}`\n🛡️ 결과 포지션: `{latest['Asset']}`\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
                    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                                   json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=15)
                    print(f"✅ [Worker] Trade signal detected ({signal}). Alert sent!", flush=True)
                else:
                    print(f"ℹ️ [Worker] Current status: {signal}. No action needed today. Skipping alert.", flush=True)
        except Exception as e:
            print(f"❌ [Worker Error]: {e}", flush=True)
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 8080))
        app.run(host='0.0.0.0', port=port)

except Exception as e:
    print(f"!!! [FATAL BOOT ERROR] !!!\n{e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
