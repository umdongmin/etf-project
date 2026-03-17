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
    try:
        from flask import Flask
        app = Flask(__name__)
    except ImportError:
        app = None
        print("⚠️ Flask 미설치 — dry-run 모드로만 동작합니다.", flush=True)

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

    if app:
        app.add_url_rule('/', 'handle_request', handle_request, methods=['GET', 'POST'])

    def run_tqqq_bot(token, chat_id, dry_run=False):
        """포트폴리오 구조 기반 신호 계산 → 계좌별 Telegram 알림 발송"""
        try:
            import datetime, requests
            from core.data import DataService
            from core.storage import StrategyStorage, AssetStorage
            from core.signal_service import compute_signals

            def _get_price(ticker, prices):
                p = prices.get(ticker) or prices.get(ticker.upper())
                if p: return float(p)
                try:
                    import yfinance as yf
                    return float(getattr(yf.Ticker(ticker).fast_info, 'last_price', None) or 0)
                except: return 0.0

            def _extract_signals(result, eq_cfg):
                """predict_portfolio_today 결과 → {name: {stage, strat_type, base_asset, lev_asset}}"""
                out = {}
                for name, sig in result.get('strategies', {}).items():
                    if not sig: continue
                    if sig.get('strat_type', 'equity') == 'equity':
                        out[name] = {
                            'strat_type': 'equity',
                            'stage': int(sig.get('stage', 0)),
                            'base_asset': eq_cfg['params'].get('base_asset', 'QQQ') if eq_cfg else 'QQQ',
                            'lev_asset':  eq_cfg['params'].get('leverage_asset', 'TQQQ') if eq_cfg else 'TQQQ',
                        }
                    else:
                        out[name] = {
                            'strat_type': 'bond',
                            'stage': int(sig.get('lev_stage', 0)),
                            'base_asset': sig.get('asset', 'TLT'),
                            'lev_asset':  sig.get('lev_asset', '') or '',
                        }
                return out

            def _alloc_from_signals(configs, signals):
                """신호 기반 목표 배분 → {ticker: portfolio_weight}"""
                STAGE_RATIOS = {0:(1.0,0.0), 1:(0.7,0.3), 2:(0.3,0.7), 3:(0.0,1.0)}
                alloc = {}
                for cfg in configs:
                    name = cfg['name']; weight = cfg.get('weight', 0)
                    sig = signals.get(name, {}); stage = sig.get('stage', 0)
                    base = sig.get('base_asset', 'QQQ'); lev = sig.get('lev_asset', '')
                    br, lr = STAGE_RATIOS.get(stage, (1.0, 0.0))
                    if br > 0 and base: alloc[base.upper()] = alloc.get(base.upper(), 0) + weight * br
                    if lr > 0 and lev:  alloc[lev.upper()]  = alloc.get(lev.upper(),  0) + weight * lr
                return alloc

            STAGE_LABELS = {0:'S0 (Base 100%)', 1:'S1 (Base 70%+Lev 30%)',
                            2:'S2 (Base 30%+Lev 70%)', 3:'S3 (Lev 100%)'}
            STAGE_ICONS  = {0:'🔵', 1:'🟡', 2:'🟠', 3:'🔴'}

            # ── 1. 포트폴리오 로드 ────────────────────────────────────
            PORTFOLIO_NAME = os.environ.get("TELEGRAM_PORTFOLIO", "A-공격형")
            portfolio_data = StrategyStorage.load_portfolio(PORTFOLIO_NAME)
            if not portfolio_data:
                print(f"❌ 포트폴리오 '{PORTFOLIO_NAME}' 로드 실패", flush=True); return
            configs  = portfolio_data.get('configs', [])
            eq_cfg   = next((c for c in configs if c['type'] == 'equity'), None)
            start_date = datetime.date(2026, 1, 1)

            # ── 2. 데이터 수집 ────────────────────────────────────────
            print("📡 데이터 수집 중...", flush=True)
            data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data()
            if not data_dict:
                print("❌ 데이터 로드 실패", flush=True); return

            cur_prices = DataService.fetch_current_prices(list(data_dict.keys()) + ['^VIX', '^VXN'])
            print(f"{'✅' if cur_prices else '⚠️'} 현재가 {'조회 완료' if cur_prices else '없음 — 종가 기준 사용'}", flush=True)

            # ── 3. 신호 계산: 종가(현재) + 실시간(변경) ──────────────
            print("🔍 종가 기준 신호 계산 중...", flush=True)
            closing_result = compute_signals(
                PORTFOLIO_NAME, data_dict, fg_df, vxn_df, macro_df, news_df,
                start_date, cur_prices=None,
            )
            if not closing_result:
                print("❌ 종가 신호 계산 실패", flush=True); return

            print("🔍 실시간 신호 계산 중...", flush=True)
            realtime_result = compute_signals(
                PORTFOLIO_NAME, data_dict, fg_df, vxn_df, macro_df, news_df,
                start_date, cur_prices=cur_prices,   # None이면 종가와 동일
            )
            if not realtime_result:
                realtime_result = closing_result

            closing_signals  = _extract_signals(closing_result,  eq_cfg)
            realtime_signals = _extract_signals(realtime_result, eq_cfg)
            target_alloc     = _alloc_from_signals(configs, realtime_signals)
            portfolio_sig    = realtime_result.get('portfolio', {})

            # ── 4. 계좌별 메시지 생성 및 전송 ────────────────────────
            today_str = datetime.date.today().strftime('%Y-%m-%d')
            accounts  = AssetStorage.list_accounts()

            for acc in accounts:
                acc_id   = acc['id']
                acc_name = acc['name']

                holdings = AssetStorage.list_holdings(acc_id)

                # 보유 현황 맵
                holdings_map = {}
                total_value  = 0.0
                for h in holdings:
                    ticker = h['asset_id'].upper()
                    price  = _get_price(ticker, cur_prices or {})
                    val    = h['quantity'] * price
                    total_value += val
                    holdings_map[ticker] = {'qty': h['quantity'], 'price': price, 'value': val}

                # 목표 수량 (실시간 신호 기반)
                target_qty = {}
                for ticker, weight in target_alloc.items():
                    price = _get_price(ticker, cur_prices or {})
                    target_qty[ticker] = round(total_value * weight / price) if price > 0 else 0

                lines = [
                    f"━━━━━━━━━━━━━━━━━━━━━━",
                    f"👤 *{acc_name}*  |  📅 {today_str}",
                    f"📊 포트폴리오: *{PORTFOLIO_NAME}*",
                    f"💰 총 평가액: ${total_value:,.0f}",
                    f"━━━━━━━━━━━━━━━━━━━━━━",
                ]
                action_statuses = []  # 전략별 status 수집 (전송 여부 판단용)

                # 전략별 메시지: 종가 신호 = 현재, 실시간 신호 = 변경
                for name, new_sig in realtime_signals.items():
                    stype     = new_sig['strat_type']
                    new_stage = new_sig['stage']
                    new_icon  = STAGE_ICONS.get(new_stage, '⚪')
                    new_label = STAGE_LABELS.get(new_stage, f'S{new_stage}')
                    alert     = ' ⚠️' if new_stage >= 2 else ''

                    cur_sig   = closing_signals.get(name, new_sig)
                    cur_stage = cur_sig['stage']
                    cur_icon  = STAGE_ICONS.get(cur_stage, '⚪')
                    cur_label = STAGE_LABELS.get(cur_stage, f'S{cur_stage}')

                    if stype == 'equity':
                        base_t = new_sig['base_asset'].upper()
                        lev_t  = new_sig['lev_asset'].upper()
                        cur_base_qty = holdings_map.get(base_t, {}).get('qty', 0)
                        cur_lev_qty  = holdings_map.get(lev_t,  {}).get('qty', 0)
                        tgt_base_qty = target_qty.get(base_t, 0)
                        tgt_lev_qty  = target_qty.get(lev_t,  0)
                        diff_base = tgt_base_qty - int(cur_base_qty)
                        diff_lev  = tgt_lev_qty  - int(cur_lev_qty)

                        if new_stage != cur_stage:
                            status = '매수' if new_stage > cur_stage else '매도'
                        elif diff_base != 0 or diff_lev != 0:
                            status = '조정'
                        else:
                            status = '보유'

                        lines.append(f"\n📈 *{name}*{alert}  [{status}]")
                        lines.append(f"  현재: {cur_icon} {cur_label}")
                        lines.append(f"        {base_t}:{int(cur_base_qty)}주  {lev_t}:{int(cur_lev_qty)}주")
                        lines.append(f"  변경: {new_icon} {new_label}")
                        lines.append(f"        {base_t}:{tgt_base_qty}주  {lev_t}:{tgt_lev_qty}주")

                        result_parts = []
                        if diff_base > 0:  result_parts.append(f"🟢 {base_t} {diff_base}주 매수")
                        elif diff_base < 0: result_parts.append(f"🔴 {base_t} {abs(diff_base)}주 매도")
                        if diff_lev > 0:   result_parts.append(f"🟢 {lev_t} {diff_lev}주 매수")
                        elif diff_lev < 0:  result_parts.append(f"🔴 {lev_t} {abs(diff_lev)}주 매도")
                        action_statuses.append(status)
                        lines.append(f"  결과: {' / '.join(result_parts) if result_parts else '변동 없음'}")

                    else:
                        base_t = new_sig['base_asset'].upper()
                        lev_t  = (new_sig['lev_asset'] or '').upper()
                        cur_base_qty = holdings_map.get(base_t, {}).get('qty', 0)
                        cur_lev_qty  = holdings_map.get(lev_t,  {}).get('qty', 0) if lev_t else 0
                        tgt_base_qty = target_qty.get(base_t, 0)
                        tgt_lev_qty  = target_qty.get(lev_t,  0) if lev_t else 0
                        diff_base = tgt_base_qty - int(cur_base_qty)
                        diff_lev  = tgt_lev_qty  - int(cur_lev_qty)

                        status = '보유' if diff_base == 0 and diff_lev == 0 else '조정'
                        lines.append(f"\n📉 *{name}*  [{status}]")
                        lines.append(f"  현재: {cur_icon} {cur_label}  {base_t}:{int(cur_base_qty)}주")
                        lines.append(f"  변경: {new_icon} {base_t}" + (f" → {lev_t}:{tgt_lev_qty}주" if lev_t and tgt_lev_qty > 0 else f":{tgt_base_qty}주"))
                        result_parts = []
                        if diff_base > 0:  result_parts.append(f"🟢 {base_t} {diff_base}주 매수")
                        elif diff_base < 0: result_parts.append(f"🔴 {base_t} {abs(diff_base)}주 매도")
                        if lev_t:
                            if diff_lev > 0:   result_parts.append(f"🟢 {lev_t} {diff_lev}주 매수")
                            elif diff_lev < 0:  result_parts.append(f"🔴 {lev_t} {abs(diff_lev)}주 매도")
                        action_statuses.append(status)
                        lines.append(f"  결과: {' / '.join(result_parts) if result_parts else '변동 없음'}")

                drift = portfolio_sig.get('drift_pct', 0)
                try: drift_str = f"{float(drift):.1f}"
                except: drift_str = str(drift)
                rebal = '필요 ⚠️' if portfolio_sig.get('rebalance_needed') else '정상 ✅'
                lines.append(f"\n🔄 Drift: `{drift_str}%`  Rebalance: {rebal}")
                lines.append("━━━━━━━━━━━━━━━━━━━━━━")

                msg = '\n'.join(lines)

                # ── 출력 / 전송 ───────────────────────────────────────
                print(f"\n{'='*50}", flush=True)
                print(f"📋 [{acc_name}] 메시지 미리보기", flush=True)
                print('='*50, flush=True)
                print(msg, flush=True)
                print('='*50 + '\n', flush=True)

                has_action = any(s != '보유' for s in action_statuses)
                needs_rebal = portfolio_sig.get('rebalance_needed', False)
                should_send = has_action or needs_rebal

                if not dry_run:
                    if should_send:
                        send_url = f"https://api.telegram.org/bot{token}/sendMessage"
                        payload  = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                        r = requests.post(send_url, json=payload, timeout=10)
                        print(f"📡 [{acc_name}] Telegram Send: {r.status_code}", flush=True)
                    else:
                        print(f"⏭️ [{acc_name}] 변동 없음 — 전송 생략", flush=True)
                else:
                    print(f"🔕 [DRY RUN] {acc_name} {'전송 예정' if should_send else '전송 생략 (변동 없음)'}", flush=True)

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
