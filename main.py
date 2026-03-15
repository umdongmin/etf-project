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
            
            # 🌟 사용자의 요청에 따라 기본 전략(tqqq_strategy.json)을 우선 로드
            default_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
            optimized_path = os.path.join(project_root, "strategies", "tqqq_strategy_Optimized.json")
            
            def get_fn(p): return os.path.basename(p)

            # [수정] 사용자가 tqqq_strategy.json을 명시적으로 원하므로 이를 우선 체크
            if os.path.exists(default_path):
                strat_path = default_path
                print(f"📦 [알림] 프로젝트 전략을 로드합니다: {get_fn(strat_path)}")
            elif os.path.exists(optimized_path):
                strat_path = optimized_path
                print(f"🚀 [알림] 최적화된 전략을 로드합니다: {get_fn(strat_path)}")
            else:
                print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다.")
                return

            if not os.path.exists(strat_path):
                print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다 -> {strat_path}", flush=True)
                return

            with open(strat_path, 'r', encoding='utf-8') as f:
                strat = json.load(f)

            # 2. 분석 실행 (실시간 데이터 수집 및 시그널 계산)
            sim_start_date = datetime.date(2026, 1, 1)
            fetch_start_date = (sim_start_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            
            print(f"📡 분석 대상: {fetch_start_date} ~ 오늘")
            
            # 2-1. 데이터 수집
            data_raw, fg_raw, vix_raw, macro_raw, news_raw, _, _ = DataService.fetch_live_data(start_date=fetch_start_date)
            all_tickers = list(data_raw.keys())
            # VIX/VXN 실시간 조회를 위해 티커 추가
            preview_tickers = all_tickers + ['^VIX', '^VXN']
            current_prices = DataService.fetch_current_prices(preview_tickers)
            
            # 2-2. 🌟 가상 종가 주입 (반환값 할당 버그 수정: Data, VIX, FG 3개 수령)
            data, vix, fg = DataService.inject_virtual_close(data_raw, current_prices, vix_df=vix_raw, fg_df=fg_raw)

            gh, _, events = StrategyEngine.run_golden_strategy(
                data, fg, vix, news_raw,
                strat.get('leverage_asset', 'TQQQ'), 
                strat.get('base_asset', 'QQQ'),
                0.0, 
                sim_start_date, 
                datetime.date.today(), 
                strat, 
                strat.get('trade_at', '종가'), 
                salt='vFinal_Batch_Live'
            )

            # 2-3. 채권 전략 분석 (신규)
            from core.storage import StrategyStorage
            bond_strat = StrategyStorage.load_strategy("Bond_V7_Custom") # 기본값 또는 저장된 값
            if not bond_strat: 
                bond_strat = {'ffv_lo': 0.0, 'ffv_hi': 0.8, 'yc_inv': -0.15, 'yc_steep': 0.1, 'tlt_rsi_max': 70}
            
            bond_res = StrategyEngine.predict_bond_signal(data_raw, macro_raw, params=bond_strat)

            # 3. 알림 전송
            if not gh.empty:
                latest = gh.iloc[-1]
                signal = str(latest.get('Trade_Label', ''))
                
                # 🌟 [수정] 매수 또는 매도 신호가 있을 때만 알람 발송
                # '진행중' 상태는 이미 포지션이 진입된 상태이므로 알람에서 제외 (중복 알람 방지)
                if ("매수" in signal or "매도" in signal) and "진행중" not in signal:

                    # 🔹 매수/매도 종류 강조
                    signal_type = "🔴 매도" if "매도" in signal else "🟢 매수"
                    
                    # 🔹 포지션 변화 추출 (현재 vs 결과)
                    # gh.iloc[-2]가 존재하면 변경 전 포지션으로 사용
                    prev_summary = gh.iloc[-2]['Summary'] if len(gh) > 1 else "정보 없음"
                    new_summary = latest['Summary']
                    
                    # 🌟 [신규] 채권 신호 포함
                    bond_text = ""
                    if bond_res:
                        bond_text = f"\n\n📉 **채권 전략 (V7) 예상 포지션**: `{bond_res['signal']}`\n• 매크로 상태: {bond_res['f_state']}\n• FFV Gap: {bond_res['gap']:.2f}\n• Yield Curve: {bond_res['yc']:.2f}"

                    header = "📢 **ETF Golden Strategy 분석 리포트**"
                    msg = f"{header}\n\n결과: {signal_type} {latest['Asset']}\n오늘 종가(예상): **${latest['Close']:,.2f}**\n\n📌 **상세 정보**\n• {latest['Summary']}\n• RSI: {latest['RSI']:.1f}, VXN: {latest['VXN']:.1f}\n\n🔄 **포지션 요약**\n• {prev_summary} ➡️ **{new_summary}**{bond_text}"
                    
                    # 텔레그램 전송 (requests 사용)
                    send_url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                    r = requests.post(send_url, json=payload, timeout=10)
                    print(f"📡 Telegram Send: {r.status_code}, {r.text}", flush=True)
                else:
                    print(f"ℹ️ [Worker] Current status: {signal}. No trade action needed. Skipping alert.", flush=True)
            else:
                print("ℹ️ No historical data available, skipping Telegram alert.", flush=True)
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
