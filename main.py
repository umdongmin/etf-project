import os
import sys
import threading
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
        """구글 서버 체크에 즉시 200 OK 응답을 보내고 분석을 시작합니다."""
        print("✅ Received request. Starting analysis in background...", flush=True)
        
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        # 실제 분석은 백그라운드 쓰레드에서 조용히 실행 (Gunicorn 타임아웃 방지)
        if token and chat_id:
            threading.Thread(target=run_tqqq_bot, args=(token, chat_id)).start()
            return "SUCCESS: Bot is working!", 200
        else:
            print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.", flush=True)
            return "SUCCESS: Server is alive but settings missing.", 200

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
            # 🌟 설정: 사용자님의 실제 투자 시작일(2026-01-01)로 시뮬레이션 시작을 맞춥니다.
            sim_start_date = datetime.date(2026, 1, 1)
            fetch_start_date = (sim_start_date - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            
            print(f"📡 데이터 수집 기간: {fetch_start_date} ~ 오늘")
            
            # 2-1. 과거 데이터 및 실시간 현재가 수집
            data_raw, fg, vix, _, _, _ = DataService.fetch_live_data(start_date=fetch_start_date)
            all_tickers = list(data_raw.keys())
            current_prices = DataService.fetch_current_prices(all_tickers)
            
            # 2-2. 🌟 가상 종가 주입 (UI의 '실시간 시그널 프리뷰'와 동일한 로직 적용)
            # 장중 현재가를 오늘자의 종가로 간주하여 정확한 실시간 시그널 도출
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
                salt='vFinal_Production_v3_Live'
            )

            # 3. 알림 전송
            if not gh.empty:
                latest = gh.iloc[-1]
                msg = f"🔔 **TQQQ 실시간 시그널**\n\n🛡️ 현재 포지션: `{latest['Asset']}`\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                              json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=15)
                print("✅ [Worker] Analysis complete and alert sent!", flush=True)
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
