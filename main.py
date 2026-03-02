import os
import sys
import threading
import time

# 🚨 로그 출력을 강제로 즉시 보이게 설정 (배포 디버깅의 핵심)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("--- [BOOT] TQQQ Trading Bot Starting ---")

# 🌟 [해결] 최상단에서 스트림릿 모듈을 가짜로 주입하여 라이브러리 충돌을 원천 차단합니다.
class Dummy:
    def __getattr__(self, name): return lambda *args, **kwargs: Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __bool__(self): return True

sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Initialized")

try:
    from flask import Flask
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/alert', methods=['GET', 'POST'])
    def health_check():
        """구글 서버의 8080 포트 체크에 즉각 'OK'를 보냅니다."""
        print("✅ Received request from Google Server. Health Check Passed.")
        
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        # 실제 분석은 응답 후 별도 쓰레드에서 딴짓하지 않고 즉시 실행
        if token and chat_id:
            threading.Thread(target=run_analysis_background, args=(token, chat_id)).start()
            return "Analysis Started", 200
        else:
            print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing in environment variables.")
            return "Config Missing", 200

    def run_analysis_background(token, chat_id):
        """무거운 라이브러리 임포트와 분석을 백그라운드에서 진행합니다."""
        try:
            print("🚀 [Background] Loading libraries and starting analysis...")
            import json
            import datetime
            import requests
            
            # 주의: core 모듈 임포트 시 streamlit 충돌이 날 수 있어 여기서도 mocking 확인
            from core.data import DataService
            from core.engine import StrategyEngine

            # 1. 전략 파일 경로 확인
            project_root = os.path.dirname(os.path.abspath(__file__))
            strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
            
            if not os.path.exists(strat_path):
                print(f"❌ Strategy file NOT found: {strat_path}")
                return

            with open(strat_path, 'r', encoding='utf-8') as f:
                strat = json.load(f)

            # 2. 분석 실행
            data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
            gh, _, events = StrategyEngine.run_golden_strategy(
                data_dict, fg_df, vix_df, 
                strat.get('leverage_asset', 'TQQQ'), 
                strat.get('base_asset', 'QQQ'),
                strat.get('cash_ratio_pct', 0.0) / 100.0, 
                datetime.date(2010, 1, 1), 
                datetime.date.today(), 
                strat, 
                strat.get('trade_at', '종가'), 
                salt='flask_vFinal_debug'
            )

            # 3. 알림 전송
            if not gh.empty:
                latest = gh.iloc[-1]
                msg = f"🔔 **TQQQ 실시간 시그널**\n\n📅 기준일: {latest.name.strftime('%Y-%m-%d')}\n🛡️ 포지션: `{latest['Asset']}`\n📊 RSI: `{latest['RSI']:.1f}`"
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                              json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, 
                              timeout=15)
                print("✅ Analysis and Alert completed successfully!")
                
        except Exception as e:
            print(f"❌ Background Error: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        # GCP Cloud Run이 부여하는 PORT 번호로 실행
        port = int(os.environ.get("PORT", 8080))
        print(f"📡 Flask Server listening on 0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port)

except Exception as e:
    print(f"!!! [FATAL BOOT ERROR] !!!\n{e}")
    import traceback
    traceback.print_exc()
    time.sleep(10) # 로그 전송 시간 확보
    sys.exit(1)
