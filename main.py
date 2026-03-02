import os
import sys
import threading
from flask import Flask

# 🌟 [해결] Cloud Run 환경에서 core 모듈 임포트 시 Streamlit 충돌 방지
class Dummy:
    def __getattr__(self, name):
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        return Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __bool__(self): return True

# 어떤 하위 모듈에서 streamlit을 불러도 에러가 나지 않게 주입합니다.
sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Success (Flask/Gunicorn Mode)")

# Flask 앱 생성 (Gunicorn이 이 app 객체를 찾아 실행합니다)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/alert', methods=['GET', 'POST'])
def home():
    """
    Cloud Run의 8080 포트 체크에 즉각 응답하여 배포 속도를 높입니다.
    """
    print("✅ HTTP Request received. Starting analysis in background...")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        # 분석 로직은 백그라운드 쓰레드에서 실행
        threading.Thread(target=run_analysis_async, args=(token, chat_id)).start()
        return "OK", 200
    
    return "Check Configuration", 200 # 배포는 통과시킴

def run_analysis_async(token, chat_id):
    """
    실제 데이터 분석 및 텔레그램 알림 전송 로직
    """
    try:
        print("🚀 [분석작업] 라이브러리 및 데이터 로딩 시작...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 설정 파일 로드
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
        
        if not os.path.exists(strat_path):
            print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다 -> {strat_path}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 실시간 데이터 수집 및 엔진 실행
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
            salt='final_alert_v11_gunicorn'
        )

        # 3. 알림 전송
        if not gh.empty:
            latest = gh.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            msg = f"🔔 **TQQQ 실시간 시그널 알림**\n\n"
            msg += f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **현재 포지션:** `{latest['Asset']}`\n"
            
            today_events = [e for e in events if e['date'].strftime('%Y-%m-%d') == last_date]
            if today_events:
                msg += "\n🎯 **시그널 감지:**\n"
                for ev in today_events:
                    msg += f"👉 {ev['type']}: {ev['reason']}\n"
            
            msg += f"\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
            
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, 
                          timeout=15)
            print("✅ [분석작업] 알림 전송 완료!")
            
    except Exception as e:
        print(f"❌ [분석작업] 치명적 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 로컬 테스트용 (Cloud Run에서는 gunicorn이 직접 app을 실행합니다)
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))
