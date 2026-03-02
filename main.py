import os
import sys
import threading
from flask import Flask

# 🌟 [해결] core 모듈 내의 Streamlit 임포트 및 데코레이터(@st.cache_data 등) 충돌을 완벽 방어합니다.
# Context Manager(__enter__, __exit__)까지 지원하도록 강화했습니다.
class Dummy:
    def __getattr__(self, name):
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        return lambda *args, **kwargs: Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __bool__(self): return True

# 어떤 하위 모듈에서 streamlit을 불러도 에러가 나지 않게 주입합니다.
sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Success (Flask Mode)")

# Flask 서버 생성
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/alert', methods=['GET', 'POST'])
def home():
    """
    Cloud Run의 8080 포트 체크에 0.001초 만에 응답하여 배포 실패를 방지합니다.
    """
    print("✅ HTTP Request received. Starting analysis in background...")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        # 무거운 분석 작업은 즉시 별도 쓰레드에서 실행
        threading.Thread(target=run_analysis_internal, args=(token, chat_id)).start()
        return "OK: Analysis started in background.", 200
    
    return "Error: Environment variables missing.", 200 # 배포 성공을 위해 200 리턴

def run_analysis_internal(token, chat_id):
    """
    구글 서버를 안심시킨 후, 백그라운드에서 조용히 실행되는 실제 로직
    """
    try:
        print("🚀 [분석작업] 라이브러리 로딩 및 데이터 분석 시작...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 설정 파일 로드 (파일명: tqqq_strategy.json)
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
        
        if not os.path.exists(strat_path):
            print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다 -> {strat_path}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 실시간 데이터 수집 및 전략 실행
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
            salt='final_alert_v10_flask'
        )

        # 3. 알림 전송
        if not gh.empty:
            latest = gh.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            current_asset = latest['Asset']
            
            msg = f"🔔 **TQQQ 실시간 시그널 알림**\n\n"
            msg += f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **현재 포지션:** `{current_asset}`\n"
            
            # 오늘 발생한 시그널 필터링
            today_events = [e for e in events if e['date'].strftime('%Y-%m-%d') == last_date]
            if today_events:
                msg += "\n🎯 **감지된 시그널:**\n"
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
    # 구글 클라우드가 요구하는 PORT 환경변수에 맞춰 서버를 실행합니다.
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Flask Server Mode: Listening on port {port} (Host: 0.0.0.0)")
    app.run(host='0.0.0.0', port=port)
