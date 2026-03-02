import os
import sys
import threading
import functions_framework

# 🌟 [해결] Cloud Run 환경에는 streamlit이 없으므로, core 모듈 임포트 시 충돌을 막기 위해 '가짜' 모듈을 주입합니다.
# 특히 @st.cache_data 같은 데코레이터 에러를 완벽히 차단합니다.
class Dummy:
    def __getattr__(self, name):
        # cache_data 등을 호출하면 함수를 그대로 돌려주는(identity) 데코레이터를 반환합니다.
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        # st.columns 같은 호출을 대비해 리스트 등을 반환해야 할 경우를 처리합니다.
        if name == 'columns':
            return lambda *args, **kwargs: [Dummy()] * 10
        # 그 외 모든 접근은 호출 가능한 Dummy 객체를 반환합니다.
        return lambda *args, **kwargs: Dummy()
    
    def __call__(self, *args, **kwargs):
        return Dummy()

    def __bool__(self):
        return True

# streamlit 모듈 전체를 Dummy 객체로 대체합니다.
sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Success (With Decorator Support)")

@functions_framework.http
def alert_handler(request):
    """
    구글 서버의 8080 포트 체크(Health Check)에 즉시 응답 성공(200 OK)을 보냅니다.
    """
    print("✅ Health Check received. Starting analysis in background...")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        # 실제 데이터 분석 및 알림 작업은 구글 서버를 안심시킨 뒤 백그라운드에서 진행합니다.
        t = threading.Thread(target=run_analysis_in_background, args=(token, chat_id))
        t.start()
        return "OK", 200
    
    print("⚠️ Environment variables (TOKEN/ID) are missing.")
    return "OK (Missing configuration)", 200

def run_analysis_in_background(token, chat_id):
    """
    배포가 완료된 후 조용히 실행되는 실제 로직입니다.
    """
    try:
        print("🚀 [분석작업] 라이브러리 및 데이터 로딩 시작...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 영어로 된 전략 파일 로드
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
        
        if not os.path.exists(strat_path):
            print(f"❌ 에러: 전략 파일을 찾을 수 없습니다: {strat_path}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 실시간 데이터 수집
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 3. 전략 분석 실행
        gh, _, events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, 
            strat.get('leverage_asset', 'TQQQ'), 
            strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            datetime.date(2010, 1, 1), 
            datetime.date.today(), 
            strat, 
            strat.get('trade_at', '종가'), 
            salt='final_alert_v5'
        )

        # 4. 분석 결과가 있으면 텔레그램 전송
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
        print(f"❌ [분석작업] 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from functions_framework._cli import _cli
    # 0.0.0.0과 PORT 지정을 통해 Cloud Run의 포트 체크를 통과합니다.
    port = os.environ.get("PORT", "8080")
    print(f"🚀 Cloud Run Mode: Listening on port {port}")
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
