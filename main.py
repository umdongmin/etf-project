import os
import sys
import threading
import functions_framework

# 🌟 [최후의 해결책] Streamlit 관련 모든 충돌을 방지하는 무조건 통과용 가짜 객체
class Dummy:
    def __getattr__(self, name):
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f)
        return lambda *args, **kwargs: Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __bool__(self): return True

# 어떤 하위 모듈에서 streamlit을 불러도 에러가 나지 않게 주입합니다.
sys.modules['streamlit'] = Dummy()

@functions_framework.http
def alert_handler(request):
    """
    구글 서버의 8080 포트 체크(Health Check)에 즉시 응답 성공(200 OK)을 보냅니다.
    이 함수가 0.1초 만에 응답해야 배포 실패(Timeout)가 뜨지 않습니다.
    """
    print("✅ Health Check received. Starting analysis in background...")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    # 작업은 별도 쓰레드에서 수행 (프로그램 부팅 지연 방지)
    t = threading.Thread(target=run_analysis_async, args=(token, chat_id))
    t.start()
    
    return "OK", 200

def run_analysis_async(token, chat_id):
    """
    구글 서버를 통과시킨 후, 백그라운드에서 조용히 라이브러리를 로딩하고 분석합니다.
    """
    try:
        if not token or not chat_id:
            print("⚠️ Telegram 환경 변수가 설정되지 않았습니다.")
            return

        print("🚀 [분석작업] 라이브러리 및 데이터 로딩 시작...")
        import json
        import datetime
        import requests
        
        # 실제 로직에 필요한 모듈을 작업 시점에 임포트 (Lazy Import)
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 파일 로드 (파일명: tqqq_strategy.json)
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
            salt='final_alert_v10'
        )

        # 3. 알림 전송
        if not gh.empty:
            latest = gh.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            msg = f"🔔 **TQQQ 실시간 알림**\n\n"
            msg += f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **포지션:** `{latest['Asset']}`\n"
            
            today_events = [e for e in events if e['date'].strftime('%Y-%m-%d') == last_date]
            if today_events:
                msg += "\n🎯 **시그널 감지:**\n"
                for ev in today_events:
                    msg += f"👉 {ev['type']}: {ev['reason']}\n"
            
            msg += f"\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
            
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, 
                          timeout=15)
            print("✅ 분석 및 알림 전송 완료!")
            
    except Exception as e:
        print(f"❌ [에러발생] 분석 도중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    from functions_framework._cli import _cli
    port = os.environ.get("PORT", "8080")
    # --host 0.0.0.0 과 --target 필수
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
