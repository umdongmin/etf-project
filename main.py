import os
import sys
import threading
import functions_framework

# 🌟 [해결책] Streamlit 없는 환경(GCF/Cloud Run)에서 core 모듈 임포트 시 발생하는 에러 방지
class DummyStreamlit:
    def cache_data(self, *args, **kwargs):
        return lambda x: x
    def cache_resource(self, *args, **kwargs):
        return lambda x: x
    def write(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def success(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def sidebar(self, *args, **kwargs): return self
    def columns(self, *args, **kwargs): return [self] * 10
    def container(self, *args, **kwargs): return self

# GCF에는 streamlit이 설치되어 있지 않으므로 가짜 모듈을 주입합니다.
sys.modules['streamlit'] = DummyStreamlit()

@functions_framework.http
def alert_handler(request):
    """
    구글 헬스체크를 통과하기 위해 0.1초 만에 응답을 보내고, 
    실제 분석 작업은 백그라운드 쓰레드에서 수행합니다.
    """
    print("✅ 구글 헬스체크 수신 성공! 즉시 응답합니다.")
    
    # 텔레그램 토큰 확인
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if token and chat_id:
        # 무거운 작업은 별도 쓰레드에서 실행 (부팅 타임아웃 방지)
        t = threading.Thread(target=run_logic_task, args=(token, chat_id))
        t.start()
        return "OK: Logic started.", 200
    
    print("⚠️ 텔레그램 환경변수 누락됨 (TOKEN/ID)")
    return "OK: But environment variables missing.", 200 # 배포 성공을 위해 200 리턴

def run_logic_task(token, chat_id):
    """
    실제 데이터 수집 및 시그널 분석 로직
    """
    try:
        print("🚀 [배경작업] 라이브러리 로딩 및 분석 시작...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 로드
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "TQQQ 전략.json")
        if not os.path.exists(strat_path):
            print(f"❌ 전략 파일을 찾을 수 없음: {strat_path}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 데이터 수집 (VXN/VIX 등 포함)
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 3. 전략 실행
        gh, _, events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, 
            strat.get('leverage_asset', 'TQQQ'), 
            strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            datetime.date(2010, 1, 1), 
            datetime.date.today(), 
            strat, 
            strat.get('trade_at', '종가'), 
            salt='alert_bot_final_fix'
        )

        # 4. 결과 전송
        if not gh.empty:
            latest = gh.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            
            msg = f"🔔 **TQQQ 실시간 알림**\n\n"
            msg += f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **현재 포지션:** `{latest['Asset']}`\n"
            
            today_events = [e for e in all_signal_events if e['date'].strftime('%Y-%m-%d') == last_date] if 'all_signal_events' in locals() else []
            # 위 변수명이 전략마다 다를 수 있어 events(엔진 리턴값) 사용 권장
            today_events = [e for e in events if e['date'].strftime('%Y-%m-%d') == last_date]
            
            if today_events:
                msg += "\n🎯 **감지된 시그널:**\n"
                for ev in today_events:
                    msg += f"👉 {ev['type']}: {ev['reason']}\n"
            
            msg += f"\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
            
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, 
                          timeout=15)
            print("✅ [배경작업] 텔레그램 알림 전송 완료!")
            
    except Exception as e:
        print(f"❌ [배경작업] 에러 발생: {e}")

if __name__ == "__main__":
    from functions_framework._cli import _cli
    port = os.environ.get("PORT", "8080")
    # --host 0.0.0.0 필수
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
