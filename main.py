import os
import sys
import threading

# 🌟 [1단계] 어떤 임포트보다 최상단에서 Streamlit 에러를 원천 차단합니다.
class DummyStreamlit:
    def __getattr__(self, name): return lambda *args, **kwargs: DummyStreamlit()
    def __call__(self, *args, **kwargs): return DummyStreamlit()
sys.modules['streamlit'] = DummyStreamlit()
print("✅ Streamlit Mocking Success")

import functions_framework

@functions_framework.http
def alert_handler(request):
    """
    구글 서버가 8080 포트를 찌르면 즉시 응답을 보내 배포를 성공시킵니다.
    """
    print("✅ alert_handler triggered")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ 환경변수 TELEGRAM_BOT_TOKEN/CHAT_ID가 누락되었습니다.")
        return "Env Missing", 200

    # 분석 작업은 별도 쓰레드에서 실행하여 부팅 타임아웃을 방지합니다.
    t = threading.Thread(target=safe_run_logic, args=(token, chat_id))
    t.start()
    return "OK: Logic started in background.", 200

def safe_run_logic(token, chat_id):
    """
    모든 과정을 try-except로 감싸서 에러가 나도 서버가 죽지 않게 하고, 로그를 남깁니다.
    """
    try:
        print("🚀 [분석시작] 라이브러리 및 데이터 로드 중...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 전략 파일 경로 확인 (영어 파일명 사용 권장: tqqq_strategy.json)
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
        
        if not os.path.exists(strat_path):
            print(f"❌ 에러: 전략 파일을 찾을 수 없습니다: {strat_path}")
            # 파일이 없으면 혹시 모르니 strategies 폴더 목록을 찍어봅니다.
            if os.path.exists(os.path.join(project_root, 'strategies')):
                print(f"📂 폴더 목록: {os.listdir(os.path.join(project_root, 'strategies'))}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 데이터 수집 및 엔진 실행
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 전략 실행
        gh, _, events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, 
            strat.get('leverage_asset', 'TQQQ'), 
            strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            datetime.date(2010, 1, 1), 
            datetime.date.today(), 
            strat, 
            strat.get('trade_at', '종가'), 
            salt='final_v5'
        )

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
        traceback.print_exc() # 상세 에러 로그 기록

if __name__ == "__main__":
    from functions_framework._cli import _cli
    # 0.0.0.0 호스트 명시와 PORT 8080 연결이 Cloud Run 성공의 핵심입니다.
    port = os.environ.get("PORT", "8080")
    print(f"🚀 [최종모드] {port} 포트에서 대기 중...")
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
