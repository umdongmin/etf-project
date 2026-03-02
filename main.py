import os
import sys
import threading
import functions_framework

# 🌟 [해결] core 모듈에서 streamlit을 임포트할 때 GCF/Cloud Run 환경에서 나는 에러를 원천 차단합니다.
class DummyStreamlit:
    def __getattr__(self, name):
        # 모든 속성에 대해 자기 자신을 반환하는 가짜 메서드를 생성합니다.
        return lambda *args, **kwargs: DummyStreamlit()
    def __call__(self, *args, **kwargs):
        return DummyStreamlit()

# streamlit 모듈을 가짜 객체로 대체하여 임포트 시 충돌을 방지합니다.
sys.modules['streamlit'] = DummyStreamlit()

@functions_framework.http
def alert_handler(request):
    """
    구글 헬스체크(PORT 8080 확인) 수신 시 0.1초 만에 응답하여 배포 실패를 방지합니다.
    """
    print("✅ 구글 헬스체크 수신 성공! 작업을 백그라운드에서 시작합니다.")
    
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ 환경변수(TELEGRAM_BOT_TOKEN/CHAT_ID)가 없습니다.")
        return "OK (Env missing)", 200

    # 무거운 분석 작업은 별도 쓰레드에서 실행 (부팅 타임아웃 방지)
    t = threading.Thread(target=run_analysis_task, args=(token, chat_id))
    t.start()
    
    return "OK: Logic started in background.", 200

def run_analysis_task(token, chat_id):
    """
    실제 데이터 수집 및 시그널 분석을 수행하는 백그라운드 작업
    """
    try:
        print("🚀 [분석작업] 라이브러리 로딩 및 전략 분석 시작...")
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 설정 파일 로드 (영어 파일명 사용)
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
        
        if not os.path.exists(strat_path):
            print(f"❌ 전략 파일을 찾을 수 없습니다: {strat_path}")
            return

        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 실시간 데이터 수집
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 3. 전략 엔진 실행
        gh, _, events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, 
            strat.get('leverage_asset', 'TQQQ'), 
            strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            datetime.date(2010, 1, 1), 
            datetime.date.today(), 
            strat, 
            strat.get('trade_at', '종가'), 
            salt='alert_bot_final_v4'
        )

        # 4. 결과 알림 전송
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

if __name__ == "__main__":
    from functions_framework._cli import _cli
    # 0.0.0.0 호스트 명시와 PORT 8080 연결이 Cloud Run 성공의 핵심입니다.
    port = os.environ.get("PORT", "8080")
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
