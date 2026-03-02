import os
import json
import datetime
import functions_framework
from core.data import DataService
from core.engine import StrategyEngine

# 1. 텔레그램 전송 함수
def send_telegram(token, chat_id, title, message):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = f"*{title}*\n\n{message}"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=15)
        print("✅ 텔레그램 전송 완료")
    except Exception as e:
        print(f"❌ 전송 실패: {e}")

# 2. 메인 시그널 처리 함수 (구글 서버가 직접 호출하는 지점)
@functions_framework.http
def alert_handler(request):
    """
    구글 클라우드 서버가 직접 이 함수를 찾아 실행합니다.
    환경변수 GOOGLE_FUNCTION_TARGET 설정이 필수입니다.
    """
    print("� [시작] TQQQ 알림 로직 실행 중...")
    
    # 환경 변수 로드
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return "에러: 환경변수(TOKEN/ID)가 없습니다.", 500

    try:
        # 경로 및 데이터 설정
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "TQQQ 전략.json")
        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 1. 실시간 데이터 수집 (VXN/VIX 포함)
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 2. 전략 엔진 실행
        golden_history, _, all_signal_events = StrategyEngine.run_golden_strategy(
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

        # 3. 결과 분석 및 알림 생성
        if not golden_history.empty:
            latest = golden_history.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            
            msg = f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **현재 포지션:** `{latest['Asset']}`\n"
            
            # 오늘 발생한 시그널 확인
            today_events = [e for e in all_signal_events if e['date'].strftime('%Y-%m-%d') == last_date]
            if today_events:
                msg += "\n🎯 **감지된 시그널:**\n"
                for ev in today_events:
                    msg += f"👉 {ev['type']}: {ev['reason']}\n"
            
            msg += f"\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
            
            # 텔레그램 발송
            send_telegram(token, chat_id, "🔔 TQQQ 실시간 알림", msg)
            return "성공: 알림이 전송되었습니다.", 200
        else:
            return "경고: 분석된 데이터가 없습니다.", 200

    except Exception as e:
        print(f"❌ 치명적 에러: {e}")
        return f"에러 발생: {e}", 500

# 로컬 테스트 및 Cloud Run 환경 직접 실행용
if __name__ == "__main__":
    import os
    from functions_framework._cli import _cli
    port = os.environ.get("PORT", "8080")
    # --host 0.0.0.0 설정이 있어야 Cloud Run의 외부 접속 요청을 수신할 수 있습니다.
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
