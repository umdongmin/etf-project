import os
import threading
import functions_framework

# 🌟 핵심: 무거운 라이브러리(pandas, yfinance 등) 임포트를 작업 함수 안으로 이동합니다.
# 이렇게 하면 서버가 0.1초 만에 실행되어 구글의 8080 포트 체크(Health Check)를 통과합니다.

@functions_framework.http
def alert_handler(request):
    """
    구글 서버가 8080 포트를 찌르면 즉시 성공 응답을 보내 배포 실패를 방지합니다.
    실제 무거운 작업은 백그라운드 쓰레드에서 조용히 실행합니다.
    """
    print("✅ 구글 헬스체크 수신! 즉각 응답하여 배포 에러를 방지합니다.")
    
    # 텔레그램 토큰 설정 확인
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ 환경변수(TOKEN/ID)가 설정되지 않았습니다.")
        # 배포 성공을 위해 일단 200 OK를 리턴합니다.
        return "Warning: Env missing, but health check passed.", 200

    # 무거운 작업(데이터 수집, 전략 분석)은 배경 쓰레드에서 실행
    t = threading.Thread(target=run_heavy_logic_in_background, args=(token, chat_id))
    t.start()
    
    return "OK: Logic started in background.", 200

def run_heavy_logic_in_background(token, chat_id):
    """
    구글 서버 응답이 끝난 뒤, 여기서 무거운 라이브러리를 불러오고 작업을 수행합니다.
    """
    try:
        print("🚀 [배경작업] 무거운 라이브러리 및 데이터 분석 시작...")
        
        # 여기서 비로소 무거운 라이브러리를 임포트합니다.
        import json
        import datetime
        import requests
        from core.data import DataService
        from core.engine import StrategyEngine

        # 1. 전략 설정 파일 로드
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "TQQQ 전략.json")
        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 2. 실시간 데이터 수집 (VXN/VIX 등 포함)
        # 이 과정이 5~10초 걸리기 때문에 배경작업이 필수입니다.
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 3. 전략 엔진 실행
        golden_history, _, all_signal_events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, 
            strat.get('leverage_asset', 'TQQQ'), 
            strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            datetime.date(2010, 1, 1), 
            datetime.date.today(), 
            strat, 
            strat.get('trade_at', '종가'), 
            salt='alert_bot_fixed_v3'
        )

        # 4. 결과 분석 및 알림 전송
        if not golden_history.empty:
            latest = golden_history.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            current_asset = latest['Asset']
            
            msg = f"🔔 **TQQQ 실시간 알림**\n\n"
            msg += f"📅 **기준일:** {last_date}\n"
            msg += f"🛡️ **현재 포지션:** `{current_asset}`\n"
            
            today_events = [e for e in all_signal_events if e['date'].strftime('%Y-%m-%d') == last_date]
            if today_events:
                msg += "\n🎯 **감지된 시그널:**\n"
                for ev in today_events:
                    msg += f"👉 {ev['type']}: {ev['reason']}\n"
            
            msg += f"\n📊 RSI: `{latest['RSI']:.1f}` / VXN: `{latest['VXN']:.1f}`"
            
            # 텔레그램 전송
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", 
                          json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, 
                          timeout=15)
            print("✅ [배경작업] 알림 전송 완료!")
            
    except Exception as e:
        print(f"❌ [배경작업] 치명적 에러 발생: {e}")

# Cloud Run/GCF 강제 실행용 (삭제하면 안 됨)
if __name__ == "__main__":
    from functions_framework._cli import _cli
    port = os.environ.get("PORT", "8080")
    # --host 0.0.0.0 이 있어야 외부 헬스체크 신호를 받을 수 있습니다.
    _cli(["--target", "alert_handler", "--port", port, "--host", "0.0.0.0"])
