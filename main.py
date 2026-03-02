import os
import json
import datetime
import functions_framework
from core.data import DataService
from core.engine import StrategyEngine

# GCF용 진입점 함수 (GitHub 연동 시 이 함수명을 '진입점'으로 사용하세요)
@functions_framework.http
def alert_handler(request):
    """
    Cloud Scheduler(HTTP) 또는 직접 호출에 의해 실행되는 핸들러
    """
    print("🔄 GCF 알림 봇 실행 시작...")
    
    # 텔레그램 설정 로드 (GCF 환경 변수 사용 권장)
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        return "Error: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID environment variables are missing.", 500

    try:
        run_daily_signal_check(bot_token, chat_id)
        return "Success: Alert sent.", 200
    except Exception as e:
        print(f"Error during execution: {e}")
        return f"Error: {e}", 500

def send_telegram(token, chat_id, title, message):
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = f"*{title}*\n\n{message}"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload, timeout=15)

def run_daily_signal_check(token, chat_id):
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 1. 전략 파일 로드 (TQQQ 전략.json 경로 확인)
    strat_path = os.path.join(project_root, "strategies", "TQQQ 전략.json")
    if not os.path.exists(strat_path):
        raise FileNotFoundError(f"Strategy file not found: {strat_path}")
        
    with open(strat_path, 'r', encoding='utf-8') as f:
        strat = json.load(f)

    # 2. 기초 데이터 로드 (VXN 등 동기화 완료된 실시간 데이터)
    data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
    
    base_asset = strat.get('base_asset', 'QQQ')
    lev_asset = strat.get('leverage_asset', 'TQQQ')
    
    # 3. 전략 엔진 실행
    start_date = datetime.date(2010, 1, 1)
    end_date = datetime.date.today()
    
    golden_history, _, all_signal_events = StrategyEngine.run_golden_strategy(
        data_dict, fg_df, vix_df, lev_asset, base_asset, 
        strat.get('cash_ratio_pct', 0.0) / 100.0, 
        start_date, end_date, strat, 
        strat.get('trade_at', '종가'), 
        strat.get('smart_params', {}), 
        salt='alert_bot_gcf'
    )

    if golden_history.empty:
        return

    latest_data = golden_history.iloc[-1]
    last_date = latest_data.name.strftime('%Y-%m-%d')
    prev_data = golden_history.iloc[-2] if len(golden_history) > 1 else latest_data
    
    current_asset = latest_data['Asset']
    prev_asset = prev_data['Asset']
    
    msg = f"📅 **기준일:** {last_date}\n\n"
    
    if current_asset != prev_asset:
        msg += f"🚨 **포지션 스위칭 발생!**\n기존 `{prev_asset}` ➡️ 신규 `{current_asset}`\n\n"
    else:
        msg += f"🛡️ **현재 포지션 유지 중:** `{current_asset}`\n\n"
        
    today_events = [e for e in all_signal_events if e['date'].strftime('%Y-%m-%d') == last_date]
    if today_events:
        msg += "🎯 **오늘 감지된 시그널:**\n"
        for ev in today_events:
            emoji = "🟢" if "매수" in ev['type'] else "🔴"
            msg += f"{emoji} {ev['type']} 발동\n👉 이유: {ev['reason']}\n"
        msg += "\n"
    
    msg += f"📊 **주요 지표 상태:**\n"
    msg += f"• RSI: `{latest_data['RSI']:.1f}` / VXN: `{latest_data['VXN']:.1f}`\n"
    msg += f"• MACD: `{latest_data['MACD']:.2f}` / ADX: `{latest_data['ADX']:.1f}`\n"
    
    send_telegram(token, chat_id, "🔔 TQQQ GCF 알림", msg)

# [추가] Cloud Run/GCF 통합 환경에서 8080 포트 에러 방지용 실행 로직
if __name__ == "__main__":
    from functions_framework._cli import _cli
    import sys
    
    # 8080 포트에서 대기하도록 설정 (이 코드가 있어야 Cloud Run이 응답 실패로 간주하지 않습니다)
    port = os.environ.get("PORT", "8080")
    print(f"🚀 Cloud Run 통합 모드 실행 중... (Port: {port})")
    
    # functions-framework를 이용해 alert_handler를 8080 포트로 띄웁니다.
    _cli(["--target", "alert_handler", "--port", port])
