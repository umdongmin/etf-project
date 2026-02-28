import os
import sys

# 윈도우 환경(cmd, powershell) 콘솔 이모티콘 출력 에러 방지용
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
import json
import datetime
import requests
import pandas as pd
import yfinance as yf

# 기존 프로젝트 모듈(core.engine 등)을 불러오기 위해 경로 동적 추가 (GitHub Actions 대응)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from core.data import DataService
from core.engine import StrategyEngine

# ==========================================
# ⚠️ 텔레그램 설정 (환경변수 또는 수동 입력)
# ==========================================
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8617487351:AAEEgvEWUWIp6zTETwMmYsKgYiERp827y3c")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8645050146")
# ==========================================

def send_telegram(title, message):
    """텔레그램 메시지 전송 함수"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # 마크다운 포맷으로 예쁘게 보내기
    text = f"*{title}*\n\n{message}"
    
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ 텔레그램 알림 전송 성공")
        else:
            print(f"❌ 텔레그램 알림 전송 실패: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ 텔레그램 전송 중 에러 발생: {e}")

def run_daily_signal_check():
    """오늘 종가(현재가) 기준으로 엔진을 돌려 시그널을 확인하는 메인 함수"""
    print("🔄 실시간 시그널 검사를 시작합니다...")
    
    # 1. 전략 파일 로드
    strat_path = os.path.join(PROJECT_ROOT, "strategies", "TQQQ 전략.json")
    try:
        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)
    except Exception as e:
        send_telegram("⚠️ 자동 알림 봇 오류", f"전략 파일을 읽어올 수 없습니다: {e}")
        return

    # 2. 기초 데이터 로드 (과거 데이터)
    print("📊 과거 데이터 로딩 중 (야후 파이낸스)...")
    # 주의: fetch_live_data()는 시간이 조금 걸릴 수 있습니다.
    data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time = DataService.fetch_live_data()
    
    # 3. 데이터가 오늘 날짜까지 업데이트 안 되었을 수 있으므로 강제 업데이트 보정
    # 장 마감 전이라면 yfinance에서 가져온 '오늘'의 마지막 가격을 종가로 간주
    base_asset = strat.get('base_asset', 'QQQ')
    lev_asset = strat.get('leverage_asset', 'TQQQ')
    
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    # 어제 날짜 확인을 위해
    
    # 4. 전략 엔진 실행 (전체 기간)
    print("⚙️ 전략 엔진 백테스트 실행 중...")
    start_date = datetime.date(2010, 1, 1)
    end_date = datetime.date.today()
    
    try:
        golden_history, closed_trades, all_signal_events = StrategyEngine.run_golden_strategy(
            data_dict, 
            fg_df, 
            vix_df, 
            lev_asset, 
            base_asset, 
            strat.get('cash_ratio_pct', 0.0) / 100.0, 
            start_date, 
            end_date, 
            strat, 
            strat.get('trade_at', '종가'), 
            strat.get('smart_params', {}), 
            salt='alert_bot'
        )
    except Exception as e:
        send_telegram("⚠️ 자동 알림 봇 오류", f"엔진 실행 중 에러가 발생했습니다:\n`{e}`")
        return

    # 5. 결과 분석 및 알림 전송 (가장 최근 = 오늘 또는 어제 데이터)
    if golden_history.empty:
        print("결과 데이터가 없습니다.")
        return
        
    latest_data = golden_history.iloc[-1]
    last_date = latest_data.name.strftime('%Y-%m-%d')
    
    # 전일(비교용) 데이터
    prev_data = golden_history.iloc[-2] if len(golden_history) > 1 else latest_data
    
    # 상태 변화 감지 (레버리지 단계 변경이나 비중 등)
    current_asset = latest_data['Asset']
    prev_asset = prev_data['Asset']
    
    # 메시지 작성
    msg = f"📅 **기준일:** {last_date}\n\n"
    
    # (1) 레버리지(포지션) 변경 여부
    if current_asset != prev_asset:
        msg += f"🚨 **포지션 스위칭 발생!**\n"
        msg += f"기존 `{prev_asset}` ➡️ 신규 `{current_asset}`\n\n"
    else:
        msg += f"🛡️ **현재 포지션 유지 중:** `{current_asset}`\n\n"
        
    # (2) 오늘 발생한 시그널 이벤트 확인
    today_events = [e for e in all_signal_events if e['date'].strftime('%Y-%m-%d') == last_date]
    if today_events:
        msg += "🎯 **오늘 감지된 시그널:**\n"
        for ev in today_events:
            emoji = "🟢" if "매수" in ev['type'] else "🔴"
            msg += f"{emoji} {ev['type']} 발동\n"
            msg += f"👉 이유: {ev['reason']}\n"
        msg += "\n"
    elif current_asset != prev_asset:
        # 이벤트 리스트엔 없는데 포지션이 변했다면 가격 리밸런싱(역삼각형) 발생
        msg += "📉 **가격 변동 리밸런싱 (동적 비중조절) 발생**\n\n"
    
    # (3) 핵심 지표 요약
    msg += f"📊 **주요 지표 상태:**\n"
    msg += f"• RSI: `{latest_data['RSI']:.1f}`\n"
    msg += f"• MACD: `{latest_data['MACD']:.2f}`\n"
    msg += f"• VIX: `{latest_data['VIX']:.1f}`\n"
    
    if current_asset != prev_asset or today_events:
        # 변화가 있을 때는 무조건 알림
        send_telegram("🔔 TQQQ 자동매매 시그널 알림", msg)
    else:
        # 변화가 없을 때는 알림을 생략하거나 간략화 가능. (테스트를 위해 일단 보냄)
        # 만약 실제 운영 시 "변화 없을 땐 조용히" 하고 싶다면 아래 들여쓰기를 활성화하세요.
        # pass
        send_telegram("ℹ️ TQQQ 데일리 상태 보고 (변동 없음)", msg)
        print("상태 변동 없음. (테스트용 데일리 알림 전송됨)")

if __name__ == "__main__":
    run_daily_signal_check()
