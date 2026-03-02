import os
import json
import datetime
import threading
import http.server
import socketserver
import requests

# 1. 시그널 계산 및 텔레그램 전송 (메인 로직)
def run_logic_background():
    print("🚀 [배경작업] 시그널 분석 및 알림 전송 시작...")
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("❌ 환경변수(TOKEN/ID)가 누락되었습니다.")
        return
    
    try:
        from core.data import DataService
        from core.engine import StrategyEngine
        
        # 전략 설정 파일 로드
        project_root = os.path.dirname(os.path.abspath(__file__))
        strat_path = os.path.join(project_root, "strategies", "TQQQ 전략.json")
        with open(strat_path, 'r', encoding='utf-8') as f:
            strat = json.load(f)

        # 실시간 데이터 수집 (VXN 등 포함)
        data_dict, fg_df, vix_df, _, _, _ = DataService.fetch_live_data()
        
        # 전략 실행
        golden_history, _, all_signal_events = StrategyEngine.run_golden_strategy(
            data_dict, fg_df, vix_df, strat.get('leverage_asset', 'TQQQ'), strat.get('base_asset', 'QQQ'),
            strat.get('cash_ratio_pct', 0.0) / 100.0, datetime.date(2010, 1, 1), datetime.date.today(), strat, 
            strat.get('trade_at', '종가'), salt='alert_bot_final_v2'
        )

        if not golden_history.empty:
            latest = golden_history.iloc[-1]
            last_date = latest.name.strftime('%Y-%m-%d')
            current_asset = latest['Asset']
            
            msg = f"� **TQQQ 알림 (Cloud Run)**\n\n"
            msg += f"� **기준일:** {last_date}\n"
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
        print(f"❌ [배경작업] 에러 발생: {e}")

# 2. 아주 단순한 웹 서버 (구글의 8080 찌르기에 즉각 대답용)
class SimpleHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_success()
        
    def do_POST(self):
        self.send_success()
        
    def send_success(self):
        # 구글 서버에게 0.001초 만에 "나 살아있어!"라고 먼저 보고합니다.
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"OK")
        
        # 실제 데이터 수집 로직은 구글 응답과 별개로 뒤에서(Thread) 천천히 실행합니다.
        threading.Thread(target=run_logic_background).start()

    def log_message(self, format, *args):
        # 지저분한 로그 출력 방지
        pass

# 3. 메인 실행부 (포트 8080 열고 무한 대기)
if __name__ == "__main__":
    # 구글 클라우드가 요구하는 PORT 환경변수를 읽어 8080 포트를 점유합니다.
    port = int(os.environ.get("PORT", "8080"))
    print(f"🚀 [최종모드] {port} 포트에서 8080 에러 방어 중...")
    
    with socketserver.TCPServer(("0.0.0.0", port), SimpleHandler) as httpd:
        # 이 무한 루프가 돌아가야 구글 클라우드가 서비스가 살아있다고 판단합니다.
        httpd.serve_forever()

# [함수 진입점] 혹시라도 필요할까봐 남겨두는 더미 코드
def alert_handler(request):
    return "OK", 200
