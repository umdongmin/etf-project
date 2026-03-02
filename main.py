import os
import sys
import threading
import time

# 🚨 로그 출력을 강제로 즉시 보이게 설정 (배포 디버깅의 핵심)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("--- 🚀 [BOOT] TQQQ Trading Bot Starting ---")

# 🌟 [1단계] 최상단에서 스트림릿 모듈을 가짜로 주입하여 라이브러리 충돌을 원천 차단합니다.
# core 폴더 등 다른 파일에서 'import streamlit'을 해도 에러가 나지 않게 "속이는" 마법입니다.
class Dummy:
    def __getattr__(self, name):
        if name in ('cache_data', 'cache_resource'):
            return lambda *args, **kwargs: (lambda f: f) # 데코레이터(@) 기능까지 지원
        return lambda *args, **kwargs: Dummy()
    def __call__(self, *args, **kwargs): return Dummy()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __bool__(self): return True

sys.modules['streamlit'] = Dummy()
print("✅ Streamlit Mocking Initialized")

try:
    from flask import Flask
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    @app.route('/alert', methods=['GET', 'POST'])
    def health_check():
        """구글 서버의 8080 포트 체크에 즉각 응답하여 성공판정을 유도합니다."""
        print("✅ Received request from Google Server. Health Check Passed.")
        
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        # 실제 분석은 0.1초 만에 응답 후, 별도 쓰레드에서 조용히 실행 (포트 타임아웃 방지)
        if token and chat_id:
            threading.Thread(target=run_analysis_internal, args=(token, chat_id)).start()
            return "Analysis Started in Background", 200
        else:
            print("⚠️ TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
            return "Health OK (Config Missing)", 200

    def run_analysis_internal(token, chat_id):
        """무거운 라이브러리 임포트와 실제 분석 로직을 여기서 수행합니다."""
        try:
            print("🚀 [배경작업] 라이브러리 로딩 및 데이터 분석 시작...")
            import json, datetime, requests
            from core.data import DataService
            from core.engine import StrategyEngine

            # 1. 전략 설정 파일 로드
            project_root = os.path.dirname(os.path.abspath(__file__))
            strat_path = os.path.join(project_root, "strategies", "tqqq_strategy.json")
            
            if not os.path.exists(strat_path):
                print(f"❌ 분석 실패: 전략 파일을 찾지 못했습니다 -> {strat_path}")
                return

            with open(strat_path, 'r', encoding='utf-8') as f:
                strat = json.load(f)

            # 2. 실시간 데이터 수집 및 전략 실행 (주요 분석 작업)
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
                salt='flask_vFinal_v22'
            )

            # 3. 분석 결과 알림 전송
            if not gh.empty:
                latest = gh.iloc[-1]
                last_date = latest.name.strftime('%Y-%m-%d')
                msg = f"🔔 **TQQQ 실시간 시그널 알림**\n\n"
                msg += f"📅 **기준일:** {last_date}\n"
                msg += f"🛡️ **현재 포지션:** `{latest['Asset']}`\n"
                
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
                print("✅ [배경작업] 분석 및 알림 전송 완료!")
                
        except Exception as e:
            print(f"❌ [배경작업 에러]: {e}")
            import traceback
            traceback.print_exc()

    if __name__ == "__main__":
        # 구글 클라우드가 요구하는 PORT 환경변수에 맞춰 서버를 실행합니다.
        port = int(os.environ.get("PORT", 8080))
        print(f"📡 Flask Server Listening on 0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port)

except Exception as e:
    # 🚨 부팅 도중 꺼지는 원인을 로그에 상세히 남깁니다.
    print(f"!!! [부팅 시 치명적 오류 발생] !!!\n{e}")
    import traceback
    traceback.print_exc()
    time.sleep(10) # 로그 전송 시간 확보
    sys.exit(1)
