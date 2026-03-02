import os
import sys
import threading
import time

# 🚨 [최하단 로그 출력 강제화] 구글 클라우드에서 에러 메시지가 즉시 보이도록 설정합니다.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("--- 🚀 [시스템 부팅] 분석 봇이 시작되었습니다 ---")

try:
    # 🌟 [1단계] 최단 시간 내에 Streamlit Mocking 완료 (충돌 방지 1순위)
    class DummyStreamlit:
        def __getattr__(self, name):
            if name in ('cache_data', 'cache_resource'):
                return lambda *args, **kwargs: (lambda f: f)
            return lambda *args, **kwargs: DummyStreamlit()
        def __call__(self, *args, **kwargs): return DummyStreamlit()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __bool__(self): return True

    sys.modules['streamlit'] = DummyStreamlit()
    print("✅ Streamlit 모듈 가짜 주입 완료")

    # Flask 서버 설정 (가장 안정적인 웹 서버 방식)
    from flask import Flask
    app = Flask(__name__)

    @app.route('/', methods=['GET', 'POST'])
    def health_check():
        """구글 서버의 8080 포트 체크에 즉각 응답하여 성공판정을 유도합니다."""
        print("✅ 구글 서버 헬스체크 신호 수신!")
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        
        # 실제 분석 작업은 응답을 보낸 직후 백그라운드에서 진행
        if token and chat_id:
            threading.Thread(target=heavy_analysis_task, args=(token, chat_id)).start()
            return "OK", 200
        else:
            print("⚠️ 환경 변수(TOKEN/ID)가 설정되지 않았습니다.")
            return "OK (Missing Config)", 200

    def heavy_analysis_task(token, chat_id):
        """무거운 라이브러리를 로드하고 데이터 분석을 수행하는 백그라운드 작업"""
        try:
            print("🚀 [배경작업] 라이브러리 로딩 및 분석 시작...")
            import json
            import datetime
            import requests
            from core.data import DataService
            from core.engine import StrategyEngine

            # 1. 전략 설정 로드 (파일명: tqqq_strategy.json)
            base_path = os.path.dirname(os.path.abspath(__file__))
            strat_path = os.path.join(base_path, "strategies", "tqqq_strategy.json")
            
            if not os.path.exists(strat_path):
                print(f"❌ 전략 파일을 찾을 수 없음: {strat_path}")
                return

            with open(strat_path, 'r', encoding='utf-8') as f:
                strat = json.load(f)

            # 2. 데이터 수집 및 시그널 분석
            data, fg, vix, _, _, _ = DataService.fetch_live_data()
            gh, _, events = StrategyEngine.run_golden_strategy(
                data, fg, vix, 
                strat.get('leverage_asset', 'TQQQ'), 
                strat.get('base_asset', 'QQQ'),
                strat.get('cash_ratio_pct', 0.0) / 100.0, 
                datetime.date(2010, 1, 1), 
                datetime.date.today(), 
                strat, 
                strat.get('trade_at', '종가'), 
                salt='final_v20_flask'
            )

            # 3. 분석 결과 전송
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

except Exception as e:
    # 🚨 부팅 도중 꺼지는 원인을 로그에 상세히 남깁니다.
    print(f"!!! [부팅 시 치명적 오류 발생] !!!\n{e}")
    import traceback
    traceback.print_exc()
    time.sleep(5) # 로그 전송 시간 확보
    sys.exit(1)

if __name__ == "__main__":
    # 구글 클라우드가 지정한 포트로 서버 실행
    port = int(os.environ.get("PORT", 8080))
    print(f"📡 Flask Server Listening on 0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port)
