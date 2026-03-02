import os
import sys
from flask import Flask

# 🚀 [로그] 부팅 시작을 알림 (로그 탐색기에서 확인 가능)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("--- 🚀 [인프라 진단용] 깡통 서버 부팅 시작 ---")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST', 'HEAD'])
def health_check():
    """구글 클라우드의 포트 8080 체크를 즉각 통과시킵니다."""
    print("✅ 구글 서버로부터 헬스체크 신호를 받았습니다!")
    return "SUCCESS: Infrastructure is Ready!", 200

@app.route('/test')
def test():
    return "Server is working perfectly!", 200

if __name__ == "__main__":
    # 구글 클라우드가 지정하는 PORT 환경변수를 사용하여 서버 가동 (기본 8080)
    port = int(os.environ.get("PORT", 8080))
    print(f"📡 {port} 포트에서 대기 시작... (Host: 0.0.0.0)")
    app.run(host='0.0.0.0', port=port)
