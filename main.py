import os
from flask import Flask

# 🚀 [로그] 부팅 시작을 알림 (로그 탐색기에서 확인 가능)
print("--- 🚀 Standard Flask Server Starting ---", flush=True)

app = Flask(__name__)

@app.route('/')
@app.route('/test')
def health_check():
    """구글 클라우드의 포트 8080 체크를 즉각 통과시킵니다."""
    print("✅ Health Check Received", flush=True)
    return "SUCCESS: Deployment is working! (Dummy Mode)", 200

if __name__ == "__main__":
    # 로컬 테스트용 기본 설정
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
