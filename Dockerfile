# Python 3.10 버전 사용 (트레이딩 라이브러리 호환성 고려)
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 빌드에 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# Cloud Run의 PORT 환경변수 대응 (gunicorn 사용 시)
# 현재 로그 상 Entrypoint가 gunicorn이므로 아래 설정을 따릅니다.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app