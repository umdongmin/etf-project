import os
import json
import sqlite3
import datetime
import pandas as pd
import pandas as pd
try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
import yfinance as yf
import feedparser
from core.llm_service import LLMService

class NewsService:
    """경제 뉴스 수집 및 Gemini 기반 정형화 서비스"""
    
    def __init__(self, db_path=None):
        self.llm = LLMService()
        # [수정] data/ 폴더 의존성 제거. 기본값으로 루트 경로 숨김 파일 지정
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".news_sentiment.db")
        else:
            self.db_path = db_path
        
        # Supabase (PostgreSQL) 설정 확인
        self.supabase_url = os.getenv("SUPABASE_DB_URL")
        self.engine = self._get_engine()
        
        self._init_db()

    def _get_engine(self):
        """SQLAlchemy 엔진 객체 반환"""
        if not HAS_SQLALCHEMY:
            return None
            
        if self.supabase_url:
            url = self.supabase_url
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            try:
                return create_engine(url)
            except Exception as e:
                print(f"SQLAlchemy Engine Error (Supabase): {e}")
        
        # SQLite
        return create_engine(f"sqlite:///{self.db_path}")

    def _get_connection(self):
        """직접 연결 객체 반환 (DDL 실행용)"""
        if self.supabase_url and HAS_PSYCOPG2:
            try:
                return psycopg2.connect(self.supabase_url)
            except Exception as e:
                print(f"Supabase Direct Connection Error, falling back to local: {e}")
        
        return sqlite3.connect(self.db_path)

    def _get_placeholder(self, conn):
        """DB 타입에 따른 플레이스홀더 반환 (? for SQLite, %s for Postgres)"""
        # SQLAlchemy 엔진을 사용할 때는 보통 %s나 ?가 자동으로 처리되지만, 직접 쿼리용
        if hasattr(conn, 'dsn'): 
            return "%s"
        return "?"

    def _init_db(self):
        """뉴스 심리 데이터 테이블 초기화 및 마이그레이션"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 기본 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    date TEXT PRIMARY KEY,
                    sentiment REAL,
                    impact REAL,
                    reliability REAL,
                    topic TEXT,
                    summary TEXT,
                    raw_headlines TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 컬럼 존재 여부 확인 및 마이그레이션
            if hasattr(conn, 'dsn'): # PostgreSQL
                cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='news_sentiment'")
                columns = [row[0] for row in cursor.fetchall()]
            else: # SQLite
                cursor.execute("PRAGMA table_info(news_sentiment)")
                columns = [row[1] for row in cursor.fetchall()]
                
            if 'reliability' not in columns:
                try:
                    cursor.execute("ALTER TABLE news_sentiment ADD COLUMN reliability REAL DEFAULT 0.5")
                    print("Migration: Added 'reliability' column to news_sentiment table.")
                except Exception as e:
                    print(f"Migration Error (reliability): {e}")

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"DB Init Error: {e}")

    def score_news_with_gemini(self, date_str, headlines):
        """Gemini 2.5 Flash를 사용하여 뉴스 헤드라인 수치화"""
        if not self.llm.client:
            print("LLM Client not available")
            return None
            
        headlines_text = "\n".join(headlines) if isinstance(headlines, list) else headlines
        
        prompt = f"""
        당신은 20년 경력의 금융 뉴스 분석 전문가입니다. 
        다음은 {date_str}의 주요 경제 뉴스 헤드라인들입니다.
        이 뉴스들이 시장 참여자들의 심리에 미치는 영향과 중요도를 분석하여 정해진 JSON 형식으로 응답해주세요.

        [뉴스 헤드라인]:
        {headlines_text}

        [출력 형식 (JSON 형식만 출력)]:
        {{
            "sentiment": -1.0 (매우 비관/공포) ~ 1.0 (매우 낙관/탐욕) 사이의 실수값,
            "impact": 0.0 (낮음) ~ 1.0 (매우 높음) 사이의 영향력 강도,
            "reliability": 0.0 (신뢰할 수 없는 노이즈) ~ 1.0 (매우 신뢰할 수 있는 팩트) 사이의 값,
            "topic": "핵심 키워드 (예: 금리 동결, 고용 지표 쇼크 등)",
            "summary": "전체적인 뉴스 분위기 한줄 요약"
        }}
        
        주의: 분석 시 기술적 분석보다는 거시 경제적 관점과 투자자 심리 변화에 집중하세요. 
        특히 여러 매체에서 공통적으로 다루는 내용은 'reliability' 점수를 높게 부여하세요.
        """
        
        try:
            # JSON 응답 강제 (Gemini 1.5/2.5 이상 지원)
            response = self.llm.client.models.generate_content(
                model=self.llm.model_name,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            # 응답 파싱
            res_json = json.loads(response.text)
            return res_json
        except Exception as e:
            print(f"Gemini News Scoring Error: {e}")
            return None

    def save_sentiment(self, date_str, news_data, raw_headlines=""):
        """분석된 결과를 DB에 저장"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholder = self._get_placeholder(conn)
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                sql = '''
                    INSERT INTO news_sentiment 
                    (date, sentiment, impact, reliability, topic, summary, raw_headlines)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                    sentiment = EXCLUDED.sentiment,
                    impact = EXCLUDED.impact,
                    reliability = EXCLUDED.reliability,
                    topic = EXCLUDED.topic,
                    summary = EXCLUDED.summary,
                    raw_headlines = EXCLUDED.raw_headlines,
                    updated_at = CURRENT_TIMESTAMP
                '''
            else: # SQLite
                sql = '''
                    INSERT OR REPLACE INTO news_sentiment 
                    (date, sentiment, impact, reliability, topic, summary, raw_headlines)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                
            cursor.execute(sql, (
                date_str, 
                news_data['sentiment'], 
                news_data['impact'], 
                news_data.get('reliability', 0.5),
                news_data['topic'], 
                news_data['summary'],
                raw_headlines
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Sentiment Error: {e}")
            return False

    def get_sentiment_data(self, start_date=None, end_date=None):
        """저장된 심리 지표를 DataFrame으로 조회"""
        try:
            query = "SELECT * FROM news_sentiment"
            conditions = []
            if start_date: conditions.append(f"date >= '{start_date}'")
            if end_date: conditions.append(f"date <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY date ASC"
            
            # SQLAlchemy 엔진이 있으면 엔진 사용, 없으면 직접 연결 사용
            db_conn = self.engine if self.engine else self._get_connection()
            df = pd.read_sql_query(query, db_conn)
            
            if self.engine is None and hasattr(db_conn, 'close'):
                db_conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except Exception as e:
            print(f"Get Sentiment Data Error: {e}")
            return pd.DataFrame()
        finally:
            pass

    def check_date_exists(self, date_str):
        """해당 날짜의 데이터가 이미 있는지 확인"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            placeholder = self._get_placeholder(conn)
            cursor.execute(f"SELECT 1 FROM news_sentiment WHERE date = {placeholder}", (date_str,))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except:
            return False

    def fetch_headlines_from_yf(self, tickers=['QQQ', 'SPY', 'DIA', '^IXIC', '^TNX', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']):
        """yfinance를 통해 주요 티커의 최신 뉴스 헤드라인 수집 (디버깅 강화)"""
        headlines = []
        print(f"--- 뉴스 수집 시작 ({len(tickers)}개 티커 시도) ---")
        
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                # hasattr 대신 getattr로 안전하게 접근
                news_items = getattr(t, 'news', [])
                if news_items:
                    print(f" [OK] {ticker}: {len(news_items)}건 발견")
                    for item in news_items:
                        # 다양한 키 시도 (yfinance 버전에 따라 다를 수 있음)
                        title = item.get('title') or item.get('headline') or item.get('summary')
                        if not title and isinstance(item, dict):
                            # 첫 번째 문자열 값을 타이틀로 시도 (비상시)
                            for k, v in item.items():
                                if k != 'link' and isinstance(v, str) and len(v) > 10:
                                    title = v
                                    break
                        
                        if title and title not in headlines:
                            headlines.append(title)
                else:
                    print(f" [EMPTY] {ticker}: 뉴스가 없습니다.")
            except Exception as e:
                print(f" [ERROR] {ticker}: {e}")
        
        # 중복 제거 및 공백 정리
        final_headlines = list(set([h.strip() for h in headlines if len(h.strip()) > 5]))
        print(f"--- 수집 완료: 총 {len(final_headlines)}개의 고유 헤드라인 확보 ---")
        return final_headlines

    def fetch_headlines_from_rss(self):
        """전문 금융 RSS 피드에서 고품질 뉴스 수집"""
        rss_urls = [
            "https://www.investing.com/rss/news_25.rss",     # Stock Market News
            "https://www.investing.com/rss/news_301.rss",    # Economy News
            "https://www.investing.com/rss/market_overview_opinion.rss", # Opinions/Analysis
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", # WSJ Markets
        ]
        
        headlines = []
        print(f"--- RSS 고품질 뉴스 수집 시작 ({len(rss_urls)}개 소스) ---")
        
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                if feed.entries:
                    print(f" [OK] {url.split('/')[-1]}: {len(feed.entries)}건 발견")
                    for entry in feed.entries:
                        title = entry.get('title', '')
                        summary = entry.get('summary', '') or entry.get('description', '')
                        
                        # 제목과 요약을 합쳐서 더 풍부한 맥락 확보
                        full_content = f"[{title}] {summary}" if summary else title
                        if full_content and full_content not in headlines:
                            headlines.append(full_content)
                else:
                    print(f" [EMPTY] {url.split('/')[-1]}: 뉴스가 없습니다.")
            except Exception as e:
                print(f" [ERROR] {url}: {e}")
                
        final_headlines = list(set([h.strip() for h in headlines if len(h.strip()) > 10]))
        print(f"--- RSS 수집 완료: 총 {len(final_headlines)}개의 전문 데이터 확보 ---")
        return final_headlines
