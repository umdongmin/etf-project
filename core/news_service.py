import os
import json
import datetime
import pandas as pd
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
import requests
from core.llm_service import LLMService

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

class NewsService:
    """경제 뉴스 수집 및 Gemini 기반 정형화 서비스 (Supabase/PostgreSQL 전용)"""
    
    def __init__(self):
        self.llm = LLMService()
        self.supabase_url = os.getenv("SUPABASE_DB_URL")
        if not self.supabase_url:
            raise ValueError("SUPABASE_DB_URL 환경 변수가 설정되지 않았습니다.")
            
        self.engine = self._get_engine()
        self._init_db()

    def _get_engine(self):
        if not HAS_SQLALCHEMY or not self.supabase_url: return None
        url = self.supabase_url
        if url.startswith("postgres://"): url = url.replace("postgres://", "postgresql://", 1)
        try: return create_engine(url)
        except: return None

    def _get_connection(self):
        if not HAS_PSYCOPG2: raise ImportError("psycopg2-binary 라이브러리가 필요합니다.")
        return psycopg2.connect(self.supabase_url)

    def _get_placeholder(self, conn): return "%s"

    def _init_db(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    date TEXT, source TEXT DEFAULT 'yfinance',
                    sentiment REAL, impact REAL, reliability REAL DEFAULT 0.5,
                    topic TEXT, summary TEXT, raw_headlines TEXT,
                    q_score REAL, z_score REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (date, source)
                )
            ''')
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='news_sentiment'")
            columns = [row[0] for row in cursor.fetchall()]
            if 'q_score' not in columns: cursor.execute("ALTER TABLE news_sentiment ADD COLUMN q_score REAL")
            if 'z_score' not in columns: cursor.execute("ALTER TABLE news_sentiment ADD COLUMN z_score REAL")
            if 'updated_at' not in columns: cursor.execute("ALTER TABLE news_sentiment ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            conn.commit(); conn.close()
        except Exception as e: print(f"News DB Init Error: {e}")

    def score_news_with_gemini(self, date_str, headlines):
        if not self.llm.client: return None
        headlines_text = "\n".join(headlines) if isinstance(headlines, list) else headlines
        prompt = f"""
        당신은 20년 경력의 미국 기술주(NASDAQ, QQQ) 전문 퀀트 트레이더이자 수석 전략가입니다.
        미국 주식 시장의 유동성 및 주가 상승/하락에 미칠 영향을 평가하세요.
        [뉴스 헤드라인]: {headlines_text}
        (반드시 JSON 형식) {{"sentiment": -1~1, "impact": 0~1, "reliability": 0~1, "topic": "...", "summary": "..."}}
        """
        try:
            response = self.llm.client.models.generate_content(
                model=self.llm.model_name, contents=prompt,
                config={'response_mime_type': 'application/json', 'temperature': 0.0}
            )
            return json.loads(response.text)
        except: return None

    def save_sentiment(self, date_str, news_data, raw_headlines="", source="yfinance", q_score=None, z_score=None):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            sql = '''
                INSERT INTO news_sentiment 
                (date, source, sentiment, impact, reliability, topic, summary, raw_headlines, q_score, z_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date, source) DO UPDATE SET
                sentiment = EXCLUDED.sentiment, impact = EXCLUDED.impact, reliability = EXCLUDED.reliability,
                topic = EXCLUDED.topic, summary = EXCLUDED.summary, raw_headlines = EXCLUDED.raw_headlines,
                q_score = EXCLUDED.q_score, z_score = EXCLUDED.z_score, updated_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql, (
                date_str, source, news_data['sentiment'], news_data['impact'], news_data.get('reliability', 0.5),
                news_data['topic'], news_data['summary'], raw_headlines, q_score, z_score
            ))
            conn.commit(); conn.close()
            # 자동 업데이트 유도
            if q_score is None: self.update_all_indicators()
            return True
        except: return False

    def update_all_indicators(self, rolling_window=60):
        log_msg = f"🔄 [{datetime.datetime.now()}] AI 지표 재계산 시작 (Window={rolling_window})"
        try:
            df = self.get_sentiment_data()
            if df.empty:
                print("⚠️ [update_all_indicators] 데이터가 없어 업데이트를 중단합니다.")
                return
            
            # 1. 일별 가중 평균 계산 (중복 날짜 제거 및 가중치 반영)
            df['weighted_score'] = df['sentiment'] * df['impact'] * df.get('reliability', 0.5)
            # 날짜(index)별 그룹화
            daily_sum = df.groupby(df.index).apply(lambda x: (x['weighted_score'].sum() / (x['impact'] * x.get('reliability', 0.5)).sum()) if (x['impact'] * x.get('reliability', 0.5)).sum() != 0 else x['sentiment'].mean())
            daily_sum = daily_sum.sort_index() # 시간 순 정렬 보장
            
            # 2. 롤링 지수 계산 (min_periods=1로 설정하여 데이터가 1개라도 있으면 계산)
            q_scores = daily_sum.rolling(window=rolling_window, min_periods=1).rank(pct=True)
            rolling_mean = daily_sum.rolling(window=rolling_window, min_periods=1).mean()
            rolling_std = daily_sum.rolling(window=rolling_window, min_periods=1).std()
            z_scores = (daily_sum - rolling_mean) / (rolling_std + 1e-9)
            
            # 3. 업데이트 데이터 구성
            update_data = []
            for date_ts, q_val in q_scores.items():
                dt_str = date_ts.strftime('%Y-%m-%d')
                z_val = z_scores[date_ts]
                
                # 수치 안정화 (NaN -> None)
                q_val_clean = float(q_val) if not pd.isna(q_val) else None
                z_val_clean = float(z_val) if not pd.isna(z_val) else None
                
                update_data.append((q_val_clean, z_val_clean, dt_str))
                
            # 4. 수파베이스(Postgres) 배치 업데이트
            if HAS_PSYCOPG2 and update_data:
                conn = self._get_connection()
                cursor = conn.cursor()
                
                # 명시적 캐스팅을 포함한 배치 업데이트
                sql = """
                    UPDATE news_sentiment AS n
                    SET q_score = CAST(v.q AS REAL), 
                        z_score = CAST(v.z AS REAL)
                    FROM (VALUES %s) AS v(q, z, d)
                    WHERE n.date = CAST(v.d AS TEXT)
                """
                execute_values(cursor, sql, update_data)
                conn.commit()
                conn.close()
                log_msg += f" | ✅ {len(update_data)}일치 데이터 일괄 업데이트 완료"
            else:
                log_msg += " | ⚠️ 업데이트 가능한 데이터가 없거나 psycopg2 라이브러리가 없습니다."
        except Exception as e:
            log_msg += f" | ❌ 에러 발생: {str(e)}"
            if 'conn' in locals() and conn: conn.close()
        
        print(log_msg)
        with open("update.log", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

    def get_sentiment_data(self, start_date=None, end_date=None, source=None):
        try:
            query = "SELECT * FROM news_sentiment"
            conditions = []
            if start_date: conditions.append(f"date >= '{start_date}'")
            if end_date: conditions.append(f"date <= '{end_date}'")
            if source: conditions.append(f"source = '{source}'")
            if conditions: query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY date ASC, reliability DESC"
            db_conn = self.engine if self.engine else self._get_connection()
            df = pd.read_sql_query(query, db_conn)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            return df
        except: return pd.DataFrame()

    def check_date_exists(self, date_str):
        try:
            conn = self._get_connection(); cursor = conn.cursor()
            cursor.execute(f"SELECT 1 FROM news_sentiment WHERE date = %s", (date_str,))
            exists = cursor.fetchone() is not None
            conn.close(); return exists
        except: return False

    def fetch_headlines_from_yf(self, target_date=None, tickers=['QQQ', 'SPY', 'DIA', '^IXIC', '^TNX', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']):
        headlines = []
        target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                news_items = getattr(t, 'news', [])
                for item in news_items:
                    pub_time = item.get('providerPublishTime')
                    if pub_time:
                        item_dt = datetime.datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
                        if item_dt != target_dt_str: continue
                    title = item.get('title') or item.get('headline'); headlines.append(title)
            except: continue
        return list(set([h.strip() for h in headlines if h]))

    def fetch_headlines_from_rss(self, target_date=None):
        rss_urls = ["https://www.investing.com/rss/news_25.rss", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    pub_struct = entry.get('published_parsed')
                    if pub_struct:
                        item_dt = datetime.datetime(*pub_struct[:3]).strftime("%Y-%m-%d")
                        if item_dt != target_dt_str: continue
                    headlines.append(entry.get('title', ''))
            except: continue
        return list(set(headlines))

    def fetch_headlines_from_macro_report(self, target_date=None):
        url = "https://finance.naver.com/research/market_info_list.naver"
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        naver_dt = target_dt_str[2:].replace('-', '.')
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}); resp.encoding = 'euc-kr'
            soup = BeautifulSoup(resp.text, 'html.parser')
            for row in soup.find_all('tr'):
                tds = row.find_all('td')
                if len(tds) >= 4 and tds[3].text.strip() == naver_dt: headlines.append(tds[0].text.strip())
        except: pass
        return list(set(headlines))

    def fetch_headlines_from_finviz(self, target_date=None):
        url = "https://finviz.com/news.ashx"
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.text, 'html.parser')
            for a in soup.select('a.nn-tab-link'): headlines.append(a.text.strip())
        except: pass
        return list(set(headlines))
