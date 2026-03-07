import os
import json
import datetime
import pandas as pd
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from core.llm_service import LLMService

class IntelligenceService:
    """AI 마켓 인텔리전스: 뉴스 + 리포트 통합 분석 및 지능형 리포트 생성 서비스 (Supabase/PostgreSQL 전용)"""
    
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

    def _init_db(self):
        try:
            conn = self._get_connection(); cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_intelligence (
                    date TEXT PRIMARY KEY, report_title TEXT, content TEXT,
                    sentiment_score REAL, reliability_score REAL DEFAULT 0.5,
                    source_count INTEGER DEFAULT 0, source_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='market_intelligence'")
            columns = [row[0] for row in cursor.fetchall()]
            if 'reliability_score' not in columns: cursor.execute("ALTER TABLE market_intelligence ADD COLUMN reliability_score REAL DEFAULT 0.5")
            if 'source_count' not in columns: cursor.execute("ALTER TABLE market_intelligence ADD COLUMN source_count INTEGER DEFAULT 0")
            conn.commit(); conn.close()
        except Exception as e: print(f"Intelligence DB Init Error: {e}")

    def sync_market_intelligence(self, date_str, news_service, analyst_reports=None):
        """뉴스 수집 -> 데이터 정형화 -> 리포트 생성 통합 동기화"""
        if not self.llm.client: return None
        
        # 1. 뉴스 데이터 수집
        macro = news_service.fetch_headlines_from_macro_report(date_str)
        micro = news_service.fetch_headlines_from_finviz(date_str)
        all_headlines = list(set(macro + micro))
        if not all_headlines: return None

        # 2. 정량 데이터 생성 및 저장
        quant_data = news_service.score_news_with_gemini(date_str, all_headlines)
        if quant_data:
            news_service.save_sentiment(date_str, quant_data, raw_headlines="\n".join(all_headlines), source="integrated_quant")

        # 3. 정성 리포트 생성
        prompt = f"당신은 월스트리트 전략가입니다. 다음 정형화 데이터와 뉴스 헤드라인을 바탕으로 마크다운 리포트를 작성하세요.\n\n[데이터]: {quant_data}\n[뉴스]: {'/'.join(all_headlines)}"
        try:
            response = self.llm.client.models.generate_content(model=self.llm.model_name, contents=prompt, config={'temperature': 0.3})
            resp_text = response.text.strip()
            report_data = {
                "date": date_str, "report_title": f"{date_str} AI 마켓 인텔리전스",
                "content": resp_text, "sentiment_score": quant_data.get('sentiment', 0.0),
                "reliability_score": quant_data.get('reliability', 0.5),
                "source_count": len(all_headlines), "source_metadata": all_headlines
            }
            self.save_report(date_str, report_data, all_headlines)
            return report_data
        except: return None

    def save_report(self, date_str, report_data, source_metadata=None):
        try:
            conn = self._get_connection(); cursor = conn.cursor()
            metadata_json = json.dumps(source_metadata) if source_metadata else "{}"
            sql = '''
                INSERT INTO market_intelligence 
                (date, report_title, content, sentiment_score, reliability_score, source_count, source_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                report_title = EXCLUDED.report_title, content = EXCLUDED.content,
                sentiment_score = EXCLUDED.sentiment_score, reliability_score = EXCLUDED.reliability_score,
                source_count = EXCLUDED.source_count, source_metadata = EXCLUDED.source_metadata,
                created_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql, (
                date_str, report_data['report_title'], report_data['content'], 
                report_data['sentiment_score'], report_data.get('reliability_score', 0.5),
                report_data.get('source_count', 0), metadata_json
            ))
            conn.commit(); conn.close(); return True
        except: return False

    def get_report(self, date_str):
        try:
            conn = self._get_connection(); cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM market_intelligence WHERE date = %s", (date_str,))
            row = cursor.fetchone(); conn.close(); return dict(row) if row else None
        except: return None

    def get_all_reports(self, limit=30):
        try:
            conn = self._get_connection(); cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT date, report_title, sentiment_score, created_at FROM market_intelligence ORDER BY date DESC LIMIT %s", (limit,))
            rows = cursor.fetchall(); conn.close(); return [dict(row) for row in rows]
        except: return []
