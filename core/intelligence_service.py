import os
import json
import sqlite3
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
    """AI 마켓 인텔리전스: 뉴스 + 리포트 통합 분석 및 지능형 리포트 생성 서비스"""
    
    def __init__(self, db_path=None):
        self.llm = LLMService()
        # [수정] data/ 폴더 의존성 제거. 기본값으로 루트 경로 숨김 파일 지정
        if db_path is None:
            self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".market_intelligence.db")
        else:
            self.db_path = db_path
            
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
                print(f"SQLAlchemy Engine Error (Supabase Intelligence): {e}")
        
        return create_engine(f"sqlite:///{self.db_path}")

    def _get_connection(self):
        """직접 연결 객체 반환 (DDL 및 호환성용)"""
        if self.supabase_url and HAS_PSYCOPG2:
            try:
                return psycopg2.connect(self.supabase_url)
            except Exception as e:
                print(f"Supabase Connection Error (Intelligence), falling back to local: {e}")
        
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """마켓 인텔리전스 테이블 초기화 및 마이그레이션"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_intelligence (
                        date TEXT PRIMARY KEY,
                        report_title TEXT,
                        content TEXT,
                        sentiment_score REAL,
                        reliability_score REAL,
                        source_count INTEGER,
                        source_metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='market_intelligence'")
                columns = [row[0] for row in cursor.fetchall()]
            else: # SQLite
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS market_intelligence (
                        date TEXT PRIMARY KEY,
                        report_title TEXT,
                        content TEXT,
                        sentiment_score REAL,
                        reliability_score REAL,
                        source_count INTEGER,
                        source_metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute("PRAGMA table_info(market_intelligence)")
                columns = [row[1] for row in cursor.fetchall()]

            # 신규 컬럼 마이그레이션
            new_cols = {
                'reliability_score': 'REAL DEFAULT 0.5',
                'source_count': 'INTEGER DEFAULT 0'
            }
            for col, col_type in new_cols.items():
                if col not in columns:
                    try:
                        cursor.execute(f"ALTER TABLE market_intelligence ADD COLUMN {col} {col_type}")
                        print(f"Migration: Added '{col}' to market_intelligence table.")
                    except: pass

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Intelligence DB Init Error: {e}")

    def generate_daily_report(self, date_str, news_headlines, analyst_reports=None):
        """뉴스 및 애널리스트 리포트를 결합하여 심층 인텔리전스 리포트 생성"""
        if not self.llm.client:
            return None
            
        news_context = "\n".join(news_headlines) if isinstance(news_headlines, list) else news_headlines
        analyst_context = analyst_reports if analyst_reports else "제공된 상세 리포트 없음 (뉴스 기반 분석 수행)"
        
        prompt = f"""
        당신은 월스트리트 출신의 'AI 수석 전략가'입니다. 
        {date_str}의 시장 데이터(뉴스 및 시황 리포트)를 바탕으로 퀀트 투자자들을 위한 지능형 마켓 리포트를 작성하세요.

        [데이터 소스 - 뉴스 헤드라인]:
        {news_context}

        [데이터 소스 - 애널리스트/시황 리포트 요약]:
        {analyst_context}

        [작성 지침]:
        1. **심층 분석**: 단순 요약을 넘어, 이 뉴스들이 '왜' 중요한지, 시장의 '숨은 의도'가 무엇인지 통찰을 제시하세요.
        2. **상충 분석**: 뉴스 데이터와 애널리스트 리포트 사이의 상충되는 의견이 있다면 이를 날카롭게 짚어내세요.
        3. **신뢰성 검증**: 수집된 데이터의 전반적인 신뢰도를 평가하고, 불확실성이 큰 경우 경고를 포함하세요.
        4. **정형화 데이터**: 전체적인 시장 심리 점수를 -1.0(공포)에서 1.0(탐욕) 사이로 계산하세요.
        5. **투자 전략 제언**: 현재 상황에서 RSI 임계치나 MACD 민감도 등 퀀트 파라미터를 어떻게 보정해야 할지 '숫자'를 포함하여 구체적으로 조언하세요.
        6. **가독성**: Markdown 형식을 사용하여 전문적인 수준의 미려한 리포트를 작성하세요.

        [주의사항]:
        - 반드시 유효한 JSON 형식이어야 합니다.
        - `content` 필드 내부의 Markdown 텍스트는 JSON 문자열 규격에 맞게 특수문자(따옴표, 역슬래시 등)가 적절히 이스케이프(Escape)되어야 합니다.
        
        [출력 형식 (JSON)]:
        {{
            "report_title": "{date_str} AI 마켓 인텔리전스: [핵심 키워드]",
            "content": "# 마켓 요약\\n...내용...",
            "sentiment_score": 0.0,
            "reliability_score": 0.0,
            "source_count": {len(news_headlines) if isinstance(news_headlines, list) else 1}
        }}
        """
        
        try:
            response = self.llm.client.models.generate_content(
                model=self.llm.model_name,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            
            # JSON 텍스트 정제 (불필요한 공백/마크다운 코드 블록 방어)
            resp_text = response.text.strip()
            if resp_text.startswith("```json"):
                resp_text = resp_text.split("```json")[1].split("```")[0].strip()
            elif resp_text.startswith("```"):
                resp_text = resp_text.split("```")[1].split("```")[0].strip()
                
            res_json = json.loads(resp_text)
            
            # DB 저장
            self.save_report(date_str, res_json, news_headlines)
            
            return res_json
        except Exception as e:
            print(f"Generate Daily Report Error: {e}")
            return None

    def save_report(self, date_str, report_data, source_metadata=None):
        """생성된 리포트를 DB에 저장"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            metadata_json = json.dumps(source_metadata) if source_metadata else "{}"
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                sql = '''
                    INSERT INTO market_intelligence 
                    (date, report_title, content, sentiment_score, reliability_score, source_count, source_metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date) DO UPDATE SET
                    report_title = EXCLUDED.report_title,
                    content = EXCLUDED.content,
                    sentiment_score = EXCLUDED.sentiment_score,
                    reliability_score = EXCLUDED.reliability_score,
                    source_count = EXCLUDED.source_count,
                    source_metadata = EXCLUDED.source_metadata,
                    created_at = CURRENT_TIMESTAMP
                '''
            else: # SQLite
                sql = '''
                    INSERT OR REPLACE INTO market_intelligence 
                    (date, report_title, content, sentiment_score, reliability_score, source_count, source_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                
            cursor.execute(sql, (
                date_str, 
                report_data['report_title'], 
                report_data['content'], 
                report_data['sentiment_score'],
                report_data.get('reliability_score', 0.5),
                report_data.get('source_count', 0),
                metadata_json
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Intelligence Report Error: {e}")
            return False

    def get_report(self, date_str):
        """특정 날짜의 리포트 조회"""
        try:
            conn = self._get_connection()
            # PostgreSQL에서 DictCursor 사용 가능여부 체크
            if hasattr(conn, 'dsn') and HAS_PSYCOPG2:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute("SELECT * FROM market_intelligence WHERE date = %s", (date_str,))
            else:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM market_intelligence WHERE date = ?", (date_str,))
                
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return dict(row)
            return None
        except Exception as e:
            print(f"Get Intelligence Report Error: {e}")
            return None

    def get_all_reports(self, limit=30):
        """최근 리포트 목록 조회"""
        try:
            conn = self._get_connection()
            query = "SELECT date, report_title, sentiment_score, created_at FROM market_intelligence ORDER BY date DESC LIMIT %s"
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, (limit,))
            else: # SQLite
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query.replace("%s", "?"), (limit,))
                
            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"Get All Reports Error: {e}")
            return []
