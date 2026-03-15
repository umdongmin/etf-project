import os
import json
import datetime
import streamlit as st

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

class StrategyStorage:
    """전략 설정을 Supabase/PostgreSQL로 관리하는 클래스 (PostgreSQL 전용)"""
    
    @classmethod
    def _get_connection(cls):
        """Supabase DB 연결 객체 반환"""
        supabase_url = os.getenv("SUPABASE_DB_URL")
        if not supabase_url or not HAS_PSYCOPG2:
            raise ValueError("SUPABASE_DB_URL 설정 또는 psycopg2 라이브러리가 필요합니다.")
        return psycopg2.connect(supabase_url)

    @classmethod
    def _init_db(cls):
        """전략 테이블 초기화 및 마이그레이션"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            # [1] strategies 테이블 정의
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY,
                    category TEXT DEFAULT 'equity',
                    params JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # [Migration] 'category' 컬럼 누락 시 추가
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='strategies' AND column_name='category'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE strategies ADD COLUMN category TEXT DEFAULT 'equity'")

            # [Migration] 초기 데이터 JSON -> DB 이전
            cursor.execute("SELECT COUNT(*) FROM strategies")
            if cursor.fetchone()[0] == 0:
                cls._auto_migrate_from_json(cursor, conn)
                
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Strategy DB Init Error: {e}")

    @classmethod
    def _auto_migrate_from_json(cls, cursor, conn):
        """기존 JSON 파일들에서 Supabase DB로 데이터 자동 이전"""
        strategy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")
        if not os.path.exists(strategy_dir): return

        json_files = [f for f in os.listdir(strategy_dir) if f.endswith(".json")]
        if not json_files: return

        print(f"--- Detected {len(json_files)} legacy strategy files. Auto-migrating to Supabase... ---")
        for filename in json_files:
            name = filename.replace(".json", "")
            filepath = os.path.join(strategy_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                params_json = json.dumps(params, ensure_ascii=False)
                cursor.execute("INSERT INTO strategies (name, params) VALUES (%s, %s) ON CONFLICT DO NOTHING", (name, params_json))
            except Exception as e:
                print(f" [ERROR] Failed to migrate {name}: {e}")
        conn.commit()

    @classmethod
    def save_strategy(cls, name, params, category='equity'):
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            params_json = json.dumps(params, ensure_ascii=False)
            
            sql = '''
                INSERT INTO strategies (name, category, params)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                category = EXCLUDED.category,
                params = EXCLUDED.params,
                updated_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql, (name, category, params_json))
            if cursor.rowcount > 0:
                conn.commit()
                st.cache_data.clear() # 데이터 변경 시 캐시 무효화
            conn.close()
            return name
        except Exception as e:
            print(f"Save Strategy Error: {e}")
            return None

    @classmethod
    @st.cache_data(ttl=60)
    def list_strategies(cls, category=None):
        """저장된 전략 이름 목록 조회 (캐시 적용)"""
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT name FROM strategies WHERE category = %s ORDER BY name ASC", (category,))
            else:
                cursor.execute("SELECT name FROM strategies ORDER BY name ASC")
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except Exception as e:
            print(f"List Strategy Error: {e}")
            return []

    @classmethod
    def load_strategy(cls, name):
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT params, category FROM strategies WHERE name = %s", (name,))
            row = cursor.fetchone()
            conn.close()
            if row:
                params = row[0]
                category = row[1]
                loaded = json.loads(params) if isinstance(params, str) else params
                return {**loaded, 'category': category} if isinstance(loaded, dict) else loaded
            return None
        except Exception as e:
            print(f"Load Strategy Error: {e}")
            return None

    @classmethod
    def delete_strategy(cls, name):
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM strategies WHERE name = %s", (name,))
            conn.commit()
            st.cache_data.clear() # 데이터 변경 시 캐시 무효화
            conn.close()
            return True
        except Exception as e:
            print(f"Delete Strategy Error: {e}")
            return False

    # --- Portfolio Persistence Methods ---

    @classmethod
    def _init_portfolio_db(cls):
        """포트폴리오 관련 테이블 초기화 및 마이그레이션"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            # [1] portfolios 메인 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    name TEXT PRIMARY KEY,
                    total_capital DOUBLE PRECISION,
                    rebalance_preset TEXT DEFAULT 'none',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # [Migration] 구형 'configs' 컬럼 감지 시 제거
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='portfolios' AND column_name='configs'")
            if cursor.fetchone():
                cursor.execute("ALTER TABLE portfolios DROP COLUMN configs")

            # [2] portfolio_items 관계 테이블 (정규화)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_items (
                    id SERIAL PRIMARY KEY,
                    portfolio_name TEXT REFERENCES portfolios(name) ON DELETE CASCADE,
                    strategy_name TEXT REFERENCES strategies(name) ON UPDATE CASCADE,
                    weight DOUBLE PRECISION,
                    sort_order INT,
                    UNIQUE(portfolio_name, strategy_name)
                )
            ''')

            # [Migration] 중복된 'params_override' 컬럼 감지 시 제거
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='portfolio_items' AND column_name='params_override'")
            if cursor.fetchone():
                cursor.execute("ALTER TABLE portfolio_items DROP COLUMN params_override")
            
            # [Migration] UNIQUE 제약 조건 강제 적용 (이미 테이블이 있는 경우 대비)
            # 중복 데이터가 있으면 제약 조건 추가가 실패하므로 우선 중복 제거
            cursor.execute('''
                DELETE FROM portfolio_items a USING portfolio_items b
                WHERE a.id > b.id 
                AND a.portfolio_name = b.portfolio_name 
                AND a.strategy_name = b.strategy_name
            ''')
            
            # 제약 조건 이름 확인 (이미 존재할 수도 있음)
            cursor.execute("""
                SELECT constraint_name 
                FROM information_schema.table_constraints 
                WHERE table_name = 'portfolio_items' AND constraint_type = 'UNIQUE'
            """)
            if not cursor.fetchone():
                try:
                    cursor.execute("ALTER TABLE portfolio_items ADD CONSTRAINT portfolio_items_unique_pair UNIQUE (portfolio_name, strategy_name)")
                except Exception as ex:
                    print(f"Migration: Failed to add UNIQUE constraint: {ex}")
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Portfolio DB Init Error: {e}")

    @classmethod
    def save_portfolio(cls, name, configs, total_capital, rebalance_preset='none'):
        """포트폴리오 정보를 정규화하여 저장"""
        cls._init_portfolio_db()
        # [Security] 공백 등 트리밍
        name = name.strip() if name else "Untitled"
        
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            # 1. portfolios 테이블 저장 (Upsert)
            sql_p = '''
                INSERT INTO portfolios (name, total_capital, rebalance_preset)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                total_capital = EXCLUDED.total_capital,
                rebalance_preset = EXCLUDED.rebalance_preset,
                updated_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql_p, (name, total_capital, rebalance_preset))
            
            # 2. 기존 아이템 삭제 후 재삽입 (정규화된 테이블의 원칙)
            cursor.execute("DELETE FROM portfolio_items WHERE portfolio_name = %s", (name,))
            
            sql_i = '''
                INSERT INTO portfolio_items (portfolio_name, strategy_name, weight, sort_order)
                VALUES (%s, %s, %s, %s)
            '''
            for idx, item in enumerate(configs):
                # item: { 'name': str, 'weight': float }
                # 정규화: 항상 글로벌 전략을 참조함 (params_override 제거됨)
                cursor.execute(sql_i, (name, item['name'], item['weight'], idx))
                
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return name
        except Exception as e:
            print(f"Save Portfolio Error (Normalized): {e}")
            return None

    @classmethod
    def load_portfolio(cls, name):
        """정규화된 테이블에서 포트폴리오 정보를 JOIN하여 로드"""
        cls._init_portfolio_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            # 1. 메인 정보 조회
            cursor.execute("SELECT total_capital, rebalance_preset FROM portfolios WHERE name = %s", (name,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                return None
                
            total_capital, rb_preset = row
            
            # 2. 아이템 및 연관된 전략 정보 JOIN 조회
            # strategies 테이블의 최신 params를 가져옴으로써 자동 동기화 실현
            sql = '''
                SELECT i.strategy_name, i.weight, s.category, s.params as global_params
                FROM portfolio_items i
                LEFT JOIN strategies s ON i.strategy_name = s.name
                WHERE i.portfolio_name = %s
                ORDER BY i.sort_order ASC
            '''
            cursor.execute(sql, (name,))
            rows = cursor.fetchall()
            
            configs = []
            for r in rows:
                strat_name, weight, category, g_params = r
                
                # 정규화 핵심: 글로벌 설정을 100% 사용 (params_override 제거)
                final_params = g_params if g_params else {}
                
                configs.append({
                    'name': strat_name,
                    'weight': weight,
                    'type': category if category else 'equity',
                    'params': final_params
                })
                
            conn.close()
            return {
                'total_capital': total_capital,
                'rebalance_preset': rb_preset,
                'configs': configs
            }
        except Exception as e:
            print(f"Load Portfolio Error (Normalized): {e}")
            return None

    @classmethod
    @st.cache_data(ttl=60)
    def list_portfolios(cls):
        """저장된 포트폴리오 목록 조회 (캐시 적용)"""
        cls._init_portfolio_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM portfolios ORDER BY updated_at DESC")
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except Exception as e:
            print(f"List Portfolios Error: {e}")
            return []

    @classmethod
    def delete_portfolio(cls, name):
        cls._init_portfolio_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM portfolios WHERE name = %s", (name,))
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return True
        except Exception as e:
            print(f"Delete Portfolio Error: {e}")
            return False
