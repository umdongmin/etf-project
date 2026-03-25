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

            # [Migration] is_default 컬럼 추가 (기본 전략 지정)
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='strategies' AND column_name='is_default'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE strategies ADD COLUMN is_default BOOLEAN DEFAULT FALSE")

            # [Migration] lev_params 컬럼 추가 — 채권 Layer2를 strategies에서 직접 관리
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='strategies' AND column_name='lev_params'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE strategies ADD COLUMN lev_params JSONB DEFAULT NULL")
                # 기존 번들 형식 {'params':{L1}, 'lev_params':{L2}} → 컬럼으로 분리
                cursor.execute("""
                    UPDATE strategies
                    SET lev_params = params->'lev_params',
                        params     = params->'params'
                    WHERE category = 'bond'
                      AND params ? 'lev_params'
                      AND params ? 'params'
                """)

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
    def save_strategy(cls, name, params, category='equity', lev_params=None):
        """전략 저장. 채권은 params(Layer1)와 lev_params(Layer2)를 분리하여 저장.

        하위 호환: params가 {'params':..., 'lev_params':...} 번들 형식이면 자동 분리.
        """
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            # 번들 형식 자동 분리 (구형 호환)
            if isinstance(params, dict) and 'params' in params and 'lev_params' in params and category == 'bond':
                lev_params = lev_params or params.get('lev_params')
                params = params.get('params', {})

            params_json = json.dumps(params, ensure_ascii=False)
            lev_params_json = json.dumps(lev_params, ensure_ascii=False) if lev_params is not None else None

            sql = '''
                INSERT INTO strategies (name, category, params, lev_params)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                category   = EXCLUDED.category,
                params     = EXCLUDED.params,
                lev_params = EXCLUDED.lev_params,
                updated_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql, (name, category, params_json, lev_params_json))
            if cursor.rowcount > 0:
                conn.commit()
                st.cache_data.clear()
            conn.close()
            return name
        except Exception as e:
            print(f"Save Strategy Error: {e}")
            return None

    @classmethod
    def list_strategies(cls, category=None):
        """저장된 전략 이름 목록 조회.

        @st.cache_data TTL=60s 래퍼를 통해 호출되므로,
        매 rerun마다 Supabase를 조회하지 않는다.
        저장/삭제 시 st.cache_data.clear()로 무효화됨.
        """
        return _cached_list_strategies(category)

    @classmethod
    def get_default_strategy(cls, category='equity'):
        """해당 카테고리의 기본 전략 이름 반환 (캐시 적용, 없으면 None)."""
        return _cached_get_default_strategy(category)

    @classmethod
    def load_strategy(cls, name):
        """전략 로드. 반환값: {..L1 params.., 'category':..., 'lev_params':... (채권만)}"""
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT params, category, lev_params FROM strategies WHERE name = %s", (name,))
            row = cursor.fetchone()
            conn.close()
            if row:
                params, category, lev_params_raw = row
                loaded = json.loads(params) if isinstance(params, str) else (params or {})

                # 하위 호환: 마이그레이션 전 번들 형식 처리
                if isinstance(loaded, dict) and 'params' in loaded and 'lev_params' in loaded and category == 'bond':
                    lev_params_raw = lev_params_raw or loaded.get('lev_params')
                    loaded = loaded.get('params', {})

                result = {**loaded, 'category': category} if isinstance(loaded, dict) else loaded
                if lev_params_raw is not None:
                    result['lev_params'] = json.loads(lev_params_raw) if isinstance(lev_params_raw, str) else lev_params_raw
                return result
            return None
        except Exception as e:
            print(f"Load Strategy Error: {e}")
            return None

    @classmethod
    def set_default_strategy(cls, name, category='equity'):
        """특정 전략을 해당 카테고리의 기본값으로 지정 (기존 기본값은 해제)"""
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE strategies SET is_default = FALSE WHERE category = %s", (category,))
            cursor.execute("UPDATE strategies SET is_default = TRUE WHERE name = %s AND category = %s", (name, category))
            conn.commit()
            conn.close()
            st.cache_data.clear()  # 캐시 무효화 → 다음 조회 시 DB 재조회
            return True
        except Exception as e:
            print(f"Set Default Strategy Error: {e}")
            return False

    @classmethod
    def _get_default_strategy_uncached(cls, category='equity'):
        """내부용 비캐시 버전 (set_default_strategy 후 직접 DB 조회 필요 시)."""
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM strategies WHERE category = %s AND is_default = TRUE LIMIT 1", (category,))
            row = cursor.fetchone()
            conn.close()
            return row[0] if row else None
        except Exception as e:
            print(f"Get Default Strategy Error: {e}")
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

            # [Migration] portfolio_items.lev_params 제거 — strategies.lev_params로 이관 완료
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='portfolio_items' AND column_name='lev_params'")
            if cursor.fetchone():
                cursor.execute("ALTER TABLE portfolio_items DROP COLUMN lev_params")

            # [Migration] is_default 컬럼 추가 (기본 포트폴리오 지정)
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='portfolios' AND column_name='is_default'")
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE portfolios ADD COLUMN is_default BOOLEAN DEFAULT FALSE")
            
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
            # strategies 테이블의 최신 params/lev_params를 가져옴으로써 자동 동기화 실현
            sql = '''
                SELECT i.strategy_name, i.weight, s.category, s.params, s.lev_params
                FROM portfolio_items i
                LEFT JOIN strategies s ON i.strategy_name = s.name
                WHERE i.portfolio_name = %s
                ORDER BY i.sort_order ASC
            '''
            cursor.execute(sql, (name,))
            rows = cursor.fetchall()

            configs = []
            for r in rows:
                strat_name, weight, category, g_params, lev_params_raw = r

                params_data = g_params if g_params else {}

                # 하위 호환: 마이그레이션 전 번들 형식 처리
                if isinstance(params_data, dict) and 'params' in params_data and 'lev_params' in params_data and category == 'bond':
                    lev_params_raw = lev_params_raw or params_data.get('lev_params')
                    params_data = params_data.get('params', {})

                cfg_item = {
                    'name': strat_name,
                    'weight': weight,
                    'type': category if category else 'equity',
                    'params': params_data,
                }
                if lev_params_raw is not None:
                    cfg_item['lev_params'] = json.loads(lev_params_raw) if isinstance(lev_params_raw, str) else lev_params_raw
                configs.append(cfg_item)
                
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
    def list_portfolios(cls):
        """저장된 포트폴리오 목록 조회 (캐시 적용).

        @st.cache_data TTL=60s 래퍼를 통해 호출되므로,
        매 rerun마다 Supabase를 조회하지 않는다.
        저장/삭제 시 st.cache_data.clear()로 무효화됨.
        """
        return _cached_list_portfolios()

    @classmethod
    def set_default_portfolio(cls, name):
        """특정 포트폴리오를 기본값으로 지정 (기존 기본값은 해제)"""
        cls._init_portfolio_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE portfolios SET is_default = FALSE")
            cursor.execute("UPDATE portfolios SET is_default = TRUE WHERE name = %s", (name,))
            conn.commit()
            conn.close()
            st.cache_data.clear()  # 캐시 무효화 → 다음 조회 시 DB 재조회
            return True
        except Exception as e:
            print(f"Set Default Portfolio Error: {e}")
            return False

    @classmethod
    def get_default_portfolio(cls):
        """기본 포트폴리오 이름 반환 (캐시 적용, 없으면 None)."""
        return _cached_get_default_portfolio()

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


class AssetStorage:
    """자산 관리 시스템을 위한 Supabase/PostgreSQL 저장소"""

    @classmethod
    def _get_connection(cls):
        """Supabase DB 연결 객체 반환"""
        supabase_url = os.getenv("SUPABASE_DB_URL")
        if not supabase_url or not HAS_PSYCOPG2:
            raise ValueError("SUPABASE_DB_URL 설정 또는 psycopg2 라이브러리가 필요합니다.")
        return psycopg2.connect(supabase_url)

    @classmethod
    def _init_asset_db(cls):
        """자산 관리 테이블 초기화"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            # [1] assets 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    ticker TEXT,
                    api_source TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [2] accounts 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS accounts (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    account_type TEXT NOT NULL,
                    currency TEXT DEFAULT 'USD',
                    broker_code TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [3] holdings 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS holdings (
                    id SERIAL PRIMARY KEY,
                    account_id INT NOT NULL REFERENCES accounts(id),
                    asset_id TEXT NOT NULL REFERENCES assets(id),
                    quantity DECIMAL(18,8) NOT NULL DEFAULT 0,
                    avg_cost DECIMAL(18,8),
                    total_cost DECIMAL(18,8),
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(account_id, asset_id)
                )
            ''')

            # [4] deposits 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS deposits (
                    id SERIAL PRIMARY KEY,
                    account_id INT NOT NULL REFERENCES accounts(id),
                    amount DECIMAL(18,8) NOT NULL,
                    currency TEXT DEFAULT 'KRW',
                    deposit_type TEXT NOT NULL,
                    deposit_date DATE NOT NULL,
                    notes TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [6] account_sync_config 테이블 (동기화 기준 총자본 저장)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account_sync_config (
                    id               SERIAL PRIMARY KEY,
                    account_id       INT NOT NULL REFERENCES accounts(id) UNIQUE,
                    total_capital    DECIMAL(18,2) NOT NULL DEFAULT 0,
                    capital_currency TEXT DEFAULT 'USD',
                    last_synced_at   TIMESTAMPTZ DEFAULT NOW(),
                    notes            TEXT,
                    updated_at       TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_holdings_account ON holdings(account_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_deposits_account_date ON deposits(account_id, deposit_date DESC)')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Asset DB Init Error: {e}")

    # --- Asset Methods ---

    @classmethod
    def save_asset(cls, asset_id, name, category, currency='USD', ticker=None, api_source=None):
        """자산 저장"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            sql = '''
                INSERT INTO assets (id, name, category, currency, ticker, api_source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                currency = EXCLUDED.currency,
                ticker = EXCLUDED.ticker,
                api_source = EXCLUDED.api_source
            '''
            cursor.execute(sql, (asset_id, name, category, currency, ticker, api_source))
            if cursor.rowcount > 0:
                conn.commit()
                st.cache_data.clear()
            conn.close()
            return asset_id
        except Exception as e:
            print(f"Save Asset Error: {e}")
            return None

    @classmethod
    def list_assets(cls, category=None):
        """자산 목록 조회 (캐시)"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT id, name FROM assets WHERE category = %s AND is_active = TRUE ORDER BY name ASC", (category,))
            else:
                cursor.execute("SELECT id, name FROM assets WHERE is_active = TRUE ORDER BY name ASC")
            rows = cursor.fetchall()
            conn.close()
            return [(row[0], row[1]) for row in rows]  # (id, name) 튜플 리스트
        except Exception as e:
            print(f"List Assets Error: {e}")
            return []

    @classmethod
    def load_asset(cls, asset_id):
        """자산 상세 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, category, currency, ticker, api_source FROM assets WHERE id = %s", (asset_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'currency': row[3],
                    'ticker': row[4],
                    'api_source': row[5]
                }
            return None
        except Exception as e:
            print(f"Load Asset Error: {e}")
            return None

    # --- Account Methods ---

    @classmethod
    def save_account(cls, name, account_type, currency='USD', broker_code=None):
        """계좌 저장"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            sql = '''
                INSERT INTO accounts (name, account_type, currency, broker_code)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE SET
                account_type = EXCLUDED.account_type,
                currency = EXCLUDED.currency,
                broker_code = EXCLUDED.broker_code
                RETURNING id
            '''
            cursor.execute(sql, (name, account_type, currency, broker_code))
            account_id = cursor.fetchone()[0]
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return account_id
        except Exception as e:
            print(f"Save Account Error: {e}")
            return None

    @classmethod
    def list_accounts(cls):
        """계좌 목록 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, account_type, currency FROM accounts WHERE is_active = TRUE ORDER BY id ASC")
            rows = cursor.fetchall()
            conn.close()
            return [{'id': row[0], 'name': row[1], 'type': row[2], 'currency': row[3]} for row in rows]
        except Exception as e:
            print(f"List Accounts Error: {e}")
            return []

    @classmethod
    def load_account(cls, account_id):
        """계좌 상세 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, account_type, currency, broker_code FROM accounts WHERE id = %s", (account_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'id': row[0],
                    'name': row[1],
                    'type': row[2],
                    'currency': row[3],
                    'broker_code': row[4]
                }
            return None
        except Exception as e:
            print(f"Load Account Error: {e}")
            return None

    # --- Holdings Methods ---

    @classmethod
    def save_holding(cls, account_id, asset_id, quantity):
        """보유 자산 수량 저장"""
        return cls.update_holding_quantity(account_id, asset_id, quantity)

    @classmethod
    def list_holdings(cls, account_id):
        """보유 현황 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            sql = '''
                SELECT h.id, h.asset_id, a.name, h.quantity, h.last_updated
                FROM holdings h
                JOIN assets a ON h.asset_id = a.id
                WHERE h.account_id = %s AND h.quantity > 0
                ORDER BY a.name ASC
            '''
            cursor.execute(sql, (account_id,))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    'id': row[0],
                    'asset_id': row[1],
                    'asset_name': row[2],
                    'quantity': float(row[3]),
                    'last_updated': row[4]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"List Holdings Error: {e}")
            return []

    @classmethod
    def update_holding_quantity(cls, account_id, asset_id, quantity):
        """보유 수량 직접 업데이트 (평균단가 계산 없음)"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            if quantity <= 0:
                cursor.execute('''
                    DELETE FROM holdings WHERE account_id = %s AND asset_id = %s
                ''', (account_id, asset_id))
            else:
                cursor.execute('''
                    INSERT INTO holdings (account_id, asset_id, quantity, last_updated)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (account_id, asset_id) DO UPDATE SET
                    quantity = EXCLUDED.quantity,
                    last_updated = NOW()
                ''', (account_id, asset_id, quantity))
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return True
        except Exception as e:
            print(f"Update Holding Quantity Error: {e}")
            return False

    # --- Deposit Methods ---

    @classmethod
    def save_deposit(cls, account_id, amount, deposit_type, deposit_date=None, currency='KRW', notes=None):
        """입출금 기록 저장"""
        cls._init_asset_db()
        if deposit_date is None:
            deposit_date = datetime.date.today()

        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            sql = '''
                INSERT INTO deposits (account_id, amount, currency, deposit_type, deposit_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
            '''
            cursor.execute(sql, (account_id, amount, currency, deposit_type, deposit_date, notes))
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Deposit Error: {e}")
            return False

    @classmethod
    def list_deposits(cls, account_id, limit=100):
        """입출금 내역 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            sql = '''
                SELECT id, amount, currency, deposit_type, deposit_date, notes
                FROM deposits
                WHERE account_id = %s
                ORDER BY deposit_date DESC, created_at DESC
                LIMIT %s
            '''
            cursor.execute(sql, (account_id, limit))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    'id': row[0],
                    'amount': float(row[1]),
                    'currency': row[2],
                    'type': row[3],
                    'date': row[4],
                    'notes': row[5]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"List Deposits Error: {e}")
            return []

    @classmethod
    def _init_signal_events_db(cls):
        """신호 이벤트 테이블 초기화"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_events (
                    id           SERIAL PRIMARY KEY,
                    account_id   INT NOT NULL REFERENCES accounts(id),
                    portfolio_name TEXT NOT NULL,
                    event_date   DATE NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strat_type   TEXT NOT NULL,
                    prev_stage   INT,
                    new_stage    INT,
                    prev_asset   TEXT,
                    new_asset    TEXT,
                    action       TEXT,
                    total_value  DECIMAL(18,2),
                    is_executed  BOOLEAN DEFAULT FALSE,
                    notes        TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_events_account ON signal_events(account_id, event_date DESC)')
            cursor.execute('''
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname = 'uq_signal_events_key'
                    ) THEN
                        ALTER TABLE signal_events
                        ADD CONSTRAINT uq_signal_events_key
                        UNIQUE (account_id, portfolio_name, event_date, strategy_name);
                    END IF;
                END $$
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Signal Events DB Init Error: {e}")

    @classmethod
    def save_signal_event(cls, account_id, portfolio_name, strategy_name, strat_type,
                          prev_stage, new_stage, prev_asset, new_asset, action,
                          total_value=None, notes=None, event_date=None):
        """신호 이벤트 저장 (같은 날 동일 전략이면 UPDATE, 없으면 INSERT)"""
        cls._init_signal_events_db()
        if event_date is None:
            event_date = datetime.date.today()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO signal_events
                (account_id, portfolio_name, event_date, strategy_name, strat_type,
                 prev_stage, new_stage, prev_asset, new_asset, action, total_value, notes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (account_id, portfolio_name, event_date, strategy_name)
                DO UPDATE SET
                    strat_type  = EXCLUDED.strat_type,
                    new_stage   = EXCLUDED.new_stage,
                    new_asset   = EXCLUDED.new_asset,
                    action      = EXCLUDED.action,
                    total_value = EXCLUDED.total_value,
                    notes       = EXCLUDED.notes
                RETURNING id
            ''', (account_id, portfolio_name, event_date, strategy_name, strat_type,
                  prev_stage, new_stage, prev_asset, new_asset, action, total_value, notes))
            eid = cursor.fetchone()[0]
            conn.commit()
            conn.close()
            return eid
        except Exception as e:
            print(f"Save Signal Event Error: {e}")
            return None

    @classmethod
    def list_signal_events(cls, account_id, limit=50):
        """신호 이벤트 목록 조회"""
        cls._init_signal_events_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, portfolio_name, event_date, strategy_name, strat_type,
                       prev_stage, new_stage, prev_asset, new_asset, action,
                       total_value, is_executed, notes, created_at
                FROM signal_events
                WHERE account_id = %s
                ORDER BY event_date DESC, created_at DESC
                LIMIT %s
            ''', (account_id, limit))
            rows = cursor.fetchall()
            conn.close()
            return [{
                'id': r[0], 'portfolio_name': r[1], 'event_date': r[2],
                'strategy_name': r[3], 'strat_type': r[4],
                'prev_stage': r[5], 'new_stage': r[6],
                'prev_asset': r[7], 'new_asset': r[8],
                'action': r[9], 'total_value': float(r[10]) if r[10] else 0,
                'is_executed': r[11], 'notes': r[12], 'created_at': r[13]
            } for r in rows]
        except Exception as e:
            print(f"List Signal Events Error: {e}")
            return []

    @classmethod
    def get_latest_signal_snapshot(cls, account_id, portfolio_name):
        """계좌+포트폴리오의 마지막 신호 스냅샷 조회 (전략별 stage/asset)"""
        cls._init_signal_events_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT ON (strategy_name)
                       strategy_name, strat_type, new_stage, new_asset, event_date
                FROM signal_events
                WHERE account_id = %s AND portfolio_name = %s
                ORDER BY strategy_name, event_date DESC, created_at DESC
            ''', (account_id, portfolio_name))
            rows = cursor.fetchall()
            conn.close()
            return {r[0]: {'strat_type': r[1], 'stage': r[2], 'asset': r[3], 'date': r[4]}
                    for r in rows}
        except Exception as e:
            print(f"Get Latest Signal Snapshot Error: {e}")
            return {}

    @classmethod
    def mark_signal_events_executed(cls, event_ids):
        """신호 이벤트 실행 완료 처리"""
        cls._init_signal_events_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE signal_events SET is_executed=TRUE WHERE id = ANY(%s)', (event_ids,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Mark Signal Events Executed Error: {e}")
            return False

    # --- Sync Config Methods ---

    @classmethod
    def save_sync_config(cls, account_id, total_capital, currency='USD', notes=None):
        """계좌 총자본 동기화 설정 저장 (upsert)"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO account_sync_config (account_id, total_capital, capital_currency, notes, last_synced_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (account_id) DO UPDATE SET
                    total_capital    = EXCLUDED.total_capital,
                    capital_currency = EXCLUDED.capital_currency,
                    notes            = EXCLUDED.notes,
                    last_synced_at   = NOW(),
                    updated_at       = NOW()
            ''', (account_id, total_capital, currency, notes))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Sync Config Error: {e}")
            return False

    @classmethod
    def load_sync_config(cls, account_id):
        """계좌 총자본 동기화 설정 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT total_capital, capital_currency, last_synced_at, notes
                FROM account_sync_config
                WHERE account_id = %s
            ''', (account_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'total_capital': float(row[0]),
                    'currency': row[1],
                    'last_synced_at': row[2],
                    'notes': row[3]
                }
            return None
        except Exception as e:
            print(f"Load Sync Config Error: {e}")
            return None

    @classmethod
    def upsert_holding_qty(cls, account_id, asset_id, quantity):
        """holdings 수량만 직접 upsert (동기화 전용 - avg_cost/total_cost 재계산 없음)"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO holdings (account_id, asset_id, quantity, last_updated)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (account_id, asset_id) DO UPDATE SET
                    quantity     = EXCLUDED.quantity,
                    last_updated = NOW()
            ''', (account_id, asset_id, quantity))
            conn.commit()
            try:
                st.cache_data.clear()
            except Exception:
                pass
            conn.close()
            return True
        except Exception as e:
            print(f"Upsert Holding Qty Error: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# KIS 주문 관리 저장소
# ═══════════════════════════════════════════════════════════════════════════

class OrderStorage:
    """KIS 주문 관리를 위한 Supabase/PostgreSQL 저장소"""

    @classmethod
    def _get_connection(cls):
        supabase_url = os.getenv("SUPABASE_DB_URL")
        if not supabase_url or not HAS_PSYCOPG2:
            raise ValueError("SUPABASE_DB_URL 설정 또는 psycopg2 라이브러리가 필요합니다.")
        return psycopg2.connect(supabase_url)

    # ── 테이블 초기화 ────────────────────────────────────────────────────

    @classmethod
    def _init_trading_db(cls):
        """KIS 거래 관련 테이블 초기화"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            # [1] kis_tokens — OAuth 토큰 캐시
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS kis_tokens (
                    mode         TEXT PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    expires_at   TIMESTAMPTZ NOT NULL,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [2] trading_config — 계좌별 거래 설정
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_config (
                    id                    SERIAL PRIMARY KEY,
                    account_id            INT NOT NULL UNIQUE,
                    is_halted             BOOLEAN DEFAULT TRUE,
                    max_single_order_usd  DECIMAL(18,2) DEFAULT 5000,
                    daily_order_limit_usd DECIMAL(18,2) DEFAULT 20000,
                    daily_loss_limit_usd  DECIMAL(18,2) DEFAULT 5000,
                    position_limit_pct    DECIMAL(5,4) DEFAULT 0.50,
                    cooldown_minutes      INT DEFAULT 30,
                    auto_approve_paper    BOOLEAN DEFAULT FALSE,
                    allowed_tickers       TEXT[] DEFAULT ARRAY['TQQQ','QQQ','QLD','TLT','TMF','BIL','IEF'],
                    updated_at            TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [3] order_log — 주문 감사 추적 (모든 시도 기록)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS order_log (
                    id               SERIAL PRIMARY KEY,
                    account_id       INT NOT NULL,
                    portfolio_name   TEXT,
                    strategy_name    TEXT,
                    signal_event_id  INT,

                    ticker           TEXT NOT NULL,
                    side             TEXT NOT NULL,
                    quantity         INT NOT NULL,
                    price            DECIMAL(18,4),
                    order_type       TEXT DEFAULT 'market',
                    estimated_amount DECIMAL(18,2),

                    safety_passed    BOOLEAN NOT NULL,
                    safety_checks    JSONB,
                    rejection_reason TEXT,

                    approval_status  TEXT DEFAULT 'pending',
                    approved_at      TIMESTAMPTZ,
                    approved_by      TEXT,

                    execution_status TEXT DEFAULT 'not_sent',
                    kis_order_no     TEXT,
                    filled_quantity  INT DEFAULT 0,
                    filled_price     DECIMAL(18,4),
                    filled_amount    DECIMAL(18,2),
                    error_message    TEXT,

                    trading_mode     TEXT NOT NULL,
                    created_at       TIMESTAMPTZ DEFAULT NOW(),
                    updated_at       TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            # 멱등성 제약: 동일 signal_event + ticker → 중복 방지
            cursor.execute('''
                CREATE UNIQUE INDEX IF NOT EXISTS uq_order_signal_ticker
                ON order_log(signal_event_id, ticker)
                WHERE signal_event_id IS NOT NULL
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_log_account_date ON order_log(account_id, created_at DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_log_status ON order_log(execution_status, created_at DESC)')

            # [4] pending_approvals — Telegram 승인 대기
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id             SERIAL PRIMARY KEY,
                    order_log_id   INT NOT NULL REFERENCES order_log(id),
                    approval_code  TEXT NOT NULL UNIQUE,
                    status         TEXT DEFAULT 'pending',
                    expires_at     TIMESTAMPTZ NOT NULL,
                    responded_at   TIMESTAMPTZ,
                    created_at     TIMESTAMPTZ DEFAULT NOW()
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pending_code ON pending_approvals(approval_code)')

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Trading DB Init Error: {e}")

    # ── KIS 토큰 관리 ────────────────────────────────────────────────────

    @classmethod
    def save_kis_token(cls, mode, access_token, expires_at):
        """KIS OAuth 토큰 저장 (UPSERT)"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO kis_tokens (mode, access_token, expires_at, created_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (mode) DO UPDATE SET
                    access_token = EXCLUDED.access_token,
                    expires_at   = EXCLUDED.expires_at,
                    created_at   = NOW()
            ''', (mode, access_token, expires_at))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Save KIS Token Error: {e}")

    @classmethod
    def load_kis_token(cls, mode):
        """KIS OAuth 토큰 로드. 반환: (access_token, expires_at) 또는 None"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'SELECT access_token, expires_at FROM kis_tokens WHERE mode = %s',
                (mode,)
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return (row[0], row[1])
            return None
        except Exception as e:
            print(f"Load KIS Token Error: {e}")
            return None

    # ── 거래 설정 ────────────────────────────────────────────────────────

    @classmethod
    def load_trading_config(cls, account_id):
        """계좌별 거래 설정 로드"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT is_halted, max_single_order_usd, daily_order_limit_usd,
                       daily_loss_limit_usd, position_limit_pct, cooldown_minutes,
                       auto_approve_paper, allowed_tickers
                FROM trading_config WHERE account_id = %s
            ''', (account_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'is_halted': row[0],
                    'max_single_order_usd': float(row[1]),
                    'daily_order_limit_usd': float(row[2]),
                    'daily_loss_limit_usd': float(row[3]),
                    'position_limit_pct': float(row[4]),
                    'cooldown_minutes': int(row[5]),
                    'auto_approve_paper': row[6],
                    'allowed_tickers': list(row[7]) if row[7] else [],
                }
            return None
        except Exception as e:
            print(f"Load Trading Config Error: {e}")
            return None

    @classmethod
    def save_trading_config(cls, account_id, config: dict):
        """계좌별 거래 설정 저장 (UPSERT)"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO trading_config
                (account_id, is_halted, max_single_order_usd, daily_order_limit_usd,
                 daily_loss_limit_usd, position_limit_pct, cooldown_minutes,
                 auto_approve_paper, allowed_tickers, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (account_id) DO UPDATE SET
                    is_halted             = EXCLUDED.is_halted,
                    max_single_order_usd  = EXCLUDED.max_single_order_usd,
                    daily_order_limit_usd = EXCLUDED.daily_order_limit_usd,
                    daily_loss_limit_usd  = EXCLUDED.daily_loss_limit_usd,
                    position_limit_pct    = EXCLUDED.position_limit_pct,
                    cooldown_minutes      = EXCLUDED.cooldown_minutes,
                    auto_approve_paper    = EXCLUDED.auto_approve_paper,
                    allowed_tickers       = EXCLUDED.allowed_tickers,
                    updated_at            = NOW()
            ''', (
                account_id,
                config.get('is_halted', True),
                config.get('max_single_order_usd', 5000),
                config.get('daily_order_limit_usd', 20000),
                config.get('daily_loss_limit_usd', 5000),
                config.get('position_limit_pct', 0.50),
                config.get('cooldown_minutes', 30),
                config.get('auto_approve_paper', False),
                config.get('allowed_tickers', ['TQQQ', 'QQQ', 'QLD', 'TLT', 'TMF', 'BIL', 'IEF']),
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Trading Config Error: {e}")
            return False

    @classmethod
    def set_trading_halt(cls, account_id, is_halted: bool):
        """거래 중단/재개 (kill switch)"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            # trading_config가 없으면 기본값으로 생성
            cursor.execute('''
                INSERT INTO trading_config (account_id, is_halted, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (account_id) DO UPDATE SET
                    is_halted  = EXCLUDED.is_halted,
                    updated_at = NOW()
            ''', (account_id, is_halted))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Set Trading Halt Error: {e}")
            return False

    # ── 주문 로그 ────────────────────────────────────────────────────────

    @classmethod
    def save_order_log(cls, order_data: dict) -> int:
        """주문 로그 저장. 반환: order_log.id 또는 None"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO order_log
                (account_id, portfolio_name, strategy_name, signal_event_id,
                 ticker, side, quantity, price, order_type, estimated_amount,
                 safety_passed, safety_checks, rejection_reason,
                 approval_status, execution_status, trading_mode)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
            ''', (
                order_data['account_id'],
                order_data.get('portfolio_name'),
                order_data.get('strategy_name'),
                order_data.get('signal_event_id'),
                order_data['ticker'],
                order_data['side'],
                order_data['quantity'],
                order_data.get('price'),
                order_data.get('order_type', 'market'),
                order_data.get('estimated_amount'),
                order_data['safety_passed'],
                json.dumps(order_data.get('safety_checks', []), ensure_ascii=False),
                order_data.get('rejection_reason'),
                order_data.get('approval_status', 'pending'),
                order_data.get('execution_status', 'not_sent'),
                order_data['trading_mode'],
            ))
            oid = cursor.fetchone()[0]
            conn.commit()
            conn.close()
            return oid
        except Exception as e:
            # 멱등성 제약 위반(중복 주문)은 정상 동작
            if 'uq_order_signal_ticker' in str(e):
                print(f"[OrderLog] Duplicate order ignored (signal_event_id={order_data.get('signal_event_id')}, ticker={order_data['ticker']})", flush=True)
                return None
            print(f"Save Order Log Error: {e}")
            return None

    @classmethod
    def update_order_execution(cls, order_log_id: int, update_data: dict):
        """주문 실행 결과 업데이트"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            set_parts = []
            values = []
            for key in ('execution_status', 'kis_order_no', 'filled_quantity',
                        'filled_price', 'filled_amount', 'error_message',
                        'approval_status', 'approved_at', 'approved_by'):
                if key in update_data:
                    set_parts.append(f"{key} = %s")
                    values.append(update_data[key])

            if not set_parts:
                return

            set_parts.append("updated_at = NOW()")
            values.append(order_log_id)

            cursor.execute(
                f"UPDATE order_log SET {', '.join(set_parts)} WHERE id = %s",
                values
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Update Order Execution Error: {e}")

    @classmethod
    def get_daily_order_total(cls, account_id: int, trading_mode: str) -> float:
        """당일 주문 총액 조회 (안전가드용)"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COALESCE(SUM(estimated_amount), 0)
                FROM order_log
                WHERE account_id = %s
                  AND trading_mode = %s
                  AND created_at::date = CURRENT_DATE
                  AND safety_passed = TRUE
                  AND execution_status NOT IN ('failed', 'cancelled')
            ''', (account_id, trading_mode))
            total = float(cursor.fetchone()[0])
            conn.close()
            return total
        except Exception as e:
            print(f"Get Daily Order Total Error: {e}")
            return 0.0

    @classmethod
    def get_recent_order_for_ticker(cls, account_id: int, ticker: str, minutes: int = 30) -> dict:
        """최근 N분 내 동일 종목 주문 조회 (쿨다운 체크용)"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, side, quantity, created_at
                FROM order_log
                WHERE account_id = %s
                  AND ticker = %s
                  AND safety_passed = TRUE
                  AND created_at > NOW() - INTERVAL '%s minutes'
                ORDER BY created_at DESC
                LIMIT 1
            ''', (account_id, ticker, minutes))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {'id': row[0], 'side': row[1], 'quantity': row[2], 'created_at': row[3]}
            return None
        except Exception as e:
            print(f"Get Recent Order Error: {e}")
            return None

    @classmethod
    def check_duplicate_order(cls, signal_event_id: int, ticker: str) -> bool:
        """동일 signal_event + ticker 주문이 이미 존재하는지 확인"""
        if signal_event_id is None:
            return False
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM order_log
                WHERE signal_event_id = %s AND ticker = %s
                LIMIT 1
            ''', (signal_event_id, ticker))
            exists = cursor.fetchone() is not None
            conn.close()
            return exists
        except Exception as e:
            print(f"Check Duplicate Order Error: {e}")
            return True  # 에러 시 중복으로 간주 (Fail-Closed)

    # ── UI 조회 ──────────────────────────────────────────────────────────

    @classmethod
    def list_order_logs(cls, trading_mode: str = None, days: int = 30, limit: int = 200) -> list[dict]:
        """
        UI용 주문 로그 목록 조회.

        Parameters
        ----------
        trading_mode : str | None — 'paper' / 'live' / None(전체)
        days : int — 최근 N일 (기본 30일)
        limit : int — 최대 행 수

        Returns
        -------
        list[dict] — 최신순 정렬
        """
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            where = ["created_at > NOW() - INTERVAL '%s days'"]
            args  = [days]
            if trading_mode:
                where.append("trading_mode = %s")
                args.append(trading_mode)
            args.append(limit)
            cursor.execute(f'''
                SELECT
                    id, portfolio_name, strategy_name, ticker, side,
                    quantity, filled_price, filled_amount,
                    safety_passed, rejection_reason,
                    approval_status, execution_status,
                    kis_order_no, trading_mode, created_at, updated_at,
                    safety_checks
                FROM order_log
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC
                LIMIT %s
            ''', args)
            cols = [d[0] for d in cursor.description]
            rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"list_order_logs Error: {e}")
            return []

    @classmethod
    def get_pnl_summary(cls, trading_mode: str = None, days: int = 90) -> dict:
        """
        UI용 손익 요약 (날짜별 누적 실현손익).

        Returns
        -------
        dict : {
            'daily': list[{date, realized_pnl, trade_count}],
            'total_buy_amount': float,
            'total_sell_amount': float,
            'total_trades': int,
            'filled_trades': int,
            'blocked_trades': int,
        }
        """
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            where = ["created_at > NOW() - INTERVAL '%s days'", "execution_status = 'filled'"]
            args  = [days]
            if trading_mode:
                where.append("trading_mode = %s")
                args.append(trading_mode)

            # 날짜별 매수/매도 금액 집계
            cursor.execute(f'''
                SELECT
                    created_at::date AS dt,
                    side,
                    COUNT(*) AS cnt,
                    COALESCE(SUM(filled_amount), 0) AS total_amount
                FROM order_log
                WHERE {' AND '.join(where)}
                GROUP BY dt, side
                ORDER BY dt
            ''', args)
            rows = cursor.fetchall()

            # 전체 통계
            where2 = ["created_at > NOW() - INTERVAL '%s days'"]
            args2  = [days]
            if trading_mode:
                where2.append("trading_mode = %s")
                args2.append(trading_mode)
            cursor.execute(f'''
                SELECT
                    COUNT(*) FILTER (WHERE execution_status = 'filled')   AS filled,
                    COUNT(*) FILTER (WHERE safety_passed = FALSE)          AS blocked,
                    COUNT(*)                                               AS total
                FROM order_log
                WHERE {' AND '.join(where2)}
            ''', args2)
            stats = cursor.fetchone()
            conn.close()

            # 날짜별 데이터 가공 (buy/sell 금액으로 실현손익 근사)
            from collections import defaultdict
            daily_map = defaultdict(lambda: {'buy': 0.0, 'sell': 0.0, 'count': 0})
            for dt, side, cnt, amt in rows:
                daily_map[dt][side] = float(amt)
                daily_map[dt]['count'] += cnt

            daily = [
                {'date': str(dt), 'sell_amount': v['sell'], 'buy_amount': v['buy'], 'trade_count': v['count']}
                for dt, v in sorted(daily_map.items())
            ]
            return {
                'daily': daily,
                'filled_trades': int(stats[0]) if stats else 0,
                'blocked_trades': int(stats[1]) if stats else 0,
                'total_trades': int(stats[2]) if stats else 0,
            }
        except Exception as e:
            print(f"get_pnl_summary Error: {e}")
            return {'daily': [], 'filled_trades': 0, 'blocked_trades': 0, 'total_trades': 0}

    # ── 승인 관리 ────────────────────────────────────────────────────────

    @classmethod
    def save_pending_approval(cls, order_log_id: int, approval_code: str, expires_at) -> int:
        """승인 대기 레코드 저장"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pending_approvals (order_log_id, approval_code, status, expires_at)
                VALUES (%s, %s, 'pending', %s)
                RETURNING id
            ''', (order_log_id, approval_code, expires_at))
            aid = cursor.fetchone()[0]
            conn.commit()
            conn.close()
            return aid
        except Exception as e:
            print(f"Save Pending Approval Error: {e}")
            return None

    @classmethod
    def get_pending_approval(cls, approval_code: str) -> dict:
        """승인 코드로 대기 레코드 조회"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pa.id, pa.order_log_id, pa.status, pa.expires_at,
                       ol.account_id, ol.ticker, ol.side, ol.quantity, ol.price,
                       ol.order_type, ol.trading_mode, ol.portfolio_name, ol.strategy_name
                FROM pending_approvals pa
                JOIN order_log ol ON ol.id = pa.order_log_id
                WHERE pa.approval_code = %s
            ''', (approval_code,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return {
                    'approval_id': row[0],
                    'order_log_id': row[1],
                    'status': row[2],
                    'expires_at': row[3],
                    'account_id': row[4],
                    'ticker': row[5],
                    'side': row[6],
                    'quantity': row[7],
                    'price': float(row[8]) if row[8] else 0,
                    'order_type': row[9],
                    'trading_mode': row[10],
                    'portfolio_name': row[11],
                    'strategy_name': row[12],
                }
            return None
        except Exception as e:
            print(f"Get Pending Approval Error: {e}")
            return None

    @classmethod
    def update_pending_approval(cls, approval_code: str, status: str):
        """승인 상태 업데이트"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE pending_approvals
                SET status = %s, responded_at = NOW()
                WHERE approval_code = %s
            ''', (status, approval_code))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Update Pending Approval Error: {e}")

    @classmethod
    def list_pending_approvals(cls, status: str = 'pending') -> list:
        """대기 중인 승인 목록 조회"""
        cls._init_trading_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT pa.approval_code, pa.expires_at, pa.created_at,
                       ol.ticker, ol.side, ol.quantity, ol.price, ol.trading_mode
                FROM pending_approvals pa
                JOIN order_log ol ON ol.id = pa.order_log_id
                WHERE pa.status = %s
                ORDER BY pa.created_at DESC
                LIMIT 20
            ''', (status,))
            rows = cursor.fetchall()
            conn.close()
            return [{
                'approval_code': r[0], 'expires_at': r[1], 'created_at': r[2],
                'ticker': r[3], 'side': r[4], 'quantity': r[5],
                'price': float(r[6]) if r[6] else 0, 'trading_mode': r[7],
            } for r in rows]
        except Exception as e:
            print(f"List Pending Approvals Error: {e}")
            return []


# ── @st.cache_data 캐시 함수 (모듈 레벨, TTL=60s) ────────────────────────────
# StrategyStorage / PortfolioStorage 목록 조회는 매 rerun마다 Supabase를
# 조회하는 대신, 이 캐시 함수를 거쳐 60초 동안 결과를 재사용한다.
# 저장·삭제·기본값 변경 등 쓰기 작업 완료 시 st.cache_data.clear()를
# 호출하면 즉시 무효화된다.

@st.cache_data(ttl=60)
def _cached_list_strategies(category=None):
    """전략 이름 목록을 DB에서 조회 (캐시 TTL 60s)."""
    supabase_url = os.getenv("SUPABASE_DB_URL")
    if not supabase_url or not HAS_PSYCOPG2:
        return []
    try:
        conn = psycopg2.connect(supabase_url)
        cursor = conn.cursor()
        if category:
            cursor.execute("SELECT name FROM strategies WHERE category = %s ORDER BY name ASC", (category,))
        else:
            cursor.execute("SELECT name FROM strategies ORDER BY name ASC")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return names
    except Exception as e:
        print(f"[_cached_list_strategies] Error: {e}")
        return []


@st.cache_data(ttl=60)
def _cached_get_default_strategy(category='equity'):
    """카테고리 기본 전략 이름을 DB에서 조회 (캐시 TTL 60s)."""
    supabase_url = os.getenv("SUPABASE_DB_URL")
    if not supabase_url or not HAS_PSYCOPG2:
        return None
    try:
        conn = psycopg2.connect(supabase_url)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM strategies WHERE category = %s AND is_default = TRUE LIMIT 1",
            (category,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"[_cached_get_default_strategy] Error: {e}")
        return None


@st.cache_data(ttl=60)
def _cached_list_portfolios():
    """포트폴리오 이름 목록을 DB에서 조회 (캐시 TTL 60s)."""
    supabase_url = os.getenv("SUPABASE_DB_URL")
    if not supabase_url or not HAS_PSYCOPG2:
        return []
    try:
        conn = psycopg2.connect(supabase_url)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM portfolios ORDER BY updated_at DESC")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        return names
    except Exception as e:
        print(f"[_cached_list_portfolios] Error: {e}")
        return []


@st.cache_data(ttl=60)
def _cached_get_default_portfolio():
    """기본 포트폴리오 이름을 DB에서 조회 (캐시 TTL 60s)."""
    supabase_url = os.getenv("SUPABASE_DB_URL")
    if not supabase_url or not HAS_PSYCOPG2:
        return None
    try:
        conn = psycopg2.connect(supabase_url)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM portfolios WHERE is_default = TRUE LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"[_cached_get_default_portfolio] Error: {e}")
        return None
