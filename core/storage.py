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

            # [4] transactions 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id SERIAL PRIMARY KEY,
                    account_id INT NOT NULL REFERENCES accounts(id),
                    asset_id TEXT NOT NULL REFERENCES assets(id),
                    tx_type TEXT NOT NULL,
                    quantity DECIMAL(18,8),
                    price DECIMAL(18,8),
                    amount DECIMAL(18,8) NOT NULL,
                    fee DECIMAL(18,8) DEFAULT 0,
                    currency TEXT DEFAULT 'USD',
                    tx_date DATE NOT NULL,
                    notes TEXT,
                    source TEXT DEFAULT 'manual',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            ''')

            # [5] deposits 테이블
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

            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_holdings_account ON holdings(account_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_account_date ON transactions(account_id, tx_date DESC)')
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
    def save_holding(cls, account_id, asset_id, quantity, price):
        """보유 자산 저장 (평균 매입가 자동 계산)"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            # 현재 보유 정보 조회
            cursor.execute('''
                SELECT quantity, avg_cost, total_cost FROM holdings
                WHERE account_id = %s AND asset_id = %s
            ''', (account_id, asset_id))
            row = cursor.fetchone()

            if row:
                # 이미 보유 중인 경우 - 평균 매입가 재계산
                old_qty, old_avg_cost, old_total_cost = row
                old_avg_cost = old_avg_cost or 0
                old_total_cost = old_total_cost or 0

                new_qty = old_qty + quantity
                if new_qty > 0:
                    new_total_cost = old_total_cost + (quantity * price)
                    new_avg_cost = new_total_cost / new_qty
                else:
                    new_total_cost = 0
                    new_avg_cost = 0
            else:
                # 신규 보유
                new_qty = quantity
                new_avg_cost = price if quantity > 0 else 0
                new_total_cost = quantity * price

            # 보유 정보 저장
            sql = '''
                INSERT INTO holdings (account_id, asset_id, quantity, avg_cost, total_cost)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (account_id, asset_id) DO UPDATE SET
                quantity = EXCLUDED.quantity,
                avg_cost = EXCLUDED.avg_cost,
                total_cost = EXCLUDED.total_cost,
                last_updated = NOW()
            '''
            cursor.execute(sql, (account_id, asset_id, new_qty, new_avg_cost, new_total_cost))
            conn.commit()
            st.cache_data.clear()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Holding Error: {e}")
            return False

    @classmethod
    def list_holdings(cls, account_id):
        """보유 현황 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            sql = '''
                SELECT h.id, h.asset_id, a.name, h.quantity, h.avg_cost, h.total_cost, h.last_updated
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
                    'avg_cost': float(row[4]) if row[4] else 0,
                    'total_cost': float(row[5]) if row[5] else 0,
                    'last_updated': row[6]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"List Holdings Error: {e}")
            return []

    # --- Transaction Methods ---

    @classmethod
    def save_transaction(cls, account_id, asset_id, tx_type, amount, quantity=None, price=None, fee=0, currency='USD', tx_date=None, notes=None):
        """거래 기록 저장 및 보유량 업데이트"""
        cls._init_asset_db()
        if tx_date is None:
            tx_date = datetime.date.today()

        try:
            conn = cls._get_connection()
            cursor = conn.cursor()

            # 거래 기록 저장
            sql_tx = '''
                INSERT INTO transactions
                (account_id, asset_id, tx_type, quantity, price, amount, fee, currency, tx_date, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            '''
            cursor.execute(sql_tx, (account_id, asset_id, tx_type, quantity, price, amount, fee, currency, tx_date, notes))

            # 보유량 업데이트 (매수/매도의 경우)
            if tx_type in ['buy', 'sell'] and quantity:
                if tx_type == 'buy':
                    # 기존 보유정보 조회
                    cursor.execute('''
                        SELECT quantity, avg_cost, total_cost FROM holdings
                        WHERE account_id = %s AND asset_id = %s
                    ''', (account_id, asset_id))
                    row = cursor.fetchone()

                    if row:
                        old_qty, old_avg_cost, old_total_cost = row
                        old_avg_cost = old_avg_cost or 0
                        old_total_cost = old_total_cost or 0
                        new_qty = old_qty + quantity
                        new_total_cost = old_total_cost + (quantity * price)
                        new_avg_cost = new_total_cost / new_qty if new_qty > 0 else 0
                    else:
                        new_qty = quantity
                        new_avg_cost = price
                        new_total_cost = quantity * price

                    # 보유정보 저장 (insert or update)
                    cursor.execute('''
                        INSERT INTO holdings (account_id, asset_id, quantity, avg_cost, total_cost)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (account_id, asset_id) DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        avg_cost = EXCLUDED.avg_cost,
                        total_cost = EXCLUDED.total_cost,
                        last_updated = NOW()
                    ''', (account_id, asset_id, new_qty, new_avg_cost, new_total_cost))

                elif tx_type == 'sell':
                    # 매도 시 수량 차감 (평균가는 변경 없음)
                    cursor.execute('''
                        UPDATE holdings
                        SET quantity = quantity - %s, last_updated = NOW()
                        WHERE account_id = %s AND asset_id = %s
                    ''', (quantity, account_id, asset_id))

            conn.commit()
            st.cache_data.clear()
            conn.close()
            return True
        except Exception as e:
            print(f"Save Transaction Error: {e}")
            return False

    @classmethod
    def list_transactions(cls, account_id, limit=100):
        """거래 내역 조회"""
        cls._init_asset_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            sql = '''
                SELECT t.id, t.asset_id, a.name, t.tx_type, t.quantity, t.price, t.amount, t.fee, t.currency, t.tx_date, t.notes
                FROM transactions t
                JOIN assets a ON t.asset_id = a.id
                WHERE t.account_id = %s
                ORDER BY t.tx_date DESC, t.created_at DESC
                LIMIT %s
            '''
            cursor.execute(sql, (account_id, limit))
            rows = cursor.fetchall()
            conn.close()
            return [
                {
                    'id': row[0],
                    'asset_id': row[1],
                    'asset_name': row[2],
                    'type': row[3],
                    'quantity': float(row[4]) if row[4] else None,
                    'price': float(row[5]) if row[5] else None,
                    'amount': float(row[6]),
                    'fee': float(row[7]) if row[7] else 0,
                    'currency': row[8],
                    'tx_date': row[9],
                    'notes': row[10]
                }
                for row in rows
            ]
        except Exception as e:
            print(f"List Transactions Error: {e}")
            return []

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
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Signal Events DB Init Error: {e}")

    @classmethod
    def save_signal_event(cls, account_id, portfolio_name, strategy_name, strat_type,
                          prev_stage, new_stage, prev_asset, new_asset, action,
                          total_value=None, notes=None, event_date=None):
        """신호 이벤트 저장"""
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
