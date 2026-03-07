import os
import json
import datetime

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
        """전략 테이블 초기화 (PostgreSQL 전용)"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    name TEXT PRIMARY KEY,
                    params JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

            # 데이터 유무 확인 및 자동 마이그레이션 (기존 로직 유지)
            cursor.execute("SELECT COUNT(*) FROM strategies")
            if cursor.fetchone()[0] == 0:
                cls._auto_migrate_from_json(cursor, conn)
                
            conn.close()
        except Exception as e:
            print(f"Strategy DB Init Error (Postgres): {e}")

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
    def save_strategy(cls, name, params):
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            params_json = json.dumps(params, ensure_ascii=False)
            
            sql = '''
                INSERT INTO strategies (name, params)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET
                params = EXCLUDED.params,
                updated_at = CURRENT_TIMESTAMP
            '''
            cursor.execute(sql, (name, params_json))
            conn.commit()
            conn.close()
            return name
        except Exception as e:
            print(f"Save Strategy Error: {e}")
            return None

    @classmethod
    def list_strategies(cls):
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
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
            cursor.execute("SELECT params FROM strategies WHERE name = %s", (name,))
            row = cursor.fetchone()
            conn.close()
            if row:
                params = row[0]
                return json.loads(params) if isinstance(params, str) else params
            return None
        except Exception as e:
            print(f"Load Strategy Error: {e}")
            return None
