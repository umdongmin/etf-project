import os
import json
import sqlite3
import datetime

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

class StrategyStorage:
    """전략 설정을 DB(Supabase/SQLite)로 관리하는 클래스"""
    
    # [수정] data/ 폴더 의존성 제거. 프로젝트 루트에 숨김 파일(.db) 형태로 저장
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".strategies.db")
    
    @classmethod
    def _get_connection(cls):
        """DB 연결 객체 반환"""
        supabase_url = os.getenv("SUPABASE_DB_URL")
        if supabase_url and HAS_PSYCOPG2:
            try:
                return psycopg2.connect(supabase_url)
            except Exception as e:
                print(f"Supabase Connection Error (Storage), falling back to local: {e}")
        
        return sqlite3.connect(cls.DB_PATH)

    @classmethod
    def _init_db(cls):
        """전략 테이블 초기화 및 자동 마이그레이션"""
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            # 1. 테이블 생성
            if hasattr(conn, 'dsn'): 
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                        name TEXT PRIMARY KEY,
                        params JSONB,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            else:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS strategies (
                        name TEXT PRIMARY KEY,
                        params TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            conn.commit()

            # 2. 데이터 유무 확인 및 자동 마이그레이션 (중요)
            cursor.execute("SELECT COUNT(*) FROM strategies")
            if cursor.fetchone()[0] == 0:
                cls._auto_migrate_from_json(cursor, conn)
                
            conn.close()
        except Exception as e:
            print(f"Strategy DB Init Error: {e}")

    @classmethod
    def _auto_migrate_from_json(cls, cursor, conn):
        """기존 JSON 파일들에서 DB로 데이터 자동 이전"""
        strategy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")
        if not os.path.exists(strategy_dir):
            return

        json_files = [f for f in os.listdir(strategy_dir) if f.endswith(".json")]
        if not json_files:
            return

        print(f"--- Detected {len(json_files)} legacy strategy files. Starting auto-migration... ---")
        
        for filename in json_files:
            name = filename.replace(".json", "")
            filepath = os.path.join(strategy_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                params_json = json.dumps(params, ensure_ascii=False)
                
                if hasattr(conn, 'dsn'):
                    cursor.execute("INSERT INTO strategies (name, params) VALUES (%s, %s) ON CONFLICT DO NOTHING", (name, params_json))
                else:
                    cursor.execute("INSERT OR IGNORE INTO strategies (name, params) VALUES (?, ?)", (name, params_json))
                
                print(f" [OK] Migrated: {name}")
            except Exception as e:
                print(f" [ERROR] Failed to migrate {name}: {e}")
        
        conn.commit()

    @classmethod
    def save_strategy(cls, name, params):
        """전략 파라미터를 DB에 저장"""
        cls._init_db() # 자동 초기화 보장
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            params_json = json.dumps(params, ensure_ascii=False)
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                sql = '''
                    INSERT INTO strategies (name, params)
                    VALUES (%s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                    params = EXCLUDED.params,
                    updated_at = CURRENT_TIMESTAMP
                '''
            else: # SQLite
                sql = "INSERT OR REPLACE INTO strategies (name, params, updated_at) VALUES (?, ?, ?)"
                
            if hasattr(conn, 'dsn'):
                cursor.execute(sql, (name, params_json))
            else:
                cursor.execute(sql, (name, params_json, datetime.datetime.now()))
                
            conn.commit()
            conn.close()
            return name
        except Exception as e:
            print(f"Save Strategy Error: {e}")
            return None

    @classmethod
    def list_strategies(cls):
        """저장된 전략 이름 목록 추출"""
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
        """특정 전략 로드"""
        cls._init_db()
        try:
            conn = cls._get_connection()
            cursor = conn.cursor()
            
            if hasattr(conn, 'dsn'): # PostgreSQL
                cursor.execute("SELECT params FROM strategies WHERE name = %s", (name,))
            else: # SQLite
                cursor.execute("SELECT params FROM strategies WHERE name = ?", (name,))
                
            row = cursor.fetchone()
            conn.close()
            
            if row:
                params = row[0]
                return json.loads(params) if isinstance(params, str) else params
            return None
        except Exception as e:
            print(f"Load Strategy Error: {e}")
            return None
