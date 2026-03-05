import sqlite3
import os
import sys
from dotenv import load_dotenv

# 현재 경로를 sys.path에 추가하여 core 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psycopg2
    from psycopg2 import sql
except ImportError:
    print("[Error] psycopg2-binary 패키지가 설치되어 있지 않습니다. pip install psycopg2-binary 를 먼저 실행하세요.")
    sys.exit(1)

# 스크립트 파일의 위치를 기준으로 프로젝트 루트 경로 계산
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
load_dotenv(env_path)

def migrate_table(table_name, local_conn, remote_conn):
    print(f"Migrating table: {table_name}...")
    local_cursor = local_conn.cursor()
    remote_cursor = remote_conn.cursor()
    
    # 1. 로컬에서 데이터 가져오기
    local_cursor.execute(f"SELECT * FROM {table_name}")
    rows = local_cursor.fetchall()
    
    if not rows:
        print(f"No data in local table {table_name}.")
        return

    # 2. 컬럼명 가져오기
    col_names = [description[0] for description in local_cursor.description]
    
    # 3. 원격(Supabase) 테이블 생성 (있는지 확인 후)
    # PostgreSQL에 맞게 가벼운 타입 매핑 필요할 수 있으나, SQLite TEXT/REAL은 대체로 호환
    # 여기서는 간단하게 기존 데이터 기반 삽입 시도
    
    col_str = ", ".join(col_names)
    placeholder_str = ", ".join(["%s"] * len(col_names))
    
    # INSERT 쿼리 (ON CONFLICT는 PK가 있는 경우에만 작동하므로 테이블별 대응 필요)
    # news_sentiment는 date가 PK
    if table_name == "news_sentiment":
        insert_sql = f"""
            INSERT INTO {table_name} ({col_str})
            VALUES ({placeholder_str})
            ON CONFLICT (date) DO UPDATE SET
            sentiment = EXCLUDED.sentiment,
            impact = EXCLUDED.impact,
            topic = EXCLUDED.topic,
            summary = EXCLUDED.summary,
            raw_headlines = EXCLUDED.raw_headlines,
            updated_at = CURRENT_TIMESTAMP
        """
    else:
        # Optuna 테이블 등은 그냥 삽입 (중복 방지 로직 미포함 - 초기 이전용)
        insert_sql = f"INSERT INTO {table_name} ({col_str}) VALUES ({placeholder_str})"
    
    count = 0
    for row in rows:
        try:
            remote_cursor.execute(insert_sql, row)
            count += 1
        except Exception as e:
            # 중복 오류 등은 무시하고 진행
            if "duplicate key" not in str(e).lower():
                print(f"  [Warning] Error in {table_name} row: {e}")
            
    remote_conn.commit()
    print(f"Successfully migrated {count}/{len(rows)} rows to {table_name}.")

def main():
    supabase_url = os.getenv("SUPABASE_DB_URL")
    
    if not supabase_url:
        print("[Error] .env 파일에 SUPABASE_DB_URL 설정이 필요합니다.")
        return

    # 1. 프로젝트 내 테이블 생성 (초기화)
    # NewsService 등을 통해 테이블이 이미 생성되어 있을 수도 있지만, 마이그레이션용으로 명시적 생성
    try:
        remote_conn = psycopg2.connect(supabase_url)
        remote_cursor = remote_conn.cursor()
        
        print("Initializing Supabase schema...")
        remote_cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sentiment (
                date TEXT PRIMARY KEY,
                sentiment REAL,
                impact REAL,
                topic TEXT,
                summary TEXT,
                raw_headlines TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        remote_conn.commit()
        
    except Exception as e:
        print(f"[Error] Remote DB connection/init failed: {e}")
        return

    # 2. 데이터 마이그레이션
    # 스크립트 파일의 위치를 기준으로 프로젝트 루트 경로 계산
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_news_db = os.path.join(base_dir, "data", "news_sentiment.db")
    
    if os.path.exists(local_news_db):
        print(f"\n[1/1] Migrating News Data ({local_news_db})")
        local_conn = sqlite3.connect(local_news_db)
        migrate_table("news_sentiment", local_conn, remote_conn)
        local_conn.close()
    else:
        print(f"\n[Skipped] News DB not found at {local_news_db}")

    remote_conn.close()
    print("\n[Migration Completed] Local news data pushed to Supabase!")
    print("\nNote: Optuna optimization history is not automatically migrated due to complex schema.")
    print("New optimizations will be stored in Supabase automatically.")

if __name__ == "__main__":
    main()
