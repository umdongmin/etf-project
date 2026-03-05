import os
import json
import sys
import datetime

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.storage import StrategyStorage

def migrate():
    """로컬 JSON 전략 파일들을 DB로 마이그레이션"""
    strategy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")
    log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migration.log")
    
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n--- 마이그레이션 시작: {datetime.datetime.now()} ---\n")
        
        if not os.path.exists(strategy_dir):
            log.write(" [!] strategies 폴더가 존재하지 않습니다.\n")
            print(" [!] strategies 폴더가 존재하지 않습니다.")
            return

        files = [f for f in os.listdir(strategy_dir) if f.endswith(".json")]
        log.write(f"대상 파일: {len(files)}개\n")
        
        success_count = 0
        for filename in files:
            name = filename.replace(".json", "")
            filepath = os.path.join(strategy_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                
                res = StrategyStorage.save_strategy(name, params)
                if res:
                    log.write(f" [OK] {name} -> DB 저장 완료\n")
                    print(f" [OK] {name} -> DB 저장 완료")
                    success_count += 1
                else:
                    log.write(f" [FAIL] {name} -> DB 저장 실패 (저장 함수 반환값 None)\n")
            except Exception as e:
                log.write(f" [ERROR] {name} 마이그레이션 실패: {e}\n")
                print(f" [ERROR] {name} 마이그레이션 실패: {e}")

        log.write(f"마이그레이션 종료: {success_count}/{len(files)} 성공\n")
        print(f"\n--- 마이그레이션 완료: {success_count}/{len(files)} 성공 ---")

if __name__ == "__main__":
    migrate()
