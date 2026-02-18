import os
import json

class StrategyStorage:
    """전략 설정을 로컬 JSON 파일로 관리하는 클래스"""
    STRATEGY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategies")

    @classmethod
    def save_strategy(cls, name, params):
        """전략 파라미터를 JSON 파일로 저장"""
        if not os.path.exists(cls.STRATEGY_DIR):
            os.makedirs(cls.STRATEGY_DIR)
        
        filepath = os.path.join(cls.STRATEGY_DIR, f"{name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
        return filepath

    @classmethod
    def list_strategies(cls):
        """저장된 전략 목록 추출"""
        if not os.path.exists(cls.STRATEGY_DIR):
            return []
        return [f.replace(".json", "") for f in os.listdir(cls.STRATEGY_DIR) if f.endswith(".json")]

    @classmethod
    def load_strategy(cls, name):
        """특정 전략 파일 로드"""
        filepath = os.path.join(cls.STRATEGY_DIR, f"{name}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
