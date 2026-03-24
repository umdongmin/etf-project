"""
core/config_loader.py — DB 우선 설정 로더

앱 시작 시 Supabase에서 기본 전략을 로드하여 session_state를 초기화.
DB에 값이 없을 때만 core/defaults.py 하드코딩 값을 폴백으로 사용.
"""
import copy
from core.defaults import (
    get_default_equity_params, get_default_bond_params, get_default_bond_lev_params,
)


def load_default_equity_params() -> tuple:
    """DB에서 기본 주식 전략을 로드.

    Returns:
        (name: str | None, params: dict)
        name: DB 전략 이름 (없으면 None)
        params: 로드된 파라미터 (없으면 하드코딩 기본값)
    """
    from core.storage import StrategyStorage
    try:
        default_name = StrategyStorage.get_default_strategy(category='equity')
        if default_name:
            loaded = StrategyStorage.load_strategy(default_name)
            if loaded and isinstance(loaded, dict) and 'buy_signals' in loaded:
                return default_name, loaded
    except Exception as e:
        print(f"[config_loader] equity 로드 실패: {e}")
    return None, get_default_equity_params()


def load_default_bond_params() -> tuple:
    """DB에서 기본 채권 전략을 로드.

    strategies 테이블의 params(Layer1) + lev_params(Layer2) 분리 구조를 사용.

    Returns:
        (name: str | None, params: dict, lev_params: dict)
    """
    from core.storage import StrategyStorage
    try:
        default_name = StrategyStorage.get_default_strategy(category='bond')
        if default_name:
            loaded = StrategyStorage.load_strategy(default_name)
            if loaded and isinstance(loaded, dict):
                lev_p = loaded.pop('lev_params', None) or get_default_bond_lev_params()
                loaded.pop('category', None)
                return default_name, loaded, lev_p
    except Exception as e:
        print(f"[config_loader] bond 로드 실패: {e}")
    return None, get_default_bond_params(), get_default_bond_lev_params()


def load_default_portfolio_name() -> str:
    """DB에서 기본 포트폴리오 이름을 반환 (없으면 None)."""
    from core.storage import StrategyStorage
    try:
        return StrategyStorage.get_default_portfolio()
    except Exception:
        return None
