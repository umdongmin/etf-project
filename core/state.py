"""
core/state.py — Streamlit 세션 상태 단일 관리자 (Single Source of Truth)

모든 UI 컴포넌트는 st.session_state를 직접 조작하지 않고
이 모듈의 get/set 함수를 통해서만 접근한다.

사용법:
    from core.state import AppState
    params = AppState.get_equity_params()
    AppState.set_equity_params(new_params)

이벤트 아키텍처에서의 역할:
    AppState는 순수한 상태 저장소(Store)이며,
    상태 변경의 부작용(dirty flag, 캐시 무효화)은
    core/dispatcher.py 에서 관리한다.
"""
import copy
import streamlit as st
from core.session_keys import SK


# ── 키 매핑: SK에 있는 것은 위임, 없는 것만 여기 정의 ───────────────────────
class _Keys:
    # SK 위임 (session_keys.py가 단일 진실의 원천)
    EQUITY_PARAMS     = SK.CURRENT_PARAMS
    BOND_PARAMS       = SK.CURRENT_BOND_PARAMS
    BOND_LEV_PARAMS   = SK.CURRENT_BOND_LEV_PARAMS
    FEE_RATE          = SK.GLOBAL_FEE_RATE
    INTEGER_SHARES    = SK.GLOBAL_INTEGER_SHARES
    PORTFOLIO_CONFIGS = SK.PORTFOLIO_CONFIGS
    PORTFOLIO_RESULT  = SK.PORTFOLIO_RESULT
    INIT_VER_BOND     = SK.INITIAL_BOND_LOAD_V28

    # state.py 전용 (SK에 없는 것)
    EQUITY_STRAT_NAME = 'default_equity_strategy_name'
    BOND_STRAT_NAME   = 'default_bond_strategy_name'
    INIT_VER_EQUITY   = 'init_ver_equity_v1'


class AppState:
    """
    세션 상태 읽기/쓰기 인터페이스.

    직접 st.session_state[key]에 접근하는 대신 이 클래스를 사용한다.
    모든 getter는 방어적 deepcopy를 반환하여 외부에서 원본을 변경할 수 없게 한다.
    """

    # ── 주식 전략 ────────────────────────────────────────────────────────────
    @staticmethod
    def get_equity_params() -> dict:
        return copy.deepcopy(st.session_state.get(_Keys.EQUITY_PARAMS, {}))

    @staticmethod
    def set_equity_params(params: dict, strategy_name: str = None):
        st.session_state[_Keys.EQUITY_PARAMS] = copy.deepcopy(params)
        if strategy_name:
            st.session_state[_Keys.EQUITY_STRAT_NAME] = strategy_name
        AppState._invalidate_engine_cache()

    @staticmethod
    def get_equity_strategy_name() -> str | None:
        return st.session_state.get(_Keys.EQUITY_STRAT_NAME)

    # ── 채권 전략 ────────────────────────────────────────────────────────────
    @staticmethod
    def get_bond_params() -> dict:
        return copy.deepcopy(st.session_state.get(_Keys.BOND_PARAMS, {}))

    @staticmethod
    def get_bond_lev_params() -> dict | None:
        return copy.deepcopy(st.session_state.get(_Keys.BOND_LEV_PARAMS))

    @staticmethod
    def set_bond_params(params: dict, lev_params: dict = None, strategy_name: str = None):
        st.session_state[_Keys.BOND_PARAMS] = copy.deepcopy(params)
        if lev_params is not None:
            st.session_state[_Keys.BOND_LEV_PARAMS] = copy.deepcopy(lev_params)
        if strategy_name:
            st.session_state[_Keys.BOND_STRAT_NAME] = strategy_name
        AppState._invalidate_engine_cache()

    @staticmethod
    def get_bond_strategy_name() -> str | None:
        return st.session_state.get(_Keys.BOND_STRAT_NAME)

    # ── 글로벌 트레이딩 설정 ─────────────────────────────────────────────────
    @staticmethod
    def get_fee_rate() -> float:
        return st.session_state.get(_Keys.FEE_RATE, 0.0)

    @staticmethod
    def set_fee_rate(fee: float):
        st.session_state[_Keys.FEE_RATE] = fee

    @staticmethod
    def get_integer_shares() -> bool:
        return st.session_state.get(_Keys.INTEGER_SHARES, False)

    @staticmethod
    def set_integer_shares(val: bool):
        st.session_state[_Keys.INTEGER_SHARES] = val

    # ── 포트폴리오 ───────────────────────────────────────────────────────────
    @staticmethod
    def get_portfolio_configs() -> list:
        return copy.deepcopy(st.session_state.get(_Keys.PORTFOLIO_CONFIGS, []))

    @staticmethod
    def set_portfolio_configs(configs: list):
        st.session_state[_Keys.PORTFOLIO_CONFIGS] = copy.deepcopy(configs)
        AppState.clear_portfolio_result()

    @staticmethod
    def get_portfolio_result() -> dict | None:
        return st.session_state.get(_Keys.PORTFOLIO_RESULT)

    @staticmethod
    def set_portfolio_result(result: dict):
        st.session_state[_Keys.PORTFOLIO_RESULT] = result

    @staticmethod
    def clear_portfolio_result():
        st.session_state.pop(_Keys.PORTFOLIO_RESULT, None)

    # ── 캐시 무효화 ──────────────────────────────────────────────────────────
    @staticmethod
    def _invalidate_engine_cache():
        """파라미터 변경 시 엔진 캐시를 무효화한다.

        @st.cache_data(ttl=3600)가 파라미터 변화를 감지하지 못하는 케이스를
        방어하기 위해 전략 파라미터 변경 시 항상 호출한다.
        """
        try:
            st.cache_data.clear()
        except Exception:
            pass

    # ── 앱 초기화 ────────────────────────────────────────────────────────────
    @staticmethod
    def init_equity(force: bool = False):
        """주식 전략 세션 초기화 (앱 최초 로드 시 1회만).

        DB 우선 → defaults.py fallback 순서로 로드.
        """
        if not force and st.session_state.get(_Keys.INIT_VER_EQUITY):
            return

        from core.config_loader import load_default_equity_params
        name, params = load_default_equity_params()
        params.pop('category', None)
        AppState.set_equity_params(params, strategy_name=name)
        st.session_state[_Keys.INIT_VER_EQUITY] = True

    # ── 포트폴리오 dirty 상태 (엔진 재실행 필요 여부) ──────────────────────────
    @staticmethod
    def mark_portfolio_stale(stale: bool = True):
        """포트폴리오 구성이 변경되어 재시뮬레이션이 필요함을 표시."""
        from core.dispatcher import _DK
        st.session_state[_DK.PORTFOLIO_DIRTY] = stale

    @staticmethod
    def is_portfolio_stale() -> bool:
        from core.dispatcher import _DK
        return st.session_state.get(_DK.PORTFOLIO_DIRTY, False)

    @staticmethod
    def init_bond(force: bool = False):
        """채권 전략 세션 초기화 (앱 최초 로드 시 1회만).

        DB 우선 → defaults.py fallback 순서로 로드.
        """
        if not force and st.session_state.get(_Keys.INIT_VER_BOND):
            return

        from core.config_loader import load_default_bond_params
        name, params, lev_params = load_default_bond_params()
        AppState.set_bond_params(params, lev_params=lev_params, strategy_name=name)
        st.session_state[_Keys.INIT_VER_BOND] = True
