"""
core/dispatcher.py — 중앙 이벤트 디스패처 (Single Unidirectional Flow)

[데이터 흐름]
    UI 이벤트 발생
        → Dispatcher.dispatch(action, payload)
            → 이전 상태와 diff 비교
            → 변경된 경우에만 AppState 갱신 + dirty flag 설정
            → 엔진은 dirty flag가 있을 때만, 사용자가 명시적으로 실행 요청 시 호출

모든 UI 컴포넌트는 이 모듈을 통해서만 상태를 변경한다.
직접 st.session_state를 수정하거나 엔진을 호출하지 않는다.
"""
import copy
import hashlib
import json
import streamlit as st

from core.session_keys import SK


# ── 액션 타입 상수 ────────────────────────────────────────────────────────────
class Action:
    # 전략 파라미터 변경
    SET_EQUITY_PARAMS        = 'set_equity_params'
    SET_BOND_PARAMS          = 'set_bond_params'

    # 포트폴리오 구성 변경
    LOAD_PORTFOLIO           = 'load_portfolio'
    UPDATE_PORTFOLIO_STRATEGY = 'update_portfolio_strategy'
    UPDATE_PORTFOLIO_WEIGHT  = 'update_portfolio_weight'
    ADD_PORTFOLIO_STRATEGY   = 'add_portfolio_strategy'
    REMOVE_PORTFOLIO_STRATEGY = 'remove_portfolio_strategy'

    # 글로벌 설정 변경
    SET_FEE_RATE             = 'set_fee_rate'
    SET_INTEGER_SHARES       = 'set_integer_shares'
    SET_DATE_RANGE           = 'set_date_range'

    # 엔진 실행 요청 (명시적 사용자 액션에만 발생)
    REQUEST_EQUITY_RUN       = 'request_equity_run'
    REQUEST_PORTFOLIO_RUN    = 'request_portfolio_run'
    REQUEST_BOND_RUN         = 'request_bond_run'


def _stable_hash(obj) -> str:
    """딕셔너리/리스트를 안정적으로 해시화.

    opt_* / rng_* 키는 엔진 결과에 영향이 없으므로 캐시 키에서 제외.
    """
    def _normalize(o):
        if isinstance(o, dict):
            return {k: _normalize(v) for k, v in sorted(o.items())
                    if not (k.startswith('opt_') or k.startswith('rng_'))}
        if isinstance(o, list):
            return [_normalize(i) for i in o]
        return o

    normalized = _normalize(obj)
    return hashlib.md5(
        json.dumps(normalized, sort_keys=True, default=str).encode()
    ).hexdigest()


class Dispatcher:
    """
    중앙 이벤트 디스패처.

    사용법:
        changed = Dispatcher.dispatch(Action.SET_EQUITY_PARAMS, {'params': new_p})
        if changed:
            st.rerun()
    """

    @staticmethod
    def dispatch(action: str, payload: dict | None = None) -> bool:
        """액션을 처리하고, 실제 상태 변경이 일어났으면 True를 반환.

        Returns:
            bool: 상태가 변경된 경우 True (호출 측에서 st.rerun() 필요 여부 판단)
        """
        payload = payload or {}

        handler = _HANDLERS.get(action)
        if handler is None:
            raise ValueError(f"[Dispatcher] 알 수 없는 액션: '{action}'")
        return handler(payload)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────
    @staticmethod
    def is_equity_dirty() -> bool:
        return st.session_state.get(SK.TRIGGER_RECALC_HISTORY, False)

    @staticmethod
    def is_bond_dirty() -> bool:
        return st.session_state.get(SK.TRIGGER_RECALC_BOND, False)

    @staticmethod
    def is_portfolio_dirty() -> bool:
        return st.session_state.get(_DK.PORTFOLIO_DIRTY, False)

    @staticmethod
    def clear_equity_dirty():
        st.session_state[SK.TRIGGER_RECALC_HISTORY] = False

    @staticmethod
    def clear_bond_dirty():
        st.session_state[SK.TRIGGER_RECALC_BOND] = False

    @staticmethod
    def clear_portfolio_dirty():
        st.session_state[_DK.PORTFOLIO_DIRTY] = False


# ── 디스패처 전용 dirty 키 (SK에 없는 것만) ──────────────────────────────────
class _DK:
    PORTFOLIO_DIRTY   = '_dispatcher_portfolio_dirty'
    EQUITY_PARAM_HASH = '_dispatcher_equity_hash'
    BOND_PARAM_HASH   = '_dispatcher_bond_hash'
    PORTFOLIO_HASH    = '_dispatcher_portfolio_hash'


# ── 개별 액션 핸들러 ──────────────────────────────────────────────────────────

def _handle_set_equity_params(payload: dict) -> bool:
    from core.state import AppState
    new_params = payload.get('params', {})
    name       = payload.get('name')

    new_hash = _stable_hash(new_params)
    if st.session_state.get(_DK.EQUITY_PARAM_HASH) == new_hash:
        return False  # 실제 변경 없음 → 엔진 불필요

    AppState.set_equity_params(new_params, strategy_name=name)
    st.session_state[_DK.EQUITY_PARAM_HASH]    = new_hash
    st.session_state[SK.TRIGGER_RECALC_HISTORY] = True
    return True


def _handle_set_bond_params(payload: dict) -> bool:
    from core.state import AppState
    new_params     = payload.get('params', {})
    new_lev_params = payload.get('lev_params')
    name           = payload.get('name')

    new_hash = _stable_hash({'p': new_params, 'lev': new_lev_params})
    if st.session_state.get(_DK.BOND_PARAM_HASH) == new_hash:
        return False

    AppState.set_bond_params(new_params, lev_params=new_lev_params, strategy_name=name)
    st.session_state[_DK.BOND_PARAM_HASH]   = new_hash
    st.session_state[SK.TRIGGER_RECALC_BOND] = True
    return True


def _handle_load_portfolio(payload: dict) -> bool:
    """포트폴리오 전체를 교체 (불러오기 버튼)."""
    configs       = payload.get('configs', [])
    total_capital = payload.get('total_capital', 10000.0)
    preset        = payload.get('rebalance_preset', 'none')

    new_hash = _stable_hash({'c': configs, 'tc': total_capital, 'pr': preset})
    if st.session_state.get(_DK.PORTFOLIO_HASH) == new_hash:
        return False

    from core.state import AppState
    AppState.set_portfolio_configs(copy.deepcopy(configs))
    st.session_state[SK.PORTFOLIO_TOTAL_CAPITAL]    = total_capital
    st.session_state[SK.PORTFOLIO_REBALANCE_PRESET] = preset
    st.session_state[_DK.PORTFOLIO_HASH]            = new_hash
    st.session_state[_DK.PORTFOLIO_DIRTY]           = True
    return True


def _handle_update_portfolio_strategy(payload: dict) -> bool:
    """포트폴리오 내 개별 전략 교체 (드롭다운 변경).

    DB 호출은 이 핸들러 안에서만 발생하며,
    실제 변경이 없으면 호출하지 않는다.
    """
    idx      = payload['index']
    new_name = payload['name']
    new_type = payload['type']

    from core.state import AppState
    configs  = AppState.get_portfolio_configs()
    old_cfg  = configs[idx]

    if old_cfg.get('name') == new_name and old_cfg.get('type') == new_type:
        return False  # 동일 → 아무것도 안 함

    # 실제 변경된 경우에만 DB 조회
    from core.storage import StrategyStorage
    loaded = StrategyStorage.load_strategy(new_name)
    if not loaded:
        return False

    lev_p = loaded.pop('lev_params', None)
    loaded.pop('category', None)
    configs[idx]['name']   = new_name
    configs[idx]['type']   = new_type
    configs[idx]['params'] = copy.deepcopy(loaded)
    if lev_p is not None:
        configs[idx]['lev_params'] = copy.deepcopy(lev_p)
    elif 'lev_params' in configs[idx]:
        del configs[idx]['lev_params']

    AppState.set_portfolio_configs(configs)
    new_hash = _stable_hash(configs)
    st.session_state[_DK.PORTFOLIO_HASH]  = new_hash
    st.session_state[_DK.PORTFOLIO_DIRTY] = True
    return True


def _handle_update_portfolio_weight(payload: dict) -> bool:
    """비중 변경 (number_input). 엔진은 트리거하지 않고 dirty만 설정."""
    idx    = payload['index']
    weight = payload['weight']

    from core.state import AppState
    configs = AppState.get_portfolio_configs()
    if abs(configs[idx].get('weight', 0.0) - weight) < 1e-9:
        return False

    configs[idx]['weight'] = weight
    AppState.set_portfolio_configs(configs)
    st.session_state[_DK.PORTFOLIO_DIRTY] = True
    return True


def _handle_add_portfolio_strategy(payload: dict) -> bool:
    from core.state import AppState
    new_cfg = payload.get('config', {})
    configs = AppState.get_portfolio_configs()
    configs.append(copy.deepcopy(new_cfg))
    AppState.set_portfolio_configs(configs)
    st.session_state[_DK.PORTFOLIO_DIRTY] = True
    return True


def _handle_remove_portfolio_strategy(payload: dict) -> bool:
    idx = payload['index']
    from core.state import AppState
    configs = AppState.get_portfolio_configs()
    if idx < 0 or idx >= len(configs):
        return False
    configs.pop(idx)
    AppState.set_portfolio_configs(configs)
    st.session_state[_DK.PORTFOLIO_DIRTY] = True
    return True


def _handle_set_fee_rate(payload: dict) -> bool:
    from core.state import AppState
    new_rate = payload['fee_rate']
    if abs(AppState.get_fee_rate() - new_rate) < 1e-9:
        return False
    AppState.set_fee_rate(new_rate)
    # 수수료 변경은 모든 계산에 영향 → 전체 dirty
    st.session_state[SK.TRIGGER_RECALC_HISTORY] = True
    st.session_state[SK.TRIGGER_RECALC_BOND]    = True
    st.session_state[_DK.PORTFOLIO_DIRTY]        = True
    return True


def _handle_set_integer_shares(payload: dict) -> bool:
    from core.state import AppState
    new_val = payload['value']
    if AppState.get_integer_shares() == new_val:
        return False
    AppState.set_integer_shares(new_val)
    st.session_state[SK.TRIGGER_RECALC_HISTORY] = True
    st.session_state[SK.TRIGGER_RECALC_BOND]    = True
    st.session_state[_DK.PORTFOLIO_DIRTY]        = True
    return True


def _handle_set_date_range(payload: dict) -> bool:
    """날짜 범위 변경. 단순 세션 저장만 하고 dirty flag는 건드리지 않음.
    (날짜는 엔진 호출 시 인자로 직접 전달되므로 별도 재계산 트리거 불필요)
    """
    st.session_state['_date_start'] = payload.get('start')
    st.session_state['_date_end']   = payload.get('end')
    return False  # 탭 내 렌더링은 다음 rerun에서 반영


def _handle_request_equity_run(payload: dict) -> bool:
    """사용자가 명시적으로 주식 전략 실행을 요청."""
    st.session_state[SK.TRIGGER_RECALC_HISTORY] = True
    return True


def _handle_request_portfolio_run(payload: dict) -> bool:
    """사용자가 명시적으로 포트폴리오 시뮬레이션을 요청."""
    st.session_state[_DK.PORTFOLIO_DIRTY] = True
    return True


def _handle_request_bond_run(payload: dict) -> bool:
    """사용자가 명시적으로 채권 전략 실행을 요청."""
    st.session_state[SK.TRIGGER_RECALC_BOND] = True
    return True


# ── 핸들러 레지스트리 ─────────────────────────────────────────────────────────
_HANDLERS: dict[str, callable] = {
    Action.SET_EQUITY_PARAMS:          _handle_set_equity_params,
    Action.SET_BOND_PARAMS:            _handle_set_bond_params,
    Action.LOAD_PORTFOLIO:             _handle_load_portfolio,
    Action.UPDATE_PORTFOLIO_STRATEGY:  _handle_update_portfolio_strategy,
    Action.UPDATE_PORTFOLIO_WEIGHT:    _handle_update_portfolio_weight,
    Action.ADD_PORTFOLIO_STRATEGY:     _handle_add_portfolio_strategy,
    Action.REMOVE_PORTFOLIO_STRATEGY:  _handle_remove_portfolio_strategy,
    Action.SET_FEE_RATE:               _handle_set_fee_rate,
    Action.SET_INTEGER_SHARES:         _handle_set_integer_shares,
    Action.SET_DATE_RANGE:             _handle_set_date_range,
    Action.REQUEST_EQUITY_RUN:         _handle_request_equity_run,
    Action.REQUEST_PORTFOLIO_RUN:      _handle_request_portfolio_run,
    Action.REQUEST_BOND_RUN:           _handle_request_bond_run,
}
