"""
core/engine_kwargs_builder.py — 엔진 호출 kwargs 중앙 집중화

모든 UI 뷰/서비스에서 StrategyEngine 호출 시 이 빌더를 사용.
smart_params 추출, equity/bond kwargs 구성을 단일화하여 결과 불일치를 근본적으로 차단.
"""
import copy
from core.defaults import REBALANCE_PRESET_CONFIGS


def extract_smart_params(params: dict) -> dict:
    """params 딕셔너리에서 smart_params 서브셋을 추출.

    이 함수만이 smart_params 키 목록의 유일한 정의이다.
    app.py, signal_service.py, portfolio_view.py 등 5곳의 중복을 제거.
    """
    return {
        'use_panic':         params.get('use_panic', True),
        'panic_ma':          params.get('panic_ma', 200),
        'panic_buy_signals': params.get('panic_buy_signals', []),
        'panic_rsi_s1':      params.get('panic_rsi_s1', 27),
        'panic_rsi_s2':      params.get('panic_rsi_s2', 28),
        'panic_rsi_s3':      params.get('panic_rsi_s3', 30),
        'use_vxn_safety':    params.get('use_vxn_safety', False),
        'vxn_exit':          params.get('vxn_exit', 31.0),
        'use_rsi_turbo':     params.get('use_rsi_turbo', False),
        'rsi_turbo':         params.get('rsi_turbo', 31.0),
        'use_sl_control':    params.get('use_sl_control', False),
        'sl_control_limit':  params.get('sl_control_limit', -15.0),
        'use_set_sl':        params.get('use_set_sl', False),
        'set_sl_limit':      params.get('set_sl_limit', -15.0),
        'dynamic_cash':      params.get('dynamic_cash', {'use': False}),
    }


def build_equity_kwargs(
    data_dict, fg_df, vxn_df, news_df,
    params: dict,
    start_date, end_date,
    allocated_capital: float = 10000.0,
    fee_rate: float = 0.0,
    integer_shares: bool = False,
    fast_mode: bool = False,
) -> dict:
    """주식 전략 엔진 호출을 위한 kwargs 딕셔너리를 구성.

    StrategyEngine.run_golden_strategy(**kwargs) 또는
    PortfolioManager.register_strategy(name, weight, kwargs, strat_type='equity') 로 직접 사용.
    """
    smart_p = extract_smart_params(params)
    return {
        'data_dict': data_dict,
        'fg_df': fg_df,
        'vxn_df': vxn_df,
        'news_df': news_df,
        'leverage_asset': params.get('leverage_asset', 'TQQQ'),
        'base_asset': params.get('base_asset', 'QQQ'),
        'cash_ratio': params.get('cash_ratio_pct', 0.0) / 100.0 if params.get('cash_ratio_pct', 0.0) > 1.0 else params.get('cash_ratio_pct', 0.0),
        'start_date': start_date,
        'end_date': end_date,
        'params': copy.deepcopy(params),
        'trade_at': params.get('trade_at', '종가'),
        'smart_params': smart_p,
        'allocated_capital': allocated_capital,
        'fee_rate': fee_rate,
        'integer_shares': integer_shares,
        'fast_mode': fast_mode,
    }


def build_bond_kwargs(
    data_dict, macro_df,
    params: dict,
    start_date, end_date,
    lev_params: dict = None,
    allocated_capital: float = 10000.0,
    fee_rate: float = 0.0,
    integer_shares: bool = False,
) -> dict:
    """채권 전략 엔진 호출을 위한 kwargs 딕셔너리를 구성.

    StrategyEngine.run_bond_strategy(**kwargs) 또는
    PortfolioManager.register_strategy(name, weight, kwargs, strat_type='bond') 로 직접 사용.
    """
    return {
        'data_dict': data_dict,
        'macro_df': macro_df,
        'start_date': start_date,
        'end_date': end_date,
        'params': copy.deepcopy(params) if params else None,
        'lev_params': copy.deepcopy(lev_params) if lev_params else None,
        'allocated_capital': allocated_capital,
        'fee_rate': fee_rate,
        'integer_shares': integer_shares,
    }


def resolve_rebalance_preset(preset_key: str) -> tuple:
    """프리셋 키를 받아 (std_preset, custom_cfg) 튜플 반환.

    REBALANCE_PRESET_CONFIGS에 정의된 프리셋은 custom_cfg로 전달하고,
    PortfolioManager의 기본 프리셋(none, annual 등)은 std_preset으로 전달.

    Returns:
        (std_preset: str, custom_cfg: dict | None)
    """
    _CUSTOM_PRESETS = {'hybrid', 'trend', 'band_5pct', 'band_7pct', 'band_10pct', 'asymmetric'}
    if preset_key in _CUSTOM_PRESETS:
        cfg = REBALANCE_PRESET_CONFIGS.get(preset_key, {'method': 'none'})
        return 'none', cfg
    return preset_key, None
