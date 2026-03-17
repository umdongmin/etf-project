"""
신호 계산 서비스 (Streamlit 의존성 없는 순수 Python)

두 가지 함수를 제공:
  compute_signals(... cur_prices=None)
    - cur_prices 있음 → inject_virtual_close 후 실시간(장중) 신호
    - cur_prices 없음 → 마지막 종가 기준 신호

main.py, asset_view.py, portfolio_realtime_view.py 모두 이 함수를 통해 신호를 계산한다.
"""

import copy
import datetime


def compute_signals(
    portfolio_name,
    data_dict,
    fg_df,
    vxn_df,
    macro_df,
    news_df,
    start_date,
    cur_prices=None,
    fee_rate=0.001,
    integer_shares=True,
):
    """포트폴리오 신호 계산.

    Parameters
    ----------
    portfolio_name : str
    data_dict, fg_df, vxn_df, macro_df, news_df : 데이터
    start_date : date  — 백테스트 시작일
    cur_prices : dict | None
        None  → 마지막 종가 기준 신호 (closing)
        dict  → inject_virtual_close 후 실시간 신호 (realtime)
    fee_rate : float
    integer_shares : bool

    Returns
    -------
    dict | None  — predict_portfolio_today() 결과
    """
    from core.data import DataService
    from core.storage import StrategyStorage
    from core.portfolio_manager import PortfolioManager

    portfolio_data = StrategyStorage.load_portfolio(portfolio_name)
    if not portfolio_data:
        return None

    configs     = portfolio_data.get('configs', [])
    eq_cfg      = next((c for c in configs if c['type'] == 'equity'), None)
    bond_cfg    = next((c for c in configs if c['type'] == 'bond'), None)
    preset      = portfolio_data.get('rebalance_preset', 'hybrid')

    # ── inject_virtual_close (실시간 전용) ──────────────────────────────
    _data_dict = data_dict
    _vxn_df    = vxn_df
    _fg_df     = fg_df
    if cur_prices:
        try:
            _data_dict, _vxn_df, _fg_df = DataService.inject_virtual_close(
                data_dict, cur_prices, vxn_df, fg_df
            )
        except Exception as e:
            print(f"[signal_service] inject_virtual_close 오류: {e}")

    # ── PortfolioManager 빌드 ────────────────────────────────────────────
    _PRESET_MAP = {
        'hybrid':     {'method': 'hybrid',    'interval_days': 63, 'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
        'trend':      {'method': 'trend',     'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
        'band_5pct':  {'method': 'threshold', 'upper_band': 0.05, 'lower_band': 0.05, 'min_interval_days': 21},
        'band_10pct': {'method': 'threshold', 'upper_band': 0.10, 'lower_band': 0.10, 'min_interval_days': 21},
        'asymmetric': {'method': 'threshold', 'upper_band': 0.15, 'lower_band': 0.05, 'min_interval_days': 21},
    }
    custom_cfg = _PRESET_MAP.get(preset)
    std_preset = preset if preset not in _PRESET_MAP else 'none'

    trend_data = None
    if preset == 'trend' and 'QQQ' in _data_dict:
        try:
            qqq_close  = _data_dict['QQQ']['close']
            trend_data = qqq_close > qqq_close.rolling(200).mean()
        except Exception:
            pass

    trend_def = (
        {eq_cfg['name']: 0.50, bond_cfg['name']: 0.50}
        if preset == 'trend' and eq_cfg and bond_cfg
        else None
    )

    pm = PortfolioManager(
        total_capital=10000.0,
        rebalance_preset=std_preset,
        custom_preset_cfg=custom_cfg,
        fee_rate=fee_rate,
        integer_shares=integer_shares,
        trend_data=trend_data,
        trend_defense_weights=trend_def,
    )

    end_date = datetime.date.today()

    if eq_cfg:
        p = eq_cfg['params']
        smart_p = {
            'use_panic':         p.get('use_panic', True),
            'panic_ma':          p.get('panic_ma', 200),
            'panic_buy_signals': p.get('panic_buy_signals', []),
            'panic_rsi_s1':      p.get('panic_rsi_s1', 27),
            'panic_rsi_s2':      p.get('panic_rsi_s2', 28),
            'panic_rsi_s3':      p.get('panic_rsi_s3', 30),
            'use_vxn_safety':    p.get('use_vxn_safety', False),
            'vxn_exit':          p.get('vxn_exit', 31.0),
            'use_rsi_turbo':     p.get('use_rsi_turbo', False),
            'rsi_turbo':         p.get('rsi_turbo', 31.0),
            'use_sl_control':    p.get('use_sl_control', False),
            'sl_control_limit':  p.get('sl_control_limit', -15.0),
            'use_set_sl':        p.get('use_set_sl', False),
            'set_sl_limit':      p.get('set_sl_limit', -15.0),
            'dynamic_cash':      p.get('dynamic_cash', {'use': False}),
        }
        pm.register_strategy(eq_cfg['name'], eq_cfg['weight'], dict(
            data_dict=_data_dict, fg_df=_fg_df, vxn_df=_vxn_df, news_df=news_df,
            leverage_asset=p.get('leverage_asset', 'TQQQ'),
            base_asset=p.get('base_asset', 'QQQ'),
            cash_ratio=p.get('cash_ratio_pct', 0.0),
            start_date=start_date, end_date=end_date,
            params=copy.deepcopy(p),
            trade_at=p.get('trade_at', '종가'),
            smart_params=smart_p, fast_mode=True,
        ), strat_type='equity')

    if bond_cfg:
        bp = bond_cfg['params'] or {
            'ffv_lo': -1.0, 'ffv_hi': 1.5,
            'yc_inv': -0.3, 'yc_steep': 0.40, 'tlt_rsi_max': 60,
        }
        pm.register_strategy(bond_cfg['name'], bond_cfg['weight'], dict(
            data_dict=_data_dict, macro_df=macro_df,
            start_date=start_date, end_date=end_date,
            params=copy.deepcopy(bp),
            lev_params=bond_cfg.get('lev_params'),
        ), strat_type='bond')

    return pm.predict_portfolio_today(_data_dict, _fg_df, _vxn_df, macro_df, news_df, start_date=start_date)
