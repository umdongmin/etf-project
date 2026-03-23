"""
신호 계산 서비스 (Streamlit 의존성 없는 순수 Python)

두 가지 함수를 제공:
  compute_signals(... cur_prices=None)
    - cur_prices 있음 → inject_virtual_close 후 실시간(장중) 신호
    - cur_prices 없음 → 마지막 종가 기준 신호

  get_cached_signals(...)
    - session_state 1분 캐시를 활용한 래퍼 (Streamlit UI 전용)
    - main.py, asset_view.py, portfolio_realtime_view.py 등에서 사용

main.py, asset_view.py, portfolio_realtime_view.py 모두 이 함수를 통해 신호를 계산한다.
"""

import copy
import datetime

from core.defaults import DEFAULT_BOND_PARAMS
from core.engine_kwargs_builder import build_equity_kwargs, build_bond_kwargs, resolve_rebalance_preset


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
    std_preset, custom_cfg = resolve_rebalance_preset(preset)

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
        eq_kwargs = build_equity_kwargs(
            data_dict=_data_dict, fg_df=_fg_df, vxn_df=_vxn_df, news_df=news_df,
            params=p, start_date=start_date, end_date=end_date,
            fee_rate=fee_rate, integer_shares=integer_shares, fast_mode=True,
        )
        pm.register_strategy(eq_cfg['name'], eq_cfg['weight'], eq_kwargs, strat_type='equity')

    if bond_cfg:
        bp = bond_cfg['params'] or DEFAULT_BOND_PARAMS
        bond_kwargs = build_bond_kwargs(
            data_dict=_data_dict, macro_df=macro_df,
            params=bp, start_date=start_date, end_date=end_date,
            lev_params=bond_cfg.get('lev_params'),
            fee_rate=fee_rate, integer_shares=integer_shares,
        )
        pm.register_strategy(bond_cfg['name'], bond_cfg['weight'], bond_kwargs, strat_type='bond')

    return pm.predict_portfolio_today(_data_dict, _fg_df, _vxn_df, macro_df, news_df, start_date=start_date)


def extract_signals(result, configs=None):
    """predict_portfolio_today() 결과 → {strategy_name: signal_dict} 표준 변환.

    모든 호출부(main.py, asset_view, sync_view)의 개별 구현을 대체한다.

    Parameters
    ----------
    result  : dict  — predict_portfolio_today() 반환값
    configs : list | None  — portfolio configs. 있으면 params에서 base/lev_asset 보완

    Returns
    -------
    dict[str, dict]  — {
        strategy_name: {
            'strat_type': 'equity' | 'bond',
            'stage': int,           # equity: stage(0~3), bond: lev_stage(0~2)
            'base_asset': str,      # equity: QQQ 등, bond: TLT/BIL 등 (signal 필드)
            'lev_asset': str,       # TQQQ, TMF 등. 없으면 ''
        }
    }
    """
    cfg_map = {c['name']: c for c in (configs or [])}
    signals = {}

    for name, sig in result.get('strategies', {}).items():
        if not sig:
            continue
        cfg    = cfg_map.get(name, {})
        params = cfg.get('params', {})
        stype  = sig.get('strat_type', 'equity')

        if stype == 'equity':
            signals[name] = {
                'strat_type': 'equity',
                'stage':      int(sig.get('stage', 0)),
                'base_asset': (sig.get('base_asset')
                               or params.get('base_asset', 'QQQ')),
                'lev_asset':  (sig.get('lev_asset')
                               or params.get('leverage_asset', 'TQQQ')),
            }
        else:
            # 채권: 자산명은 'signal' 필드 (TLT/BIL 등)
            signals[name] = {
                'strat_type': 'bond',
                'stage':      int(sig.get('lev_stage', 0)),
                'base_asset': (sig.get('signal')
                               or sig.get('asset')
                               or params.get('base_asset', 'TLT')),
                'lev_asset':  sig.get('lev_asset', '') or '',
            }

    return signals


def alloc_from_signals(configs, signals):
    """신호 dict → {ticker: portfolio_weight} 배분 맵.

    main.py의 _alloc_from_signals()를 공용 함수로 이관.

    Parameters
    ----------
    configs : list  — portfolio configs (name, weight 포함)
    signals : dict  — extract_signals() 결과

    Returns
    -------
    dict[str, float]  — {'TQQQ': 0.45, 'QQQ': 0.25, 'TLT': 0.30}
    """
    from core.constants import STAGE_RATIOS
    alloc = {}
    for cfg in configs:
        name   = cfg['name']
        weight = cfg.get('weight', 0)
        sig    = signals.get(name, {})
        stage  = sig.get('stage', 0)
        base   = sig.get('base_asset', 'QQQ')
        lev    = sig.get('lev_asset', '')
        br, lr = STAGE_RATIOS.get(stage, (1.0, 0.0))
        if br > 0 and base:
            alloc[base.upper()] = alloc.get(base.upper(), 0) + weight * br
        if lr > 0 and lev:
            alloc[lev.upper()]  = alloc.get(lev.upper(),  0) + weight * lr
    return alloc


def get_cached_signals(
    portfolio_name,
    data_dict,
    fg_df,
    vxn_df,
    macro_df,
    news_df,
    start_date,
    cur_prices=None,
    ttl_seconds=60,
):
    """compute_signals()의 session_state 캐시 래퍼 (Streamlit UI 전용).

    동일 포트폴리오 + 동일 모드(closing/realtime)의 결과를 ttl_seconds 동안 재사용.
    탭 이동, 위젯 상호작용 등으로 인한 불필요한 재계산을 방지한다.

    Parameters
    ----------
    ttl_seconds : int  — 캐시 유효 시간 (기본 60초)
    나머지 파라미터는 compute_signals()와 동일.
    """
    try:
        import streamlit as st
    except ImportError:
        # Streamlit 없는 환경(main.py bot)에서는 캐시 없이 직접 계산
        return compute_signals(
            portfolio_name, data_dict, fg_df, vxn_df, macro_df,
            news_df, start_date, cur_prices=cur_prices,
        )

    from core.session_keys import SK

    mode      = 'realtime' if cur_prices else 'closing'
    # start_date를 캐시 키에 포함 → 날짜 변경 시 자동 재계산
    date_str  = str(start_date) if start_date else 'all'
    cache_key = SK.sig_cache(portfolio_name, f"{mode}_{date_str}")
    time_key  = SK.sig_cache_ts(portfolio_name, f"{mode}_{date_str}")

    now = datetime.datetime.now()
    cached    = st.session_state.get(cache_key)
    cached_ts = st.session_state.get(time_key)

    if cached and cached_ts and (now - cached_ts).total_seconds() < ttl_seconds:
        return cached

    result = compute_signals(
        portfolio_name, data_dict, fg_df, vxn_df, macro_df,
        news_df, start_date, cur_prices=cur_prices,
    )
    if result:
        st.session_state[cache_key] = result
        st.session_state[time_key]  = now
    return result
