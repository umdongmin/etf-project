"""
core/defaults.py — 전체 애플리케이션의 하드코딩 기본값 단일 진실의 원천 (Single Source of Truth)

모든 뷰/서비스에서 from core.defaults import ... 로 참조.
직접 수정하지 말고, DB 저장값으로 override 하는 구조.
"""
import copy


# ── 채권 Layer1 기본 파라미터 (V7_Custom v26 골드 스탠다드) ──────────────────
DEFAULT_BOND_PARAMS = {
    'ffv_lo': -1.11, 'ffv_hi': 2.93,
    'yc_inv': -0.40, 'yc_steep': 0.13,
    'tlt_rsi_max': 70,
    'fomc_use': True, 'fomc_hi': 0.64, 'fomc_lo': -0.37,
    'f1_lookback': 15, 'f1_rise_thr': 0.27, 'f1_fall_thr': -0.27, 'f1_freeze_days': 28,
    'f2_cpi_window': 142, 'f2_accel_window': 114, 'f2_ppi_window': 135,
    'f3_emp_window': 72, 'f3_gap_hi': 0.47, 'f3_chg_thr': -0.11, 'f3_gap_lo': -0.21,
}

# ── 채권 레버리지 신호 조건 기본값 ────────────────────────────────────────────
BOND_SIG_DEFAULTS = {
    'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
    'use_sar': False, 'bb_lower': False,
    'use_willr': False, 'willr_val': -80,
    'di_plus_cross': False, 'di_minus_cross': False,
    'use_rsi_wait': False, 'rsi_wait_val': 35,
    'rsi_inc': False, 'macd_golden': False,
}

BOND_SELL_DEFAULTS = {
    'use_sar': False, 'bb_upper': False,
    'use_willr': False, 'willr_val': -20,
    'di_minus_above': False, 'di_minus_cross': False,
    'use_chandelier': False, 'chandelier_mult': 3.0,
    'use_rsi_wait': False, 'rsi_wait_val': 70,
    'macd_dead': False,
}

# ── 채권 Layer2 레버리지 기본 파라미터 ───────────────────────────────────────
DEFAULT_BOND_LEV_PARAMS = {
    'use_leverage': True,
    'buy_signals': [
        {**BOND_SIG_DEFAULTS, 'rsi_val': 35, 'rsi_cross': False,
         'macd_inc': False, 'macd_signal_below': False,
         'use_adx': True, 'adx_op': '<=', 'adx_val': 40,
         'di_plus_above': False},
        {**BOND_SIG_DEFAULTS, 'rsi_val': 0, 'rsi_cross': True,
         'macd_inc': True, 'macd_signal_below': True,
         'use_adx': True, 'adx_op': '<=', 'adx_val': 40,
         'di_plus_above': False},
    ],
    'sell_signals': [
        {**BOND_SELL_DEFAULTS, 'rsi_val': 72, 'rsi_dec': True,
         'rsi_dead': False, 'macd_dec': False, 'macd_signal_above': False},
        {**BOND_SELL_DEFAULTS, 'rsi_val': 0, 'rsi_dec': False,
         'rsi_dead': True, 'macd_dec': True, 'macd_signal_above': True},
    ],
    's3_protection': [],
    'panic_buy_signals': [
        {**BOND_SIG_DEFAULTS, 'rsi_val': 35, 'rsi_cross': False,
         'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False,
         'di_plus_above': False},
        {**BOND_SIG_DEFAULTS, 'rsi_val': 35, 'rsi_cross': False,
         'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': True, 'macd_golden': False,
         'di_plus_above': False},
        {**BOND_SIG_DEFAULTS, 'rsi_val': 30, 'rsi_cross': False,
         'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False,
         'di_plus_above': False},
    ],
    'buy_reb_up': 0.02, 'buy_reb_down': -1.0,
    'sell_reb_up': 0.05, 'sell_reb_down': -0.02,
    'use_fixed_reb': True, 'use_atr_reb': True,
    'atr_mult_buy_up': 10.0, 'atr_mult_buy_down': 2.0, 'atr_mult_sell': 7.0,
    'sell_mode': 'instant',
    'cash_ratio': 0.0, 'trade_at': '종가',
    'panic_ma': 200, 'use_panic': True,
    'use_reb_interlock': True, 'reb_interlock_macd_below': True,
    'base_asset': 'TLT', 'leverage_asset': 'TMF',
    'stage_fracs': {1: 0.30, 2: 0.70, 3: 1.0},
}

# ── 주식 전략 기본 파라미터 (TQQQ Golden Strategy) ─────────────────────────
DEFAULT_EQUITY_PARAMS = {
    'buy_signals': [
        {'rsi_val': 35, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [10, 60], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
        {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': True, 'rsi_inc': False, 'macd_inc': True, 'macd_signal_below': True, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [0, 45], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
        {'rsi_val': 52, 'use_rsi_wait': False, 'rsi_wait_val': 52, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '>=', 'adx_val': 25, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': True, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.13,
         'opt_rsi_val': False, 'rng_rsi_val': [0, 45], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
    ],
    'sell_signals': [
        {'rsi_val': 70, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': True, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': False, 'di_minus_above': False, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
        {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': True, 'rsi_dec': False, 'macd_dec': True, 'macd_signal_above': True, 'macd_dead': False, 'di_minus_above': False, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
        {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': False, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': True, 'di_minus_above': True, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
        {'rsi_val': 50, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': True, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': False, 'di_minus_above': True, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': True, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0,
         'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
         'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
    ],
    's3_protection': [
        {'use_daily_drop': False, 'drop_limit': -3.0, 'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0, 'use_vxn_jump': False, 'vxn_jump': 15.0, 'use_gap_down': True, 'gap_limit': -3.0, 'use_drop_acc': False, 'acc_limit': -7.0, 'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0},
        {'use_daily_drop': False, 'drop_limit': -3.0, 'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0, 'use_vxn_jump': False, 'vxn_jump': 15.0, 'use_gap_down': False, 'gap_limit': -3.0, 'use_drop_acc': True, 'acc_limit': -7.0, 'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0},
        {'use_daily_drop': False, 'drop_limit': -3.0, 'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0, 'use_vxn_jump': False, 'vxn_jump': 15.0, 'use_gap_down': False, 'gap_limit': -3.0, 'use_drop_acc': False, 'acc_limit': -7.0, 'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0},
    ],
    'buy_reb_up': 0.018, 'rng_buy_reb_up': [0.1, 5.0], 'buy_reb_down': -1.0, 'rng_buy_reb_down': [-15.0, -1.0],
    'sell_reb_up': 0.03, 'rng_sell_reb_up': [0.5, 10.0], 'sell_reb_down': -0.035, 'rng_sell_reb_down': [-10.0, -0.5],
    'use_reb_interlock': False, 'reb_interlock_dev': 1.13, 'reb_interlock_vxn': 22.0,
    'opt_buy_reb_up': False, 'opt_buy_reb_down': False, 'opt_sell_reb_up': False, 'opt_sell_reb_down': False,
    'base_asset': 'QQQ', 'leverage_asset': 'TQQQ', 'cash_ratio_pct': 0.0, 'trade_at': '종가',
    'use_fixed_reb': True, 'use_atr_reb': True,
    'atr_mult_buy_up': 10.0, 'rng_atr_m_buy_up': [1.0, 20.0], 'atr_mult_buy_down': 3.0, 'rng_atr_m_buy_down': [0.5, 5.0], 'atr_mult_sell': 10.0, 'rng_atr_m_sell': [1.0, 6.0],
    'opt_atr_m_buy_up': False, 'opt_atr_m_buy_down': False, 'opt_atr_m_sell': False,
    'atr_period_buy': 20, 'atr_period_sell': 20,
    'use_panic': True, 'panic_ma': 200,
    'panic_buy_signals': [
        {'rsi_val': 27, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_cross': False, 'di_minus_cross': False, 'use_sar': False, 'opt_rsi_val': False, 'rng_rsi_val': [15, 50], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50]},
        {'rsi_val': 28, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_cross': False, 'di_minus_cross': False, 'use_sar': False, 'opt_rsi_val': False, 'rng_rsi_val': [15, 50], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50]},
        {'rsi_val': 30, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_cross': False, 'di_minus_cross': False, 'use_sar': False, 'opt_rsi_val': False, 'rng_rsi_val': [15, 50], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50]},
    ],
    'use_vxn_safety': False, 'vxn_exit': 31.0,
    'use_rsi_turbo': False, 'rsi_turbo': 31.0,
    'use_sl_control': False, 'sl_control_limit': -15.0,
    'use_set_sl': False, 'set_sl_limit': -15.0,
    'dynamic_cash': {
        'use': False,
        'bull_vxn': 18.0,
        'caution_vxn': 35.0,
        'bear_vxn': 40.0,
        'ma200_buffer': 0.0,
        'cash_bull': 0.0,
        'cash_neutral': 20.0,
        'cash_caution': 30.0,
        'cash_bear': 80.0,
        'opt_bull_vxn': False, 'rng_bull_vxn': [10.0, 25.0],
        'opt_caution_vxn': False, 'rng_caution_vxn': [25.0, 40.0],
        'opt_bear_vxn': False, 'rng_bear_vxn': [35.0, 50.0],
        'opt_ma200_buffer': False, 'rng_ma200_buffer': [0.0, 5.0],
        'opt_cash_bull': False, 'rng_cash_bull': [0.0, 20.0],
        'opt_cash_neutral': False, 'rng_cash_neutral': [0.0, 50.0],
        'opt_cash_caution': False, 'rng_cash_caution': [0.0, 80.0],
        'opt_cash_bear': False, 'rng_cash_bear': [0.0, 100.0],
    },
}


# ── 리밸런싱 프리셋 설정 (PortfolioManager 및 모든 뷰에서 공유) ──────────────
REBALANCE_PRESET_CONFIGS = {
    'none':        {'method': 'none'},
    'annual':      {'method': 'calendar', 'interval_days': 252},
    'semi_annual': {'method': 'calendar', 'interval_days': 126},
    'quarterly':   {'method': 'calendar', 'interval_days': 63},
    'band_5pct':   {'method': 'threshold', 'band': 0.05, 'upper_band': 0.05, 'lower_band': 0.05, 'min_interval_days': 21},
    'band_10pct':  {'method': 'threshold', 'band': 0.10, 'upper_band': 0.10, 'lower_band': 0.10, 'min_interval_days': 21},
    'asymmetric':  {'method': 'threshold', 'upper_band': 0.15, 'lower_band': 0.05, 'min_interval_days': 21},
    'hybrid':      {'method': 'hybrid', 'interval_days': 63, 'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
    'trend':       {'method': 'trend', 'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
}

REBALANCE_PRESET_LABELS = {
    'none':        '🚫 리밸런싱 없음',
    'annual':      '📅 연간 (1년)',
    'semi_annual': '📅 반기 (6개월)',
    'quarterly':   '📅 분기 (3개월)',
    'band_5pct':   '📊 밴드 ±5%',
    'band_10pct':  '📊 밴드 ±10%',
    'asymmetric':  '⚖️ 비대칭 (상 15% / 하 5%)',
    'hybrid':      '🔀 Hybrid (분기+밴드7%)',
    'trend':       '📈 Trend-Filtered (MA200)',
}


# ── Helper: 딥카피 반환 (외부에서 원본을 직접 수정하지 못하도록 보호) ────────────
def get_default_equity_params():
    return copy.deepcopy(DEFAULT_EQUITY_PARAMS)

def get_default_bond_params():
    return copy.deepcopy(DEFAULT_BOND_PARAMS)

def get_default_bond_lev_params():
    return copy.deepcopy(DEFAULT_BOND_LEV_PARAMS)
