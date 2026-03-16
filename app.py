import streamlit as st

# [중요] Streamlit 설정은 모든 출력 및 다른 라이브러리 실행보다 먼저 호출되어야 함
st.set_page_config(page_title="ETF Golden Strategy", page_icon="📈", layout="wide")

import pandas as pd
import numpy as np
import datetime
import os
import time

# 커스텀 모듈 임포트
from core.data import DataService
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.backtest_view import BacktestView
from ui.history_view import HistoryLabView
from ui.quant_lab_view import QuantLabView
from ui.intelligence_view import IntelligenceView
from ui.portfolio_view import PortfolioView
from ui.portfolio_realtime_view import PortfolioRealtimeView
from ui.tester_view import TesterView
from utils.metrics import calculate_metrics

class GoldenStrategyApp:
    def __init__(self):
        self.ticker_map = {
            "QQQ": "나스닥 100 (QQQ)",
            "QLD": "나스닥 2배 (QLD)",
            "TQQQ": "나스닥 3배 (TQQQ)"
        }
        
    def run(self):
        # 사이드바 공통 메뉴
        st.sidebar.title("🚀 Golden Strategy v3.0")
        menu = st.sidebar.radio("메인 메뉴", [
            "🛰️ 실시간 모니터링",
            "📈 포트폴리오 매니저",
            "🔬 퀀트 분석 연구실",
            "📜 주식 전략 설정",
            "📉 채권 전략 설정",
            "🧠 AI 뉴스 분석 리포트"
        ])
        
        
        # 세션 상태 초기화
        if 'trigger_recalc' not in st.session_state:
            st.session_state.trigger_recalc = False
            
        if 'current_params' not in st.session_state:
            st.session_state.current_params = {
                'buy_signals': [
                    {'rsi_val': 35, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 
                     'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
                     'opt_rsi_val': False, 'rng_rsi_val': [10, 60], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': True, 'rsi_inc': False, 'macd_inc': True, 'macd_signal_below': True, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 
                     'use_dev200': False, 'dev200_op': '>=', 'dev200_val': 1.0,
                     'opt_rsi_val': False, 'rng_rsi_val': [0, 45], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0], 'opt_dev200_val': False, 'rng_dev200_val': [0.8, 1.3]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 
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
                    {
                        'use_daily_drop': False, 'drop_limit': -3.0, 
                        'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0,
                        'use_vxn_jump': False, 'vxn_jump': 15.0, 
                        'use_gap_down': True, 'gap_limit': -3.0,
                        'use_drop_acc': False, 'acc_limit': -7.0,
                        'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0
                    },
                    {
                        'use_daily_drop': False, 'drop_limit': -3.0, 
                        'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0,
                        'use_vxn_jump': False, 'vxn_jump': 15.0, 
                        'use_gap_down': False, 'gap_limit': -3.0,
                        'use_drop_acc': True, 'acc_limit': -7.0,
                        'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0
                    },
                    {
                        'use_daily_drop': False, 'drop_limit': -3.0, 
                        'use_ma60': False, 'ma60_limit': 0.0, 'use_ma200': False, 'ma200_limit': 0.0,
                        'use_vxn_jump': False, 'vxn_jump': 15.0, 
                        'use_gap_down': False, 'gap_limit': -3.0,
                        'use_drop_acc': False, 'acc_limit': -7.0,
                        'use_exit_all': False, 'use_chandelier': False, 'chandelier_mult': 3.0
                    }
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
                    {'rsi_val': 30, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_cross': False, 'di_minus_cross': False, 'use_sar': False, 'opt_rsi_val': False, 'rng_rsi_val': [15, 50], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50]}
                ],
                'use_vxn_safety': False, 'vxn_exit': 31.0,
                'use_rsi_turbo': False, 'rsi_turbo': 31.0,
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
                    'opt_cash_bear': False, 'rng_cash_bear': [0.0, 100.0]
                }
            }

        # [신규] 채권 전략 초기화 로직 (Bond_V7_Custom 기본 적용)
        # [Definitive Fix] 채권 전략 초기화 로직 (Bond_V7_Custom v26 골드 스탠다드 적용)
        if ('current_bond_params' not in st.session_state
                or 'current_bond_lev_params' not in st.session_state
                or st.session_state.get('initial_bond_load_v28', False) is False):
            # 골드 스탠다드 (최적 수익률 구현용 풀스펙)
            GOLDEN_BOND_PARAMS = {
                'ffv_lo': -1.0, 'ffv_hi': 1.5, 'yc_inv': -0.3, 'yc_steep': 0.4, 'tlt_rsi_max': 60
            }
            _sig_defaults = {
                'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
                'use_sar': False, 'bb_lower': False,
                'use_willr': False, 'willr_val': -80,
                'di_plus_cross': False, 'di_minus_cross': False,
                'use_rsi_wait': False, 'rsi_wait_val': 35,
                'rsi_inc': False, 'macd_golden': False,
            }
            _sell_defaults = {
                'use_sar': False, 'bb_upper': False,
                'use_willr': False, 'willr_val': -20,
                'di_minus_above': False, 'di_minus_cross': False,
                'use_chandelier': False, 'chandelier_mult': 3.0,
                'use_rsi_wait': False, 'rsi_wait_val': 70,
                'macd_dead': False,
            }
            GOLDEN_LEV_PARAMS = {
                'use_leverage': True,
                'buy_signals': [
                    {**_sig_defaults, 'rsi_val': 35, 'rsi_cross': False,
                     'macd_inc': False, 'macd_signal_below': False},
                    {**_sig_defaults, 'rsi_val': 0, 'rsi_cross': True,
                     'macd_inc': True, 'macd_signal_below': True},
                ],
                'sell_signals': [
                    {**_sell_defaults, 'rsi_val': 70, 'rsi_dec': True,
                     'rsi_dead': False, 'macd_dec': False, 'macd_signal_above': False},
                    {**_sell_defaults, 'rsi_val': 0, 'rsi_dec': False,
                     'rsi_dead': True, 'macd_dec': True, 'macd_signal_above': True},
                ],
                's3_protection': [], 'panic_buy_signals': [],
                'buy_reb_up': 0.01, 'buy_reb_down': -0.02,
                'sell_reb_up': 0.02, 'sell_reb_down': -0.02,
                'use_fixed_reb': True, 'use_atr_reb': False,
                'cash_ratio': 0.0, 'trade_at': '종가',
                'panic_ma': 200, 'use_panic': False,
                'base_asset': 'TLT', 'leverage_asset': 'TMF',
            }

            # 1. DB 로드 시도
            custom_strat = StrategyStorage.load_strategy("Bond_V7_Custom")
            
            if custom_strat and isinstance(custom_strat, dict) and 'ffv_lo' in custom_strat:
                # DB에 값이 있으면 기본값은 가져오되, 레버리지 시그널이 부실하면 골든 스탠다드로 병합
                st.session_state.current_bond_params = {k: custom_strat.get(k, v) for k, v in GOLDEN_BOND_PARAMS.items()}
                
                db_lev = custom_strat.get('lev_params') or custom_strat
                if not db_lev.get('buy_signals'):
                    # 레버리지 시그널이 없으면 골든 스탠다드 강제 적용
                    st.session_state.current_bond_lev_params = GOLDEN_LEV_PARAMS
                else:
                    st.session_state.current_bond_lev_params = db_lev
            else:
                # 2. DB에 없거나 비정상이면 완전 초기화
                st.session_state.current_bond_params = GOLDEN_BOND_PARAMS
                st.session_state.current_bond_lev_params = GOLDEN_LEV_PARAMS
            
            # 3. UI 위젯 동기화
            TesterView.sync_bond_params_to_widgets(
                st.session_state.current_bond_params, 
                lev_params=st.session_state.current_bond_lev_params,
                key_prefix="bond_"
            )
            
            # 4. 상태 저장 및 강제 재실행 (모든 레이어 동기화 보장)
            st.session_state.trigger_recalc_bond = True
            st.session_state.initial_bond_load_v28 = True
            st.rerun()
        
        if 'comparison_list' not in st.session_state:
            st.session_state.comparison_list = []
            
        cp = st.session_state.current_params
        
        # 데이터 로딩 (st.cache_data 활용, TTL 1시간)
        with st.spinner('실시간 데이터를 가져오는 중...'):
            data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time = DataService.load_all_data()
        
        if not data_dict:
            st.error("데이터 로드 실패")
            st.stop()
            
        # [수정] 날짜 선택 및 파라미터 설정을 로직 최상단으로 이동 (UnboundLocalError 방지)
        start_d, end_d = None, None
        if 'QQQ' in data_dict:
            years = list(range(datetime.datetime.now().year, 2009, -1))
            min_d, max_d = data_dict['QQQ'].index.min().date(), data_dict['QQQ'].index.max().date()
            current_year_jan1 = datetime.date(datetime.datetime.now().year, 1, 1)

            if 'start_input' not in st.session_state:
                st.session_state.start_input = current_year_jan1 if min_d <= current_year_jan1 else min_d
            if 'end_input' not in st.session_state:
                st.session_state.end_input = datetime.date.today()

            def on_year_change():
                if st.session_state.q_year != "직접 선택":
                    yr = int(st.session_state.q_year)
                    st.session_state.start_input = datetime.date(yr, 1, 1)
                    st.session_state.end_input = datetime.date(yr, 12, 31)

            st.sidebar.divider()
            st.sidebar.subheader("💸 거래 수수료 설정")
            use_fee = st.sidebar.toggle("수수료 적용", value=st.session_state.get('global_use_fee', True), key='global_use_fee')
            if use_fee:
                fee_pct = st.sidebar.number_input("수수료율 (%)", min_value=0.0, max_value=1.0,
                    value=st.session_state.get('global_fee_pct', 0.10), step=0.01,
                    format="%.3f", help="0.10% = 한국 증권사 해외 ETF 기준 (수수료 0.07% + 슬리피지)", key='global_fee_pct')
                st.session_state.global_fee_rate = fee_pct / 100.0
            else:
                st.session_state.global_fee_rate = 0.0

            st.sidebar.subheader("📦 주식 수량 설정")
            st.sidebar.toggle("정수 주식 수량 적용", value=st.session_state.get('global_integer_shares', True), key='global_integer_shares',
                help="활성화 시 소수점 주식 없이 정수 단위로만 매매 (잔여금은 현금으로 보존)")

            st.sidebar.divider()
            st.sidebar.subheader("📅 분석 기간 설정")
            st.sidebar.selectbox(
                "연도 선택 (1/1 ~ 12/31 자동설정)", 
                ["직접 선택"] + years,
                key="q_year",
                on_change=on_year_change
            )
            
            start_d = st.sidebar.date_input("시작일", key="start_input")
            end_d = st.sidebar.date_input("종료일", key="end_input")

        # [수정] 세션 파라미터를 기반으로 로직 파라미터 구성
        smart_params = {
            'use_panic': cp['use_panic'], 
            'panic_ma': cp['panic_ma'],
            'panic_buy_signals': cp.get('panic_buy_signals', []),
            'panic_rsi_s1': cp.get('panic_rsi_s1', 27),
            'panic_rsi_s2': cp.get('panic_rsi_s2', 28),
            'panic_rsi_s3': cp.get('panic_rsi_s3', 30),
            'use_vxn_safety': cp['use_vxn_safety'],
            'vxn_exit': cp['vxn_exit'],
            'use_rsi_turbo': cp['use_rsi_turbo'],
            'rsi_turbo': cp['rsi_turbo'],
            'use_sl_control': cp.get('use_sl_control', False),
            'sl_control_limit': cp.get('sl_control_limit', -15.0),
            # [신규] 개별 거래 세트 기준 손절 라인
            'use_set_sl': cp.get('use_set_sl', False),
            'set_sl_limit': cp.get('set_sl_limit', -15.0),
            # [신규] 동적 현금 비중 제어
            'dynamic_cash': cp.get('dynamic_cash', {})
        }
        cash_ratio = cp['cash_ratio_pct'] / 100.0

        # [신규] 실시간 시그널 프리뷰 처리 인터럽트
        if st.session_state.get('trigger_preview'):
            with st.spinner('실시간 신호를 계산 중입니다... (장마감 전 종가베팅 지원)'):
                # 1. 현재가 및 VIX/VXN 수집
                all_tickers = list(data_dict.keys()) + ['^VIX', '^VXN']
                cur_prices = DataService.fetch_current_prices(all_tickers)
                
                if cur_prices:
                    # 2. 가상 종가 주입 및 지표 재계산 (VIX/VXN 포함)
                    try:
                        v_dict, v_vxn_df, v_fg_df = DataService.inject_virtual_close(data_dict, cur_prices, vxn_df, fg_df)
                    except Exception as ve:
                        st.error(f"데이터 주입 오류: {ve}")
                        v_dict = None

                    if v_dict:
                        try:
                            # 3. 오늘자 신호 예측
                            pred = StrategyEngine.predict_today_signal(
                                v_dict, v_fg_df, v_vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], 
                                cash_ratio, start_d, end_d, cp, cp['trade_at'], smart_params
                            )
                        except Exception as e:
                            pred = None
                            st.error(f"예측 엔진 오류: {e}")
                    else:
                        pred = None
                    
                    if pred:
                        st.session_state.preview_result = pred
                        st.session_state.trigger_preview = False
                        st.toast("✅ 실시간 신호 계산 완료!", icon="🚀")
                    else:
                        st.error("신호 예측에 실패했습니다. (결과 데이터 없음 - 엔진 내부 필터링 확인 필요)")
                else:
                    st.error("실시간 가격을 전혀 가져올 수 없습니다. 네트워크 상태를 확인해 주세요.")
                
            st.session_state.trigger_preview = False
                
        # 메뉴별 렌더링
        if menu == "🛰️ 실시간 모니터링":
            st.title("🛰️ 실시간 포트폴리오 모니터링")
            PortfolioView.render_realtime_tab(data_dict, fg_df, vxn_df, macro_df, news_df,
                                              start_d or data_dict['QQQ'].index.min().date(),
                                              datetime.date.today())
        elif menu == "🔬 퀀트 분석 연구실":
            QuantLabView.render(
                data_dict, fg_df, vxn_df, news_df, 
                cp['leverage_asset'], cp['base_asset'], cp['trade_at'], 
                cp, smart_params=smart_params, start_d=start_d, end_d=end_d
            )
        elif menu == "📈 포트폴리오 매니저":
            PortfolioView.render(data_dict, fg_df, vxn_df, macro_df, news_df, start_d, end_d)
        elif menu == "📜 주식 전략 설정":
            HistoryLabView.render(data_dict, fg_df, vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], cp['trade_at'], cp, smart_params=smart_params, salt="v2.3 (CrashLogs)")
        elif menu == "📉 채권 전략 설정":
            st.title("📉 Bond Strategy Lab (V7)")
            
            # [신규] 채권 전용 기간 프리셋 추가 (📜 주식 전략 설정과 동일)
            start_b, end_b = TesterView.render_period_selector(key_suffix="bond")
            st.divider()

            # [신규] 탭 시스템 적용 (분석 결과 & 상세 로그 + 전략 설정)
            b_tab1, b_tab2 = st.tabs(["📊 분석 결과 & 상세 로그", "⚙️ 채권 전략 설정"])
            
            with b_tab2:
                # [신규] 채권 전략 파일 관리 (메매 전략 설정과 동일 구조)
                st.subheader("💾 채권 전략 파일 관리")
                col_b1, col_b2 = st.columns(2)
                
                with col_b1:
                    with st.form("save_bond_strat_form"):
                        st.write("📝 현재 채권 전략 저장")
                        new_bond_name = st.text_input("새 전략 이름", value="Bond_V7_Custom")
                        if st.form_submit_button("💾 신규 저장", use_container_width=True):
                            # [개선] UI 위젯의 현재 상태를 즉시 읽어서 통합 저장
                            p, lp = TesterView.get_bond_params_from_widgets(key_prefix="bond_")
                            bundle = {
                                'params': p,
                                'lev_params': lp
                            }
                            StrategyStorage.save_strategy(new_bond_name, bundle, category='bond')
                            st.success(f"'{new_bond_name}' 저장 완료 (UI 현재값 반영)")
                            st.rerun()

                with col_b2:
                    bond_list = StrategyStorage.list_strategies(category='bond')
                    if bond_list:
                        with st.form("load_bond_strat_form"):
                            st.write("📂 저장된 채권 전략 불러오기")
                            selected_bond = st.selectbox("전략 선택", bond_list)
                            if st.form_submit_button("✨ 적용하기", use_container_width=True):
                                loaded = StrategyStorage.load_strategy(selected_bond)
                                if loaded:
                                    # [개선] 번들 데이터 해제 및 세션 반영
                                    if 'params' in loaded and 'lev_params' in loaded:
                                        st.session_state.current_bond_params = loaded['params']
                                        st.session_state.current_bond_lev_params = loaded['lev_params']
                                    else:
                                        # 하위 호환성 (구형 데이터)
                                        st.session_state.current_bond_params = loaded
                                        st.session_state.current_bond_lev_params = None
                                    
                                    # [신규] UI 위젯 키 강제 동기화 (Apply 즉시 반영)
                                    TesterView.sync_bond_params_to_widgets(
                                        st.session_state.current_bond_params,
                                        lev_params=st.session_state.get('current_bond_lev_params'),
                                        key_prefix="bond_"
                                    )
                                    
                                    st.session_state.trigger_recalc_bond = True
                                    st.success(f"'{selected_bond}' 로드 및 UI 동기화 완료")
                                    st.rerun()
                    else:
                        st.info("저장된 채권 전략이 없습니다.")

                st.divider()
                st.subheader("⚖️ 채권 전략 상세 설정")
                with st.form("bond_settings_form"):
                    # 기존 설정 UI 호출
                    new_bond_params, new_lev_params = TesterView.render_bond_settings(
                        st.session_state.current_bond_params,
                        current_lev_params=st.session_state.get('current_bond_lev_params'),
                    )
                    st.write("---")
                    bond_submitted = st.form_submit_button("🚀 설정값 확정 및 재시뮬레이션 실행", use_container_width=True, type="primary")

                    if bond_submitted:
                        st.session_state.current_bond_params = new_bond_params
                        st.session_state.current_bond_lev_params = new_lev_params
                        st.session_state.trigger_recalc_bond = True
                        st.rerun()
            
            with b_tab1:
                if macro_df.empty:
                    st.error("⚠️ 매크로 데이터(FRED)를 불러오지 못했습니다. API 키 설정을 확인하거나 잠시 후 다시 시도해 주세요.")
                    st.info("데이터가 없으면 채권 전략 분석을 수행할 수 없습니다.")
                
                # [수정] 성과 변경 감지 및 자동 계산 로직 (trigger_recalc_bond 또는 해시 변경 시)
                # [수정] 레버리지(lev_params) 설정 변경도 감지 대상에 추가
                current_bond_hash = hash(
                    str(start_b) + str(end_b) +
                    str(st.session_state.current_bond_params) +
                    str(st.session_state.get('current_bond_lev_params')) +
                    str(len(macro_df)) +
                    str(st.session_state.get('global_fee_rate', 0.0)) +
                    str(st.session_state.get('global_integer_shares', False))
                )
                should_calc_bond = st.session_state.get('trigger_recalc_bond', False) or (st.session_state.get('last_bond_hash') != current_bond_hash)
                

                if should_calc_bond:
                    with st.spinner('채권 분석 데이터 업데이트 중...'):
                        # 채권 전략 실행 (V7)
                        bond_bench = {'TLT': StrategyEngine.run_benchmark(data_dict, 'TLT', start_b, end_b)}
                        bond_history, b_closed, _ = StrategyEngine.run_bond_strategy(
                            data_dict, macro_df, start_b, end_b,
                            params=st.session_state.current_bond_params,
                            lev_params=st.session_state.get('current_bond_lev_params'),
                            fee_rate=st.session_state.get('global_fee_rate', 0.0),
                            integer_shares=st.session_state.get('global_integer_shares', False)
                        )
                        
                        # [수정] 부분 업데이트
                        curr_res = st.session_state.get('last_bt_result', {})
                        curr_res['bond'] = {'history': bond_history, 'closed_trades': b_closed, 'benchmarks': bond_bench}
                        st.session_state.last_bt_result = curr_res
                        st.session_state.last_bond_hash = current_bond_hash
                        st.session_state.trigger_recalc_bond = False # 완료 후 해제


                res = st.session_state.get('last_bt_result', {}).get('bond')
                if res and res['history'] is not None and not res['history'].empty:
                    # [신규] 주식 전략 설정과 동일한 요약 정보 추가
                    bh_bond = res['benchmarks'].get('TLT', pd.DataFrame())
                    empty_df = pd.DataFrame()
                    
                    st.subheader("📊 전체 기간 요약")
                    HistoryLabView.render_overall_summary(res['history'], bh_bond, empty_df, empty_df, b_names=("TLT", "", ""))
                    
                    st.divider()
                    HistoryLabView.render_yearly_table(res['history'], bh_bond, empty_df, empty_df, b_names=("TLT", "", ""))
 
                    # [신규 추가] 채권 전략 상세 분석 로그 (주식 전략과 동일한 구성)
                    BacktestView.render_closed_sets(res['closed_trades'])
                    BacktestView.render_execution_log(res['history'], 'TLT')
                    
                    st.divider()
                    BacktestView.render_daily_log(res['history'], res['benchmarks'], 'TLT')

                    st.divider()
                    # 상세 분석 결과 렌더링 (차트 및 로그) - 채권 전용 필터 적용
                    BacktestView.render_results(res['history'], res['benchmarks'], res['closed_trades'], 'TLT', 'TLT', smart_params=None, is_bond=True)
                else:
                    st.info("시뮬레이션 데이터를 준비 중입니다... (파라미터를 변경해 보세요)")
                    # [신규] 진단 정보 표시 (결과 미출력 원인 파악용)
                    with st.expander("🔍 분석 진단 정보 (결과가 나오지 않을 경우 확인)", expanded=False):
                        st.write(f"- 분석 기간: {start_b} ~ {end_b}")
                        st.write(f"- 매크로 데이터 수: {len(macro_df)} 행")
                        if not macro_df.empty:
                            st.write("- 매크로 데이터 최신 날짜:", macro_df.index.max())
                            st.write("- 주요 지표 결측치: ", macro_df.isna().sum().to_dict())
                        tlt_len = len(data_dict.get('TLT', []))
                        st.write(f"- TLT 데이터 수: {tlt_len} 행")
                        if tlt_len == 0:
                            st.warning("⚠️ TLT 데이터 없음 — yfinance Rate Limit 가능성. 아래 버튼으로 캐시를 초기화하세요.")
                            if st.button("🔄 데이터 캐시 초기화 후 재시도", key="clear_cache_bond"):
                                DataService.load_all_data.clear()
                                st.rerun()
                        st.write(f"- 최적화 파라미터: {st.session_state.current_bond_params}")

        elif menu == "🧠 AI 뉴스 분석 리포트":
            IntelligenceView.render()

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
