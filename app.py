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
from ui.rolling_view import RollingLabView
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
            "📊 실시간 백테스트", 
            "🔬 퀀트 분석 연구실", 
            "🧬 롤링 윈도우 분석",
            "📜 매매 전략 설정", 
            "🧠 AI 마켓 인텔리전스"
        ])
        
        # [신규] 전역 백그라운드 최적화 모니터링
        if 'background_opt' in st.session_state and st.session_state.background_opt:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🔥 AI 최적화 탐색 중")
            state = st.session_state.background_opt
            
            if not state['done']:
                progress = min(state['current'] / st.session_state.opt_n_iter, 1.0) if 'opt_n_iter' in st.session_state else 0
                st.sidebar.progress(progress)
                
                # 하트비트 체크 (5초 이상 갱신 없으면 경고)
                last_hb = state.get('heartbeat', time.time())
                gap = int(time.time() - last_hb)
                hb_color = "green" if gap < 10 else "orange" if gap < 30 else "red"
                st.sidebar.markdown(f"⏱️ 마지막 활동: :{hb_color}[{gap}초 전]")
                st.sidebar.caption(f"진행: {state['current']}회 완료 | 최고: {state['best_v']:.2f}")
                
                col1, col2 = st.sidebar.columns(2)
                if col1.button("🔄 화면 갱신", key="refresh_bg_opt"):
                    st.rerun()
                if col2.button("🛑 탐색 중단", key="stop_background_opt"):
                    state['stop_requested'] = True
                    st.sidebar.warning("중단 요청 중... (현재 Trial 완료 후 중단)")
                    st.rerun()
                
                if st.session_state.get('opt_n_jobs', 1) > 2:
                    st.sidebar.warning("⚠️ 많은 코어 사용 중: UI가 느려질 수 있습니다.")
            else:
                if state['error']:
                    st.sidebar.error(f"❌ 오류 발생: {state['error']}")
                    if st.sidebar.button("확인", key="clear_bg_error"):
                        st.session_state.background_opt = None
                        st.rerun()
                else:
                    if 'opt_notified' not in st.session_state or not st.session_state.opt_notified:
                        st.sidebar.success("✅ 탐색 완료!")
                        st.toast("🎉 최적화 탐색이 완료되었습니다!", icon="📊")
                        st.session_state.opt_notified = True
                    
                    # 결과 병합
                    st.session_state.opt_results = state['results']
                    st.session_state.opt_baseline = {
                        'score': state['base_info'][0], 
                        'cagr': state['base_info'][1], 
                        'mdd': state['base_info'][2]
                    }
                    
                    st.sidebar.info("✨ 탐색이 완료되었습니다. 결과를 확인하세요.")
                    col1, col2 = st.sidebar.columns(2)
                    if col1.button("결과 보기", key="show_opt_result"):
                        st.session_state.background_opt = None
                        st.session_state.opt_notified = False
                        st.rerun()
                    if col2.button("닫기", key="close_bg_opt"):
                        st.session_state.background_opt = None
                        st.session_state.opt_notified = False
                        st.rerun()
        
        # 세션 상태 초기화
        if 'current_params' not in st.session_state:
            st.session_state.current_params = {
                'buy_signals': [
                    {'rsi_val': 35, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 'opt_rsi_val': False, 'rng_rsi_val': [10, 60], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': True, 'rsi_inc': False, 'macd_inc': True, 'macd_signal_below': True, 'macd_golden': False, 'bb_lower': False, 'use_adx': True, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 'opt_rsi_val': False, 'rng_rsi_val': [0, 45], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40, 'use_willr': False, 'willr_val': -80, 'di_plus_above': False, 'di_plus_cross': False, 'use_sar': False, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '>', 'news_q_val': 0.8, 'use_news_z': False, 'news_z_op': '>', 'news_z_val': 1.0, 'opt_rsi_val': False, 'rng_rsi_val': [0, 45], 'opt_adx_val': False, 'rng_adx_val': [5, 80], 'opt_willr_val': False, 'rng_willr_val': [-95, -50], 'opt_news_q': False, 'rng_news_q_val': [0.5, 0.99], 'opt_news_z': False, 'rng_news_z_val': [-1.0, 4.0]}
                ],
                'sell_signals': [
                    {'rsi_val': 70, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': True, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': False, 'di_minus_above': False, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0, 'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': True, 'rsi_dec': False, 'macd_dec': True, 'macd_signal_above': True, 'macd_dead': False, 'di_minus_above': False, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0, 'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0]},
                    {'rsi_val': 0, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': False, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': True, 'di_minus_above': True, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': False, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0, 'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0]},
                    {'rsi_val': 50, 'use_rsi_wait': False, 'rsi_wait_val': 70, 'rsi_dead': False, 'rsi_dec': True, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': False, 'di_minus_above': True, 'bb_upper': False, 'use_chandelier': False, 'chandelier_mult': 3.0, 'use_sar': True, 'use_willr': False, 'willr_val': -20, 'di_minus_cross': False, 'use_news_q': False, 'news_q_op': '<', 'news_q_val': 0.2, 'use_news_z': False, 'news_z_op': '<', 'news_z_val': -1.0, 'opt_rsi_val': False, 'rng_rsi_val': [40, 95], 'opt_chandelier_mult': False, 'rng_chandelier_mult': [1.0, 5.0], 'opt_willr_val': False, 'rng_willr_val': [-50, -5], 'opt_news_q': False, 'rng_news_q_val': [0.01, 0.5], 'opt_news_z': False, 'rng_news_z_val': [-4.0, 1.0]}
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
                'use_rsi_turbo': False, 'rsi_turbo': 31.0
            }
        
        if 'comparison_list' not in st.session_state:
            st.session_state.comparison_list = []
            
        cp = st.session_state.current_params
        
        # 데이터 로딩
        if 'global_data' not in st.session_state:
            with st.spinner('실시간 데이터를 가져오는 중...'):
                st.session_state.global_data = DataService.load_all_data()
        
        data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time = st.session_state.global_data
        
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
                st.session_state.end_input = max_d

            def on_year_change():
                if st.session_state.q_year != "직접 선택":
                    yr = int(st.session_state.q_year)
                    st.session_state.start_input = datetime.date(yr, 1, 1)
                    st.session_state.end_input = datetime.date(yr, 12, 31)

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
            'set_sl_limit': cp.get('set_sl_limit', -15.0)
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
        if menu == "📊 실시간 백테스트":
            if 'QQQ' in data_dict:
                with st.spinner('백테스팅 계산 중...'):
                    bench_tickers = ['QQQ', 'QLD', 'TQQQ']
                    bh_histories = {t: StrategyEngine.run_benchmark(data_dict, t, start_d, end_d) for t in bench_tickers}
                    golden_history, closed_trades, _ = StrategyEngine.run_golden_strategy(data_dict, fg_df, vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], cash_ratio, start_d, end_d, cp, cp['trade_at'], smart_params=smart_params)
                    
                BacktestView.render_results(golden_history, bh_histories, closed_trades, cp['base_asset'], cp['leverage_asset'], smart_params=smart_params)
            else:
                st.error("데이터 로드 실패")
        elif menu == "🔬 퀀트 분석 연구실":
            QuantLabView.render(
                data_dict, fg_df, vxn_df, news_df, 
                cp['leverage_asset'], cp['base_asset'], cp['trade_at'], 
                cp, smart_params=smart_params, start_d=start_d, end_d=end_d
            )
        elif menu == "🧬 롤링 윈도우 분석":
            RollingLabView.render(
                data_dict, fg_df, vxn_df, news_df,
                cp['leverage_asset'], cp['base_asset'], cp['trade_at'],
                cp, smart_params=smart_params
            )
        elif menu == "📜 매매 전략 설정":
            HistoryLabView.render(data_dict, fg_df, vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], cp['trade_at'], cp, smart_params=smart_params, salt="v2.3 (CrashLogs)")
        elif menu == "🧠 AI 마켓 인텔리전스":
            IntelligenceView.render()

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
