import streamlit as st

# [중요] Streamlit 설정은 모든 출력 및 다른 라이브러리 실행보다 먼저 호출되어야 함
st.set_page_config(page_title="ETF Golden Strategy", page_icon="📈", layout="wide")

import pandas as pd
import numpy as np
import datetime
import os

# 커스텀 모듈 임포트
from core.data import DataService
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.market_view import MarketView
from ui.backtest_view import BacktestView
from ui.history_view import HistoryLabView
from ui.chart_view import ChartView
from ui.tester_view import TesterView
from utils.metrics import calculate_metrics

class GoldenStrategyApp:
    def __init__(self):
        self.ticker_map = {
            "QQQ": "나스닥 100 (QQQ)",
            "QLD": "나스닥 2배 (QLD)",
            "TQQQ": "나스닥 3배 (TQQQ)",
            "SOXX": "반도체 1배 (SOXX)",
            "USD": "반도체 2배 (USD)",
            "SOXL": "반도체 3배 (SOXL)",
            "TMF": "채권 3배 (TMF)",
            "TLT": "채권 1배 (TLT)"
        }
        
    def run(self):
        # 사이드바 공통 메뉴
        st.sidebar.title("🚀 Golden Strategy v3.0")
        menu = st.sidebar.radio("메인 메뉴", ["📊 실시간 백테스트", "📈 인사이트 센터", "📜 매매 전략 설정", "📰 주요 마켓 이슈"])
        
        # 세션 상태 초기화
        if 'current_params' not in st.session_state:
            st.session_state.current_params = {
                'buy_signals': [
                    {'rsi_val': 35, 'rsi_cross': False, 'rsi_inc': False, 'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False, 'bb_lower': False},
                    {'rsi_val': 0, 'rsi_cross': True, 'rsi_inc': False, 'macd_inc': True, 'macd_signal_below': True, 'macd_golden': False, 'bb_lower': False}
                ],
                'sell_signals': [
                    {'rsi_val': 70, 'rsi_dead': False, 'rsi_dec': True, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': False, 'bb_upper': False},
                    {'rsi_val': 0, 'rsi_dead': True, 'rsi_dec': False, 'macd_dec': True, 'macd_signal_above': True, 'macd_dead': False, 'bb_upper': False},
                    {'rsi_val': 0, 'rsi_dead': False, 'rsi_dec': False, 'macd_dec': False, 'macd_signal_above': False, 'macd_dead': True, 'bb_upper': False}
                ],
                'buy_reb_up': 0.02, 'buy_reb_down': -0.05, 'sell_reb_up': 0.03, 'sell_reb_down': -0.035,
                'base_asset': 'QQQ', 'leverage_asset': 'QLD', 'cash_ratio_pct': 0, 'trade_at': '종가',
                'use_panic': True, 'panic_ma': 200, 
                'panic_rsi_s1': 27, 'panic_rsi_s2': 28, 'panic_rsi_s3': 30,
                'use_vix_safety': True, 'vix_exit': 31,
                'use_rsi_turbo': True, 'rsi_turbo': 31
            }
        
        if 'comparison_list' not in st.session_state:
            st.session_state.comparison_list = []
            
        cp = st.session_state.current_params
        
        # 데이터 로딩
        if 'global_data' not in st.session_state:
            with st.spinner('실시간 데이터를 가져오는 중...'):
                st.session_state.global_data = DataService.load_all_data()
        
        data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time = st.session_state.global_data
        
        if not data_dict:
            st.error("데이터 로드 실패")
            st.stop()
            
        # [복구] 사이드바 핵심 설정 (사용자 편의성 제고)
        st.sidebar.header("📍 핵심 자산 설정")
        base_asset_opts = ["QQQ", "QLD", "TQQQ", "SOXX", "USD", "SOXL"]
        cp['base_asset'] = st.sidebar.selectbox("기준 자산 (시그널)", base_asset_opts, index=base_asset_opts.index(cp['base_asset']), format_func=lambda x: self.ticker_map[x])
        
        leverage_asset_opts = ["QLD", "TQQQ", "USD", "SOXL", "TMF"]
        cp['leverage_asset'] = st.sidebar.selectbox("매매 대상 자산", leverage_asset_opts, index=leverage_asset_opts.index(cp['leverage_asset']), format_func=lambda x: self.ticker_map[x])
        
        cp['cash_ratio_pct'] = st.sidebar.slider("현금 비중 (%)", 0, 50, value=cp['cash_ratio_pct'], step=5)
        
        trade_at_opts = ["종가", "익일 시가"]
        cp['trade_at'] = st.sidebar.radio("매매 시점 선택", trade_at_opts, index=trade_at_opts.index(cp['trade_at']), horizontal=True)

        # [수정] 세션 파라미터를 기반으로 로직 파라미터 구성
        smart_params = {
            'use_panic': cp['use_panic'], 
            'panic_ma': cp['panic_ma'],
            'panic_rsi_s1': cp['panic_rsi_s1'],
            'panic_rsi_s2': cp['panic_rsi_s2'],
            'panic_rsi_s3': cp['panic_rsi_s3'],
            'use_vix_safety': cp['use_vix_safety'],
            'vix_exit': cp['vix_exit'],
            'use_rsi_turbo': cp['use_rsi_turbo'],
            'rsi_turbo': cp['rsi_turbo']
        }
        cash_ratio = cp['cash_ratio_pct'] / 100.0

        # 메뉴별 렌더링
        if menu == "📊 실시간 백테스트":
            if 'QQQ' in data_dict:
                min_d, max_d = data_dict['QQQ'].index.min().date(), data_dict['QQQ'].index.max().date()
                start_d = st.sidebar.date_input("시작일", value=datetime.date(2026, 1, 1) if min_d <= datetime.date(2026, 1, 1) else min_d)
                end_d = st.sidebar.date_input("종료일", value=max_d)
                
                with st.spinner('백테스팅 계산 중...'):
                    is_semi = cp['base_asset'] in ['SOXX', 'USD', 'SOXL'] or cp['leverage_asset'] in ['SOXX', 'USD', 'SOXL']
                    bench_tickers = ['SOXX', 'USD', 'SOXL'] if is_semi else ['QQQ', 'QLD', 'TQQQ']
                    bh_histories = {t: StrategyEngine.run_benchmark(data_dict, t, start_d, end_d) for t in bench_tickers}
                    golden_history, closed_trades = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, cp['leverage_asset'], cp['base_asset'], cash_ratio, start_d, end_d, cp, cp['trade_at'], smart_params=smart_params)
                    
                BacktestView.render_results(golden_history, bh_histories, closed_trades, cp['base_asset'], cp['leverage_asset'], smart_params=smart_params)
            else:
                st.error("데이터 로드 실패")
        elif menu == "📈 인사이트 센터":
            ChartView.render(data_dict, fg_df, vix_df, cp['leverage_asset'], cp['base_asset'], cp['trade_at'], cp, smart_params=smart_params, comparison_list=st.session_state.comparison_list)
        elif menu == "📜 매매 전략 설정":
            HistoryLabView.render(data_dict, fg_df, vix_df, cp['leverage_asset'], cp['base_asset'], cp['trade_at'], cp, smart_params=smart_params)
        elif menu == "📰 주요 마켓 이슈":
            MarketView.render(data_dict, macro_df, fetch_status, fetch_time, fg_df)

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
