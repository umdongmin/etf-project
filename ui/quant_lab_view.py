import streamlit as st
import pandas as pd
import datetime
from ui.chart_view import ChartView
from ui.data_analysis_view import DataAnalysisView
from core.engine import StrategyEngine

class QuantLabView:
    @staticmethod
    def render(data_dict, fg_df, vix_df, news_df, leverage_asset, base_asset, trade_at, params, smart_params=None, start_d=None, end_d=None):
        st.header("🔬 퀀트 분석 연구실 (Quant Strategy Lab)")
        st.caption("차트를 통한 시각적 통찰력과, 정밀한 시그널 데이터를 넘나들며 전략의 원인과 결과를 심층 분석합니다.")
        
        # 1. 공통 데이터 전처리 (백테스트 실행)
        long_start = datetime.date(2010, 1, 1) if not start_d else start_d
        long_end = datetime.date.today() if not end_d else end_d

        if base_asset not in data_dict:
            st.error("데이터가 준비되지 않았습니다.")
            return

        with st.spinner('전 기간 데이터 시뮬레이션 및 분석 준비 중...'):
            cash_r = params.get('cash_ratio_pct', 0.0) / 100.0
            
            is_semi = base_asset in ['SOXX', 'USD', 'SOXL'] or leverage_asset in ['SOXX', 'USD', 'SOXL']
            bench_tickers = ['SOXX', 'USD', 'SOXL'] if is_semi else ['QQQ', 'QLD', 'TQQQ']
            bh_histories = {t: StrategyEngine.run_benchmark(data_dict, t, long_start, long_end) for t in bench_tickers}
            
            golden_history, closed_trades, signal_logs = StrategyEngine.run_golden_strategy(
                data_dict, fg_df, vix_df, news_df, leverage_asset, base_asset, 
                cash_r, long_start, long_end, params, trade_at, smart_params=smart_params
            )

        # 2. 탭 UI 구성
        tab_visual, tab_data = st.tabs(["📊 시각적 인사이트 (차트뷰)", "📝 심층 데이터 분석 (시그널 로그)"])

        # [Tab 1: 기존 인사이트 센터 기능]
        with tab_visual:
            # st.header 를 표시하지 않기 위해 flag 전달 (ChartView 내 수정 필요)
            ChartView.render(
                data_dict, fg_df, vix_df, news_df, leverage_asset, base_asset, 
                trade_at, params, smart_params=smart_params, comparison_list=st.session_state.get('comparison_list'),
                hide_header=True
            )

        # [Tab 2: 기존 데이터 분석 센터 기능]
        with tab_data:
            DataAnalysisView.render(
                signal_logs, 
                start_date=start_d, 
                end_date=end_d, 
                base_df=data_dict.get(base_asset), 
                base_name=base_asset,
                golden_history=golden_history,
                closed_trades=closed_trades,
                bh_histories=bh_histories,
                hide_header=True
            )
