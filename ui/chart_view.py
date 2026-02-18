import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.backtest_view import BacktestView
from ui.history_view import HistoryLabView
from utils.metrics import calculate_metrics

class ChartView:
    """차트 통합 분석 화면 UI 구성 클래스"""
    @staticmethod
    def render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, params, smart_params=None, comparison_list=None):
        st.header("📈 인사이트 센터 (Insight Center)")
        st.caption("내 전략의 상세 분석과 더불어, 바구니에 담긴 다른 전략들과의 다이내믹한 대조 분석을 수행합니다.")
        
        # 전 구간 데이터 확보
        long_start = datetime.date(2010, 1, 1)
        long_end = datetime.date.today()
        
        with st.spinner('전 기간 데이터 시뮬레이션 중...'):
            golden_history, all_closed_trades = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, params['cash_ratio_pct']/100.0, long_start, long_end, params, trade_at, smart_params=smart_params)
            
            semi_group = ['SOXX', 'USD', 'SOXL']
            is_semi = base_asset in semi_group or leverage_asset in semi_group
            b1, b2, b3 = ("SOXX", "USD", "SOXL") if is_semi else ("QQQ", "QLD", "TQQQ")
            bh_1 = StrategyEngine.run_benchmark(data_dict, b1, long_start, long_end)
            bh_2 = StrategyEngine.run_benchmark(data_dict, b2, long_start, long_end)
            bh_3 = StrategyEngine.run_benchmark(data_dict, b3, long_start, long_end)
            b_long_name = b1
            bh_long = bh_1
            
            # [보정] 수익률 선계산 (슬라이싱 시 첫날 누락 방지)
            bh_1['Daily_Ret'] = bh_1['Value'].pct_change()
            golden_history['Daily_Ret'] = golden_history['Value'].pct_change()

        # 공통 컬러 팔레트
        colors = ['#2979ff', '#00e676', '#ffea00', '#aa00ff', '#ff9100']

        if golden_history.empty:
            st.warning("분석할 데이터가 충분하지 않습니다.")
            return

        selected_comparisons = []
        st.divider()
        c1, c2, c3 = st.columns([2, 1, 1])
        
        with c1:
            stored_strats = StrategyStorage.list_strategies()
            loaded_name = st.session_state.get('loaded_strat_name')
            name_opts = [loaded_name] if loaded_name else []
            name_opts.extend([s for s in stored_strats if s != loaded_name])
            if not name_opts: name_opts = ["현재 분석 전략"]
            new_comp_name = st.selectbox("📝 비교군 이름", options=name_opts, key="integrated_comp_name_ui", label_visibility="collapsed")
        
        with c2:
            if st.button("➕ 현재 결과를 비교 목록에 담기", use_container_width=True):
                if any(item['name'] == new_comp_name for item in st.session_state.comparison_list):
                    st.warning(f"이미 '{new_comp_name}'이(가) 비교 목록에 있습니다.")
                else:
                    target_params = StrategyStorage.load_strategy(new_comp_name) if (new_comp_name in stored_strats and new_comp_name != loaded_name) else None
                    if target_params:
                        with st.spinner(f"'{new_comp_name}' 전략 데이터 시뮬레이션 중..."):
                            t_hist, _ = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, target_params.get('leverage_asset', leverage_asset), target_params.get('base_asset', base_asset), (target_params.get('cash_ratio_pct', 0)/100.0), long_start, long_end, target_params, target_params.get('trade_at', trade_at), smart_params=target_params)
                            t_metrics = calculate_metrics(t_hist)
                            st.session_state.comparison_list.append({'name': new_comp_name, 'base': target_params.get('base_asset', base_asset), 'lev': target_params.get('leverage_asset', leverage_asset), 'history': t_hist, 'metrics': t_metrics})
                    else:
                        m = calculate_metrics(golden_history)
                        st.session_state.comparison_list.append({'name': new_comp_name, 'base': base_asset, 'lev': leverage_asset, 'history': golden_history.copy(), 'metrics': m})
                    st.success(f"'{new_comp_name}' 추가 완료!"); st.rerun()

        with c3:
            if st.button("🗑️ 비교 목록 관리", use_container_width=True):
                st.session_state.show_comp_manager = not st.session_state.get('show_comp_manager', False)

        if comparison_list:
            selected_names = st.multiselect("🔍 대조 분석 대상 선택 (멀티 선택 가능)", options=[item['name'] for item in comparison_list], default=[])
            selected_comparisons = [item for item in comparison_list if item['name'] in selected_names]
            if st.session_state.get('show_comp_manager'):
                with st.expander("🗑️ 비교 목록 관리 (전체 목록)", expanded=True):
                    for i, item in enumerate(comparison_list):
                        col1, col2 = st.columns([4, 1])
                        col1.write(f"**{i+1}. {item['name']}** ({item['base']} / {item['lev']})")
                        if col2.button("삭제", key=f"del_integrated_{i}"):
                            st.session_state.comparison_list.pop(i); st.rerun()
                    if st.button("목록 전체 초기화", key="clear_all_integrated"):
                        st.session_state.comparison_list = []; st.rerun()
        else:
            st.info("💡 비교할 대상이 없습니다. 위에서 이름을 입력하고 [비교 목록에 담기] 버튼을 눌러보세요.")

        tab_core, tab_cycle, tab_signal = st.tabs(["🏆 전략 핵심 인사이트", "🌀 마켓 사이클", "🔬 기술적 신호 탐색"])

        with tab_core:
            # 1. 연도별 성과 비교
            actual_start = golden_history.index.min().date()
            if actual_start > long_start: st.info(f"ℹ️ 자산 상장일 관계로 분석이 **{actual_start}**부터 시작되었습니다.")
            if smart_params and smart_params.get('use_panic'):
                st.info(f"🛡️ **하락장 대응 활성**: 주가 < SMA{smart_params.get('panic_ma', 200)}일 때 매수 RSI 제한 ({smart_params.get('panic_rsi_s1', 27)}/{smart_params.get('panic_rsi_s2', 28)}/{smart_params.get('panic_rsi_s3', 30)}).")

            st.subheader(f"📊 연도별 수익률 비교 ({actual_start.year}~)")
            y_val_s = golden_history['Value'].resample('YE').last()
            y_ret_s = y_val_s.pct_change()
            if not y_val_s.empty: y_ret_s.iloc[0] = (y_val_s.iloc[0] / golden_history['Value'].iloc[0]) - 1
            
            fig_y = go.Figure()
            fig_y.add_trace(go.Bar(x=y_ret_s.index.year, y=y_ret_s*100, name='내 전략', marker_color='#ff1744'))
            y_val_b = bh_1['Value'].resample('YE').last()
            y_ret_b = y_val_b.pct_change()
            if not y_val_b.empty: y_ret_b.iloc[0] = (y_val_b.iloc[0] / bh_1['Value'].iloc[0]) - 1
            fig_y.add_trace(go.Bar(x=y_ret_b.index.year, y=y_ret_b*100, name=b1, marker_color='rgba(100, 100, 100, 0.4)'))
            
            for i, target in enumerate(selected_comparisons):
                t_hist = target['history'].copy()
                if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
                t_hist.index = pd.to_datetime(t_hist.index).normalize()
                t_val = t_hist['Value'].resample('YE').last()
                t_ret = t_val.pct_change()
                if not t_val.empty: t_ret.iloc[0] = (t_val.iloc[0] / t_hist['Value'].iloc[0]) - 1 if not t_hist.empty else 0
                fig_y.add_trace(go.Bar(x=list(t_ret.index.year), y=list(t_ret*100), name=str(target['name']), marker_color=colors[i % len(colors)]))
            
            fig_y.update_layout(barmode='group', template='plotly_white', height=400)
            st.plotly_chart(fig_y, use_container_width=True)

            HistoryLabView.render_yearly_table(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3), smart_params=smart_params, selected_comparisons=selected_comparisons)
            HistoryLabView.render_overall_summary(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3), selected_comparisons=selected_comparisons)
            
            st.divider()

            # 2. 시장 상황별 분석 (Market Regime)
            st.subheader("🛡️ 시장 상황별 성과 분석 (Market Regime)")
            regime_df = golden_history[golden_history['SMA200'] > 0].copy()
            if regime_df.empty: st.info("SMA200 데이터 부족.")
            else:
                regime_df['Is_Bull'] = regime_df['Close'] >= regime_df['SMA200']
                bh_regime = bh_1.loc[bh_1.index.isin(regime_df.index)].copy()
                bh_regime['Is_Bull'] = regime_df['Is_Bull']
                
                def get_regime_stats(df):
                    if df.empty: return {"CAGR": 0, "MDD": 0}
                    rets = df['Daily_Ret'].dropna()
                    if rets.empty: return {"CAGR": 0, "MDD": 0}
                    cagr = ((1 + rets).prod() ** (252 / len(rets))) - 1
                    dd = (df['Value'] / df['Value'].cummax() - 1).min()
                    return {"CAGR": cagr, "MDD": dd}

                # 데이터 구성 및 요약 테이블
                results = []
                for label, df_source in [("내 전략", regime_df), (b1, bh_regime)]:
                    bull_df = df_source[df_source['Is_Bull']]
                    bear_df = df_source[~df_source['Is_Bull']]
                    results.append({"name": label, "bull": get_regime_stats(bull_df), "bear": get_regime_stats(bear_df)})

                for target in selected_comparisons:
                    t_hist = target['history'].copy()
                    if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
                    t_regime = t_hist.loc[t_hist.index.isin(regime_df.index)].copy()
                    t_regime['Daily_Ret'] = t_regime['Value'].pct_change()
                    t_regime['Is_Bull'] = regime_df['Is_Bull']
                    results.append({"name": target['name'], "bull": get_regime_stats(t_regime[t_regime['Is_Bull']]), "bear": get_regime_stats(t_regime[~t_regime['Is_Bull']])})

                summary_data = []
                for r in results:
                    summary_data.append({"전략": r['name'], "상승장 CAGR": f"{r['bull']['CAGR']*100:+.1f}%", "상승장 MDD": f"{r['bull']['MDD']*100:.1f}%", "하락장 CAGR": f"{r['bear']['CAGR']*100:+.1f}%", "하락장 MDD": f"{r['bear']['MDD']*100:.1f}%"})
                st.dataframe(pd.DataFrame(summary_data).set_index("전략"), use_container_width=True)
                
                # 차트 시각화
                fig_regime = go.Figure()
                for i, r in enumerate(results):
                    color = "#00c853" if r['name'] == "내 전략" else ("#66b3ff" if r['name'] == b1 else colors[(i-2) % len(colors)])
                    fig_regime.add_trace(go.Bar(x=["상승장 CAGR", "하락장 CAGR"], y=[r['bull']['CAGR']*100, r['bear']['CAGR']*100], name=r['name'], marker_color=color, text=[f"{r['bull']['CAGR']*100:+.1f}%", f"{r['bear']['CAGR']*100:+.1f}%"], textposition="auto"))
                fig_regime.update_layout(title="시장 국면별 연평균 수익률(CAGR) 비교", yaxis_title="CAGR (%)", barmode='group', template='plotly_white', height=500)
                st.plotly_chart(fig_regime, use_container_width=True)
                st.info(f"💡 **분석 가이드**: 주가가 200일 이동평균선(SMA200) 위에 있을 때를 '상승장', 아래에 있을 때를 '하락장'으로 정의합니다. 각 전략이 하락장에서도 벤치마크({b1}) 대비 자산을 잘 방어하는지(MDD) 및 수익을 내는지(CAGR) 확인해 보세요.")

            st.divider()

            # 3. 전략 성격 분석 (Radar Chart)
            st.subheader("🎯 전략 성격 분석 (Radar Chart)")
            metrics_s = calculate_metrics(golden_history)
            fig_radar = go.Figure()
            def get_radar_data(m): return [min(100, max(0, m.get('CAGR',0)*200)), min(100, max(0, 100+m.get('MDD',0)*200)), min(100, max(0, m.get('Sharpe',0)*40)), min(100, max(0, m.get('Win_Rate',0)*100)), min(100, max(0, 100-m.get('Max_Recovery',0)*10))]
            cat = ['수익성', '안정성', '효율성', '승률', '회복력']
            fig_radar.add_trace(go.Scatterpolar(r=get_radar_data(metrics_s), theta=cat, fill='toself', name='내 전략', line_color='#ff1744'))
            for i, target in enumerate(selected_comparisons):
                fig_radar.add_trace(go.Scatterpolar(r=get_radar_data(target['metrics']), theta=cat, fill='toself', name=target['name'], line_color=colors[i % len(colors)]))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=500); st.plotly_chart(fig_radar, use_container_width=True)

        with tab_cycle:
            # 1. 월별 계절성
            HistoryLabView.render_seasonality(golden_history, bh_long, b_name=b_long_name)
            st.divider()
            # 2. 미 대선 주기 및 중간선거 분석
            col1, col2 = st.columns(2)
            with col1: HistoryLabView.render_election_cycle(golden_history, bh_long, b_name=b_long_name)
            with col2: HistoryLabView.render_midterm_analysis(golden_history, bh_long, b_name=b_long_name)

        with tab_signal:
            st.subheader("📈 지표 상관관계 분석 (Signal Lab)")
            c1, c2 = st.columns([1, 2]); sel_year = c1.selectbox("📅 분석 연도", range(datetime.date.today().year, 2009, -1), index=0)
            hist_sub = golden_history[golden_history.index.year == sel_year]
            if hist_sub.empty: st.warning(f"{sel_year}년 데이터 없음.")
            else:
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
                fig.add_trace(go.Scatter(x=hist_sub.index, y=hist_sub['Close'], name='Price', line=dict(color='#66b3ff')), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist_sub.index, y=hist_sub['VIX'], name='VIX', line=dict(color='#ff9999')), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist_sub.index, y=hist_sub['RSI'], name='RSI', line=dict(color='#9b59b6')), row=3, col=1)
                fig.update_layout(height=700, template='plotly_white', hovermode='x unified'); st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **분석 가이드**: 주가 변동과 VIX(공포지수), RSI(과매수/과매도)의 상관관계를 시계열로 분석합니다. 특정 지표가 매매 신호로서 얼마나 선행성을 가지는지 확인해 보세요.")
