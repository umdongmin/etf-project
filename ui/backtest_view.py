import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.metrics import calculate_metrics
from core.engine import StrategyEngine

class BacktestView:
    """백테스트 결과 표시 UI 구성 클래스"""
    @staticmethod
    def render_results(golden_history, bh_histories, closed_trades, base_asset, leverage_asset, smart_params=None):
        if golden_history.empty:
            st.warning("⚠️ 선택하신 기간에 대한 백테스트 결과가 없습니다. (자산 상장일 이전이거나 데이터 수집 오류)")
            return
            
        # [신규] 실시간 시그널 프리뷰 (Close Betting)
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 🛑 실시간 시그널 프리뷰 (Beta)")
                
                # [신규] 서머타임 기간 판별 (현재 2026년 기준: 3/8 ~ 11/1)
                now = datetime.datetime.now()
                dst_start = datetime.datetime(now.year, 3, 8) # 2026년 3월 둘째 일요일
                dst_end = datetime.datetime(now.year, 11, 1)  # 2026년 11월 첫째 일요일
                is_dst = dst_start <= now < dst_end
                
                dst_status = "☀️ **현재 서머타임 적용 중**" if is_dst else "❄️ **현재 서머타임 해제 중**"
                dst_color = "#e67e22" if is_dst else "#3498db"
                
                # [신규] 분석 기간 정보 (사용자 투자 시작일 반영 여부 확인용)
                analysis_start = golden_history.index.min().strftime('%Y-%m-%d')
                analysis_end = golden_history.index.max().strftime('%Y-%m-%d')
                
                st.markdown(f"""
                <div style="font-size: 13px; color: #444; background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 4px solid {dst_color}; margin-bottom: 10px;">
                    🇺🇸 <b>미국 정규장 운영 시간 (한국 기준):</b><br>
                    • <b>서머타임 (03/08 ~ 11/01):</b> 22:30 ~ 05:00<br>
                    • <b>평시/겨울 (11/02 ~ 03/07):</b> 23:30 ~ 06:00<br>
                    <div style="margin-top: 5px; color: {dst_color}; font-size: 14px;">{dst_status}</div>
                    <hr style="margin: 8px 0; border: 0.5px solid #eee;">
                    📊 <b>분석 기준 기간:</b> <span style="color:#d35400; font-weight:bold;">{analysis_start} ~ {analysis_end}</span> (데이터 최신화 완료)
                </div>
                """, unsafe_allow_html=True)
                st.caption(f"분석 시작일({analysis_start}) 기준의 상태를 바탕으로 오늘({analysis_end})의 예상 매매 신호를 도출합니다.")
            with col2:
                # 버튼 클릭 시 app.py에서 처리하도록 신호 전달
                # 버튼을 누르면 예측 프로세스가 시작됨
                if st.button("현재가 기준 신호 계산", use_container_width=True, type="primary"):
                    st.session_state.trigger_preview = True
            
            # 프리뷰 결과가 세션에 있으면 표시
            if st.session_state.get('preview_result'):
                res = st.session_state['preview_result']
                p_sig = str(res['signal'])
                current_stage = int(golden_history.iloc[-1]['Stage'])
                preview_stage = int(res.get('stage', current_stage))
                
                # 색상 결정
                if p_sig.startswith("매도"): p_bg = "rgba(52, 152, 219, 0.2)"; p_txt = "#3498db"
                elif p_sig.startswith("매수"): p_bg = "rgba(231, 76, 60, 0.2)"; p_txt = "#e74c3c"
                else: p_bg = "rgba(127, 140, 141, 0.15)"; p_txt = "#555"
                
                # 상태 변화 체크
                status_change = ""
                if current_stage != preview_stage:
                    status_change = f" <span style='color:#e67e22; font-size:14px;'>(⚠️ S{current_stage} → S{preview_stage} 변경 예상)</span>"
                else:
                    status_change = f" <span style='color:#27ae60; font-size:14px;'>(✅ S{current_stage} 상태 유지 예상)</span>"

                # 시간 체크 (실시간 여부)
                is_today = res['date'].date() == datetime.datetime.now().date()
                time_label = "🔴 실시간(장중)" if is_today else "⚪ 장마감후(마지막 종가)"
                
                st.markdown(f"""
                <div style="background-color:{p_bg}; border: 2px solid {p_txt}; padding:18px; border-radius:12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="font-weight:bold; color:{p_txt}; font-size:16px;">오늘의 예상 시그널: <span style="font-size:22px; font-weight:900;">{p_sig}</span>{status_change}</span>
                        <span style="font-size:12px; background:#eee; padding:3px 8px; border-radius:15px; color:#666;">{time_label}</span>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; font-size:13px; color:#444; border-top:1px solid rgba(0,0,0,0.1); pt:10px; margin-top:10px; padding-top:10px;">
                        <div><b>기준일:</b> {res['date'].strftime('%Y-%m-%d')}</div>
                        <div><b>현재가:</b> ${res['price']:.2f}</div>
                        <div><b>RSI:</b> {res['rsi']:.1f}</div>
                        <div><b>VXN:</b> {res['vxn']:.1f}</div>
                    </div>
                    <div style="margin-top:12px; font-size:14px; padding:8px; background:rgba(255,255,255,0.5); border-radius:6px;">
                        📝 <b>전략 요약:</b> {res['summary']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("프리뷰 로그 닫기"):
                    del st.session_state['preview_result']
                    st.rerun()

        st.divider()
        latest = golden_history.iloc[-1]
        st.subheader(f"📍 상태 요약 ({latest.name.date()})")
        
        # Trade_Label 사용 (매도 우선 체크하여 과매수 오매칭 방지)
        sig = str(latest['Trade_Label']) if 'Trade_Label' in latest else "중립"
        if sig.startswith("매도"): sig_color = "#3498db" # Blue
        elif sig.startswith("매수"): sig_color = "#e74c3c" # Red
        elif "진행중" in sig: sig_color = "#f39c12" # Orange
        else: sig_color = "#7f8c8d" # Gray (중립)
        
        st.markdown(f'<div style="background-color:{sig_color}; padding:20px; border-radius:10px; text-align:center; color:white;">'
                    f'<h2 style="margin:0;">{sig} 상태</h2>'
                    f'<div style="margin-top:10px; font-weight:bold;">{latest["Summary"]}</div></div>', unsafe_allow_html=True)
        
        # metrics 계산 시 PortfolioValue 컬럼명 정규화
        metrics = {}
        for k, v in bh_histories.items():
            temp_df = v.copy()
            if 'Value' in temp_df.columns: temp_df.rename(columns={'Value': 'PortfolioValue'}, inplace=True)
            metrics[k] = calculate_metrics(temp_df)
        
        temp_strat = golden_history.copy()
        if 'Value' in temp_strat.columns: temp_strat.rename(columns={'Value': 'PortfolioValue'}, inplace=True)
        metrics['Strategy'] = calculate_metrics(temp_strat)
        
        s_m = metrics['Strategy']
        
        # 그룹별 지표 렌더링
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown("##### 📈 수익 및 성장")
            st.metric("누적 수익률", f"{s_m['Cumulative Return']:.1%}")
            st.metric("CAGR", f"{s_m['CAGR']:.1%}")
            st.metric("최종 자산", f"${s_m['Final Value']:,.0f}")

        with m_col2:
            st.markdown("##### 🛡️ 리스크 및 방어")
            st.metric("MDD", f"{s_m['MDD']:.1%}")
            st.metric("MAR Index", f"{s_m['MAR']:.2f}")
            st.metric("최대 회복 기간", f"{s_m['Max_Recovery']}일")

        with m_col3:
            st.markdown("##### ⚡ 매매 효율성")
            st.metric("Sharpe Ratio", f"{s_m['Sharpe']:.2f}")
            st.metric("승률 (일별)", f"{s_m['Win_Rate']:.1%}")
            st.metric("손익비", f"{s_m['Profit_Factor']:.2f}")
        
        st.subheader("성과 비교 차트 (상세 분석)")
        fig = go.Figure()
        
        for k, v in bh_histories.items():
            fig.add_trace(go.Scatter(x=v.index, y=v['Value']/10000, name=k, opacity=0.5, line=dict(width=1.5)))
        
        fig.add_trace(go.Scatter(x=golden_history.index, y=golden_history['Value']/10000, name='Strategy', line=dict(color='blue', width=2.5)))
        
        v_exit = smart_params.get('vix_exit', 29) if smart_params else 29
        r_turbo = smart_params.get('rsi_turbo', 31) if smart_params else 31
        
        buy_pts = golden_history[golden_history['Trade_Label'].str.contains('매수', na=False)]
        sell_pts = golden_history[golden_history['Trade_Label'].str.contains('매도', na=False)]
        
        safety_sells = sell_pts[sell_pts['VIX'] >= v_exit] if 'VIX' in sell_pts.columns else pd.DataFrame()
        normal_sells = sell_pts.drop(safety_sells.index)
        
        turbo_buys = buy_pts[buy_pts['RSI'] <= r_turbo] if 'RSI' in buy_pts.columns else pd.DataFrame()
        normal_buys = buy_pts.drop(turbo_buys.index)

        if not normal_buys.empty:
            fig.add_trace(go.Scatter(x=normal_buys.index, y=normal_buys['Value']/10000, mode='markers', 
                                     name='Normal Buy', marker=dict(symbol='triangle-up', size=8, color='#af52de', opacity=0.8)))
        if not normal_sells.empty:
            fig.add_trace(go.Scatter(x=normal_sells.index, y=normal_sells['Value']/10000, mode='markers', 
                                     name='Normal Sell', marker=dict(symbol='triangle-down', size=8, color='#00b0ff', opacity=0.8)))
        if not turbo_buys.empty:
            fig.add_trace(go.Scatter(x=turbo_buys.index, y=turbo_buys['Value']/10000, mode='markers', 
                                     name='Turbo Buy ▲', marker=dict(symbol='triangle-up', size=14, color='#7b1fa2', line=dict(color='white', width=1))))
        if not safety_sells.empty:
            fig.add_trace(go.Scatter(x=safety_sells.index, y=safety_sells['Value']/10000, mode='markers', 
                                     name='Safety Sell ▼', marker=dict(symbol='triangle-down', size=14, color='#01579b', line=dict(color='white', width=1))))

        fig.update_layout(
            template='plotly_white', hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(l=0, r=0, t=30, b=0), height=450
        )
        fig.update_xaxes(tickformat="%Y-%m-%d")



        if smart_params:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='red', opacity=0.2), name='Safety Mode (VIX↑)', showlegend=True, legendgroup='safety'))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='blue', opacity=0.2), name='Turbo Mode (RSI↓)', showlegend=True, legendgroup='turbo'))

            def add_backtest_vrects(df, mask, color, lg):
                y_max = (df['Value'] / 10000.0).max() * 2 
                intervals, start = [], None
                for i in range(len(df)):
                    if mask.iloc[i] and start is None: start = df.index[i]
                    elif not mask.iloc[i] and start is not None: intervals.append((start, df.index[i-1])); start = None
                if start is not None: intervals.append((start, df.index[-1]))
                for s, e in intervals:
                    fig.add_trace(go.Scatter(x=[s, e, e, s], y=[0, 0, y_max, y_max], fill='toself', fillcolor=color, opacity=0.08, line_width=0, showlegend=False, legendgroup=lg, hoverinfo='skip'))

            if 'VIX' in golden_history.columns: add_backtest_vrects(golden_history, golden_history['VIX'] >= v_exit, "red", "safety")
            if 'RSI' in golden_history.columns: add_backtest_vrects(golden_history, golden_history['RSI'] <= r_turbo, "blue", "turbo")

        st.plotly_chart(fig, use_container_width=True)
        BacktestView.render_returns_heatmap(golden_history, bh_histories, leverage_asset)
        BacktestView.render_monte_carlo(golden_history)

        st.divider()
        st.subheader("전략 성과 요약")
        sum_df = pd.DataFrame([{"전략": k, "수익률": f"{v['Cumulative Return']:.2%}", "CAGR": f"{v['CAGR']:.2%}", "MDD": f"{v['MDD']:.2%}", "Sharpe": f"{v['Sharpe']:.2f}", "MAR": f"{v['MAR']:.2f}", "승률": f"{v['Win_Rate']:.1%}", "손익비": f"{v['Profit_Factor']:.2f}"} for k, v in metrics.items()])
        st.table(sum_df.set_index("전략"))
        
        st.divider()
        with st.expander("전체 일별 상세로그 확인"):
            def style_signal(row):
                sig = str(row['Trade_Label']) if 'Trade_Label' in row else ""
                status = str(row.get('Trade_Status', ""))
                if sig.startswith("매도"): return ['background-color: rgba(52, 152, 219, 0.1); color: #3498db; font-weight: bold'] * len(row)
                elif sig.startswith("매수"): return ['background-color: rgba(231, 76, 60, 0.1); color: #e74c3c; font-weight: bold'] * len(row)
                elif "익절" in status: return ['color: #e74c3c; font-weight: bold'] * len(row)
                elif "손절" in status: return ['color: #3498db; font-weight: bold'] * len(row)
                elif "진행중" in sig: return ['background-color: rgba(243, 156, 18, 0.05); color: #f39c12'] * len(row)
                return [''] * len(row)

            log_df = golden_history.copy()
            if 'QQQ' in bh_histories:
                log_df = log_df.join(bh_histories['QQQ']['Value'].rename('QQQ_Value'), how='left')
            log_df.index = log_df.index.date
            
            base_w_col = f'{base_asset}_Weight'
            lev_group_cols = [f'{t}_Weight' for t in ['QLD', 'TQQQ'] if f'{t}_Weight' in log_df.columns]
            # 가치 관련 컬럼을 가장 앞쪽에 배치하여 확인 용이하도록 수정
            cols = [
                'Value', 'QQQ_Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'Trade_Ret', 'Trade_Status', 'Trade_Entry_Date'
            ] + lev_group_cols + [
                base_w_col, 
                'RSI', 'RSI_Sig', 
                'MACD', 'Signal_Line', 'MACD_Hist', 'MACD_Sig', 
                'BB_Lower', 'BB_Upper', 'Peak_Price', 'ATR_20',
                'WILLR', 'WILLR_Sig', 
                'ADX', 'DMP', 'DMN', 'SAR', 'STOCH_K', 'STOCH_D', 'VXN'
            ]
            display_df = log_df[[c for c in cols if c in log_df.columns]].sort_index(ascending=False)
            
            # 컬럼명 리네임 (가독성 향상)
            display_df = display_df.rename(columns={
                'RSI_Sig': 'RSI Signal',
                'MACD_Sig': 'MACD Signal',
                'WILLR_Sig': 'Williams Signal',
                'WILLR': 'Williams',
                'DMP': 'DI+',
                'DMN': 'DI-',
                'SAR': 'Parabolic'
            })
            
            formats = {
                'Value': '${:,.0f}', 'QQQ_Value': '${:,.0f}', 'Trade_Ret': '{:.2f}%', 'Lev_Weight': '{:.1%}', 
                'RSI': '{:.1f}', 'MACD': '{:.2f}', 'Signal_Line': '{:.2f}', 'MACD_Hist': '{:.2f}', 
                'BB_Lower': '${:.2f}', 'BB_Upper': '${:.2f}', 'Peak_Price': '${:.2f}', 'ATR_20': '{:.2f}',
                'VXN': '{:.1f}', 'ADX': '{:.1f}', 
                'Williams': '{:.1f}', 'DI+': '{:.1f}', 'DI-': '{:.1f}', 'Parabolic': '{:.2f}',
                'STOCH_K': '{:.1f}', 'STOCH_D': '{:.1f}'
            }
            formats.update({c: '{:.1%}' for c in lev_group_cols + [base_w_col]})
            
            def highlight_signals(val):
                if any(x in str(val) for x in ['골든크로스', '과매도', '양선', '이탈']):
                    return 'color: #e74c3c; font-weight: bold'
                if any(x in str(val) for x in ['데드크로스', '과매수', '음선']):
                    return 'color: #3498db; font-weight: bold'
                return ''

            st.dataframe(
                display_df.style.apply(style_signal, axis=1)
                .applymap(highlight_signals, subset=[c for c in ['RSI Signal', 'MACD Signal', 'Williams Signal'] if c in display_df.columns])
                .format(formats, na_rep='-'), 
                use_container_width=True, height=600
            )

        if closed_trades is not None and not closed_trades.empty:
            st.divider()
            with st.expander("💰 완료된 한 세트 거래 내역 (Closed Trades Journal)", expanded=False):
                journal_df = closed_trades.copy()
                journal_df['Entry_Date'] = pd.to_datetime(journal_df['Entry_Date']).dt.date
                journal_df['Exit_Date'] = pd.to_datetime(journal_df['Exit_Date']).dt.date
                j_formats = {'Buy_Price': '${:,.2f}', 'Sell_Price': '${:,.2f}', 'Return': '{:.2f}%', 'MDD': '{:.2f}%'}
                st.dataframe(journal_df.style.format(j_formats).apply(lambda x: ['color: #e74c3c; font-weight: bold' if v > 0 else 'color: #3498db; font-weight: bold' for v in x] if x.name == 'Return' else ['']*len(x), axis=0), use_container_width=True, height=400)

        # [신규] 실시간 거래 체결 내역 (이미양식 대응 및 접기 기능 추가)
        st.divider()
        with st.expander("📝 실시간 거래 체결 내역 (Actual Execution Log)", expanded=False):
            # Trade_Label이 '매수(' 또는 '매도('로 시작하는 행만 필터링
            exec_df = golden_history[golden_history['Trade_Label'].str.startswith(('매수(', '매도('), na=False)].copy()
            
            if not exec_df.empty:
                exec_df.index = exec_df.index.date
                
                # 이미지 양식에 맞춘 고정 컬럼 구성
                # 모든 자산 비중 컬럼이 존재하도록 보장 (없으면 0.0)
                all_weight_cols = ['QLD_Weight', 'TQQQ_Weight', f'{base_asset}_Weight']
                for col in all_weight_cols:
                    if col not in exec_df.columns:
                        exec_df[col] = 0.0
                
                # 최종 출력 컬럼 순서 (이미지 양식과 동일 + Stage 추가)
                e_cols = ['Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'QLD_Weight', 'TQQQ_Weight', 
                          f'{base_asset}_Weight', 
                          'RSI', 'MACD', 'VXN', 'ADX', 'WILLR', 'DMP', 'DMN', 'SAR']
                
                display_exec_df = exec_df[[c for c in e_cols if c in exec_df.columns]].sort_index(ascending=False)
                
                # 컬럼명 리네임
                display_exec_df = display_exec_df.rename(columns={
                    'WILLR': 'Williams',
                    'DMP': 'DI+',
                    'DMN': 'DI-',
                    'SAR': 'Parabolic'
                })
                
                # 포맷팅 설정
                e_formats = {
                    'Value': '${:,.0f}', 
                    'Lev_Weight': '{:.1%}', 
                    'RSI': '{:.1f}', 
                    'MACD': '{:.2f}', 
                    'VXN': '{:.1f}',
                    'ADX': '{:.1f}',
                    'Williams': '{:.1f}',
                    'DI+': '{:.1f}',
                    'DI-': '{:.1f}',
                    'Parabolic': '{:.2f}'
                }
                e_formats.update({c: '{:.1%}' for c in all_weight_cols})
                
                def style_exec(row):
                    sig = str(row['Trade_Label'])
                    if sig.startswith("매도"): 
                        return ['background-color: rgba(52, 152, 219, 0.1); color: #3498db; font-weight: bold'] * len(row)
                    elif sig.startswith("매수"): 
                        return ['background-color: rgba(231, 76, 60, 0.1); color: #e74c3c; font-weight: bold'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(display_exec_df.style.apply(style_exec, axis=1).format(e_formats, na_rep='-'), use_container_width=True)
            else:
                st.write("해당 기간 동안 실행된 매매 내역이 없습니다.")


    @staticmethod
    def render_returns_heatmap(strategy_history, bh_histories, leverage_asset):
        """연도별/월별 수익률 히트맵 렌더링"""
        st.divider()
        st.subheader("📅 월간 수익률 히트맵 비교 (%)")
        if strategy_history.empty: return

        def get_heatmap_data(history):
            if history.empty: return None
            monthly_val = history['Value'].resample('ME').last()
            initial_series = pd.Series([10000.0], index=[monthly_val.index[0] - pd.DateOffset(months=1)])
            monthly_ret = pd.concat([initial_series, monthly_val]).pct_change().dropna() * 100
            df_ret = monthly_ret.to_frame(name='Return')
            df_ret['Year'], df_ret['Month'] = df_ret.index.year, df_ret.index.month
            pivot = df_ret.pivot(index='Year', columns='Month', values='Return').reindex(columns=range(1, 13))
            ann_ret = history['Value'].resample('YE').last().pct_change() * 100
            if not ann_ret.empty: ann_ret.iloc[0] = (history['Value'].resample('YE').last().iloc[0] / 10000.0 - 1) * 100
            pivot['Annual'] = ann_ret.values
            return pivot

        def render_plotly_heatmap(pivot, title, colorscale='RdYlGn'):
            fig = go.Figure(data=go.Heatmap(z=pivot.iloc[:, :-1].values, x=['1M','2M','3M','4M','5M','6M','7M','8M','9M','10M','11M','12M'], y=pivot.index, colorscale=colorscale, zmin=-10, zmax=10, text=pivot.iloc[:, :-1].applymap(lambda x: f"{x:.1f}%" if not np.isnan(x) else ""), texttemplate="%{text}", hoverinfo="all"))
            for i, year in enumerate(pivot.index):
                ann = pivot.iloc[i, -1]
                fig.add_annotation(x=12.2, y=year, text=f"<b>{ann:.1f}%</b>", showarrow=False, font=dict(color="blue" if ann > 0 else "red"))
            fig.update_layout(title=title, xaxis_title="Month", yaxis_title="Year", height=300 + (len(pivot)*20), template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        try:
            b_name = list(bh_histories.keys())[0] if bh_histories else "Benchmark"
            s_pivot, b_pivot = get_heatmap_data(strategy_history), get_heatmap_data(bh_histories.get(b_name, pd.DataFrame()))
            if s_pivot is not None:
                common_years = s_pivot.index.intersection(b_pivot.index) if b_pivot is not None else s_pivot.index
                render_plotly_heatmap(s_pivot.loc[common_years], f"🔥 My Strategy ({leverage_asset}) Monthly Returns (%)")
                if b_pivot is not None:
                    render_plotly_heatmap(b_pivot.loc[common_years], f"📊 {b_name} Monthly Returns (%)")
                    diff = s_pivot.loc[common_years] - b_pivot.loc[common_years]
                    diff['Annual'] = s_pivot.loc[common_years, 'Annual'] - b_pivot.loc[common_years, 'Annual']
                    render_plotly_heatmap(diff, "🆚 Alpha (Strategy - Benchmark) (%)", colorscale='PiYG')
        except: pass

    @staticmethod
    def render_monte_carlo(history):
        """몬테카를로 시뮬레이션 결과 시각화"""
        st.divider()
        st.subheader("🎲 미래 자산 시뮬레이션 (Monte Carlo)")
        sim_data = StrategyEngine.run_monte_carlo(history)
        if sim_data is None: return
            
        fig = go.Figure()
        for i in range(sim_data.shape[1]):
            fig.add_trace(go.Scatter(x=list(range(sim_data.shape[0])), y=sim_data[:, i], line=dict(color='gray', width=0.5), opacity=0.1, showlegend=False))
        median_path = np.median(sim_data, axis=1)
        fig.add_trace(go.Scatter(x=list(range(sim_data.shape[0])), y=median_path, name='Median (50th)', line=dict(color='blue', width=2.5)))
        fig.add_hline(y=sum(sim_data[0,:])/sim_data.shape[1], line_dash="dash", line_color="red", opacity=0.5, annotation_text="Current")
        fig.update_layout(title="Future Asset Value Projection (1 Year)", template='plotly_white', height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        p5, p50, p95 = np.percentile(sim_data[-1, :], 5), np.percentile(sim_data[-1, :], 50), np.percentile(sim_data[-1, :], 95)
        c1, c2, c3 = st.columns(3)
        c1.metric("상위 5% (낙관)", f"${p95:,.0f}"); c2.metric("중간값 (기대)", f"${p50:,.0f}"); c3.metric("하위 5% (비관)", f"${p5:,.0f}")
        last_val = history['Value'].iloc[-1]
        st.info(f"💡 분석 결과: 향후 1년 뒤 자산이 현재(${last_val:,.0f})보다 높을 확률은 약 **{(sim_data[-1, :] > last_val).mean():.1%}** 입니다.")
