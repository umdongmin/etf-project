import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataAnalysisView:
    @staticmethod
    def render(signal_logs, start_date=None, end_date=None, base_df=None, base_name="QQQ", 
               golden_history=None, closed_trades=None, bh_histories=None, hide_header=False):
        if not hide_header:
            st.header("📊 데이터 분석 센터")
        
        # 1. 시그널 선택 라디오 / 콤보박스
        signal_types = [
            "전체 보기",
            "매수신호1", 
            "매수신호2", 
            "매도신호1", 
            "매도신호2", 
            "매도신호3", 
            "매도신호4"
        ]
        
        selected_signal = st.selectbox("조회할 신호를 선택하세요:", signal_types)
        
        # [신규] '실제로 거래가 된 시그널만 표시' 체크박스 추가
        show_executed_only = st.checkbox("✅ 실제로 거래된 시그널만 표시", value=False)
        
        # 2. 데이터 프레임 변환
        if not signal_logs:
            st.info("해당 기간에 발생한 신호 로그가 없습니다.")
            return

        df_all = pd.DataFrame(signal_logs)
        
        # 날짜(Date) 필터링 - app.py 의 공통 메뉴에서 start_date, end_date 를 넘겨받음
        if start_date:
            df_all = df_all[df_all['date'].dt.date >= start_date]
        if end_date:
            df_all = df_all[df_all['date'].dt.date <= end_date]
            
        # [신규] 실행(executed) 여부 필터링 적용
        if show_executed_only and 'executed' in df_all.columns:
            df_all = df_all[df_all['executed'] == True]
            
        # [신규] 차트 데이터 준비
        chart_df = None
        if base_df is not None and not base_df.empty:
            chart_df = base_df.copy()
            if start_date:
                chart_df = chart_df[chart_df.index.date >= start_date]
            if end_date:
                chart_df = chart_df[chart_df.index.date <= end_date]
                
        if df_all.empty:
            st.info(f"설정하신 기간({start_date} ~ {end_date}) 내에 기록된 로그가 없습니다.")
            return
            
        # 3. 데이터 분리 (시그널 vs 가격변동)
        df_signals = df_all[df_all['type'] != '가격변동']
        df_reb = df_all[df_all['type'] == '가격변동']
        
        # 시그널 필터링
        if selected_signal != "전체 보기":
            df_signals = df_signals[df_signals['type'] == selected_signal]
            
        if df_signals.empty and df_reb.empty:
            st.info(f"해당 기간에 표시할 기록({selected_signal})이 없습니다.")
            return

        
        # 4. 차트 보조지표 필터 박스 추가 및 렌더링
        if chart_df is not None and not chart_df.empty:
            st.subheader(f"📈 {base_name} 가격 및 다중 패널 데이터 분석")
            
            # [신규] 보조지표 선택 체크박스 레이아웃
            st.markdown("**차트 표시 옵션:**")
            row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
            with row1_col1: show_psar = st.checkbox("PSAR (Overlaid)", value=True)
            with row1_col2: show_macd = st.checkbox("MACD", value=False)
            with row1_col3: show_rsi = st.checkbox("RSI", value=False)
            with row1_col4: show_dmi = st.checkbox("DMI", value=False)
            with row1_col5: show_willr = st.checkbox("Williams %R", value=False)
            
            show_reb_price = st.checkbox("📉 가격변동 리밸런싱 표시 (역삼각형)", value=False, help="고정 %, ATR 트리거 등으로 발생한 매매 내역을 표시합니다.")
            
            # 테이블용 데이터 합치기
            df_table = df_signals.copy()
            if show_reb_price:
                df_table = pd.concat([df_table, df_reb]).sort_values('date')
            
            if df_table.empty:
                st.info("표시할 시그널 또는 리밸런싱 내역이 없습니다.")
                return

            df_disp = df_table.rename(columns={
                'date': '발생 일자',
                'type': '신호 종류',
                'reason': '발생 사유 (상세)',
                'price': '당일 종가'
            })
            df_disp['발생 일자'] = df_disp['발생 일자'].dt.strftime('%Y-%m-%d')

            
            # 서브플롯 패널 구성
            active_subplots = []
            if show_macd: active_subplots.append("MACD")
            if show_dmi: active_subplots.append("DMI")
            if show_willr: active_subplots.append("WILLR")
            if show_rsi: active_subplots.append("RSI")
            
            num_rows = 1 + len(active_subplots)
            row_heights = [0.5] + [0.5 / len(active_subplots)] * len(active_subplots) if len(active_subplots) > 0 else [1.0]
            
            fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.05, row_heights=row_heights)
            
            # [Row 1] 기준 자산 가격 차트
            has_ohlc = all(c in chart_df.columns for c in ['Open', 'High', 'Low', 'Close'])
            if has_ohlc:
                fig.add_trace(go.Candlestick(
                    x=chart_df.index,
                    open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'],
                    name=f"{base_name} 캔들"
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df['Close'],
                    mode='lines',
                    name=f"{base_name} 종가",
                    line=dict(color='#2979ff', width=2)
                ), row=1, col=1)
            
            # PSAR Overlay
            if show_psar and 'PSAR' in chart_df.columns:
                fig.add_trace(go.Scatter(
                    x=chart_df.index, y=chart_df['PSAR'], mode='markers',
                    name='PSAR', marker=dict(color='black', size=2, symbol='circle')
                ), row=1, col=1)

            # 시그널 스캐터 차트 레이아웃
            if not df_signals.empty:
                signal_dates = pd.to_datetime(df_signals['date'])
                signal_prices = df_signals['price']
                signal_texts = df_signals['type'] + "<br>" + df_signals['reason']
                
                # 시그널 마커 색상 (기존 로직: 선택된 시그널 기반 또는 개별 판단)
                if selected_signal == "전체 보기":
                    marker_color = ['#af52de' if '매수' in t else '#00b0ff' for t in df_signals['type']]
                else:
                    marker_color = '#af52de' if '매수' in selected_signal else '#00b0ff'
                    
                fig.add_trace(go.Scatter(
                    x=signal_dates, y=signal_prices, mode='markers', name='전략 시그널',
                    marker=dict(symbol='circle', size=10, color=marker_color, line=dict(width=1, color='white')),
                    text=signal_texts, hoverinfo='text+x+y'
                ), row=1, col=1)

            # [신규] 가격변동 리밸런싱 (역삼각형)
            if show_reb_price and not df_reb.empty:
                reb_dates = pd.to_datetime(df_reb['date'])
                reb_prices = df_reb['price']
                reb_texts = "가격변동: " + df_reb['reason']
                
                # 색상 결정: 사유에 '매수'가 있으면 Violet, '매도'나 '샹들리에'가 있으면 Azure Blue
                reb_colors = []
                for r in df_reb['reason']:
                    if '매수' in r: reb_colors.append('#af52de')
                    else: reb_colors.append('#00b0ff')
                
                fig.add_trace(go.Scatter(
                    x=reb_dates, y=reb_prices, mode='markers', name='가격 리밸런싱',
                    marker=dict(symbol='triangle-down', size=12, color=reb_colors, line=dict(width=1, color='white')),
                    text=reb_texts, hoverinfo='text+x+y'
                ), row=1, col=1)

                
            # [Subplots: Row 2+]
            curr_row = 2
            
            # 1. MACD
            if show_macd and 'MACD' in chart_df.columns and 'Signal_Line' in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD'], name='MACD', line=dict(color='blue')), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['Signal_Line'], name='Signal', line=dict(color='orange')), row=curr_row, col=1)
                # MACD_Hist 계산 및 출력
                macd_h = chart_df['MACD'] - chart_df['Signal_Line']
                colors = ['green' if val >= 0 else 'red' for val in macd_h]
                fig.add_trace(go.Bar(x=chart_df.index, y=macd_h, name='MACD Hist', marker_color=colors), row=curr_row, col=1)
                curr_row += 1
                
            # 2. DMI/ADX
            if show_dmi and 'ADX' in chart_df.columns and 'DMP' in chart_df.columns and 'DMN' in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['DMP'], name='DI+', line=dict(color='blue')), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['DMN'], name='DI-', line=dict(color='red')), row=curr_row, col=1)
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['ADX'], name='ADX', line=dict(color='gold', width=2)), row=curr_row, col=1)
                curr_row += 1
                
            # 3. Williams %R
            if show_willr and 'WILLR' in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['WILLR'], name='Williams %R', line=dict(color='purple')), row=curr_row, col=1)
                fig.add_hline(y=-20, line_dash="dash", line_color="gray", annotation_text="-20", row=curr_row, col=1)
                fig.add_hline(y=-80, line_dash="dash", line_color="gray", annotation_text="-80", row=curr_row, col=1)
                curr_row += 1
                
            # 4. RSI
            if show_rsi and 'RSI' in chart_df.columns:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI'], name='RSI', line=dict(color='magenta')), row=curr_row, col=1)
                if 'RSI_SMA' in chart_df.columns:
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI_SMA'], name='RSI SMA', line=dict(color='orange', dash='dot')), row=curr_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
                curr_row += 1
            
            # y축 타이틀 업데이트
            fig.update_yaxes(title_text="Price", row=1, col=1)
            row_idx = 2
            for sp in active_subplots:
                fig.update_yaxes(title_text=sp, row=row_idx, col=1)
                row_idx += 1

            # 전체 레이아웃 옵션 설정
            # 캔들스틱 범례에서 rangeslider 제거 기능 포함
            fig.update_layout(
                xaxis_title="Date",
                hovermode='x unified',
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, traceorder="normal"),
                height=400 + (180 * len(active_subplots)), # 높이 동적 조정
                xaxis_rangeslider_visible=False
            )
            fig.update_xaxes(tickformat="%Y-%m-%d")



            
            st.plotly_chart(fig, width='stretch')
            st.divider()

        # 5. 테이블 출력 (합쳐진 데이터 사용)
        st.write(f"### 상세 리포트 (필터링된 항목: {len(df_table)}건)")
        st.dataframe(
            df_disp.set_index('발생 일자'), 
            width='stretch',
            column_config={
                "신호 종류": st.column_config.TextColumn("분류", width="small"),
                "발생 사유 (상세)": st.column_config.TextColumn("사유", width="large"),
                "당일 종가": st.column_config.NumberColumn("종가", format="$%.2f")
            }
        )


        # 5. [신규] 상세 로그 및 거래 내역 섹션
        if golden_history is not None:
            st.divider()
            st.subheader("📝 상세 분석 데이터")
            
            # 기간 필터링 적용 (복사본 사용)
            hist_df = golden_history.copy()
            if start_date:
                hist_df = hist_df[hist_df.index.date >= start_date]
            if end_date:
                hist_df = hist_df[hist_df.index.date <= end_date]
                
            # 5-1. 전체 일별 상세로그 확인
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

                log_df = hist_df.copy()
                # 벤치마크 데이터 결합 (QQQ_Value 등)
                if bh_histories and 'QQQ' in bh_histories:
                    log_df = log_df.join(bh_histories['QQQ']['Value'].rename('QQQ_Value'), how='left')
                
                log_df.index = log_df.index.date
                
                base_w_col = f'{base_name}_Weight'
                lev_group_cols = [f'{t}_Weight' for t in ['QLD', 'TQQQ'] if f'{t}_Weight' in log_df.columns]
                cols = ['Value', 'QQQ_Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'Trade_Ret', 'Trade_Status', 'Trade_Entry_Date'] + lev_group_cols + [
                    base_w_col, 'Cash_Weight', 
                    'RSI', 'RSI_Sig', 
                    'MACD', 'Signal_Line', 'MACD_Hist', 'MACD_Sig', 
                    'BB_Lower', 'BB_Upper', 'Peak_Price', 'ATR_20',
                    'WILLR', 'WILLR_Sig', 
                    'ADX', 'DMP', 'DMN', 'SAR', 'STOCH_K', 'STOCH_D', 'FG', 'VXN',
                    'News_Q'
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
                    'Value': '${:,.0f}', 'QQQ_Value': '${:,.0f}', 'Trade_Ret': '{:.2f}%', 'Lev_Weight': '{:.1%}', 'Cash_Weight': '{:.1%}', 
                    'RSI': '{:.1f}', 'MACD': '{:.2f}', 'Signal_Line': '{:.2f}', 'MACD_Hist': '{:.2f}', 
                    'BB_Lower': '${:.2f}', 'BB_Upper': '${:.2f}', 'Peak_Price': '${:.2f}', 'ATR_20': '{:.2f}',
                    'FG': '{:.0f}', 'VXN': '{:.1f}', 'STOCH_K': '{:.1f}', 'STOCH_D': '{:.1f}',
                    'ADX': '{:.1f}', 'Williams': '{:.1f}', 'DI+': '{:.1f}', 'DI-': '{:.1f}', 'Parabolic': '{:.2f}',
                    'News_Q': '{:.2f}'
                }
                formats.update({c: '{:.1%}' for c in lev_group_cols + [base_w_col]})
                st.dataframe(display_df.style.apply(style_signal, axis=1).format(formats, na_rep='-'), width='stretch', height=600)

            # 5-2. 💰 완료된 한 세트 거래 내역
            if closed_trades is not None and not closed_trades.empty:
                journal_df = closed_trades.copy()
                journal_df['Entry_Date'] = pd.to_datetime(journal_df['Entry_Date'])
                journal_df['Exit_Date'] = pd.to_datetime(journal_df['Exit_Date'])
                
                # 기간 필터링 (청산일 기준)
                if start_date:
                    journal_df = journal_df[journal_df['Exit_Date'].dt.date >= start_date]
                if end_date:
                    journal_df = journal_df[journal_df['Exit_Date'].dt.date <= end_date]
                
                if not journal_df.empty:
                    with st.expander("💰 완료된 한 세트 거래 내역 (Closed Trades Journal)", expanded=False):
                        journal_df['Entry_Date'] = journal_df['Entry_Date'].dt.date
                        journal_df['Exit_Date'] = journal_df['Exit_Date'].dt.date
                        j_formats = {'Buy_Price': '${:,.2f}', 'Sell_Price': '${:,.2f}', 'Return': '{:.2f}%', 'MDD': '{:.2f}%'}
                        st.dataframe(journal_df.style.format(j_formats).apply(lambda x: ['color: #e74c3c; font-weight: bold' if v > 0 else 'color: #3498db; font-weight: bold' for v in x] if x.name == 'Return' else ['']*len(x), axis=0), width='stretch', height=400)

            # 5-3. 📝 실시간 거래 체결 내역
            with st.expander("📝 실시간 거래 체결 내역 (Actual Execution Log)", expanded=False):
                exec_df = hist_df[hist_df['Trade_Label'].str.startswith(('매수(', '매도('), na=False)].copy()
                
                if not exec_df.empty:
                    exec_df.index = exec_df.index.date
                    all_weight_cols = ['QLD_Weight', 'TQQQ_Weight', f'{base_name}_Weight']
                    for col in all_weight_cols:
                        if col not in exec_df.columns:
                            exec_df[col] = 0.0
                    
                    e_cols = ['Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'QLD_Weight', 'TQQQ_Weight', 
                              f'{base_name}_Weight', 
                              'Cash_Weight', 'RSI', 'MACD', 'VXN']
                    
                    display_exec_df = exec_df[[c for c in e_cols if c in exec_df.columns]].sort_index(ascending=False)
                    e_formats = {'Value': '${:,.0f}', 'Lev_Weight': '{:.1%}', 'Cash_Weight': '{:.1%}', 'RSI': '{:.1f}', 'MACD': '{:.2f}', 'VXN': '{:.1f}'}
                    e_formats.update({c: '{:.1%}' for c in all_weight_cols})
                    
                    def style_exec(row):
                        sig = str(row['Trade_Label'])
                        if sig.startswith("매도"): return ['background-color: rgba(52, 152, 219, 0.1); color: #3498db; font-weight: bold'] * len(row)
                        elif sig.startswith("매수"): return ['background-color: rgba(231, 76, 60, 0.1); color: #e74c3c; font-weight: bold'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(display_exec_df.style.apply(style_exec, axis=1).format(e_formats, na_rep='-'), width='stretch')
                else:
                    st.info("해당 기간 동안 실행된 매매 내역이 없습니다.")
