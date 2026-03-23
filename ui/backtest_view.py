import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.metrics import calculate_metrics
from core.engine import StrategyEngine


def _safe_float(val, default=0.0) -> float:
    """안전한 float 변환 (문자열/None/etc 모두 처리)"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


class BacktestView:
    """백테스트 결과 표시 UI 구성 클래스"""
    @staticmethod
    def render_results(golden_history, bh_histories, closed_trades, base_asset, leverage_asset, smart_params=None, is_bond=False):
        if golden_history.empty:
            st.warning("⚠️ 선택하신 기간에 대한 백테스트 결과가 없습니다. (자산 상장일 이전이거나 데이터 수집 오류)")
            return
            
        if not is_bond:
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
                    if st.button("현재가 기준 신호 계산", width='stretch', type="primary"):
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
                            <div><b>기준일:</b> {(res.get('date') or datetime.datetime.now()).strftime('%Y-%m-%d') if hasattr(res.get('date'), 'strftime') else str(res.get('date', ''))}</div>
                            <div><b>현재가:</b> ${_safe_float(res.get('price')):.2f}</div>
                            <div><b>RSI:</b> {_safe_float(res.get('rsi')):.1f}</div>
                            <div><b>MACD:</b> {_safe_float(res.get('macd')):.2f}</div>
                            <div><b>MACD Sig:</b> {_safe_float(res.get('macd_signal')):.2f}</div>
                            <div><b>MACD Hist:</b> {_safe_float(res.get('macd_hist')):.2f}</div>
                            <div><b>Stage:</b> S{preview_stage}</div>
                            <div><b>VXN:</b> {_safe_float(res.get('vxn')):.1f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        if not is_bond:
            BacktestView.render_status_summary(golden_history)
        
        # [중요] 상세 로그, 거래 내역 등은 '주식 전략 설정' 메뉴로 이동됨
        # HistoryLabView에서 BacktestView의 정적 메서드들을 호출하여 표시함

    @staticmethod
    def render_status_summary(golden_history):
        """현재 상태(매수/매도/대기) 요약 표시"""
        latest = golden_history.iloc[-1]
        st.subheader(f"📍 상태 요약 ({latest.name.date()})")
        
        sig = str(latest['Trade_Label']) if 'Trade_Label' in latest else "중립"
        if sig.startswith("매도"): sig_color = "#3498db"
        elif sig.startswith("매수"): sig_color = "#e74c3c"
        elif "진행중" in sig: sig_color = "#f39c12"
        else: sig_color = "#7f8c8d"
        
        st.markdown(f'<div style="background-color:{sig_color}; padding:20px; border-radius:10px; text-align:center; color:white;">'
                    f'<h2 style="margin:0;">{sig} 상태</h2>'
                    f'<div style="margin-top:10px; font-weight:bold;">{latest.get("Summary", "내역 없음")}</div></div>', unsafe_allow_html=True)

    @staticmethod
    def render_metrics_summary(metrics):
        """주요 성과 지표 메트릭 표시"""
        s_m = metrics.get('Strategy')
        if not s_m: return
        
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

    @staticmethod
    def render_summary_table(metrics):
        """전략 간 성과 요약 테이블"""
        st.subheader("전략 성과 요약")
        sum_df = pd.DataFrame([{"전략": k, "수익률": f"{v['Cumulative Return']:.2%}", "CAGR": f"{v['CAGR']:.2%}", "MDD": f"{v['MDD']:.2%}", "Sharpe": f"{v['Sharpe']:.2f}", "MAR": f"{v['MAR']:.2f}", "승률": f"{v['Win_Rate']:.1%}", "손익비": f"{v['Profit_Factor']:.2f}"} for k, v in metrics.items()])
        st.table(sum_df.set_index("전략"))

    @staticmethod
    def render_daily_log(golden_history, bh_histories, base_asset):
        """전체 일별 상세 로그 리스트"""
        with st.expander("📄 전체 일별 상세로그 확인", expanded=False):
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
            # 중복 제거를 위해 set/dict 사용 (순서 유지)
            lev_group_cols = list(dict.fromkeys([f'{t}_Weight' for t in ['QLD', 'TQQQ', 'TMF', 'TMV'] if f'{t}_Weight' in log_df.columns]))
            base_group_cols = list(dict.fromkeys([f'{t}_Weight' for t in [base_asset, 'TLT', 'TBF', 'BIL'] if f'{t}_Weight' in log_df.columns]))
            
            # [신규] 채권 관련 지표 컬럼 추가
            bond_cols = ['gap', 't10y2y', 'yc_mom', 'tlt_rsi', 'ff_rate', 'core_cpi']
            
            # 가치 관련 컬럼을 가장 앞쪽에 배치하여 확인 용이하도록 수정
            all_cols = [
                'Value', 'QQQ_Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'Base_Weight', 'Trade_Ret', 'Trade_Status', 'Trade_Entry_Date'
            ] + lev_group_cols + base_group_cols + [
                'RSI', 'RSI_Sig', 
                'MACD', 'Signal_Line', 'MACD_Hist', 'MACD_Sig', 
                'BB_Lower', 'BB_Upper', 'Peak_Price', 'ATR_20',
                'WILLR', 'WILLR_Sig', 
                'ADX', 'DMP', 'DMN', 'SAR', 'STOCH_K', 'STOCH_D', 'VXN'
            ] + bond_cols
            
            # 최종 컬럼 리스트에서 한 번 더 중복 제거 (KeyError 방지 핵심)
            cols = list(dict.fromkeys([c for c in all_cols if c in log_df.columns]))
            display_df = log_df[cols].sort_index(ascending=False)
            
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
                'Value': '${:,.0f}', 'QQQ_Value': '${:,.0f}', 'Trade_Ret': '{:.2f}%', 'Lev_Weight': '{:.1%}', 'Base_Weight': '{:.1%}',
                'RSI': '{:.1f}', 'MACD': '{:.2f}', 'Signal_Line': '{:.2f}', 'MACD_Hist': '{:.2f}', 
                'BB_Lower': '${:.2f}', 'BB_Upper': '${:.2f}', 'Peak_Price': '${:.2f}', 'ATR_20': '{:.2f}',
                'VXN': '{:.1f}', 'ADX': '{:.1f}', 
                'Williams': '{:.1f}', 'DI+': '{:.1f}', 'DI-': '{:.1f}', 'Parabolic': '{:.2f}',
                'STOCH_K': '{:.1f}', 'STOCH_D': '{:.1f}',
                'gap': '{:.2f}', 't10y2y': '{:.2f}', 'yc_mom': '{:.2f}', 'tlt_rsi': '{:.1f}',
                'ff_rate': '{:.1f}', 'core_cpi': '{:.1f}'
            }
            formats.update({c: '{:.1%}' for c in lev_group_cols + base_group_cols})
            
            def highlight_signals(val):
                if any(x in str(val) for x in ['골든크로스', '과매도', '양선', '이탈']):
                    return 'color: #e74c3c; font-weight: bold'
                if any(x in str(val) for x in ['데드크로스', '과매수', '음선']):
                    return 'color: #3498db; font-weight: bold'
                return ''

            # [수정] 존재하지 않는 컬럼에 대한 포맷팅/매핑 제외 (KeyError 방지)
            active_formats = {k: v for k, v in formats.items() if k in display_df.columns}
            active_subset = [c for c in ['RSI Signal', 'MACD Signal', 'Williams Signal'] if c in display_df.columns]

            styler = display_df.style.apply(style_signal, axis=1)
            if active_subset:
                styler = styler.map(highlight_signals, subset=active_subset)

            st.dataframe(
                styler.format(active_formats, na_rep='-'), 
                width='stretch', height=600
            )

    @staticmethod
    def render_closed_sets(closed_trades):
        """완료된 세트 거래 내역"""
        if closed_trades is not None and not closed_trades.empty:
            with st.expander("💰 완료된 한 세트 거래 내역 (Closed Trades Journal)", expanded=False):
                journal_df = closed_trades.copy()
                journal_df['Entry_Date'] = pd.to_datetime(journal_df['Entry_Date']).dt.date
                journal_df['Exit_Date'] = pd.to_datetime(journal_df['Exit_Date']).dt.date
                # 표시 컬럼 및 포맷 정의 (신규 필드 포함)
                j_col_order = ['Entry_Date', 'Exit_Date', 'Asset', 'Shares', 'Buy_Price', 'Sell_Price',
                               'Total_Cost', 'Total_Revenue', 'Fee_Cost', 'Net_Gain', 'Return', 'MDD', 'Status']
                j_display_cols = [c for c in j_col_order if c in journal_df.columns]
                j_formats = {
                    'Buy_Price': '${:,.2f}', 'Sell_Price': '${:,.2f}',
                    'Total_Cost': '${:,.2f}', 'Total_Revenue': '${:,.2f}',
                    'Fee_Cost': '${:,.4f}', 'Net_Gain': '${:,.2f}',
                    'Shares': '{:,.4f}', 'Return': '{:.2f}%', 'MDD': '{:.2f}%'
                }
                j_formats = {k: v for k, v in j_formats.items() if k in journal_df.columns}
                def _color_return(x):
                    if x.name in ('Return', 'Net_Gain'):
                        return ['color: #e74c3c; font-weight: bold' if v > 0 else 'color: #3498db; font-weight: bold' for v in x]
                    return ['']*len(x)
                st.dataframe(journal_df[j_display_cols].style.format(j_formats).apply(_color_return, axis=0), width='stretch', height=400)

    @staticmethod
    def render_execution_log(golden_history, base_asset, is_bond=False):
        """실시간 매매 체결 내역 (주식/채권 공용)"""
        with st.expander("📝 실시간 거래 체결 내역 (Actual Execution Log)", expanded=False):
            # Trade_Label이 '매수(' 또는 '매도('로 시작하는 행만 필터링
            exec_df = golden_history[golden_history['Trade_Label'].str.startswith(('매수(', '매도('), na=False)].copy()

            if not exec_df.empty:
                exec_df.index = exec_df.index.date

                if is_bond:
                    # ── 채권 전략 컬럼 구성 ──
                    all_weight_cols = ['TLT_Weight', 'TMF_Weight', 'TBF_Weight', 'TMV_Weight', 'BIL_Weight']
                    for col in all_weight_cols:
                        if col not in exec_df.columns:
                            exec_df[col] = 0.0

                    e_cols = ['Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight',
                              'TLT_Weight', 'TMF_Weight', 'TBF_Weight', 'TMV_Weight', 'BIL_Weight',
                              'lev_asset', 'lev_frac', 'f_state', 'tlt_rsi', 'gap', 't10y2y']

                    display_exec_df = exec_df[[c for c in e_cols if c in exec_df.columns]].sort_index(ascending=False)

                    # 컬럼명 리네임
                    display_exec_df = display_exec_df.rename(columns={
                        'lev_asset': 'Lev Asset',
                        'lev_frac': 'Lev Ratio',
                        'f_state': 'Regime',
                        'tlt_rsi': 'TLT RSI',
                        'gap': 'FFV Gap',
                        't10y2y': '10Y-2Y',
                    })

                    e_formats = {
                        'Value': '${:,.0f}',
                        'Lev_Weight': '{:.1%}',
                        'Lev Ratio': '{:.0%}',
                        'TLT RSI': '{:.1f}',
                        'FFV Gap': '{:+.2f}',
                        '10Y-2Y': '{:.2f}',
                    }
                    e_formats.update({c: '{:.1%}' for c in all_weight_cols})
                else:
                    # ── 주식 전략 컬럼 구성 (기존 로직) ──
                    all_weight_cols = ['QLD_Weight', 'TQQQ_Weight', f'{base_asset}_Weight']
                    for col in all_weight_cols:
                        if col not in exec_df.columns:
                            exec_df[col] = 0.0

                    e_cols = ['Value', 'Trade_Label', 'Asset', 'Stage', 'Lev_Weight', 'QLD_Weight', 'TQQQ_Weight',
                              f'{base_asset}_Weight',
                              'RSI', 'MACD', 'VXN', 'ADX', 'WILLR', 'DMP', 'DMN', 'SAR']

                    display_exec_df = exec_df[[c for c in e_cols if c in exec_df.columns]].sort_index(ascending=False)

                    display_exec_df = display_exec_df.rename(columns={
                        'WILLR': 'Williams',
                        'DMP': 'DI+',
                        'DMN': 'DI-',
                        'SAR': 'Parabolic'
                    })

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

                # [수정] 존재하지 않는 컬럼에 대한 포맷팅 제외 (KeyError 방지)
                active_e_formats = {k: v for k, v in e_formats.items() if k in display_exec_df.columns}

                st.dataframe(display_exec_df.style.apply(style_exec, axis=1).format(active_e_formats, na_rep='-'), width='stretch')
            else:
                st.write("해당 기간 동안 실행된 매매 내역이 없습니다.")



