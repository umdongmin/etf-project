import streamlit as st

class TesterView:
    """전략 분석기 입력 UI 구성 클래스"""
    @staticmethod
    def render_config(current_params, ticker_map=None):
        st.subheader("🌐 기본 자산 및 거래 설정")
        c1, c2, c3 = st.columns(3)
        
        base_asset_opts = ["QQQ", "QLD", "TQQQ", "SOXX", "USD", "SOXL"]
        leverage_asset_opts = ["QLD", "TQQQ", "USD", "SOXL", "TMF"]
        trade_at_opts = ["종가", "익일 시가"]
        
        def fmt(x): return ticker_map[x] if ticker_map and x in ticker_map else x

        base_asset = c1.selectbox("기준 자산 (시그널)", base_asset_opts, index=base_asset_opts.index(current_params.get('base_asset', 'QQQ')), format_func=fmt, key="conf_base")
        leverage_asset = c2.selectbox("매매 대상 자산", leverage_asset_opts, index=leverage_asset_opts.index(current_params.get('leverage_asset', 'QLD')), format_func=fmt, key="conf_lev")
        trade_at = c3.radio("매매 시점", trade_at_opts, index=trade_at_opts.index(current_params.get('trade_at', '종가')), horizontal=True, key="conf_trade_at")
        
        st.divider()
        col_buy, col_sell = st.columns(2)
        buy_signals = []
        with col_buy:
            st.subheader("📥 Buy Conditions (OR)")
            for i in range(1, 3):
                with st.expander(f"매수 신호 {i}", expanded=False):
                    p = current_params['buy_signals'][i-1] if i <= len(current_params['buy_signals']) else {}
                    if i == 1: # B2: RSI 과매도
                        buy_signals.append({
                            'rsi_val': st.number_input(f"RSI 매수 기준 {i}", 0, 100, p.get('rsi_val', 35), key=f"b_rsi_{i}"),
                            'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', False), key=f"b_cross_{i}"),
                            'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}"),
                            'macd_inc': st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', False), key=f"b_m_inc_{i}"),
                            'macd_signal_below': st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', False), key=f"b_m_below_{i}"),
                            'macd_golden': st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}"),
                            'bb_lower': st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        })
                    else: # B1: RSI 골든크로스 + MACD 상승
                        buy_signals.append({
                            'rsi_val': st.number_input(f"RSI 매수 기준 {i}", 0, 100, p.get('rsi_val', 0), key=f"b_rsi_{i}"),
                            'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', True), key=f"b_cross_{i}"),
                            'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}"),
                            'macd_inc': st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', True), key=f"b_m_inc_{i}"),
                            'macd_signal_below': st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', True), key=f"b_m_below_{i}"),
                            'macd_golden': st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}"),
                            'bb_lower': st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        })
        
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            for i in range(1, 4):
                with st.expander(f"매도 신호 {i}", expanded=False):
                    p = current_params['sell_signals'][i-1] if i <= len(current_params['sell_signals']) else {}
                    if i == 1: # S1: RSI 꺾임 (70 이상)
                        sell_signals.append({
                            'rsi_val': st.number_input(f"RSI 매도 기준 {i}", 0, 100, p.get('rsi_val', 70), key=f"s_rsi_{i}"),
                            'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', False), key=f"s_dead_{i}"),
                            'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', True), key=f"s_dec_{i}"),
                            'macd_dec': st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', False), key=f"s_m_dec_{i}"),
                            'macd_signal_above': st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', False), key=f"s_m_above_{i}"),
                            'macd_dead': st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', False), key=f"s_m_dead_{i}"),
                            'bb_upper': st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=f"s_bb_up_{i}")
                        })
                    elif i == 2: # S2: RSI 데드크로스 + MACD 하락
                        sell_signals.append({
                            'rsi_val': st.number_input(f"RSI 매도 기준 {i}", 0, 100, p.get('rsi_val', 0), key=f"s_rsi_{i}"),
                            'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', True), key=f"s_dead_{i}"),
                            'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', False), key=f"s_dec_{i}"),
                            'macd_dec': st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', True), key=f"s_m_dec_{i}"),
                            'macd_signal_above': st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', True), key=f"s_m_above_{i}"),
                            'macd_dead': st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', False), key=f"s_m_dead_{i}"),
                            'bb_upper': st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=f"s_bb_up_{i}")
                        })
                    else: # S5: MACD 데드크로스
                        sell_signals.append({
                            'rsi_val': st.number_input(f"RSI 매도 기준 {i}", 0, 100, p.get('rsi_val', 0), key=f"s_rsi_{i}"),
                            'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', False), key=f"s_dead_{i}"),
                            'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', False), key=f"s_dec_{i}"),
                            'macd_dec': st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', False), key=f"s_m_dec_{i}"),
                            'macd_signal_above': st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', False), key=f"s_m_above_{i}"),
                            'macd_dead': st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', True), key=f"s_m_dead_{i}"),
                            'bb_upper': st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=f"s_bb_up_{i}")
                        })
        
        st.divider()
        st.subheader("⚖️ 리밸런싱 및 스마트 조건")
        c1, c2, c3 = st.columns(3)
        reb_params = {
            'buy_reb_up': c1.number_input("매수 후 리밸런싱(↑) %", 0.0, 100.0, current_params.get('buy_reb_up', 0.03) * 100, 0.1, format="%.1f", key="r_b_u") / 100.0,
            'buy_reb_down': c1.number_input("매수 후 리밸런싱(↓) %", -100.0, 0.0, current_params.get('buy_reb_down', -0.03) * 100, 0.1, format="%.1f", key="r_b_d") / 100.0,
            'sell_reb_up': c2.number_input("매도 후 리밸런싱(↑) %", 0.0, 100.0, current_params.get('sell_reb_up', 0.03) * 100, 0.1, format="%.1f", key="r_s_u") / 100.0,
            'sell_reb_down': c2.number_input("매도 후 리밸런싱(↓) %", -100.0, 0.0, current_params.get('sell_reb_down', -0.03) * 100, 0.1, format="%.1f", key="r_s_d") / 100.0,
            'cash_ratio': c3.number_input("현금 유지 비율(%)", 0.0, 100.0, current_params.get('cash_ratio', 0.0) * 100, 1.0, key="conf_cash") / 100.0
        }
        
        with st.expander("🛡️ 하락장 및 스마트 대응 설정 (Smart Filter)"):
            c1, c2 = st.columns(2)
            smart_config = {
                'use_panic': c1.toggle("하락장 매수 지연 (Panic Mode)", value=current_params.get('use_panic', True)),
                'panic_ma': c1.selectbox("지연 기준 이평선", [60, 120, 200], index=[60,120,200].index(current_params.get('panic_ma', 200))),
                'use_vix_safety': c2.toggle("VIX 대피 시스템 (Safety)", value=current_params.get('use_vix_safety', True)),
                'vix_exit': c2.slider("대피 VIX 레벨", 15, 45, current_params.get('vix_exit', 28)),
                'use_rsi_turbo': c2.toggle("RSI 과매도 가속 (Turbo)", value=current_params.get('use_rsi_turbo', True)),
                'rsi_turbo': c2.slider("가속 RSI 레벨", 20, 45, current_params.get('rsi_turbo', 31))
            }
            st.write("**Panic Mode 상세 (매수 시 RSI 제한)**")
            sc1, sc2, sc3 = st.columns(3)
            smart_config.update({
                'panic_rsi_s1': sc1.number_input("S1 제한 RSI", 20, 45, current_params.get('panic_rsi_s1', 32)),
                'panic_rsi_s2': sc2.number_input("S2 제한 RSI", 20, 45, current_params.get('panic_rsi_s2', 32)),
                'panic_rsi_s3': sc3.number_input("S3 제한 RSI", 20, 45, current_params.get('panic_rsi_s3', 30))
            })
            
        return {
            'buy_signals': buy_signals, 'sell_signals': sell_signals,
            'base_asset': base_asset, 'leverage_asset': leverage_asset, 'trade_at': trade_at,
            **reb_params, **smart_config
        }
