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
                            'rsi_val': st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 35), key=f"b_rsi_{i}"),
                            'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', False), key=f"b_cross_{i}"),
                            'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}"),
                            'macd_inc': st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', False), key=f"b_m_inc_{i}"),
                            'macd_signal_below': st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', False), key=f"b_m_below_{i}"),
                            'macd_golden': st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}"),
                            'use_adx': st.checkbox(f"ADX 필터 사용 {i}", value=p.get('use_adx', False), key=f"b_use_adx_{i}"),
                            'adx_op': st.selectbox(f"ADX 연산자 {i}", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=f"b_adx_op_{i}"),
                            'adx_val': st.number_input(f"ADX 기준 {i}", 0, 100, p.get('adx_val', 40), key=f"b_adx_{i}"),
                            'bb_lower': st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        })
                    else: # B1: RSI 골든크로스 + MACD 상승
                        buy_signals.append({
                            'rsi_val': st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 0), key=f"b_rsi_{i}"),
                            'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', True), key=f"b_cross_{i}"),
                            'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}"),
                            'macd_inc': st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', True), key=f"b_m_inc_{i}"),
                            'macd_signal_below': st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', True), key=f"b_m_below_{i}"),
                            'macd_golden': st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}"),
                            'use_adx': st.checkbox(f"ADX 필터 사용 {i}", value=p.get('use_adx', False), key=f"b_use_adx_{i}"),
                            'adx_op': st.selectbox(f"ADX 연산자 {i}", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=f"b_adx_op_{i}"),
                            'adx_val': st.number_input(f"ADX 기준 {i}", 0, 100, p.get('adx_val', 40), key=f"b_adx_{i}"),
                            'bb_lower': st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        })
        
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            for i in range(1, 4):
                with st.expander(f"매도 신호 {i}", expanded=False):
                    p = current_params['sell_signals'][i-1] if i <= len(current_params['sell_signals']) else {}
                    sell_signals.append({
                        'rsi_val': st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 70 if i==1 else 0), key=f"s_rsi_{i}"),
                        'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', i==2), key=f"s_dead_{i}"),
                        'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', i==1), key=f"s_dec_{i}"),
                        'macd_dec': st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', i==2), key=f"s_m_dec_{i}"),
                        'macd_signal_above': st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', i==2), key=f"s_m_above_{i}"),
                        'macd_dead': st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', i==3), key=f"s_m_dead_{i}"),
                        'di_minus_above': st.checkbox(f"매도 우세 (DI->DI+) {i}", value=p.get('di_minus_above', False), key=f"s_di_{i}"),
                        'bb_upper': st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=f"s_bb_up_{i}")
                    })
            
            st.write("---")
            st.subheader("🛡️ S3(레버리지 100%) 보호 설정")
            st.caption("두 신호 중 하나만이라도 충족되면(OR) 방어 매도가 작동합니다.")
            
            s3_p_list = current_params.get('s3_protection', [{}, {}])
            while len(s3_p_list) < 2: s3_p_list.append({})
            
            s3_protection = []
            for i in range(1, 4):
                with st.expander(f"보호 신호 {i}", expanded=False):
                    p = s3_p_list[i-1] if i <= len(s3_p_list) else {}
                    use_drop = st.checkbox(f"일일 급락 매도 제어 {i}", value=p.get('use_daily_drop', False), key=f"s_drop_use_{i}")
                    drop_lim = st.number_input(f"급락 기준 (%) {i}", -10.0, 0.0, p.get('drop_limit', -3.0), 0.1, format="%.1f", key=f"s_drop_lim_{i}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        use_ma60 = st.checkbox(f"60일 이평선 하회 {i}", value=p.get('use_ma60', False), key=f"s_ma60_use_{i}")
                        ma60_lim = st.number_input(f"60일 기준 (%) {i}", -20.0, 5.0, p.get('ma60_limit', 0.0), 0.1, format="%.1f", key=f"s_ma60_lim_{i}")
                    with col2:
                        use_ma200 = st.checkbox(f"200일 이평선 하회 {i}", value=p.get('use_ma200', False), key=f"s_ma200_use_{i}")
                        ma200_lim = st.number_input(f"200일 기준 (%) {i}", -20.0, 5.0, p.get('ma200_limit', 0.0), 0.1, format="%.1f", key=f"s_ma200_lim_{i}")
                    
                    use_vj = st.checkbox(f"VIX 급등 시그널 사용 {i}", value=p.get('use_vix_jump', False), key=f"s_vj_use_{i}")
                    vj_val = st.number_input(f"VIX 급등 기준 (%) {i}", 0.0, 100.0, p.get('vix_jump', 15.0), 0.1, format="%.1f", key=f"s_vj_{i}")
                    use_gap = st.checkbox(f"패닉 시가 갭 발생 {i} (-3% 이하)", value=p.get('use_gap_down', False), key=f"s_gap_use_{i}")
                    gap_lim = st.number_input(f"시가 갭 기준 (%) {i}", -10.0, 0.0, p.get('gap_limit', -3.0), 0.1, format="%.1f", key=f"s_gap_lim_{i}")
                    use_acc = st.checkbox(f"급격한 하락 발생 {i} (-7% 이하)", value=p.get('use_drop_acc', False), key=f"s_acc_use_{i}")
                    acc_lim = st.number_input(f"누적 하락 기준 (%) {i}", -20.0, 0.0, p.get('acc_limit', -7.0), 0.1, format="%.1f", key=f"s_acc_lim_{i}")
                    use_exit = st.checkbox(f"전량매도 (S0로 이동) {i}", value=p.get('use_exit_all', False), key=f"s_exit_all_{i}")
                    
                    s3_protection.append({
                        'use_daily_drop': use_drop, 'drop_limit': drop_lim,
                        'use_ma60': use_ma60, 'ma60_limit': ma60_lim,
                        'use_ma200': use_ma200, 'ma200_limit': ma200_lim,
                        'use_vix_jump': use_vj, 'vix_jump': vj_val,
                        'use_gap_down': use_gap, 'gap_limit': gap_lim,
                        'use_drop_acc': use_acc, 'acc_limit': acc_lim,
                        'use_exit_all': use_exit
                    })
        
        st.divider()
        st.subheader("⚖️ 리밸런싱 및 스마트 조건")
        c1, c2, c3 = st.columns(3)
        reb_params = {
            'buy_reb_up': c1.number_input("매수 후 리밸런싱(↑) %", 0.0, 100.0, float(current_params.get('buy_reb_up', 0.03)) * 100, 0.1, format="%.1f", key="r_b_u") / 100.0,
            'buy_reb_down': c1.number_input("매수 후 리밸런싱(↓) %", -100.0, 0.0, float(current_params.get('buy_reb_down', -0.03)) * 100, 0.1, format="%.1f", key="r_b_d") / 100.0,
            'sell_reb_up': c2.number_input("매도 후 리밸런싱(↑) %", 0.0, 100.0, float(current_params.get('sell_reb_up', 0.03)) * 100, 0.1, format="%.1f", key="r_s_u") / 100.0,
            'sell_reb_down': c2.number_input("매도 후 리밸런싱(↓) %", -100.0, 0.0, float(current_params.get('sell_reb_down', -0.03)) * 100, 0.1, format="%.1f", key="r_s_d") / 100.0,
            'cash_ratio_pct': c3.number_input("현금 유지 비율(%)", 0.0, 100.0, float(current_params.get('cash_ratio_pct', current_params.get('cash_ratio', 0.0) * 100)), 1.0, key="conf_cash")
        }
        
        with st.expander("🛡️ 하락장 및 스마트 대응 설정 (Smart Filter)"):
            c1, c2 = st.columns(2)
            smart_config = {
                'use_panic': c1.toggle("하락장 매수 지연 (Panic Mode)", value=current_params.get('use_panic', True)),
                'panic_ma': c1.selectbox("지연 기준 이평선", [60, 120, 200], index=[60,120,200].index(current_params.get('panic_ma', 200))),
                'use_vix_safety': c2.toggle("VIX 대피 시스템 (Safety)", value=current_params.get('use_vix_safety', True)),
                'vix_exit': c2.slider("대피 VIX 레벨", 15, 45, current_params.get('vix_exit', 28)),
                'use_rsi_turbo': c2.toggle("RSI 과매도 가속 (Turbo)", value=current_params.get('use_rsi_turbo', True)),
                'rsi_turbo': c2.slider("가속 RSI 레벨", 20, 45, current_params.get('rsi_turbo', 31)),
                'use_sl_control': st.toggle("손절제어 사용 (Stop-Loss Control)", value=current_params.get('use_sl_control', False)),
                'sl_control_limit': st.slider("손절제어 임계값 (%) (이하일 때 매도차단)", -50, 0, current_params.get('sl_control_limit', -15))
            }
            st.write("**Panic Mode 상세 (매수 시 RSI 제한)**")
            sc1, sc2, sc3 = st.columns(3)
            smart_config.update({
                'panic_rsi_s1': sc1.number_input("S1 제한 RSI", 20, 45, current_params.get('panic_rsi_s1', 32)),
                'panic_rsi_s2': sc2.number_input("S2 제한 RSI", 20, 45, current_params.get('panic_rsi_s2', 32)),
                'panic_rsi_s3': sc3.number_input("S3 제한 RSI", 20, 45, current_params.get('panic_rsi_s3', 30))
            })
            
        return {
            'buy_signals': buy_signals, 'sell_signals': sell_signals, 's3_protection': s3_protection,
            'base_asset': base_asset, 'leverage_asset': leverage_asset, 'trade_at': trade_at,
            **reb_params, **smart_config
        }
