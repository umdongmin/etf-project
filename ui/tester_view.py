import streamlit as st

class TesterView:
    """전략 분석기 입력 UI 구성 클래스"""
    @staticmethod
    def render_opt_range(label, p_dict, field_name, min_v, max_v, step=1, format=None, key_suffix=None):
        opt_key = f"opt_{field_name}"
        rng_key = f"rng_{field_name}"
        if field_name == "news_q" and rng_key not in p_dict and "rng_news_q_val" in p_dict: rng_key = "rng_news_q_val"
        if field_name == "news_z" and rng_key not in p_dict and "rng_news_z_val" in p_dict: rng_key = "rng_news_z_val"
        return p_dict.get(opt_key, False), p_dict.get(rng_key, [min_v, max_v])

    @staticmethod
    def render_top_part(current_params, ticker_map=None, key_prefix=""):
        st.subheader("🌐 기본 자산 및 거래 설정")
        c1, c2, c3 = st.columns(3)
        base_asset_opts = ["QQQ", "QLD", "TQQQ"]
        leverage_asset_opts = ["QLD", "TQQQ"]
        trade_at_opts = ["종가", "익일 시가"]
        def fmt(x): return ticker_map[x] if ticker_map and x in ticker_map else x
        def k(base_key): return f"{key_prefix}{base_key}"
        base_asset = c1.selectbox("기준 자산 (시그널)", base_asset_opts, index=base_asset_opts.index(current_params.get('base_asset', 'QQQ')), format_func=fmt, key=k("conf_base"))
        leverage_asset = c2.selectbox("매매 대상 자산", leverage_asset_opts, index=leverage_asset_opts.index(current_params.get('leverage_asset', 'QLD')), format_func=fmt, key=k("conf_lev"))
        trade_at = c3.radio("매매 시점", trade_at_opts, index=trade_at_opts.index(current_params.get('trade_at', '종가')), horizontal=True, key=k("conf_trade_at"))
        st.divider()
        col_buy, col_sell = st.columns(2)
        buy_signals = []
        with col_buy:
            st.subheader("📥 Buy Conditions (OR)")
            for i in range(1, 4):
                with st.expander(f"매수 신호 {i}", expanded=False):
                    p = current_params['buy_signals'][i-1] if i <= len(current_params['buy_signals']) else {}
                    use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 {i}", value=p.get('use_rsi_wait', False), key=k(f"b_rsi_wait_use_{i}"), help="RSI가 기준치에 도달하면 즉시 매수하지 않고, 다음 조건이 충족될 때까지 대기 상태를 유지합니다.")
                    rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 {i}", 0, 100, p.get('rsi_wait_val', 35), key=k(f"b_rsi_wait_val_{i}"))
                    rsi_val = st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 35), key=k(f"b_rsi_{i}"))
                    opt_rsi, rng_rsi = TesterView.render_opt_range("RSI", p, "rsi_val", 10, 60, 1, key_suffix=k(f"b_rsi_{i}"))
                    rsi_cross = st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', False), key=k(f"b_cross_{i}"))
                    rsi_inc = st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=k(f"b_inc_{i}"))
                    macd_inc = st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', False), key=k(f"b_m_inc_{i}"))
                    macd_signal_below = st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', False), key=k(f"b_m_below_{i}"))
                    macd_golden = st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=k(f"b_m_golden_{i}"))
                    use_adx = st.checkbox(f"ADX 필터 사용 {i}", value=p.get('use_adx', False), key=k(f"b_use_adx_{i}"))
                    adx_op = st.selectbox(f"ADX 연산자 {i}", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=k(f"b_adx_op_{i}"))
                    adx_val = st.number_input(f"ADX 기준 {i}", 0, 100, p.get('adx_val', 40), key=k(f"b_adx_{i}"))
                    opt_adx, rng_adx = TesterView.render_opt_range("ADX", p, "adx_val", 5, 80, 1, key_suffix=k(f"b_adx_{i}"))
                    bb_lower = st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=k(f"b_bb_low_{i}"))
                    use_willr = st.checkbox(f"Williams %R 상향 돌파 {i}", value=p.get('use_willr', False), key=k(f"b_willr_use_{i}"))
                    willr_val = st.number_input(f"W%R 상향 기준값 {i}", -100, 0, p.get('willr_val', -80), key=k(f"b_willr_val_{i}"))
                    opt_willr, rng_willr = TesterView.render_opt_range("Williams %R", p, "willr_val", -95, -50, 1, key_suffix=k(f"b_willr_{i}"))
                    di_plus_above = st.checkbox(f"매수 우세 상태유지 (DI+ > DI-) {i}", value=p.get('di_plus_above', False), key=k(f"b_di_plus_above_{i}"))
                    di_plus_cross = st.checkbox(f"DI+ 골든크로스 순간 (DI+ > DI-) {i}", value=p.get('di_plus_cross', False), key=k(f"b_di_plus_cross_{i}"))
                    use_sar = st.checkbox(f"파라볼릭 SAR 상향 돌파 {i}", value=p.get('use_sar', False), key=k(f"b_sar_use_{i}"))
                    st.markdown("---")
                    use_dev200 = st.checkbox(f"200일 이격도 필터 {i}", value=p.get('use_dev200', False), key=k(f"b_dev200_use_{i}"))
                    col_d1, col_d2 = st.columns(2)
                    dev200_op = col_d1.selectbox(f"이격도 조건 {i}", [">", ">=", "<", "<="], index=[">", ">=", "<", "<="].index(p.get('dev200_op', '>=')), key=k(f"b_dev200_op_{i}"))
                    dev200_val = col_d2.number_input(f"이격도 기준 {i}", 0.5, 1.5, float(p.get('dev200_val', 1.0)), 0.01, key=k(f"b_dev200_val_{i}"))
                    opt_dev200, rng_dev200 = TesterView.render_opt_range("200일 이격도", p, "dev200_val", 0.8, 1.3, 0.01, format="%.2f", key_suffix=k(f"b_dev200_{i}"))
                    buy_signals.append({'use_rsi_wait': use_rsi_wait, 'rsi_wait_val': rsi_wait_val, 'rsi_val': rsi_val, 'opt_rsi_val': opt_rsi, 'rng_rsi_val': rng_rsi, 'rsi_cross': rsi_cross, 'rsi_inc': rsi_inc, 'macd_inc': macd_inc, 'macd_signal_below': macd_signal_below, 'macd_golden': macd_golden, 'use_adx': use_adx, 'adx_op': adx_op, 'adx_val': adx_val, 'opt_adx_val': opt_adx, 'rng_adx_val': rng_adx, 'bb_lower': bb_lower, 'use_willr': use_willr, 'willr_val': willr_val, 'opt_willr_val': opt_willr, 'rng_willr_val': rng_willr, 'di_plus_above': di_plus_above, 'di_plus_cross': di_plus_cross, 'use_sar': use_sar, 'use_dev200': use_dev200, 'dev200_op': dev200_op, 'dev200_val': dev200_val, 'opt_dev200_val': opt_dev200, 'rng_dev200_val': rng_dev200})
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            for i in range(1, 5):
                with st.expander(f"매도 신호 {i}", expanded=False):
                    p = current_params['sell_signals'][i-1] if i <= len(current_params['sell_signals']) else {}
                    use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 {i}", value=p.get('use_rsi_wait', False), key=k(f"s_rsi_wait_use_{i}"))
                    rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 {i}", 0, 100, p.get('rsi_wait_val', 70), key=k(f"s_rsi_wait_val_{i}"))
                    rsi_val = st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 70 if i==1 else 0), key=k(f"s_rsi_{i}"))
                    opt_rsi, rng_rsi = TesterView.render_opt_range("RSI", p, "rsi_val", 40, 95, 1, key_suffix=k(f"s_rsi_{i}"))
                    rsi_dead = st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', i==2), key=k(f"s_dead_{i}"))
                    rsi_dec = st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', i==1), key=k(f"s_dec_{i}"))
                    macd_dec = st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', i==2), key=k(f"s_m_dec_{i}"))
                    macd_signal_above = st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', i==2), key=k(f"s_m_above_{i}"))
                    macd_dead = st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', i==3), key=k(f"s_m_dead_{i}"))
                    di_minus_above = st.checkbox(f"매도 우세 상태유지 (DI+ > DI-) {i}", value=p.get('di_minus_above', False), key=k(f"s_di_{i}"))
                    di_minus_cross = st.checkbox(f"DI- 데드크로스 순간 (DI- > DI+) {i}", value=p.get('di_minus_cross', False), key=k(f"s_di_minus_cross_{i}"))
                    bb_upper = st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=k(f"s_bb_up_{i}"))
                    use_chan = st.checkbox(f"샹들리에 엑시트 (고점방어) {i}", value=p.get('use_chandelier', i==4), key=k(f"s_chan_use_{i}"))
                    chan_mult = st.number_input(f"샹들리에 변동성 승수 {i}", 0.1, 10.0, float(p.get('chandelier_mult', 3.0)), 0.1, key=k(f"s_chan_mult_{i}"))
                    opt_chan, rng_chan = TesterView.render_opt_range("샹들리에", p, "chandelier_mult", 1.0, 5.0, 0.1, format="%.1f", key_suffix=k(f"s_chan_{i}"))
                    use_sar = st.checkbox(f"파라볼릭 SAR 하향 이탈 {i}", value=p.get('use_sar', False), key=k(f"s_sar_use_{i}"))
                    use_willr = st.checkbox(f"Williams %R 하향 이탈 {i}", value=p.get('use_willr', False), key=k(f"s_willr_use_{i}"))
                    willr_val = st.number_input(f"W%R 하향 기준값 {i}", -100, 0, p.get('willr_val', -20), key=k(f"s_willr_val_{i}"))
                    opt_willr, rng_willr = TesterView.render_opt_range("Williams %R", p, "willr_val", -50, -5, 1, key_suffix=k(f"s_willr_{i}"))
                    st.markdown("---")
                    use_dev200 = st.checkbox(f"200일 이격도 필터 {i}", value=p.get('use_dev200', False), key=k(f"s_dev200_use_{i}"))
                    col_d1, col_d2 = st.columns(2)
                    dev200_op = col_d1.selectbox(f"이격도 조건 {i}", [">", ">=", "<", "<="], index=[">", ">=", "<", "<="].index(p.get('dev200_op', '>=')), key=k(f"s_dev200_op_{i}"))
                    dev200_val = col_d2.number_input(f"이격도 기준 {i}", 0.5, 1.5, float(p.get('dev200_val', 1.0)), 0.01, key=k(f"s_dev200_val_{i}"))
                    opt_dev200, rng_dev200 = TesterView.render_opt_range("200일 이격도", p, "dev200_val", 0.8, 1.3, 0.01, format="%.2f", key_suffix=k(f"s_dev200_{i}"))
                    
                    # News Sentiment Filter
                    st.markdown("---")
                    st.markdown("###### 📰 뉴스 심리 필터")
                    col_nq, col_nz = st.columns(2)
                    use_news_q = col_nq.checkbox(f"뉴스 심리 (Q) 필터 {i}", value=p.get('use_news_q', False), key=k(f"s_news_q_use_{i}"))
                    news_q_op = col_nq.selectbox(f"뉴스 심리 (Q) 연산자 {i}", ["<=", ">="], index=0 if p.get('news_q_op', "<=") == "<=" else 1, key=k(f"s_news_q_op_{i}"))
                    news_q_val = col_nq.number_input(f"뉴스 심리 (Q) 기준 {i}", -1.0, 1.0, float(p.get('news_q_val', 0.0)), 0.01, format="%.2f", key=k(f"s_news_q_val_{i}"))
                    opt_news_q, rng_news_q = TesterView.render_opt_range("뉴스 심리 (Q)", p, "news_q", -0.5, 0.5, 0.01, format="%.2f", key_suffix=k(f"s_news_q_{i}"))

                    use_news_z = col_nz.checkbox(f"뉴스 심리 (Z) 필터 {i}", value=p.get('use_news_z', False), key=k(f"s_news_z_use_{i}"))
                    news_z_op = col_nz.selectbox(f"뉴스 심리 (Z) 연산자 {i}", ["<=", ">="], index=0 if p.get('news_z_op', "<=") == "<=" else 1, key=k(f"s_news_z_op_{i}"))
                    news_z_val = col_nz.number_input(f"뉴스 심리 (Z) 기준 {i}", -3.0, 3.0, float(p.get('news_z_val', 0.0)), 0.01, format="%.2f", key=k(f"s_news_z_val_{i}"))
                    opt_news_z, rng_news_z = TesterView.render_opt_range("뉴스 심리 (Z)", p, "news_z", -2.0, 2.0, 0.01, format="%.2f", key_suffix=k(f"s_news_z_{i}"))

                    sell_signals.append({
                        'use_rsi_wait': use_rsi_wait, 'rsi_wait_val': rsi_wait_val, 'rsi_val': rsi_val,
                        'opt_rsi_val': opt_rsi, 'rng_rsi_val': rng_rsi,
                        'rsi_dead': rsi_dead, 'rsi_dec': rsi_dec, 'macd_dec': macd_dec,
                        'macd_signal_above': macd_signal_above, 'macd_dead': macd_dead,
                        'di_minus_above': di_minus_above, 'di_minus_cross': di_minus_cross,
                        'bb_upper': bb_upper, 'use_chandelier': use_chan, 'chandelier_mult': chan_mult,
                        'opt_chandelier_mult': opt_chan, 'rng_chandelier_mult': rng_chan,
                        'use_sar': use_sar, 'use_willr': use_willr, 'willr_val': willr_val,
                        'opt_willr_val': opt_willr, 'rng_willr_val': rng_willr,
                        'use_news_q': use_news_q, 'news_q_op': news_q_op, 'news_q_val': news_q_val,
                        'opt_news_q': opt_news_q, 'rng_news_q_val': rng_news_q,
                        'use_news_z': use_news_z, 'news_z_op': news_z_op, 'news_z_val': news_z_val,
                        'opt_news_z': opt_news_z, 'rng_news_z_val': rng_news_z,
                        'use_dev200': use_dev200, 'dev200_op': dev200_op, 'dev200_val': dev200_val,
                        'opt_dev200_val': opt_dev200, 'rng_dev200_val': rng_dev200
                    })
            st.write("---")
            st.subheader("🛡️ S3(레버리지 100%) 보호 설정")
            s3_p_list = current_params.get('s3_protection', [{}, {}])
            while len(s3_p_list) < 2: s3_p_list.append({})
            s3_protection = []
            for i in range(1, 4):
                with st.expander(f"보호 신호 {i}", expanded=False):
                    p = s3_p_list[i-1] if i <= len(s3_p_list) else {}
                    use_drop = st.checkbox(f"일일 급락 매도 제어 {i}", value=p.get('use_daily_drop', False), key=k(f"s_drop_use_{i}"))
                    drop_lim = st.number_input(f"급락 기준 (%) {i}", -10.0, 0.0, p.get('drop_limit', -3.0), 0.1, format="%.1f", key=k(f"s_drop_lim_{i}"))
                    col1, col2 = st.columns(2)
                    use_ma60 = col1.checkbox(f"60일 이평선 하회 {i}", value=p.get('use_ma60', False), key=k(f"s_ma60_use_{i}"))
                    ma60_lim = col1.number_input(f"60일 기준 (%) {i}", -50.0, 50.0, p.get('ma60_limit', 0.0), 0.1, format="%.1f", key=k(f"s_ma60_lim_{i}"))
                    use_ma200 = col2.checkbox(f"200일 이평선 하회 {i}", value=p.get('use_ma200', False), key=k(f"s_ma200_use_{i}"))
                    ma200_lim = col2.number_input(f"200일 기준 (%) {i}", -50.0, 50.0, p.get('ma200_limit', 0.0), 0.1, format="%.1f", key=k(f"s_ma200_lim_{i}"))
                    use_vj = st.checkbox(f"VXN 급등 시그널 사용 {i}", value=p.get('use_vxn_jump', False), key=k(f"s_vj_use_{i}"))
                    vj_val = st.number_input(f"VXN 급등 기준 {i}", 0.0, 100.0, p.get('vxn_jump', 15.0), 0.1, format="%.1f", key=k(f"s_vj_{i}"))
                    use_gap = st.checkbox(f"패닉 시가 갭 발생 {i}", value=p.get('use_gap_down', False), key=k(f"s_gap_use_{i}"))
                    gap_lim = st.number_input(f"시가 갭 기준 {i}", -10.0, 0.0, p.get('gap_limit', -3.0), 0.1, format="%.1f", key=k(f"s_gap_lim_{i}"))
                    use_acc = st.checkbox(f"급격한 하락 발생 {i}", value=p.get('use_drop_acc', False), key=k(f"s_acc_use_{i}"))
                    acc_lim = st.number_input(f"누적 하락 기준 {i}", -20.0, 0.0, p.get('acc_limit', -7.0), 0.1, format="%.1f", key=k(f"s_acc_lim_{i}"))
                    use_chan = st.checkbox(f"S3 샹들리에 엑시트 사용 {i}", value=p.get('use_chandelier', False), key=k(f"s3_chan_use_{i}"))
                    chan_mult = st.number_input(f"S3 샹들리에 승수 {i}", 0.1, 10.0, float(p.get('chandelier_mult', 3.0)), 0.1, key=k(f"s3_chan_mult_{i}"))
                    use_exit = st.checkbox(f"전량매도 (S0로 이동) {i}", value=p.get('use_exit_all', False), key=k(f"s_exit_all_{i}"))
                    s3_protection.append({'use_daily_drop': use_drop, 'drop_limit': drop_lim, 'use_ma60': use_ma60, 'ma60_limit': ma60_lim, 'use_ma200': use_ma200, 'ma200_limit': ma200_lim, 'use_vxn_jump': use_vj, 'vxn_jump': vj_val, 'use_gap_down': use_gap, 'gap_limit': gap_lim, 'use_drop_acc': use_acc, 'acc_limit': acc_lim, 'use_chandelier': use_chan, 'chandelier_mult': chan_mult, 'use_exit_all': use_exit})
        return {'buy_signals': buy_signals, 'sell_signals': sell_signals, 's3_protection': s3_protection, 'base_asset': base_asset, 'leverage_asset': leverage_asset, 'trade_at': trade_at}

    @staticmethod
    def render_bottom_part(current_params, ticker_map=None, reb_mode=None, key_prefix=""):
        def k(base_key): return f"{key_prefix}{base_key}"
        
        # 폼 외부(HistoryView)에서 선택된 핵심 모드 상태를 가져옴
        use_fixed = current_params.get('use_fixed_reb', True)
        use_atr = current_params.get('use_atr_reb', False)
        
        c1, c2, c3 = st.columns(3)
        
        # (1) 고정 비율 설정 섹션
        with c1:
            st.markdown("##### 🔹 고정 비율 세부 설정")
            # 매수 상향
            buy_reb_up = st.number_input("매수 상향 (%)", 0.0, 100.0, float(current_params.get('buy_reb_up', 0.02)) * 100, 0.1, key=k("r_b_u")) / 100.0
            opt_rbu, rng_rbu = TesterView.render_opt_range("매수 상향", current_params, "buy_reb_up", 0.1, 5.0, 0.1, key_suffix=k("r_b_u")) # opt_mode 제거됨
            
            # 매수 하향
            buy_reb_down = st.number_input("매수 하향 (%)", -100.0, 0.0, float(current_params.get('buy_reb_down', -0.07)) * 100, 0.1, key=k("r_b_d")) / 100.0
            opt_rbd, rng_rbd = TesterView.render_opt_range("매수 하향", current_params, "buy_reb_down", -15.0, -1.0, 0.1, key_suffix=k("r_b_d")) # opt_mode 제거됨
            
            # 매도 상향
            sell_reb_up = st.number_input("매도 상향 (%)", 0.0, 100.0, float(current_params.get('sell_reb_up', 0.03)) * 100, 0.1, key=k("r_s_u")) / 100.0
            opt_rsu, rng_rsu = TesterView.render_opt_range("매도 상향", current_params, "sell_reb_up", 0.5, 10.0, 0.1, key_suffix=k("r_s_u")) # opt_mode 제거됨
            
            # 매도 하향
            sell_reb_down = st.number_input("매도 하향 (%)", -100.0, 0.0, float(current_params.get('sell_reb_down', -0.035)) * 100, 0.1, key=k("r_s_d")) / 100.0
            opt_rsd, rng_rsd = TesterView.render_opt_range("매도 하향", current_params, "sell_reb_down", -10.0, -0.5, 0.1, key_suffix=k("r_s_d")) # opt_mode 제거됨
            
            fixed_params = {
                'buy_reb_up': buy_reb_up, 'opt_buy_reb_up': opt_rbu, 'rng_buy_reb_up': rng_rbu,
                'buy_reb_down': buy_reb_down, 'opt_buy_reb_down': opt_rbd, 'rng_buy_reb_down': rng_rbd,
                'sell_reb_up': sell_reb_up, 'opt_sell_reb_up': opt_rsu, 'rng_sell_reb_up': rng_rsu,
                'sell_reb_down': sell_reb_down, 'opt_sell_reb_down': opt_rsd, 'rng_sell_reb_down': rng_rsd
            }
        # (3) 리밸런싱 인터락 설정 섹션 (신기능)
        with c3:
            st.markdown("##### 🚦 리밸런싱 인터락")
            use_reb_interlock = st.toggle("인터락 활성화", 
                                       value=current_params.get('use_reb_interlock', False), 
                                       help="이격도와 VXN이 모두 높을 때 리밸런싱 추가 매수를 차단합니다.",
                                       key=k("use_reb_interlock"))
            
            reb_interlock_dev = st.number_input("이격도 임계값", 1.00, 2.00, 
                                             float(current_params.get('reb_interlock_dev', 1.13)), 0.01, 
                                             key=k("reb_interlock_dev"))
            
            reb_interlock_vxn = st.number_input("VXN 임계값", 10.0, 60.0, 
                                             float(current_params.get('reb_interlock_vxn', 22.0)), 0.5, 
                                             key=k("reb_interlock_vxn"))
            
            fixed_params.update({
                'use_reb_interlock': use_reb_interlock,
                'reb_interlock_dev': reb_interlock_dev,
                'reb_interlock_vxn': reb_interlock_vxn
            })

            st.markdown("---")
            st.markdown("##### 🛑 매수 신호 인터락")
            use_buy_interlock = st.toggle("매수 인터락 활성화",
                                          value=current_params.get('use_buy_interlock', False),
                                          help="VXN과 이격도가 동시에 임계값 초과 시 매수 신호를 차단합니다.",
                                          key=k("use_buy_interlock"))

            interlock_vxn = st.number_input("VXN 임계값", 10.0, 60.0,
                                            float(current_params.get('interlock_vxn', 25.0)), 0.5,
                                            key=k("interlock_vxn"))

            interlock_dev = st.number_input("이격도(Dev200) 임계값", 1.00, 2.00,
                                            float(current_params.get('interlock_dev', 1.1)), 0.01,
                                            key=k("interlock_dev"))

            interlock_min_stage = st.selectbox("적용 스테이지",
                                               options=[1, 2, 3],
                                               index=current_params.get('interlock_min_stage', 1) - 1,
                                               format_func=lambda x: {1: "S1~S3 (전체)", 2: "S2~S3", 3: "S3만"}[x],
                                               key=k("interlock_min_stage"))

            fixed_params.update({
                'use_buy_interlock': use_buy_interlock,
                'interlock_vxn': interlock_vxn,
                'interlock_dev': interlock_dev,
                'interlock_min_stage': interlock_min_stage
            })

        # (2) ATR 변동성 설정 섹션
        with c2:
            st.markdown("##### 🔸 ATR 변동성 세부 설정")
            # 매수 상향 (불타기)
            atr_m_buy_up = st.number_input("매수 상향 승수 (불타기)", 0.1, 25.0, float(current_params.get('atr_mult_buy_up', 10.0)), 0.1, key=k("atr_m_buy_up"))
            opt_amb_u, rng_amb_u = TesterView.render_opt_range("ATR 매수 상향", current_params, "atr_m_buy_up", 1.0, 20.0, 0.1, key_suffix=k("atr_m_buy_up")) # opt_mode 제거됨
            
            # 매수 하향 (물타기)
            atr_m_buy_down = st.number_input("매수 하향 승수 (물타기)", 0.1, 15.0, float(current_params.get('atr_mult_buy_down', 1.5)), 0.1, key=k("atr_m_buy_down"))
            opt_amb_d, rng_amb_d = TesterView.render_opt_range("ATR 매수 하향", current_params, "atr_m_buy_down", 0.5, 5.0, 0.1, key_suffix=k("atr_m_buy_down")) # opt_mode 제거됨
            
            # 매도 승수
            atr_m_sell = st.number_input("매도 승수 (K2)", 0.1, 15.0, float(current_params.get('atr_mult_sell', 3.0)), 0.1, key=k("atr_m_sell"))
            opt_ams, rng_ams = TesterView.render_opt_range("ATR 매도 승수", current_params, "atr_m_sell", 1.0, 6.0, 0.1, key_suffix=k("atr_m_sell")) # opt_mode 제거됨
            
            # ATR 기준일
            atr_p_buy = st.selectbox("매수 ATR 기준일", [14, 20], index=0 if current_params.get('atr_period_buy', 20) == 14 else 1, key=k("atr_p_buy"))
            atr_p_sell = st.selectbox("매도 ATR 기준일", [14, 20], index=1 if current_params.get('atr_period_sell', 20) == 20 else 0, key=k("atr_p_sell"))
            
            atr_params = {
                'atr_mult_buy_up': atr_m_buy_up, 'opt_atr_m_buy_up': opt_amb_u, 'rng_atr_m_buy_up': rng_amb_u,
                'atr_mult_buy_down': atr_m_buy_down, 'opt_atr_m_buy_down': opt_amb_d, 'rng_atr_m_buy_down': rng_amb_d,
                'atr_mult_sell': atr_m_sell, 'opt_atr_m_sell': opt_ams, 'rng_atr_m_sell': rng_ams,
                'atr_period_buy': atr_p_buy, 'atr_period_sell': atr_p_sell
            }

        # (3) 공통 설정 및 파라미터 취합
        with c3:
            st.markdown("##### 🧱 공통 설정")
            cash_ratio_pct = st.number_input("현금 유지 비율(%)", 0.0, 100.0, float(current_params.get('cash_ratio_pct', 0.0)), 1.0, key=k("conf_cash"))
            
        reb_params = {
            'use_fixed_reb': use_fixed,
            'use_atr_reb': use_atr,
            **fixed_params,
            **atr_params,
            'cash_ratio_pct': cash_ratio_pct
        }
        
        with st.expander("🛡️ 하락장 및 스마트 대응 설정 (Smart Filter)"):
            c1, c2 = st.columns(2)
            smart_config = {
                'use_panic': c1.toggle("하락장 매수 지연 (Panic Mode)", value=current_params.get('use_panic', True), key=k("use_panic_chk")),
                'panic_ma': c1.selectbox("지연 기준 이평선", [60, 120, 200], index=[60,120,200].index(current_params.get('panic_ma', 200)), key=k("panic_ma_sel")),
                'use_vxn_safety': c2.toggle("VXN 대피 시스템 (Safety)", value=current_params.get('use_vxn_safety', current_params.get('use_vix_safety', True)), key=k("use_vxn_safety_chk")),
                'vxn_exit': c2.slider("대피 VXN 레벨", 15.0, 45.0, float(current_params.get('vxn_exit', current_params.get('vix_exit', 28.0))), step=0.5, key=k("vxn_exit_sli")),
                'vxn_dev_filter': c2.number_input("VXN 대피 이격도 필터 (0=비활성)", 0.0, 2.0, float(current_params.get('vxn_dev_filter', 0.0)), step=0.01, help="0이면 VXN 조건만 사용. 값 설정 시 Dev200 >= 값일 때만 VXN 대피 발동 (과열 구간 필터)", key=k("vxn_dev_filter_inp")),
                'use_rsi_turbo': c2.toggle("RSI 과매도 가속 (Turbo)", value=current_params.get('use_rsi_turbo', True), key=k("use_rsi_turbo_chk")),
                'rsi_turbo': c2.slider("가속 RSI 레벨", 20.0, 45.0, float(current_params.get('rsi_turbo', 31.0)), step=0.5, key=k("rsi_turbo_sli")),
                'use_sl_control': st.toggle("손절제어 사용 (Stop-Loss Control)", value=current_params.get('use_sl_control', False), key=k("use_sl_control_chk")),
                'sl_control_limit': st.slider("손절제어 임계값 (%)", -50.0, 0.0, float(current_params.get('sl_control_limit', -15.0)), step=1.0, key=k("sl_control_lim_sli")),
                'use_set_sl': st.toggle("개별 거래 세트 기준 손절 라인 사용", value=current_params.get('use_set_sl', False), key=k("use_set_sl_chk")),
                'set_sl_limit': st.slider("거래 세트별 손절 라인 (%)", -50.0, 0.0, float(current_params.get('set_sl_limit', -15.0)), step=1.0, key=k("set_sl_lim_sli")),
                'panic_rsi_s1': current_params.get('panic_rsi_s1', 27),
                'panic_rsi_s2': current_params.get('panic_rsi_s2', 28),
                'panic_rsi_s3': current_params.get('panic_rsi_s3', 30)
            }
            panic_buy_signals_list = []
            for i in range(1, 4):
                with st.expander(f"하락장 매수 신호 S{i}", expanded=False):
                    p_list = current_params.get('panic_buy_signals', [])
                    p = p_list[i-1] if i <= len(p_list) else {}
                    
                    use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 (S{i})", value=p.get('use_rsi_wait', False), key=k(f"pb_rsi_wait_use_{i}"))
                    rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 (S{i})", 0, 100, p.get('rsi_wait_val', 35), key=k(f"pb_rsi_wait_val_{i}"))
                    rsi_val = st.number_input(f"RSI 기준 (S{i})", 0, 100, p.get('rsi_val', 27 if i==1 else (28 if i==2 else 30)), key=k(f"pb_rsi_{i}"))
                    opt_rsi, rng_rsi = TesterView.render_opt_range(f"RSI (S{i})", p, "rsi_val", 15, 50, 1, key_suffix=k(f"pb_rsi_{i}"))
                    
                    rsi_cross = st.checkbox(f"RSI Golden Cross (S{i})", value=p.get('rsi_cross', False), key=k(f"pb_cross_{i}"))
                    rsi_inc = st.checkbox(f"RSI 증가 (S{i})", value=p.get('rsi_inc', False), key=k(f"pb_inc_{i}"))
                    macd_inc = st.checkbox(f"MACD 증가 (S{i})", value=p.get('macd_inc', False), key=k(f"pb_m_inc_{i}"))
                    macd_signal_below = st.checkbox(f"MACD < Signal (S{i})", value=p.get('macd_signal_below', False), key=k(f"pb_m_below_{i}"))
                    macd_golden = st.checkbox(f"MACD Golden Cross (S{i})", value=p.get('macd_golden', False), key=k(f"pb_m_golden_{i}"))
                    use_adx = st.checkbox(f"ADX 필터 사용 (S{i})", value=p.get('use_adx', False), key=k(f"pb_use_adx_{i}"))
                    adx_op = st.selectbox(f"ADX 연산자 (S{i})", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=k(f"pb_adx_op_{i}"))
                    adx_val = st.number_input(f"ADX 기준 (S{i})", 0, 100, p.get('adx_val', 40), key=k(f"pb_adx_{i}"))
                    opt_adx, rng_adx = TesterView.render_opt_range(f"ADX (S{i})", p, "adx_val", 5, 80, 1, key_suffix=k(f"pb_adx_{i}"))
                    
                    bb_lower = st.checkbox(f"Bollinger Lower (S{i})", value=p.get('bb_lower', False), key=k(f"pb_bb_low_{i}"))
                    use_willr = st.checkbox(f"Williams %R 상향 돌파 (S{i})", value=p.get('use_willr', False), key=k(f"pb_willr_use_{i}"))
                    willr_val = st.number_input(f"W%R 상향 기준값 (S{i})", -100, 0, p.get('willr_val', -80), key=k(f"pb_willr_val_{i}"))
                    opt_willr, rng_willr = TesterView.render_opt_range(f"W%R (S{i})", p, "willr_val", -95, -50, 1, key_suffix=k(f"pb_willr_{i}"))
                    
                    di_plus_above = st.checkbox(f"매수 우세 상태유지 (DI+ > DI-) (S{i})", value=p.get('di_plus_above', False), key=k(f"pb_di_plus_above_{i}"))
                    di_plus_cross = st.checkbox(f"DI+ 골든크로스 순간 (DI+ > DI-) (S{i})", value=p.get('di_plus_cross', False), key=k(f"pb_di_plus_cross_{i}"))
                    use_sar = st.checkbox(f"파라볼릭 SAR 상향 돌파 (S{i})", value=p.get('use_sar', False), key=k(f"pb_sar_use_{i}"))
                    
                    panic_buy_signals_list.append({
                        'use_rsi_wait': use_rsi_wait, 'rsi_wait_val': rsi_wait_val, 'rsi_val': rsi_val,
                        'opt_rsi_val': opt_rsi, 'rng_rsi_val': rng_rsi,
                        'rsi_cross': rsi_cross, 'rsi_inc': rsi_inc, 'macd_inc': macd_inc,
                        'macd_signal_below': macd_signal_below, 'macd_golden': macd_golden,
                        'use_adx': use_adx, 'adx_op': adx_op, 'adx_val': adx_val,
                        'opt_adx_val': opt_adx, 'rng_adx_val': rng_adx,
                        'bb_lower': bb_lower, 'use_willr': use_willr, 'willr_val': willr_val,
                        'opt_willr_val': opt_willr, 'rng_willr_val': rng_willr,
                        'di_plus_above': di_plus_above, 'di_plus_cross': di_plus_cross, 'use_sar': use_sar
                    })
            
            smart_config['panic_buy_signals'] = panic_buy_signals_list
            
        # --- [신규] 국면별 동적 현금 비중 제어 섹션 (별도 익스펜더) ---
        with st.expander("🌊 시장 국면별 동적 현금 비중 제어 (Dynamic Cash Allocation)"):
            dc = current_params.get('dynamic_cash', {
                'use': False, 'bull_vxn': 18.0, 'caution_vxn': 35.0, 'bear_vxn': 40.0,
                'cash_bull': 0.0, 'cash_neutral': 20.0, 'cash_caution': 30.0, 'cash_bear': 80.0
            })
            
            use_dc = st.toggle("국면별 동적 현금 비중 제어 활성화", value=dc.get('use', False), key=k("dc_use_toggle"))
            
            dc_params = {'use': use_dc}
            col_th, col_val = st.columns(2)
            with col_th:
                st.caption("🎯 국면 판단 기준 (VXN)")
                dc_params['bull_vxn'], dc_params['rng_bull_vxn'] = TesterView.render_opt_range(
                    "강세/중립 경계 (VXN)", dc, "bull_vxn", 10.0, 25.0, 0.5, key_suffix=k("dc_bull_v")
                )
                if not dc_params.get('opt_bull_vxn'):
                    dc_params['bull_vxn'] = st.slider("강세/중립 경계 (VXN)", 10.0, 25.0, float(dc.get('bull_vxn', 18.0)), 0.5, key=k("dc_bull_v_sli"))

                dc_params['caution_vxn'], dc_params['rng_caution_vxn'] = TesterView.render_opt_range(
                    "중립/주의 경계 (VXN)", dc, "caution_vxn", 25.0, 40.0, 0.5, key_suffix=k("dc_caution_v")
                )
                if not dc_params.get('opt_caution_vxn'):
                    dc_params['caution_vxn'] = st.slider("중립/주의 경계 (VXN)", 25.0, 40.0, float(dc.get('caution_vxn', 35.0)), 0.5, key=k("dc_caution_v_sli"))

                dc_params['bear_vxn'], dc_params['rng_bear_vxn'] = TesterView.render_opt_range(
                    "주의/하락장 경계 (VXN)", dc, "bear_vxn", 35.0, 50.0, 0.5, key_suffix=k("dc_bear_v")
                )
                if not dc_params.get('opt_bear_vxn'):
                    dc_params['bear_vxn'] = st.slider("주의/하락장 경계 (VXN)", 35.0, 50.0, float(dc.get('bear_vxn', 40.0)), 0.5, key=k("dc_bear_v_sli"))

                st.write("") 
                dc_params['ma200_buffer'], dc_params['rng_ma200_buffer'] = TesterView.render_opt_range(
                    "하락장 판정 MA200 버퍼 (%)", dc, "ma200_buffer", 0.0, 5.0, 0.1, key_suffix=k("dc_ma_buf")
                )
                if not dc_params.get('opt_ma200_buffer'):
                    dc_params['ma200_buffer'] = st.slider("하락장 판정 MA200 버퍼 (%)", 0.0, 5.0, float(dc.get('ma200_buffer', 0.0)), 0.1, key=k("dc_ma_buf_sli"))
                st.caption("💡 주가가 [MA200 * (1 - 버퍼%)] 아래로 내려가면 즉시 하락장(Bear)으로 판정합니다.")
            
            with col_val:
                if not dc_params.get('opt_cash_bear'): dc_params['cash_bear'] = st.number_input("하락장 현금 (%)", 0.0, 100.0, float(dc.get('cash_bear', 80.0)), 1.0, key=k("dc_cash_bear_val"))
            smart_config['dynamic_cash'] = dc_params
        return {**reb_params, **smart_config}

    @staticmethod
    def render_period_selector(key_suffix=""):
        import datetime
        st.markdown("##### 📅 분석 대상 기간 설정")
        col1, col2 = st.columns([1.5, 2])
        period_opts = ["전체 (2010~)", "최근 5년", "최근 3년", "최근 1년", "직접 입력"]
        
        state_key = f"period_sel_type_{key_suffix}" if key_suffix else "period_sel_type"
        box_key = f"period_sel_type_box_{key_suffix}" if key_suffix else "period_sel_type_box"
        
        if state_key not in st.session_state: 
            st.session_state[state_key] = "전체 (2010~)"
            
        sel_type = col1.selectbox("기간 프리셋", period_opts, index=period_opts.index(st.session_state[state_key]), key=box_key)
        st.session_state[state_key] = sel_type
        end_date = datetime.date.today()
        start_date = datetime.date(2010, 1, 1)
        if sel_type == "최근 5년": start_date = end_date - datetime.timedelta(days=5*365)
        elif sel_type == "최근 3년": start_date = end_date - datetime.timedelta(days=3*365)
        elif sel_type == "최근 1년": start_date = end_date - datetime.timedelta(days=1*365)
        elif sel_type == "직접 입력":
            c1, c2 = col2.columns(2)
            start_date = c1.date_input("시작일", value=datetime.date(2023, 1, 1), key="period_custom_start")
            end_date = c2.date_input("종료일", value=end_date, key="period_custom_end")
        if sel_type != "직접 입력": col2.info(f"선택 범위: **{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}**")
        return start_date, end_date

    @staticmethod
    def render_summary_dashboard(results_df):
        if results_df is None or results_df.empty:
            st.warning("📊 시뮬레이션 결과가 없습니다.")
            return
        total_ret = (results_df['Value'].iloc[-1] / 10000.0 - 1) * 100
        days = (results_df.index[-1] - results_df.index[0]).days
        years = days / 365.25
        cagr = ((results_df['Value'].iloc[-1] / 10000.0) ** (1/years) - 1) * 100 if years > 0 else 0
        roll_max = results_df['Value'].cummax()
        drawdown = (results_df['Value'] - roll_max) / roll_max
        mdd = drawdown.min() * 100
        st.markdown("##### 🚀 실시간 시뮬레이션 성과 요약")
        m1, m2, m3 = st.columns(3)
        m1.metric("누적 수익률", f"{total_ret:+.1f}%")
        m2.metric("연평균 수익률(CAGR)", f"{cagr:.1f}%")
        m3.metric("최대 낙폭(MDD)", f"{mdd:.1f}%")

    @staticmethod
    def render_bond_settings(current_params, current_lev_params=None, key_prefix="bond_"):
        """채권 전략 설정 UI — Layer 1 (자산 선택) / Layer 2 (레버리지 실행) 분리"""
        def k(base_key): return f"{key_prefix}{base_key}"
        lp = current_lev_params or {}

        st.subheader("📉 채권 전략 설정")
        tab1, tab2 = st.tabs(["Layer 1 — 자산 선택 (WHAT)", "Layer 2 — 레버리지 실행 (HOW)"])

        # ── Tab 1: Layer 1 (V7) ──────────────────────────────────────────────
        with tab1:
            st.caption("매크로 신호 기반으로 TLT / TBF / BIL 중 무엇을 보유할지 결정합니다.")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("##### FFV 내재가치 임계값")
                ffv_lo = st.number_input("FFV 하한 (Undervalued — TLT 진입 허용)", -5.0, 5.0,
                                         float(current_params.get('ffv_lo', -1.0)), 0.1, key=k("ffv_lo"))
                ffv_hi = st.number_input("FFV 상한 (Overvalued — TBF 진입 차단)", -5.0, 5.0,
                                         float(current_params.get('ffv_hi', 1.5)), 0.1, key=k("ffv_hi"))
                st.markdown("---")
                st.markdown("##### TLT 진입 RSI 필터")
                tlt_rsi_max = st.slider("TLT 진입 허용 RSI 최대치", 30, 90,
                                        int(current_params.get('tlt_rsi_max', 60)), key=k("rsi_max"))

            with c2:
                st.markdown("##### 수익률 곡선(YC) 오버라이드")
                yc_inv = st.number_input("역전 임계값 (10Y-2Y, 이하면 TBF 전환)", -1.0, 1.0,
                                         float(current_params.get('yc_inv', -0.3)), 0.01, key=k("yc_inv"))
                yc_steep = st.number_input("스티프닝 모멘텀 임계값 (63일 변화량)", -1.0, 1.0,
                                           float(current_params.get('yc_steep', 0.4)), 0.01, key=k("yc_steep"))
                st.info("수익률 곡선이 스티프닝 국면이면 다른 조건에 관계없이 TLT 우선 매수합니다.")

        # ── Tab 2: Layer 2 (V8) ──────────────────────────────────────────────
        with tab2:
            st.caption("Layer 1이 TLT를 선택하면 TLT 지표로 TMF 스테이지를, TBF를 선택하면 TBF 지표로 TMV 스테이지를 결정합니다.")

            use_leverage = st.toggle("레버리지 전략 활성화 (TLT→TMF / TBF→TMV)",
                                     value=lp.get('use_leverage', True), key=k("use_lev"))

            if not use_leverage:
                st.info("비활성화 상태: Layer 1 결과(TLT/TBF/BIL)를 그대로 사용합니다.")
                lev_params_out = None
            else:
                # 스테이지 비중 안내
                st.markdown("""
| 스테이지 | TLT 구간 | TBF 구간 |
|---|---|---|
| S0 (신호 없음) | TLT 100% | TBF 100% |
| S1 | TMF 30% + TLT 70% | TMV 30% + TBF 70% |
| S2 | TMF 70% + TLT 30% | TMV 70% + TBF 30% |
| S3 | TMF 100% | TMV 100% |
""")
                st.divider()

                col_buy, col_sell = st.columns(2)

                # ── 매수 조건 (buy_signals) ──
                with col_buy:
                    st.markdown("##### 매수 조건 (OR) — 기초자산 기준")
                    st.caption("TLT 지표로 TMF 진입 / TBF 지표로 TMV 진입")
                    buy_signals = []

                    # 조건 1: RSI 과매도 단독
                    with st.expander("매수 신호 1 — RSI 과매도 단독 진입", expanded=True):
                        p1 = lp.get('buy_signals', [{}])[0] if lp.get('buy_signals') else {}
                        rsi_val_1 = st.slider("RSI ≤ (단독 진입)", 10, 70,
                                              int(p1.get('rsi_val', 35)), key=k("b1_rsi"))
                        buy_signals.append({
                            'rsi_val': rsi_val_1, 'rsi_cross': False, 'rsi_inc': False,
                            'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False,
                            'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
                            'use_sar': False, 'bb_lower': False,
                            'use_willr': False, 'willr_val': -80,
                            'di_plus_cross': False, 'di_minus_cross': False,
                            'use_rsi_wait': False, 'rsi_wait_val': 35,
                        })

                    # 조건 2: RSI 골든크로스 + MACD 상승
                    with st.expander("매수 신호 2 — RSI 골든크로스 + MACD 확인", expanded=False):
                        p2 = lp.get('buy_signals', [{}, {}])[1] if len(lp.get('buy_signals', [])) > 1 else {}
                        macd_signal_below_2 = st.checkbox("MACD < Signal (히스토그램 음수 구간)",
                                                           value=p2.get('macd_signal_below', True),
                                                           key=k("b2_macd_below"))
                        macd_inc_2 = st.checkbox("MACD 값 증가 중",
                                                  value=p2.get('macd_inc', True), key=k("b2_macd_inc"))
                        buy_signals.append({
                            'rsi_val': 0, 'rsi_cross': True, 'rsi_inc': False,
                            'macd_inc': macd_inc_2, 'macd_signal_below': macd_signal_below_2,
                            'macd_golden': False,
                            'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
                            'use_sar': False, 'bb_lower': False,
                            'use_willr': False, 'willr_val': -80,
                            'di_plus_cross': False, 'di_minus_cross': False,
                            'use_rsi_wait': False, 'rsi_wait_val': 35,
                        })

                # ── 매도 조건 (sell_signals) ──
                with col_sell:
                    st.markdown("##### 매도 조건 (OR) — 기초자산 기준")
                    st.caption("신호 발생 시 즉시 S0로 다운그레이드")
                    sell_signals = []

                    # 조건 1: RSI 과매수 + 하락 중
                    with st.expander("매도 신호 1 — RSI 과매수 이익실현", expanded=True):
                        sp1 = lp.get('sell_signals', [{}])[0] if lp.get('sell_signals') else {}
                        rsi_sell_1 = st.slider("RSI ≥ (이익실현)", 50, 90,
                                               int(sp1.get('rsi_val', 70)), key=k("s1_rsi"))
                        rsi_dec_1 = st.checkbox("RSI 하락 중 조건 추가",
                                                value=sp1.get('rsi_dec', True), key=k("s1_rsi_dec"))
                        sell_signals.append({
                            'rsi_val': rsi_sell_1, 'rsi_dec': rsi_dec_1, 'rsi_dead': False,
                            'macd_dec': False, 'macd_dead': False, 'macd_signal_above': False,
                            'use_sar': False, 'bb_upper': False,
                            'use_willr': False, 'willr_val': -20,
                            'di_minus_above': False, 'di_minus_cross': False,
                            'use_chandelier': False, 'chandelier_mult': 3.0,
                            'use_rsi_wait': False, 'rsi_wait_val': 70,
                        })

                    # 조건 2: RSI 데드크로스 + MACD 감소
                    with st.expander("매도 신호 2 — RSI 데드크로스 + MACD 반전", expanded=False):
                        sp2 = lp.get('sell_signals', [{}, {}])[1] if len(lp.get('sell_signals', [])) > 1 else {}
                        macd_dec_2 = st.checkbox("MACD 감소 중", value=sp2.get('macd_dec', True), key=k("s2_macd_dec"))
                        macd_sig_above_2 = st.checkbox("MACD > Signal (히스토그램 양수 구간)",
                                                        value=sp2.get('macd_signal_above', True),
                                                        key=k("s2_macd_above"))
                        sell_signals.append({
                            'rsi_val': 0, 'rsi_dec': False, 'rsi_dead': True,
                            'macd_dec': macd_dec_2, 'macd_dead': False,
                            'macd_signal_above': macd_sig_above_2,
                            'use_sar': False, 'bb_upper': False,
                            'use_willr': False, 'willr_val': -20,
                            'di_minus_above': False, 'di_minus_cross': False,
                            'use_chandelier': False, 'chandelier_mult': 3.0,
                            'use_rsi_wait': False, 'rsi_wait_val': 70,
                        })

                st.divider()
                st.markdown("##### 가격 트리거 (스테이지 전환)")
                pc1, pc2 = st.columns(2)
                with pc1:
                    reb_up = st.slider("상승 트리거 reb_up (S업)", 0.5, 5.0,
                                       float(lp.get('buy_reb_up', 0.01)) * 100, 0.5,
                                       format="%.1f%%", key=k("reb_up")) / 100
                with pc2:
                    reb_dn = st.slider("하락 트리거 reb_down (S다운)", 0.5, 5.0,
                                       float(abs(lp.get('buy_reb_down', 0.02))) * 100, 0.5,
                                       format="%.1f%%", key=k("reb_dn")) / 100

                lev_params_out = {
                    'use_leverage': True,
                    'buy_signals':  buy_signals,
                    'sell_signals': sell_signals,
                    's3_protection': [],
                    'panic_buy_signals': [],
                    'buy_reb_up':    reb_up,
                    'buy_reb_down':  -reb_dn,
                    'sell_reb_up':   0.02,
                    'sell_reb_down': -0.02,
                    'use_fixed_reb': True,
                    'use_atr_reb':   False,
                    'cash_ratio':    0.0,
                    'trade_at':      '종가',
                    'panic_ma':      200,
                    'use_panic':     False,
                    'base_asset':    'TLT',
                    'leverage_asset':'TMF',
                }

        layer1_params = {
            'ffv_lo': ffv_lo, 'ffv_hi': ffv_hi,
            'yc_inv': yc_inv, 'yc_steep': yc_steep,
            'tlt_rsi_max': tlt_rsi_max,
        }
        return layer1_params, lev_params_out

    @staticmethod
    def sync_bond_params_to_widgets(params, lev_params=None, key_prefix="bond_"):
        """세션 파라미터를 UI 위젯 키로 강제 동기화 (Apply 시 사용)"""
        def k(base_key): return f"{key_prefix}{base_key}"
        
        # Layer 1
        st.session_state[k("ffv_lo")] = float(params.get('ffv_lo', -1.0))
        st.session_state[k("ffv_hi")] = float(params.get('ffv_hi', 1.5))
        st.session_state[k("rsi_max")] = int(params.get('tlt_rsi_max', 60))
        st.session_state[k("yc_inv")] = float(params.get('yc_inv', -0.3))
        st.session_state[k("yc_steep")] = float(params.get('yc_steep', 0.4))
        
        # Layer 2
        lp = lev_params or {}
        st.session_state[k("use_lev")] = lp.get('use_leverage', True)
        
        # Buy Signals
        b_sigs = lp.get('buy_signals', [])
        if len(b_sigs) > 0:
            st.session_state[k("b1_rsi")] = int(b_sigs[0].get('rsi_val', 35))
        if len(b_sigs) > 1:
            st.session_state[k("b2_macd_below")] = b_sigs[1].get('macd_signal_below', True)
            st.session_state[k("b2_macd_inc")] = b_sigs[1].get('macd_inc', True)
            
        # Sell Signals
        s_sigs = lp.get('sell_signals', [])
        if len(s_sigs) > 0:
            st.session_state[k("s1_rsi")] = int(s_sigs[0].get('rsi_val', 70))
            st.session_state[k("s1_rsi_dec")] = s_sigs[0].get('rsi_dec', True)
        if len(s_sigs) > 1:
            st.session_state[k("s2_macd_dec")] = s_sigs[1].get('macd_dec', True)
            st.session_state[k("s2_macd_above")] = s_sigs[1].get('macd_signal_above', True)
            
        # Rebalance
        st.session_state[k("reb_up")] = float(lp.get('buy_reb_up', 0.01)) * 100
        st.session_state[k("reb_dn")] = float(abs(lp.get('buy_reb_down', 0.02))) * 100

    @staticmethod
    def get_bond_params_from_widgets(key_prefix="bond_"):
        """UI 위젯의 현재 상태를 읽어서 파라미터 번들 생성 (Save 시 사용)"""
        def k(base_key): return f"{key_prefix}{base_key}"
        s = st.session_state
        
        # Layer 1
        params = {
            'ffv_lo': s.get(k("ffv_lo"), -1.0),
            'ffv_hi': s.get(k("ffv_hi"), 1.5),
            'tlt_rsi_max': s.get(k("rsi_max"), 60),
            'yc_inv': s.get(k("yc_inv"), -0.3),
            'yc_steep': s.get(k("yc_steep"), 0.4)
        }
        
        # Layer 2
        use_lev = s.get(k("use_lev"), True)
        
        buy_signals = [
            {
                'rsi_val': s.get(k("b1_rsi"), 35), 'rsi_cross': False, 'rsi_inc': False,
                'macd_inc': False, 'macd_signal_below': False, 'macd_golden': False,
                'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
                'use_sar': False, 'bb_lower': False, 'use_willr': False, 'willr_val': -80,
                'di_plus_cross': False, 'di_minus_cross': False, 'use_rsi_wait': False, 'rsi_wait_val': 35,
            },
            {
                'rsi_val': 0, 'rsi_cross': True, 'rsi_inc': False,
                'macd_inc': s.get(k("b2_macd_inc"), True), 
                'macd_signal_below': s.get(k("b2_macd_below"), True),
                'macd_golden': False, 'use_adx': False, 'adx_op': '<=', 'adx_val': 40,
                'use_sar': False, 'bb_lower': False, 'use_willr': False, 'willr_val': -80,
                'di_plus_cross': False, 'di_minus_cross': False, 'use_rsi_wait': False, 'rsi_wait_val': 35,
            }
        ]
        
        sell_signals = [
            {
                'rsi_val': s.get(k("s1_rsi"), 70), 'rsi_dec': s.get(k("s1_rsi_dec"), True), 'rsi_dead': False,
                'macd_dec': False, 'macd_dead': False, 'macd_signal_above': False,
                'use_sar': False, 'bb_upper': False, 'use_willr': False, 'willr_val': -20,
                'di_minus_above': False, 'di_minus_cross': False, 'use_chandelier': False, 'chandelier_mult': 3.0,
                'use_rsi_wait': False, 'rsi_wait_val': 70,
            },
            {
                'rsi_val': 0, 'rsi_dec': False, 'rsi_dead': True,
                'macd_dec': s.get(k("s2_macd_dec"), True), 'macd_dead': False,
                'macd_signal_above': s.get(k("s2_macd_above"), True),
                'use_sar': False, 'bb_upper': False, 'use_willr': False, 'willr_val': -20,
                'di_minus_above': False, 'di_minus_cross': False, 'use_chandelier': False, 'chandelier_mult': 3.0,
                'use_rsi_wait': False, 'rsi_wait_val': 70,
            }
        ]
        
        lev_params = {
            'use_leverage': use_lev,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_reb_up': s.get(k("reb_up"), 1.0) / 100,
            'buy_reb_down': -s.get(k("reb_dn"), 2.0) / 100,
            'sell_reb_up': 0.02, 'sell_reb_down': -0.02,
            'use_fixed_reb': True, 'use_atr_reb': False, 'cash_ratio': 0.0,
            'trade_at': '종가', 'panic_ma': 200, 'use_panic': False,
            'base_asset': 'TLT', 'leverage_asset': 'TMF'
        }
        
        return params, lev_params
