import streamlit as st

class TesterView:
    """전략 분석기 입력 UI 구성 클래스"""
    @staticmethod
    def render_opt_range(label, p_dict, field_name, min_v, max_v, step=1, opt_mode=False, format=None, key_suffix=None):
        """[신규] 최적화 🎯 체크박스 및 범위 슬라이더를 묶어서 렌더링하는 헬퍼 함수"""
        # field_name: 딕셔너리에서 사용할 기본 키 이름 (예: 'rsi_val')
        # key_suffix: 위젯 키를 고유하게 만들기 위한 접미사 (예: 'b1')
        
        opt_key = f"opt_{field_name}"
        rng_key = f"rng_{field_name}"
        
        # [특별 처리] 뉴스 지표는 rng 키에 _val이 붙는 경우가 있음 (호환성 유지)
        if field_name == "news_q" and rng_key not in p_dict and "rng_news_q_val" in p_dict:
            rng_key = "rng_news_q_val"
        if field_name == "news_z" and rng_key not in p_dict and "rng_news_z_val" in p_dict:
            rng_key = "rng_news_z_val"

        w_suffix = key_suffix if key_suffix else field_name
        w_key = f"opt_chk_{w_suffix}"
        
        # 🎯 체크박스
        is_opt = st.checkbox(f"🎯 {label} 최적화", value=p_dict.get(opt_key, False), key=w_key) if opt_mode else p_dict.get(opt_key, False)
        
        # 범위 데이터
        rng = p_dict.get(rng_key, [min_v, max_v])
        
        if opt_mode and is_opt:
            s_key = f"rng_sld_{w_suffix}"
            rng = st.slider(f"   ㄴ {label} 탐색 범위 설정", min_v, max_v, value=rng, step=step, key=s_key, format=format)
        
        return is_opt, rng

    @staticmethod
    def render_top_part(current_params, ticker_map=None):
        st.subheader("🌐 기본 자산 및 거래 설정")
        c1, c2, c3 = st.columns(3)
        
        base_asset_opts = ["QQQ", "QLD", "TQQQ"]
        leverage_asset_opts = ["QLD", "TQQQ"]
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
            opt_mode = st.session_state.get('opt_mode_active', False)
            
            for i in range(1, 4):
                with st.expander(f"매수 신호 {i}", expanded=False):
                    p = current_params['buy_signals'][i-1] if i <= len(current_params['buy_signals']) else {}
                    if i == 1: # B2: RSI 과매도
                        # RSI 1차 대기
                        use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 {i}", value=p.get('use_rsi_wait', False), key=f"b_rsi_wait_use_{i}", help="RSI가 기준치에 도달하면 즉시 매수하지 않고, 다음 조건이 충족될 때까지 대기 상태를 유지합니다.")
                        rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 {i}", 0, 100, p.get('rsi_wait_val', 35), key=f"b_rsi_wait_val_{i}")
                        # RSI 기준
                        rsi_val = st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 35), key=f"b_rsi_{i}")
                        opt_rsi, rng_rsi = TesterView.render_opt_range("RSI", p, "rsi_val", 10, 60, 1, opt_mode, key_suffix=f"b_rsi_{i}")
                        
                        # 지표들
                        rsi_cross = st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', False), key=f"b_cross_{i}")
                        rsi_inc = st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}")
                        macd_inc = st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', False), key=f"b_m_inc_{i}")
                        macd_signal_below = st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', False), key=f"b_m_below_{i}")
                        macd_golden = st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}")
                        
                        # ADX
                        use_adx = st.checkbox(f"ADX 필터 사용 {i}", value=p.get('use_adx', False), key=f"b_use_adx_{i}")
                        adx_op = st.selectbox(f"ADX 연산자 {i}", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=f"b_adx_op_{i}")
                        adx_val = st.number_input(f"ADX 기준 {i}", 0, 100, p.get('adx_val', 40), key=f"b_adx_{i}")
                        opt_adx, rng_adx = TesterView.render_opt_range("ADX", p, "adx_val", 5, 80, 1, opt_mode, key_suffix=f"b_adx_{i}")
                        
                        bb_lower = st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        
                        # Williams %R
                        use_willr = st.checkbox(f"Williams %R 상향 돌파 {i}", value=p.get('use_willr', False), key=f"b_willr_use_{i}")
                        willr_val = st.number_input(f"W%R 상향 기준값 {i}", -100, 0, p.get('willr_val', -80), key=f"b_willr_val_{i}")
                        opt_willr, rng_willr = TesterView.render_opt_range("Williams %R", p, "willr_val", -95, -50, 1, opt_mode, key_suffix=f"b_willr_{i}")
                        
                        di_plus_above = st.checkbox(f"매수 우세 상태유지 (DI+ > DI-) {i}", value=p.get('di_plus_above', False), key=f"b_di_plus_above_{i}")
                        di_plus_cross = st.checkbox(f"DI+ 골든크로스 순간 (DI+ > DI-) {i}", value=p.get('di_plus_cross', False), key=f"b_di_plus_cross_{i}")
                        use_sar = st.checkbox(f"파라볼릭 SAR 상향 돌파 {i}", value=p.get('use_sar', False), key=f"b_sar_use_{i}")
                        
                        # AI 뉴스 필터
                        use_news_q = st.checkbox(f"AI 뉴스 분위수 필터(Q) {i}", value=p.get('use_news_q', False), key=f"b_news_q_use_{i}")
                        news_q_op = st.selectbox(f"AI 뉴스 분위수 조건식 {i}", [">", ">=", "<", "<="], index=[">", ">=", "<", "<="].index(p.get('news_q_op', '>')), key=f"b_news_q_op_{i}")
                        news_q_val = st.number_input(f"AI 뉴스 분위수 임계값 {i}", 0.0, 1.0, float(p.get('news_q_val', 0.8)), 0.05, key=f"b_news_q_val_{i}")
                        # 분위수는 0.5~0.99 범위가 일반적
                        opt_news_q, rng_news_q = TesterView.render_opt_range("AI 분위수(Q)", p, "news_q", 0.5, 0.99, 0.01, opt_mode, format="%.2f", key_suffix=f"b_news_q_{i}")
                        
                        use_news_z = st.checkbox(f"AI 뉴스 Z-점수 필터(Z) {i}", value=p.get('use_news_z', False), key=f"b_news_z_use_{i}")
                        news_z_op = st.selectbox(f"AI 뉴스 Z-점수 조건식 {i}", [">", ">=", "<", "<="], index=0, key=f"b_news_z_op_{i}")
                        news_z_val = st.number_input(f"AI 뉴스 Z-점수 임계값 {i}", -3.0, 3.0, float(p.get('news_z_val', 1.0)), 0.1, key=f"b_news_z_val_{i}")
                        # Z점수는 -2.0~4.0 범위
                        opt_news_z, rng_news_z = TesterView.render_opt_range("AI Z-점수(Z)", p, "news_z", -1.0, 4.0, 0.1, opt_mode, format="%.1f", key_suffix=f"b_news_z_{i}")

                        buy_signals.append({
                            'use_rsi_wait': use_rsi_wait, 'rsi_wait_val': rsi_wait_val, 'rsi_val': rsi_val,
                            'opt_rsi_val': opt_rsi, 'rng_rsi_val': rng_rsi,
                            'rsi_cross': rsi_cross, 'rsi_inc': rsi_inc, 'macd_inc': macd_inc,
                            'macd_signal_below': macd_signal_below, 'macd_golden': macd_golden,
                            'use_adx': use_adx, 'adx_op': adx_op, 'adx_val': adx_val,
                            'opt_adx_val': opt_adx, 'rng_adx_val': rng_adx,
                            'bb_lower': bb_lower, 'use_willr': use_willr, 'willr_val': willr_val,
                            'opt_willr_val': opt_willr, 'rng_willr_val': rng_willr,
                            'di_plus_above': di_plus_above, 'di_plus_cross': di_plus_cross, 'use_sar': use_sar,
                            'use_news_q': use_news_q, 'news_q_op': news_q_op, 'news_q_val': news_q_val,
                            'opt_news_q': opt_news_q, 'rng_news_q_val': rng_news_q,
                            'use_news_z': use_news_z, 'news_z_op': news_z_op, 'news_z_val': news_z_val,
                            'opt_news_z': opt_news_z, 'rng_news_z_val': rng_news_z
                        })
                    else: # B1: RSI 골든크로스 + MACD 상승
                        # RSI 1차 대기
                        use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 {i}", value=p.get('use_rsi_wait', False), key=f"b_rsi_wait_use_{i}")
                        rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 {i}", 0, 100, p.get('rsi_wait_val', 35), key=f"b_rsi_wait_val_{i}")
                        # RSI 기준
                        rsi_val = st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 0), key=f"b_rsi_{i}")
                        opt_rsi, rng_rsi = TesterView.render_opt_range("RSI", p, "rsi_val", 0, 45, 1, opt_mode, key_suffix=f"b_rsi_{i}")
                        
                        rsi_cross = st.checkbox(f"RSI Golden Cross {i}", value=p.get('rsi_cross', True), key=f"b_cross_{i}")
                        rsi_inc = st.checkbox(f"RSI 증가 {i}", value=p.get('rsi_inc', False), key=f"b_inc_{i}")
                        macd_inc = st.checkbox(f"MACD 증가 {i}", value=p.get('macd_inc', True), key=f"b_m_inc_{i}")
                        macd_signal_below = st.checkbox(f"MACD < Signal {i}", value=p.get('macd_signal_below', True), key=f"b_m_below_{i}")
                        macd_golden = st.checkbox(f"MACD Golden Cross {i}", value=p.get('macd_golden', False), key=f"b_m_golden_{i}")
                        
                        # ADX
                        use_adx = st.checkbox(f"ADX 필터 사용 {i}", value=p.get('use_adx', False), key=f"b_use_adx_{i}")
                        adx_op = st.selectbox(f"ADX 연산자 {i}", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=f"b_adx_op_{i}")
                        adx_val = st.number_input(f"ADX 기준 {i}", 0, 100, p.get('adx_val', 40), key=f"b_adx_{i}")
                        opt_adx, rng_adx = TesterView.render_opt_range("ADX", p, "adx_val", 5, 80, 1, opt_mode, key_suffix=f"b_adx_{i}")
                        
                        bb_lower = st.checkbox(f"Bollinger Lower {i}", value=p.get('bb_lower', False), key=f"b_bb_low_{i}")
                        
                        # Williams %R
                        use_willr = st.checkbox(f"Williams %R 상향 돌파 {i}", value=p.get('use_willr', False), key=f"b_willr_use_{i}")
                        willr_val = st.number_input(f"W%R 상향 기준값 {i}", -100, 0, p.get('willr_val', -80), key=f"b_willr_val_{i}")
                        opt_willr, rng_willr = TesterView.render_opt_range("Williams %R", p, "willr_val", -95, -50, 1, opt_mode, key_suffix=f"b_willr_{i}")
                        
                        di_plus_above = st.checkbox(f"매수 우세 상태유지 (DI+ > DI-) {i}", value=p.get('di_plus_above', False), key=f"b_di_plus_above_{i}")
                        di_plus_cross = st.checkbox(f"DI+ 골든크로스 순간 (DI+ > DI-) {i}", value=p.get('di_plus_cross', False), key=f"b_di_plus_cross_{i}")
                        use_sar = st.checkbox(f"파라볼릭 SAR 상향 돌파 {i}", value=p.get('use_sar', False), key=f"b_sar_use_{i}")
                        
                        # AI 뉴스 필터
                        use_news_q = st.checkbox(f"AI 뉴스 분위수 필터(Q) {i}", value=p.get('use_news_q', False), key=f"b_news_q_use_{i}")
                        news_q_op = st.selectbox(f"AI 뉴스 분위수 조건식 {i}", [">", ">=", "<", "<="], index=[">", ">=", "<", "<="].index(p.get('news_q_op', '>')), key=f"b_news_q_op_{i}")
                        news_q_val = st.number_input(f"AI 뉴스 분위수 임계값 {i}", 0.0, 1.0, float(p.get('news_q_val', 0.8)), 0.05, key=f"b_news_q_val_{i}")
                        opt_news_q, rng_news_q = TesterView.render_opt_range("AI 분위수(Q)", p, "news_q", 0.5, 0.99, 0.01, opt_mode, format="%.2f", key_suffix=f"b_news_q_{i}")
                        
                        use_news_z = st.checkbox(f"AI 뉴스 Z-점수 필터(Z) {i}", value=p.get('use_news_z', False), key=f"b_news_z_use_{i}")
                        news_z_op = st.selectbox(f"AI 뉴스 Z-점수 조건식 {i}", [">", ">=", "<", "<="], index=0, key=f"b_news_z_op_{i}")
                        news_z_val = st.number_input(f"AI 뉴스 Z-점수 임계값 {i}", -3.0, 3.0, float(p.get('news_z_val', 1.0)), 0.1, key=f"b_news_z_val_{i}")
                        opt_news_z, rng_news_z = TesterView.render_opt_range("AI Z-점수(Z)", p, "news_z", -1.0, 4.0, 0.1, opt_mode, format="%.1f", key_suffix=f"b_news_z_{i}")

                        buy_signals.append({
                            'use_rsi_wait': use_rsi_wait, 'rsi_wait_val': rsi_wait_val, 'rsi_val': rsi_val,
                            'opt_rsi_val': opt_rsi, 'rng_rsi_val': rng_rsi,
                            'rsi_cross': rsi_cross, 'rsi_inc': rsi_inc, 'macd_inc': macd_inc,
                            'macd_signal_below': macd_signal_below, 'macd_golden': macd_golden,
                            'use_adx': use_adx, 'adx_op': adx_op, 'adx_val': adx_val,
                            'opt_adx_val': opt_adx, 'rng_adx_val': rng_adx,
                            'bb_lower': bb_lower, 'use_willr': use_willr, 'willr_val': willr_val,
                            'opt_willr_val': opt_willr, 'rng_willr_val': rng_willr,
                            'di_plus_above': di_plus_above, 'di_plus_cross': di_plus_cross, 'use_sar': use_sar,
                            'use_news_q': use_news_q, 'news_q_op': news_q_op, 'news_q_val': news_q_val,
                            'opt_news_q': opt_news_q, 'rng_news_q_val': rng_news_q,
                            'use_news_z': use_news_z, 'news_z_op': news_z_op, 'news_z_val': news_z_val,
                            'opt_news_z': opt_news_z, 'rng_news_z_val': rng_news_z
                        })
        
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            # [수정] 매도 신호를 4개까지 설정할 수 있도록 반복문 범위 확장
            for i in range(1, 5):
                with st.expander(f"매도 신호 {i}", expanded=False):
                    p = current_params['sell_signals'][i-1] if i <= len(current_params['sell_signals']) else {}
                    opt_mode = st.session_state.get('opt_mode_active', False)
                    
                    # RSI 1차 대기
                    use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 {i}", value=p.get('use_rsi_wait', False), key=f"s_rsi_wait_use_{i}", help="RSI가 기준치에 도달하면 즉시 매도하지 않고, 다음 조건이 충족될 때까지 대기 상태를 유지합니다.")
                    rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 {i}", 0, 100, p.get('rsi_wait_val', 70), key=f"s_rsi_wait_val_{i}", help="1차 대기 상태로 진입하기 위한 RSI 기준값입니다.")
                    # RSI 기준
                    rsi_val = st.number_input(f"RSI 기준 {i}", 0, 100, p.get('rsi_val', 70 if i==1 else 0), key=f"s_rsi_{i}", help="매도 실행을 위한 핵심 RSI 기준값입니다.")
                    opt_rsi, rng_rsi = TesterView.render_opt_range("RSI", p, "rsi_val", 40, 95, 1, opt_mode, key_suffix=f"s_rsi_{i}")

                    rsi_dead = st.checkbox(f"RSI Dead Cross {i}", value=p.get('rsi_dead', i==2), key=f"s_dead_{i}", help="RSI가 기준선을 하향 돌파할 때 신호를 발생시킵니다.")
                    rsi_dec = st.checkbox(f"RSI 감소 {i}", value=p.get('rsi_dec', i==1), key=f"s_dec_{i}", help="RSI 지표가 전날보다 하락 중일 때 조건을 충족합니다.")
                    macd_dec = st.checkbox(f"MACD 감소 {i}", value=p.get('macd_dec', i==2), key=f"s_m_dec_{i}", help="MACD 선이 전날보다 하락 중일 때 조건을 충족합니다.")
                    macd_signal_above = st.checkbox(f"MACD > Signal {i}", value=p.get('macd_signal_above', i==2), key=f"s_m_above_{i}", help="MACD 선이 시그널 선보다 위에 있을 때 조건을 충족합니다.")
                    macd_dead = st.checkbox(f"MACD Dead Cross {i}", value=p.get('macd_dead', i==3), key=f"s_m_dead_{i}", help="MACD 선이 시그널 선을 하향 돌파할 때 신호를 발생시킵니다.")
                    di_minus_above = st.checkbox(f"매도 우세 상태유지 (DI- > DI+) {i}", value=p.get('di_minus_above', False), key=f"s_di_{i}", help="매도 세력(DI-)이 매수 세력(DI+)보다 높은 상태를 요구합니다.")
                    di_minus_cross = st.checkbox(f"DI- 데드크로스 순간 (DI- > DI+) {i}", value=p.get('di_minus_cross', False), key=f"s_di_minus_cross_{i}", help="DI-가 DI+를 상향 돌파하는 순간 신호를 발생시킵니다.")
                    bb_upper = st.checkbox(f"Bollinger Upper {i}", value=p.get('bb_upper', False), key=f"s_bb_up_{i}", help="주가가 볼린저 밴드 상단선 이상일 때 과매수로 판단합니다.")
                    # [신규] 샹들리에 엑시트 (ATR 고점 추적) 조건 추가
                    use_chan = st.checkbox(f"샹들리에 엑시트 (고점방어) {i}", value=p.get('use_chandelier', i==4), key=f"s_chan_use_{i}", help="고점 대비 일정 수준 하락 시 수익 보존 및 손실 방어를 위해 매도합니다.")
                    chan_mult = st.number_input(f"샹들리에 변동성 승수 {i}", 0.1, 10.0, float(p.get('chandelier_mult', 3.0)), 0.1, key=f"s_chan_mult_{i}", help="낮을수록 민감하게 반응합니다. 보통 3.0을 사용합니다.")
                    opt_chan, rng_chan = TesterView.render_opt_range("샹들리에", p, "chandelier_mult", 1.0, 5.0, 0.1, opt_mode, format="%.1f", key_suffix=f"s_chan_{i}")
                    # [신규] 파라볼릭 SAR 데드크로스 조건 추가
                    use_sar = st.checkbox(f"파라볼릭 SAR 하향 이탈 {i}", value=p.get('use_sar', False), key=f"s_sar_use_{i}", help="SAR 점이 하락 추세로 전환될 때 신호를 발생시킵니다.")
                    # [신규] Williams %R 하향 이탈 추가
                    use_willr = st.checkbox(f"Williams %R 하향 이탈 {i}", value=p.get('use_willr', False), key=f"s_willr_use_{i}", help="Williams %R이 과열권에서 이탈하며 하향 돌파할 때 신호를 발생시킵니다.")
                    willr_val = st.number_input(f"W%R 하향 기준값 {i}", -100, 0, p.get('willr_val', -20), key=f"s_willr_val_{i}", help="보통 -20을 사용합니다.")
                    opt_willr, rng_willr = TesterView.render_opt_range("Williams %R", p, "willr_val", -50, -5, 1, opt_mode, key_suffix=f"s_willr_{i}")
                    
                    use_news_q = st.checkbox(f"AI 뉴스 분위수 필터(Q) {i}", value=p.get('use_news_q', False), key=f"s_news_q_use_{i}", help="최근 60일 뉴스 데이터 중 현재 점수의 상대적 위치(0~1)를 나타냅니다. 0.9는 상위 10% 안에 드는 강력한 긍정을 의미합니다.")
                    news_q_op = st.selectbox(f"AI 뉴스 분위수 조건식 {i}", [">", ">=", "<", "<="], index=[">", ">=", "<", "<="].index(p.get('news_q_op', '<')), key=f"s_news_q_op_{i}")
                    news_q_val = st.number_input(f"AI 뉴스 분위수 임계값 {i} (0.0~1.0)", 0.0, 1.0, float(p.get('news_q_val', 0.2)), 0.05, key=f"s_news_q_val_{i}")
                    opt_news_q, rng_news_q = TesterView.render_opt_range("AI 분위수(Q)", p, "news_q", 0.01, 0.5, 0.01, opt_mode, format="%.2f", key_suffix=f"s_news_q_{i}")
                    
                    use_news_z = st.checkbox(f"AI 뉴스 Z-점수 필터(Z) {i}", value=p.get('use_news_z', False), key=f"s_news_z_use_{i}", help="최근 평균 대비 현재 심리가 얼마나 급격하게 변했는는지를 나타냅니다. 2.0 이상이면 평소보다 매우 강한 호재, -2.0 이하면 매우 강한 악재를 의미합니다.")
                    news_z_op = st.selectbox(f"AI 뉴스 Z-점수 조건식 {i}", [">", ">=", "<", "<="], index=2 if p.get('news_z_op', '<') == '<' else 0, key=f"s_news_z_op_{i}")
                    news_z_val = st.number_input(f"AI 뉴스 Z-점수 임계값 {i} (-3.0~3.0)", -3.0, 3.0, float(p.get('news_z_val', -1.5)), 0.1, key=f"s_news_z_val_{i}")
                    opt_news_z, rng_news_z = TesterView.render_opt_range("AI Z-점수(Z)", p, "news_z", -4.0, 1.0, 0.1, opt_mode, format="%.1f", key_suffix=f"s_news_z_{i}")

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
                        'opt_news_z': opt_news_z, 'rng_news_z_val': rng_news_z
                    })
            
            st.write("---")
            st.subheader("🛡️ S3(레버리지 100%) 보호 설정")
            st.caption("각 보호 신호 **내부 항목은 AND(모두 충족)**, **신호 간(1,2,3)은 OR(하나라도 충족)**로 작동합니다.")
            
            s3_p_list = current_params.get('s3_protection', [{}, {}])
            while len(s3_p_list) < 2: s3_p_list.append({})
            
            s3_protection = []
            for i in range(1, 4):
                with st.expander(f"보호 신호 {i}", expanded=False):
                    p = s3_p_list[i-1] if i <= len(s3_p_list) else {}
                    use_drop = st.checkbox(f"일일 급락 매도 제어 {i}", value=p.get('use_daily_drop', False), key=f"s_drop_use_{i}", help="전날 대비 자산 가격이 크게 하락할 경우 레버리지를 축소합니다.")
                    drop_lim = st.number_input(f"급락 기준 (%) {i}", -10.0, 0.0, p.get('drop_limit', -3.0), 0.1, format="%.1f", key=f"s_drop_lim_{i}", help="급락으로 판단할 하락 비율(%)입니다.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        use_ma60 = st.checkbox(f"60일 이평선 하회 {i}", value=p.get('use_ma60', False), key=f"s_ma60_use_{i}", help="60일 이평선 아래로 내려갈 때(추세 붕괴) 보호 로직을 가동합니다.")
                        ma60_lim = st.number_input(f"60일 기준 (%) {i}", -50.0, 50.0, p.get('ma60_limit', 0.0), 0.1, format="%.1f", key=f"s_ma60_lim_{i}")
                    with col2:
                        use_ma200 = st.checkbox(f"200일 이평선 하회 {i}", value=p.get('use_ma200', False), key=f"s_ma200_use_{i}", help="200일 이평선 아래로 내려갈 때(장기 추세 붕괴) 보호 로직을 가동합니다.")
                        ma200_lim = st.number_input(f"200일 기준 (%) {i}", -50.0, 50.0, p.get('ma200_limit', 0.0), 0.1, format="%.1f", key=f"s_ma200_lim_{i}")
                    
                    use_vj = st.checkbox(f"VXN 급등 시그널 사용 {i}", value=p.get('use_vxn_jump', p.get('use_vix_jump', False)), key=f"s_vj_use_{i}", help="변동성 지수(VXN)가 단기 급등하여 심리 지표가 악화될 때 보호 로직을 가동합니다.")
                    vj_val = st.number_input(f"VXN 급등 기준 (%) {i}", 0.0, 100.0, p.get('vxn_jump', p.get('vix_jump', 15.0)), 0.1, format="%.1f", key=f"s_vj_{i}", help="변동성 지수의 전날 대비 기습 상승 기준입니다.")
                    use_gap = st.checkbox(f"패닉 시가 갭 발생 {i} (-3% 이하)", value=p.get('use_gap_down', False), key=f"s_gap_use_{i}", help="당일 시가가 전날 종가 대비 크게 하향하여 패닉이 발생할 때 보호 로직을 가동합니다.")
                    gap_lim = st.number_input(f"시가 갭 기준 (%) {i}", -10.0, 0.0, p.get('gap_limit', -3.0), 0.1, format="%.1f", key=f"s_gap_lim_{i}")
                    use_acc = st.checkbox(f"급격한 하락 발생 {i} (-7% 이하)", value=p.get('use_drop_acc', False), key=f"s_acc_use_{i}", help="연속적인 하락으로 인해 하락폭이 가팔라질 때 보호 로직을 가동합니다.")
                    acc_lim = st.number_input(f"누적 하락 기준 (%) {i}", -20.0, 0.0, p.get('acc_limit', -7.0), 0.1, format="%.1f", key=f"s_acc_lim_{i}")
                    
                    # [신규] S3 보호용 샹들리에 엑시트 옵션 추가
                    use_chan = st.checkbox(f"S3 샹들리에 엑시트 사용 {i}", value=p.get('use_chandelier', False), key=f"s3_chan_use_{i}", help="보호 모드 내에서도 ATR 손절선을 적용합니다.")
                    chan_mult = st.number_input(f"S3 샹들리에 승수 {i}", 0.1, 10.0, float(p.get('chandelier_mult', 3.0)), 0.1, key=f"s3_chan_mult_{i}")
                    
                    use_exit = st.checkbox(f"전량매도 (S0로 이동) {i}", value=p.get('use_exit_all', False), key=f"s_exit_all_{i}", help="보호 조건 충족 시 비중 축소가 아닌, 전량 현금화(S0 단계)를 단행합니다.")
                    
                    s3_protection.append({
                        'use_daily_drop': use_drop, 'drop_limit': drop_lim,
                        'use_ma60': use_ma60, 'ma60_limit': ma60_lim,
                        'use_ma200': use_ma200, 'ma200_limit': ma200_lim,
                        'use_vxn_jump': use_vj, 'vxn_jump': vj_val,
                        'use_gap_down': use_gap, 'gap_limit': gap_lim,
                        'use_drop_acc': use_acc, 'acc_limit': acc_lim,
                        'use_chandelier': use_chan, 'chandelier_mult': chan_mult,
                        'use_exit_all': use_exit
                    })
        
        return {
            'buy_signals': buy_signals, 'sell_signals': sell_signals, 's3_protection': s3_protection,
            'base_asset': base_asset, 'leverage_asset': leverage_asset, 'trade_at': trade_at
        }

    @staticmethod
    def render_bottom_part(current_params, ticker_map=None, reb_mode=None):
        # 폼 외부(HistoryView)에서 선택된 핵심 모드 상태를 가져옴
        use_fixed = current_params.get('use_fixed_reb', True)
        use_atr = current_params.get('use_atr_reb', False)
        
        opt_mode = st.session_state.get('opt_mode_active', False)
        
        c1, c2, c3 = st.columns(3)
        
        # (1) 고정 비율 설정 섹션
        with c1:
            st.markdown("##### 🔹 고정 비율 세부 설정")
            # 매수 상향
            buy_reb_up = st.number_input("매수 상향 (%)", 0.0, 100.0, float(current_params.get('buy_reb_up', 0.02)) * 100, 0.1, key="r_b_u", disabled=not use_fixed) / 100.0
            opt_rbu, rng_rbu = TesterView.render_opt_range("매수 상향", current_params, "buy_reb_up", 0.1, 5.0, 0.1, opt_mode and use_fixed)
            
            # 매수 하향
            buy_reb_down = st.number_input("매수 하향 (%)", -100.0, 0.0, float(current_params.get('buy_reb_down', -0.07)) * 100, 0.1, key="r_b_d", disabled=not use_fixed) / 100.0
            opt_rbd, rng_rbd = TesterView.render_opt_range("매수 하향", current_params, "buy_reb_down", -15.0, -1.0, 0.1, opt_mode and use_fixed)
            
            # 매도 상향
            sell_reb_up = st.number_input("매도 상향 (%)", 0.0, 100.0, float(current_params.get('sell_reb_up', 0.03)) * 100, 0.1, key="r_s_u", disabled=not use_fixed) / 100.0
            opt_rsu, rng_rsu = TesterView.render_opt_range("매도 상향", current_params, "sell_reb_up", 0.5, 10.0, 0.1, opt_mode and use_fixed)
            
            # 매도 하향
            sell_reb_down = st.number_input("매도 하향 (%)", -100.0, 0.0, float(current_params.get('sell_reb_down', -0.035)) * 100, 0.1, key="r_s_d", disabled=not use_fixed) / 100.0
            opt_rsd, rng_rsd = TesterView.render_opt_range("매도 하향", current_params, "sell_reb_down", -10.0, -0.5, 0.1, opt_mode and use_fixed)
            
            fixed_params = {
                'buy_reb_up': buy_reb_up, 'opt_buy_reb_up': opt_rbu, 'rng_buy_reb_up': rng_rbu,
                'buy_reb_down': buy_reb_down, 'opt_buy_reb_down': opt_rbd, 'rng_buy_reb_down': rng_rbd,
                'sell_reb_up': sell_reb_up, 'opt_sell_reb_up': opt_rsu, 'rng_sell_reb_up': rng_rsu,
                'sell_reb_down': sell_reb_down, 'opt_sell_reb_down': opt_rsd, 'rng_sell_reb_down': rng_rsd
            }

        # (2) ATR 변동성 설정 섹션
        with c2:
            st.markdown("##### 🔸 ATR 변동성 세부 설정")
            # 매수 상향 (불타기)
            atr_m_buy_up = st.number_input("매수 상향 승수 (불타기)", 0.1, 25.0, float(current_params.get('atr_mult_buy_up', 10.0)), 0.1, key="atr_m_buy_up", disabled=not use_atr)
            opt_amb_u, rng_amb_u = TesterView.render_opt_range("ATR 매수 상향", current_params, "atr_m_buy_up", 1.0, 20.0, 0.1, opt_mode and use_atr)
            
            # 매수 하향 (물타기)
            atr_m_buy_down = st.number_input("매수 하향 승수 (물타기)", 0.1, 15.0, float(current_params.get('atr_mult_buy_down', 1.5)), 0.1, key="atr_m_buy_down", disabled=not use_atr)
            opt_amb_d, rng_amb_d = TesterView.render_opt_range("ATR 매수 하향", current_params, "atr_m_buy_down", 0.5, 5.0, 0.1, opt_mode and use_atr)
            
            # 매도 승수
            atr_m_sell = st.number_input("매도 승수 (K2)", 0.1, 15.0, float(current_params.get('atr_mult_sell', 3.0)), 0.1, key="atr_m_sell", disabled=not use_atr)
            opt_ams, rng_ams = TesterView.render_opt_range("ATR 매도 승수", current_params, "atr_m_sell", 1.0, 6.0, 0.1, opt_mode and use_atr)
            
            # ATR 기준일
            atr_p_buy = st.selectbox("매수 ATR 기준일", [14, 20], index=0 if current_params.get('atr_period_buy', 20) == 14 else 1, key="atr_p_buy", disabled=not use_atr)
            atr_p_sell = st.selectbox("매도 ATR 기준일", [14, 20], index=1 if current_params.get('atr_period_sell', 20) == 20 else 0, key="atr_p_sell", disabled=not use_atr)
            
            atr_params = {
                'atr_mult_buy_up': atr_m_buy_up, 'opt_atr_m_buy_up': opt_amb_u, 'rng_atr_m_buy_up': rng_amb_u,
                'atr_mult_buy_down': atr_m_buy_down, 'opt_atr_m_buy_down': opt_amb_d, 'rng_atr_m_buy_down': rng_amb_d,
                'atr_mult_sell': atr_m_sell, 'opt_atr_m_sell': opt_ams, 'rng_atr_m_sell': rng_ams,
                'atr_period_buy': atr_p_buy, 'atr_period_sell': atr_p_sell
            }

        # (3) 공통 설정 및 파라미터 취합
        with c3:
            st.markdown("##### 🧱 공통 설정")
            cash_ratio_pct = st.number_input("현금 유지 비율(%)", 0.0, 100.0, float(current_params.get('cash_ratio_pct', 0.0)), 1.0, key="conf_cash")
            
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
                'use_panic': c1.toggle("하락장 매수 지연 (Panic Mode)", value=current_params.get('use_panic', True), key="use_panic_chk"),
                'panic_ma': c1.selectbox("지연 기준 이평선", [60, 120, 200], index=[60,120,200].index(current_params.get('panic_ma', 200)), key="panic_ma_sel"),
                'use_vxn_safety': c2.toggle("VXN 대피 시스템 (Safety)", value=current_params.get('use_vxn_safety', current_params.get('use_vix_safety', True)), key="use_vxn_safety_chk"),
                'vxn_exit': c2.slider("대피 VXN 레벨", 15.0, 45.0, float(current_params.get('vxn_exit', current_params.get('vix_exit', 28.0))), step=0.5, key="vxn_exit_sli"),
                'use_rsi_turbo': c2.toggle("RSI 과매도 가속 (Turbo)", value=current_params.get('use_rsi_turbo', True), key="use_rsi_turbo_chk"),
                'rsi_turbo': c2.slider("가속 RSI 레벨", 20.0, 45.0, float(current_params.get('rsi_turbo', 31.0)), step=0.5, key="rsi_turbo_sli"),
                'use_sl_control': st.toggle("손절제어 사용 (Stop-Loss Control)", value=current_params.get('use_sl_control', False), key="use_sl_control_chk"),
                'sl_control_limit': st.slider("손절제어 임계값 (%)", -50.0, 0.0, float(current_params.get('sl_control_limit', -15.0)), step=1.0, key="sl_control_lim_sli"),
                'use_set_sl': st.toggle("개별 거래 세트 기준 손절 라인 사용", value=current_params.get('use_set_sl', False), key="use_set_sl_chk"),
                'set_sl_limit': st.slider("거래 세트별 손절 라인 (%)", -50.0, 0.0, float(current_params.get('set_sl_limit', -15.0)), step=1.0, key="set_sl_lim_sli"),
                'panic_rsi_s1': current_params.get('panic_rsi_s1', 27),
                'panic_rsi_s2': current_params.get('panic_rsi_s2', 28),
                'panic_rsi_s3': current_params.get('panic_rsi_s3', 30)
            }
            
            st.write("**하락장 매수 지연 해제 조건 (Panic Buy Signals)**")
            
            panic_buy_signals_list = []
            for i in range(1, 4):
                with st.expander(f"하락장 매수 신호 S{i}", expanded=False):
                    p_list = current_params.get('panic_buy_signals', [])
                    p = p_list[i-1] if i <= len(p_list) else {}
                    
                    use_rsi_wait = st.checkbox(f"RSI 1차 대기 사용 (S{i})", value=p.get('use_rsi_wait', False), key=f"pb_rsi_wait_use_{i}")
                    rsi_wait_val = st.number_input(f"대기 진입 RSI 기준 (S{i})", 0, 100, p.get('rsi_wait_val', 35), key=f"pb_rsi_wait_val_{i}")
                    rsi_val = st.number_input(f"RSI 기준 (S{i})", 0, 100, p.get('rsi_val', 27 if i==1 else (28 if i==2 else 30)), key=f"pb_rsi_{i}")
                    opt_rsi, rng_rsi = TesterView.render_opt_range(f"RSI (S{i})", p, "rsi_val", 15, 50, 1, opt_mode, key_suffix=f"pb_rsi_{i}")
                    
                    rsi_cross = st.checkbox(f"RSI Golden Cross (S{i})", value=p.get('rsi_cross', False), key=f"pb_cross_{i}")
                    rsi_inc = st.checkbox(f"RSI 증가 (S{i})", value=p.get('rsi_inc', False), key=f"pb_inc_{i}")
                    macd_inc = st.checkbox(f"MACD 증가 (S{i})", value=p.get('macd_inc', False), key=f"pb_m_inc_{i}")
                    macd_signal_below = st.checkbox(f"MACD < Signal (S{i})", value=p.get('macd_signal_below', False), key=f"pb_m_below_{i}")
                    macd_golden = st.checkbox(f"MACD Golden Cross (S{i})", value=p.get('macd_golden', False), key=f"pb_m_golden_{i}")
                    use_adx = st.checkbox(f"ADX 필터 사용 (S{i})", value=p.get('use_adx', False), key=f"pb_use_adx_{i}")
                    adx_op = st.selectbox(f"ADX 연산자 (S{i})", ["<=", ">="], index=0 if p.get('adx_op', "<=") == "<=" else 1, key=f"pb_adx_op_{i}")
                    adx_val = st.number_input(f"ADX 기준 (S{i})", 0, 100, p.get('adx_val', 40), key=f"pb_adx_{i}")
                    opt_adx, rng_adx = TesterView.render_opt_range(f"ADX (S{i})", p, "adx_val", 5, 80, 1, opt_mode, key_suffix=f"pb_adx_{i}")
                    
                    bb_lower = st.checkbox(f"Bollinger Lower (S{i})", value=p.get('bb_lower', False), key=f"pb_bb_low_{i}")
                    use_willr = st.checkbox(f"Williams %R 상향 돌파 (S{i})", value=p.get('use_willr', False), key=f"pb_willr_use_{i}")
                    willr_val = st.number_input(f"W%R 상향 기준값 (S{i})", -100, 0, p.get('willr_val', -80), key=f"pb_willr_val_{i}")
                    opt_willr, rng_willr = TesterView.render_opt_range(f"W%R (S{i})", p, "willr_val", -95, -50, 1, opt_mode, key_suffix=f"pb_willr_{i}")
                    
                    di_plus_above = st.checkbox(f"매수 우세 상태유지 (DI+ > DI-) (S{i})", value=p.get('di_plus_above', False), key=f"pb_di_plus_above_{i}")
                    di_plus_cross = st.checkbox(f"DI+ 골든크로스 순간 (DI+ > DI-) (S{i})", value=p.get('di_plus_cross', False), key=f"pb_di_plus_cross_{i}")
                    use_sar = st.checkbox(f"파라볼릭 SAR 상향 돌파 (S{i})", value=p.get('use_sar', False), key=f"pb_sar_use_{i}")
                    
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
            
        return {**reb_params, **smart_config}

    @staticmethod
    def render_period_selector():
        """[신규] 분석 대상 기간 선택 UI (매매 전략 설정 탭 전용)"""
        import datetime
        st.markdown("##### 📅 분석 대상 기간 설정")
        col1, col2 = st.columns([1.5, 2])
        
        period_opts = ["전체 (2010~)", "최근 5년", "최근 3년", "최근 1년", "직접 입력"]
        
        # 세션 상태 초기화
        if 'period_sel_type' not in st.session_state:
            st.session_state.period_sel_type = "전체 (2010~)"
        
        sel_type = col1.selectbox("기간 프리셋", period_opts, index=period_opts.index(st.session_state.period_sel_type), key="period_sel_type_box")
        st.session_state.period_sel_type = sel_type
        
        end_date = datetime.date.today()
        start_date = datetime.date(2010, 1, 1)
        
        if sel_type == "최근 5년":
            start_date = end_date - datetime.timedelta(days=5*365)
        elif sel_type == "최근 3년":
            start_date = end_date - datetime.timedelta(days=3*365)
        elif sel_type == "최근 1년":
            start_date = end_date - datetime.timedelta(days=1*365)
        elif sel_type == "직접 입력":
            c1, c2 = col2.columns(2)
            start_date = c1.date_input("시작일", value=datetime.date(2023, 1, 1), key="period_custom_start")
            end_date = c2.date_input("종료일", value=end_date, key="period_custom_end")
        
        if sel_type != "직접 입력":
            col2.info(f"선택 범위: **{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}**")
            
        return start_date, end_date

    @staticmethod
    def render_summary_dashboard(results_df):
        """[신규] 실시간 성과 요약 대시보드 (매매 전략 설정 탭 전용)"""
        if results_df is None or results_df.empty:
            st.warning("📊 시뮬레이션 결과가 없습니다.")
            return

        # 핵심 지표 계산
        total_ret = (results_df['Value'].iloc[-1] / 10000.0 - 1) * 100
        days = (results_df.index[-1] - results_df.index[0]).days
        years = days / 365.25
        cagr = ((results_df['Value'].iloc[-1] / 10000.0) ** (1/years) - 1) * 100 if years > 0 else 0
        
        roll_max = results_df['Value'].cummax()
        drawdown = (results_df['Value'] - roll_max) / roll_max
        mdd = drawdown.min() * 100
        
        # UI 렌더링
        st.markdown("##### 🚀 실시간 시뮬레이션 성과 요약")
        m1, m2, m3 = st.columns(3)
        m1.metric("누적 수익률", f"{total_ret:+.1f}%")
        m2.metric("연평균 수익률(CAGR)", f"{cagr:.1f}%")
        m3.metric("최대 낙폭(MDD)", f"{mdd:.1f}%")

