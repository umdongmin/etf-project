import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @st.cache_data(ttl=3600, show_spinner=False)
    @staticmethod
    def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_date, end_date, params=None, trade_at="종가", smart_params=None, salt=None):
        if base_asset not in data_dict: return pd.DataFrame(), pd.DataFrame()
        base_df = data_dict[base_asset]
        combined = base_df[['Close', 'RSI']].copy()
        
        # [추가] 시가 데이터 확보
        if 'Open' in base_df.columns:
            combined['Open'] = base_df['Open']
        else:
            combined['Open'] = combined['Close'] # 시가 없으면 종가 대체
        
        fg_clean = fg_df[~fg_df.index.duplicated(keep='first')]
        vix_clean = vix_df[~vix_df.index.duplicated(keep='first')]
        combined = combined.join(fg_clean['FearGreed'].rename('FG'), how='left')
        combined = combined.join(vix_clean['Close'].rename('VIX'), how='left')
        
        # [신규] 스토캐스틱 데이터 병합
        if 'STOCH_K' in base_df.columns:
            combined['STOCH_K'] = base_df['STOCH_K']
            combined['STOCH_D'] = base_df['STOCH_D']
            
        # [신규] VVIX와 PCCR 데이터 병합
        if 'VVIX' in vix_clean.columns:
            combined = combined.join(vix_clean['VVIX'], how='left')
        if 'PCCR' in vix_clean.columns:
            combined = combined.join(vix_clean['PCCR'], how='left')
        
        macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
        signal_col = [c for c in combined.columns if 'MACDs_' in c]
        if macd_col and signal_col:
            combined['MACD'], combined['Signal_Line'] = combined[macd_col[0]], combined[signal_col[0]]
        else:
            macd = ta.macd(combined['Close'])
            if macd is not None:
                combined = pd.concat([combined, macd], axis=1)
                combined['MACD'], combined['Signal_Line'] = combined['MACD_12_26_9'], combined['MACDs_12_26_9']
        
        combined['MACD_Hist'] = combined['MACD'] - combined['Signal_Line']
        combined['Prev2_MACD_Hist'] = combined['MACD_Hist'].shift(2)
        
        combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
        combined['SMA200'] = combined['Close'].rolling(window=200).mean()
        combined['SMA60'] = combined['Close'].rolling(window=60).mean()
        
        # [신규] 사용자 정의 Panic MA 계산 (예: SMA120 등)
        if smart_params and 'panic_ma' in smart_params:
            p_ma = smart_params['panic_ma']
            if f'SMA{p_ma}' not in combined.columns:
                combined[f'SMA{p_ma}'] = combined['Close'].rolling(window=p_ma).mean()
        
        # [신규] ADX 및 DI 지표 계산 추가 (시그널용)
        adx_df = ta.adx(base_df['High'], base_df['Low'], base_df['Close'], length=14)
        if adx_df is not None:
            combined = pd.concat([combined, adx_df], axis=1)
            combined['ADX'] = combined['ADX_14']
            combined['DMP'] = combined['DMP_14']
            combined['DMN'] = combined['DMN_14']
        else:
            combined['ADX'], combined['DMP'], combined['DMN'] = 0, 0, 0
        
        # [신규] 급락 진단용 지표 계산 (Base Asset 기준)
        # Drop_Acc: 2거래일 누적 수익률 (최근 하강 가속도 측정용)
        combined['Drop_Acc'] = combined['Close'].pct_change(periods=2) * 100
        # Gap_Down: 전일 종가 대비 당일 시가 갭 (패닉 시가 측정용)
        combined['Gap_Down'] = (combined['Open'] / combined['Close'].shift(1) - 1) * 100
        
        # [신규] ATR 지표 계산 (20일 기준, 기초자산 QQQ 등 타겟)
        combined['ATR'] = ta.atr(base_df['High'], base_df['Low'], base_df['Close'], length=20)
        combined['ATR'] = combined['ATR'].ffill().fillna(0)
        
        cols_to_shift = ['RSI', 'RSI_SMA', 'MACD', 'Signal_Line', 'SMA200', 'SMA60', 'MACD_Hist', 'ADX', 'DMP', 'DMN', 'VIX', 'Drop_Acc', 'Gap_Down', 'ATR']
        if 'STOCH_K' in combined.columns:
            cols_to_shift += ['STOCH_K', 'STOCH_D']
            
        for col in cols_to_shift:
            combined[f'Prev_{col}'] = combined[col].shift(1)
        combined['Prev_ADX'] = combined['ADX'].shift(1)
        combined['Prev_Close'] = combined['Close'].shift(1)
        
        combined['FG'] = combined['FG'].ffill().fillna(50)
        combined['VIX'] = combined['VIX'].ffill().fillna(15)
        # [신규] VVIX, PCCR 결측치 처리
        if 'VVIX' in combined.columns: combined['VVIX'] = combined['VVIX'].ffill()
        if 'PCCR' in combined.columns: combined['PCCR'] = combined['PCCR'].ffill()
        combined = combined.fillna(0)
        
        # [신규] 자산 상장일 고려: 모든 자산의 실제 데이터가 존재하는 날부터 시작
        leverage_df = data_dict.get(leverage_asset, pd.DataFrame())
        
        # [수정] dropna()를 사용하여 실제 데이터가 있는 날짜 탐색
        b_min = base_df.dropna(subset=['Close']).index.min()
        l_min = leverage_df.dropna(subset=['Close']).index.min() if not leverage_df.empty else b_min
        common_start = max(b_min, l_min)
        
        dates = combined.index
        dates = dates[dates >= common_start]
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame(), pd.DataFrame()

        portfolio_value, cash = 10000.0, 10000.0 * cash_ratio
        current_planned_asset, rebalance_stage, last_entry_price = base_asset, 0, 0
        peak_price, peak_macd = 0, 0
        holdings = {t: 0.0 for t in data_dict.keys()}
        
        initial_price = base_df['Close'].asof(dates[0])
        if pd.isna(initial_price) or initial_price <= 0:
            # [수정] 데이터 부재 시 백업 로직 강화
            valid_closes = base_df['Close'].dropna()
            initial_price = valid_closes.iloc[0] if not valid_closes.empty else 1.0
            
        holdings[base_asset] = (portfolio_value * (1 - cash_ratio)) / initial_price if initial_price > 0 else 0
        
        def get_v_level(planned, stage):
            # Stage는 이제 항상 레버리지 노출 수준(0-3)을 의미합니다.
            return stage

        history = []
        trade_slots = [] # list of {'buy_price': val, 'date': date}
        closed_trades = [] # list of {'entry_date': d, 'exit_date': d, 'asset': a, 'buy_price': p, 'sell_price': p, 'ret': r, 'status': s}
        
        # [추가] 지연 실행용 상태 변수
        pending_rebalance = None # {'target_lev_pct': val, 'rebalance_stage': val, 'planned': asset}
        
        # [수정] 자산 그룹 결정 (Natural Transition용 및 에러 방지)
        semi_group = ['SOXX', 'USD', 'SOXL']
        nasdaq_group = ['QQQ', 'QLD', 'TQQQ']
        bond_group = ['TLT', 'TLT', 'TMF'] # 채권은 1x, 1x, 3x로 임시 매핑 (2배수가 없으므로)
        
        # [신규] 공통 타겟 그룹 사전 정의 (UnboundLocalError 방지)
        is_semi_mode = (base_asset in semi_group or leverage_asset in semi_group)
        is_bond_mode = (base_asset in bond_group or leverage_asset in bond_group)
        target_group = semi_group if is_semi_mode else (bond_group if is_bond_mode else nasdaq_group)
        
        for date in dates:
            row = combined.loc[date]
            price = row['Close']
            
            # [신규] 일별 변수 초기화 (로그용)
            trade_ret_val = None
            trade_status_val = ""
            trade_entry_date_val = None
            s3_drop_triggered_val = 0
            s3_detail_val = ""
            s3_sell3_event_val = 0 # [복구] S3 + 매도신호3 발생 여부
            
            # [신규] 일별 로직 변수 초기화 (NameError 방지)
            trade_label = ""
            rebalance_cause = ""
            is_delayed = False
            delay_reason = ""
            
            # [수정] 익일 시가 매매 처리: 전날 신호가 있었다면 오늘 시가로 먼저 체결
            if trade_at == "익일 시가" and pending_rebalance is not None:
                # 오늘 시가(Open)로 체결 (데이터가 없는 경우 종가로 대체, 그것도 없으면 건너뜀)
                try:
                    
                    trade_prices = {}
                    # [수정] 모든 스위칭 후보 자산의 시가를 확보
                    check_targets = list(set([base_asset, leverage_asset] + target_group))
                    for t in check_targets:
                        if t not in data_dict: continue
                        if date in data_dict[t].index:
                            trade_prices[t] = data_dict[t].loc[date, 'Open'] if 'Open' in data_dict[t].columns else data_dict[t].loc[date, 'Close']
                        else:
                            trade_prices[t] = data_dict[t]['Close'].asof(date)
                    
                    # 필수 자산(현재 보유 중인 것 + 갈아탈 것)의 가격이 없으면 스킵
                    required = [t for t, qty in holdings.items() if qty > 0] + [pending_rebalance['planned']]
                    if any(t not in trade_prices or pd.isna(trade_prices[t]) for t in required):
                        continue
                except:
                    # 데이터 부재 시 체결 지연
                    continue
                
                # 거래 시점의 자산 가치 재계산
                current_total_val = cash + sum(holdings[t] * trade_prices[t] for t in holdings if t in trade_prices)
                cash = current_total_val * cash_ratio
                etf_funds = current_total_val * (1 - cash_ratio)
                
                t_lev_asset, t_lev_pct = pending_rebalance['planned'], pending_rebalance['target_lev_pct']
                rebalance_stage = pending_rebalance['rebalance_stage']
                
                # [수정] 순차적 포트폴리오 관리 (상시 활성화)
                # 1. 현재 자산 평가 (trade_prices 기준)
                h_vals = {t: holdings[t] * trade_prices[t] for t in holdings if t in trade_prices}
                curr_total_lev_val = sum(h_vals.get(t, 0) for t in target_group[1:]) # 현재 합산 레버리지
                
                # 2. 목표 레버리지 총액 및 업그레이드 Cap 계산 (30/70/100)
                normalized_pct = t_lev_pct / 100.0 if t_lev_pct > 1 else t_lev_pct
                target_lev_val = etf_funds * normalized_pct
                
                tqqq_cap = target_lev_val
                if target_lev_val > etf_funds * 0.9 and t_lev_asset == target_group[2]:
                    # 100% 레버리지 구간에서의 터보 업그레이드 단계 적용
                    if rebalance_stage == 1: tqqq_cap = 0.30 * target_lev_val
                    elif rebalance_stage == 2: tqqq_cap = 0.70 * target_lev_val

                new_h_vals = {t: 0.0 for t in holdings}
                
                if curr_total_lev_val > target_lev_val + 0.01:
                    # 감축(Sell) 상황: QLD부터 채우고 남은 자리를 TQQQ로 채움 (TQQQ 우선 매도)
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                    rem = target_lev_val - new_h_vals[target_group[1]]
                    new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                else:
                    # 증액(Buy) 상황: 기존 레버리지 자산들을 유지하며 부족분만 현재 타겟으로 충전
                    # 1. TQQQ 보존 및 업그레이드 (Cap 한도 내)
                    if t_lev_asset == target_group[2]:
                        # 가속 모드: QLD를 팔아서라도 tqqq_cap까지 채움 (업그레이드)
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0) + h_vals.get(target_group[1], 0), tqqq_cap)
                    else:
                        # 일반 모드: 기존 TQQQ만 유지
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), tqqq_cap)
                    
                    # 2. QLD 보전 및 충전 (남은 비중 한도 내)
                    rem = target_lev_val - new_h_vals[target_group[2]]
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)
                    
                    # 3. 최종 부족분 충전 (현재 시장 타겟 자산으로)
                    rem = target_lev_val - (new_h_vals[target_group[1]] + new_h_vals[target_group[2]])
                    if rem > 0:
                        new_h_vals[t_lev_asset] += rem

                # 3. 나머지는 Base 자산
                allocated_lev = sum(new_h_vals.values())
                new_h_vals[base_asset] = max(0, etf_funds - allocated_lev)
                
                # 수량 업데이트
                for t in check_targets:
                    holdings[t] = new_h_vals.get(t, 0) / trade_prices[t] if t in trade_prices and trade_prices[t] > 0 else 0
                
                # [신규] FIFO 기반 가상 장부 업데이트 (익일 시가 체결 버전)
                new_stage_open = pending_rebalance['rebalance_stage']
                old_v = get_v_level(current_planned_asset, rebalance_stage)
                new_v = get_v_level(t_lev_asset, new_stage_open)

                if new_v > old_v:
                    # 비중 확대
                    for _ in range(new_v - old_v):
                        # 실제 매매된 가속 자산 가격 사용
                        trade_slots.append({'buy_price': trade_prices[t_lev_asset], 'date': date, 'asset': t_lev_asset})
                elif new_v < old_v:
                    # 비중 축소: 정산
                    diff = old_v - new_v
                    rets = []
                    entry_dates = []
                    for _ in range(diff):
                        if trade_slots:
                            slot = trade_slots.pop(0)
                            asset_sold = slot.get('asset', t_lev_asset)
                            sell_p = trade_prices[asset_sold]
                            r = (sell_p / slot['buy_price'] - 1) * 100
                            rets.append(r)
                            entry_dates.append(slot['date'])
                            closed_trades.append({
                                'Entry_Date': slot['date'], 'Exit_Date': date, 'Asset': asset_sold,
                                'Buy_Price': slot['buy_price'], 'Sell_Price': sell_p,
                                'Return': r, 'Status': "익절" if r > 0 else "손절"
                            })
                    if rets and history:
                        # 익일 시가는 어제 행에 기록 (대표 수익률은 평균값)
                        history[-1]['Trade_Ret'] = np.mean(rets)
                        history[-1]['Trade_Status'] = "익절" if history[-1]['Trade_Ret'] > 0 else "손절"
                        history[-1]['Trade_Entry_Date'] = ", ".join([d.strftime('%Y-%m-%d') for d in entry_dates])

                current_planned_asset = t_lev_asset
                rebalance_stage = new_stage_open
                last_entry_price = trade_prices[base_asset]
                pending_rebalance = None # 처리 완료
            
            rsi, fg, vix, macd_val, signal_line, rsi_sma = row['RSI'], row['FG'], row['VIX'], row['MACD'], row['Signal_Line'], row['RSI_SMA']
            macd_hist = row['MACD_Hist']
            bb_l, bb_u = row.get('BB_Lower', 0), row.get('BB_Upper', 0)
            prev_rsi, prev_rsi_sma, prev_macd, prev_macd_hist = row['Prev_RSI'], row['Prev_RSI_SMA'], row['Prev_MACD'], row['Prev_MACD_Hist']
            prev2_macd_hist = row['Prev2_MACD_Hist']
            prev_vix = row.get('Prev_VIX', 0)
            vix_jump_pct = ((vix / prev_vix - 1) * 100) if prev_vix > 0 else 0
            
            # [신규] 다이버전스용 고점 추적
            if current_planned_asset != base_asset:
                if price > peak_price:
                    peak_price = price
                    peak_macd = macd_val
            else:
                peak_price, peak_macd = 0, 0
            
            if params:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                is_macd_golden_cross = (prev_macd_hist < 0) and (macd_hist > 0)
                is_macd_dead_cross = (prev_macd_hist > 0) and (macd_hist < 0)
                
                # [신규] 지표 값 확보
                adx_val = row.get('ADX', 0)
                d_plus, d_minus = row.get('DMP', 0), row.get('DMN', 0)
                
                is_buy_signal = False
                signal_reason = ""
                for bp in params['buy_signals']:
                    cond, active = True, False
                    reasons = []
                    if bp.get('rsi_val', 0) > 0: 
                        cond &= (rsi < bp['rsi_val']); active = True
                        reasons.append(f"RSI<{bp['rsi_val']}")
                    if bp.get('rsi_cross'): 
                        cond &= is_rsi_golden_cross; active = True
                        reasons.append("RSI 골든크로스")
                    if bp.get('use_adx'):
                        op = bp.get('adx_op', '<=')
                        val = bp.get('adx_val', 40)
                        if op == '<=':
                            cond &= (adx_val <= val)
                        else:
                            cond &= (adx_val >= val)
                        active = True
                        reasons.append(f"ADX{op}{val}")
                    if bp.get('rsi_inc'): 
                        cond &= (rsi > prev_rsi); active = True
                        reasons.append("RSI 상승")
                    if bp.get('macd_inc'): 
                        cond &= (macd_val > prev_macd); active = True
                        reasons.append("MACD 상승")
                    if bp.get('macd_signal_below'): 
                        cond &= (macd_hist < 0); active = True
                        reasons.append("MACD 시그널 하단")
                    if bp.get('macd_golden'): 
                        cond &= is_macd_golden_cross; active = True
                        reasons.append("MACD 골든크로스")
                    if bp.get('bb_lower'): 
                        cond &= (price <= bb_l); active = True
                        reasons.append("BB 하단 터치")
                    
                    if active and cond: 
                        is_buy_signal = True
                        signal_reason = " / ".join(reasons)
                        break
                
                is_sell_signal = False
                sell_signal_idx = -1 # [신규] 어떤 매도 신호가 발생했는지 추적
                for idx, sp in enumerate(params['sell_signals']):
                    cond, active = True, False
                    reasons = []
                    if sp.get('rsi_val', 0) > 0: 
                        cond &= (rsi >= sp['rsi_val']); active = True
                        reasons.append(f"RSI>={sp['rsi_val']}")
                    if sp.get('rsi_dead'): 
                        cond &= is_rsi_dead_cross; active = True
                        reasons.append("RSI 데드크로스")
                    if sp.get('rsi_dec'): 
                        cond &= (rsi < prev_rsi); active = True
                        reasons.append("RSI 하락")
                    if sp.get('macd_dec'): 
                        cond &= (macd_val < prev_macd); active = True
                        reasons.append("MACD 하락")
                    if sp.get('macd_signal_above'): 
                        cond &= (macd_hist > 0); active = True
                        reasons.append("MACD 시그널 상단")
                    if sp.get('macd_dead'): 
                        cond &= is_macd_dead_cross; active = True
                        reasons.append("MACD 데드크로스")
                    if sp.get('di_minus_above'):
                        cond &= (d_minus > d_plus); active = True
                        reasons.append("DI->DI+")
                    if sp.get('bb_upper'): 
                        cond &= (price >= bb_u); active = True
                        reasons.append("BB 상단 터치")
                    
                    if active and cond: 
                        is_sell_signal = True
                        sell_signal_idx = idx + 1 # 1-based index
                        signal_reason = " / ".join(reasons)
                        break

                # [복구] S3(레버리지 3단계)에서 매도신호3 발생 시에만 기록
                if is_sell_signal and rebalance_stage == 3 and sell_signal_idx == 3:
                    s3_sell3_event_val = 1

                # [기존] 일반 매도 시그널 순회 끝
                pass
                
                b_up, b_down, s_up, s_down = params['buy_reb_up'], params['buy_reb_down'], params['sell_reb_up'], params['sell_reb_down']
            else:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_buy_signal = False
                signal_reason = ""
                if is_rsi_golden_cross and (macd_val > prev_macd) and (macd_hist < 0) and (adx_val <= 40):
                    is_buy_signal = True
                    signal_reason = "기본(RSI골든크로스+MACD상승)"
                elif (rsi < 35) and (adx_val <= 40):
                    is_buy_signal = True
                    signal_reason = "기본(RSI과매도)"
                
                sell_cond_1 = (rsi >= 70) and (rsi < prev_rsi)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                sell_cond_2 = (macd_hist > 0) and (macd_val < prev_macd) and is_rsi_dead_cross
                sell_cond_5 = (prev_macd_hist > 0) and (macd_hist < 0)
                
                is_sell_signal = False
                if sell_cond_1: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI꺾임)"
                elif sell_cond_2: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI데드크로스+MACD)"
                elif sell_cond_5: 
                    is_sell_signal = True
                    signal_reason = "기본(MACD데드크로스)"
                
                b_up, b_down, s_up, s_down = 0.03, -0.03, 0.03, -0.03

            # [수정] 스마트 레버리지 필터링 (VIX Safety vs RSI Turbo 분리)
            effective_lev_asset = leverage_asset
            smart_reason = ""
            
            if smart_params:
                if smart_params.get('use_vix_safety') and vix > smart_params.get('vix_exit', 31):
                    effective_lev_asset = target_group[0]
                    smart_reason = f"대피(VIX:{vix:.1f})"
                elif smart_params.get('use_rsi_turbo') and rsi < smart_params.get('rsi_turbo', 31):
                    effective_lev_asset = target_group[2]
                    smart_reason = f"가속(RSI:{rsi:.1f})"
                else:
                    effective_lev_asset = leverage_asset
                    if smart_params.get('use_vix_safety') or smart_params.get('use_rsi_turbo'):
                        smart_reason = "정상(Normal)"

            try:
                currently_holding = [t for t, qty in holdings.items() if qty > 0]
                check_targets = list(set([base_asset, leverage_asset, effective_lev_asset] + currently_holding + target_group))
                prices = {t: data_dict[t].loc[date, 'Close'] for t in data_dict.keys() if date in data_dict[t].index}
                if base_asset not in prices or any(t not in prices for t in currently_holding):
                    continue
            except (KeyError, ValueError):
                continue 
            
            current_total_val = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings if t in prices)
            
            # [신규 추가] 수익률 nan 방지를 위한 수치 정규화 (Defensive Check)
            if pd.isna(current_total_val) or current_total_val <= 0:
                current_total_val = history[-1]['Value'] if history else 10000.0
                
            rebalance_needed = False
            new_planned = current_planned_asset
            new_stage = rebalance_stage
            
            if last_entry_price == 0:
                last_entry_price = prices[base_asset]

            etf_funds = current_total_val * (1 - cash_ratio)
            lev_candidates = ['QLD', 'TQQQ', 'SOXL', 'USD', 'TMF']
            current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            current_lev_pct = current_lev_funds / etf_funds if etf_funds > 0 else 0

            # [리밸런싱 판정 우선순위 조정] 매입(신호/가격)이 매도(신호/가격)보다 우선함
            is_buy_process = (current_planned_asset != base_asset)
            is_mode_consistent = (current_planned_asset == effective_lev_asset)

            # [신규] ATR 샹들리에 엑시트를 위한 최고가(Peak Price) 실시간 갱신
            if is_buy_process:
                if peak_price == 0 or prices[base_asset] > peak_price:
                    peak_price = prices[base_asset]
            else:
                peak_price = 0 # 매도 프로세스 중에는 고점 추적 안함
            
            # [수정] 리밸런싱 트리거 판정 (하이브리드: 고정 % + ATR 변동성 병합)
            use_fixed = params.get('use_fixed_reb', True)
            use_atr = params.get('use_atr_reb', False)
            
            price_change = prices[base_asset] / last_entry_price
            price_diff_pct = (price_change - 1) * 100
            
            buy_price_hit = False
            sell_price_hit = False
            
            # (1) 시그널 매입 판정 (모드 공통)
            buy_signal_hit = is_buy_signal and not is_mode_consistent and effective_lev_asset in prices

            # (2) 고정 비율 트리거 계산
            fixed_buy_hit = False
            fixed_sell_hit = False
            if use_fixed:
                up_thresh = abs(b_up if current_planned_asset == leverage_asset else s_up)
                down_thresh = abs(b_down if current_planned_asset == leverage_asset else s_down)
                price_trigger = (price_change >= 1 + up_thresh or price_change <= 1 - down_thresh)
                fixed_buy_hit = (rebalance_stage in [1, 2] and is_buy_process and is_mode_consistent and price_trigger)
                fixed_sell_hit = (rebalance_stage in [1, 2, 3] and not is_buy_process and price_trigger)

            # (3) ATR 트리거 계산
            atr_buy_hit = False
            atr_sell_hit = False
            if use_atr:
                prev_atr = row.get('Prev_ATR', 0)
                k_up = params.get('atr_mult_buy_up', 10.0)
                k_down = params.get('atr_mult_buy_down', 1.5)
                k_sell = params.get('atr_mult_sell', 3.0)
                
                # ATR 매수 트리거 (상향/하향)
                hit_down = (prices[base_asset] <= last_entry_price - (k_down * prev_atr))
                hit_up = (prices[base_asset] >= last_entry_price + (k_up * prev_atr))
                atr_buy_hit = (rebalance_stage in [1, 2] and is_mode_consistent and (hit_down or hit_up))
                
                # ATR 샹들리에 엑시트
                if is_buy_process and peak_price > 0:
                    atr_sell_hit = (prices[base_asset] <= peak_price - (k_sell * prev_atr))
                atr_sell_hit = (rebalance_stage in [1, 2, 3] and atr_sell_hit)

            # (4) 최종 판정 (OR 결합)
            buy_price_hit = fixed_buy_hit or atr_buy_hit
            sell_price_hit = fixed_sell_hit or atr_sell_hit
            
            if buy_signal_hit or buy_price_hit:
                if buy_signal_hit:
                    new_planned = effective_lev_asset
                    if effective_lev_asset == base_asset:
                        target_lev_pct, new_stage = 0.0, 0
                    else:
                        if current_lev_pct < 0.25: target_lev_pct, new_stage = 0.30, 1
                        elif current_lev_pct < 0.65: target_lev_pct, new_stage = 0.70, 2
                        else:
                            if current_planned_asset != effective_lev_asset and effective_lev_asset == target_group[2] and current_lev_pct > 0.9:
                                curr_tqqq_w = (holdings.get(target_group[2], 0) * prices.get(target_group[2], 0)) / etf_funds if etf_funds > 0 else 0
                                if curr_tqqq_w < 0.25: target_lev_pct, new_stage = 1.0, 1
                                elif curr_tqqq_w < 0.65: target_lev_pct, new_stage = 1.0, 2
                                else: target_lev_pct, new_stage = 1.0, 3
                            else:
                                target_lev_pct, new_stage = 1.0, 3
                    rebalance_cause = "시그널"
                else:
                    # 가격 트리거에 의한 매입/상향 (1->2, 2->3)
                    new_stage = min(3, rebalance_stage + 1)
                    target_lev_pct = (0.70 if new_stage == 2 else 1.0)
                    if fixed_buy_hit and atr_buy_hit: rebalance_cause = "고정+ATR(매수)"
                    elif fixed_buy_hit: rebalance_cause = "고정(매수)"
                    else: rebalance_cause = "ATR(매수)"
                rebalance_needed = True
                
            # (2) 매도 판정 (자산 교체 신호)
            elif is_sell_signal and current_planned_asset != base_asset:
                new_planned = base_asset
                # [수정] 점진적 매도 로직 복원 (3->2, 2->1, 1->0)
                if current_lev_pct > 0.75: target_lev_pct, new_stage = 0.70, 2
                elif current_lev_pct > 0.35: target_lev_pct, new_stage = 0.30, 1
                else: target_lev_pct, new_stage = 0.0, 0
                rebalance_cause = "시그널"
                rebalance_needed = True
                
            elif sell_price_hit:
                # 레버리지 비중 축소 (3->2, 2->1, 1->0)
                new_stage = max(0, rebalance_stage - 1)
                target_lev_pct = (0.70 if new_stage == 2 else (0.30 if new_stage == 1 else 0.0))
                
                if fixed_sell_hit and atr_sell_hit: rebalance_cause = "고정+ATR(매도)"
                elif fixed_sell_hit: rebalance_cause = "고정(매도)"
                else: rebalance_cause = "ATR(샹들리에)"
                rebalance_needed = True

            # [신규] S3(레버리지 100%) 전용 보호 조건 처리 (Emergency Defensive)
            # 여러 개의 신호 중 하나라도 충족되면(OR) 작동함
            s3_p_list = params.get('s3_protection', [])
            if not isinstance(s3_p_list, list): s3_p_list = [s3_p_list]
            
            if rebalance_stage == 3:
                for idx, s3_p in enumerate(s3_p_list, 1):
                    is_drop_ok = True
                    if s3_p.get('use_daily_drop'):
                        daily_ret = (price / row['Prev_Close'] - 1) * 100
                        is_drop_ok = (daily_ret <= s3_p.get('drop_limit', -3.0))
                    
                    is_ma60_ok = True
                    if s3_p.get('use_ma60'):
                        sma60 = row.get('SMA60', 0)
                        limit = s3_p.get('ma60_limit', 0.0)
                        is_ma60_ok = (price < sma60 * (1 + limit / 100.0))
                    
                    is_ma200_ok = True
                    if s3_p.get('use_ma200'):
                        sma200 = row.get('SMA200', 0)
                        limit = s3_p.get('ma200_limit', 0.0)
                        is_ma200_ok = (price < sma200 * (1 + limit / 100.0))

                    is_vix_ok = True
                    if s3_p.get('use_vix_jump'):
                        is_vix_ok = (vix_jump_pct >= s3_p.get('vix_jump', 15.0))
                    
                    is_gap_ok = True
                    if s3_p.get('use_gap_down'):
                        gap_down = row.get('Gap_Down', 0)
                        is_gap_ok = (gap_down <= s3_p.get('gap_limit', -3.0))
                    
                    is_acc_ok = True
                    if s3_p.get('use_drop_acc'):
                        drop_acc = row.get('Drop_Acc', 0)
                        is_acc_ok = (drop_acc <= s3_p.get('acc_limit', -7.0))
                    
                    # S3 보호 조건 결합 방식 정교화: 
                    # 1. 보호 신호 1, 2, 3 '내부' 항목들은 모두 만족해야 함 (AND)
                    # 2. 보호 신호 1, 2, 3 '끼리'는 하나라도 만족하면 작동 (OR)
                    
                    active_conditions = []
                    if s3_p.get('use_daily_drop'): active_conditions.append(is_drop_ok)
                    if s3_p.get('use_ma60'): active_conditions.append(is_ma60_ok)
                    if s3_p.get('use_ma200'): active_conditions.append(is_ma200_ok)
                    if s3_p.get('use_vix_jump'): active_conditions.append(is_vix_ok)
                    if s3_p.get('use_gap_down'): active_conditions.append(is_gap_ok)
                    if s3_p.get('use_drop_acc'): active_conditions.append(is_acc_ok)
                    
                    # 하나라도 체크되어 있고, 체크된 모든 조건이 참(True)인 경우 작동
                    if len(active_conditions) > 0 and all(active_conditions):
                        dr = (price / row['Prev_Close'] - 1) * 100
                        rebalance_needed = True
                        new_planned = base_asset
                        
                        # 로그 상세화 (#N:신호번호, V:VIX, G:Gap, A:Acc, M60:MA60, M200:MA200)
                        detail_log = f"#{idx} D:{dr:+.1f}%"
                        if s3_p.get('use_ma60'): detail_log += f" M60({s3_p.get('ma60_limit', 0.0):+.1f}%)"
                        if s3_p.get('use_ma200'): detail_log += f" M200({s3_p.get('ma200_limit', 0.0):+.1f}%)"
                        if s3_p.get('use_vix_jump'): detail_log += f" V:{vix_jump_pct:+.1f}%"
                        if s3_p.get('use_gap_down'): detail_log += f" G:{row.get('Gap_Down', 0):+.1f}%"
                        if s3_p.get('use_drop_acc'): detail_log += f" A:{row.get('Drop_Acc', 0):+.1f}%"

                        s3_detail_val = detail_log
                        
                        if s3_p.get('use_exit_all', False):
                            new_stage = 0
                            target_lev_pct = 0.0
                            rebalance_cause = f"S3-전량매도({detail_log})"
                        else:
                            new_stage = 2
                            target_lev_pct = 0.70
                            rebalance_cause = f"S3-복합위기({detail_log})"
                        
                        s3_drop_triggered_val = dr 
                        break # 신호 간에는 OR이므로 하나라도 충족되면 검사 종료

            passed_panic = False
            if rebalance_needed and new_planned != base_asset and smart_params and smart_params.get('use_panic', False):
                panic_ma = smart_params.get('panic_ma', 200)
                ma_val = combined.loc[date, f'SMA{panic_ma}']
                if prices.get(base_asset, price) < ma_val:
                    if new_stage == 1: target_rsi = smart_params.get('panic_rsi_s1', 27)
                    elif new_stage == 2: target_rsi = smart_params.get('panic_rsi_s2', 28)
                    else: target_rsi = smart_params.get('panic_rsi_s3', 30)
                    if rsi > target_rsi:
                        rebalance_needed, is_delayed, delay_reason = False, True, f"MA-RSI"
                    else:
                        passed_panic = True
                
            # [신규] 손절제어 (Stop-Loss Control) 로직
            # 오직 '매도(비중 축소)' 상황에서만 손절제어를 체크함
            if rebalance_needed and not is_delayed and smart_params and smart_params.get('use_sl_control', False):
                # 1. 목표 레버리지 비중이 현재보다 낮거나(매도), 2. 안전자산(base_asset)으로의 완전 회귀 시 작동
                is_selling_move = (target_lev_pct < current_lev_pct - 0.01) or (new_planned == base_asset and current_planned_asset != base_asset)
                
                if is_selling_move:
                    if trade_slots:
                        # FIFO 기준으로 가장 오래된 물량의 수익률 체크
                        oldest_slot = trade_slots[0]
                        asset_to_check = oldest_slot.get('asset', leverage_asset)
                        if asset_to_check in prices:
                            curr_ret = (prices[asset_to_check] / oldest_slot['buy_price'] - 1) * 100
                            sl_limit = smart_params.get('sl_control_limit', -15)
                            if curr_ret <= sl_limit:
                                # 지연 사유에 원래의 리밸런싱 원인(시그널/가격조건)을 포함하여 투명성 강화
                                cause_info = f" [{rebalance_cause}]" if rebalance_cause else ""
                                rebalance_needed, is_delayed, delay_reason = False, True, f"손절제어{cause_info}({curr_ret:.1f}%)"

            if rebalance_needed:
                if trade_at == "종가":
                    h_vals = {t: holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates}
                    curr_total_lev_val = sum(h_vals.values())
                    target_lev_val = etf_funds * target_lev_pct
                    new_h_vals = {t: 0.0 for t in holdings}

                    if curr_total_lev_val > target_lev_val + 0.01:
                        new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                        rem = target_lev_val - new_h_vals[target_group[1]]
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                    else:
                        tqqq_cap = target_lev_val
                        if target_lev_val > etf_funds * 0.9 and effective_lev_asset == target_group[2]:
                            if new_stage == 1: tqqq_cap = 0.30 * target_lev_val
                            elif new_stage == 2: tqqq_cap = 0.70 * target_lev_val

                        is_upgrade_move = (target_lev_val > etf_funds * 0.9 and new_stage in [1, 2])
                        if effective_lev_asset == target_group[2] and is_upgrade_move:
                            new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0) + h_vals.get(target_group[1], 0), tqqq_cap)
                        else:
                            new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), tqqq_cap)
                        
                        rem = target_lev_val - new_h_vals[target_group[2]]
                        new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)
                        
                        rem = target_lev_val - (new_h_vals[target_group[1]] + new_h_vals[target_group[2]])
                        if rem > 0:
                            new_h_vals[effective_lev_asset] += rem
                    
                    new_h_vals[base_asset] = max(0, etf_funds - sum(new_h_vals.values()))
                    for t in check_targets:
                        holdings[t] = new_h_vals.get(t, 0) / prices[t] if t in prices and prices[t] > 0 else 0
                    
                    old_v = get_v_level(current_planned_asset, rebalance_stage)
                    new_v = get_v_level(new_planned, new_stage)

                    if new_v > old_v:
                        for _ in range(new_v - old_v):
                            trade_slots.append({'buy_price': prices[new_planned], 'date': date, 'asset': new_planned})
                    elif new_v < old_v:
                        diff = old_v - new_v
                        rets = []
                        entry_dates = []
                        for _ in range(diff):
                            if trade_slots:
                                slot = trade_slots.pop(0)
                                asset_sold = slot.get('asset', leverage_asset)
                                sell_p = prices[asset_sold]
                                r = (sell_p / slot['buy_price'] - 1) * 100
                                rets.append(r)
                                entry_dates.append(slot['date'])
                                closed_trades.append({
                                    'Entry_Date': slot['date'], 'Exit_Date': date, 'Asset': asset_sold,
                                    'Buy_Price': slot['buy_price'], 'Sell_Price': sell_p,
                                    'Return': r, 'Status': "익절" if r > 0 else "손절"
                                })
                        if rets:
                            trade_ret_val = np.mean(rets)
                            trade_status_val = "익절" if trade_ret_val > 0 else "손절"
                            trade_entry_date_val = ", ".join([d.strftime('%Y-%m-%d') for d in entry_dates])

                    current_planned_asset = new_planned
                    rebalance_stage = new_stage
                    last_entry_price = prices[base_asset]
                    peak_price = prices[base_asset] # 리밸런싱 시 고점 초기화

                    # [추가] 매매 후 현금 및 레버리지 자산 총액 재계산 (로그용)
                    cash = current_total_val * cash_ratio
                    current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
                else:
                    pending_rebalance = {'target_lev_pct': target_lev_pct, 'rebalance_stage': new_stage, 'planned': new_planned}
                    # [신규] '익일 시가' 매매 시에도 기준가격 계산의 베이스는 '신호 발생일 종가'로 고속
                    last_entry_price = prices[base_asset]
                    peak_price = prices[base_asset]

            # 시각적 가독성을 위해 trade_label 설정
            if rebalance_needed:
                prefix = "매도" if new_planned == base_asset else "매수"
                # [수정] rebalance_cause가 "시그널"이나 "가격조건"이 아닌 경우(예: S3보호)에도 해당 원인이 표시되도록 수정
                if rebalance_cause == "시그널":
                    condition_label = signal_reason
                elif "ATR" in rebalance_cause:
                    condition_label = f"{rebalance_cause} (ATR:{prev_atr:.2f})"
                elif "고정" in rebalance_cause:
                    condition_label = f"{rebalance_cause} ({price_diff_pct:+.1f}%)"
                else:
                    condition_label = rebalance_cause
                
                # [수정] 하락장(Panic) 구간에서 지연을 뚫고 들어온 경우, 'RSI조건'을 우선 표시함
                if passed_panic:
                    if new_stage == 1: limit = smart_params.get('panic_rsi_s1', 27)
                    elif new_stage == 2: limit = smart_params.get('panic_rsi_s2', 28)
                    else: limit = smart_params.get('panic_rsi_s3', 30)
                    condition_label = f"RSI조건(RSI<{limit})"

                trade_label = f"{prefix}(S{new_stage}): {condition_label}"
                if smart_params and smart_params.get('use_smart') and smart_reason:
                    trade_label += f" [{smart_reason}]"
            elif is_delayed:
                trade_label = f"지연({delay_reason})"
                if delay_reason.startswith("MA"):
                    trade_label = f"지연({delay_reason}:{rsi:.1f})"
            else:
                state_label = f"진행중(S{rebalance_stage})" if 0 < rebalance_stage < 3 else "중립"
                if is_buy_signal or is_sell_signal:
                    sig_type = "매수시그널" if is_buy_signal else "매도시그널"
                    trade_label = f"{state_label}: {sig_type}({signal_reason})"
                else:
                    trade_label = state_label

            summary_p = []
            for t in target_group[:0:-1]: # TQQQ, QLD 순기록
                w_val = (holdings.get(t, 0) * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
                if w_val >= 0.01: summary_p.append(f"{t} {w_val*100:.0f}%")
            b_w = (holdings.get(base_asset, 0) * prices.get(base_asset, 0)) / current_total_val if current_total_val > 0 else 0
            if b_w >= 0.01: summary_p.append(f"{base_asset} {b_w*100:.0f}%")
            
            # [수정] 맵핑용 레버리지 자산 총액 및 현금 비중 실시간 재계산 (로그 정확도 향상)
            log_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            log_cash_weight = cash / current_total_val if current_total_val > 0 else 0
            log_lev_weight = log_lev_funds / current_total_val if current_total_val > 0 else 0

            # [수정] 자산 명칭과 Stage를 로그 기록 시점에 동기화
            log_asset_name = current_planned_asset
            if 0 < rebalance_stage < 3:
                log_asset_name = f"{current_planned_asset}(S{int(rebalance_stage)})"
            elif rebalance_stage == 3:
                # S3는 레버리지 자산일 때만 명시적으로 표시 (base_asset이면 그냥 표시)
                if current_planned_asset != base_asset:
                    log_asset_name = f"{current_planned_asset}(S3)"

            history.append({
                'Date': date,
                'Value': current_total_val,
                'Cash_Weight': log_cash_weight,
                'Lev_Weight': log_lev_weight,
                'Stage': int(rebalance_stage),
                'Asset': log_asset_name,
                'Drop_Acc': row.get('Drop_Acc', 0),
                'Gap_Down': row.get('Gap_Down', 0),
                'S3_Drop_Limit': s3_drop_triggered_val, # [기록]
                'S3_Sell3_Event': s3_sell3_event_val, # [복구]
                'S3_Detail': s3_detail_val, # [신규 추가]
                'Trade_Label': trade_label,
                'Summary': ", ".join(summary_p),
                'Trade_Ret': trade_ret_val,
                'Trade_Status': trade_status_val,
                'Trade_Entry_Date': trade_entry_date_val,
                'Close': prices[base_asset],
                'RSI': rsi,
                'MACD': macd_val,
                'FG': fg,
                'VIX': vix,
                'ADX': row.get('ADX', 0),
                'DMP': row.get('DMP', 0),
                'DMN': row.get('DMN', 0),
                'Smart_Mode': smart_reason,
                'SMA200': row.get('SMA200', 0),
                'SMA60': row.get('SMA60', 0),
                'Target': f"{new_planned}(S{new_stage})" if new_stage < 3 else new_planned,
                'STOCH_K': row.get('STOCH_K', 0),
                'STOCH_D': row.get('STOCH_D', 0)
            })
            for t in lev_candidates:
                if t in holdings:
                    history[-1][f'{t}_Weight'] = (holdings[t] * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
            history[-1][f'{base_asset}_Weight'] = (holdings.get(base_asset, 0) * prices.get(base_asset, 0)) / current_total_val if current_total_val > 0 else 0

        return pd.DataFrame(history).set_index('Date'), pd.DataFrame(closed_trades)

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_benchmark(data_dict, ticker, start_date, end_date):
        if ticker not in data_dict: return pd.DataFrame()
        df = data_dict[ticker].dropna(subset=['Close'])
        dates = df.index
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame()
        shares = 10000.0 / df.loc[dates[0], 'Close']
        return pd.DataFrame([{'Date': d, 'Value': shares * df.loc[d, 'Close']} for d in dates]).set_index('Date')

    @staticmethod
    def run_monte_carlo(history, days=252, n_sims=200):
        if history.empty or len(history) < 30: return None
        rets = history['Value'].pct_change().dropna()
        mu, sigma, last_v = rets.mean(), rets.std(), history['Value'].iloc[-1]
        sims = np.zeros((days, n_sims))
        for i in range(n_sims):
            sims[:, i] = last_v * np.exp(np.cumsum(np.random.normal(mu, sigma, days)))
        return sims
