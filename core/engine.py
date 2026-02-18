import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_date, end_date, params=None, trade_at="종가", smart_params=None):
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
        
        cols_to_shift = ['RSI', 'RSI_SMA', 'MACD', 'Signal_Line', 'SMA200', 'SMA60', 'MACD_Hist']
        if 'STOCH_K' in combined.columns:
            cols_to_shift += ['STOCH_K', 'STOCH_D']
            
        for col in cols_to_shift:
            combined[f'Prev_{col}'] = combined[col].shift(1)
        
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
        if pd.isna(initial_price):
            initial_price = base_df['Close'].dropna().iloc[0] if not base_df['Close'].dropna().empty else 1.0
            
        holdings[base_asset] = (portfolio_value * (1 - cash_ratio)) / initial_price
        
        def get_v_level(planned, stage):
            if planned == base_asset:
                return 3 - stage if stage > 0 else 0
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
                
                is_buy_signal = False
                signal_reason = ""
                for bp in params['buy_signals']:
                    cond, active = True, False
                    reasons = []
                    if bp['rsi_val'] > 0: 
                        cond &= (rsi < bp['rsi_val']); active = True
                        reasons.append(f"RSI<{bp['rsi_val']}")
                    if bp['rsi_cross']: 
                        cond &= is_rsi_golden_cross; active = True
                        reasons.append("RSI 골든크로스")
                    if bp['rsi_inc']: 
                        cond &= (rsi > prev_rsi); active = True
                        reasons.append("RSI 상승")
                    if bp['macd_inc']: 
                        cond &= (macd_val > prev_macd); active = True
                        reasons.append("MACD 상승")
                    if bp['macd_signal_below']: 
                        cond &= (macd_hist < 0); active = True
                        reasons.append("MACD 시그널 하단")
                    if bp['macd_golden']: 
                        cond &= is_macd_golden_cross; active = True
                        reasons.append("MACD 골든크로스")
                    if bp['bb_lower']: 
                        cond &= (price <= bb_l); active = True
                        reasons.append("BB 하단 터치")
                    
                    if active and cond: 
                        is_buy_signal = True
                        signal_reason = " / ".join(reasons)
                        break
                
                is_sell_signal = False
                for sp in params['sell_signals']:
                    cond, active = True, False
                    reasons = []
                    if sp['rsi_val'] > 0: 
                        cond &= (rsi >= sp['rsi_val']); active = True
                        reasons.append(f"RSI>={sp['rsi_val']}")
                    if sp['rsi_dead']: 
                        cond &= is_rsi_dead_cross; active = True
                        reasons.append("RSI 데드크로스")
                    if sp['rsi_dec']: 
                        cond &= (rsi < prev_rsi); active = True
                        reasons.append("RSI 하락")
                    if sp['macd_dec']: 
                        cond &= (macd_val < prev_macd); active = True
                        reasons.append("MACD 하락")
                    if sp['macd_signal_above']: 
                        cond &= (macd_hist > 0); active = True
                        reasons.append("MACD 시그널 상단")
                    if sp['macd_dead']: 
                        cond &= is_macd_dead_cross; active = True
                        reasons.append("MACD 데드크로스")
                    if sp['bb_upper']: 
                        cond &= (price >= bb_u); active = True
                        reasons.append("BB 상단 터치")
                    
                    if active and cond: 
                        is_sell_signal = True
                        signal_reason = " / ".join(reasons)
                        break
                
                b_up, b_down, s_up, s_down = params['buy_reb_up'], params['buy_reb_down'], params['sell_reb_up'], params['sell_reb_down']
            else:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_buy_signal = False
                signal_reason = ""
                if is_rsi_golden_cross and (macd_val > prev_macd) and (macd_hist < 0):
                    is_buy_signal = True
                    signal_reason = "기본(RSI골든크로스+MACD상승)"
                elif (rsi < 35):
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
                prices = {t: data_dict[t].loc[date, 'Close'] for t in check_targets if t in data_dict and date in data_dict[t].index}
                if base_asset not in prices or any(t not in prices for t in currently_holding):
                    continue
            except (KeyError, ValueError):
                continue 
            
            current_total_val = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings if t in prices)
            rebalance_needed = False
            new_planned = current_planned_asset
            new_stage = rebalance_stage
            
            if last_entry_price == 0:
                last_entry_price = prices[base_asset]

            etf_funds = current_total_val * (1 - cash_ratio)
            lev_candidates = ['QLD', 'TQQQ', 'SOXL', 'USD', 'TMF']
            current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            current_lev_pct = current_lev_funds / etf_funds if etf_funds > 0 else 0
            
            rebalance_cause = "" 
            is_delayed = False 
            delay_reason = ""
            
            price_change = prices[base_asset] / last_entry_price
            price_diff_pct = (price_change - 1) * 100
            
            up_thresh = abs(b_up if current_planned_asset == leverage_asset else s_up)
            down_thresh = abs(b_down if current_planned_asset == leverage_asset else s_down)
            price_trigger = (price_change >= 1 + up_thresh or price_change <= 1 - down_thresh)
            
            if is_buy_signal and current_planned_asset != effective_lev_asset and effective_lev_asset in prices:
                new_planned = effective_lev_asset
                if effective_lev_asset == base_asset:
                    target_lev_pct, new_stage = 0.0, 3
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
                rebalance_needed = True
            elif is_sell_signal and current_planned_asset != base_asset:
                new_planned = base_asset
                if current_lev_pct > 0.75: target_lev_pct, new_stage = 0.70, 1
                elif current_lev_pct > 0.35: target_lev_pct, new_stage = 0.30, 2
                else: target_lev_pct, new_stage = 0.0, 3
                rebalance_cause = "시그널"
                rebalance_needed = True
            elif rebalance_stage in [1, 2]:
                is_buy_process = (current_planned_asset != base_asset)
                is_mode_consistent = (current_planned_asset == effective_lev_asset)
                if ((price_trigger and (not is_buy_process or is_mode_consistent))):
                    new_stage = rebalance_stage + 1
                    target_lev_pct = (0.70 if new_stage == 2 else 1.0) if is_buy_process else (0.30 if new_stage == 2 else 0.0)
                    rebalance_cause = "가격조건"
                    rebalance_needed = True

            if rebalance_needed and new_planned != base_asset and smart_params and smart_params.get('use_panic', False):
                panic_ma = smart_params.get('panic_ma', 200)
                ma_val = combined.loc[date, f'SMA{panic_ma}']
                if prices[base_asset] < ma_val:
                    if new_stage == 1: target_rsi = smart_params.get('panic_rsi_s1', 27)
                    elif new_stage == 2: target_rsi = smart_params.get('panic_rsi_s2', 28)
                    else: target_rsi = smart_params.get('panic_rsi_s3', 30)
                    if rsi > target_rsi:
                        rebalance_needed, is_delayed, delay_reason = False, True, f"MA-RSI"
                
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

                    # [추가] 매매 후 현금 및 레버리지 자산 총액 재계산 (로그용)
                    cash = current_total_val * cash_ratio
                    current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
                else:
                    pending_rebalance = {'target_lev_pct': target_lev_pct, 'rebalance_stage': new_stage, 'planned': new_planned}
                    # [신규] '익일 시가' 매매 시에도 기준가격 계산의 베이스는 '신호 발생일 종가'로 고속
                    last_entry_price = prices[base_asset]

            # 시각적 가독성을 위해 trade_label 설정
            if rebalance_needed:
                prefix = "매도" if new_planned == base_asset else "매수"
                condition_label = f"가격조건({price_diff_pct:+.1f}%)" if rebalance_cause == "가격조건" else (signal_reason if rebalance_cause == "시그널" else "MACD데드크로스집행")
                trade_label = f"{prefix}(S{new_stage}): {condition_label}"
                if smart_params and smart_params.get('use_smart') and smart_reason:
                    trade_label += f" [{smart_reason}]"
            elif is_delayed:
                trade_label = f"지연({delay_reason}:{rsi:.1f})" if delay_reason.startswith("MA") else "지연"
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
            
            history.append({
                'Date': date,
                'Value': current_total_val,
                'Cash_Weight': cash / current_total_val if current_total_val > 0 else 0,
                'Lev_Weight': current_lev_funds / current_total_val if current_total_val > 0 else 0,
                'Asset': f"{current_planned_asset}(S{rebalance_stage})" if 0 < rebalance_stage < 3 else current_planned_asset,
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
