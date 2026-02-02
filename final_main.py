import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from backtest import calculate_metrics

def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0, start_date=None, end_date=None):
    """
    RSI, F&G, VIX를 결합한 'Golden Combination' 전략을 실행합니다.
    - 매수(leverage_asset): RSI <= 35 OR RSI Golden Cross 등
    - 매도(SOXX복귀): RSI 및 MACD 과열 신호 등
    """
    # 데이터 병합 (인덱스 정렬 보장)
    base_df = data_dict[base_asset]
    combined = base_df[['Close', 'RSI']].copy()
    
    # Fear & Greed 및 VIX 데이터의 인덱스를 QQQ와 동일하게 맞춤 (날짜 포맷 정규화 및 중복 제거)
    fg_df.index = pd.to_datetime(fg_df.index)
    vix_df.index = pd.to_datetime(vix_df.index)
    
    fg_clean = fg_df[~fg_df.index.duplicated(keep='first')]
    vix_clean = vix_df[~vix_df.index.duplicated(keep='first')]
    
    combined = combined.join(fg_clean['FearGreed'].rename('FG'), how='left')
    combined = combined.join(vix_clean['Close'].rename('VIX'), how='left')
    # RSI SMA 및 MACD 컬럼 준비
    # pandas_ta MACD 컬럼명 추정 (보통 MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9)
    macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
    signal_col = [c for c in combined.columns if 'MACDs_' in c]
    
    if macd_col and signal_col:
        combined['MACD'] = combined[macd_col[0]]
        combined['Signal_Line'] = combined[signal_col[0]]
    else:
        # 컬럼 없으면 새로 계산
        macd = ta.macd(combined['Close'])
        combined = pd.concat([combined, macd], axis=1)
        combined['MACD'] = combined['MACD_12_26_9']
        combined['Signal_Line'] = combined['MACDs_12_26_9']

    # RSI SMA 계산 (14일)
    combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
    
    # 볼린저 밴드 추가 (20일, 2표준편차)
    bbands = ta.bbands(combined['Close'], length=20, std=2)
    if bbands is not None:
        # 컬럼명 유연하게 매칭 (BBL_*, BBU_*)
        l_col = [c for c in bbands.columns if c.startswith('BBL_')]
        u_col = [c for c in bbands.columns if c.startswith('BBU_')]
        if l_col and u_col:
            combined['BB_Lower'] = bbands[l_col[0]]
            combined['BB_Upper'] = bbands[u_col[0]]
        else:
            combined['BB_Lower'] = 0
            combined['BB_Upper'] = 0
    else:
        combined['BB_Lower'] = 0
        combined['BB_Upper'] = 0
    
    # Market Regime 지표 추가 (SMA50, SMA200)
    combined['SMA50'] = ta.sma(combined['Close'], length=50)
    combined['SMA200'] = ta.sma(combined['Close'], length=200)
    # ADX 추가 (추세 강도 파악용)
    adx = ta.adx(base_df['High'], base_df['Low'], base_df['Close'], length=14)
    if adx is not None:
        combined = pd.concat([combined, adx], axis=1)
        combined['ADX'] = combined['ADX_14']
    else:
        combined['ADX'] = 25 # 기본값
    
    # 이전 값 계산 (Shift)
    combined['Prev_RSI'] = combined['RSI'].shift(1)
    combined['Prev_RSI_SMA'] = combined['RSI_SMA'].shift(1)
    combined['Prev_MACD'] = combined['MACD'].shift(1)
    combined['Prev_Signal_Line'] = combined['Signal_Line'].shift(1)
    
    # 데이터 보정 (결측치 처리)
    combined['FG'] = combined['FG'].ffill().fillna(50)
    combined['VIX'] = combined['VIX'].ffill().fillna(15)
    combined = combined.fillna(0) # 나머지 초기 NaN 0으로 채움
    
    # FG 값 변화 확인을 위한 샘플링 출력
    # print(f"\n최근 Fear & Greed 데이터 샘플:\n{combined['FG'].iloc[-5:].to_string()}")
    # print(f"가장 낮은 FG 값: {combined['FG'].min():.2f}, 가장 높은 FG 값: {combined['FG'].max():.2f}")
    
    dates = combined.index
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
    
    if len(dates) == 0:
        print("경고: 선택한 기간에 해당하는 데이터가 없습니다.")
        return pd.DataFrame(columns=['Date', 'Value', 'Asset']).set_index('Date')

    portfolio_value = 10000.0
    cash = portfolio_value * cash_ratio
    etf_val = portfolio_value * (1 - cash_ratio)
    
    current_planned_asset = base_asset
    previous_asset = base_asset
    rebalance_stage = 0  # 0: 안정, 1: 30%, 2: 65%, 3: 100%
    last_entry_price = 0
    
    holdings = {t: 0.0 for t in data_dict.keys()}
    holdings[base_asset] = etf_val / base_df.loc[dates[0], 'Close']
    
    history = []
    
    for date in dates:
        prices = {t: data_dict[t].loc[date, 'Close'] for t in data_dict.keys()}
        row = combined.loc[date]
        rsi, fg, vix = row['RSI'], row['FG'], row['VIX']
        
        # 추가 지표 로드
        macd_val = row['MACD']
        signal_line = row['Signal_Line']
        rsi_sma = row['RSI_SMA']
        prev_rsi = row['Prev_RSI']
        prev_rsi_sma = row['Prev_RSI_SMA']
        prev_macd = row['Prev_MACD']
        prev_signal_line = row['Prev_Signal_Line']
        
        # 0. 시장 상황(Regime) 판단
        sma50 = row.get('SMA50', 0)
        sma200 = row.get('SMA200', 0)
        adx_val = row.get('ADX', 0)
        
        regime = 'Sideways'
        if not pd.isna(sma200) and not pd.isna(sma50):
            is_bull = row['Close'] > sma200 and sma50 > sma200
            is_bear = row['Close'] < sma200 and sma50 < sma200
            
            if adx_val < 20:
                regime = 'Sideways'
            elif is_bull:
                regime = 'Bull'
            elif is_bear:
                regime = 'Bear'
        
        # 시장 상황에 따른 파라미터 조절 (최적화 결과가 있으면 로드, 없으면 기본값 사용)
        opt_params = None
        if os.path.exists('optimization_results.json'):
            try:
                import json
                with open('optimization_results.json', 'r') as f:
                    all_opt = json.load(f)
                    curr_year = str(date.year)
                    
                    # 1. 우선순위: 해당 연도의 국면 최적값(Yearly)
                    if "Yearly" in all_opt and curr_year in all_opt["Yearly"] and regime in all_opt["Yearly"][curr_year]:
                        opt_params = all_opt["Yearly"][curr_year][regime]
                    # 2. 보조순위: 전체 기간 통합 국면 최적값(Global)
                    elif "Global" in all_opt and regime in all_opt["Global"]:
                        opt_params = all_opt["Global"][regime]
            except:
                pass

        # TEST 데이터 주석 시작
        # 1. 시그널 임계값 (RSI, FG): JSON을 보지 않고 항상 기본값 사용 (사용자 요청)
        # if regime == 'Bull':
        #     buy_rsi_threshold, sell_rsi_threshold, buy_fg_threshold = 35, 75, 30
        # elif regime == 'Bear':
        #     buy_rsi_threshold, sell_rsi_threshold, buy_fg_threshold = 30, 65, 20
        # else: # Sideways
        #     buy_rsi_threshold, sell_rsi_threshold, buy_fg_threshold = 35, 70, 25

        # # 2. 리밸런싱 임계값 (reb_threshold): JSON 최적화 결과가 있으면 로드, 없으면 기본값(0.03) 사용
        # # 고도화: 매수(Buy)/매도(Sell) 및 상승(Up)/하락(Down) 분리
        # if opt_params:
        #     buy_reb_up = opt_params.get('buy_reb_up', 0.03)
        #     buy_reb_down = opt_params.get('buy_reb_down', 0.03)
        #     sell_reb_up = opt_params.get('sell_reb_up', 0.03)
        #     sell_reb_down = opt_params.get('sell_reb_down', 0.03)
        # else:
        #     buy_reb_up = buy_reb_down = sell_reb_up = sell_reb_down = 0.03
        # TEST 데이터 주석 끝
        buy_reb_up = buy_reb_down = sell_reb_up = sell_reb_down = 0.03

        
        # 1. 시그널 판단 (신규 조건 + Regime 가중치)
        price = row['Close']
        bb_l, bb_u = row.get('BB_Lower', 0), row.get('BB_Upper', 0)
        
        is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
        is_macd_improving = (macd_val > prev_macd)
        is_macd_below_signal = (macd_val < signal_line)
        
        # [신규] 볼린저 밴드 + RSI 매수 조건: 주가 <= 하단 밴드 AND RSI < 30
        is_bb_rsi_buy = (price <= bb_l) and (rsi < 35)
        
        is_buy_signal = (is_rsi_golden_cross and is_macd_improving and is_macd_below_signal) or (rsi < 35) 
        # or is_bb_rsi_buy
        
        # 매도 조건 (Regime별 RSI 기준 적용)
        # [신규] 볼린저 밴드 + RSI 매도 조건: 주가 >= 상단 밴드 OR RSI > 70
        is_bb_rsi_sell = (price >= bb_u) or (rsi > 70)
        
        sell_cond_1 = (rsi >= 70) and (rsi < prev_rsi)
        
        # 2. MACD > Signal AND MACD 하락 AND RSI Dead Cross
        is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
        sell_cond_2 = (macd_val > signal_line) and (macd_val < prev_macd) and is_rsi_dead_cross
        
        # 3. MACD Dead Cross (MACD가 Signal Line 하향 돌파)
        is_macd_dead_cross = (prev_macd > prev_signal_line) and (macd_val < signal_line)
        sell_cond_3 = is_macd_dead_cross
        
        is_sell_signal = sell_cond_1 or sell_cond_2 or sell_cond_3 
        # or is_bb_rsi_sell
        
        # 신호 조건 로깅을 위한 변수
        signal_conditions = []
        
        if is_buy_signal:
            new_planned = leverage_asset
            
            # BUY 신호 조건 기록
            if rsi < 35:
                signal_conditions.append(f"RSI < 35 ({rsi:.1f})")
            if is_rsi_golden_cross and is_macd_improving and is_macd_below_signal:
                signal_conditions.append(f"RSI Golden Cross + MACD Improving")
            if is_bb_rsi_buy:
                signal_conditions.append(f"BB Lower Touch + RSI < 30")
                
        elif is_sell_signal:
            new_planned = base_asset
            
            # SELL 신호 조건 기록
            if sell_cond_1:
                signal_conditions.append(f"RSI >= 70 & Declining ({rsi:.1f})")
            if sell_cond_2:
                signal_conditions.append(f"MACD > Signal & Declining + RSI Dead Cross")
            if sell_cond_3:
                signal_conditions.append(f"MACD Dead Cross")
            if is_bb_rsi_sell:
                signal_conditions.append(f"BB Upper Touch or RSI > 70")
                
        else:
            new_planned = current_planned_asset
            
        # 총 가치 계산 (거래 전 가치)
        current_total_val = cash + sum(holdings[t] * prices[t] for t in holdings)
        rebalance_needed = False
        
        # 현재 레버리지 자산 비중 계산
        etf_funds = current_total_val * (1 - cash_ratio)
        current_lev_val = holdings[leverage_asset] * prices[leverage_asset]
        current_lev_pct = current_lev_val / etf_funds if etf_funds > 0 else 0
        
        # 신호에 따른 리밸런싱 로직
        if new_planned != current_planned_asset:
            # 신호 변경 발생
            previous_asset = current_planned_asset
            current_planned_asset = new_planned
            reason = "Signal Change"
            
            if new_planned == leverage_asset:
                signal_type = "BUY"
                # BUY 신호로 변경 시: 현재 비중보다 높은 다음 단계로 진입
                if current_lev_pct < 0.25: # (마진 고려)
                    target_lev_pct = 0.30
                    rebalance_stage = 1
                elif current_lev_pct < 0.65:
                    target_lev_pct = 0.70
                    rebalance_stage = 2
                else:
                    target_lev_pct = 1.0
                    rebalance_stage = 3
            else:
                signal_type = "SELL"
                # SELL 신호로 변경 시: 현재 비중보다 낮은 다음 단계로 진입
                if current_lev_pct > 0.75: # (마진 고려)
                    target_lev_pct = 0.70
                    rebalance_stage = 1
                elif current_lev_pct > 0.35:
                    target_lev_pct = 0.30
                    rebalance_stage = 2
                else:
                    target_lev_pct = 0.0
                    rebalance_stage = 3
            
            last_entry_price = prices[base_asset]
            rebalance_needed = True
            
        # 추가 진입 조건 확인 (설정된 % 상승 또는 하강 시 추가 진입)
        elif rebalance_stage in [1, 2]:
            price_change_ratio = prices[base_asset] / last_entry_price
            
            # 현재 상태(Buy/Sell)에 맞는 임계값 선택
            if current_planned_asset == leverage_asset: # Buy State
                curr_up, curr_down = buy_reb_up, buy_reb_down
            else: # Sell State
                curr_up, curr_down = sell_reb_up, sell_reb_down
                
            # 상향/하향 임계값을 각각 적용
            if price_change_ratio <= (1 - curr_down) or price_change_ratio >= (1 + curr_up):
                # signal check 추가 (해당 방향의 신호가 유지되고 있는지 확인)
                is_signal_active = (current_planned_asset == leverage_asset and is_buy_signal) or \
                                   (current_planned_asset == base_asset and is_sell_signal)
                
                if is_signal_active:
                    rebalance_stage += 1
                    last_entry_price = prices[base_asset]
                    rebalance_needed = True
                    
                    # Stage 번호에 따라 목표 비중 결정
                    if current_planned_asset == leverage_asset:
                        # BUY 상태: Stage 1→30%, Stage 2→70%, Stage 3→100%
                        if rebalance_stage == 2:
                            target_lev_pct = 0.70
                        else:  # Stage 3
                            target_lev_pct = 1.0
                        signal_type = "BUY"
                    else:
                        # SELL 상태: Stage 1→70%, Stage 2→30%, Stage 3→0%
                        if rebalance_stage == 2:
                            target_lev_pct = 0.30
                        else:  # Stage 3
                            target_lev_pct = 0.0
                        signal_type = "SELL"
                    
                    if price_change_ratio <= (1 - curr_down):
                        reason = f"{curr_down*100:.1f}% Price Drop"
                    else:
                        reason = f"{curr_up*100:.1f}% Price Rise"
                
        # 리밸런싱 실행
        if rebalance_needed:
            condition_str = " | ".join(signal_conditions) if signal_conditions else "N/A"
            print(f"[{date.date()}] {signal_type} ({reason}) | Regime: {regime}: Stage {rebalance_stage}, Target: {current_planned_asset}, Price: {prices[current_planned_asset]:.2f}, RSI: {rsi:.1f}, FG: {fg:.1f}, VIX: {vix:.1f}")
            if signal_conditions:
                 print(f"            Conditions: {condition_str}")
            
            new_cash = current_total_val * cash_ratio
            etf_funds = current_total_val * (1 - cash_ratio)
            
            # 레버리지 자산과 QQQ 비중 계산
            lev_val = etf_funds * target_lev_pct
            qqq_val = etf_funds * (1 - target_lev_pct)
            
            # 홀딩스 업데이트
            for t in holdings: holdings[t] = 0
            holdings[leverage_asset] = lev_val / prices[leverage_asset]
            holdings[base_asset] = qqq_val / prices[base_asset]
            
            if target_lev_pct >= 1.0:
                print(f"            Portfolio: {current_total_val:.0f}, {leverage_asset}: 100%")
            else:
                print(f"            Portfolio: {current_total_val:.0f}, {leverage_asset}: {target_lev_pct*100:.0f}%, {base_asset}: {(1-target_lev_pct)*100:.0f}%")
            
            cash = new_cash
        
        asset_label = f"{current_planned_asset}(S{rebalance_stage})" if rebalance_stage < 3 else current_planned_asset
        history.append({'Date': date, 'Value': current_total_val, 'Asset': asset_label})
        
    return pd.DataFrame(history).set_index('Date')

def run_benchmark(data_dict, ticker='QQQ', initial_capital=10000.0, start_date=None, end_date=None):
    """
    단순 보유 (Buy & Hold) 전략을 실행합니다.
    시작일에 전액 매수 후 변동 없이 보유합니다.
    """
    df = data_dict[ticker]
    dates = df.index
    if start_date:
        dates = dates[dates >= pd.to_datetime(start_date)]
    if end_date:
        dates = dates[dates <= pd.to_datetime(end_date)]
        
    if len(dates) == 0:
        return pd.DataFrame(columns=['Date', 'Value', 'Asset']).set_index('Date')
    
    # 선택된 기간의 첫날 종가로 전액 매수 수량 계산
    shares = initial_capital / df.loc[dates[0], 'Close']
    
    history = []
    for date in dates:
        price = df.loc[date, 'Close']
        value = shares * price
        history.append({'Date': date, 'Value': value, 'Asset': ticker})
        
    return pd.DataFrame(history).set_index('Date')

if __name__ == "__main__":
    # print("데이터 로딩 시작...")
    # 스크립트 위치 기준 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    try:
        def read_data(filename):
            # 중앙 데이터 경로 설정 (C:\TestCode\data)
            data_dir = os.path.join(base_dir, "data")
            path = os.path.join(data_dir, filename)
            
            if os.path.exists(path):
                return pd.read_csv(path, index_col=0, parse_dates=True)
            raise FileNotFoundError(f"{filename}을(를) 찾을 수 없습니다.")

        data_dict = {
            'QQQ': read_data('QQQ_data.csv'),
            'QLD': read_data('QLD_data.csv'),
            'TQQQ': read_data('TQQQ_data.csv'),
            'TLT': read_data('TLT_data.csv'),
            'TMF': read_data('TMF_data.csv')
        }
        # print("ETF 데이터 로드 완료.")
        
        # 데이터 정제 (숫자 변환 및 결측치 제거)
        for ticker, df in data_dict.items():
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            data_dict[ticker] = df.dropna(subset=['Close'])
        
        fg_df = read_data('fear_greed_data.csv')
        vix_df = read_data('vix_data.csv')
        # print("보조 지표 데이터 로드 완료.")
        
        # 보조 데이터도 날짜 변환
        fg_df.index = pd.to_datetime(fg_df.index, errors='coerce')
        vix_df.index = pd.to_datetime(vix_df.index, errors='coerce')
    except FileNotFoundError:
        print("데이터 파일이 없습니다. data_loader.py와 gen_mock_data.py를 먼저 실행해 주세요.")
        exit()

    # print("백테스트 시작...")
    years = range(2025, 2027)
    
    annual_results = []
    
    for year in years:
        s_date = f"{year}-01-01"
        e_date = f"{year}-12-31"
        # print(f"\n{'='*20} {year}년 백테스트 {'='*20}")
        
        # 1. 벤치마크 (QQQ)
        bh_history = run_benchmark(data_dict, ticker='QQQ', start_date=s_date, end_date=e_date)
        # 2. Golden Combination 전략 (QLD 사용)
        golden_history = run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0, start_date=s_date, end_date=e_date)
        
        if not bh_history.empty and not golden_history.empty:
            bh_metrics = calculate_metrics(bh_history.rename(columns={'Value': 'PortfolioValue'}))
            gd_metrics = calculate_metrics(golden_history.rename(columns={'Value': 'PortfolioValue'}))
            
            annual_results.append({
                'Year': year,
                'QQQ_ROI': bh_metrics['Cumulative Return'],
                'QQQ_MDD': bh_metrics['MDD'],
                'Strategy_ROI': gd_metrics['Cumulative Return'],
                'Strategy_MDD': gd_metrics['MDD']
            })
            
            print(f"\n[{year} 결과 요약]")
            print(f"Benchmark (QQQ): ROI {bh_metrics['Cumulative Return']:.2%}, MDD {bh_metrics['MDD']:.2%}")
            print(f"Strategy (QLD):  ROI {gd_metrics['Cumulative Return']:.2%}, MDD {gd_metrics['MDD']:.2%}")

    # 전체 기간 (2025-2026) 백테스트
    # print(f"\n{'='*20} 전체 기간 (2025-2026) {'='*20}")
    s_full = '2025-01-01'
    e_full = '2026-12-31'
    bh_full = run_benchmark(data_dict, ticker='QQQ', start_date=s_full, end_date=e_full)
    gd_full = run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0, start_date=s_full, end_date=e_full)
    
    full_bh_metrics = calculate_metrics(bh_full.rename(columns={'Value': 'PortfolioValue'}))
    full_gd_metrics = calculate_metrics(gd_full.rename(columns={'Value': 'PortfolioValue'}))
    
    print(f"\n[전체 기간 결과]")
    print(f"{'전략':<20} | {'ROI':>10} | {'CAGR':>10} | {'MDD':>10}")
    print("-" * 55)
    print(f"{'QQQ Buy & Hold':<20} | {full_bh_metrics['Cumulative Return']:>10.2%} | {full_bh_metrics['CAGR']:>10.2%} | {full_bh_metrics['MDD']:>10.2%}")
    print(f"{'Regime Strategy':<20} | {full_gd_metrics['Cumulative Return']:>10.2%} | {full_gd_metrics['CAGR']:>10.2%} | {full_gd_metrics['MDD']:>10.2%}")

    # 시각화 저장
    plt.figure(figsize=(12, 7))
    plt.plot(bh_full['Value'] / bh_full['Value'].iloc[0], label='QQQ Buy & Hold', color='gray', alpha=0.5)
    plt.plot(gd_full['Value'] / gd_full['Value'].iloc[0], label='Regime Strategy (QLD)', color='blue', linewidth=2)
    plt.title('Strategy Performance Comparison (2015-2025)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('regime_strategy_result.png')
    print("\n최종 결과 차트가 regime_strategy_result.png로 저장되었습니다.")
