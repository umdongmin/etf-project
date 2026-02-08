import streamlit as st

# [중요] Streamlit 설정은 모든 출력 및 다른 라이브러리 실행보다 먼저 호출되어야 함
st.set_page_config(page_title="ETF Golden Strategy", page_icon="📈", layout="wide")

import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import yfinance as yf
import requests
import datetime
import matplotlib.dates as mdates
import platform

import matplotlib.font_manager as fm

# 한글 폰트 설정 (운영체제별)
def set_korean_font():
    sys_name = platform.system()
    if sys_name == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif sys_name == 'Darwin': # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else: # Linux (Streamlit Cloud 등)
        # 나눔 폰트 설치 시도 확인 및 가용한 한글 폰트 탐색
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if 'NanumGothic' in available_fonts:
            plt.rcParams['font.family'] = 'NanumGothic'
        elif 'NanumBarunGothic' in available_fonts:
            plt.rcParams['font.family'] = 'NanumBarunGothic'
        elif 'DejaVu Sans' in available_fonts:
            plt.rcParams['font.family'] = 'DejaVu Sans' # 최후의 보루
        else:
            # 폰트 경로 직접 탐색 (Debian/Ubuntu 계열)
            nanum_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
            if os.path.exists(nanum_path):
                fm.fontManager.addfont(nanum_path)
                plt.rcParams['font.family'] = 'NanumGothic'
            else:
                st.warning("⚠️ 차트 한글 폰트를 찾을 수 없습니다. 배포 환경에서 나눔 폰트 설치가 필요할 수 있습니다.")

set_korean_font()
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# --- Performance Metrics Logic (Merged from backtest.py) ---

def calculate_metrics(history):
    """성과 지표 계산 함수 (MDD, CAGR 등)"""
    if history.empty:
        return {'Final Value': 0, 'Cumulative Return': 0, 'CAGR': 0, 'MDD': 0}
    
    df = history.copy()
    if 'PortfolioValue' not in df.columns and 'Value' in df.columns:
        df.rename(columns={'Value': 'PortfolioValue'}, inplace=True)
        
    df['Daily_Return'] = df['PortfolioValue'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    # CAGR 계산
    days = (df.index[-1] - df.index[0]).days
    if days <= 0: return {'Final Value': df['PortfolioValue'].iloc[-1], 'Cumulative Return': 0, 'CAGR': 0, 'MDD': 0}
    
    years = days / 365.25
    final_value = df['PortfolioValue'].iloc[-1]
    initial_value = df.get('PortfolioValue', pd.Series([10000.0])).iloc[0]
    if initial_value == 0: initial_value = 10000.0
    
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    # MDD 계산
    df['Rolling_Max'] = df['PortfolioValue'].cummax()
    df['Drawdown'] = df['PortfolioValue'] / df['Rolling_Max'] - 1
    mdd = df['Drawdown'].min()
    
    # 샤프 지수 및 변동성 계산 (연율화)
    daily_std = df['Daily_Return'].std()
    if daily_std > 0:
        ann_vol = daily_std * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0

    # MAR 지수 계산
    mar = cagr / abs(mdd) if mdd != 0 else 0
    
    # 승률 및 손익비 계산
    winning_days = df[df['Daily_Return'] > 0]['Daily_Return']
    losing_days = df[df['Daily_Return'] < 0]['Daily_Return']
    
    win_rate = len(winning_days) / len(df.dropna()) if len(df.dropna()) > 0 else 0
    
    gross_profit = winning_days.sum()
    gross_loss = abs(losing_days.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0)
    
    # 최대 회복 기간 계산 (일 수)
    under_water = df['PortfolioValue'] < df['Rolling_Max']
    recovery_periods = []
    current_period = 0
    for is_down in under_water:
        if is_down:
            current_period += 1
        else:
            if current_period > 0:
                recovery_periods.append(current_period)
            current_period = 0
    if current_period > 0: recovery_periods.append(current_period)
    max_recovery = max(recovery_periods) if recovery_periods else 0
    
    return {
        'Final Value': final_value,
        'Cumulative Return': df['Cumulative_Return'].iloc[-1] if not df['Cumulative_Return'].empty else 0,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe,
        'MAR': mar,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Max_Recovery': max_recovery
    }

# --- Core Logic Classes ---

class DataService:
    """데이터 수집 및 전처리를 담당하는 클래스"""
    @staticmethod
    def calculate_indicators(df):
        if 'Close' not in df.columns: return df
        close_prices = pd.to_numeric(df['Close'], errors='coerce')
        df['RSI'] = ta.rsi(close_prices, length=14)
        df['RSI_SMA'] = df['RSI'].rolling(window=14).mean()
        
        macd = ta.macd(close_prices)
        if macd is not None:
            # MACD 컬럼명 정규화
            m_col = [c for c in macd.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
            s_col = [c for c in macd.columns if 'MACDs_' in c]
            if m_col and s_col:
                df['MACD'] = macd[m_col[0]]
                df['Signal_Line'] = macd[s_col[0]]
            else:
                df = pd.concat([df, macd], axis=1)
        
        # 볼린저 밴드 추가 (20일, 2표준편차)
        bbands = ta.bbands(close_prices, length=20, std=2)
        if bbands is not None:
            # 컬럼명 유연하게 매칭 (BBL_*, BBU_*)
            l_col = [c for c in bbands.columns if c.startswith('BBL_')]
            u_col = [c for c in bbands.columns if c.startswith('BBU_')]
            if l_col and u_col:
                df['BB_Lower'] = bbands[l_col[0]]
                df['BB_Upper'] = bbands[u_col[0]]
        return df

    @classmethod
    def fetch_live_data(cls):
        print("라이브 데이터 수집 시작...")
        # [수정] 배포 서버(UTC)와 로컬(KST) 간의 날짜 차이 해결을 위해 KST(UTC+9) 강제 적용
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        end_date = now_kst.strftime('%Y-%m-%d')
        start_date = '2010-01-01'
        
        data_dict = {}
        tickers = ['QQQ', 'QLD', 'TQQQ', 'SOXX', 'USD', 'SOXL', 'TLT', 'TMF']
        # [수정] 여러 번 호출하지 않고 한 번에 묶어서 다운로드 (Rate Limit 회피)
        try:
            full_data = yf.download(tickers + ['^VIX'], start=start_date, end=end_date, progress=False, group_by='ticker')
            
            for ticker in tickers:
                if ticker in full_data and not full_data[ticker].empty:
                    df = full_data[ticker].copy()
                    # [수정] 결측치(NaN) 전파 방지를 위해 ffill/bfill 적용
                    df = df.ffill().bfill() 
                    df = cls.calculate_indicators(df)
                    data_dict[ticker] = df
            
            vix_df = full_data['^VIX'].copy() if '^VIX' in full_data else pd.DataFrame()
        except Exception as e:
            print(f"Main Download Error: {e}")
            vix_df = pd.DataFrame()

        fg_df = pd.DataFrame()
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()
            series = data.get('fear_and_greed_historical', {}).get('data', [])
            fg_df = pd.DataFrame(series)
            fg_df['x'] = pd.to_datetime(fg_df['x'], unit='ms')
            fg_df.rename(columns={'x': 'Date', 'y': 'FearGreed'}, inplace=True)
            fg_df.set_index('Date', inplace=True)
        except:
            fg_df = pd.DataFrame(columns=['FearGreed'])

        macro_data_map = {}
        fetch_status = {'ETF': 'Success', 'VIX': 'Pending', 'FearGreed': 'Pending', 'Macro': 'Pending'}
        
        # 매크로 지표도 한 번에 묶어서 다운로드
        m_tickers_raw = ['^TNX', '^IRX', '^PCCR']
        try:
            m_data = yf.download(m_tickers_raw, period='5d', progress=False, group_by='ticker')
            m_map = {'^TNX': 'US10Y', '^IRX': 'US03M', '^PCCR': 'PCCR'}
            for t_raw, t_name in m_map.items():
                if t_raw in m_data and not m_data[t_raw].empty:
                    val = m_data[t_raw]['Close'].dropna().iloc[-1]
                    macro_data_map[t_name] = val
            fetch_status['Macro'] = 'Success'
        except:
            fetch_status['Macro'] = 'Error'
        
        if not vix_df.empty:
            macro_data_map['VIX'] = vix_df['Close'].dropna().iloc[-1]
            fetch_status['VIX'] = 'Success'
        if not fg_df.empty: fetch_status['FearGreed'] = 'Success'
        
        macro_df = pd.DataFrame([macro_data_map])
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_all_data():
        return DataService.fetch_live_data()

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @staticmethod
    def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_date, end_date, params=None, trade_at="종가", smart_params=None):
        if base_asset not in data_dict: return pd.DataFrame()
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
        
        macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
        signal_col = [c for c in combined.columns if 'MACDs_' in c]
        if macd_col and signal_col:
            combined['MACD'], combined['Signal_Line'] = combined[macd_col[0]], combined[signal_col[0]]
        else:
            macd = ta.macd(combined['Close'])
            combined = pd.concat([combined, macd], axis=1)
            combined['MACD'], combined['Signal_Line'] = combined['MACD_12_26_9'], combined['MACDs_12_26_9']
        
        combined['MACD_Hist'] = combined['MACD'] - combined['Signal_Line']
        
        combined['MACD_Hist'] = combined['MACD'] - combined['Signal_Line']
        combined['Prev2_MACD_Hist'] = combined['MACD_Hist'].shift(2)
        
        combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
        combined['SMA200'] = combined['Close'].rolling(window=200).mean()
        combined['SMA60'] = combined['Close'].rolling(window=60).mean()
        for col in ['RSI', 'RSI_SMA', 'MACD', 'Signal_Line', 'SMA200', 'SMA60', 'MACD_Hist']:
            combined[f'Prev_{col}'] = combined[col].shift(1)
        
        combined['FG'] = combined['FG'].ffill().fillna(50)
        combined['VIX'] = combined['VIX'].ffill().fillna(15)
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
        if len(dates) == 0: return pd.DataFrame()

        portfolio_value, cash = 10000.0, 10000.0 * cash_ratio
        current_planned_asset, rebalance_stage, last_entry_price = base_asset, 0, 0
        peak_price, peak_macd = 0, 0
        holdings = {t: 0.0 for t in data_dict.keys()}
        
        # [수정] 첫날 가격이 NaN인 상황에 대비해 가장 빠른 유효가를 찾아 초기 주식 수 계산
        initial_price = base_df['Close'].asof(dates[0])
        if pd.isna(initial_price):
            initial_price = base_df['Close'].dropna().iloc[0] if not base_df['Close'].dropna().empty else 1.0
            
        holdings[base_asset] = (portfolio_value * (1 - cash_ratio)) / initial_price
        history = []
        
        # [추가] 지연 실행용 상태 변수
        pending_rebalance = None # {'target_lev_pct': val, 'rebalance_stage': val, 'planned': asset}
        
        for date in dates:
            row = combined.loc[date]
            price = row['Close']
            
            # [수정] 익일 시가 매매 처리: 전날 신호가 있었다면 오늘 시가로 먼저 체결
            if trade_at == "익일 시가" and pending_rebalance is not None:
                # 오늘 시가(Open)로 체결 (데이터가 없는 경우 종가로 대체, 그것도 없으면 건너뜀)
                try:
                    semi_group = ['SOXX', 'USD', 'SOXL']
                    is_semi = (base_asset in semi_group or leverage_asset in semi_group)
                    target_group = semi_group if is_semi else ['QQQ', 'QLD', 'TQQQ']
                    
                    trade_prices = {}
                    # [수정] 모든 스위칭 후보 자산의 시가를 확보
                    for t in list(set([base_asset, leverage_asset] + target_group)):
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
                
                # [수정] 순차적 포트폴리오 리밸런싱 (Natural Transition) 익일 시가 처리
                if smart_params and smart_params.get('use_sequential', True):
                    is_turbo_mode = (t_lev_asset == target_group[2])
                    
                    # 1. 현재 자산 평가 (trade_prices 기준)
                    h_vals = {t: holdings[t] * trade_prices[t] for t in holdings if t in trade_prices}
                    curr_total_lev_val = sum(h_vals.get(t, 0) for t in target_group[1:]) # 현재 합산 레버리지
                    
                    # 2. 목표 레버리지 총액
                    normalized_pct = t_lev_pct / 100.0 if t_lev_pct > 1 else t_lev_pct
                    target_lev_val = etf_funds * normalized_pct
                    
                    new_h_vals = {t: 0.0 for t in holdings}
                    
                    if curr_total_lev_val > target_lev_val + 0.01:
                        # 감축(Sell) 상황: QLD부터 채우고 남은 자리를 TQQQ로 채움 (TQQQ 우선 매도)
                        new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                        rem = target_lev_val - new_h_vals[target_group[1]]
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                    else:
                        # 증액(Buy) 상황
                        if is_turbo_mode:
                            # 가속(Turbo): QLD -> TQQQ 스왑 우선
                            new_h_vals[target_group[2]] = h_vals.get(target_group[2], 0) # 기존 TQQQ 유지
                            rem = target_lev_val - new_h_vals[target_group[2]]
                            
                            swap_n = min(h_vals.get(target_group[1], 0), rem)
                            new_h_vals[target_group[2]] += swap_n
                            rem -= swap_n
                            
                            from_b = min(h_vals.get(base_asset, 0), rem)
                            new_h_vals[target_group[2]] += from_b
                            
                            # 남은 QLD 유지 (이론상 0)
                            new_h_vals[target_group[1]] = h_vals.get(target_group[1], 0) - swap_n
                        else:
                            # 일반(Normal): TQQQ/QLD 유지 후 부족분 QLD 매수
                            new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), target_lev_val)
                            rem = target_lev_val - new_h_vals[target_group[2]]
                            
                            new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)
                            rem -= new_h_vals[target_group[1]]
                            
                            from_b = min(h_vals.get(base_asset, 0), rem)
                            new_h_vals[target_group[1]] += from_b
                    
                    # 3. 나머지는 Base 자산
                    allocated_lev = sum(new_h_vals.values())
                    new_h_vals[base_asset] = etf_funds - allocated_lev
                    
                    # holdings 업데이트
                    for t in holdings:
                        if t in trade_prices and trade_prices[t] > 0:
                            holdings[t] = new_h_vals.get(t, 0) / trade_prices[t]
                        else:
                            holdings[t] = 0
                else:
                    # [기존] 표준 리밸런싱 (단일 자산 올인)
                    for t in list(holdings.keys()): holdings[t] = 0
                    l_pct = t_lev_pct / 100.0 if t_lev_pct > 1 else t_lev_pct
                    if t_lev_asset == base_asset:
                        holdings[base_asset] = etf_funds / trade_prices[base_asset]
                    else:
                        holdings[t_lev_asset] = (etf_funds * l_pct) / trade_prices[t_lev_asset]
                        holdings[base_asset] = (etf_funds * (1 - l_pct)) / trade_prices[base_asset]
                
                current_planned_asset = t_lev_asset
                rebalance_stage = pending_rebalance['rebalance_stage']
                last_entry_price = trade_prices[base_asset]
                pending_rebalance = None # 처리 완료
            
            rsi, fg, vix, macd_val, signal_line, rsi_sma = row['RSI'], row['FG'], row['VIX'], row['MACD'], row['Signal_Line'], row['RSI_SMA']
            macd_hist = row['MACD_Hist']
            bb_l, bb_u = row.get('BB_Lower', 0), row.get('BB_Upper', 0)
            prev_rsi, prev_rsi_sma, prev_macd, prev_signal_line = row['Prev_RSI'], row['Prev_RSI_SMA'], row['Prev_MACD'], row['Prev_Signal_Line']
            prev_macd_hist = row['Prev_MACD_Hist']
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
                # sell_cond_3 = (macd_hist > 0) and (prev_macd_hist > prev2_macd_hist) and (macd_hist < prev_macd_hist) 
                # sell_cond_4 = (macd_hist > 0) and (price >= peak_price * 0.98) and (peak_price > 0) and (macd_val < peak_macd * 0.8)
                sell_cond_5 = (prev_macd_hist > 0) and (macd_hist < 0)
                
                is_sell_signal = False
                if sell_cond_1: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI꺾임)"
                elif sell_cond_2: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI데드크로스+MACD)"
                # elif sell_cond_3:
                #     is_sell_signal = True
                #     signal_reason = "기본(히스토그램 기울기급변)"
                # elif sell_cond_4:
                #     is_sell_signal = True
                #     signal_reason = "기본(가격-MACD 다이버전스)"
                elif sell_cond_5: 
                    is_sell_signal = True
                    signal_reason = "기본(MACD데드크로스)"
                
                b_up, b_down, s_up, s_down = 0.03, -0.03, 0.03, -0.03

            # [수정] 자산 그룹 결정 (Natural Transition용 및 에러 방지)
            semi_group = ['SOXX', 'USD', 'SOXL']
            nasdaq_group = ['QQQ', 'QLD', 'TQQQ']
            bond_group = ['TLT', 'TLT', 'TMF'] # 채권은 1x, 1x, 3x로 임시 매핑 (2배수가 없으므로)
            
            if base_asset in semi_group or leverage_asset in semi_group:
                target_group = semi_group
            elif base_asset in bond_group or leverage_asset in bond_group:
                target_group = bond_group
            else:
                target_group = nasdaq_group

            # [신규] 스마트 레버리지 자산 결정
            effective_lev_asset = leverage_asset
            smart_reason = ""
            if smart_params and smart_params.get('use_smart'):
                # 1단(Safe), 2단(Normal), 3단(Turbo)
                if vix > smart_params.get('vix_exit', 31):
                    effective_lev_asset = target_group[0]
                    smart_reason = f"스마트(VIX대피:{vix:.1f})"
                elif rsi < smart_params.get('rsi_turbo', 31):
                    effective_lev_asset = target_group[2]
                    smart_reason = f"스마트(RSI가속:{rsi:.1f})"
                else:
                    effective_lev_asset = target_group[1]
                    smart_reason = "스마트(Normal)"

            # 현재 가격 추출 (전략 실행용)
            try:
                # [수정] 현재 보유 중인 자산들과 신호용 기준 자산, 그리고 타겟 자산의 가격을 확보
                currently_holding = [t for t, qty in holdings.items() if qty > 0]
                check_targets = list(set([base_asset, leverage_asset, effective_lev_asset] + currently_holding + target_group))
                prices = {t: data_dict[t].loc[date, 'Close'] for t in check_targets if t in data_dict and date in data_dict[t].index}
                
                # 기준 자산 가격이 없으면 데이터 공백이므로 패스
                if base_asset not in prices:
                    continue
                # 현재 보유한 주식의 가격 정보가 하나라도 누락되면 가치 평가가 불가능하므로 패스
                if any(t not in prices for t in currently_holding):
                    continue
            except (KeyError, ValueError):
                continue # 현재 날짜에 자산 데이터가 없으면 건너뜀
            
            current_total_val = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings if t in prices)
            rebalance_needed = False
            
            # [수정] 변수 초기화 (에러 방지)
            new_planned = current_planned_asset
            new_stage = rebalance_stage
            
            # [신규] 첫 거래 기준가 초기화
            if last_entry_price == 0:
                last_entry_price = prices[base_asset]

            etf_funds = current_total_val * (1 - cash_ratio)
            # [수정] 모든 레버리지 후보 자산의 합산 비중 계산 (스마트 모드 대응)
            lev_candidates = ['QLD', 'TQQQ', 'SOXL', 'USD', 'TMF']
            current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            current_lev_pct = current_lev_funds / etf_funds if etf_funds > 0 else 0
            
            # [수정] 자산/단계 결정 및 통합 가드 적용
            rebalance_cause = "" # 매매 원인 저장
            is_delayed = False # 지연 여부 플래그
            delay_reason = ""
            
            # 가격 변동 체크 (기준 자산 기반)
            price_change = prices[base_asset] / last_entry_price
            price_diff_pct = (price_change - 1) * 100
            
            # [수정] 파라미터 부호 설정에 관계없이 하락/상승폭을 절대값으로 체크하도록 강건화
            up_thresh = abs(b_up if current_planned_asset == leverage_asset else s_up)
            down_thresh = abs(b_down if current_planned_asset == leverage_asset else s_down)
            
            price_trigger = (price_change >= 1 + up_thresh or price_change <= 1 - down_thresh)
            
            if is_buy_signal and current_planned_asset != effective_lev_asset and effective_lev_asset in prices:
                # 1. 자산 전환 시도 (매수 방향)
                new_planned = effective_lev_asset
                if effective_lev_asset == base_asset:
                    # [중요] 세이프 모드(1배수 대피) 시에는 레버리지 비중을 0으로 강제
                    target_lev_pct, new_stage = 0.0, 3
                else:
                    if current_lev_pct < 0.25: target_lev_pct, new_stage = 0.30, 1
                    elif current_lev_pct < 0.65: target_lev_pct, new_stage = 0.70, 2
                    else: target_lev_pct, new_stage = 1.0, 3
                rebalance_cause = "시그널"
                rebalance_needed = True
            elif is_sell_signal and current_planned_asset != base_asset:
                # 2. 자산 전환 시도 (매도 방향)
                new_planned = base_asset
                if current_lev_pct > 0.75: target_lev_pct, new_stage = 0.70, 1
                elif current_lev_pct > 0.35: target_lev_pct, new_stage = 0.30, 2
                else: target_lev_pct, new_stage = 0.0, 3
                rebalance_cause = "시그널"
                rebalance_needed = True
            elif rebalance_stage in [1, 2]:
                # 3. 동일 자산 내 추가 단계 진행 (가격 변동 OR MACD 데드크로스 가속)
                is_accelerated_sell = False
                # is_accelerated_sell = (current_planned_asset == base_asset and sell_cond_5)
                # [수정] 가격 트리거 방향성 및 모드 일관성 체크
                # 1. 매수(증액) 중: 현재 타고 있는 말(current_planned_asset)과 지금 타야 하는 말(effective_lev_asset)이 같을 때만 허용
                # 2. 매도(감축) 중: 리스크 관리가 최우선이므로 모드와 상관없이 가격 조건 맞으면 즉시 단계 진행
                is_buy_process = (current_planned_asset != base_asset)
                is_mode_consistent = (current_planned_asset == effective_lev_asset)
                
                # 매도 중이거나, 매수 중이면서 모드가 일치할 때만 가격 트리거 작동
                if ((price_trigger and (not is_buy_process or is_mode_consistent)) or is_accelerated_sell):
                    new_stage = rebalance_stage + 1
                    target_lev_pct = (0.70 if new_stage == 2 else 1.0) if is_buy_process else (0.30 if new_stage == 2 else 0.0)
                    rebalance_cause = "MACD데드크로스집행" if is_accelerated_sell else "가격조건"
                    rebalance_needed = True

            # [통합 가드] S1, S2, S3 진입 시 리스크 관리 적용 (자산 전환 포함)
            if rebalance_needed:
                # 1. 이평선 방어막 체크 (레버리지 자산 진입/증액 시)
                if new_planned != base_asset and smart_params and smart_params.get('use_panic', False):
                    panic_ma = smart_params.get('panic_ma', 200)
                    ma_val = combined.loc[date, f'SMA{panic_ma}']
                    if prices[base_asset] < ma_val:
                        # 단계별 RSI 제한선 결정
                        if new_stage == 1:
                            target_rsi = smart_params.get('panic_rsi_s1', 32)
                        elif new_stage == 2:
                            target_rsi = smart_params.get('panic_rsi_s2', 32)
                        else: # S3
                            target_rsi = smart_params.get('panic_rsi_s3', 30)
                        
                        if rsi > target_rsi:
                            rebalance_needed, is_delayed, delay_reason = False, True, f"MA{panic_ma}-RSI"
                
                # 2. 시그널 결합 옵션 체크
                if rebalance_needed and smart_params and smart_params.get('couple_signal', False):
                    # 결합 옵션 ON일 때는 [가격변동] AND [시그널]이 동시 충족되어야 함
                    required_sig = is_buy_signal if current_planned_asset == leverage_asset else is_sell_signal
                    if not (required_sig and price_trigger):
                        rebalance_needed, is_delayed, delay_reason = False, True, "시그널결합"

            if rebalance_needed:
                if trade_at == "종가":
                    # [수정] 순차적 포트폴리오 리밸런싱 (Natural Transition) 종가 처리
                    if smart_params and smart_params.get('use_sequential', True):
                        is_turbo_mode = (effective_lev_asset == target_group[2])
                        
                        # 1. 현재 자산 평가
                        h_vals = {t: holdings[t] * prices[t] for t in holdings if t in prices}
                        curr_total_lev_val = sum(h_vals.get(t, 0) for t in target_group[1:]) # 현재 합산 레버리지
                        
                        # 2. 목표 레버리지 총액
                        target_lev_val = etf_funds * target_lev_pct
                        
                        # 가중치 초기화
                        new_h_vals = {t: 0.0 for t in holdings}
                        
                        if curr_total_lev_val > target_lev_val + 0.01:
                            # 감축(Sell) 상황: QLD부터 채우고 남은 자리를 TQQQ로 채움 (TQQQ 우선 매도)
                            new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                            rem = target_lev_val - new_h_vals[target_group[1]]
                            new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                        else:
                            # 증액(Buy) 상황
                            if is_turbo_mode:
                                # 가속(Turbo): QLD -> TQQQ 스왑 우선
                                new_h_vals[target_group[2]] = h_vals.get(target_group[2], 0) # 기존 TQQQ 유지
                                rem = target_lev_val - new_h_vals[target_group[2]]
                                
                                swap_n = min(h_vals.get(target_group[1], 0), rem)
                                new_h_vals[target_group[2]] += swap_n
                                rem -= swap_n
                                
                                from_b = min(h_vals.get(base_asset, 0), rem)
                                new_h_vals[target_group[2]] += from_b
                                
                                # 남은 QLD 유지 (이론상 0)
                                new_h_vals[target_group[1]] = h_vals.get(target_group[1], 0) - swap_n
                            else:
                                # 일반(Normal): TQQQ/QLD 유지 후 부족분 QLD 매수
                                new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), target_lev_val)
                                rem = target_lev_val - new_h_vals[target_group[2]]
                                
                                new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)
                                rem -= new_h_vals[target_group[1]]
                                
                                from_b = min(h_vals.get(base_asset, 0), rem)
                                new_h_vals[target_group[1]] += from_b
                        
                        # 3. 나머지는 Base 자산
                        allocated_lev = sum(new_h_vals.values())
                        new_h_vals[base_asset] = etf_funds - allocated_lev
                        
                        for t in holdings:
                            if t in prices and prices[t] > 0:
                                holdings[t] = new_h_vals.get(t, 0) / prices[t]
                            else:
                                holdings[t] = 0
                    else:
                        # [기존] 표준 리벨런싱 (단일 자산 올인)
                        # [추가] 매도 시에는 자산 성격 유지 (Case 3 대응)
                        # 현재 보유한 레버리지 자산 찾기
                        curr_lev_asset = next((t for t in target_group[1:] if holdings.get(t,0) > 0), effective_lev_asset)
                        
                        # 매도 상황(감축)이면 현재 자산 유지, 매수(증액) 상황이면 목표 자산(effective_lev_asset) 사용
                        active_asset = curr_lev_asset if current_lev_pct > target_lev_pct + 0.01 else effective_lev_asset
                        
                        for t in list(holdings.keys()): holdings[t] = 0
                        if active_asset == base_asset:
                            holdings[base_asset] = etf_funds / prices[base_asset]
                        else:
                            holdings[active_asset] = (etf_funds * target_lev_pct) / prices[active_asset]
                            holdings[base_asset] = (etf_funds * (1 - target_lev_pct)) / prices[base_asset]
                    
                    current_planned_asset, rebalance_stage = new_planned, new_stage
                    if trade_at == "종가":
                        last_entry_price = prices[base_asset]
                        # [추가] 매매 후 현금 및 레버리지 자산 총액 재계산 (로그용)
                        cash = current_total_val * cash_ratio
                        current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
                else:
                    # 익일 시가 대기
                    pending_rebalance = {'target_lev_pct': target_lev_pct, 'rebalance_stage': new_stage, 'planned': new_planned}
                    # [신규] '익일 시가' 매매 시에도 기준가격 계산의 베이스는 '신호 발생일 종가'로 고정하여 
                    # 주말/휴일 사이의 갭에 의한 조기 트리거 방지
                    last_entry_price = prices[base_asset]

            # 시각적 가독성을 위해 trade_label 설정
            if rebalance_needed:
                prefix = "매도" if new_planned == base_asset else "매수"
                # [신규] 가격조건 매매 시 실제 변동폭 표시
                condition_label = f"가격조건({price_diff_pct:+.1f}%)" if rebalance_cause == "가격조건" else (signal_reason if rebalance_cause == "시그널" else "MACD데드크로스집행")
                trade_label = f"{prefix}(S{new_stage}): {condition_label}"
                if smart_params and smart_params.get('use_smart') and smart_reason:
                    trade_label += f" [{smart_reason}]"
            elif is_delayed:
                # [원복] 지연 원인 표시
                if delay_reason.startswith("MA"):
                    trade_label = f"지연({delay_reason}:{rsi:.1f})"
                else:
                    trade_label = f"지연(시그널결합)"
            elif rebalance_stage in [1, 2]:
                trade_label = f"진행중(S{rebalance_stage})"
            else:
                trade_label = "중립"
            
            # [신규] 비중 계산 로직 (가독성 향상 및 에러 방지)
            if current_planned_asset == base_asset:
                # 매도 중 (base_asset으로 복귀 중)
                curr_w = 0.70 if rebalance_stage == 1 else (0.30 if rebalance_stage == 2 else 0.0)
            else:
                # 매수 중 (leverage_asset으로 진입 중)
                curr_w = 0.30 if rebalance_stage == 1 else (0.70 if rebalance_stage == 2 else 1.0)
            
            # [수정] 실제 비중이 가장 큰 자산을 로그에 표시
            main_asset = base_asset
            max_w = (holdings.get(base_asset, 0) * prices[base_asset]) / current_total_val if current_total_val > 0 else 0
            for t in target_group[1:]:
                w = (holdings.get(t, 0) * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
                if w > max_w:
                    max_w = w
                    main_asset = t

            # [수정] 개별 레버리지 자산 비중 추가
            history_item = {
                'Date': date, 
                'Close': prices[base_asset],
                'Value': current_total_val, 
                'Cash': cash,
                'Trade_Label': trade_label,
                'Asset': f"{main_asset}(S{rebalance_stage})" if rebalance_stage < 3 else main_asset,
                'Target': f"{new_planned}(S{new_stage})" if new_stage < 3 else new_planned,
                'Lev_Weight': current_lev_funds / current_total_val if current_total_val > 0 else 0,
                f'{base_asset}_Weight': (holdings.get(base_asset, 0) * prices[base_asset]) / current_total_val if current_total_val > 0 else 0,
                'Cash_Weight': cash / current_total_val if current_total_val > 0 else 0,
                'RSI': rsi, 'FG': fg, 'VIX': vix, 'MACD': macd_val, 'Signal_Line': signal_line,
                'SMA200': combined.loc[date, 'SMA200'],
                'SMA60': combined.loc[date, 'SMA60']
            }
            # 타겟 그룹의 개별 레버리지 자산 비중 동적 추가 (QLD_Weight, TQQQ_Weight 등)
            for t in target_group[1:]:
                history_item[f'{t}_Weight'] = (holdings.get(t, 0) * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
            
            history.append(history_item)
        return pd.DataFrame(history).set_index('Date')

    @staticmethod
    def run_benchmark(data_dict, ticker, start_date, end_date):
        if ticker not in data_dict: return pd.DataFrame()
        df = data_dict[ticker].copy()
        
        # [수정] 데이터 공백(NaN) 제거 및 유효 범위 한정
        df = df.dropna(subset=['Close'])
        dates = df.index
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame()
        
        shares = 10000.0 / df.loc[dates[0], 'Close']
        history = [{'Date': date, 'Value': shares * df.loc[date, 'Close']} for date in dates]
        return pd.DataFrame(history).set_index('Date')

    @staticmethod
    def run_monte_carlo(history, days=252, n_sims=200):
        """과거 변동성을 기반으로 미래 수익률 시뮬레이션"""
        if history.empty or len(history) < 30: return None
        
        returns = history['Value'].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        last_val = history['Value'].iloc[-1]
        
        sim_results = np.zeros((days, n_sims))
        for i in range(n_sims):
            # Geometric Brownian Motion (GBM)
            daily_returns = np.random.normal(mu, sigma, days)
            price_paths = last_val * np.exp(np.cumsum(daily_returns))
            sim_results[:, i] = price_paths
            
        return sim_results

# --- View Classes ---

class MarketView:
    """주요 마켓 이슈 화면 UI 구성 클래스"""
    @staticmethod
    def render(data_dict, macro_df, fetch_status, fetch_time, fg_df):
        st.header("📰 주요 마켓 이슈 및 밸류어이션")
        st.caption(f"최종 데이터 업데이트: {fetch_time}")
        
        st.subheader("📊 시장 밸류에이션 및 심리")
        latest_fg = fg_df['FearGreed'].iloc[-1] if not fg_df.empty else 50
        
        v1, v2, v3 = st.columns(3)
        if 'QQQ' in data_dict and not data_dict['QQQ'].empty:
            latest_qqq = data_dict['QQQ']['Close'].iloc[-1]
            v1.metric("나스닥 100 PER", f"{(latest_qqq / 15.64):.1f}x", delta="평균 24~26x 대비 높음", delta_color="inverse")
            v2.metric("버핏 지수 (추정)", f"{(223.6 * (latest_qqq / 520.0)):.1f}%", delta="200% 이상 과열", delta_color="inverse")
        else:
            v1.metric("나스닥 100 PER", "N/A")
            v2.metric("버핏 지수 (추정)", "N/A")
        v3.metric("Fear & Greed Index", f"{latest_fg:.0f}", delta="Sentiment", delta_color="off")
        
        st.divider()
        st.subheader("💡 핵심 체크 지표 (Market Radar)")
        m1, m2, m3 = st.columns(3)
        l10y = macro_df.get('US10Y', pd.Series([0.0])).iloc[-1]
        if l10y > 15: l10y /= 10.0
        m1.metric("미 국채 10년물 금리", f"{l10y:.2f}%")
        m2.metric("VIX 공포 지수", f"{macro_df.get('VIX', pd.Series([15.0])).iloc[-1]:.2f}")
        
        pccr_val = macro_df.get('PCCR')
        pccr_display = f"{pccr_val.iloc[-1]:.2f}" if pccr_val is not None and not pccr_val.empty else "N/A"
        m3.metric("풋/콜 비율 (PCCR)", pccr_display)
        
        st.divider()
        # 3. 주요 일정 캘린더
        st.subheader("📅 주요 경제 일정")
        today = datetime.date.today()

        fomc_events = [
            {"date": datetime.date(2025, 12, 10), "actual": "5.50%", "consensus": "5.50%"},
            {"date": datetime.date(2026, 1, 29), "actual": None, "consensus": "5.50% (동결)"},
            {"date": datetime.date(2026, 3, 19), "actual": None, "consensus": "5.25% (인하)"},
        ]
        cpi_events = [
            {"date": datetime.date(2025, 12, 12), "actual": "3.1%", "consensus": "3.1%"},
            {"date": datetime.date(2026, 1, 13), "actual": "3.1%", "consensus": "3.0%"},
            {"date": datetime.date(2026, 2, 12), "actual": None, "consensus": "2.9%"},
        ]

        def get_past_and_next(events, today):
            past = [e for e in events if e['date'] < today]
            future = [e for e in events if e['date'] >= today]
            return (past[-1] if past else None), (future[0] if future else None)

        col_fomc, col_cpi = st.columns(2)
        with col_fomc:
            st.markdown("**🏦 FOMC 금리 결정**")
            p, n = get_past_and_next(fomc_events, today)
            if p: st.info(f"◀ 직전 ({p['date']}): {p['actual']}")
            if n: st.success(f"▶ 예정 ({n['date']}): {n['consensus']}")

        with col_cpi:
            st.markdown("**📈 CPI 소비자물가**")
            p, n = get_past_and_next(cpi_events, today)
            if p: st.info(f"◀ 직전 ({p['date']}): {p['actual']}")
            if n: st.success(f"▶ 예정 ({n['date']}): {n['consensus']}")

        with st.expander("💻 빅테크 주요 실적발표", expanded=True):
            st.markdown("- **1월 5주차 (1/26~1/30)** : MSFT, TSLA, AAPL 등")
            st.markdown("- **4월 4주차** : 1분기 실적 발표 시즌")

        st.divider()
        with st.expander("🛠️ 데이터 수집 상태 정보", expanded=False):
            st.table(pd.DataFrame([fetch_status]).T)

class BacktestView:
    """백테스트 결과 표시 UI 구성 클래스"""
    @staticmethod
    def render_results(golden_history, bh_histories, base_asset, leverage_asset):
        if golden_history.empty:
            st.warning("⚠️ 선택하신 기간에 대한 백테스트 결과가 없습니다. (자산 상장일 이전이거나 데이터 수집 오류)")
            return
            
        latest = golden_history.iloc[-1]
        st.subheader(f"📍 상태 요약 ({latest.name.date()})")
        
        # [수정] Trade_Label 사용 (매도 우선 체크하여 과매수 오매칭 방지)
        sig = str(latest['Trade_Label']) if 'Trade_Label' in latest else "중립"
        if sig.startswith("매도"): sig_color = "#3498db" # Blue
        elif sig.startswith("매수"): sig_color = "#e74c3c" # Red
        elif "진행중" in sig: sig_color = "#f39c12" # Orange
        else: sig_color = "#7f8c8d" # Gray (중립)
        
        st.markdown(f'<div style="background-color:{sig_color}; padding:20px; border-radius:10px; text-align:center; color:white;">'
                    f'<h2 style="margin:0;">{sig} 상태</h2>'
                    f'<b>{leverage_asset} {latest["Lev_Weight"]:.0%}, {base_asset} {latest[f"{base_asset}_Weight"]:.0%}</b></div>', unsafe_allow_html=True)
        
        metrics = {k: calculate_metrics(v.rename(columns={'Value': 'PortfolioValue'})) for k, v in {**bh_histories, 'Strategy': golden_history}.items()}
        s_m = metrics['Strategy']
        
        # 그룹별 지표 렌더링
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown("##### 📈 수익 및 성장")
            st.metric("누적 수익률", f"{s_m['Cumulative Return']:.1%}", help="전체 투자 기간 동안의 총 수익률입니다.")
            st.metric("CAGR", f"{s_m['CAGR']:.1%}", help="연평균 복리 성장률입니다.")
            st.metric("최종 자산", f"${s_m['Final Value']:,.0f}", help="10,000달러로 시작했을 때의 최종 금액입니다.")

        with m_col2:
            st.markdown("##### 🛡️ 리스크 및 방어")
            st.metric("MDD", f"{s_m['MDD']:.1%}", help="전고점 대비 최대 하락폭입니다. (가장 고통스러운 구간)")
            st.metric("MAR Index", f"{s_m['MAR']:.2f}", help="CAGR을 MDD로 나눈 값으로, 위험 대비 수익률을 나타냅니다. (0.5 이상 양호, 1.0 이상 우수)")
            st.metric("최대 회복 기간", f"{s_m['Max_Recovery']}일", help="하락 후 전고점을 탈환하는 데 걸린 가장 긴 기간입니다.")

        with m_col3:
            st.markdown("##### ⚡ 매매 효율성")
            st.metric("Sharpe Ratio", f"{s_m['Sharpe']:.2f}", help="변동성(리스크) 대비 수익 효율입니다. (1.0 이상 우수)")
            st.metric("승률 (일별)", f"{s_m['Win_Rate']:.1%}", help="전체 거래일 중 수익이 난 날의 비중입니다.")
            st.metric("손익비", f"{s_m['Profit_Factor']:.2f}", help="총 이익을 총 손실로 나눈 값입니다. (1.0 이상이면 수익적)")
        
        st.subheader("성과 비교 차트")
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, v in bh_histories.items(): ax.plot(v['Value']/10000, label=k, alpha=0.5)
        ax.plot(golden_history['Value']/10000, label='Strategy', linewidth=2, color='blue')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # [신규] 월간 수익률 히트맵 (비교 기능 포함)
        BacktestView.render_returns_heatmap(golden_history, bh_histories)

        # [신규] 몬테카를로 시뮬레이션
        BacktestView.render_monte_carlo(golden_history)

        st.divider()
        st.subheader("전략 성과 요약")
        sum_df = pd.DataFrame([{"전략": k, "수익률": f"{v['Cumulative Return']:.2%}", "CAGR": f"{v['CAGR']:.2%}", "MDD": f"{v['MDD']:.2%}", "Sharpe": f"{v['Sharpe']:.2f}", "MAR": f"{v['MAR']:.2f}", "승률": f"{v['Win_Rate']:.1%}", "손익비": f"{v['Profit_Factor']:.2f}"} for k, v in metrics.items()])
        st.table(sum_df.set_index("전략"))
        
        # [신규] 전체 거래 내역 확인 및 색상 강조
        st.divider()
        with st.expander("📝 전체 일별 상세 로그 확인 (시그널 포함)"):
            def style_signal(row):
                sig = str(row['Trade_Label']) if 'Trade_Label' in row else ""
                # 배경색 및 텍스트색 결정 (매도 우선 체크하여 과매수 오매칭 방지)
                if sig.startswith("매도"):
                    return ['background-color: rgba(52, 152, 219, 0.1); color: #3498db; font-weight: bold'] * len(row)
                elif sig.startswith("매수"):
                    return ['background-color: rgba(231, 76, 60, 0.1); color: #e74c3c; font-weight: bold'] * len(row)
                elif "진행중" in sig:
                    return ['background-color: rgba(243, 156, 18, 0.05); color: #f39c12'] * len(row)
                return [''] * len(row)

            # 로그 데이터 가공
            log_df = golden_history.copy()
            log_df.index = log_df.index.date
            
            # 가독성을 위해 컬럼 순서 조정 및 정리 (QLD_Weight, TQQQ_Weight 등 동적 포함)
            base_w_col = f'{base_asset}_Weight'
            lev_group_cols = [f'{t}_Weight' for t in ['QLD', 'TQQQ', 'USD', 'SOXL', 'TMF'] if f'{t}_Weight' in log_df.columns]
            cols = ['Value', 'Trade_Label', 'Asset', 'Lev_Weight'] + lev_group_cols + [base_w_col, 'Cash_Weight', 'RSI', 'MACD', 'VIX']
            
            display_df = log_df[[c for c in cols if c in log_df.columns]].sort_index(ascending=False)
            
            # 포맷팅 딕셔너리 동적 생성
            formats = {'Value': '${:,.0f}', 'Lev_Weight': '{:.1%}', 'Cash_Weight': '{:.1%}', 'RSI': '{:.1f}', 'MACD': '{:.2f}', 'VIX': '{:.1f}'}
            formats.update({c: '{:.1%}' for c in lev_group_cols + [base_w_col]})

            styled_log = display_df.style.apply(style_signal, axis=1)\
                                  .format(formats)
            
            st.dataframe(styled_log, use_container_width=True, height=600)

    @staticmethod
    def render_returns_heatmap(strategy_history, bh_histories):
        """연도별/월별 수익률 히트맵 렌더링 (전략 vs 벤치마크 3단 통합)"""
        st.divider()
        st.subheader("📅 월간 수익률 히트맵 비교 (%)")
        
        if strategy_history.empty:
            st.info("데이터가 충분하지 않습니다.")
            return

        def get_heatmap_data(history):
            if history.empty: return None
            monthly_val = history['Value'].resample('ME').last()
            initial_series = pd.Series([10000.0], index=[monthly_val.index[0] - pd.DateOffset(months=1)])
            monthly_combined = pd.concat([initial_series, monthly_val])
            monthly_ret = monthly_combined.pct_change().dropna() * 100
            
            df_ret = monthly_ret.to_frame(name='Return')
            df_ret['Year'] = df_ret.index.year
            df_ret['Month'] = df_ret.index.month
            
            pivot = df_ret.pivot(index='Year', columns='Month', values='Return')
            pivot = pivot.reindex(columns=range(1, 13))
            
            annual_ret = history['Value'].resample('YE').last().pct_change() * 100
            annual_ret.iloc[0] = (history['Value'].resample('YE').last().iloc[0] / 10000.0 - 1) * 100
            pivot['Annual'] = annual_ret.values
            return pivot

        def plot_it(pivot, title, ax, is_diff=False):
            vmin, vmax = (-10, 10) if not is_diff else (-5, 5)
            cmap = 'RdYlGn' if not is_diff else 'PiYG'
            
            im = ax.imshow(pivot.iloc[:, :-1], cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(12))
            ax.set_xticklabels(['1M', '2M', '3M', '4M', '5M', '6M', '7M', '8M', '9M', '10M', '11M', '12M'])
            ax.set_yticks(np.arange(len(pivot)))
            ax.set_yticklabels(pivot.index)
            
            for i in range(len(pivot)):
                for j in range(12):
                    val = pivot.iloc[i, j]
                    if not np.isnan(val):
                        color = 'white' if abs(val) > 7 else 'black'
                        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=8)
                ann_val = pivot.iloc[i, -1]
                ax.text(12.2, i, f'{ann_val:.1f}%', ha='left', va='center', weight='bold', color='blue' if ann_val > 0 else 'red', fontsize=9)
            
            ax.set_title(title, fontsize=12, pad=10, weight='bold')
            ax.spines[:].set_visible(False)
            ax.set_xticks(np.arange(13)-0.5, minor=True); ax.set_yticks(np.arange(len(pivot)+1)-0.5, minor=True)
            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)

        try:
            b_name = list(bh_histories.keys())[0] if bh_histories else "Benchmark"
            b_history = bh_histories[b_name] if bh_histories else pd.DataFrame()

            s_pivot = get_heatmap_data(strategy_history)
            b_pivot = get_heatmap_data(b_history)
            
            if s_pivot is None:
                st.info("데이터가 부족하여 차트를 그릴 수 없습니다.")
                return

            common_years = s_pivot.index.intersection(b_pivot.index) if b_pivot is not None else s_pivot.index
            s_view = s_pivot.loc[common_years]
            b_view = b_pivot.loc[common_years] if b_pivot is not None else None
            
            num_charts = 3 if b_view is not None else 1
            fig, axes = plt.subplots(num_charts, 1, figsize=(12, (len(common_years) * 0.7 + 1.5) * num_charts))
            if num_charts == 1: axes = [axes]
            
            plot_it(s_view, f"🔥 My Strategy Monthly Returns (%)", axes[0])
            if b_view is not None:
                plot_it(b_view, f"📊 Benchmark ({b_name}) Monthly Returns (%)", axes[1])
                diff_view = s_view - b_view
                plot_it(diff_view, "✨ Strategy Alpha (Strategy - Benchmark) (%)", axes[2], is_diff=True)
            
            plt.tight_layout(pad=3.0)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
            if b_view is not None:
                st.caption(f"※ Alpha 매트릭스: [내 전략 수익률] - [{b_name} 수익률]. 양수(초록)일수록 시장 대비 초과 성과를 기록했음을 의미합니다.")

        except Exception as e:
            st.error(f"히트맵 생성 중 오류 발생: {e}")

    @staticmethod
    def render_monte_carlo(history):
        """몬테카를로 시뮬레이션 결과 시각화"""
        st.divider()
        st.subheader("🎲 미래 자산 시뮬레이션 (Monte Carlo)")
        st.caption("과거 전략의 평균 수익률과 변동성을 바탕으로 향후 1년(252거래일)의 시나리오 200개를 생성합니다.")
        
        sim_data = StrategyEngine.run_monte_carlo(history)
        if sim_data is None:
            st.info("시뮬레이션을 위한 데이터가 부족합니다.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(sim_data.shape[1]):
            ax.plot(sim_data[:, i], color='gray', alpha=0.1, linewidth=1)
            
        # 백분위수 계산
        p5 = np.percentile(sim_data[-1, :], 5)
        p50 = np.percentile(sim_data[-1, :], 50)
        p95 = np.percentile(sim_data[-1, :], 95)
        
        ax.axhline(sum(sim_data[0,:])/sim_data.shape[1], color='red', linestyle='--', label='Current', alpha=0.5)
        ax.plot(np.median(sim_data, axis=1), color='blue', linewidth=2, label='Median (50th)')
        ax.set_title("Future Asset Value Projection (1 Year)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # 통계 요약
        c1, c2, c3 = st.columns(3)
        c1.metric("상위 5% (낙관)", f"${p95:,.0f}")
        c2.metric("중간값 (기대)", f"${p50:,.0f}")
        c3.metric("하위 5% (비관)", f"${p5:,.0f}")
        
        last_val = history['Value'].iloc[-1]
        probs_above = (sim_data[-1, :] > last_val).mean()
        st.info(f"💡 분석 결과: 향후 1년 뒤 자산이 현재(${last_val:,.0f})보다 높을 확률은 약 **{probs_above:.1%}** 입니다.")

class HistoryLabView:
    """역사적 마켓 랩 화면 UI 구성 클래스"""
    
    EVENTS = {
        2010: "플래시 크래시, 유럽 재정 위기 시작",
        2011: "미국 국가신용등급 강등, 그리스 디폴트 우려",
        2012: "유로존 붕괴 위기 고조, 페이스북 IPO",
        2013: "테이퍼 탠트럼 (긴축 발작), 양적완화 축소 논의",
        2014: "유가 급락, 알리바바 역대 최대 IPO",
        2015: "중국 증시 폭락, 미국 7년 만의 금리 인상",
        2016: "브렉시트 투표, 트럼프 대통령 당선",
        2017: "FAANG 주도 기술주 랠리, 비트코인 열풍",
        2018: "미중 무역 전쟁, 연준의 지속적 금리 인상",
        2019: "장단기 금리 역전, 레포 금리 급등",
        2020: "코로나19 팬데믹, 사상 최대 유동성 공급",
        2021: "포스트 팬데믹 인플레이션 시작, 공급망 병목",
        2022: "40년 만의 고물가, 공격적 금리 인상 (Bear Market)",
        2023: "SVB 은행 파산 위기, 생성형 AI (ChatGPT) 열풍",
        2024: "AI 반도체 잭팟, 금리 인하 기대감 형성"
    }

    @staticmethod
    def render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, params, smart_params=None):
        st.header("📜 역사적 마켓 랩 (Historical Market Lab)")
        st.caption("2010년부터 현재까지의 시장 역사를 전략적 관점에서 분석합니다.")
        
        # 전체 기간 데이터 확보
        start_date = datetime.date(2010, 1, 1)
        end_date = datetime.date.today()
        
        with st.spinner('전 기간 역사적 데이터 시뮬레이션 중...'):
            golden_history = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, 0, start_date, end_date, params, trade_at, smart_params=smart_params)
            
            # [수정] 섹터별 동적 벤치마크 로드 (나스닥 vs 반도체)
            semi_group = ['SOXX', 'USD', 'SOXL']
            is_semi = base_asset in semi_group or leverage_asset in semi_group
            b1, b2, b3 = ("SOXX", "USD", "SOXL") if is_semi else ("QQQ", "QLD", "TQQQ")
            
            bh_1 = StrategyEngine.run_benchmark(data_dict, b1, start_date, end_date)
            bh_2 = StrategyEngine.run_benchmark(data_dict, b2, start_date, end_date)
            bh_3 = StrategyEngine.run_benchmark(data_dict, b3, start_date, end_date)
            
        if golden_history.empty:
            st.warning("분석할 데이터가 충분하지 않습니다.")
            return

        # 실제 시뮬레이션 시작일 표시
        actual_start = golden_history.index.min().date()
        if actual_start > start_date:
            st.info(f"ℹ️ {leverage_asset} 등의 자산 상장일 관계로 분석이 **{actual_start}**부터 시작되었습니다.")
        
        # 리스크 관리 설정 표시
        if smart_params and smart_params.get('use_panic'):
            panic_ma = smart_params.get('panic_ma', 200)
            st.info(f"🛡️ **하락장 대응 활성**: 주가 < SMA{panic_ma}일 때 매수를 RSI {smart_params.get('panic_rsi_s1', 32)}/{smart_params.get('panic_rsi_s2', 32)}/{smart_params.get('panic_rsi_s3', 30)} 이하로 제한합니다.")
        
        # [추가] 벤치마크 누락 경고
        missingbench = [t for t in [b1, b2, b3] if t not in data_dict]
        if missingbench:
            st.warning(f"⚠️ {', '.join(missingbench)} 데이터 수집 오류로 일부 벤치마크 정보가 표시되지 않을 수 있습니다.")

        tab1, tab2, tab3, tab4 = st.tabs(["📅 연도별 성과 & 이벤트", "🇺🇸 미 대선 주기 분석", "🗳️ 중간선거 주기", "🍂 월별 계절성"])
        
        with tab1:
            HistoryLabView.render_yearly_table(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3), smart_params=smart_params)
            
        with tab2:
            HistoryLabView.render_election_cycle(golden_history, bh_1, b_name=b1)
            
        with tab3:
            HistoryLabView.render_midterm_analysis(golden_history, bh_1, b_name=b1)

        with tab4:
            HistoryLabView.render_seasonality(golden_history, bh_1, b_name=b1)

    @staticmethod
    def render_yearly_table(s_h, b1_h, b2_h, b3_h, b_names=("QQQ", "QLD", "TQQQ"), smart_params=None):
        st.subheader(f"연도별 성과 및 리스크 분석 (vs {b_names[0]}군)")
        
        def calculate_mdd(series):
            s = series.dropna()
            if s.empty: return 0
            roll_max = s.cummax()
            dd = (s - roll_max) / roll_max
            return dd.min()

        years = sorted(s_h.index.year.unique())
        data = []
        
        for y in years:
            # 해당 연도 데이터 슬라이싱 (안전한 접근을 위해 helper 정의)
            def safe_get_yearly(df, year):
                if df.empty or 'Value' not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
                    return pd.Series(dtype='float64')
                try:
                    return df[df.index.year == year]['Value']
                except:
                    return pd.Series(dtype='float64')

            s_y = safe_get_yearly(s_h, y)
            q_y = safe_get_yearly(b1_h, y)
            l_y = safe_get_yearly(b2_h, y)
            t_y = safe_get_yearly(b3_h, y)

            # 수익률 계산 (연초 대비 혹은 전년 말 대비)
            # 여기서는 단순화를 위해 해당 연도 내의 (최종/최초 - 1) 사용
            # 실제로는 resample().pct_change() 가 더 정확할 수 있음
            def get_ret(ser):
                s = ser.dropna()
                if s.empty or len(s) < 2: return 0
                return (s.iloc[-1] / s.iloc[0] - 1) * 100

            s_ret = get_ret(s_y)
            q_ret = get_ret(q_y)
            l_ret = get_ret(l_y)
            t_ret = get_ret(t_y)
            
            s_mdd = calculate_mdd(s_y) * 100
            q_mdd = calculate_mdd(q_y) * 100
            l_mdd = calculate_mdd(l_y) * 100
            t_mdd = calculate_mdd(t_y) * 100
            
            # [신규] 방어(Panic Mode) 발생 일수 계산
            panic_days = 0
            if 'Trade_Label' in s_h.columns:
                s_y_full = s_h[s_h.index.year == y]
                panic_ma = smart_params.get('panic_ma', 200) if smart_params else 200
                panic_days = s_y_full['Trade_Label'].str.contains(f"지연\(MA{panic_ma}", na=False).sum()
            
            event = HistoryLabView.EVENTS.get(y, "-")
            
            data.append({
                "연도": f"{y}년",
                "전략 수익": f"{s_ret:+.1f}%",
                f"{b_names[0]} 수익": f"{q_ret:+.1f}%",
                f"{b_names[1]} 수익": f"{l_ret:+.1f}%",
                "전략 MDD": f"{s_mdd:.1f}%",
                f"{b_names[0]} MDD": f"{q_mdd:.1f}%",
                "방어(일)": f"{panic_days}일" if panic_days > 0 else "-",
                "주요 이벤트": event
            })
            
        df = pd.DataFrame(data).set_index("연도")
        
        # 스타일링 함수 정의
        def style_returns(val):
            try:
                num = float(val.replace('%','').replace('+',''))
                color = '#00c853' if num >= 0 else '#ff1744' # 초록(양수), 빨강(음수)
                return f'color: {color}; font-weight: bold;'
            except:
                return ''

        def style_mdd(val):
            try:
                if isinstance(val, str):
                    num_str = val.replace('%','')
                    if 'nan' in num_str.lower(): return ''
                    num = float(num_str)
                else:
                    num = float(val)
                
                if np.isnan(num): return ''
                
                # MDD는 낮을수록(음수 폭이 클수록) 더 진한 빨강계열 배경
                alpha = min(abs(num)/50.0, 1.0) * 0.4
                return f'background-color: rgba(255, 23, 68, {alpha});'
            except:
                return ''

        def style_panic(val):
            if val == "-": return ''
            return 'background-color: rgba(0, 200, 83, 0.2); color: #00c853; font-weight: bold;'

        styled_df = df.style.applymap(style_returns, subset=[c for c in df.columns if "수익" in c])\
                            .applymap(style_mdd, subset=[c for c in df.columns if "MDD" in c])\
                            .applymap(style_panic, subset=["방어(일)"])
        
        st.dataframe(styled_df, use_container_width=True, height=520)

        # 하단 요약 (연평균 수익률, 누적 수익률)
        st.divider()
        st.markdown("### 📊 전체 기간 통합 요약 (2010~현재)")
        
        def get_summary(ser, name):
            if ser.empty: return None
            total_ret = (ser.iloc[-1] / 10000.0 - 1) * 100
            years_count = (ser.index[-1] - ser.index[0]).days / 365.25
            cagr = ((ser.iloc[-1] / 10000.0) ** (1/years_count) - 1) * 100 if years_count > 0 else 0
            mdd = calculate_mdd(ser) * 100
            return {"구분": name, "누적 수익률": f"{total_ret:+.1f}%", "연평균(CAGR)": f"{cagr:.1f}%", "최대낙폭(MDD)": f"{mdd:.1f}%"}

        summary_data = [
            get_summary(s_h['Value'], "내 전략 🔥"),
            get_summary(b1_h['Value'] if not b1_h.empty else pd.Series(), f"{b_names[0]}(1x)"),
            get_summary(b2_h['Value'] if not b2_h.empty else pd.Series(), f"{b_names[1]}(2x)"),
            get_summary(b3_h['Value'] if not b3_h.empty else pd.Series(), f"{b_names[2]}(3x)")
        ]
        
        summary_df = pd.DataFrame([s for s in summary_data if s]).set_index("구분")
        st.dataframe(summary_df, use_container_width=True)

    @staticmethod
    def render_election_cycle(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"미 대통령 임기 주기별 평균 성과 (Strategy vs {b_name})")
        st.caption(f"대선 1년차(취임)부터 4년차(대선)까지의 전략과 시장({b_name}) 성과를 비교합니다.")
        
        def get_cycle_data(hist, col_name):
            if hist.empty or 'Value' not in hist.columns or not isinstance(hist.index, pd.DatetimeIndex):
                return pd.Series(dtype='float64')
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['YearType'] = [(year - 2009) % 4 + 1 for year in df.index.year]
                return df.groupby('YearType')[col_name].mean() * 100
            except:
                return pd.Series(dtype='float64')

        s_avg = get_cycle_data(strategy_hist, 'Strategy').reindex([1, 2, 3, 4], fill_value=0.0)
        b_avg = get_cycle_data(bench_hist, 'QQQ').reindex([1, 2, 3, 4], fill_value=0.0)
        
        order = [1, 2, 3, 4]
        # b_avg가 비어있을 수 있으므로 대응
        plot_df = pd.DataFrame({'내 전략': s_avg, '나스닥(QQQ)': b_avg}).reindex(order).fillna(0)
        plot_df.index = [f"{i}년차" for i in order]

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_df.plot(kind='bar', ax=ax, color=['#00c853', '#66b3ff'], alpha=0.8)
        ax.set_title("Strategy vs QQQ by Election Cycle Year")
        ax.set_ylabel("Average Annual Return (%)")
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 값 표시
        for i in range(len(plot_df)):
            ax.text(i - 0.15, plot_df.iloc[i, 0] + 1, f"{plot_df.iloc[i, 0]:.1f}%", ha='center', fontsize=9, weight='bold', color='green')
            ax.text(i + 0.15, plot_df.iloc[i, 1] + 1, f"{plot_df.iloc[i, 1]:.1f}%", ha='center', fontsize=9, weight='bold', color='blue')
            
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        st.markdown("""
        *   **1년차 (취임)**: 신정부 정책 기대감 반영
        *   **2년차 (중간선거)**: 중간선거 해, 불확실성과 불만 표출로 변동성 심화 구간
        *   **3년차 (상승가속)**: 전통적으로 가장 강력한 랠리가 발생하는 해 (재선 목적 부양책)
        *   **4년차 (대선)**: 대선 당일 전후 불확실성 해소 및 랠리
        """)

    @staticmethod
    def render_midterm_analysis(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"중간선거 주기 집중 분석 (Strategy vs {b_name})")
        st.caption(f"중간선거 해(2년차)의 전략과 시장({b_name}) 성과를 직접 비교합니다.")
        
        def process_midterm(hist, col_name):
            if hist.empty or 'Value' not in hist.columns or not isinstance(hist.index, pd.DatetimeIndex):
                return pd.DataFrame(columns=[col_name, 'IsMidterm'])
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['IsMidterm'] = [((y - 2009) % 4 + 1) == 2 for y in df.index.year]
                return df
            except:
                return pd.DataFrame(columns=[col_name, 'IsMidterm'])

        s_df = process_midterm(strategy_hist, 'Strategy')
        b_df = process_midterm(bench_hist, b_name)
        
        # 1. 중간선거 vs 비중간선거 평균 수익률 비교
        s_avg = s_df.groupby('IsMidterm')['Strategy'].mean() * 100
        b_avg = b_df.groupby('IsMidterm')[b_name].mean() * 100
        
        # [수정] 데이터가 부족해도 인덱스 갯수를 맞추기 위해 reindex 사용
        avg_comp = pd.DataFrame({'내 전략': s_avg, f'시장({b_name})': b_avg}).reindex([False, True], fill_value=0.0)
        avg_comp.index = ["비중간선거", "중간선거"]
        
        # 2. 역대 중간선거 연도별 개별 성과
        s_mid = s_df[s_df['IsMidterm']].copy()
        s_mid['Year'] = s_mid.index.year
        b_mid = b_df[b_df['IsMidterm']].copy()
        b_mid['Year'] = b_mid.index.year
        
        mid_yearly = pd.merge(s_mid[['Year', 'Strategy']], b_mid[['Year', b_name]], on='Year')
        mid_yearly = mid_yearly.set_index('Year') * 100

        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**평균 성과 비교**")
            fig, ax = plt.subplots(figsize=(6, 8))
            avg_comp.plot(kind='bar', ax=ax, color=['#00c853', '#66b3ff'], alpha=0.8)
            ax.set_ylabel("Avg Annual Return (%)")
            
            for i in range(len(avg_comp)):
                ax.text(i - 0.15, avg_comp.iloc[i, 0] + (0.5 if avg_comp.iloc[i, 0] >= 0 else -1.5), f"{avg_comp.iloc[i, 0]:.1f}%", ha='center', fontsize=9, weight='bold', color='green')
                ax.text(i + 0.15, avg_comp.iloc[i, 1] + (0.5 if avg_comp.iloc[i, 1] >= 0 else -1.5), f"{avg_comp.iloc[i, 1]:.1f}%", ha='center', fontsize=9, weight='bold', color='blue')
                
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            
        with col2:
            st.write(f"**역대 중간선거 연도별 성과 비교 (vs {b_name})**")
            fig, ax = plt.subplots(figsize=(10, 5))
            mid_yearly.rename(columns={b_name: f'시장({b_name})'}).plot(kind='bar', ax=ax, color=['#00c853', '#66b3ff'], alpha=0.8)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylabel("Annual Return (%)")
            ax.set_xticklabels([f"{y}년" for y in mid_yearly.index], rotation=0)
            
            for i in range(len(mid_yearly)):
                ax.text(i - 0.15, mid_yearly.iloc[i, 0] + (1 if mid_yearly.iloc[i, 0] >= 0 else -3), f"{mid_yearly.iloc[i, 0]:.1f}%", ha='center', fontsize=9, weight='bold', color='green')
                ax.text(i + 0.15, mid_yearly.iloc[i, 1] + (1 if mid_yearly.iloc[i, 1] >= 0 else -3), f"{mid_yearly.iloc[i, 1]:.1f}%", ha='center', fontsize=9, weight='bold', color='blue')
                
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)

    @staticmethod
    def render_seasonality(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"지난 15년간의 월별 평균 수익률 & 승률 (vs {b_name})")
        
        def get_monthly_stats(hist):
            full_months = range(1, 13)
            if hist.empty or 'Value' not in hist.columns or not isinstance(hist.index, pd.DatetimeIndex):
                return pd.Series(index=full_months, data=0.0), pd.Series(index=full_months, data=0.0)
            try:
                returns = hist['Value'].pct_change().dropna()
                if returns.empty:
                    return pd.Series(index=full_months, data=0.0), pd.Series(index=full_months, data=0.0)
                df_rets = returns.to_frame(name='Return')
                df_rets['Month'] = df_rets.index.month
                avg = df_rets.groupby('Month')['Return'].mean() * 100 * 21
                win = df_rets.groupby('Month')['Return'].apply(lambda x: (x > 0).mean()) * 100
                return avg.reindex(full_months, fill_value=0.0), win.reindex(full_months, fill_value=0.0)
            except:
                return pd.Series(index=full_months, data=0.0), pd.Series(index=full_months, data=0.0)

        s_avg, s_win = get_monthly_stats(strategy_hist)
        b_avg, _ = get_monthly_stats(bench_hist)
        
        plot_df = pd.DataFrame({'내 전략': s_avg, '나스닥(QQQ)': b_avg})
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 월간 수익률 막대 (그룹형)
        plot_df.plot(kind='bar', ax=ax1, color=['#00c853', '#66b3ff'], alpha=0.7)
        ax1.set_ylabel("Avg Monthly Return (%)", color='black')
        ax1.set_xticks(range(12))
        ax1.set_xticklabels([f"{m}월" for m in range(1, 13)], rotation=0)
        ax1.axhline(0, color='black', linewidth=0.8)
        
        # 전략 승률 라인
        ax2 = ax1.twinx()
        ax2.plot(range(12), s_win.values, color='#ff1744', marker='o', linewidth=2.5, label='내 전략 승률')
        ax2.set_ylabel("Strategy Win Rate (%)", color='#ff1744')
        ax2.set_ylim(40, 70)
        
        # 범례 합치기
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(f"Strategy vs {b_name} Seasonality Analysis")
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        st.info("💡 **전략의 파란색/초록색 막대**가 QQQ보다 높다면, 해당 월에 시장 대비 초과 수익을 내는 경향이 있음을 의미합니다.")


class ChartView:
    """차트 통합 분석 화면 UI 구성 클래스"""
    @staticmethod
    def render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, params, smart_params=None):
        st.header("📊 차트 통합 분석 (Indicator Correlation)")
        st.caption("VIX, RSI, MACD와 가격 간의 상관관계를 시각적으로 분석하여 최적의 진입 시점을 도출합니다.")
        
        # 필터링 설정
        c1, c2 = st.columns([1, 2])
        current_year = datetime.date.today().year
        selected_year = c1.selectbox("📅 분석 연도 선택", range(current_year, 2009, -1), index=0)
        
        start_date = datetime.date(selected_year, 1, 1)
        end_date = datetime.date(selected_year, 12, 31)
        
        with st.spinner(f'{selected_year}년 데이터 분석 중...'):
            history = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, 0, start_date, end_date, params, trade_at, smart_params=smart_params)
            
        if history.empty:
            st.warning(f"{selected_year}년에 해당하는 충분한 데이터가 없습니다.")
            return

        # 차트 생성 (3개 패널)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        plt.subplots_adjust(hspace=0.1)
        
        # 1. 가격 차트 필드
        ax1.plot(history.index, history['Close'], color='#66b3ff', label=f'{base_asset} Price', alpha=0.8, linewidth=1.5)
        panic_ma = smart_params.get('panic_ma', 200) if smart_params else 200
        if f'SMA{panic_ma}' in history.columns:
            ax1.plot(history.index, history[f'SMA{panic_ma}'], color='#f39c12', label=f'SMA{panic_ma}', alpha=0.6, linestyle='--')
        ax1.set_title(f"{selected_year}년 {base_asset} 흐름 및 매매 시그널 (MA{panic_ma} 기준)", fontsize=14, weight='bold')
        ax1.set_ylabel("Price ($)")
        ax1.grid(True, linestyle='--', alpha=0.4)
        
        # 매매 마커 표시
        buy_points = history[history['Trade_Label'].str.contains('매수', na=False)]
        sell_points = history[history['Trade_Label'].str.contains('매도', na=False)]
        delay_points = history[history['Trade_Label'].str.contains('지연', na=False)]
        
        if not buy_points.empty:
            ax1.scatter(buy_points.index, buy_points['Close'], marker='^', color='green', s=100, label='Buy (Stage)', zorder=5)
        if not sell_points.empty:
            ax1.scatter(sell_points.index, sell_points['Close'], marker='v', color='red', s=100, label='Sell (Stage)', zorder=5)
        
        # 지연 구간 하이라이트 (SMA200-RSI 제약)
        has_delay = False
        panic_ma = smart_params.get('panic_ma', 200) if smart_params else 200
        for idx in delay_points.index:
            ax1.axvspan(idx, idx + datetime.timedelta(days=1), color='yellow', alpha=0.3, label=f'매수 지연(MA{panic_ma}-RSI)' if not has_delay else "")
            has_delay = True

        ax1.legend(loc='upper left', fontsize=9)

        # 2. VIX 차트 필드
        ax2.plot(history.index, history['VIX'], color='#ff9999', label='VIX Index', linewidth=1.5)
        v_exit = smart_params.get('vix_exit', 29) if smart_params else 29
        p_vix = smart_params.get('panic_vix', 35) if smart_params else 35
        ax2.axhline(v_exit, color='red', linestyle='--', alpha=0.6, label=f'VIX Exit ({v_exit})')
        ax2.axhline(p_vix, color='orange', linestyle='-', alpha=0.8, label=f'Panic Threshold ({p_vix})')
        ax2.fill_between(history.index, history['VIX'], p_vix, where=(history['VIX'] >= p_vix), color='red', alpha=0.2)
        ax2.set_ylabel("VIX")
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc='upper left', fontsize=9)

        # 3. RSI 차트 필드
        ax3.plot(history.index, history['RSI'], color='#9b59b6', label='RSI (14)', linewidth=1.2)
        r_turbo = smart_params.get('rsi_turbo', 31) if smart_params else 31
        p_r1 = smart_params.get('panic_rsi_s1', 32) if smart_params else 32
        p_r2 = smart_params.get('panic_rsi_s2', 32) if smart_params else 32
        p_r3 = smart_params.get('panic_rsi_s3', 30) if smart_params else 30
        
        ax3.axhline(70, color='red', linestyle='--', alpha=0.4)
        ax3.axhline(35, color='green', linestyle='--', alpha=0.4)
        ax3.axhline(r_turbo, color='blue', linestyle='--', alpha=0.6, label=f'Turbo ({r_turbo})')
        ax3.axhline(p_r1, color='orange', linestyle=':', alpha=0.6, label=f'Panic S1 ({p_r1})')
        ax3.axhline(p_r3, color='purple', linestyle='-', alpha=0.8, label=f'Panic S3 ({p_r3})')
        
        ax3.fill_between(history.index, history['RSI'], 30, where=(history['RSI'] <= 30), color='green', alpha=0.1)
        ax3.set_ylabel("RSI")
        ax3.set_ylim(0, 100)
        ax3.grid(True, linestyle='--', alpha=0.4)
        ax3.legend(loc='upper left', fontsize=9)
        
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        # X축 날짜 포맷
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)

        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        
        # 상세 데이터 표
        with st.expander("🔎 상세 시뮬레이션 데이터 보기"):
            display_df = history[['Close', 'SMA200', 'VIX', 'RSI', 'MACD', 'Trade_Label']].copy()
            st.dataframe(display_df, use_container_width=True)

class TesterView:
    """전략 분석기 입력 UI 구성 클래스"""
    @staticmethod
    def render_config():
        col_buy, col_sell = st.columns(2)
        buy_signals = []
        with col_buy:
            st.subheader("📥 Buy Conditions (OR)")
            for i in range(1, 3):
                with st.expander(f"매수 신호 {i}", expanded=(i==1)):
                    def_rsi = 35 if i == 1 else 0
                    def_rsi_cross = True if i == 2 else False
                    def_macd_inc = True if i == 2 else False
                    def_macd_below = True if i == 2 else False
                    buy_signals.append({
                        'rsi_val': st.number_input(f"RSI 매수 기준 {i}", 0, 100, def_rsi, key=f"b_rsi_{i}"),
                        'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=def_rsi_cross, key=f"b_cross_{i}"),
                        'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=False, key=f"b_inc_{i}"),
                        'macd_inc': st.checkbox(f"MACD 증가 {i}", value=def_macd_inc, key=f"b_m_inc_{i}"),
                        'macd_signal_below': st.checkbox(f"MACD 시그널 하단 {i}", value=def_macd_below, key=f"b_m_bel_{i}"),
                        'macd_golden': st.checkbox(f"MACD 골드크로스 {i}", value=False, key=f"b_m_gold_{i}"),
                        'bb_lower': st.checkbox(f"볼린저 하단 터치 {i}", value=False, key=f"b_bb_l_{i}")
                    })
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            for i in range(1, 4):
                with st.expander(f"매도 신호 {i}", expanded=(i==1)):
                    def_rsi = 70 if i == 1 else 0
                    def_rsi_dec = True if i == 1 else False
                    def_macd_above = True if i == 2 else False
                    def_macd_dec = True if i == 2 else False
                    def_rsi_dead = True if i == 2 else False
                    def_macd_dead = True if i == 3 else False
                    sell_signals.append({
                        'rsi_val': st.number_input(f"RSI 매도 기준 {i}", 0, 100, def_rsi, key=f"s_rsi_{i}"),
                        'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=def_rsi_dead, key=f"s_rsi_dead_{i}"),
                        'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=def_rsi_dec, key=f"s_rsi_dec_{i}"),
                        'macd_dec': st.checkbox(f"MACD 감소 {i}", value=def_macd_dec, key=f"s_macd_dec_{i}"),
                        'macd_signal_above': st.checkbox(f"MACD 시그널 상단 {i}", value=def_macd_above, key=f"s_macd_above_{i}"),
                        'macd_dead': st.checkbox(f"MACD 데드크로스 {i}", value=def_macd_dead, key=f"s_macd_dead_{i}"),
                        'bb_upper': st.checkbox(f"볼린저 상단 터치 {i}", value=False, key=f"s_bb_u_{i}")
                    })
        
        st.divider(); st.subheader("🔄 리밸런싱 전략")
        r1, r2, r3, r4 = st.columns(4)
        buy_reb_up = r1.number_input("Buy Reb Up (%)", value=3.0, step=0.5)/100
        buy_reb_down = r2.number_input("Buy Reb Down (%)", value=-3.0, step=0.5)/100
        sell_reb_up = r3.number_input("Sell Reb Up (%)", value=3.0, step=0.5)/100
        sell_reb_down = r4.number_input("Sell Reb Down (%)", value=-3.0, step=0.5)/100
        
        return {
            'buy_signals': buy_signals, 'sell_signals': sell_signals,
            'buy_reb_up': buy_reb_up, 'buy_reb_down': buy_reb_down, 'sell_reb_up': sell_reb_up, 'sell_reb_down': sell_reb_down
        }

# --- Main Application Orchestrator ---

class GoldenStrategyApp:
    """전체 애플리케이션의 흐름을 조율하는 클래스"""
    def __init__(self):
        # 자산 맵핑
        self.ticker_map = {
            "QQQ": "나스닥 100 (QQQ)", "QLD": "나스닥 2배 (QLD)", "TQQQ": "나스닥 3배 (TQQQ)",
            "SOXX": "반도체 1배 (SOXX)", "USD": "반도체 2배 (USD)", "SOXL": "반도체 3배 (SOXL)",
            "TLT": "장기채 (TLT)", "TMF": "장기채 3배 (TMF)"
        }
        
    def run(self):
        st.title("📈 ETF Golden Strategy")
        
        # 사이드바 메뉴
        menu = st.sidebar.radio("📌 메뉴", ["📊 백테스트", "🧪 전략 분석기", "📊 차트 분석", "📜 역사적 마켓 랩", "📰 주요 마켓 이슈"])
        if st.sidebar.button("🔄 데이터 새로고침 (Live)"):
            st.cache_data.clear()
            st.rerun()
            
        # 데이터 로딩
        with st.spinner('실시간 데이터를 가져오는 중...'):
            data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time = DataService.load_all_data()
        
        if not data_dict:
            st.error("데이터 로드 실패")
            st.stop()
        
        # 백테스트 및 전략 분석기 공통 설정
        st.sidebar.header("전략 및 기간 설정")
        base_asset = st.sidebar.selectbox("기준 자산 (시그널)", ["QQQ", "QLD", "TQQQ", "SOXX", "USD", "SOXL"], index=0, format_func=lambda x: self.ticker_map[x])
        leverage_asset = st.sidebar.selectbox("매매 대상 자산", ["QLD", "TQQQ", "USD", "SOXL", "TMF"], index=0, format_func=lambda x: self.ticker_map[x])
        cash_ratio = st.sidebar.slider("현금 비중 (%)", 0, 50, 0, step=5) / 100.0
        
        # [신규] 스마트 레버리지 모드 설정
        st.sidebar.markdown("---")
        st.sidebar.subheader("🚀 스마트 레버리지")
        use_smart = st.sidebar.checkbox("스마트 레버리지 활성화", value=True, help="VIX와 RSI를 분석해 자산을 자동으로 스위칭합니다.")
        
        smart_params = {'use_smart': use_smart}
        if use_smart:
            with st.sidebar.expander("가속/탈출 파라미터"):
                v_exit = st.slider("VIX 도망점 (Safety)", 20, 50, 31)
                r_turbo = st.slider("RSI 가속점 (Turbo)", 20, 40, 31)
                use_sequential = st.checkbox("순차적 포트폴리오 관리 (자산 유지/우선 매도)", value=True, help="가속 시 2배수를 3배수로 스왑하고, 매도 시 3배수를 우선 매도하는 자연스러운 자산 이동을 적용합니다.")
                smart_params.update({'vix_exit': v_exit, 'rsi_turbo': r_turbo, 'use_sequential': use_sequential})
        
        # [신규] 리스크 방어 모드 설정 (가변 RSI 매수 제한)
        st.sidebar.subheader("🛡️ 리스크 관리")
        use_panic = st.sidebar.checkbox("하락장 매수 제한", value=True, help="주가가 설정한 이평선 아래일 때 추가 매수 조건을 강화합니다.")
        panic_ma = st.sidebar.selectbox("방어막 기준 이평선", [200, 60], index=0, help="하락장 판정의 기준이 되는 이동평균선입니다.")
        
        smart_params.update({'use_panic': use_panic, 'panic_ma': panic_ma})
        if use_panic:
            with st.sidebar.expander(f"SMA{panic_ma} 방어 파라미터"):
                st.caption(f"주가가 {panic_ma}일 이평선 하단일 때 적용")
                p_r1 = st.slider("1단계 매수 RSI", 15, 45, 32)
                p_r2 = st.slider("2단계 매수 RSI", 15, 40, 32)
                p_r3 = st.slider("3단계 매수 RSI", 15, 40, 30)
                smart_params.update({'panic_rsi_s1': p_r1, 'panic_rsi_s2': p_r2, 'panic_rsi_s3': p_r3})
                
        couple_signal = st.sidebar.checkbox("추가 매매 시 시그널 결합", value=False, help="2·3단계 리밸런싱 시에도 가격 조건과 함께 보조지표 시그널이 충족되어야 합나다.")
        smart_params.update({'couple_signal': couple_signal})
        st.sidebar.markdown("---")

        # [신규] 매매 시점 선택 유의: 종가 매매는 당일 체결, 익일 시가는 다음날 아침 체결
        trade_at = st.sidebar.radio("매매 시점 선택", ["종가", "익일 시가"], index=0, horizontal=True)

        # 마켓 이슈 화면
        if menu == "📰 주요 마켓 이슈":
            MarketView.render(data_dict, macro_df, fetch_status, fetch_time, fg_df)
            st.stop()

        # [신규] 차트 분석 화면 처리
        if menu == "📊 차트 분석":
            ChartView.render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, st.session_state.get('current_params'), smart_params=smart_params)
            st.stop()

        # [신규] 역사적 마켓 랩 화면 처리
        if menu == "📜 역사적 마켓 랩":
            HistoryLabView.render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, st.session_state.get('current_params'), smart_params=smart_params)
            st.stop()
        
        if 'QQQ' in data_dict:
            min_d = data_dict['QQQ'].index.min().date()
            max_d = data_dict['QQQ'].index.max().date()
        else:
            st.error("⚠️ 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해 주세요.")
            st.stop()

        # [신규] 연 단위 기간 선택 및 동기화 로직
        current_year = datetime.date.today().year
        year_options = ["직접 설정"] + [str(y) for y in range(current_year, 2009, -1)]
        
        # 초기 세션 상태 설정
        if 'prev_year_choice' not in st.session_state:
            st.session_state.prev_year_choice = "직접 설정"
        if 'start_d_input' not in st.session_state:
            st.session_state['start_d_input'] = datetime.date(2026, 1, 1) if min_d <= datetime.date(2026, 1, 1) else min_d
        if 'end_d_input' not in st.session_state:
            st.session_state['end_d_input'] = max_d
            
        selected_year = st.sidebar.selectbox("📅 기간 퀵 선택 (연 단위)", year_options, index=0)

        # 연도 선택이 변경된 경우 위젯의 키 값을 직접 업데이트
        if selected_year != st.session_state.prev_year_choice:
            if selected_year != "직접 설정":
                y = int(selected_year)
                new_s = datetime.date(y, 1, 1)
                new_e = datetime.date(y, 12, 31)
                # 데이터 유효 범위 보정
                st.session_state['start_d_input'] = max(new_s, min_d)
                st.session_state['end_d_input'] = min(new_e, max_d)
            st.session_state.prev_year_choice = selected_year

        # 위젯 호출 (key를 지정하면 session_state의 해당 키 값이 value로 사용됨)
        start_d = st.sidebar.date_input("시작일", key="start_d_input")
        end_d = st.sidebar.date_input("종료일", key="end_d_input")
        
        params = st.session_state.get('current_params')
        if menu == "🧪 전략 분석기":
            st.header("🧪 전략 분석기")
            params = TesterView.render_config()
            st.session_state.current_params = params
            
        # 핵심 실행부
        with st.spinner('백테스팅 계산 중...'):
            # [수정] 기준 자산 섹터에 따른 동적 벤치마크 그룹 설정
            semi_group = ['SOXX', 'USD', 'SOXL']
            is_semi_mode = base_asset in semi_group or leverage_asset in semi_group
            bench_tickers = semi_group if is_semi_mode else ['QQQ', 'QLD', 'TQQQ']
            
            bh_histories = {t: StrategyEngine.run_benchmark(data_dict, t, start_d, end_d) for t in bench_tickers}
            golden_history = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_d, end_d, params, trade_at, smart_params=smart_params)
            
        # 결과 렌더링
        BacktestView.render_results(golden_history, bh_histories, base_asset, leverage_asset)
        st.info("💡 팁: '전략 분석기'에서 설정을 변경하면 즉시 백테스트 결과가 업데이트됩니다.")

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
