import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import os
from backtest import calculate_metrics

# 페이지 설정
st.set_page_config(
    page_title="ETF Golden Strategy",
    page_icon="📈",
    layout="wide",
)

# --- 스타일 설정 ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 로직 함수 (기본 final_main.py 기반) ---

def read_data(base_dir, filename):
    # 중앙 데이터 경로 설정 (C:\TestCode\data)
    data_dir = os.path.join(base_dir, "data")
    path = os.path.join(data_dir, filename)
    
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    return None

# --- 라이브 데이터 페칭 로직 (Cloud 환경용) ---
import yfinance as yf
import requests
import datetime

def calculate_indicators(df):
    """지표 계산 함수 (data_loader.py 로직 복사)"""
    if 'Close' not in df.columns: return df
    close_prices = pd.to_numeric(df['Close'], errors='coerce')
    df['RSI'] = ta.rsi(close_prices, length=14)
    macd = ta.macd(close_prices)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    return df

def fetch_live_data():
    """실시간 데이터 수집 함수"""
    print("라이브 데이터 수집 시작...")
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = '2019-01-01'
    
    # 1. ETF 데이터
    data_dict = {}
    tickers = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TMF']
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        if not df.empty:
            df = calculate_indicators(df)
            data_dict[ticker] = df
            
    # 2. VIX 데이터
    vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_df = vix_df.loc[:, ~vix_df.columns.duplicated()]
    
    # 3. Fear & Greed 데이터
    fg_df = pd.DataFrame()
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        series = data.get('fear_and_greed_historical', {}).get('data', [])
        fg_df = pd.DataFrame(series)
        fg_df['x'] = pd.to_datetime(fg_df['x'], unit='ms')
        fg_df.rename(columns={'x': 'Date', 'y': 'FearGreed'}, inplace=True)
        fg_df.set_index('Date', inplace=True)
    except Exception as e:
        print(f"F&G Fetch Error: {e}")
        fg_df = pd.DataFrame(columns=['FearGreed'])
        fg_df.index = pd.to_datetime(fg_df.index)

    # 4. 거시 경제 데이터 (실시간 연동 - 안정성 강화)
    macro_data_map = {}
    fetch_status = {
        'ETF': 'Success',
        'VIX': 'Pending',
        'FearGreed': 'Pending',
        'US10Y': 'Pending',
        'US03M': 'Pending',
        'PCCR': 'N/A' # Yahoo Finance 지원 안 함
    }
    fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 10년물, 3개월물 금리, 풋/콜 비율 수집
    m_tickers = {
        '^TNX': 'US10Y',   # 10년물 금리
        '^IRX': 'US03M',    # 3개월물 금리
        '^PCC': 'PCCR'     # CBOE Put/Call Ratio
    }
    
    print("\n[매크로 지표 수집 시작]")
    for ticker, col_name in m_tickers.items():
        try:
            m_df = yf.download(ticker, period='5d', progress=False)
            if not m_df.empty:
                if isinstance(m_df.columns, pd.MultiIndex):
                    m_df.columns = m_df.columns.get_level_values(0)
                
                target_col = next((c for c in ['Close', 'Adj Close', 'Price', 'Last'] if c in m_df.columns), None)
                if target_col:
                    valid_data = m_df[target_col].dropna()
                    if not valid_data.empty:
                        val = valid_data.iloc[-1]
                        macro_data_map[col_name] = val
                        fetch_status[col_name] = 'Success'
                        print(f"  - {ticker} ({col_name}) 완료: {val:.2f}")
                    else:
                        fetch_status[col_name] = 'No Data'
                else:
                    fetch_status[col_name] = 'Col Missing'
            else:
                fetch_status[col_name] = 'Empty'
        except Exception as e:
            fetch_status[col_name] = f'Error: {str(e)[:20]}'
            print(f"  - {ticker} ({col_name}) 에러: {e}")
    
    # VIX는 위에서 이미 vix_df로 수집했으므로 그 값을 활용 (중복 방지 및 안정성)
    if not vix_df.empty:
        if 'Close' in vix_df.columns:
            latest_vix_val = vix_df['Close'].dropna().iloc[-1]
            macro_data_map['VIX'] = latest_vix_val
            fetch_status['VIX'] = 'Success'
            print(f"  - ^VIX (VIX) 연동 완료: {latest_vix_val:.2f}")
        else:
            fetch_status['VIX'] = 'Col Missing'
    else:
        fetch_status['VIX'] = 'Empty'
        print("  - 경고: vix_df가 비어있어 macro_df에 VIX를 넣지 못했습니다.")

    # Fear & Greed 상태 업데이트
    if not fg_df.empty:
        fetch_status['FearGreed'] = 'Success'
    else:
        fetch_status['FearGreed'] = 'Fail'

    # 데이터셋 구성
    macro_df = pd.DataFrame([macro_data_map])
    print(f"[매크로 지표 수집 종료] 수집된 지표 수: {len(macro_data_map)}")
    print(f"수집된 컬럼: {macro_df.columns.tolist()}\n")
            
    return data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time

@st.cache_data(ttl=3600) # 1시간 캐시
def load_all_data():
    """실시간 데이터 수집만 수행하도록 변경"""
    return fetch_live_data()

def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0.0, start_date=None, end_date=None):
    qqq = data_dict[base_asset]
    combined = qqq[['Close', 'RSI']].copy()
    
    fg_df.index = pd.to_datetime(fg_df.index)
    vix_df.index = pd.to_datetime(vix_df.index)
    
    fg_clean = fg_df[~fg_df.index.duplicated(keep='first')]
    vix_clean = vix_df[~vix_df.index.duplicated(keep='first')]
    
    combined = combined.join(fg_clean['FearGreed'].rename('FG'), how='left')
    combined = combined.join(vix_clean['Close'].rename('VIX'), how='left')
    
    # RSI SMA 및 MACD 컬럼 준비 (동일 로직 적용)
    macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
    signal_col = [c for c in combined.columns if 'MACDs_' in c]
    
    if macd_col and signal_col:
        combined['MACD'] = combined[macd_col[0]]
        combined['Signal_Line'] = combined[signal_col[0]]
    else:
        macd = ta.macd(combined['Close'])
        combined = pd.concat([combined, macd], axis=1)
        combined['MACD'] = combined['MACD_12_26_9']
        combined['Signal_Line'] = combined['MACDs_12_26_9']
    combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
    
    combined['Prev_RSI'] = combined['RSI'].shift(1)
    combined['Prev_RSI_SMA'] = combined['RSI_SMA'].shift(1)
    combined['Prev_MACD'] = combined['MACD'].shift(1)
    combined['Prev_Signal_Line'] = combined['Signal_Line'].shift(1)
    
    combined['FG'] = combined['FG'].ffill().fillna(50)
    combined['VIX'] = combined['VIX'].ffill().fillna(15)
    combined = combined.fillna(0)
    
    dates = combined.index
    if start_date:
        dates = dates[dates.date >= start_date]
    if end_date:
        dates = dates[dates.date <= end_date]
        
    if len(dates) == 0:
        return pd.DataFrame()

    portfolio_value = 10000.0
    cash = portfolio_value * cash_ratio
    etf_val = portfolio_value * (1 - cash_ratio)
    
    current_planned_asset = base_asset
    rebalance_stage = 0
    last_entry_price = 0
    
    holdings = {t: 0.0 for t in data_dict.keys()}
    holdings[base_asset] = etf_val / qqq.loc[dates[0], 'Close']
    
    history = []
    
    for date in dates:
        prices = {t: data_dict[t].loc[date, 'Close'] for t in data_dict.keys()}
        row = combined.loc[date]
        rsi, fg, vix = row['RSI'], row['FG'], row['VIX']
        macd_val, signal_line, rsi_sma = row['MACD'], row['Signal_Line'], row['RSI_SMA']
        prev_rsi, prev_rsi_sma = row['Prev_RSI'], row['Prev_RSI_SMA']
        prev_macd, prev_signal_line = row['Prev_MACD'], row['Prev_Signal_Line']
        
        is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
        is_macd_improving = (macd_val > prev_macd)
        is_macd_below_signal = (macd_val < signal_line)
        is_buy_signal = (is_rsi_golden_cross and is_macd_improving and is_macd_below_signal) or (rsi < 35)
        
        sell_cond_1 = (rsi >= 70) and (rsi < prev_rsi)
        is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
        sell_cond_2 = (macd_val > signal_line) and (macd_val < prev_macd) and is_rsi_dead_cross
        sell_cond_3 = (prev_macd > prev_signal_line) and (macd_val < signal_line)
        is_sell_signal = sell_cond_1 or sell_cond_2 or sell_cond_3
        
        # 신호 조건 로깅 (UI 표시용)
        signal_conditions = []
        if is_buy_signal:
            if rsi < 35: signal_conditions.append(f"RSI < 35 ({rsi:.1f})")
            if is_rsi_golden_cross and is_macd_improving and is_macd_below_signal: signal_conditions.append("RSI Golden Cross + MACD Improving")
        if is_sell_signal:
            if sell_cond_1: signal_conditions.append(f"RSI >= 70 & Declining ({rsi:.1f})")
            if sell_cond_2: signal_conditions.append("MACD > Signal & Declining + RSI Dead Cross")
            if sell_cond_3: signal_conditions.append("MACD Dead Cross")
        
        current_total_val = cash + sum(holdings[t] * prices[t] for t in holdings)
        rebalance_needed = False
        
        etf_funds = current_total_val * (1 - cash_ratio)
        current_lev_val = holdings[leverage_asset] * prices[leverage_asset]
        current_lev_pct = current_lev_val / etf_funds if etf_funds > 0 else 0
        
        if is_buy_signal and current_planned_asset != leverage_asset:
            current_planned_asset = leverage_asset
            if current_lev_pct < 0.25: target_lev_pct, rebalance_stage = 0.30, 1
            elif current_lev_pct < 0.65: target_lev_pct, rebalance_stage = 0.70, 2
            else: target_lev_pct, rebalance_stage = 1.0, 3
            last_entry_price, rebalance_needed = prices[base_asset], True
        elif is_sell_signal and current_planned_asset != base_asset:
            current_planned_asset = base_asset
            if current_lev_pct > 0.75: target_lev_pct, rebalance_stage = 0.70, 1
            elif current_lev_pct > 0.35: target_lev_pct, rebalance_stage = 0.30, 2
            else: target_lev_pct, rebalance_stage = 0.0, 3
            last_entry_price, rebalance_needed = prices[base_asset], True
        elif rebalance_stage in [1, 2]:
            price_change_ratio = prices[base_asset] / last_entry_price
            if price_change_ratio <= 0.97 or price_change_ratio >= 1.03:
                is_signal_active = (current_planned_asset == leverage_asset and is_buy_signal) or \
                                   (current_planned_asset == base_asset and is_sell_signal)
                if is_signal_active:
                    rebalance_stage += 1
                    if current_planned_asset == leverage_asset:
                        target_lev_pct = 0.70 if rebalance_stage == 2 else 1.0
                    else:
                        target_lev_pct = 0.30 if rebalance_stage == 2 else 0.0
                    last_entry_price, rebalance_needed = prices[base_asset], True
        
        if rebalance_needed:
            new_cash = current_total_val * cash_ratio
            etf_funds = current_total_val * (1 - cash_ratio)
            lev_val = etf_funds * target_lev_pct
            qqq_val = etf_funds * (1 - target_lev_pct)
            for t in holdings: holdings[t] = 0
            holdings[leverage_asset] = lev_val / prices[leverage_asset]
            holdings[base_asset] = qqq_val / prices[base_asset]
            cash = new_cash
        
        asset_label = f"{current_planned_asset}(S{rebalance_stage})" if rebalance_stage < 3 else current_planned_asset
        
        # 현재 비중 계산 (rebalance_needed가 발생하지 않은 날도 유지되는 비중 기록)
        # 루프 시작 부분에서 target_lev_pct를 관리하거나 현재 보유 비중을 직접 계산
        if current_planned_asset == leverage_asset:
            # 매수 상태일 때의 단계별 비중
            current_target_lev = 0.30 if rebalance_stage == 1 else (0.70 if rebalance_stage == 2 else 1.0)
        else:
            # 매도 상태(QQQ 복귀 중)일 때의 단계별 비중
            current_target_lev = 0.70 if rebalance_stage == 1 else (0.30 if rebalance_stage == 2 else 0.0)

        # 시그널 상태 기록 (UI 표시용)
        # 시그널 상태 기록 (UI 표시용) - 비중 변화가 있을 때만 매수/매도 시그널 표시
        signal_state = "중립"
        current_condition = "N/A"
        
        if rebalance_needed:
            if current_planned_asset == leverage_asset:
                signal_state = "매수"
            else:
                signal_state = "매도"
            
            # 신호 발생 시 조건 저장
            if 'signal_conditions' in locals() and signal_conditions:
                current_condition = " | ".join(signal_conditions)

        history.append({
            'Date': date, 
            'Value': current_total_val, 
            'Signal': signal_state,
            'Condition': current_condition,
            'Asset': asset_label,
            'Lev_Weight': current_target_lev,
            f'{base_asset}_Weight': 1.0 - current_target_lev,
            'RSI': rsi,
            'FG': fg,
            'VIX': vix,
            'MACD': macd_val,
            'Signal_Line': signal_line
        })
        
    return pd.DataFrame(history).set_index('Date')

def run_benchmark(data_dict, ticker='QQQ', start_date=None, end_date=None):
    df = data_dict[ticker]
    dates = df.index
    if start_date:
        dates = dates[dates.date >= start_date]
    if end_date:
        dates = dates[dates.date <= end_date]
        
    if len(dates) == 0:
        return pd.DataFrame()
        
    shares = 10000.0 / df.loc[dates[0], 'Close']
    history = [{'Date': date, 'Value': shares * df.loc[date, 'Close']} for date in dates]
    return pd.DataFrame(history).set_index('Date')

# --- UI 구현 ---

st.title("📈 ETF Golden Strategy")
st.write("나스닥 자산 재배분 전략 및 시장 모니터링")

# --- 사이드바 메뉴 ---
st.sidebar.title("📌 메뉴")
menu = st.sidebar.radio(
    "화면 선택",
    ["📊 백테스트", "📰 주요 마켓 이슈"],
    index=0
)

def market_issues_view(data_dict, macro_df, fetch_status, fetch_time, fg_df):
    st.header("📰 주요 마켓 이슈 및 밸류어이션")
    st.caption(f"최종 데이터 업데이트: {fetch_time}")
    
    # 1. 시장 밸류에이션 및 심리 (Valuation & Sentiment)
    st.subheader("📊 시장 밸류에이션 및 심리")
    
    # 최근 데이터 추출
    latest_qqq_price = data_dict['QQQ']['Close'].iloc[-1]
    qqq_eps_est = 15.64 
    current_per = latest_qqq_price / qqq_eps_est
    
    # Fear & Greed
    latest_fg = fg_df['FearGreed'].iloc[-1] if not fg_df.empty else 50
    def get_fg_label(val):
        if val <= 25: return "Extreme Fear (극도의 공포)"
        if val <= 45: return "Fear (공포)"
        if val <= 55: return "Neutral (중립)"
        if val <= 75: return "Greed (탐욕)"
        return "Extreme Greed (극도의 탐욕)"
    
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        st.metric(label="나스닥 100 PER", value=f"{current_per:.1f}x", delta="평균 24~26x 대비 높음", delta_color="inverse")
    with v_col2:
        base_buffett = 223.6
        current_buffett = base_buffett * (latest_qqq_price / 520.0)
        st.metric(label="버핏 지수 (추정)", value=f"{current_buffett:.1f}%", delta="200% 이상 과열", delta_color="inverse")
    with v_col3:
        st.metric(label="Fear & Greed Index", value=f"{latest_fg:.0f}", delta=get_fg_label(latest_fg), delta_color="off")

    st.divider()

    # 2. 핵심 거시경제 지표 (Market Radar)
    st.subheader("💡 핵심 체크 지표 (Market Radar)")
    
    latest_10y = 0.0
    latest_03m = 0.0
    latest_vix = 0.0
    latest_pccr = 0.46 # Default fallback
    
    if macro_df is not None and not macro_df.empty:
        # 10년물 금리 보정
        latest_10y = macro_df.get('US10Y', pd.Series([0.0])).iloc[-1]
        if latest_10y > 15: latest_10y /= 10.0
            
        # 3개월물 금격
        latest_03m = macro_df.get('US03M', pd.Series([0.0])).iloc[-1]
        
        # VIX
        latest_vix = macro_df.get('VIX', pd.Series([0.0])).iloc[-1]
        
        # PCCR
        if 'PCCR' in macro_df.columns:
            latest_pccr = macro_df['PCCR'].iloc[-1]

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="미 국채 10년물 금리", value=f"{latest_10y:.2f}%", help="상승 시 성장주(나스닥)에 하방 압력")
    with m_col2:
        st.metric(label="VIX 공포 지수", value=f"{latest_vix:.2f}", help="20 이상 시 변동성 확대, 30 이상 시 패닉")
    with m_col3:
        st.metric(label="풋/콜 비율 (PCCR)", value=f"{latest_pccr:.2f}", help="0.7 이하: 과열, 1.0 이상: 공포")

    # 장단기 금리차 별도 강조
    yield_spread = latest_10y - latest_03m
    spread_color = "normal" if yield_spread > 0 else "inverse"
    st.info(f"📐 **장단기 금리차 (10Y-3M): {yield_spread:+.2f}%** {' (정상)' if yield_spread > 0 else ' (역전 - 침체 전조)'}")
    
    st.divider()

    # 3. 주요 일정 캘린더
    st.subheader("📅 주요 경제 일정")
    today = datetime.date.today()

    # 1. FOMC 일정 (데이터 보강: 날짜, 실제값, 컨센서스)
    fomc_events = [
        {"date": datetime.date(2025, 12, 10), "actual": "5.50%", "consensus": "5.50%"},
        {"date": datetime.date(2026, 1, 29), "actual": None, "consensus": "5.50% (동결)"},
        {"date": datetime.date(2026, 3, 19), "actual": None, "consensus": "5.25% (인하)"},
        {"date": datetime.date(2026, 4, 30), "actual": None, "consensus": "-"}
    ]
    
    # 2. CPI 일정 (데이터 보강: 날짜, 실제값, 컨센서스)
    cpi_events = [
        {"date": datetime.date(2025, 12, 12), "actual": "3.1%", "consensus": "3.1%"},
        {"date": datetime.date(2026, 1, 13), "actual": "3.1%", "consensus": "3.0%"},
        {"date": datetime.date(2026, 2, 12), "actual": None, "consensus": "2.9%"},
        {"date": datetime.date(2026, 3, 12), "actual": None, "consensus": "-"}
    ]

    def get_past_and_next(events, today):
        past = [e for e in events if e['date'] < today]
        future = [e for e in events if e['date'] >= today]
        latest_past = past[-1] if past else None
        next_upcoming = future[0] if future else None
        return latest_past, next_upcoming

    
    col_fomc, col_cpi = st.columns(2)
    
    with col_fomc:
        st.markdown("**🏦 FOMC 금리 결정**")
        p, n = get_past_and_next(fomc_events, today)
        if p:
            st.info(f"◀ **직전 ({p['date']})**\n\n결과: {p['actual']} / 예상: {p['consensus']}")
        if n:
            st.success(f"▶ **예정 ({n['date']})**\n\n예상: {n['consensus']}")

    with col_cpi:
        st.markdown("**📈 CPI 소비자물가**")
        p, n = get_past_and_next(cpi_events, today)
        if p:
            st.info(f"◀ **직전 ({p['date']})**\n\n결과: {p['actual']} (전년대비) / 예상: {p['consensus']}")
        if n:
            st.success(f"▶ **예정 ({n['date']})**\n\n예상: {n['consensus']}")

    # 3. 빅테크 실적발표 (Earnings)
    with st.expander("💻 빅테크 주요 실적발표 주간", expanded=True):
        st.info("빅테크(애플, MS, 테슬라 등)의 실적은 나스닥의 변동성을 좌우합니다.")
        earnings_weeks = [
            ("1월 5주차 (1/26~1/30)", "MSFT, TSLA, AAPL 등 어닝 슈퍼위크"),
            ("4월 4주차", "1분기 실적 발표 시즌"),
            ("7월 4주차", "2분기 실적 발표 시즌"),
            ("10월 4주차", "3분기 실적 발표 시즌")
        ]
        for period, desc in earnings_weeks:
            st.markdown(f"- **{period}** : {desc}")

    # 4. 데이터 수집 상태 (Debug Info)
    with st.expander("🛠️ 데이터 수집 상태 정보", expanded=False):
        status_df = pd.DataFrame([fetch_status]).T
        status_df.columns = ["Status"]
        st.table(status_df)
        st.caption("N/A는 수집이 불가능하거나 라이브 지원이 안 되는 지표입니다.")

    # 5. 기타 주의사항
    st.warning("⚠️ 위 일정은 미국 현지 사정에 따라 변경될 수 있으며, 한국 시간 기준은 보통 다음 날 새벽입니다.")

# 세션 상태 및 데이터 새로고침 관리
if 'use_live' not in st.session_state:
    st.session_state['use_live'] = True

if st.sidebar.button("🔄 데이터 새로고침 (Live)"):
    st.cache_data.clear()
    st.rerun()

# 라이브 모드 알림
st.sidebar.success("✅ 실시간 마켓 데이터 모드")

with st.spinner('실시간 마켓 데이터를 가져오는 중...'):
    data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time = load_all_data()

if not data_dict:
    st.error("마켓 데이터를 가져올 수 없습니다. 인터넷 연결 상태를 확인해주세요.")
    st.stop()

# --- 주요 마켓 이슈 화면 분기 ---
if menu == "📰 주요 마켓 이슈":
    market_issues_view(data_dict, macro_df, fetch_status, fetch_time, fg_df)
    st.stop()

# 사이드바 설정 (백테스트용)
st.sidebar.header("전략 설정")

# 자산 선택
ticker_map = {
    "QQQ": "QQQ (1x)",
    "QLD": "QLD (2x) 🌟",
    "TQQQ": "TQQQ (3x) 🌟",
    "TLT": "TLT (1x)",
    "TMF": "TMF (3x)"
}
base_asset = st.sidebar.selectbox(
    "대표 자산 (시그널 기준)", 
    ["QQQ", "QLD", "TQQQ", "TLT"], 
    index=0,
    format_func=lambda x: ticker_map[x]
)
leverage_asset = st.sidebar.selectbox(
    "레버리지 자산 (매매 대상)", 
    ["QLD", "TQQQ", "TMF"], 
    index=0,
    format_func=lambda x: ticker_map[x]
)

cash_ratio = st.sidebar.slider("현금 비중 (%)", 0, 50, 0, step=5) / 100.0

# 기간 설정 추가
st.sidebar.header("기간 설정")
min_date = data_dict['QQQ'].index.min().date()
max_date = data_dict['QQQ'].index.max().date()

start_date = st.sidebar.date_input("시작일", min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("종료일", max_date, min_value=min_date, max_value=max_date)

# 날짜 필터링 함수
def filter_by_date(df, start, end):
    if df is None or df.empty:
        return df
    # 인덱스가 날짜 형식이 아닐 경우 변환 시도
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            return df
            
    return df[(df.index.date >= start) & (df.index.date <= end)]

# 데이터 필터링 적용
filtered_data_dict = {t: filter_by_date(df, start_date, end_date) for t, df in data_dict.items()}
filtered_fg_df = filter_by_date(fg_df, start_date, end_date)
filtered_vix_df = filter_by_date(vix_df, start_date, end_date)

# 데이터 검증 (ETF 데이터는 필수, 보조지표는 선택)
if any(df.empty for df in filtered_data_dict.values()):
    st.error("주요 ETF 데이터(QQQ, QLD, TQQQ)를 가져오지 못했습니다. 잠시 후 다시 시도해주세요.")
    st.stop()

if filtered_fg_df.empty:
    st.warning("⚠️ Fear & Greed 지수를 가져오지 못해 기본값(50)을 사용합니다.")

if filtered_vix_df.empty:
    st.warning("⚠️ VIX 지수를 가져오지 못해 기본값(15)을 사용합니다.")

# 백테스팅 실행
with st.spinner('백테스팅 계산 중...'):
    # 공통 벤치마크 계산 (QQQ, QLD, TQQQ)
    bh_qqq_history = run_benchmark(data_dict, 'QQQ', start_date=start_date, end_date=end_date)
    bh_qld_history = run_benchmark(data_dict, 'QLD', start_date=start_date, end_date=end_date)
    bh_tqqq_history = run_benchmark(data_dict, 'TQQQ', start_date=start_date, end_date=end_date)
    
    # 전략 실행 (필터링되지 않은 원본 데이터 전달, 날짜는 내부에서 처리)
    golden_history = run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_date=start_date, end_date=end_date)

# --- 현재 상태 및 시그널 요약 추가 ---
latest_row = golden_history.iloc[-1]
latest_date = latest_row.name
current_signal = latest_row['Signal']
lev_w = latest_row['Lev_Weight']
qqq_w = latest_row[f'{base_asset}_Weight']

st.subheader(f"📍 상태 요약 ({latest_date.date()})")

# 앱 상단에 크게 신호 표시
def get_latest_signal_info(signal):
    if signal == "매수":
        return "🔥 매수 시그널 (진입/유지)", "green"
    elif signal == "매도":
        return "⚠️ 매도 시그널 (수익실현/대기)", "red"
    else:
        return "⚖️ 중립 상태 (신호 없음)", "gray"

sig_text, sig_color = get_latest_signal_info(current_signal)
st.markdown(f"""
    <div style="background-color:{sig_color}; padding:20px; border-radius:10px; text-align:center; color:white;">
        <h2 style="margin:0;">{sig_text}</h2>
        <div style="margin-top:10px; font-size:1.2em; font-weight:bold;">
            비중: {leverage_asset} {lev_w:.0%} | {base_asset} {qqq_w:.0%}
        </div>
    </div>
    """, unsafe_allow_html=True)

# 매매 조건 표시 (지표 수치 대신 이유 표시)
if latest_row.get('Condition', 'N/A') != 'N/A' and current_signal in ["매수", "매도"]:
    st.info(f"📋 **체결 조건:** {latest_row['Condition']}")

st.write("")

# 지표 계산
qqq_metrics = calculate_metrics(bh_qqq_history.rename(columns={'Value': 'PortfolioValue'}))
qld_metrics = calculate_metrics(bh_qld_history.rename(columns={'Value': 'PortfolioValue'}))
tqqq_metrics = calculate_metrics(bh_tqqq_history.rename(columns={'Value': 'PortfolioValue'}))
golden_metrics = calculate_metrics(golden_history.rename(columns={'Value': 'PortfolioValue'}))

# 메인 지표 표시 (3열)
col1, col2, col3 = st.columns(3)
with col1:
    # 기본 벤치마크(QQQ) 대비 비교
    st.metric("누적 수익률", f"{golden_metrics['Cumulative Return']:.1%}", 
              delta=f"{(golden_metrics['Cumulative Return'] - qqq_metrics['Cumulative Return']):.1%}")
with col2:
    st.metric("CAGR (연간 성장률)", f"{golden_metrics['CAGR']:.1%}")
with col3:
    st.metric("MDD (최대 낙폭)", f"{golden_metrics['MDD']:.1%}")

# 차트 시각화
st.subheader("성과 비교 차트")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bh_qqq_history['Value'] / 10000, label='Benchmark (QQQ)', color='gray', alpha=0.5)
ax.plot(bh_qld_history['Value'] / 10000, label='Benchmark (QLD)', color='orange', alpha=0.5)
ax.plot(bh_tqqq_history['Value'] / 10000, label='Benchmark (TQQQ)', color='red', alpha=0.5)
ax.plot(golden_history['Value'] / 10000, label=f'Golden Strat ({leverage_asset})', color='blue', linewidth=2)
ax.set_ylabel('Normalized Value')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 상세 성과 데이터 (전략별 요약)
st.subheader("전략 성과 요약")
summary_data = [
    {"전략": "Benchmark (QQQ)", "최종가치": f"{qqq_metrics['Final Value']:,.0f}", "수익률(ROI)": f"{qqq_metrics['Cumulative Return']:.2%}", "CAGR": f"{qqq_metrics['CAGR']:.2%}", "MDD": f"{qqq_metrics['MDD']:.2%}"},
    {"전략": "Benchmark (QLD)", "최종가치": f"{qld_metrics['Final Value']:,.0f}", "수익률(ROI)": f"{qld_metrics['Cumulative Return']:.2%}", "CAGR": f"{qld_metrics['CAGR']:.2%}", "MDD": f"{qld_metrics['MDD']:.2%}"},
    {"전략": "Benchmark (TQQQ)", "최종가치": f"{tqqq_metrics['Final Value']:,.0f}", "수익률(ROI)": f"{tqqq_metrics['Cumulative Return']:.2%}", "CAGR": f"{tqqq_metrics['CAGR']:.2%}", "MDD": f"{tqqq_metrics['MDD']:.2%}"},
    {"전략": f"Golden Strat ({leverage_asset})", "최종가치": f"{golden_metrics['Final Value']:,.0f}", "수익률(ROI)": f"{golden_metrics['Cumulative Return']:.2%}", "CAGR": f"{golden_metrics['CAGR']:.2%}", "MDD": f"{golden_metrics['MDD']:.2%}"}
]
st.table(pd.DataFrame(summary_data).set_index("전략"))

st.info("💡 팁: 핸드폰 브라우저 메뉴에서 '홈 화면에 추가'를 누르면 앱처럼 사용할 수 있습니다.")
