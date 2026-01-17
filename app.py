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
    # 확인할 경로 목록 (우선순위 순)
    paths = [
        os.path.join(base_dir, filename),           # 1. 로컬 개발 환경 (상위 폴더)
        os.path.join(base_dir, 'back', filename),   # 2. 로컬 백업 폴더
        filename,                                   # 3. 같은 폴더 (Streamlit Cloud 배포)
        os.path.join(os.getcwd(), filename)         # 4. 현재 작업 폴더
    ]
    
    for path in paths:
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
    tickers = ['QQQ', 'QLD', 'TQQQ']
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
        headers = {'User-Agent': 'Mozilla/5.0'}
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
        
    return data_dict, fg_df, vix_df

@st.cache_data(ttl=3600) # 1시간 캐시
def load_all_data(use_live=False):
    if use_live:
        return fetch_live_data()
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    data_dict = {}
    for ticker in ['QQQ', 'QLD', 'TQQQ']:
        df = read_data(base_dir, f'{ticker}_data.csv')
        if df is not None:
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df.sort_index()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            data_dict[ticker] = df.dropna(subset=['Close'])
    
    fg_df = read_data(base_dir, 'fear_greed_data.csv')
    vix_df = read_data(base_dir, 'vix_data.csv')
    
    if fg_df is not None:
        fg_df.index = pd.to_datetime(fg_df.index, errors='coerce')
    if vix_df is not None:
        vix_df.index = pd.to_datetime(vix_df.index, errors='coerce')
        
    return data_dict, fg_df, vix_df

def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0.0):
    qqq = data_dict[base_asset]
    combined = qqq[['Close', 'RSI']].copy()
    
    fg_df.index = pd.to_datetime(fg_df.index)
    vix_df.index = pd.to_datetime(vix_df.index)
    
    combined = combined.join(fg_df['FearGreed'].rename('FG'), how='left')
    combined = combined.join(vix_df['Close'].rename('VIX'), how='left')
    
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
            last_entry_price, rebalance_needed = prices[current_planned_asset], True
        elif is_sell_signal and current_planned_asset != base_asset:
            current_planned_asset = base_asset
            if current_lev_pct > 0.75: target_lev_pct, rebalance_stage = 0.70, 1
            elif current_lev_pct > 0.35: target_lev_pct, rebalance_stage = 0.30, 2
            else: target_lev_pct, rebalance_stage = 0.0, 3
            last_entry_price, rebalance_needed = prices[current_planned_asset], True
        elif rebalance_stage in [1, 2]:
            price_change_ratio = prices[current_planned_asset] / last_entry_price
            if price_change_ratio <= 0.97 or price_change_ratio >= 1.03:
                is_signal_active = (current_planned_asset == leverage_asset and is_buy_signal) or \
                                   (current_planned_asset == base_asset and is_sell_signal)
                if is_signal_active:
                    rebalance_stage += 1
                    if current_planned_asset == leverage_asset:
                        target_lev_pct = 0.70 if rebalance_stage == 2 else 1.0
                    else:
                        target_lev_pct = 0.30 if rebalance_stage == 2 else 0.0
                    last_entry_price, rebalance_needed = prices[current_planned_asset], True
        
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
        if rebalance_needed:
            if current_planned_asset == leverage_asset:
                signal_state = "매수"
            else:
                signal_state = "매도"
        
        history.append({
            'Date': date, 
            'Value': current_total_val, 
            'Signal': signal_state,
            'Asset': asset_label,
            'Lev_Weight': current_target_lev,
            f'{base_asset}_Weight': 1.0 - current_target_lev
        })
        
    return pd.DataFrame(history).set_index('Date')

def run_benchmark(data_dict, ticker='QQQ'):
    df = data_dict[ticker]
    dates = df.index
    shares = 10000.0 / df.loc[dates[0], 'Close']
    history = [{'Date': date, 'Value': shares * df.loc[date, 'Close']} for date in dates]
    return pd.DataFrame(history).set_index('Date')

# --- UI 구현 ---

st.title("📈 ETF Golden Strategy Mobile")
st.write("핸드폰으로 확인하는 자산 재배분 전략 대시보드")

# 데이터 로딩
# 사이드바에 새로고침 버튼 추가
if st.sidebar.button("🔄 데이터 최신화 (Live)"):
    st.cache_data.clear() # 캐시 삭제하여 강제 갱신
    use_live = True
else:
    use_live = False

with st.spinner('데이터를 불러오는 중...'):
    # 라이브 모드일 경우 파일이 없어도 되므로 예외 처리 완화 가능하지만, 
    # 일단 기존 로직 유지하되 우선순위 둠
    data_dict, fg_df, vix_df = load_all_data(use_live=use_live)

if not data_dict:
    st.error("데이터 파일을 찾을 수 없습니다. c:\\TestCode 경로에 데이터가 있는지 확인해주세요.")
    st.stop()

# 사이드바 설정
st.sidebar.header("전략 설정")

# 자산 선택
ticker_map = {
    "QQQ": "QQQ (1x)",
    "QLD": "QLD (2x)",
    "TQQQ": "TQQQ (3x)"
}
base_asset = st.sidebar.selectbox(
    "대표 자산 (시그널 기준)", 
    ["QQQ", "QLD", "TQQQ"], 
    index=0,
    format_func=lambda x: ticker_map[x]
)
leverage_asset = st.sidebar.selectbox(
    "레버리지 자산 (매매 대상)", 
    ["QLD", "TQQQ"], 
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
    return df[(df.index.date >= start) & (df.index.date <= end)]

# 데이터 필터링 적용
filtered_data_dict = {t: filter_by_date(df, start_date, end_date) for t, df in data_dict.items()}
filtered_fg_df = filter_by_date(fg_df, start_date, end_date)
filtered_vix_df = filter_by_date(vix_df, start_date, end_date)

# 데이터가 비어있는지 확인
if any(df.empty for df in filtered_data_dict.values()) or filtered_fg_df.empty or filtered_vix_df.empty:
    st.warning("선택한 기간에 데이터가 충분하지 않습니다. 다른 기간을 선택해 주세요.")
    st.stop()

# 백테스팅 실행
with st.spinner('백테스팅 계산 중...'):
    bh_history = run_benchmark(filtered_data_dict, base_asset)
    bh_leverage_history = run_benchmark(filtered_data_dict, leverage_asset)
    
    # 전략 실행 및 추가 정보 추출을 위해 로직을 가져옴
    golden_history = run_golden_strategy(filtered_data_dict, filtered_fg_df, filtered_vix_df, leverage_asset, base_asset, cash_ratio)

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
st.write("")

# 지표 계산
bh_metrics = calculate_metrics(bh_history.rename(columns={'Value': 'PortfolioValue'}))
golden_metrics = calculate_metrics(golden_history.rename(columns={'Value': 'PortfolioValue'}))

# 메인 지표 표시 (3열)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("누적 수익률", f"{golden_metrics['Cumulative Return']:.1%}", 
              delta=f"{(golden_metrics['Cumulative Return'] - bh_metrics['Cumulative Return']):.1%}")
with col2:
    st.metric("CAGR (연간 성장률)", f"{golden_metrics['CAGR']:.1%}")
with col3:
    st.metric("MDD (최대 낙폭)", f"{golden_metrics['MDD']:.1%}")

# 차트 시각화
st.subheader("성과 비교 차트")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bh_history['Value'] / 10000, label=f'{base_asset} Buy & Hold', color='gray', alpha=0.5)
ax.plot(bh_leverage_history['Value'] / 10000, label=f'{leverage_asset} Buy & Hold', color='#ff7f0e', alpha=0.5)
ax.plot(golden_history['Value'] / 10000, label=f'Golden Strat ({leverage_asset})', color='#1f77b4', linewidth=2)
ax.set_ylabel('Normalized Value')
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# 상세 데이터 표 (최근 10일)
st.subheader("최근 성과 데이터")
display_df = golden_history.tail(10).copy()
display_df['Value'] = display_df['Value'].map('{:,.0f}'.format)
st.table(display_df)

st.info("💡 팁: 핸드폰 브라우저 메뉴에서 '홈 화면에 추가'를 누르면 앱처럼 사용할 수 있습니다.")
