import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import requests
import datetime

def fetch_data(tickers, start_date, end_date):
    """
    지정된 티커의 과거 데이터를 가져옵니다.
    """
    data = {}
    for ticker in tickers:
        print(f"{ticker} 데이터 가져오는 중...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            print(f"경고: {ticker}에 대한 데이터를 찾을 수 없습니다.")
            continue
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.loc[:, ~df.columns.duplicated()]
        data[ticker] = df
    return data

def calculate_indicators(df):
    """
    pandas_ta를 사용하여 기술적 지표를 계산합니다.
    """
    if 'Close' not in df.columns:
        print("에러: 'Close' 컬럼을 찾을 수 없습니다.")
        return df
        
    close_prices = pd.to_numeric(df['Close'], errors='coerce')
    df['RSI'] = ta.rsi(close_prices, length=14)
    
    bbands = ta.bbands(close_prices, length=20, std=2)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    
    df['SMA_20'] = ta.sma(close_prices, length=20)
    df['SMA_50'] = ta.sma(close_prices, length=50)
    df['SMA_200'] = ta.sma(close_prices, length=200)
    
    macd = ta.macd(close_prices)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
    
    return df

def get_fear_and_greed_historical():
    """
    CNN Fear & Greed Index의 과거 데이터를 가져옵니다.
    """
    url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        series = data.get('fear_and_greed_historical', {}).get('data', [])
        if not series: return pd.DataFrame()
        df = pd.DataFrame(series)
        df['x'] = pd.to_datetime(df['x'], unit='ms')
        df.rename(columns={'x': 'Date', 'y': 'FearGreed'}, inplace=True)
        df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"Fear & Greed Fetch Error: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # 데이터 저장 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "data")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    start_date = '2019-01-01'
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 1. ETF 데이터 (QQQ, QLD, TQQQ, TLT, TMF)
    etf_tickers = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TMF']
    data = fetch_data(etf_tickers, start_date, end_date)
    for ticker, df in data.items():
        if not df.empty:
            df = calculate_indicators(df)
            save_path = os.path.join(data_dir, f"{ticker}_data.csv")
            df.to_csv(save_path, index_label='Date')
            print(f"{ticker} 데이터 저장 완료: {save_path}")
            
    # 2. VIX 데이터
    vix_data = fetch_data(['^VIX'], start_date, end_date)
    if '^VIX' in vix_data and not vix_data['^VIX'].empty:
        save_path = os.path.join(data_dir, "vix_data.csv")
        vix_data['^VIX'].to_csv(save_path, index_label='Date')
        print(f"VIX 데이터 저장 완료: {save_path}")
        
    # 3. Fear & Greed 데이터
    print("Fear & Greed 데이터 가져오는 중...")
    fg_df = get_fear_and_greed_historical()
    if not fg_df.empty:
        save_path = os.path.join(data_dir, "fear_greed_data.csv")
        fg_df.to_csv(save_path)
        print(f"Fear & Greed 데이터 저장 완료: {save_path}")

    # 4. 거시 경제 지표 (10년물 금리 등)
    print("거시 경제 지표 가져오는 중...")
    macro_tickers = ['^TNX'] # 10Y Yield
    macro_data = fetch_data(macro_tickers, start_date, end_date)
    if macro_data:
        macro_combined = pd.DataFrame()
        if '^TNX' in macro_data:
            macro_combined['US10Y'] = macro_data['^TNX']['Close']
        
        if not macro_combined.empty:
            save_path = os.path.join(data_dir, "macro_data.csv")
            macro_combined.to_csv(save_path, index_label='Date')
            print(f"거시 경제 데이터 저장 완료: {save_path}")
