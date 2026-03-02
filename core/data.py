import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import datetime
import streamlit as st

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
                # pandas_ta 버전에 따라 직접 컬럼명 매칭이 안 될 경우 concat으로 전체 유지
                df = pd.concat([df, macd], axis=1)
        
        # 볼린저 밴드 추가 (20일, 2표준편차)
        bbands = ta.bbands(close_prices, length=20, std=2)
        if bbands is not None:
            l_col = [c for c in bbands.columns if c.startswith('BBL_')]
            u_col = [c for c in bbands.columns if c.startswith('BBU_')]
            if l_col and u_col:
                df['BB_Lower'] = bbands[l_col[0]]
                df['BB_Upper'] = bbands[u_col[0]]
                
        # [신규] 슬로우 스토캐스틱 추가 (12, 5, 5)
        if 'High' in df.columns and 'Low' in df.columns:
            stoch = ta.stoch(df['High'], df['Low'], close_prices, k=12, d=5, smooth_k=5)
            if stoch is not None:
                k_col = [c for c in stoch.columns if 'STOCHk_' in c]
                d_col = [c for c in stoch.columns if 'STOCHd_' in c]
                if k_col and d_col:
                    df['STOCH_K'] = stoch[k_col[0]]
                    df['STOCH_D'] = stoch[d_col[0]]
        # [신규 추가] 보조지표(DMI, Williams %R, PSAR) 계산
        if 'High' in df.columns and 'Low' in df.columns:
            # 1. DMI(ADX) 추가 (14일 기준)
            adx = ta.adx(df['High'], df['Low'], close_prices, length=14)
            if adx is not None:
                adx_col = [c for c in adx.columns if c.startswith('ADX_')]
                dmp_col = [c for c in adx.columns if c.startswith('DMP_')]
                dmn_col = [c for c in adx.columns if c.startswith('DMN_')]
                if adx_col: df['ADX'] = adx[adx_col[0]]
                if dmp_col: df['DMP'] = adx[dmp_col[0]]
                if dmn_col: df['DMN'] = adx[dmn_col[0]]
                
            # 2. Williams %R 추가 (14일 기준)
            willr = ta.willr(df['High'], df['Low'], close_prices, length=14)
            if willr is not None:
                # pandas_ta willr 반환 형태가 Series 인 경우를 고려
                if isinstance(willr, pd.Series):
                    df['WILLR'] = willr
                else:
                    willr_col = [c for c in willr.columns if 'WILLR' in c]
                    if willr_col: df['WILLR'] = willr[willr_col[0]]
                
            # 3. PSAR 추가 (단기 및 장기 방향성 추적 점)
            psar = ta.psar(df['High'], df['Low'], close_prices)
            if psar is not None:
                # 상승/하락 PSAR 컬럼 중 존재하는 값을 결합하여 단일 컬럼으로 만듦
                l_col = [c for c in psar.columns if c.startswith('PSARl_')]
                s_col = [c for c in psar.columns if c.startswith('PSARs_')]
                
                psar_combined = pd.Series(index=df.index, dtype=float)
                if l_col: psar_combined = psar_combined.fillna(psar[l_col[0]])
                if s_col: psar_combined = psar_combined.fillna(psar[s_col[0]])
                df['PSAR'] = psar_combined
                
        return df

    @classmethod
    def fetch_live_data(cls, start_date='2010-01-01'):
        print(f"라이브 데이터 수집 시작 (시작일: {start_date})...")
        # KST(UTC+9) 강제 적용
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        end_date = now_kst.strftime('%Y-%m-%d')
        # start_date = '2010-01-01'  # 🌟 파라미터로 대체됨
        
        data_dict = {}
        tickers = ['QQQ', 'QLD', 'TQQQ', 'SOXX', 'USD', 'SOXL', 'TLT', 'TMF']
        try:
            full_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
            for ticker in tickers:
                if ticker in full_data and not full_data[ticker].empty:
                    df = full_data[ticker].copy()
                    df = df.ffill().bfill() 
                    df = cls.calculate_indicators(df)
                    data_dict[ticker] = df
            
            # VIX 및 VXN 데이터 별도 수집 (동기화)
            # VIX 및 VXN 데이터 별도 수집 (동기화)
            try:
                # 개별 다운로드로 MultiIndex 문제 회피 및 안정성 확보
                v_tickers = {'VIX': '^VIX', 'VXN': '^VXN'}
                v_data_list = []
                for name, ticker in v_tickers.items():
                    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not raw.empty:
                        c_data = raw['Close']
                        s = c_data.iloc[:, 0].copy() if isinstance(c_data, pd.DataFrame) else c_data.copy()
                        s.name = name
                        # 인덱스 정규화 (시간 제거)
                        try: s.index = s.index.tz_localize(None).normalize()
                        except: s.index = s.index.normalize()
                        v_data_list.append(s)
                
                vix_df = pd.DataFrame()
                if v_data_list:
                    vix_df = pd.concat(v_data_list, axis=1)
                    vix_df = vix_df[~vix_df.index.duplicated(keep='first')]
                    vix_df = vix_df.ffill().bfill()
            except Exception as e:
                print(f"VIX/VXN Download Error: {e}")
                vix_df = pd.DataFrame()
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
        m_tickers_raw = ['^TNX', '^IRX', '^PCCR']
        try:
            m_data = yf.download(m_tickers_raw, period='5d', progress=False, group_by='ticker')
            # 🌟 속도 저하의 주범인 PCCR(유령 티커)을 제거했습니다.
            m_map = {'^TNX': 'US10Y', '^IRX': 'US03M'} 
            for t_raw, t_name in m_map.items():
                if t_raw in m_data and not m_data[t_raw].empty:
                    val = m_data[t_raw]['Close'].dropna().iloc[-1]
                    macro_data_map[t_name] = val
            fetch_status['Macro'] = 'Success'
        except:
            fetch_status['Macro'] = 'Error'
        
        if not vix_df.empty:
            if 'VIX' in vix_df.columns:
                vix_vals = vix_df['VIX'].dropna()
                if not vix_vals.empty:
                    macro_data_map['VIX'] = vix_vals.iloc[-1]
                    fetch_status['VIX'] = 'Success'
            if 'VXN' in vix_df.columns:
                vxn_vals = vix_df['VXN'].dropna()
                if not vxn_vals.empty:
                    macro_data_map['VXN'] = vxn_vals.iloc[-1]
                    fetch_status['VXN'] = 'Success'
        if not fg_df.empty: fetch_status['FearGreed'] = 'Success'
        
        macro_df = pd.DataFrame([macro_data_map])
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"데이터 수집 완료 ({len(data_dict)} 개 티커)")
        return data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time

    @classmethod
    def fetch_current_prices(cls, tickers):
        """핵심 티커들의 현재가를 실시간으로 수집하여 반환 (장중 프리뷰용)"""
        prices = {}
        try:
            # period='1d'로 가장 최근 1분봉 또는 마지막가를 가져옴
            data = yf.download(tickers, period='1d', interval='1m', progress=False)
            if not data.empty:
                # pandas_ta 등으로 가공하지 않고 원본 종가만 취함
                if len(tickers) > 1:
                    close_data = data['Close']
                    for ticker in tickers:
                        if ticker in close_data.columns:
                            prices[ticker] = close_data[ticker].dropna().iloc[-1]
                else:
                    prices[tickers[0]] = data['Close'].dropna().iloc[-1]
            else:
                # download 실패 시 t.fast_info 시도
                for ticker in tickers:
                    try:
                        t = yf.Ticker(ticker)
                        prices[ticker] = t.fast_info['last_price']
                    except: pass
        except Exception as e:
            print(f"현재가 수집 오류: {e}")
        return prices

    @classmethod
    def inject_virtual_close(cls, data_dict, current_prices):
        """기존 데이터셋 마지막 행에 실시간 현재가를 '임시 종가'로 주입하고 지표 재계산"""
        cloned_dict = {}
        today = datetime.datetime.now().date()
        
        for ticker, df in data_dict.items():
            if ticker in current_prices and not df.empty:
                # 1. 원본 복사
                new_df = df.copy()
                price = current_prices[ticker]
                
                # 만약 마지막 데이터가 오늘이라면 덮어쓰기, 아니면 새로 추가
                if new_df.index[-1].date() == today:
                    new_df.loc[new_df.index[-1], 'Close'] = price
                    if 'High' in new_df.columns: new_df.loc[new_df.index[-1], 'High'] = max(new_df.loc[new_df.index[-1], 'High'], price)
                    if 'Low' in new_df.columns: new_df.loc[new_df.index[-1], 'Low'] = min(new_df.loc[new_df.index[-1], 'Low'], price)
                else:
                    # 새로운 행 추가
                    last_row = new_df.iloc[-1].copy()
                    new_row = pd.Series(index=new_df.columns, dtype=float)
                    # 이전 행의 값들을 기본으로 하되 가격 정보는 현재가로 채움
                    for col in new_df.columns:
                        new_row[col] = last_row[col]
                    
                    new_row['Open'] = price
                    new_row['High'] = price
                    new_row['Low'] = price
                    new_row['Close'] = price
                    
                    # 지표들은 calculate_indicators에서 갱신됨
                    new_df = pd.concat([new_df, pd.DataFrame([new_row], index=[pd.Timestamp(today)])])
                
                # [중요] 지표 재계산 (RSI, MACD 등은 전체 데이터 기반이므로 다시 돌려야 함)
                new_df = cls.calculate_indicators(new_df)
                cloned_dict[ticker] = new_df
            else:
                cloned_dict[ticker] = df.copy()
        return cloned_dict

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_all_data():
        return DataService.fetch_live_data()
