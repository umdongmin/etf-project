import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import datetime
import os
import streamlit as st
from core.news_service import NewsService

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
        tickers = ['QQQ', 'QLD', 'TQQQ']
        
        # [수정] 미 동부 시간(EST) 기준으로 '오늘' 날짜 및 장 마감 여부 계산
        # 현재는 단순 5시간 차이로 계산 (서머타임 고려 시 4시간)
        now_est = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5) 
        today_est = now_est.date()
        # [신규] 장 마감(16:00) 이후거나 주말인 경우 '오늘' 데이터도 확정으로 간주
        is_market_closed = now_est.hour >= 16 or now_est.weekday() >= 5

        try:
            print(f"메인 티커 데이터 다운로드 중: {tickers}...")
            full_data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker')
            print("메인 데이터 다운로드 완료. 처리 중...")
            for ticker in tickers:
                if ticker in full_data and not full_data[ticker].empty:
                    df = full_data[ticker].copy()
                    
                    # [신규] 장중 불완전한 데이터 필터링 (로직 고도화)
                    if df.index[-1].date() >= today_est:
                        if not is_market_closed:
                            # ⚠️ 장중(16시 이전)일 때는 아직 종가가 확정되지 않았으므로 제외 (프리뷰에서만 활용)
                            df = df[df.index.date < today_est]
                        else:
                            # ✅ 장 마감 이후에는 '오늘' 데이터를 확정 종가로 포함
                            pass
                        
                    df = df.ffill().bfill() 
                    df = cls.calculate_indicators(df)
                    data_dict[ticker] = df
            print(f"티커별 지표 계산 완료 ({len(data_dict)}개 티커)")
            
            # VIX 및 VXN 데이터 별도 수집 (동기화)
            # VIX 및 VXN 데이터 별도 수집 (동기화)
            try:
                # 개별 다운로드로 MultiIndex 문제 회피 및 안정성 확보
                v_tickers = {'VIX': '^VIX', 'VXN': '^VXN'}
                v_data_list = []
                for name, ticker in v_tickers.items():
                    print(f"{name} ({ticker}) 다운로드 중...")
                    raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not raw.empty:
                        c_data = raw['Close']
                        s = c_data.iloc[:, 0].copy() if isinstance(c_data, pd.DataFrame) else c_data.copy()
                        s.name = name
                        # 인덱스 정규화 (시간 제거)
                        try: s.index = s.index.tz_localize(None).normalize()
                        except: s.index = s.index.normalize()
                        v_data_list.append(s)
                
                vxn_df = pd.DataFrame()
                if v_data_list:
                    vxn_df = pd.concat(v_data_list, axis=1)
                    vxn_df = vxn_df[~vxn_df.index.duplicated(keep='first')]
                    
                    # [신규] VIX/VXN도 장중 데이터 필터링 (마켓 클로즈 여부 반영)
                    if not vxn_df.empty and vxn_df.index[-1].date() >= today_est:
                        if not is_market_closed:
                            vxn_df = vxn_df[vxn_df.index.date < today_est]
                        
                    vxn_df = vxn_df.ffill().bfill()
            except Exception as e:
                print(f"VIX/VXN Download Error: {e}")
                vxn_df = pd.DataFrame()
        except Exception as e:
            print(f"Main Download Error: {e}")
            vxn_df = pd.DataFrame()

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
        fetch_status = {'ETF': 'Success', 'VXN': 'Pending', 'FearGreed': 'Pending', 'Macro': 'Pending'}
        m_tickers_raw = ['^TNX', '^IRX']
        try:
            m_data = yf.download(m_tickers_raw, period='5d', progress=False, group_by='ticker')
            m_map = {'^TNX': 'US10Y', '^IRX': 'US03M'} 
            for t_raw, t_name in m_map.items():
                if t_raw in m_data and not m_data[t_raw].empty:
                    val = m_data[t_raw]['Close'].dropna().iloc[-1]
                    macro_data_map[t_name] = val
            fetch_status['Macro'] = 'Success'
        except:
            fetch_status['Macro'] = 'Error'
        
            if 'VXN' in vxn_df.columns:
                vxn_vals = vxn_df['VXN'].dropna()
                if not vxn_vals.empty:
                    macro_data_map['VXN'] = vxn_vals.iloc[-1]
                    fetch_status['VXN'] = 'Success'
        if not fg_df.empty: fetch_status['FearGreed'] = 'Success'
        
        macro_df = pd.DataFrame([macro_data_map])
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # [신규] 뉴스 심리 데이터 로드 (Supabase 우선 지원)
        news_df = pd.DataFrame()
        try:
            news_svc = NewsService()
            # 로컬 파일이 있거나, Supabase 설정이 되어 있는 경우 로드 시도
            if news_svc.supabase_url or os.path.exists(news_svc.db_path):
                news_df = news_svc.get_sentiment_data()
                if news_df.empty:
                    print(" [INFO] 뉴스 데이터셋이 비어 있습니다. (수파베이스 또는 로컬)")
            else:
                print(" [WARNING] 뉴스 DB 연결 경로(로컬/원격)를 찾을 수 없습니다.")
        except Exception as e:
            print(f"News Data Load Error: {e}")

        print(f"데이터 수집 완료 ({len(data_dict)} 개 티커)")
        return data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time

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
    def inject_virtual_close(cls, data_dict, current_prices, vxn_df, fg_df=None):
        """가상 종가를 주입하여 시뮬레이션을 실시간 데이터 기반으로 수행 (VIX/VXN 포함)"""
        cloned_dict = {}
        # [수정] 미 동부 시간(EST) 기준으로 '오늘' 날짜 계산
        now_est = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5) 
        today_ts = pd.Timestamp(now_est.date())
        
        # 1. 메인 티커 데이터 주입
        for ticker, df in data_dict.items():
            if ticker in current_prices and not df.empty:
                new_df = df.copy()
                new_df.index = new_df.index.normalize()
                price = current_prices[ticker]
                
                if new_df.index[-1] == today_ts:
                    new_df.loc[today_ts, 'Close'] = price
                else:
                    last_row = new_df.iloc[-1].copy()
                    new_row = pd.Series(index=new_df.columns, dtype=float)
                    for col in new_df.columns:
                        new_row[col] = last_row[col]
                    new_row['Open'] = price
                    new_row['High'] = price
                    new_row['Low'] = price
                    new_row['Close'] = price
                    # 인덱스 이름(Date 등) 유지
                    new_df = pd.concat([new_df, pd.DataFrame([new_row], index=[today_ts])])
                
                new_df = cls.calculate_indicators(new_df)
                cloned_dict[ticker] = new_df
            else:
                cloned_dict[ticker] = df.copy()

        # 2. VIX/VXN 데이터 주입 (있는 경우)
        new_vxn_df = vxn_df.copy() if vxn_df is not None else pd.DataFrame()
        if not new_vxn_df.empty:
            new_vxn_df.index = new_vxn_df.index.normalize()
            vix_val = current_prices.get('^VIX')
            vxn_val = current_prices.get('^VXN')
            
            # [수정] 0.0 값도 유효하므로 'is not None'으로 체크
            if (vix_val is not None) or (vxn_val is not None):
                if new_vxn_df.index[-1] == today_ts:
                    if vix_val is not None: new_vxn_df.loc[today_ts, 'VIX'] = vix_val
                    if vxn_val is not None: new_vxn_df.loc[today_ts, 'VXN'] = vxn_val
                else:
                    new_v_row = new_vxn_df.iloc[-1].copy()
                    if vix_val is not None: new_v_row['VIX'] = vix_val
                    if vxn_val is not None: new_v_row['VXN'] = vxn_val
                    new_vxn_df = pd.concat([new_vxn_df, pd.DataFrame([new_v_row], index=[today_ts])])

        # 3. Fear & Greed 주입 (현재가 수집 시 같이 가져왔다면 - 향후 확장 대비)
        new_fg_df = fg_df.copy() if fg_df is not None else pd.DataFrame()
        # (현재 Fear & Greed는 실시간 API 수집이 번거로우므로 기존 값 ffill 하도록 둠)

        return cloned_dict, new_vxn_df, new_fg_df

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_all_data():
        return DataService.fetch_live_data()
