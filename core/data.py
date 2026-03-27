import pandas as pd
import pandas_ta as ta
import yfinance as yf
import requests
import datetime
import time
import os
import json
from pathlib import Path
import streamlit as st
from core.news_service import NewsService

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
# 캐시에 저장할 원시 컬럼 (지표 제외)
_OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
_VXN_COLS   = ['VIX', 'VXN']

class DataService:
    """데이터 수집 및 전처리를 담당하는 클래스"""

    @staticmethod
    def _yf_download_with_retry(*args, max_retries=3, **kwargs):
        """yf.download를 Rate Limit 오류 시 exponential backoff으로 재시도"""
        # [수정] 대기 시간 대폭 연장 (배포 환경의 엄격한 제한 대응)
        delays = [30, 90, 180] 
        ticker_str = str(args[0]) if args else "Unknown"
        
        for attempt in range(max_retries):
            try:
                result = yf.download(*args, **kwargs)
                if result is not None and not result.empty:
                    return result
                
                # 빈 결과일 경우 티커 존재 여부나 네트워크 일시 오류 가능성
                if attempt < max_retries - 1:
                    print(f"[yfinance] 빈 데이터 반환 ({ticker_str}) — {delays[attempt]}초 후 재시도...")
                    time.sleep(delays[attempt])
            except Exception as e:
                err_str = str(e).lower()
                # yfinance의 내부 Rate Limit 예외명들 체크
                is_rate_limit = any(x in err_str for x in ['ratelimit', 'too many requests', '429', 'rate limited'])
                
                if is_rate_limit:
                    if attempt < max_retries - 1:
                        wait = delays[attempt]
                        print(f"[yfinance] Rate limit 감지 ({ticker_str}) — {wait}초 후 재시도 ({attempt+1}/{max_retries})...")
                        time.sleep(wait)
                    else:
                        print(f"[yfinance] Rate limit 재시도 초과 ({ticker_str}): {e}")
                        # 에러를 던지지 않고 빈 DF 반환하여 엔진에서 대체 로직(Approximation)이 돌게 함
                        return pd.DataFrame()
                else:
                    # 기타 치명적 에러 (네트워크 끊김 등)
                    print(f"[yfinance] 예외 발생 ({ticker_str}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(delays[attempt])
                    else:
                        return pd.DataFrame()
        return pd.DataFrame()

    # ──────────────────────────────────────────────
    # Parquet 파일 캐시 (앱 기동 속도 개선 핵심)
    # ──────────────────────────────────────────────

    @classmethod
    def _load_price_cache(cls):
        """parquet 캐시에서 raw OHLCV + VIX/VXN 로드.

        Returns
        -------
        raw_dict  : dict[ticker, DataFrame]  — OHLCV only
        vxn_raw   : DataFrame                — VIX/VXN columns
        last_date : str | None               — 'YYYY-MM-DD', 없으면 None
        """
        meta_path = CACHE_DIR / "cache_meta.json"
        if not meta_path.exists():
            return {}, pd.DataFrame(), None
        try:
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            last_date = meta.get('last_date')
            if not last_date:
                return {}, pd.DataFrame(), None

            tickers = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TBF', 'BIL', 'TMF', 'TMV', '^TNX']
            raw_dict = {}
            for ticker in tickers:
                safe = ticker.replace('^', '_')
                path = CACHE_DIR / f"raw_{safe}.parquet"
                if path.exists():
                    raw_dict[ticker] = pd.read_parquet(path)

            vxn_raw = pd.DataFrame()
            vxn_path = CACHE_DIR / "raw_vix_vxn.parquet"
            if vxn_path.exists():
                vxn_raw = pd.read_parquet(vxn_path)

            print(f"[Cache] 로드 완료: {len(raw_dict)}개 티커, 마지막 날짜: {last_date}")
            return raw_dict, vxn_raw, last_date
        except Exception as e:
            print(f"[Cache] 로드 실패 (전체 다운로드로 대체): {e}")
            return {}, pd.DataFrame(), None

    @classmethod
    def _save_price_cache(cls, raw_dict, vxn_raw, last_date_str):
        """raw OHLCV + VIX/VXN 를 parquet 캐시로 저장."""
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            for ticker, df in raw_dict.items():
                safe = ticker.replace('^', '_')
                cols = [c for c in _OHLCV_COLS if c in df.columns]
                df[cols].to_parquet(CACHE_DIR / f"raw_{safe}.parquet")

            if not vxn_raw.empty:
                cols = [c for c in _VXN_COLS if c in vxn_raw.columns]
                if cols:
                    vxn_raw[cols].to_parquet(CACHE_DIR / "raw_vix_vxn.parquet")

            meta = {
                'last_date': last_date_str,
                'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open(CACHE_DIR / "cache_meta.json", 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"[Cache] 저장 완료: {len(raw_dict)}개 티커, 마지막 날짜: {last_date_str}")
        except Exception as e:
            print(f"[Cache] 저장 실패 (무시하고 계속): {e}")

    @staticmethod
    def _merge_raw(cached: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        """캐시 DataFrame + 신규 DataFrame 병합 (중복 제거, 정렬)."""
        if cached.empty:
            return new
        if new.empty:
            return cached
        merged = pd.concat([cached, new])
        merged = merged[~merged.index.duplicated(keep='last')].sort_index()
        return merged

    @staticmethod
    def calculate_indicators(df):
        # yfinance 최신 버전에서 단일 티커도 MultiIndex 컬럼을 반환할 수 있으므로 평탄화
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
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

        # [신규] 이동평균선 추가
        df['SMA60'] = close_prices.rolling(window=60).mean()
        df['SMA200'] = close_prices.rolling(window=200).mean()

        # [신규] MACD Histogram 명시적 추가
        if macd is not None:
            h_col = [c for c in macd.columns if 'MACDh_' in c]
            if h_col:
                df['MACD_Hist'] = macd[h_col[0]]
                
        # [신규] 이전치(Prev_) 데이터 생성 (시그널 판별용)
        df['Prev_Close'] = close_prices.shift(1)
        df['Prev_RSI'] = df['RSI'].shift(1)
        df['Prev_RSI_SMA'] = df['RSI_SMA'].shift(1)
        if 'MACD' in df.columns:
            df['Prev_MACD'] = df['MACD'].shift(1)
        if 'MACD_Hist' in df.columns:
            df['Prev_MACD_Hist'] = df['MACD_Hist'].shift(1)
            df['Prev2_MACD_Hist'] = df['MACD_Hist'].shift(2)
                
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
                
                # DMP/DMN 이전치 추가
                df['Prev_DMP'] = df['DMP'].shift(1)
                df['Prev_DMN'] = df['DMN'].shift(1)
                
            # 2. Williams %R 추가 (14일 기준)
            willr = ta.willr(df['High'], df['Low'], close_prices, length=14)
            if willr is not None:
                # pandas_ta willr 반환 형태가 Series 인 경우를 고려
                if isinstance(willr, pd.Series):
                    df['WILLR'] = willr
                else:
                    willr_col = [c for c in willr.columns if 'WILLR' in c]
                    if willr_col: df['WILLR'] = willr[willr_col[0]]
                # Williams %R 이전치 추가
                df['Prev_WILLR'] = df['WILLR'].shift(1)
                
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
                df['Prev_SAR'] = df['PSAR'].shift(1)

        # [신규 추가] ATR 변동성 지표 (20일 기준)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], close_prices, length=20)
        df['Prev_ATR_20'] = df['ATR_20'].shift(1)
        df['ATR_14'] = ta.atr(df['High'], df['Low'], close_prices, length=14)
        df['Prev_ATR_14'] = df['ATR_14'].shift(1)

        # [신규 추가] RSI slope (기울기) — Buy2_Cross 필터용 (C-4B)
        df['RSI_slope3'] = df['RSI'].diff(3)
        df['RSI_slope5'] = df['RSI'].diff(5)

        # [신규 추가] StochRSI (14, 14, 3, 3) — Buy2_Cross 강도 확인용 (C-4B)
        try:
            srsi = ta.stochrsi(close_prices, length=14, rsi_length=14, k=3, d=3)
            if srsi is not None:
                k_col = [c for c in srsi.columns if 'STOCHRSIk_' in c]
                d_col = [c for c in srsi.columns if 'STOCHRSId_' in c]
                if k_col: df['StochRSI_K'] = srsi[k_col[0]]
                if d_col: df['StochRSI_D'] = srsi[d_col[0]]
        except Exception:
            pass

        return df

    @staticmethod
    def calculate_bond_indicators(df, prices_all=None):
        """채권 전략(V7)을 위한 지표 계산"""
        df = df.copy()
        
        # 1. FFV (10Y 적정금리) - macro_backtest.py 고정 수식 적용
        NEUTRAL_REAL_RATE = 0.75  # r*
        TERM_PREMIUM      = 0.50  # 기간 프리미엄
        
        if 'exp_inf' in df.columns:
            df['ffv'] = df['exp_inf'] + NEUTRAL_REAL_RATE + TERM_PREMIUM
            
            # app.py 등에서 넘겨준 TNX 가격 데이터와 결합
            if prices_all is not None and isinstance(prices_all, pd.DataFrame) and '^TNX' in prices_all.columns:
                # ── 데이터 정렬 중요: 일일 TNX 데이터를 매크로 인덱스에 맞게 리인덱싱 후 채움 ──
                df['tnx_val'] = prices_all['^TNX'].reindex(df.index).ffill()
                df['gap'] = df['tnx_val'] - df['ffv']
        
        # 2. TLT RSI (일일 데이터 기반으로 먼저 계산 후 매크로 인덱스에 매핑)
        if prices_all is not None and isinstance(prices_all, pd.DataFrame) and 'TLT' in prices_all.columns:
            daily_rsi = ta.rsi(prices_all['TLT'], length=14)
            df['tlt_rsi'] = daily_rsi.reindex(df.index).ffill()
        
        # 3. 수익률 곡선 모멘텀 (10Y-2Y)
        if 't10y2y' in df.columns:
            df['yc_mom'] = df['t10y2y'].diff(63) # 약 3개월 모멘텀
        
        return df

    @classmethod
    def fetch_live_data(cls, start_date='2010-01-01', skip_cache=False):
        # KST(UTC+9) 강제 적용
        now_kst = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=9)
        # yfinance end는 exclusive → 내일 날짜를 end로 사용해야 오늘 데이터 포함
        end_date     = now_kst.strftime('%Y-%m-%d')           # 필터링용 (오늘)
        end_date_yf  = (now_kst + datetime.timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance 전달용 (내일)

        # ── Parquet 캐시 로드 ──────────────────────────────
        cache_raw_dict, cache_vxn_raw, cache_last_date = {}, pd.DataFrame(), None
        if skip_cache:
            print(f"[실시간 모드] Parquet 캐시 무시, yfinance 전량 다운로드 (시작일: {start_date})...")
        else:
            cache_raw_dict, cache_vxn_raw, cache_last_date = cls._load_price_cache()

        if not skip_cache and cache_last_date:
            cache_last_dt = pd.to_datetime(cache_last_date)
            incremental_start = (cache_last_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            if incremental_start <= end_date_yf:
                start_date = incremental_start
                print(f"[Cache] 증분 조회: {start_date} ~ {end_date_yf}")
            else:
                # 캐시가 오늘까지 최신 → yfinance 조회 불필요
                start_date = None
                print(f"[Cache] 최신 상태 (마지막: {cache_last_date}). yfinance 스킵.")
        elif not skip_cache:
            print(f"라이브 데이터 수집 시작 (시작일: {start_date})...")
        # ───────────────────────────────────────────────────

        data_dict = {}
        tickers = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TBF', 'BIL', 'TMF', 'TMV', '^TNX']
        
        # [수정] 미 동부 시간(EST) 기준으로 '오늘' 날짜 및 장 마감 여부 계산
        # 현재는 단순 5시간 차이로 계산 (서머타임 고려 시 4시간)
        now_est = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5) 
        today_est = now_est.date()
        # [신규] 장 마감(16:00) 이후거나 주말인 경우 '오늘' 데이터도 확정으로 간주
        is_market_closed = now_est.hour >= 16 or now_est.weekday() >= 5

        # ── raw OHLCV 수집 (start_date=None 이면 캐시만 사용) ──────────
        new_raw_dict = {}   # 신규 다운로드분 raw OHLCV
        new_vxn_raw  = pd.DataFrame()
        vxn_df       = pd.DataFrame()

        # 증분 구간이 7일 이하면 재시도 1회만 (실패해도 캐시로 충분)
        incremental_days = (pd.to_datetime(end_date_yf) - pd.to_datetime(start_date)).days if start_date else 0
        is_short_increment = cache_last_date is not None and incremental_days <= 7
        yf_max_retries = 1 if is_short_increment else 3

        try:
            if start_date is not None:
                print(f"메인 티커 데이터 다운로드 중: {tickers}...")
                full_data = cls._yf_download_with_retry(tickers, start=start_date, end=end_date_yf, progress=False, group_by='ticker', max_retries=yf_max_retries)
                print("메인 데이터 다운로드 완료. 처리 중...")
                for ticker in tickers:
                    try:
                        if isinstance(full_data.columns, pd.MultiIndex):
                            if ticker in full_data.columns.get_level_values(0):
                                df_raw = full_data[ticker]
                            elif ticker in full_data.columns.get_level_values(1):
                                df_raw = full_data.xs(ticker, level=1, axis=1)
                            else:
                                continue
                        elif ticker in full_data:
                            df_raw = full_data[ticker]
                        else:
                            continue
                        if df_raw.empty:
                            continue
                        if isinstance(df_raw, pd.Series):
                            df_raw = df_raw.to_frame(name='Close')
                        if isinstance(df_raw.columns, pd.MultiIndex):
                            df_raw.columns = df_raw.columns.get_level_values(0)
                    except Exception as e:
                        print(f"  [{ticker}] 데이터 추출 실패: {e}")
                        continue

                    # 장중 불완전 데이터 필터링
                    if not df_raw.empty and df_raw.index[-1].date() >= today_est:
                        if not is_market_closed:
                            df_raw = df_raw[df_raw.index.date < today_est]
                    new_raw_dict[ticker] = df_raw

            # ── 캐시와 병합 후 지표 계산 ──────────────────────────────
            merged_raw_dict = {}
            for ticker in tickers:
                cached = cache_raw_dict.get(ticker, pd.DataFrame())
                new    = new_raw_dict.get(ticker, pd.DataFrame())
                merged = cls._merge_raw(cached, new)
                if not merged.empty:
                    merged_raw_dict[ticker] = merged

            for ticker, df_raw in merged_raw_dict.items():
                df = df_raw.ffill().bfill()
                df = cls.calculate_indicators(df)
                data_dict[ticker] = df
            print(f"티커별 지표 계산 완료 ({len(data_dict)}개 티커)")

            # 핵심 티커 누락 시 개별 재시도 (신규 다운로드 구간에서만)
            if start_date is not None:
                critical = ['QQQ', 'TQQQ', 'TLT', 'TMF', 'TBF']
                missing = [t for t in critical if t not in data_dict]
                if missing:
                    print(f"[경고] 누락 티커 개별 재시도: {missing}")
                    time.sleep(5)
                    for t in missing:
                        try:
                            raw = cls._yf_download_with_retry(t, start=start_date, end=end_date_yf, progress=False, max_retries=yf_max_retries)
                            if not raw.empty:
                                if isinstance(raw.columns, pd.MultiIndex):
                                    raw.columns = raw.columns.get_level_values(0)
                                merged = cls._merge_raw(cache_raw_dict.get(t, pd.DataFrame()), raw)
                                df = merged.ffill().bfill()
                                df = cls.calculate_indicators(df)
                                data_dict[t] = df
                                merged_raw_dict[t] = merged
                                print(f"  [{t}] 개별 재시도 성공 ({len(df)}행)")
                        except Exception as e:
                            print(f"  [{t}] 개별 재시도 실패: {e}")

            # 핵심 티커가 여전히 없으면: 캐시가 있으면 경고만, 없으면 예외
            still_missing = [t for t in ['QQQ', 'TLT'] if t not in data_dict]
            if still_missing:
                if is_short_increment:
                    print(f"[Cache] 증분 조회 실패 → 캐시 데이터 사용 (누락: {still_missing})")
                else:
                    raise RuntimeError(f"핵심 티커 로드 실패 (캐시 저장 안함): {still_missing}")

            # ── VIX / VXN ─────────────────────────────────────────────
            try:
                if start_date is not None:
                    v_tickers = {'VIX': '^VIX', 'VXN': '^VXN'}
                    v_data_list = []
                    for name, ticker in v_tickers.items():
                        print(f"{name} ({ticker}) 다운로드 중...")
                        raw = cls._yf_download_with_retry(ticker, start=start_date, end=end_date_yf, progress=False, max_retries=yf_max_retries)
                        if not raw.empty:
                            c_data = raw['Close']
                            s = c_data.iloc[:, 0].copy() if isinstance(c_data, pd.DataFrame) else c_data.copy()
                            s.name = name
                            try: s.index = s.index.tz_localize(None).normalize()
                            except: s.index = s.index.normalize()
                            v_data_list.append(s)
                    if v_data_list:
                        new_vxn_raw = pd.concat(v_data_list, axis=1)
                        new_vxn_raw = new_vxn_raw[~new_vxn_raw.index.duplicated(keep='first')]

                # 캐시와 병합
                merged_vxn = cls._merge_raw(cache_vxn_raw, new_vxn_raw)
                if not merged_vxn.empty:
                    vxn_df = merged_vxn.copy()
                    # 장중 필터링
                    if vxn_df.index[-1].date() >= today_est and not is_market_closed:
                        vxn_df = vxn_df[vxn_df.index.date < today_est]
                    vxn_df = vxn_df.ffill().bfill()
                    if 'VIX' in vxn_df.columns: vxn_df['Prev_VIX'] = vxn_df['VIX'].shift(1)
                    if 'VXN' in vxn_df.columns: vxn_df['Prev_VXN'] = vxn_df['VXN'].shift(1)
            except Exception as e:
                print(f"VIX/VXN Download Error: {e}")
                vxn_df = cache_vxn_raw.copy() if not cache_vxn_raw.empty else pd.DataFrame()

            # ── parquet 캐시 저장 (신규 데이터가 있을 때만) ────────────
            if start_date is not None and merged_raw_dict:
                last_dates = [df.index.max() for df in merged_raw_dict.values() if not df.empty]
                if last_dates:
                    last_date_str = max(last_dates).strftime('%Y-%m-%d')
                    cls._save_price_cache(merged_raw_dict, merged_vxn if not merged_vxn.empty else new_vxn_raw, last_date_str)

        except Exception as e:
            print(f"Main Download Error: {e}")
            # 캐시가 있으면 캐시 데이터로 fallback
            if cache_raw_dict:
                print("[Cache] 다운로드 실패 → 캐시 데이터로 fallback")
                for ticker, df_raw in cache_raw_dict.items():
                    df = df_raw.ffill().bfill()
                    df = cls.calculate_indicators(df)
                    data_dict[ticker] = df
                if not cache_vxn_raw.empty:
                    vxn_df = cache_vxn_raw.copy().ffill().bfill()
                    if 'VIX' in vxn_df.columns: vxn_df['Prev_VIX'] = vxn_df['VIX'].shift(1)
                    if 'VXN' in vxn_df.columns: vxn_df['Prev_VXN'] = vxn_df['VXN'].shift(1)
            else:
                vxn_df = pd.DataFrame()

        fg_df = pd.DataFrame()  # CNN F&G 미사용 — engine.py에서 FG=50 기본값 처리

        # FRED 매크로 데이터 — 캐시 우선 로드, 없거나 오래되면 API 증분 호출
        fetch_status = {'VXN': 'Pending', 'FRED': 'Pending'}
        _fred_cache_path = CACHE_DIR / "macro_fred.parquet"
        _fred_macro_series = [
            ('DFEDTARU', 'ff_rate',   0),
            ('CPILFESL', 'core_cpi', 30),
            ('PPIACO',   'ppi',      14),
            ('UNRATE',   'unrate',   30),
            ('NROU',     'nrou',     45),
            ('T10YIE',   'exp_inf',   0),
            ('T10Y2Y',   't10y2y',    0),
            ('DFII10',   'real_rate_10y', 0),   # 10Y TIPS 실질금리
            ('T5YIFR',   'breakeven_5y5y', 0),  # 5Y5Y Forward Breakeven 인플레 기대
            ('BAMLH0A0HYM2', 'hy_spread', 0),   # ICE BofA HY OAS (신용 리스크 프리미엄)
        ]
        macro_df = pd.DataFrame()

        # 캐시 로드
        if _fred_cache_path.exists():
            try:
                macro_df = pd.read_parquet(_fred_cache_path)
                print(f"✅ FRED 캐시 로드 (마지막: {macro_df.index[-1].date()})")
            except Exception as e:
                print(f"FRED 캐시 로드 실패: {e}")

        # 캐시에 없는 신규 시리즈가 있으면 전체 재수집 강제 (컬럼 추가 시 자동 대응)
        _expected_cols = {col for _, col, _ in _fred_macro_series}
        _missing_cols = _expected_cols - set(macro_df.columns) if not macro_df.empty else set()
        if _missing_cols:
            print(f"⚠️ FRED 캐시에 신규 컬럼 누락 {_missing_cols} → 전체 재수집 강제")
            macro_df = pd.DataFrame()  # 캐시 무효화 → 전체 재수집

        # 캐시가 3일 이상 오래됐거나 없으면 API 증분 호출
        _cache_age = (datetime.date.today() - macro_df.index[-1].date()).days if not macro_df.empty else 999
        if _cache_age > 3:
            try:
                api_key = os.getenv("FRED_API_KEY")
                if api_key:
                    fred_start = (macro_df.index[-1].date() - datetime.timedelta(days=7)).strftime('%Y-%m-%d') \
                        if not macro_df.empty else '2010-01-01'
                    print(f"FRED API 증분 호출 ({fred_start} 이후)...")
                    macro_data_list = []
                    for sid, col, lag in _fred_macro_series:
                        try:
                            url = (f"https://api.stlouisfed.org/fred/series/observations"
                                   f"?series_id={sid}&api_key={api_key}"
                                   f"&file_type=json&observation_start={fred_start}&observation_end={end_date}")
                            r = requests.get(url, timeout=10)
                            if r.status_code == 200:
                                obs = r.json().get('observations', [])
                                temp = pd.DataFrame(obs)
                                if not temp.empty:
                                    temp['date'] = pd.to_datetime(temp['date'])
                                    temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
                                    temp = temp.set_index('date')[['value']].rename(columns={'value': col})
                                    if lag > 0:
                                        temp.index = temp.index + datetime.timedelta(days=lag)
                                    macro_data_list.append(temp)
                        except Exception as e:
                            print(f"FRED Fetch Error ({sid}): {e}")

                    if macro_data_list:
                        new_macro = pd.concat(macro_data_list, axis=1)
                        if not macro_df.empty:
                            macro_df = pd.concat([macro_df, new_macro])
                            macro_df = macro_df[~macro_df.index.duplicated(keep='last')].sort_index()
                        else:
                            macro_df = new_macro
                        if not macro_df.empty:
                            now_kst = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Seoul').normalize().tz_localize(None)
                            full_idx = pd.date_range(start=macro_df.index.min(), end=now_kst, freq='D')
                            macro_df = macro_df.reindex(full_idx).ffill().bfill()
                        print(f"✅ FRED API 증분 완료 ({len(macro_df)}행)")
            except Exception as e:
                print(f"FRED API 호출 실패: {e}")

        if not macro_df.empty:
            fetch_status['FRED'] = 'Success'
        else:
            print("⚠️ FRED 데이터 없음 — Bond 전략 비활성화")
            fetch_status['FRED'] = 'Error'

        if not vxn_df.empty and 'VXN' in vxn_df.columns:
            vxn_vals = vxn_df['VXN'].dropna()
            if not vxn_vals.empty:
                fetch_status['VXN'] = 'Success'

        # [신규] 채권 데이터에 채권 전용 지표 추가
        if 'TLT' in data_dict:
            # ^TNX를 포함한 전체 종가 데이터를 넘겨 FFV Gap 계산
            prices_all = pd.DataFrame({t: data_dict[t]['Close'] for t in data_dict if 'Close' in data_dict[t].columns})
            macro_df = cls.calculate_bond_indicators(macro_df, prices_all)

        # [신규] FOMC 회의록 감성 점수 주입 (fomc_sentiment: -1.0 ~ +1.0)
        # fomc_scores.csv → 회의록 발표일 기준 ffill → 일별 시계열로 macro_df에 병합
        try:
            fomc_scores_path = Path(__file__).resolve().parent.parent / "data" / "fomc_scores.csv"
            if fomc_scores_path.exists():
                fomc_df = pd.read_csv(fomc_scores_path, dtype={"date": str})
                fomc_df["date"] = pd.to_datetime(fomc_df["date"], format="%Y%m%d")
                fomc_df = fomc_df.set_index("date")[["score"]].rename(columns={"score": "fomc_sentiment"})
                fomc_df = fomc_df[~fomc_df.index.duplicated(keep="last")].sort_index()

                if not macro_df.empty:
                    # macro_df 인덱스 범위에 맞춰 리인덱싱 후 ffill (다음 회의록 전까지 동일 점수 유지)
                    fomc_df = fomc_df.reindex(macro_df.index).ffill()
                    macro_df["fomc_sentiment"] = fomc_df["fomc_sentiment"]
                    print(f"FOMC 감성 점수 주입 완료 ({fomc_df['fomc_sentiment'].notna().sum()}일)")
                else:
                    print("macro_df 비어있어 FOMC 감성 점수 스킵")
            else:
                print(f"FOMC 감성 점수 파일 없음 (스킵): {fomc_scores_path}")
        except Exception as e:
            print(f"FOMC 감성 점수 주입 오류 (무시): {e}")

        # [수정] macro_df는 이제 시계열 데이터를 포함함
        # macro_df = pd.DataFrame([macro_data_map])
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # [삭제] 뉴스 심리 데이터 로드 (AI 보조 지표 기능 제거로 인한 삭제)
        news_df = pd.DataFrame()

        print(f"데이터 수집 완료 ({len(data_dict)} 개 티커)")

        # 데이터 검증
        data_warnings = DataService.validate_data(data_dict)
        if data_warnings:
            fetch_status['DataWarnings'] = data_warnings
            for w in data_warnings:
                print(f"⚠️ [데이터 검증] {w}")
        else:
            fetch_status['DataWarnings'] = []

        return data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time

    @classmethod
    def validate_data(cls, data_dict: dict) -> list:
        """데이터 유효성 검증 — 최신성 및 필수 컬럼 누락 확인"""
        warnings = []
        today = datetime.date.today()
        # 주말 포함 최대 허용 지연: 3거래일 (금요일 → 수요일까지 허용)
        max_delay_days = 5
        required_cols = ['Close', 'RSI', 'MACD']

        for ticker, df in data_dict.items():
            if df is None or df.empty:
                warnings.append(f"{ticker}: 데이터 없음")
                continue

            # ① 최신성 검사
            last_date = df.index[-1].date()
            delay = (today - last_date).days
            if delay > max_delay_days:
                warnings.append(f"{ticker}: 마지막 데이터 {last_date} ({delay}일 지연)")

            # ③ 필수 컬럼 누락 검사
            missing = [c for c in required_cols if c not in df.columns or df[c].isna().all()]
            if missing:
                warnings.append(f"{ticker}: 컬럼 누락 또는 전체 NaN {missing}")

        return warnings

    @classmethod
    def fetch_current_prices(cls, tickers, use_cache=True):
        """핵심 티커들의 현재가를 실시간으로 수집하여 반환 (장중 프리뷰용)

        Parameters:
        - tickers: 티커 리스트
        - use_cache: True면 session_state 5분 TTL 캐싱 활용
        """
        import streamlit as st

        prices = {}
        from core.session_keys import SK
        now = datetime.datetime.now()
        bucket = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
        tickers_key = ','.join(sorted(tickers))
        bucket_str  = bucket.strftime('%H%M')
        cache_key   = SK.cur_prices_cache(tickers_key, bucket_str)
        time_key    = SK.cur_prices_cache_ts(tickers_key, bucket_str)

        # session_state TTL 캐시 확인 (5분)
        if use_cache and hasattr(st, 'session_state'):
            cached    = st.session_state.get(cache_key)
            cached_ts = st.session_state.get(time_key)
            if cached is not None and cached_ts and (now - cached_ts).total_seconds() < 300:
                return cached

        try:
            # 1. 종가 다운로드 시도 (1분 간격 제거 → 5분 간격으로 변경하여 레이트 제한 완화)
            try:
                data = cls._yf_download_with_retry(
                    tickers, period='1d', interval='5m', progress=False,
                    max_retries=2  # 재시도 횟수 축소
                )
                if not data.empty:
                    if len(tickers) > 1:
                        close_data = data['Close']
                        for ticker in tickers:
                            if ticker in close_data.columns:
                                prices[ticker] = close_data[ticker].dropna().iloc[-1]
                    else:
                        prices[tickers[0]] = data['Close'].dropna().iloc[-1]
            except Exception as e1:
                print(f"⚠️ 1분/5분봉 다운로드 실패: {e1}")
                pass

            # 2. 여전히 부족한 데이터는 fast_info로 채움 (레이트 제한에 더 강함)
            for ticker in tickers:
                if ticker not in prices:
                    try:
                        t = yf.Ticker(ticker)
                        prices[ticker] = t.fast_info['last_price']
                        print(f"✓ {ticker} fast_info에서 조회")
                    except Exception as e2:
                        print(f"⚠️ {ticker} 현재가 조회 실패: {e2}")
                        prices[ticker] = None
        except Exception as e:
            print(f"❌ 현재가 수집 오류: {e}")

        # session_state 캐싱 저장
        if use_cache and hasattr(st, 'session_state'):
            st.session_state[cache_key] = prices
            st.session_state[time_key]  = now

        return prices

    @staticmethod
    def get_price(ticker, price_map=None):
        """단일 티커 현재가 조회 — 전역 공용 함수.

        1순위: price_map dict에서 조회 (대소문자 무관)
        2순위: yfinance fast_info
        3순위: yfinance history 최근 종가

        Parameters
        ----------
        ticker    : str   — 티커 심볼 (예: 'QQQ', 'TQQQ')
        price_map : dict | None  — 미리 조회된 가격 dict (fetch_current_prices 결과 등)

        Returns
        -------
        float — 가격, 조회 실패 시 0.0
        """
        if price_map:
            p = price_map.get(ticker) or price_map.get(ticker.upper()) or price_map.get(ticker.lower())
            if p:
                return float(p)
        try:
            t = yf.Ticker(ticker)
            price = (getattr(t.fast_info, 'last_price', None)
                     or getattr(t.fast_info, 'regularMarketPrice', None))
            if price:
                return float(price)
            hist = t.history(period='2d')
            if not hist.empty and 'Close' in hist.columns:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def get_prices(tickers, price_map=None):
        """복수 티커 현재가 배치 조회 — 전역 공용 함수.

        1순위: price_map에서 이미 있는 것 재사용
        2순위: 누락된 티커만 yf.download 배치 조회
        3순위: 여전히 없는 티커는 fast_info 개별 fallback

        Parameters
        ----------
        tickers   : list[str]
        price_map : dict | None  — 부분적으로 채워진 가격 dict (업데이트 후 반환)

        Returns
        -------
        dict[str, float]  — {TICKER: price}
        """
        result = {}
        if price_map:
            for t in tickers:
                p = price_map.get(t) or price_map.get(t.upper())
                if p:
                    result[t.upper()] = float(p)

        missing = [t for t in tickers if t.upper() not in result]
        if not missing:
            return result

        # 배치 다운로드
        try:
            data = yf.download(missing, period='1d', auto_adjust=True, progress=False)
            if not data.empty and 'Close' in data.columns:
                close = data['Close']
                if hasattr(close, 'columns'):
                    for t in missing:
                        if t in close.columns and not close[t].dropna().empty:
                            result[t.upper()] = float(close[t].dropna().iloc[-1])
                else:
                    if not close.dropna().empty:
                        result[missing[0].upper()] = float(close.dropna().iloc[-1])
        except Exception:
            pass

        # 여전히 없는 티커 → fast_info 개별 fallback
        still_missing = [t for t in missing if t.upper() not in result]
        for t in still_missing:
            price = DataService.get_price(t)
            if price:
                result[t.upper()] = price

        return result

    @classmethod
    def inject_virtual_close(cls, data_dict, current_prices, vxn_df, fg_df=None):
        """가상 종가를 주입하여 시뮬레이션을 실시간 데이터 기반으로 수행 (VIX/VXN 포함)

        [수정] 과거 지표 보존: 가상 종가 주입 후 전체 지표를 재계산하되,
        기존 행의 지표값은 원본을 유지하고 마지막(가상) 행만 새 지표값을 적용.
        → pandas_ta의 EWM 정규화/부동소수점 누적 오차로 인한 과거 시그널 오염 방지.
        """
        cloned_dict = {}
        # [수정] 미 동부 시간(EST) 기준으로 '오늘' 날짜 계산
        now_est = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=5)
        today_ts = pd.Timestamp(now_est.date())

        # 지표 컬럼 식별용 (OHLCV + 날짜 관련은 제외)
        _OHLCV = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Dividends', 'Stock Splits'}

        # 1. 메인 티커 데이터 주입
        for ticker, df in data_dict.items():
            if ticker in current_prices and not df.empty:
                new_df = df.copy()
                new_df.index = new_df.index.normalize()
                price = current_prices[ticker]

                is_update = (new_df.index[-1] == today_ts)

                if is_update:
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

                # [수정] 과거 지표 보존: 원본 지표 백업 → 재계산 → 과거 행 복원
                # 가상 행만 새 지표값을 사용하여 과거 시그널 오염 방지
                hist_idx = new_df.index[:-1]  # 가상 행 제외한 과거 인덱스
                indicator_cols = [c for c in new_df.columns if c not in _OHLCV]
                hist_indicators = new_df.loc[hist_idx, indicator_cols].copy() if len(indicator_cols) > 0 else None

                new_df = cls.calculate_indicators(new_df)

                # 과거 행의 지표값을 원본으로 복원 (마지막 가상 행만 새 값 유지)
                if hist_indicators is not None and len(hist_idx) > 0:
                    # 재계산으로 새로 생긴 컬럼은 유지, 기존 컬럼만 복원
                    restore_cols = [c for c in indicator_cols if c in new_df.columns]
                    if restore_cols:
                        new_df.loc[hist_idx, restore_cols] = hist_indicators[restore_cols]

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
    @st.cache_data(ttl=21600)  # 6시간 캐시 (Rate Limit 방지)
    def load_all_data():
        return DataService.fetch_live_data(start_date='2008-01-01')

    @staticmethod
    def load_all_data_realtime():
        """실시간 모드: Parquet 캐시를 무시하고 yfinance에서 전량 다운로드.
        증분 병합 과정 없이 원본 데이터로 지표를 계산하여 시그널 오차를 방지."""
        return DataService.fetch_live_data(start_date='2008-01-01', skip_cache=True)
