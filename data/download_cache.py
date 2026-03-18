"""
parquet 캐시 전체 다운로드 스크립트
────────────────────────────────────────────────────────────────
사용법 (로컬):
    cd c:\TestCode\rebalance_program
    python scripts/download_cache.py

기능:
    - 2010-01-01 ~ 오늘까지 yfinance 전체 다운로드
    - data/cache/ 폴더에 parquet 파일로 저장
    - 이후 앱 기동 시 마지막 날짜 이후분만 yfinance 조회 (수십 행)

배포 루틴:
    1. python scripts/download_cache.py
    2. git add data/cache/
    3. git push  →  Streamlit Cloud 자동 재배포
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import datetime
import json
import pandas as pd
import yfinance as yf
import requests
import time

CACHE_DIR = ROOT / "data" / "cache"
START_DATE = "2010-01-01"
TICKERS    = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TBF', 'BIL', 'TMF', 'TMV', '^TNX']
VIX_TICKERS = {'VIX': '^VIX', 'VXN': '^VXN'}


def download_with_retry(tickers, start, end, max_retries=3, **kwargs):
    delays = [15, 45, 90]
    ticker_str = str(tickers)[:60]
    for attempt in range(max_retries):
        try:
            result = yf.download(tickers, start=start, end=end, progress=False, **kwargs)
            if result is not None and not result.empty:
                return result
            if attempt < max_retries - 1:
                print(f"  빈 데이터 반환 ({ticker_str}) — {delays[attempt]}초 후 재시도...")
                time.sleep(delays[attempt])
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(x in err for x in ['ratelimit', 'too many', '429', 'rate limited'])
            if attempt < max_retries - 1:
                wait = delays[attempt]
                reason = "Rate limit" if is_rate_limit else "오류"
                print(f"  {reason} ({ticker_str}) — {wait}초 후 재시도 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  재시도 초과 ({ticker_str}): {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    end_date = now.strftime('%Y-%m-%d')
    print(f"\n{'='*55}")
    print(f"  parquet 캐시 전체 다운로드")
    print(f"  기간: {START_DATE} ~ {end_date}")
    print(f"{'='*55}\n")

    # ── 1. 메인 티커 ──────────────────────────────────────────
    print(f"[1/3] 메인 티커 다운로드: {TICKERS}")
    full_data = download_with_retry(TICKERS, start=START_DATE, end=end_date, group_by='ticker')
    saved_tickers = []

    if not full_data.empty:
        for ticker in TICKERS:
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

                if isinstance(df_raw, pd.Series):
                    df_raw = df_raw.to_frame(name='Close')
                if isinstance(df_raw.columns, pd.MultiIndex):
                    df_raw.columns = df_raw.columns.get_level_values(0)

                ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_raw.columns]
                df_raw = df_raw[ohlcv_cols].dropna(how='all')
                if df_raw.empty:
                    continue

                safe = ticker.replace('^', '_')
                out_path = CACHE_DIR / f"raw_{safe}.parquet"
                df_raw.to_parquet(out_path)
                saved_tickers.append(ticker)
                print(f"  ✓ {ticker}: {len(df_raw)}행 → {out_path.name}")
            except Exception as e:
                print(f"  ✗ {ticker} 실패: {e}")
    else:
        print("  ✗ 메인 티커 다운로드 실패")

    # ── 2. VIX / VXN ──────────────────────────────────────────
    print(f"\n[2/3] VIX / VXN 다운로드")
    v_series = []
    for name, ticker in VIX_TICKERS.items():
        raw = download_with_retry(ticker, start=START_DATE, end=end_date)
        if not raw.empty:
            c = raw['Close']
            s = c.iloc[:, 0].copy() if isinstance(c, pd.DataFrame) else c.copy()
            s.name = name
            try:    s.index = s.index.tz_localize(None).normalize()
            except: s.index = s.index.normalize()
            v_series.append(s)
            print(f"  ✓ {name}: {len(s)}행")
        else:
            print(f"  ✗ {name} 다운로드 실패")

    if v_series:
        vxn_df = pd.concat(v_series, axis=1)
        vxn_df = vxn_df[~vxn_df.index.duplicated(keep='first')]
        vxn_df.to_parquet(CACHE_DIR / "raw_vix_vxn.parquet")
        print(f"  → raw_vix_vxn.parquet 저장 ({len(vxn_df)}행)")

    # ── 3. cache_meta.json ────────────────────────────────────
    print(f"\n[3/3] 메타 정보 저장")
    # 저장된 티커 중 마지막 날짜 계산
    last_dates = []
    for ticker in saved_tickers:
        safe = ticker.replace('^', '_')
        path = CACHE_DIR / f"raw_{safe}.parquet"
        if path.exists():
            idx = pd.read_parquet(path).index
            if len(idx):
                last_dates.append(idx.max())

    if last_dates:
        last_date_str = max(last_dates).strftime('%Y-%m-%d')
        meta = {
            'last_date': last_date_str,
            'updated_at': now.strftime('%Y-%m-%d %H:%M:%S'),
            'tickers': saved_tickers,
        }
        with open(CACHE_DIR / "cache_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  ✓ cache_meta.json — last_date: {last_date_str}")
    else:
        print("  ✗ 저장된 티커 없음 — meta 저장 스킵")

    print(f"\n완료! 저장 위치: {CACHE_DIR}")
    print(f"저장된 티커: {saved_tickers}")
    print(f"\n다음 단계:")
    print(f"  git add data/cache/")
    print(f"  git commit -m 'chore: update market data cache'")
    print(f"  git push")


if __name__ == "__main__":
    main()
