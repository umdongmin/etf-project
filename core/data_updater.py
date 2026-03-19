"""
일별 데이터 자동 업데이트 서비스
─────────────────────────────────────────────────────
Cloud Scheduler → Cloud Run → run_daily_update() 순서로 호출됨

수행 작업:
  1. OHLCV 증분 업데이트 (yfinance, 전일 종가)
  2. FRED 증분 업데이트 (3일 이상 오래됐을 때만)
  3. FOMC 신규 회의록 점수화 (신규 있을 때만)
  4. 변경된 파일 GitHub push

환경변수:
  FRED_API_KEY    — FRED API 키
  GEMINI_API_KEY  — Gemini API 키 (FOMC 점수화용)
  GITHUB_TOKEN    — GitHub Personal Access Token
  GITHUB_REPO     — 예: username/rebalance_program
  GITHUB_BRANCH   — 예: main (기본값)
"""

import os
import json
import time
import base64
import datetime
import requests
import pandas as pd
from pathlib import Path

# .env 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
DATA_DIR  = ROOT / "data"

TICKERS     = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TBF', 'BIL', 'TMF', 'TMV', '^TNX']
VIX_TICKERS = {'VIX': '^VIX', 'VXN': '^VXN'}
FRED_SERIES = [
    ('DFEDTARU', 'ff_rate',   0),
    ('CPILFESL', 'core_cpi', 30),
    ('PPIACO',   'ppi',      14),
    ('UNRATE',   'unrate',   30),
    ('NROU',     'nrou',     45),
    ('T10YIE',   'exp_inf',   0),
    ('T10Y2Y',   't10y2y',    0),
]


# ─────────────────────────────────────────────────────────────
# 1. OHLCV 증분 업데이트
# ─────────────────────────────────────────────────────────────

def _download_with_retry(tickers, start, end, max_retries=3, **kwargs):
    import yfinance as yf
    delays = [15, 45, 90]
    for attempt in range(max_retries):
        try:
            result = yf.download(tickers, start=start, end=end, progress=False, **kwargs)
            if result is not None and not result.empty:
                return result
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delays[attempt])
            else:
                print(f"  yfinance 재시도 초과: {e}")
    return pd.DataFrame()


def update_ohlcv() -> list:
    """OHLCV 증분 업데이트. 변경된 파일 경로 목록 반환."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    changed = []
    now      = datetime.datetime.now()
    end_date = now.strftime('%Y-%m-%d')

    # 캐시 메타에서 마지막 날짜 확인
    meta_path = CACHE_DIR / "cache_meta.json"
    last_date = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding='utf-8') as f:
                last_date = json.load(f).get('last_date')
        except Exception:
            pass

    start_date = '2010-01-01'
    if last_date:
        # 마지막 날짜 다음날부터 증분
        start_dt = datetime.datetime.strptime(last_date, '%Y-%m-%d') + datetime.timedelta(days=1)
        start_date = start_dt.strftime('%Y-%m-%d')
        if start_dt.date() >= datetime.date.today():
            print(f"  OHLCV 이미 최신 ({last_date}) — 스킵")
            return []

    print(f"  OHLCV 증분 다운로드: {start_date} ~ {end_date}")
    full_data = _download_with_retry(TICKERS, start=start_date, end=end_date, group_by='ticker')
    saved_tickers = []

    if not full_data.empty:
        for ticker in TICKERS:
            try:
                if isinstance(full_data.columns, pd.MultiIndex):
                    if ticker in full_data.columns.get_level_values(0):
                        df_new = full_data[ticker]
                    elif ticker in full_data.columns.get_level_values(1):
                        df_new = full_data.xs(ticker, level=1, axis=1)
                    else:
                        continue
                elif ticker in full_data:
                    df_new = full_data[ticker]
                else:
                    continue

                if isinstance(df_new, pd.Series):
                    df_new = df_new.to_frame(name='Close')
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.get_level_values(0)

                ohlcv_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_new.columns]
                df_new = df_new[ohlcv_cols].dropna(how='all')
                if df_new.empty:
                    continue

                safe     = ticker.replace('^', '_')
                out_path = CACHE_DIR / f"raw_{safe}.parquet"

                # 기존 캐시와 병합
                if out_path.exists():
                    df_old = pd.read_parquet(out_path)
                    df_merged = pd.concat([df_old, df_new])
                    df_merged = df_merged[~df_merged.index.duplicated(keep='last')].sort_index()
                else:
                    df_merged = df_new

                df_merged.to_parquet(out_path)
                saved_tickers.append(ticker)
                changed.append(out_path)
                print(f"  ✓ {ticker}: +{len(df_new)}행 추가")
            except Exception as e:
                print(f"  ✗ {ticker} 실패: {e}")

    # VIX / VXN
    v_series = []
    for name, ticker in VIX_TICKERS.items():
        raw = _download_with_retry(ticker, start=start_date, end=end_date)
        if not raw.empty:
            c = raw['Close']
            s = c.iloc[:, 0].copy() if isinstance(c, pd.DataFrame) else c.copy()
            s.name = name
            try:    s.index = s.index.tz_localize(None).normalize()
            except: s.index = s.index.normalize()
            v_series.append(s)

    if v_series:
        vxn_path = CACHE_DIR / "raw_vix_vxn.parquet"
        df_new_vxn = pd.concat(v_series, axis=1)
        if vxn_path.exists():
            df_old_vxn = pd.read_parquet(vxn_path)
            df_new_vxn = pd.concat([df_old_vxn, df_new_vxn])
            df_new_vxn = df_new_vxn[~df_new_vxn.index.duplicated(keep='last')].sort_index()
        df_new_vxn.to_parquet(vxn_path)
        changed.append(vxn_path)

    # cache_meta.json 업데이트
    if saved_tickers:
        last_dates = []
        for ticker in saved_tickers:
            p = CACHE_DIR / f"raw_{ticker.replace('^','_')}.parquet"
            if p.exists():
                idx = pd.read_parquet(p).index
                if len(idx):
                    last_dates.append(idx.max())
        if last_dates:
            last_date_str = max(last_dates).strftime('%Y-%m-%d')
            meta = {'last_date': last_date_str, 'updated_at': now.strftime('%Y-%m-%d %H:%M:%S'), 'tickers': saved_tickers}
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            changed.append(meta_path)
            print(f"  ✓ cache_meta.json 업데이트 ({last_date_str})")

    return changed


# ─────────────────────────────────────────────────────────────
# 2. FRED 증분 업데이트
# ─────────────────────────────────────────────────────────────

def update_fred() -> list:
    """FRED 증분 업데이트. 변경된 파일 경로 목록 반환."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("  FRED_API_KEY 없음 — 스킵")
        return []

    fred_path = CACHE_DIR / "macro_fred.parquet"
    macro_df  = pd.DataFrame()

    if fred_path.exists():
        try:
            macro_df  = pd.read_parquet(fred_path)
            cache_age = (datetime.date.today() - macro_df.index[-1].date()).days
            if cache_age <= 3:
                print(f"  FRED 캐시 최신 ({macro_df.index[-1].date()}) — 스킵")
                return []
        except Exception:
            pass

    fred_start = (macro_df.index[-1].date() - datetime.timedelta(days=7)).strftime('%Y-%m-%d') \
        if not macro_df.empty else '2010-01-01'
    end_date   = datetime.date.today().strftime('%Y-%m-%d')

    print(f"  FRED 증분 호출: {fred_start} 이후")
    macro_data_list = []
    for sid, col, lag in FRED_SERIES:
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={sid}&api_key={api_key}"
                   f"&file_type=json&observation_start={fred_start}&observation_end={end_date}")
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                obs  = r.json().get('observations', [])
                temp = pd.DataFrame(obs)
                if not temp.empty:
                    temp['date']  = pd.to_datetime(temp['date'])
                    temp['value'] = pd.to_numeric(temp['value'], errors='coerce')
                    temp = temp.set_index('date')[['value']].rename(columns={'value': col})
                    if lag > 0:
                        temp.index = temp.index + datetime.timedelta(days=lag)
                    macro_data_list.append(temp)
        except Exception as e:
            print(f"  FRED Fetch Error ({sid}): {e}")

    if not macro_data_list:
        return []

    new_macro = pd.concat(macro_data_list, axis=1)
    if not macro_df.empty:
        macro_df = pd.concat([macro_df, new_macro])
        macro_df = macro_df[~macro_df.index.duplicated(keep='last')].sort_index()
    else:
        macro_df = new_macro

    now_kst  = pd.Timestamp.now(tz='UTC').tz_convert('Asia/Seoul').normalize().tz_localize(None)
    full_idx = pd.date_range(start=macro_df.index.min(), end=now_kst, freq='D')
    macro_df = macro_df.reindex(full_idx).ffill().bfill()
    macro_df.to_parquet(fred_path)
    print(f"  ✓ macro_fred.parquet 업데이트 ({len(macro_df)}행)")
    return [fred_path]


# ─────────────────────────────────────────────────────────────
# 3. FOMC 신규 회의록 점수화
# ─────────────────────────────────────────────────────────────

def update_fomc() -> list:
    """FOMC 신규 회의록 크롤링 + 감성 점수화. 변경된 파일 경로 목록 반환."""
    scores_path  = DATA_DIR / "fomc_scores.csv"
    minutes_dir  = DATA_DIR / "fomc_minutes"
    minutes_dir.mkdir(parents=True, exist_ok=True)

    # 기존 점수 로드 → 마지막 처리 날짜 확인
    existing = {}
    if scores_path.exists():
        import csv
        with open(scores_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                existing[row['date']] = row
    last_scored = max(existing.keys()) if existing else '20100101'
    print(f"  FOMC 마지막 점수화 날짜: {last_scored}")

    # 신규 회의록 링크 수집
    import re
    from bs4 import BeautifulSoup
    FED_BASE     = "https://www.federalreserve.gov"
    CALENDAR_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"
    HEADERS      = {"User-Agent": "Mozilla/5.0"}

    try:
        r    = requests.get(CALENDAR_URL, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        new_links = []
        for a in soup.find_all("a", href=True):
            m = re.search(r"fomcminutes(\d{8})\.htm", a["href"])
            if m:
                date_str = m.group(1)
                if date_str > last_scored:
                    new_links.append({"date": date_str, "url": FED_BASE + a["href"]})
    except Exception as e:
        print(f"  FOMC 캘린더 수집 실패: {e}")
        return []

    if not new_links:
        print("  신규 FOMC 회의록 없음 — 스킵")
        return []

    print(f"  신규 FOMC 회의록 {len(new_links)}개 발견")

    # Gemini 클라이언트
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("  GEMINI_API_KEY 없음 — 스킵")
        return []

    from google import genai
    client    = genai.Client(api_key=api_key)
    MODEL     = "gemini-2.5-flash"
    PROMPT    = """You are a senior Federal Reserve analyst. Analyze FOMC meeting minutes and return JSON only:
{{"score": <float -1.0 to +1.0>, "reasoning": "<3-line Korean summary>"}}
Scoring: +1.0=extremely hawkish, 0.0=neutral, -1.0=extremely dovish
FOMC Minutes:\n{text}"""
    MAX_CHARS = 12000

    for item in new_links:
        date_str = item['date']
        try:
            # 회의록 다운로드
            r    = requests.get(item['url'], headers=HEADERS, timeout=30)
            soup = BeautifulSoup(r.text, "html.parser")
            article = soup.find("div", id="article") or soup.find("body")
            text    = article.get_text(separator="\n", strip=True) if article else ""
            if len(text) < 500:
                print(f"  ⚠️ {date_str} 텍스트 너무 짧음 — 스킵")
                continue

            # 회의록 저장
            (minutes_dir / f"{date_str}.txt").write_text(text, encoding='utf-8')

            # Gemini 점수화
            response = client.models.generate_content(model=MODEL, contents=PROMPT.format(text=text[:MAX_CHARS]))
            raw      = response.text.strip().strip('`').replace('```json','').replace('```','').strip()
            result   = json.loads(raw)
            score    = max(-1.0, min(1.0, round(float(result['score']), 1)))

            existing[date_str] = {
                'date': date_str, 'score': score,
                'reasoning': result.get('reasoning', ''),
                'model_version': MODEL,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            print(f"  ✓ {date_str} 점수화 완료: {score:+.1f}")
            time.sleep(2.0)
        except Exception as e:
            print(f"  ✗ {date_str} 실패: {e}")

    # fomc_scores.csv 저장
    import csv
    sorted_rows = sorted(existing.values(), key=lambda x: x['date'])
    with open(scores_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date','score','reasoning','model_version','created_at'])
        writer.writeheader()
        writer.writerows(sorted_rows)
    print(f"  ✓ fomc_scores.csv 업데이트 ({len(sorted_rows)}건)")
    return [scores_path]


# ─────────────────────────────────────────────────────────────
# 4. GitHub push
# ─────────────────────────────────────────────────────────────

def push_to_github(changed_files: list) -> bool:
    """변경된 파일을 GitHub API로 push. 성공 여부 반환."""
    token  = os.getenv("GITHUB_TOKEN")
    repo   = os.getenv("GITHUB_REPO")
    branch = os.getenv("GITHUB_BRANCH", "main")

    if not token or not repo:
        print("  GITHUB_TOKEN 또는 GITHUB_REPO 없음 — GitHub push 스킵")
        return False

    headers    = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    api_base   = f"https://api.github.com/repos/{repo}/contents"
    pushed     = 0
    today_str  = datetime.date.today().strftime('%Y-%m-%d')

    for file_path in changed_files:
        file_path = Path(file_path)
        if not file_path.exists():
            continue

        # 레포 내 상대 경로
        try:
            rel_path = file_path.relative_to(ROOT).as_posix()
        except ValueError:
            continue

        try:
            content_b64 = base64.b64encode(file_path.read_bytes()).decode()

            # 현재 파일 SHA 조회 (업데이트 시 필요)
            r   = requests.get(f"{api_base}/{rel_path}?ref={branch}", headers=headers, timeout=10)
            sha = r.json().get('sha') if r.status_code == 200 else None

            payload = {
                "message": f"chore: update cache {today_str}",
                "content": content_b64,
                "branch":  branch,
            }
            if sha:
                payload["sha"] = sha

            r = requests.put(f"{api_base}/{rel_path}", headers=headers, json=payload, timeout=30)
            if r.status_code in (200, 201):
                print(f"  ✓ GitHub push: {rel_path}")
                pushed += 1
            else:
                print(f"  ✗ GitHub push 실패 ({rel_path}): {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"  ✗ GitHub push 오류 ({rel_path}): {e}")

    print(f"  GitHub push 완료: {pushed}/{len(changed_files)}개")
    return pushed > 0


# ─────────────────────────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────────────────────────

def run_daily_update():
    """일별 데이터 업데이트 전체 실행."""
    print("\n📦 [Daily Update] 시작", flush=True)
    changed = []

    print("\n[1/3] OHLCV 증분 업데이트", flush=True)
    changed += update_ohlcv()

    print("\n[2/3] FRED 증분 업데이트", flush=True)
    changed += update_fred()

    print("\n[3/3] FOMC 신규 점수화", flush=True)
    changed += update_fomc()

    if changed:
        print(f"\n[GitHub] 변경 파일 {len(changed)}개 push 중...", flush=True)
        push_to_github(changed)
    else:
        print("\n[GitHub] 변경 없음 — push 스킵", flush=True)

    print("\n✅ [Daily Update] 완료", flush=True)
