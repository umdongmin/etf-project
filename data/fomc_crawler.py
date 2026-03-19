"""
FOMC 회의록 크롤러
==================
Fed 홈페이지에서 2010~현재 FOMC 회의록 텍스트를 수집하여
data/fomc_minutes/ 디렉토리에 날짜별 텍스트 파일로 저장.

사용법:
    python data/fomc_crawler.py              # 전체 수집 (2010~현재)
    python data/fomc_crawler.py --year 2024  # 특정 연도만
    python data/fomc_crawler.py --update     # 신규 회의록만 추가

저장 구조:
    data/fomc_minutes/
    ├── 20100127.txt
    ├── 20100316.txt
    └── ...
"""

import os
import sys
import time
import argparse
import logging
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MINUTES_DIR = PROJECT_ROOT / "data" / "fomc_minutes"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FED_BASE = "https://www.federalreserve.gov"
CALENDAR_URL = f"{FED_BASE}/monetarypolicy/fomccalendars.htm"

# 연도별 과거 페이지 URL 패턴
# fomchistoricalYYYY.htm 은 2020년까지만 존재, 2021~현재는 캘린더 페이지에서 수집
HISTORICAL_URL_PATTERN = f"{FED_BASE}/monetarypolicy/fomchistorical{{year}}.htm"
HISTORICAL_MAX_YEAR = 2020  # 이 연도까지만 historical 페이지 존재


def fetch_html(url: str, retries: int = 3, delay: float = 2.0) -> str | None:
    """HTML 페이지 가져오기 (재시도 포함)"""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"[attempt {attempt+1}/{retries}] {url} → {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    logger.error(f"Failed to fetch: {url}")
    return None


def parse_minutes_links_from_calendar(html: str) -> list[dict]:
    """
    현재 캘린더 페이지 (fomccalendars.htm) 파싱
    → {date: 'YYYYMMDD', url: '/monetarypolicy/fomcminutes...htm'} 목록 반환
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        # 회의록 링크 패턴: /monetarypolicy/fomcminutes20XXXXXX.htm
        match = re.search(r"fomcminutes(\d{8})\.htm", href)
        if match:
            date_str = match.group(1)
            full_url = FED_BASE + href if href.startswith("/") else href
            results.append({"date": date_str, "url": full_url})

    return results


def parse_minutes_links_from_historical(html: str, year: int) -> list[dict]:
    """
    과거 연도별 페이지 파싱 (fomchistoricalYYYY.htm)
    → {date: 'YYYYMMDD', url: '...'} 목록 반환
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        match = re.search(r"fomcminutes(\d{8})\.htm", href)
        if match:
            date_str = match.group(1)
            if date_str.startswith(str(year)):
                full_url = FED_BASE + href if href.startswith("/") else href
                results.append({"date": date_str, "url": full_url})

    return results


def extract_minutes_text(html: str) -> str:
    """
    회의록 HTML에서 본문 텍스트만 추출
    - <div id="article"> 또는 <div class="col-xs-12 col-sm-8 col-md-8"> 영역 우선
    - 없으면 <body> 전체 텍스트
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1순위: id="article" (최신 형식)
    article = soup.find("div", id="article")
    if article:
        return article.get_text(separator="\n", strip=True)

    # 2순위: 주요 컨텐츠 div (구형식)
    for cls in ["col-xs-12 col-sm-8 col-md-8", "content"]:
        div = soup.find("div", class_=cls)
        if div:
            return div.get_text(separator="\n", strip=True)

    # 3순위: body 전체
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def collect_all_links(start_year: int = 2010) -> list[dict]:
    """
    전체 회의록 링크 수집
    - 과거(start_year ~ 2020): 연도별 historical 페이지 (fomchistoricalYYYY.htm)
    - 최근(2021~현재): 메인 캘린더 페이지 (fomccalendars.htm) — 다년도 통합 제공
    """
    current_year = datetime.now().year
    all_links = []
    seen_dates = set()

    # 과거 연도별 페이지 순회 (2020년까지만 historical 페이지 존재)
    hist_end = min(HISTORICAL_MAX_YEAR + 1, current_year)
    for year in range(start_year, hist_end):
        url = HISTORICAL_URL_PATTERN.format(year=year)
        logger.info(f"Fetching historical page: {year}")
        html = fetch_html(url)
        if not html:
            logger.warning(f"  → {year}년 historical 페이지 없음, 스킵")
            continue
        links = parse_minutes_links_from_historical(html, year)
        for lk in links:
            if lk["date"] not in seen_dates:
                all_links.append(lk)
                seen_dates.add(lk["date"])
        time.sleep(1.0)

    # 2021~현재: 메인 캘린더 페이지 (다년도 링크 포함)
    logger.info("Fetching calendar page (2021~현재 포함)")
    html = fetch_html(CALENDAR_URL)
    if html:
        links = parse_minutes_links_from_calendar(html)
        for lk in links:
            if lk["date"] not in seen_dates:
                all_links.append(lk)
                seen_dates.add(lk["date"])

    all_links.sort(key=lambda x: x["date"])
    logger.info(f"Total minutes links found: {len(all_links)}")
    return all_links


def download_minutes(links: list[dict], skip_existing: bool = True) -> int:
    """
    회의록 텍스트 다운로드 및 저장
    Returns: 신규 저장된 파일 수
    """
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, item in enumerate(links, 1):
        date_str = item["date"]
        url = item["url"]
        out_path = MINUTES_DIR / f"{date_str}.txt"

        if skip_existing and out_path.exists():
            logger.debug(f"[{i}/{len(links)}] Skip (exists): {date_str}")
            continue

        logger.info(f"[{i}/{len(links)}] Downloading: {date_str} → {url}")
        html = fetch_html(url)
        if not html:
            logger.warning(f"  ❌ Failed: {date_str}")
            continue

        text = extract_minutes_text(html)
        if len(text) < 500:
            logger.warning(f"  ⚠️  Text too short ({len(text)} chars): {date_str}")

        out_path.write_text(text, encoding="utf-8")
        logger.info(f"  ✅ Saved: {out_path.name} ({len(text):,} chars)")
        saved += 1
        time.sleep(1.5)  # Rate limit 준수

    return saved


def run(args):
    """메인 실행 로직"""
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)

    if args.update:
        # 신규 회의록만 추가 (skip_existing=True 동일하지만 로그 명시)
        logger.info("=== UPDATE MODE: 신규 회의록만 수집 ===")
        links = collect_all_links(start_year=2010)
        saved = download_minutes(links, skip_existing=True)
        logger.info(f"업데이트 완료: {saved}개 신규 저장")

    elif args.year:
        # 특정 연도만
        year = args.year
        logger.info(f"=== YEAR MODE: {year}년 수집 ===")
        current_year = datetime.now().year
        if year == current_year:
            html = fetch_html(CALENDAR_URL)
            links = parse_minutes_links_from_calendar(html) if html else []
        else:
            url = HISTORICAL_URL_PATTERN.format(year=year)
            html = fetch_html(url)
            links = parse_minutes_links_from_historical(html, year) if html else []
        logger.info(f"{year}년 회의록 링크: {len(links)}개")
        saved = download_minutes(links, skip_existing=not args.force)
        logger.info(f"완료: {saved}개 저장")

    else:
        # 전체 수집 (2010~현재)
        start_year = args.start_year or 2010
        logger.info(f"=== FULL MODE: {start_year}~현재 전체 수집 ===")
        links = collect_all_links(start_year=start_year)
        saved = download_minutes(links, skip_existing=not args.force)
        logger.info(f"전체 완료: {saved}개 신규 저장 / 총 {len(links)}개 링크")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOMC 회의록 크롤러")
    parser.add_argument("--year", type=int, help="특정 연도만 수집 (예: 2024)")
    parser.add_argument("--start-year", type=int, default=2010, help="시작 연도 (기본: 2010)")
    parser.add_argument("--update", action="store_true", help="신규 회의록만 추가 수집")
    parser.add_argument("--force", action="store_true", help="기존 파일 덮어쓰기")
    args = parser.parse_args()
    run(args)
