"""
FOMC 회의록 감성 점수화 스크립트
==================================
data/fomc_minutes/*.txt 파일을 읽어 Gemini 2.5 Flash로
매파(+1.0) ~ 비둘기(-1.0) 감성 점수를 산출하고
data/fomc_scores.csv 에 저장.

사용법:
    python data/fomc_sentiment_scorer.py              # 미처리 파일 전체 점수화
    python data/fomc_sentiment_scorer.py --date 20220316  # 특정 날짜 재점수화
    python data/fomc_sentiment_scorer.py --verify 20220316  # 일관성 검증 (3회 실행)
    python data/fomc_sentiment_scorer.py --update     # 신규 파일만 처리

저장 구조:
    data/fomc_scores.csv
    - date: YYYYMMDD
    - score: -1.0 ~ +1.0 (매파: 양수, 비둘기: 음수)
    - reasoning: 판단 근거 3줄 요약
    - model_version: gemini 모델명
    - created_at: 분석 일시
"""

import os
import sys
import time
import argparse
import logging
import csv
import json
import re
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MINUTES_DIR = PROJECT_ROOT / "data" / "fomc_minutes"
SCORES_CSV = PROJECT_ROOT / "data" / "fomc_scores.csv"
CSV_FIELDS = ["date", "score", "reasoning", "model_version", "created_at"]

# 회의록 최대 토큰 절감을 위해 앞부분 우선 사용 (약 12,000자 ≈ 3,000 토큰)
MAX_CHARS = 12_000

MODEL_NAME = "gemini-2.5-flash"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 감성 점수화 프롬프트
# ─────────────────────────────────────────────
SCORING_PROMPT = """You are a senior Federal Reserve analyst with 20 years of experience.

Analyze the following FOMC meeting minutes and provide a hawkish/dovish sentiment score.

Scoring criteria:
- +1.0: Extremely hawkish (aggressive rate hikes, strong inflation concern, tightening bias)
- +0.5: Moderately hawkish (rate hike signaled, above-target inflation concerns)
-  0.0: Neutral (balanced risks, data-dependent stance, no clear bias)
- -0.5: Moderately dovish (rate cuts signaled, growth concerns, easing bias)
- -1.0: Extremely dovish (emergency cuts, recession fears, QE expansion)

Return your response in the following JSON format ONLY (no markdown, no extra text):
{{
  "score": <float between -1.0 and +1.0, one decimal place>,
  "reasoning": "<3-line summary in Korean explaining the key hawkish/dovish signals found>"
}}

FOMC Minutes (excerpt):
{minutes_text}
"""


def load_gemini_client():
    """Gemini 클라이언트 초기화"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    from google import genai
    client = genai.Client(api_key=api_key)
    return client


def score_minutes(client, text: str, retries: int = 3) -> dict | None:
    """
    회의록 텍스트 → 감성 점수 및 근거 반환
    Returns: {"score": float, "reasoning": str} or None
    """
    excerpt = text[:MAX_CHARS]
    prompt = SCORING_PROMPT.format(minutes_text=excerpt)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            raw = response.text.strip()

            # JSON 파싱 (마크다운 코드블록 제거)
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            result = json.loads(raw)

            score = float(result["score"])
            score = max(-1.0, min(1.0, round(score, 1)))  # 범위 클램핑
            reasoning = str(result.get("reasoning", "")).strip()

            return {"score": score, "reasoning": reasoning}

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"  [attempt {attempt+1}/{retries}] JSON 파싱 실패: {e}")
            logger.debug(f"  Raw response: {raw[:300]}")
            if attempt < retries - 1:
                time.sleep(2.0)
        except Exception as e:
            logger.warning(f"  [attempt {attempt+1}/{retries}] API 오류: {e}")
            if attempt < retries - 1:
                time.sleep(5.0)

    return None


def load_existing_scores() -> dict:
    """기존 fomc_scores.csv 로드 → {date: row_dict} 딕셔너리 반환"""
    if not SCORES_CSV.exists():
        return {}
    scores = {}
    with open(SCORES_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores[row["date"]] = row
    return scores


def save_scores(scores: dict):
    """scores 딕셔너리 전체를 fomc_scores.csv로 저장 (날짜 오름차순)"""
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(scores.values(), key=lambda x: x["date"])
    with open(SCORES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(sorted_rows)
    logger.info(f"Saved {len(sorted_rows)} records → {SCORES_CSV}")


def append_score(scores: dict, date_str: str, result: dict):
    """단일 점수 결과를 scores 딕셔너리에 추가/갱신"""
    scores[date_str] = {
        "date": date_str,
        "score": result["score"],
        "reasoning": result["reasoning"],
        "model_version": MODEL_NAME,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def run_verify(client, date_str: str, runs: int = 3):
    """특정 날짜 회의록 N회 반복 점수화 → 일관성(stddev) 검증"""
    txt_path = MINUTES_DIR / f"{date_str}.txt"
    if not txt_path.exists():
        logger.error(f"회의록 파일 없음: {txt_path}")
        return

    text = txt_path.read_text(encoding="utf-8")
    results = []
    for i in range(runs):
        logger.info(f"[{i+1}/{runs}] Scoring {date_str}...")
        r = score_minutes(client, text)
        if r:
            results.append(r)
            logger.info(f"  Score: {r['score']:+.1f} | {r['reasoning'][:60]}...")
        time.sleep(2.0)

    if len(results) < 2:
        logger.error("결과 부족으로 일관성 검증 불가")
        return

    scores_list = [r["score"] for r in results]
    mean_score = sum(scores_list) / len(scores_list)
    stddev = (sum((s - mean_score) ** 2 for s in scores_list) / len(scores_list)) ** 0.5

    print(f"\n=== 일관성 검증 결과: {date_str} ===")
    print(f"점수 목록: {scores_list}")
    print(f"평균: {mean_score:+.2f} | 표준편차: {stddev:.3f}")
    print(f"판정: {'✅ 통과 (stddev < 0.2)' if stddev < 0.2 else '❌ 실패 (stddev ≥ 0.2)'}")


def run_score_all(client, force: bool = False):
    """미처리 회의록 전체 점수화"""
    txt_files = sorted(MINUTES_DIR.glob("*.txt"))
    if not txt_files:
        logger.error(f"회의록 파일 없음: {MINUTES_DIR}")
        logger.info("먼저 fomc_crawler.py를 실행하세요.")
        return

    scores = load_existing_scores()
    processed = 0
    skipped = 0
    failed = 0

    logger.info(f"총 {len(txt_files)}개 파일 | 기존 점수: {len(scores)}개")

    for i, path in enumerate(txt_files, 1):
        date_str = path.stem  # 파일명 = 날짜 (YYYYMMDD)

        if not force and date_str in scores:
            logger.debug(f"[{i}/{len(txt_files)}] Skip (exists): {date_str}")
            skipped += 1
            continue

        text = path.read_text(encoding="utf-8")
        if len(text) < 200:
            logger.warning(f"[{i}/{len(txt_files)}] 텍스트 너무 짧음, 스킵: {date_str}")
            skipped += 1
            continue

        logger.info(f"[{i}/{len(txt_files)}] Scoring: {date_str} ({len(text):,} chars)")
        result = score_minutes(client, text)

        if result:
            append_score(scores, date_str, result)
            logger.info(f"  ✅ Score: {result['score']:+.1f} | {result['reasoning'][:80]}...")
            processed += 1
            # 10건마다 중간 저장 (중단 대비)
            if processed % 10 == 0:
                save_scores(scores)
        else:
            logger.error(f"  ❌ Failed: {date_str}")
            failed += 1

        time.sleep(1.5)  # API Rate limit

    save_scores(scores)
    logger.info(f"완료: 신규 {processed}개 | 스킵 {skipped}개 | 실패 {failed}개")
    print_summary(scores)


def run_score_single(client, date_str: str):
    """특정 날짜 단건 재점수화"""
    txt_path = MINUTES_DIR / f"{date_str}.txt"
    if not txt_path.exists():
        logger.error(f"회의록 파일 없음: {txt_path}")
        return

    text = txt_path.read_text(encoding="utf-8")
    logger.info(f"Scoring: {date_str} ({len(text):,} chars)")
    result = score_minutes(client, text)

    if result:
        scores = load_existing_scores()
        append_score(scores, date_str, result)
        save_scores(scores)
        logger.info(f"✅ Score: {result['score']:+.1f}")
        logger.info(f"Reasoning: {result['reasoning']}")
    else:
        logger.error("점수화 실패")


def print_summary(scores: dict):
    """점수 분포 요약 출력"""
    if not scores:
        return
    vals = [float(v["score"]) for v in scores.values()]
    print(f"\n=== 감성 점수 요약 ({len(vals)}개 회의) ===")
    print(f"평균: {sum(vals)/len(vals):+.2f}")
    print(f"최대(매파): {max(vals):+.1f} | 최소(비둘기): {min(vals):+.1f}")
    hawkish = sum(1 for v in vals if v > 0.3)
    neutral = sum(1 for v in vals if -0.3 <= v <= 0.3)
    dovish = sum(1 for v in vals if v < -0.3)
    print(f"매파: {hawkish}회 | 중립: {neutral}회 | 비둘기: {dovish}회")

    # 수동 검증 기준점 출력
    reference_dates = {
        "20200315": "코로나 긴급인하 (기대: -1.0)",
        "20220316": "첫 금리인상 (기대: +0.8~+1.0)",
        "20240918": "인하 전환 (기대: -0.5~-0.8)",
    }
    print("\n[수동 검증 기준점]")
    for date, label in reference_dates.items():
        if date in scores:
            s = float(scores[date]["score"])
            print(f"  {date} {label} → 실제 점수: {s:+.1f}")
        else:
            print(f"  {date} {label} → 미처리")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOMC 회의록 감성 점수화")
    parser.add_argument("--date", type=str, help="특정 날짜 재점수화 (예: 20220316)")
    parser.add_argument("--verify", type=str, help="일관성 검증 - 3회 반복 점수화 (예: 20220316)")
    parser.add_argument("--update", action="store_true", help="신규 파일만 처리 (기존 스킵)")
    parser.add_argument("--force", action="store_true", help="기존 점수 덮어쓰기")
    parser.add_argument("--summary", action="store_true", help="현재 점수 요약만 출력")
    args = parser.parse_args()

    if args.summary:
        scores = load_existing_scores()
        print_summary(scores)
        sys.exit(0)

    client = load_gemini_client()

    if args.verify:
        run_verify(client, args.verify)
    elif args.date:
        run_score_single(client, args.date)
    else:
        # 전체 또는 업데이트 (update/force 모두 run_score_all로 처리)
        run_score_all(client, force=args.force)
