import sys
import os
import datetime

# 프로젝트 루트 경로 추가 (core 모듈 임계값 방지)
sys.path.append(os.getcwd())

from core.news_service import NewsService

def seed_2024_historical_news():
    news_svc = NewsService()
    
    # 2024년 월별 주요 경제 뉴 데이터 (PoC용)
    historical_news = {
        "2024-01-15": [
            "US economy adds 353,000 jobs in January, smashing expectations",
            "Unemployment rate steady at 3.7%",
            "Average hourly earnings jump 0.6% monthly",
            "GDP growth reported at 3.3% for Q4 2023"
        ],
        "2024-02-13": [
            "CPI inflation tops 3.2% annual pace, higher than forecast",
            "Retail sales see month-over-month drop of 0.8%",
            "Bond yields rise as inflation cools slower than hoped",
            "Fed conveys need for patience on interest rate cuts"
        ],
        "2024-03-20": [
            "Fed holds interest rates steady at 5.25%-5.50%",
            "Fed projects three interest rate cuts by end of year",
            "US adds 303,000 jobs in March, beating expectations",
            "PCE index rises to 2.8% on year-over-year basis"
        ],
        "2024-04-15": [
            "CPI rises 3.4% year-over-year, slight deceleration",
            "Retail sales remain flat, falling short of Wall Street expectations",
            "S&P 500 and Nasdaq reach near all-time highs on rate cut hopes",
            "Core inflation cooling during April period"
        ],
        "2024-05-15": [
            "Overall consumer prices unchanged monthly, up 3.3% annually",
            "S&P 500 reaches new all-time highs above 5,300",
            "Unemployment rate ticks up slightly to 4.0%",
            "Fed Chair Powell suggests rate cuts if labor market weakens"
        ],
        "2024-06-12": [
            "Fed maintains interest rates at two-decade high",
            "US economy adds 206,000 jobs, cooling slightly",
            "Unemployment rate climbs to 4.1%",
            "S&P 500 and Nasdaq set new record highs driven by AI rally"
        ],
        "2024-07-11": [
            "CPI dips below 3% for the first time since 2021 (2.9%)",
            "Expectations grow for September rate cut",
            "US retail sales accelerate by 1%, highest in a year",
            "Job growth reported at 114,000, weakest in a year"
        ],
        "2024-08-23": [
            "Fed Chair Powell says 'the time has come' for rate cuts",
            "Annual inflation drops to 2.5%, lower than expected",
            "Job growth revised down by 818,000 for past year",
            "S&P 500 recovers from early August yen carry trade crash"
        ],
        "2024-09-18": [
            "Fed cuts interest rates by 0.50 percentage points (Jumbo Cut)",
            "First rate cut in four years after inflation cools",
            "Job market shows resilience with 254,000 positions added",
            "Indices hit new records following Fed's bold pivot"
        ],
        "2024-10-15": [
            "US economy adds only 12,000 jobs in October, missing expectations",
            "Consumer confidence sees largest decline in three years",
            "Third-quarter GDP rises by 2.8% (slightly below 3.1% forecast)",
            "Markets volatile ahead of US General Elections"
        ],
        "2024-11-06": [
            "S&P 500 and Bitcoin hit all-time highs following US Election results",
            "Fed expected to cut rates by 0.25 percentage points in November",
            "Consumer optimism shows another increase as uncertainty clears",
            "Markets rally on pro-growth policy expectations"
        ],
        "2024-12-18": [
            "Fed raises economic growth forecast for 2025",
            "S&P 500 concludes the year with strong double-digit gains",
            "US economy concludes year strongly with 256,000 jobs added",
            "Inflation (Core PCE) stabilizes near 2.5%"
        ],
        "2025-01-15": [
            "Markets await new economic policies from administration",
            "Inflation data shows sticky core prices in early 2025",
            "Nvidia and tech leaders continue to drive market indices",
            "Fed signals cautious approach to further rate cuts in 2025"
        ]
    }

    print(f"--- 1년치 역사적 경제 뉴스 정형화 시작 (총 {len(historical_news)}개 데이터) ---")
    
    for date_str, headlines in historical_news.items():
        if news_svc.check_date_exists(date_str):
            print(f"[{date_str}] 데이터가 이미 존재합니다. 건너뜁니다.")
            continue
            
        print(f"[{date_str}] Gemini 분석 중...")
        res = news_svc.score_news_with_gemini(date_str, headlines)
        if res:
            if news_svc.save_sentiment(date_str, res, raw_headlines="; ".join(headlines)):
                print(f" [OK] 저장 성공: {res['topic']} (Sentiment: {res['sentiment']})")
        else:
            print(f" [FAIL] 분석 실패: {date_str}")
            
    print("--- 1년치 뉴스 데이터 시딩(Seeding) 완료 ---")

if __name__ == "__main__":
    seed_2024_historical_news()
