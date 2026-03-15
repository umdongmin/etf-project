import os
import json
import datetime
import pandas as pd
import yfinance as yf
import feedparser
from bs4 import BeautifulSoup
import requests
from core.llm_service import LLMService

class NewsService:
    """경제 뉴스 수집 서비스 (뉴스 헤드라인 크롤링 전용)"""
    
    def __init__(self):
        self.llm = LLMService()
        # DB 관련 초기화 제거

    def fetch_headlines_from_yf(self, target_date=None, tickers=['QQQ', 'SPY', 'DIA', '^IXIC', '^TNX', 'AAPL', 'NVDA', 'MSFT', 'TSLA', 'GOOGL', 'AMZN']):
        headlines = []
        target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                news_items = getattr(t, 'news', [])
                for item in news_items:
                    pub_time = item.get('providerPublishTime')
                    if pub_time:
                        item_dt = datetime.datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d")
                        if item_dt != target_dt_str: continue
                    title = item.get('title') or item.get('headline'); headlines.append(title)
            except: continue
        return list(set([h.strip() for h in headlines if h]))

    def fetch_headlines_from_rss(self, target_date=None):
        rss_urls = ["https://www.investing.com/rss/news_25.rss", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"]
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        for url in rss_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    pub_struct = entry.get('published_parsed')
                    if pub_struct:
                        item_dt = datetime.datetime(*pub_struct[:3]).strftime("%Y-%m-%d")
                        if item_dt != target_dt_str: continue
                    headlines.append(entry.get('title', ''))
            except: continue
        return list(set(headlines))

    def fetch_headlines_from_macro_report(self, target_date=None):
        url = "https://finance.naver.com/research/market_info_list.naver"
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        naver_dt = target_dt_str[2:].replace('-', '.')
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}); resp.encoding = 'euc-kr'
            soup = BeautifulSoup(resp.text, 'html.parser')
            for row in soup.find_all('tr'):
                tds = row.find_all('td')
                if len(tds) >= 4 and tds[3].text.strip() == naver_dt: headlines.append(tds[0].text.strip())
        except: pass
        return list(set(headlines))

    def fetch_headlines_from_finviz(self, target_date=None):
        url = "https://finviz.com/news.ashx"
        headlines = []; target_dt_str = target_date if target_date else datetime.date.today().strftime("%Y-%m-%d")
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.text, 'html.parser')
            for a in soup.select('a.nn-tab-link'): headlines.append(a.text.strip())
        except: pass
        return list(set(headlines))
