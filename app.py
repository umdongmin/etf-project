import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import yfinance as yf
import requests
import datetime
from backtest import calculate_metrics

# Streamlit 설정은 가장 처음에 호출되어야 합니다 (데코레이터보다 먼저)
st.set_page_config(page_title="ETF Golden Strategy", page_icon="📈", layout="wide")

# --- Core Logic Classes ---

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
                df = pd.concat([df, macd], axis=1)
        
        # 볼린저 밴드 추가 (20일, 2표준편차)
        bbands = ta.bbands(close_prices, length=20, std=2)
        if bbands is not None:
            # 컬럼명 유연하게 매칭 (BBL_*, BBU_*)
            l_col = [c for c in bbands.columns if c.startswith('BBL_')]
            u_col = [c for c in bbands.columns if c.startswith('BBU_')]
            if l_col and u_col:
                df['BB_Lower'] = bbands[l_col[0]]
                df['BB_Upper'] = bbands[u_col[0]]
        return df

    @classmethod
    def fetch_live_data(cls):
        print("라이브 데이터 수집 시작...")
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        start_date = '2019-01-01'
        
        data_dict = {}
        tickers = ['QQQ', 'QLD', 'TQQQ', 'TLT', 'TMF']
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            if not df.empty:
                df = cls.calculate_indicators(df)
                data_dict[ticker] = df
                
        vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df = vix_df.loc[:, ~vix_df.columns.duplicated()]
        
        fg_df = pd.DataFrame()
        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            r = requests.get(url, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            series = data.get('fear_and_greed_historical', {}).get('data', [])
            fg_df = pd.DataFrame(series)
            fg_df['x'] = pd.to_datetime(fg_df['x'], unit='ms')
            fg_df.rename(columns={'x': 'Date', 'y': 'FearGreed'}, inplace=True)
            fg_df.set_index('Date', inplace=True)
        except Exception as e:
            print(f"F&G Fetch Error: {e}")
            fg_df = pd.DataFrame(columns=['FearGreed'])

        macro_data_map = {}
        fetch_status = {
            'ETF': 'Success', 'VIX': 'Pending', 'FearGreed': 'Pending', 
            'US10Y': 'Pending', 'US03M': 'Pending', 'PCCR': 'N/A'
        }
        m_tickers = {'^TNX': 'US10Y', '^IRX': 'US03M', '^PCC': 'PCCR'}
        
        for ticker, col_name in m_tickers.items():
            try:
                m_df = yf.download(ticker, period='5d', progress=False)
                if not m_df.empty:
                    if isinstance(m_df.columns, pd.MultiIndex): m_df.columns = m_df.columns.get_level_values(0)
                    target_col = next((c for c in ['Close', 'Adj Close', 'Price', 'Last'] if c in m_df.columns), None)
                    if target_col:
                        val = m_df[target_col].dropna().iloc[-1]
                        macro_data_map[col_name] = val
                        fetch_status[col_name] = 'Success'
            except: fetch_status[col_name] = 'Error'
        
        if not vix_df.empty and 'Close' in vix_df.columns:
            macro_data_map['VIX'] = vix_df['Close'].dropna().iloc[-1]
            fetch_status['VIX'] = 'Success'
        
        if not fg_df.empty: fetch_status['FearGreed'] = 'Success'
        
        macro_df = pd.DataFrame([macro_data_map])
        fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_all_data():
        return DataService.fetch_live_data()

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @staticmethod
    def run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_date, end_date, params=None):
        if base_asset not in data_dict: return pd.DataFrame()
        qqq = data_dict[base_asset]
        combined = qqq[['Close', 'RSI']].copy()
        
        fg_clean = fg_df[~fg_df.index.duplicated(keep='first')]
        vix_clean = vix_df[~vix_df.index.duplicated(keep='first')]
        combined = combined.join(fg_clean['FearGreed'].rename('FG'), how='left')
        combined = combined.join(vix_clean['Close'].rename('VIX'), how='left')
        
        macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
        signal_col = [c for c in combined.columns if 'MACDs_' in c]
        if macd_col and signal_col:
            combined['MACD'], combined['Signal_Line'] = combined[macd_col[0]], combined[signal_col[0]]
        else:
            macd = ta.macd(combined['Close'])
            combined = pd.concat([combined, macd], axis=1)
            combined['MACD'], combined['Signal_Line'] = combined['MACD_12_26_9'], combined['MACDs_12_26_9']
        
        combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
        for col in ['RSI', 'RSI_SMA', 'MACD', 'Signal_Line']:
            combined[f'Prev_{col}'] = combined[col].shift(1)
        
        combined['FG'] = combined['FG'].ffill().fillna(50)
        combined['VIX'] = combined['VIX'].ffill().fillna(15)
        combined = combined.fillna(0)
        
        dates = combined.index
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame()

        portfolio_value, cash = 10000.0, 10000.0 * cash_ratio
        current_planned_asset, rebalance_stage, last_entry_price = base_asset, 0, 0
        holdings = {t: 0.0 for t in data_dict.keys()}
        holdings[base_asset] = (portfolio_value * (1 - cash_ratio)) / qqq.loc[dates[0], 'Close']
        history = []
        
        for date in dates:
            row = combined.loc[date]
            price = row['Close']
            rsi, fg, vix, macd_val, signal_line, rsi_sma = row['RSI'], row['FG'], row['VIX'], row['MACD'], row['Signal_Line'], row['RSI_SMA']
            bb_l, bb_u = row.get('BB_Lower', 0), row.get('BB_Upper', 0)
            prev_rsi, prev_rsi_sma, prev_macd, prev_signal_line = row['Prev_RSI'], row['Prev_RSI_SMA'], row['Prev_MACD'], row['Prev_Signal_Line']
            
            if params:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                is_macd_golden_cross = (prev_macd < prev_signal_line) and (macd_val > signal_line)
                is_macd_dead_cross = (prev_macd > prev_signal_line) and (macd_val < signal_line)
                
                is_buy_signal = False
                for bp in params['buy_signals']:
                    cond, active = True, False
                    if bp['rsi_val'] > 0: cond &= (rsi < bp['rsi_val']); active = True
                    if bp['rsi_cross']: cond &= is_rsi_golden_cross; active = True
                    if bp['rsi_inc']: cond &= (rsi > prev_rsi); active = True
                    if bp['macd_inc']: cond &= (macd_val > prev_macd); active = True
                    if bp['macd_signal_below']: cond &= (macd_val < signal_line); active = True
                    if bp['macd_golden']: cond &= is_macd_golden_cross; active = True
                    if bp['bb_lower']: cond &= (price <= bb_l); active = True
                    if active and cond: is_buy_signal = True; break
                
                is_sell_signal = False
                for sp in params['sell_signals']:
                    cond, active = True, False
                    if sp['rsi_val'] > 0: cond &= (rsi >= sp['rsi_val']); active = True
                    if sp['rsi_dead']: cond &= is_rsi_dead_cross; active = True
                    if sp['rsi_dec']: cond &= (rsi < prev_rsi); active = True
                    if sp['macd_dec']: cond &= (macd_val < prev_macd); active = True
                    if sp['macd_signal_above']: cond &= (macd_val > signal_line); active = True
                    if sp['macd_dead']: cond &= is_macd_dead_cross; active = True
                    if sp['bb_upper']: cond &= (price >= bb_u); active = True
                    if active and cond: is_sell_signal = True; break
                
                b_up, b_down, s_up, s_down = params['buy_reb_up'], params['buy_reb_down'], params['sell_reb_up'], params['sell_reb_down']
            else:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_buy_signal = (is_rsi_golden_cross and (macd_val > prev_macd) and (macd_val < signal_line)) or (rsi < 35)
                sell_cond_1 = (rsi >= 70) and (rsi < prev_rsi)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                sell_cond_2 = (macd_val > signal_line) and (macd_val < prev_macd) and is_rsi_dead_cross
                sell_cond_3 = (prev_macd > prev_signal_line) and (macd_val < signal_line)
                is_sell_signal = sell_cond_1 or sell_cond_2 or sell_cond_3
                b_up, b_down, s_up, s_down = 0.03, -0.03, 0.03, -0.03

            prices = {t: data_dict[t].loc[date, 'Close'] for t in data_dict.keys()}
            current_total_val = cash + sum(holdings[t] * prices[t] for t in holdings)
            rebalance_needed = False
            
            etf_funds = current_total_val * (1 - cash_ratio)
            current_lev_pct = (holdings[leverage_asset] * prices[leverage_asset]) / etf_funds if etf_funds > 0 else 0
            
            if is_buy_signal and current_planned_asset != leverage_asset:
                current_planned_asset, rebalance_needed = leverage_asset, True
                if current_lev_pct < 0.25: target_lev_pct, rebalance_stage = 0.30, 1
                elif current_lev_pct < 0.65: target_lev_pct, rebalance_stage = 0.70, 2
                else: target_lev_pct, rebalance_stage = 1.0, 3
                last_entry_price = prices[base_asset]
            elif is_sell_signal and current_planned_asset != base_asset:
                current_planned_asset, rebalance_needed = base_asset, True
                if current_lev_pct > 0.75: target_lev_pct, rebalance_stage = 0.70, 1
                elif current_lev_pct > 0.35: target_lev_pct, rebalance_stage = 0.30, 2
                else: target_lev_pct, rebalance_stage = 0.0, 3
                last_entry_price = prices[base_asset]
            elif rebalance_stage in [1, 2]:
                price_change = prices[base_asset] / last_entry_price
                # 리벨런싱 트리거 판단: 매수 상태일 때는 b_up/b_down, 매도 상태일 때는 s_up/s_down
                trigger = (price_change >= 1 + b_up or price_change <= 1 + b_down) if current_planned_asset == leverage_asset else (price_change >= 1 + s_up or price_change <= 1 + s_down)
                
                # [수정] 신호 유지 조건 제거: 단계 전이 시 신호가 반드시 유지되어야 하는 제약을 없앰 (사용자 전략 변경 반영을 위해)
                if trigger:
                    rebalance_stage += 1
                    target_lev_pct = (0.70 if rebalance_stage == 2 else 1.0) if current_planned_asset == leverage_asset else (0.30 if rebalance_stage == 2 else 0.0)
                    last_entry_price, rebalance_needed = prices[base_asset], True

            if rebalance_needed:
                cash = current_total_val * cash_ratio
                etf_funds = current_total_val * (1 - cash_ratio)
                for t in holdings: holdings[t] = 0
                holdings[leverage_asset] = (etf_funds * target_lev_pct) / prices[leverage_asset]
                holdings[base_asset] = (etf_funds * (1 - target_lev_pct)) / prices[base_asset]

            curr_w = (0.30 if rebalance_stage == 1 else (0.70 if rebalance_stage == 2 else 1.0)) if current_planned_asset == leverage_asset else (0.70 if rebalance_stage == 1 else (0.30 if rebalance_stage == 2 else 0.0))
            history.append({
                'Date': date, 'Value': current_total_val, 'Signal': '매수' if rebalance_needed and current_planned_asset == leverage_asset else ('매도' if rebalance_needed else '중립'),
                'Asset': f"{current_planned_asset}(S{rebalance_stage})" if rebalance_stage < 3 else current_planned_asset,
                'Lev_Weight': curr_w, f'{base_asset}_Weight': 1 - curr_w, 'RSI': rsi, 'FG': fg, 'VIX': vix, 'MACD': macd_val, 'Signal_Line': signal_line
            })
        return pd.DataFrame(history).set_index('Date')

    @staticmethod
    def run_benchmark(data_dict, ticker, start_date, end_date):
        if ticker not in data_dict: return pd.DataFrame()
        df = data_dict[ticker]
        dates = df.index
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame()
        shares = 10000.0 / df.loc[dates[0], 'Close']
        history = [{'Date': date, 'Value': shares * df.loc[date, 'Close']} for date in dates]
        return pd.DataFrame(history).set_index('Date')

# --- View Classes ---

class MarketView:
    """주요 마켓 이슈 화면 UI 구성 클래스"""
    @staticmethod
    def render(data_dict, macro_df, fetch_status, fetch_time, fg_df):
        st.header("📰 주요 마켓 이슈 및 밸류어이션")
        st.caption(f"최종 데이터 업데이트: {fetch_time}")
        
        st.subheader("📊 시장 밸류에이션 및 심리")
        latest_fg = fg_df['FearGreed'].iloc[-1] if not fg_df.empty else 50
        
        v1, v2, v3 = st.columns(3)
        if 'QQQ' in data_dict and not data_dict['QQQ'].empty:
            latest_qqq = data_dict['QQQ']['Close'].iloc[-1]
            v1.metric("나스닥 100 PER", f"{(latest_qqq / 15.64):.1f}x", delta="평균 24~26x 대비 높음", delta_color="inverse")
            v2.metric("버핏 지수 (추정)", f"{(223.6 * (latest_qqq / 520.0)):.1f}%", delta="200% 이상 과열", delta_color="inverse")
        else:
            v1.metric("나스닥 100 PER", "N/A")
            v2.metric("버핏 지수 (추정)", "N/A")
        v3.metric("Fear & Greed Index", f"{latest_fg:.0f}", delta="Sentiment", delta_color="off")
        
        st.divider()
        st.subheader("💡 핵심 체크 지표 (Market Radar)")
        m1, m2, m3 = st.columns(3)
        l10y = macro_df.get('US10Y', pd.Series([0.0])).iloc[-1]
        if l10y > 15: l10y /= 10.0
        m1.metric("미 국채 10년물 금리", f"{l10y:.2f}%")
        m2.metric("VIX 공포 지수", f"{macro_df.get('VIX', pd.Series([0.0])).iloc[-1]:.2f}")
        m3.metric("풋/콜 비율 (PCCR)", f"{macro_df.get('PCCR', pd.Series([0.46])).iloc[-1]:.2f}")
        
        st.divider()
        # 3. 주요 일정 캘린더
        st.subheader("📅 주요 경제 일정")
        today = datetime.date.today()

        fomc_events = [
            {"date": datetime.date(2025, 12, 10), "actual": "5.50%", "consensus": "5.50%"},
            {"date": datetime.date(2026, 1, 29), "actual": None, "consensus": "5.50% (동결)"},
            {"date": datetime.date(2026, 3, 19), "actual": None, "consensus": "5.25% (인하)"},
        ]
        cpi_events = [
            {"date": datetime.date(2025, 12, 12), "actual": "3.1%", "consensus": "3.1%"},
            {"date": datetime.date(2026, 1, 13), "actual": "3.1%", "consensus": "3.0%"},
            {"date": datetime.date(2026, 2, 12), "actual": None, "consensus": "2.9%"},
        ]

        def get_past_and_next(events, today):
            past = [e for e in events if e['date'] < today]
            future = [e for e in events if e['date'] >= today]
            return (past[-1] if past else None), (future[0] if future else None)

        col_fomc, col_cpi = st.columns(2)
        with col_fomc:
            st.markdown("**🏦 FOMC 금리 결정**")
            p, n = get_past_and_next(fomc_events, today)
            if p: st.info(f"◀ 직전 ({p['date']}): {p['actual']}")
            if n: st.success(f"▶ 예정 ({n['date']}): {n['consensus']}")

        with col_cpi:
            st.markdown("**📈 CPI 소비자물가**")
            p, n = get_past_and_next(cpi_events, today)
            if p: st.info(f"◀ 직전 ({p['date']}): {p['actual']}")
            if n: st.success(f"▶ 예정 ({n['date']}): {n['consensus']}")

        with st.expander("💻 빅테크 주요 실적발표", expanded=True):
            st.markdown("- **1월 5주차 (1/26~1/30)** : MSFT, TSLA, AAPL 등")
            st.markdown("- **4월 4주차** : 1분기 실적 발표 시즌")

        st.divider()
        with st.expander("🛠️ 데이터 수집 상태 정보", expanded=False):
            st.table(pd.DataFrame([fetch_status]).T)

class BacktestView:
    """백테스트 결과 표시 UI 구성 클래스"""
    @staticmethod
    def render_results(golden_history, bh_histories, base_asset, leverage_asset):
        latest = golden_history.iloc[-1]
        st.subheader(f"📍 상태 요약 ({latest.name.date()})")
        
        sig_color = "green" if latest['Signal'] == "매수" else ("red" if latest['Signal'] == "매도" else "gray")
        st.markdown(f'<div style="background-color:{sig_color}; padding:20px; border-radius:10px; text-align:center; color:white;">'
                    f'<h2 style="margin:0;">{latest["Signal"]} 상태</h2>'
                    f'<b>{leverage_asset} {latest["Lev_Weight"]:.0%}, {base_asset} {latest[f"{base_asset}_Weight"]:.0%}</b></div>', unsafe_allow_html=True)
        
        metrics = {k: calculate_metrics(v.rename(columns={'Value': 'PortfolioValue'})) for k, v in {**bh_histories, 'Strategy': golden_history}.items()}
        c1, c2, c3 = st.columns(3)
        c1.metric("누적 수익률", f"{metrics['Strategy']['Cumulative Return']:.1%}")
        c2.metric("CAGR", f"{metrics['Strategy']['CAGR']:.1%}")
        c3.metric("MDD", f"{metrics['Strategy']['MDD']:.1%}")
        
        st.subheader("성과 비교 차트")
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, v in bh_histories.items(): ax.plot(v['Value']/10000, label=k, alpha=0.5)
        ax.plot(golden_history['Value']/10000, label='Strategy', linewidth=2, color='blue')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader("전략 성과 요약")
        sum_df = pd.DataFrame([{"전략": k, "최종가치": f"{v['Final Value']:,.0f}", "수익률": f"{v['Cumulative Return']:.2%}", "CAGR": f"{v['CAGR']:.2%}", "MDD": f"{v['MDD']:.2%}"} for k, v in metrics.items()])
        st.table(sum_df.set_index("전략"))

class TesterView:
    """전략 분석기 입력 UI 구성 클래스"""
    @staticmethod
    def render_config():
        col_buy, col_sell = st.columns(2)
        buy_signals = []
        with col_buy:
            st.subheader("📥 Buy Conditions (OR)")
            for i in range(1, 3):
                with st.expander(f"매수 신호 {i}", expanded=(i==1)):
                    def_rsi = 35 if i == 1 else 0
                    def_rsi_cross = True if i == 2 else False
                    def_macd_inc = True if i == 2 else False
                    def_macd_below = True if i == 2 else False
                    buy_signals.append({
                        'rsi_val': st.number_input(f"RSI 매수 기준 {i}", 0, 100, def_rsi, key=f"b_rsi_{i}"),
                        'rsi_cross': st.checkbox(f"RSI Golden Cross {i}", value=def_rsi_cross, key=f"b_cross_{i}"),
                        'rsi_inc': st.checkbox(f"RSI 증가 {i}", value=False, key=f"b_inc_{i}"),
                        'macd_inc': st.checkbox(f"MACD 증가 {i}", value=def_macd_inc, key=f"b_m_inc_{i}"),
                        'macd_signal_below': st.checkbox(f"MACD 시그널 하단 {i}", value=def_macd_below, key=f"b_m_bel_{i}"),
                        'macd_golden': st.checkbox(f"MACD 골드크로스 {i}", value=False, key=f"b_m_gold_{i}"),
                        'bb_lower': st.checkbox(f"볼린저 하단 터치 {i}", value=False, key=f"b_bb_l_{i}")
                    })
        sell_signals = []
        with col_sell:
            st.subheader("📤 Sell Conditions (OR)")
            for i in range(1, 4):
                with st.expander(f"매도 신호 {i}", expanded=(i==1)):
                    def_rsi = 70 if i == 1 else 0
                    def_rsi_dec = True if i == 1 else False
                    def_macd_above = True if i == 2 else False
                    def_macd_dec = True if i == 2 else False
                    def_rsi_dead = True if i == 2 else False
                    def_macd_dead = True if i == 3 else False
                    sell_signals.append({
                        'rsi_val': st.number_input(f"RSI 매도 기준 {i}", 0, 100, def_rsi, key=f"s_rsi_{i}"),
                        'rsi_dead': st.checkbox(f"RSI Dead Cross {i}", value=def_rsi_dead, key=f"s_rsi_dead_{i}"),
                        'rsi_dec': st.checkbox(f"RSI 감소 {i}", value=def_rsi_dec, key=f"s_rsi_dec_{i}"),
                        'macd_dec': st.checkbox(f"MACD 감소 {i}", value=def_macd_dec, key=f"s_macd_dec_{i}"),
                        'macd_signal_above': st.checkbox(f"MACD 시그널 상단 {i}", value=def_macd_above, key=f"s_macd_above_{i}"),
                        'macd_dead': st.checkbox(f"MACD 데드크로스 {i}", value=def_macd_dead, key=f"s_macd_dead_{i}"),
                        'bb_upper': st.checkbox(f"볼린저 상단 터치 {i}", value=False, key=f"s_bb_u_{i}")
                    })
        
        st.divider(); st.subheader("🔄 리밸런싱 전략")
        r1, r2, r3, r4 = st.columns(4)
        buy_reb_up = r1.number_input("Buy Reb Up (%)", value=3.0, step=0.5)/100
        buy_reb_down = r2.number_input("Buy Reb Down (%)", value=-3.0, step=0.5)/100
        sell_reb_up = r3.number_input("Sell Reb Up (%)", value=3.0, step=0.5)/100
        sell_reb_down = r4.number_input("Sell Reb Down (%)", value=-3.0, step=0.5)/100
        
        return {
            'buy_signals': buy_signals, 'sell_signals': sell_signals,
            'buy_reb_up': buy_reb_up, 'buy_reb_down': buy_reb_down, 'sell_reb_up': sell_reb_up, 'sell_reb_down': sell_reb_down
        }

# --- Main Application Orchestrator ---

class GoldenStrategyApp:
    """전체 애플리케이션의 흐름을 조율하는 클래스"""
    def __init__(self):
        # 자산 맵핑
        self.ticker_map = {
            "QQQ": "QQQ (1x)", "QLD": "QLD (2x) 🌟", "TQQQ": "TQQQ (3x) 🌟",
            "TLT": "TLT (1x)", "TMF": "TMF (3x)"
        }
        
    def run(self):
        st.title("📈 ETF Golden Strategy")
        
        # 사이드바 메뉴
        menu = st.sidebar.radio("📌 메뉴", ["📊 백테스트", "🧪 전략 분석기", "📰 주요 마켓 이슈"])
        if st.sidebar.button("🔄 데이터 새로고침 (Live)"):
            st.cache_data.clear()
            st.rerun()
            
        # 데이터 로딩
        with st.spinner('실시간 데이터를 가져오는 중...'):
            data_dict, fg_df, vix_df, macro_df, fetch_status, fetch_time = DataService.load_all_data()
        
        if not data_dict:
            st.error("데이터 로드 실패")
            st.stop()
        
        # 마켓 이슈 화면
        if menu == "📰 주요 마켓 이슈":
            MarketView.render(data_dict, macro_df, fetch_status, fetch_time, fg_df)
            st.stop()
            
        # 백테스트 및 전략 분석기 공통 설정
        st.sidebar.header("전략 및 기간 설정")
        base_asset = st.sidebar.selectbox("기준 자산 (시그널)", ["QQQ", "QLD", "TQQQ"], index=0, format_func=lambda x: self.ticker_map[x])
        leverage_asset = st.sidebar.selectbox("매매 대상 자산", ["QLD", "TQQQ", "TMF"], index=0, format_func=lambda x: self.ticker_map[x])
        cash_ratio = st.sidebar.slider("현금 비중 (%)", 0, 50, 0, step=5) / 100.0
        
        # 기본 날짜 설정
        if 'QQQ' in data_dict:
            min_d = data_dict['QQQ'].index.min().date()
            max_d = data_dict['QQQ'].index.max().date()
        else:
            # 데이터가 하나도 없는 경우 (비정상 상황)
            st.error("⚠️ 데이터를 불러올 수 없습니다. 잠시 후 다시 시도해 주세요.")
            st.stop()

        start_d = st.sidebar.date_input("시작일", datetime.date(2026, 1, 1))
        end_d = st.sidebar.date_input("종료일", max_d)
        
        params = None
        if menu == "🧪 전략 분석기":
            st.header("🧪 전략 분석기")
            params = TesterView.render_config()
            
        # 핵심 실행부
        with st.spinner('백테스팅 계산 중...'):
            bh_histories = {t: StrategyEngine.run_benchmark(data_dict, t, start_d, end_d) for t in ['QQQ', 'QLD', 'TQQQ']}
            golden_history = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, cash_ratio, start_d, end_d, params)
            
        # 결과 렌더링
        BacktestView.render_results(golden_history, bh_histories, base_asset, leverage_asset)
        st.info("💡 팁: '전략 분석기'에서 설정을 변경하면 즉시 백테스트 결과가 업데이트됩니다.")

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
