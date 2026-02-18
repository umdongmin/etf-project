import streamlit as st
import pandas as pd
import datetime

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
            # [최신화] 2026년 2월 기준 추정치 반영 (QQQ 500 내외 기준)
            v1.metric("나스닥 100 PER", f"{(latest_qqq / 15.6):.1f}x", delta="평균 24~26x 대비 높음", delta_color="inverse")
            v2.metric("버핏 지수 (추정)", f"{(220.1 * (latest_qqq / 505.0)):.1f}%", delta="200% 이상 과열", delta_color="inverse")
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
        m2.metric("VIX 공포 지수", f"{macro_df.get('VIX', pd.Series([15.0])).iloc[-1]:.2f}")
        
        pccr_val = macro_df.get('PCCR')
        pccr_display = f"{pccr_val.iloc[-1]:.2f}" if pccr_val is not None and not pccr_val.empty else "N/A"
        m3.metric("풋/콜 비율 (PCCR)", pccr_display)
        
        st.divider()
        st.subheader("📅 주요 경제 일정 (2026)")
        today = datetime.date.today()

        fomc_events = [
            {"date": datetime.date(2026, 1, 28), "actual": "5.50% (동결)", "consensus": "5.50%"},
            {"date": datetime.date(2026, 3, 18), "actual": None, "consensus": "5.25% 기대"},
            {"date": datetime.date(2026, 4, 29), "actual": None, "consensus": "5.00% 기대"},
        ]
        cpi_events = [
            {"date": datetime.date(2026, 1, 13), "actual": "3.1%", "consensus": "3.1%"},
            {"date": datetime.date(2026, 2, 13), "actual": "3.1%", "consensus": "3.0%"},
            {"date": datetime.date(2026, 3, 11), "actual": None, "consensus": "2.9% 전망"},
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
            st.markdown("- **2월 4주차 (2/25 예정)** : NVDA (4분기 실적 발표)")
            st.markdown("- **4월 4주차 (4/27~5/01)** : 2026년 1분기 실적 발표 시즌 (AAPL, MSFT 등)")

        st.divider()
        with st.expander("🛠️ 데이터 수집 상태 정보", expanded=False):
            st.table(pd.DataFrame([fetch_status]).T)
