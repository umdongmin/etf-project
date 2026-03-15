import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from core.engine import StrategyEngine
from core.llm_service import LLMService

class RollingLabView:
    """전략의 강건성 검증을 위한 롤링 윈도우 분석 UI 클래스"""

    @staticmethod
    def render(data_dict, fg_df, vxn_df, news_df, leverage_asset, base_asset, trade_at, current_params, smart_params=None):
        st.header("🧬 롤링 윈도우 분석 (Rolling Lab)")
        st.caption("시점 독립적 성과 검증: 특정 기간을 고정하고 전진하며 테스트하여 전략의 일관성과 강건성을 분석합니다.")

        if base_asset not in data_dict:
            st.error("데이터가 준비되지 않았습니다.")
            return

        # 1. 설정 섹션
        with st.expander("⚙️ 분석 설정", expanded=True):
            col1, col2, col3 = st.columns(3)
            window_years = col1.slider("윈도우 크기 (년)", 1, 10, 3)
            step_years = col2.slider("전진 간격 (년)", 1, 5, 1)
            
            # 데이터 시작/종료일 파악
            full_dates = data_dict[base_asset].index
            min_date = full_dates.min().date()
            max_date = full_dates.max().date()
            
            st.info(f"데이터 전체 기간: {min_date} ~ {max_date}")

        # 2. 롤링 백테스트 실행 로직 (1월 1일 ~ 12월 31일 기준 정렬)
        windows = []
        current_start_year = min_date.year
        
        while True:
            # 시작일: 해당 연도 1월 1일
            start_d = datetime.date(current_start_year, 1, 1)
            # 종료일: n년 뒤 12월 31일
            end_d = datetime.date(current_start_year + window_years - 1, 12, 31)
            
            if end_d > max_date:
                break
            
            windows.append((start_d, end_d))
            current_start_year += step_years

        if not windows:
            st.warning("설정한 연도 범위가 데이터 전체 기간보다 큽니다. 설정을 조정해 주세요.")
            return

        st.subheader(f"📊 {window_years}년 롤링 분석 결과 ({len(windows)}개 구간)")
        
        results = []
        spider_data = []

        with st.spinner(f'{len(windows)}개 구간 시뮬레이션 중...'):
            cash_ratio = current_params.get('cash_ratio_pct', 0.0) / 100.0
            
            for i, (start_d, end_d) in enumerate(windows):
                # 전략 실행 (fast_mode=True로 속도 최적화)
                hist, _, _ = StrategyEngine.run_golden_strategy(
                    data_dict, fg_df, vxn_df, news_df, leverage_asset, base_asset, 
                    cash_ratio, start_d, end_d, current_params, trade_at, 
                    smart_params=smart_params, fast_mode=True, salt=f"rolling_{i}_{window_years}"
                )
                
                if not hist.empty:
                    # 수익률 및 MDD 계산
                    total_ret = (hist['Value'].iloc[-1] / hist['Value'].iloc[0] - 1) * 100
                    
                    # CAGR (연평균 수익률) 계산
                    years_count = (end_d - start_d).days / 365.25
                    cagr = ((hist['Value'].iloc[-1] / hist['Value'].iloc[0]) ** (1/years_count) - 1) * 100 if years_count > 0 else 0
                    
                    # MDD
                    roll_max = hist['Value'].cummax()
                    drawdown = (hist['Value'] - roll_max) / roll_max
                    mdd = drawdown.min() * 100
                    
                    # Spider Chart용 데이터 가공 (0% 시작)
                    hist_indexed = (hist['Value'] / hist['Value'].iloc[0] - 1) * 100
                    # 날짜 대신 경과 일수(0부터 시작)로 인덱스 변경
                    days_passed = [(d - hist.index[0]).days for d in hist.index]
                    
                    spider_data.append({
                        'name': f"{start_d.year}~{end_d.year}",
                        'days': days_passed,
                        'returns': hist_indexed.values
                    })
                    
                    results.append({
                        "구간": f"{start_d.year}~{end_d.year}",
                        "시작일": start_d.strftime('%Y-%m-%d'),
                        "종료일": end_d.strftime('%Y-%m-%d'),
                        "최종 수익률": f"{total_ret:+.1f}%",
                        "연평균(CAGR)": f"{cagr:.1f}%",
                        "MDD": f"{mdd:.1f}%",
                        "raw_ret": total_ret,
                        "raw_cagr": cagr,
                        "raw_mdd": mdd
                    })

        if not results:
            st.error("분석 결과가 없습니다.")
            return

        # 3. Spider Chart 렌더링
        fig = go.Figure()
        for data in spider_data:
            fig.add_trace(go.Scatter(
                x=data['days'], y=data['returns'],
                mode='lines',
                name=data['name'],
                opacity=0.5,
                line=dict(width=1.5)
            ))
        
        fig.update_layout(
            title=f"Spider Chart: {window_years}년 구간별 성과 중첩 (0% 시작)",
            xaxis_title="경과 일수 (Days)",
            yaxis_title="수익률 (%)",
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. 성과 요약 통계
        res_df = pd.DataFrame(results)
        avg_ret = res_df['raw_ret'].mean()
        avg_cagr = res_df['raw_cagr'].mean()
        std_ret = res_df['raw_ret'].std()
        avg_mdd = res_df['raw_mdd'].mean()
        win_rate = (res_df['raw_ret'] > 0).mean() * 100

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("평균 수익률", f"{avg_ret:+.1f}%")
        c2.metric("평균 CAGR", f"{avg_cagr:.1f}%")
        c3.metric("수익률 표준편차", f"{std_ret:.1f}%")
        c4.metric("평균 MDD", f"{avg_mdd:.1f}%")
        c5.metric("수익 구간 확률", f"{win_rate:.1f}%")

        # [신규] AI 분석 리포트 생성 버튼
        st.write("")
        llm = LLMService()
        if llm.is_available():
            if st.button("🤖 AI 전략 강건성 정밀 진단 리포트 생성", width='stretch'):
                with st.spinner('AI가 롤링 데이터를 심층 분석 중입니다...'):
                    summary_metrics = {
                        'avg_ret': avg_ret,
                        'avg_cagr': avg_cagr,
                        'std_ret': std_ret,
                        'avg_mdd': avg_mdd,
                        'win_rate': win_rate
                    }
                    report = llm.generate_rolling_report(results, summary_metrics, window_years)
                    st.markdown("---")
                    st.subheader("📝 AI 전략 진단 결과")
                    st.markdown(report)
        else:
            st.info("💡 사이드바에서 Gemini API 키를 설정하면 AI 정밀 진단 리포트를 받아볼 수 있습니다.")

        # 5. 마스터 테이블
        st.divider()
        st.subheader("📋 롤링 분석 마스터 테이블 (RAW)")
        
        # 스타일링을 위한 함수
        def color_ret(val):
            try:
                num = float(val.replace('%', '').replace('+', ''))
                color = '#00c853' if num >= 0 else '#ff1744'
                return f'color: {color}; font-weight: bold'
            except: return ''

        display_df = res_df[["구간", "시작일", "종료일", "최종 수익률", "연평균(CAGR)", "MDD"]].set_index("구간")
        st.dataframe(
            display_df.style.map(color_ret, subset=["최종 수익률", "연평균(CAGR)"]),
            width='stretch'
        )

        # 데이터 다운로드 버튼
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 분석 결과 CSV 다운로드",
            data=csv,
            file_name=f"rolling_analysis_{window_years}y.csv",
            mime='text/csv',
        )
