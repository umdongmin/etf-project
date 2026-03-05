import streamlit as st
import datetime
import pandas as pd
from core.intelligence_service import IntelligenceService
from core.news_service import NewsService

class IntelligenceView:
    """AI 마켓 인텔리전스 화면 UI 구성 클래스"""
    
    @staticmethod
    def render():
        st.header("🧠 AI 마켓 인텔리전스 (Market Intelligence)")
        st.caption("뉴스 수집, AI 심층 분석, 리포트 관리를 한곳에서 수행합니다.")
        
        intel_service = IntelligenceService()
        news_service = NewsService()
        
        tab1, tab2, tab3 = st.tabs(["📜 리포트 열람", "📡 뉴스 수집", "🧠 AI 분석 생성"])
        
        # --- Tab 1: 리포트 열람 ---
        with tab1:
            st.sidebar.divider()
            st.sidebar.subheader("📜 리포트 히스토리")
            recent_reports = intel_service.get_all_reports(limit=20)
            
            selected_report_date = None
            if recent_reports:
                report_options = {f"{r['date']} - {r['report_title']}": r['date'] for r in recent_reports}
                options_list = list(report_options.keys())
                
                # 세션 상태를 확인하여 최근 생성된 리포트 자동 선택
                default_idx = 0
                if 'last_generated_date' in st.session_state:
                    for i, (opt_text, opt_date) in enumerate(report_options.items()):
                        if opt_date == st.session_state['last_generated_date']:
                            default_idx = i
                            break
                
                selected_option = st.sidebar.selectbox("과거 리포트 선택", options_list, index=default_idx, key="history_select")
                selected_report_date = report_options[selected_option]
                
                # 열람 후 세션 상태 초기화 (원할 경우)
                if selected_report_date == st.session_state.get('last_generated_date'):
                    st.session_state['last_generated_date'] = None
            else:
                st.info("저장된 리포트가 없습니다. 다른 탭에서 리포트를 먼저 생성해 주세요.")

            if selected_report_date:
                report = intel_service.get_report(selected_report_date)
                if report:
                    IntelligenceView._display_report(report)
                else:
                    st.error("리포트를 불러오는 데 실패했습니다.")

        # --- Tab 2: 뉴스 수집 ---
        with tab2:
            st.subheader("📡 시장 뉴스 수집 및 정형화")
            st.info("""
            **💡 수집 방식 안내**
            *   **뉴스**: `yfinance`를 통해 QQQ, NVDA 등 주요 티커의 최신 헤드라인을 실시간으로 수집합니다. 별도 유료 API 없이 무료 기능을 사용하므로 야후 서버 상태에 따라 간혹 누락될 수 있습니다.
            *   **애널리스트 리포트**: 'AI 분석 생성' 탭에서 직접 텍스트를 입력하여 AI에게 더 풍부한 맥락을 제공할 수 있습니다.
            """)
            
            sc1, sc2 = st.columns([1, 2])
            with sc1:
                collect_date = st.date_input("수집 날짜", datetime.date.today(), key="collect_date")
                collect_date_str = collect_date.strftime("%Y-%m-%d")
            
            with sc2:
                existing_news = news_service.get_sentiment_data(start_date=collect_date_str, end_date=collect_date_str)
                if not existing_news.empty:
                    st.success(f"✅ {collect_date_str} 데이터가 이미 존재합니다.")
                else:
                    st.info(f"ℹ️ {collect_date_str} 데이터가 없습니다. 수집이 필요합니다.")

            # 수집 버튼 영역
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🌐 일반 뉴스 수집 (yfinance)", use_container_width=True):
                    with st.spinner("야후 파이낸스에서 주요 티커 뉴스(QQQ, SPY 등)를 수집 중입니다..."):
                        headlines = news_service.fetch_headlines_from_yf()
                        IntelligenceView._process_and_save_news(news_service, collect_date_str, headlines)
            
            with c2:
                if st.button("💎 고품질 RSS 수집 (Investing.com)", use_container_width=True):
                    with st.spinner("Investing.com 및 글로벌 경제 전문 RSS 피드를 수집 중입니다..."):
                        headlines = news_service.fetch_headlines_from_rss()
                        IntelligenceView._process_and_save_news(news_service, collect_date_str, headlines)

            # 이미 데이터가 있을 경우 내용 미리보기
            if not existing_news.empty:
                with st.expander("🔍 현재 저장된 뉴스 데이터 미리보기"):
                    st.write(f"**핵심 토픽**: {existing_news['topic'].iloc[0]}")
                    st.write(f"**심리 점수**: {existing_news['sentiment'].iloc[0]} (영향력: {existing_news['impact'].iloc[0]})")
                    st.write("**원본 헤드라인**:")
                    raw_h = existing_news['raw_headlines'].iloc[0]
                    if raw_h:
                        for h in raw_h.split('\n')[:10]:
                            st.text(f"• {h}")
                    if st.button("🗑️ 해당 날짜 뉴스 삭제 및 재수집 준비"):
                        # 삭제 로직 (필요시 구현)
                        st.info("삭제 기능은 현재 DB에서 직접 수행해야 합니다.")

        # --- Tab 3: AI 분석 생성 ---
        with tab3:
            st.subheader("🧠 AI 심층 리포트 생성")
            st.write("이미 수집된 뉴스 데이터와 시황 리포트를 결합하여 고품질 인텔리전스 리포트를 생성합니다.")
            
            ac1, ac2 = st.columns([1, 2])
            with ac1:
                analysis_date = st.date_input("분석 대상 날짜", datetime.date.today(), key="analysis_date")
                analysis_date_str = analysis_date.strftime("%Y-%m-%d")
            
            with ac2:
                news_df = news_service.get_sentiment_data(start_date=analysis_date_str, end_date=analysis_date_str)
                if news_df.empty:
                    st.warning(f"⚠️ {analysis_date_str} 뉴스가 없습니다. '뉴스 수집' 탭을 먼저 실행하세요.")
                else:
                    st.success(f"✅ {analysis_date_str} 분석 준비 완료 (뉴스 {len(news_df)}건 확인)")

            # 애널리스트 리포트 입력
            analyst_input = st.text_area("📄 시황 분석 자료/애널리스트 리포트 입력 (선택)", 
                                       placeholder="뉴스 외에 참고할 시황 자료를 입력하면 분석 품질이 올라갑니다.", height=200)

            if st.button("🚀 AI 마켓 인텔리전스 리포트 생성", use_container_width=True, disabled=news_df.empty):
                with st.spinner(f"{analysis_date_str} 리포트를 생성하는 중입니다..."):
                    # 뉴스 헤드라인 추출
                    headlines = []
                    if 'raw_headlines' in news_df.columns:
                        raw_data = news_df['raw_headlines'].iloc[0]
                        headlines = raw_data.split("\n") if raw_data else []
                    
                    # AI 분석 실행
                    report = intel_service.generate_daily_report(analysis_date_str, headlines, analyst_input)
                    
                    if report:
                        st.session_state['last_generated_date'] = analysis_date_str
                        st.success("✅ 리포트가 성공적으로 생성되었습니다! '리포트 열람' 탭에서 확인하세요.")
                        st.balloons()
                        # 약간의 지연 후 리런하여 탭이 변경되어도 데이터가 갱신되게 함
                        st.rerun()
                    else:
                        st.error("리포트 생성 중 오류가 발생했습니다.")

    @staticmethod
    def _process_and_save_news(news_service, date_str, headlines):
        """뉴스 수집 후 정형화 및 저장 공통 로직"""
        if headlines:
            st.write(f"📋 {len(headlines)}개의 정보를 확보했습니다. Gemini 분석을 시작합니다...")
            res = news_service.score_news_with_gemini(date_str, headlines)
            if res:
                news_service.save_sentiment(date_str, res, raw_headlines="\n".join(headlines))
                st.success(f"🎉 {date_str} 뉴스 수집 및 정형화 성공!")
                st.json(res)
                st.rerun()
            else:
                st.error("Gemini 정형화 분석 중 오류가 발생했습니다.")
        else:
            st.error("데이터를 수집하지 못했습니다. 잠시 후 다시 시도해 주세요.")

    @staticmethod
    def _display_report(report):
        """리포트 상세 렌더링"""
        st.divider()
        st.title(report['report_title'])
        
        # 지표 요약 레이아웃
        m1, m2, m3, m4 = st.columns(4)
        score = report['sentiment_score']
        score_color = "normal"
        if score > 0.4: score_label = "낙관 (Greed)"; score_color = "normal"
        elif score < -0.4: score_label = "공포 (Fear)"; score_color = "inverse"
        else: score_label = "중립 (Neutral)"; score_color = "off"
            
        m1.metric("종합 심리 지수", f"{score:.2f}", score_label, delta_color=score_color)
        
        # 신뢰도 지표 추가
        rel_score = report.get('reliability_score', 0.0)
        if rel_score > 0.7: rel_label = "높음 (Trust)"; rel_color = "normal"
        elif rel_score < 0.4: rel_label = "낮음 (Noise)"; rel_color = "inverse"
        else: rel_label = "보통 (Middle)"; rel_color = "off"
        
        m2.metric("분석 신뢰도", f"{rel_score:.2f}", rel_label, delta_color=rel_color)
        m3.metric("분석 기준일", report['date'])
        m4.metric("출처 수", f"{report.get('source_count', 0)}개")
        
        # 리포트 본문 (Markdown)
        st.markdown(report['content'])
        
        # 하단 메타데이터
        with st.expander("🔗 사용된 데이터 출처 정보", expanded=False):
            if report.get('source_metadata'):
                metadata = report['source_metadata']
                if isinstance(metadata, str):
                    try: metadata = json.loads(metadata)
                    except: pass
                
                if isinstance(metadata, list):
                    for idx, item in enumerate(metadata):
                        st.write(f"{idx+1}. {item}")
                else:
                    st.write(metadata)
            else:
                st.write("출처 정보가 기록되지 않았습니다.")
