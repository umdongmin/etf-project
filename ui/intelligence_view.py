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
        
        tab1, tab2 = st.tabs(["📜 리포트 열람", "⚡ 실시간 시장 데이터 동기화"])
        
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
            else:
                st.info("저장된 리포트가 없습니다. 동기화 버튼을 눌러 첫 리포트를 생성해 주세요.")

            if selected_report_date:
                report = intel_service.get_report(selected_report_date)
                if report:
                    IntelligenceView._display_report(report)
                else:
                    st.error("리포트를 불러오는 데 실패했습니다.")

        # --- Tab 2: 실시간 시장 데이터 동기화 ---
        with tab2:
            st.subheader("⚡ 실시간 시장 데이터 통합 동기화")
            st.write("뉴스 수집, AI 데이터 정형화, 심층 리포트 생성을 단 한 번의 클릭으로 완료합니다.")
            
            sync_date = st.date_input("기준 날짜", datetime.date.today(), key="sync_date")
            sync_date_str = sync_date.strftime("%Y-%m-%d")
            
            st.info(f"""
            **🔍 동기화 대상 소스**
            1.  🏛️ **Macro**: 국내 대형 증권사 리서치 리포트 요약
            2.  🎯 **Micro**: Finviz 실시간 글로벌 헤드라인 및 주요 지표 뉴스
            3.  🧠 **AI Engine**: 구글 Gemini 2.5 Flash (**데이터 0.0** / **리포트 0.3**)
            """)

            if st.button("🚀 통합 동기화 시작 (Sync Now)", width='stretch'):
                status_area = st.empty()
                with st.spinner("시장 데이터 동기화 중..."):
                    # 1. Macro 수집 시작 표시
                    status_area.info("⏳ [1/4] 🏛️ Macro 리포트 수집 중 (네이버 금융)...")
                    
                    # 2. Micro 수집 시작 표시
                    status_area.info("⏳ [2/4] 🎯 Micro 헤드라인 수집 중 (Finviz)...")
                    
                    # 3. 데이터 정형화 표시
                    status_area.info("⏳ [3/4] 🎲 퀀트 데이터 정형화 중 (Temperature 0.0)...")
                    
                    # 4. 리포트 생성 표시
                    status_area.info("⏳ [4/4] 📑 프리미엄 리포트 집필 중 (Temperature 0.3)...")
                    
                    analyst_reports = st.session_state.get('manual_analyst_input', "")
                    report = intel_service.sync_market_intelligence(sync_date_str, news_service, analyst_reports=analyst_reports)
                    
                    if report:
                        status_area.empty()
                        st.session_state['last_generated_date'] = sync_date_str
                        st.success(f"✅ {sync_date_str} 시장 데이터 동기화 완료!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("동기화 중 오류가 발생했습니다. 뉴스 데이터 확보 여부를 확인해 주세요.")

            # 과거 데이터 존재 여부 확인
            existing_data = news_service.get_sentiment_data(start_date=sync_date_str, end_date=sync_date_str)
            if not existing_data.empty:
                st.markdown("---")
                st.caption(f"📍 {sync_date_str}에 대한 데이터가 이미 시스템에 동기화되어 있습니다.")

            # 애널리스트 리포트 입력 (통합 동기화 시 참고용으로 이동 가능)
            with st.expander("📄 [고급] 추가 분석 자료 입력", expanded=False):
                st.write("이미 확보한 뉴스 외에 수기로 분석 자료를 입력하여 AI에게 더 풍부한 맥락을 제공할 수 있습니다.")
                st.text_area("시황 분석 자료/애널리스트 리포트", 
                            placeholder="입력된 내용은 AI 분석 리포트 생성 시 '참고 데이터'로 활용됩니다.", 
                            height=200, key="manual_analyst_input")

    @staticmethod
    def _display_report(report):
        """리포트 상세 렌더링"""
        st.divider()
        # [수정] 제목 크기 및 색상 강조
        st.markdown(f"### 📑 {report['report_title']}")
        
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
