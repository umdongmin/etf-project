import streamlit as st
import datetime
import pandas as pd
import json
from core.intelligence_service import IntelligenceService
from core.news_service import NewsService

class IntelligenceView:
    """AI 뉴스 분석 리포트 화면 UI 구성 클래스 (통합 간소화 버전)"""
    
    @staticmethod
    def render():
        st.header("📑 AI 뉴스 분석 리포트 (News Analysis Report)")
        st.caption("뉴스 수집, AI 심층 분석, 리포트 관리를 한곳에서 수행합니다.")
        
        intel_service = IntelligenceService()
        news_service = NewsService()
        
        # --- 리포트 데이터 로드 ---
        recent_reports = intel_service.get_all_reports(limit=30)
        
        # --- 메인 화면 상단: 리포트 생성 및 선택 컨트롤 ---
        st.divider()
        
        # 1행: 날짜 선택 및 생성 버튼
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c1:
            sync_date = st.date_input("기준 날짜", datetime.date.today(), key="sync_date")
            sync_date_str = sync_date.strftime("%Y-%m-%d")
        
        with c2:
            st.write("") # 간격 맞춤
            st.write("")
            generate_btn = st.button("🚀 새 리포트 작성", width='stretch', type="primary")

        with c3:
            # 과거 리포트 선택 (기존 사이드바에서 이동)
            if recent_reports:
                report_options = {f"{r['date']} - {r['report_title']}": r['date'] for r in recent_reports}
                options_list = list(report_options.keys())
                
                # 세션 상태를 확인하여 최근 생성된 리포트 자동 선택
                default_idx = 0
                if 'last_generated_date' in st.session_state:
                    for i, opt_date in enumerate(report_options.values()):
                        if opt_date == st.session_state['last_generated_date']:
                            default_idx = i
                            break
                
                selected_option = st.selectbox("📂 과거 리포트 열람", options_list, index=default_idx, key="history_select")
                selected_report_date = report_options[selected_option]
            else:
                st.info("저장된 리포트 없음")
                selected_report_date = None

        # 애널리스트 리포트 입력 (필요 시 펼쳐서 입력)
        with st.expander("📄 [고급] 추가 분석 자료 입력 (참고용)", expanded=False):
            st.write("확보된 뉴스 외에 수기로 분석 자료를 입력하여 AI에게 더 풍부한 맥락을 제공할 수 있습니다.")
            st.text_area("시황 분석 자료/애널리스트 리포트", 
                        placeholder="입력된 내용은 AI 분석 리포트 생성 시 '참고 데이터'로 활용됩니다.", 
                        height=150, key="manual_analyst_input")

        # 리포트 생성 로직
        if generate_btn:
            status_area = st.empty()
            with st.spinner(f"{sync_date_str} 마켓 분석 중..."):
                status_area.info("⏳ [1/2] 🏛️ 뉴스 데이터 수집 및 분석 중...")
                
                analyst_reports = st.session_state.get('manual_analyst_input', "")
                report = intel_service.sync_market_intelligence(sync_date_str, news_service, analyst_reports=analyst_reports)
                
                if report:
                    status_area.empty()
                    st.session_state['last_generated_date'] = sync_date_str
                    st.success(f"✅ {sync_date_str} 리포트 생성 완료!")
                    st.balloons()
                    st.rerun()
                else:
                    status_area.error("리포트 생성 중 오류가 발생했습니다. 뉴스 데이터 확보 여부를 확인해 주세요.")

        # 리포트 상세 표시
        if selected_report_date:
            report = intel_service.get_report(selected_report_date)
            if report:
                IntelligenceView._display_report(report)
            else:
                st.error("리포트를 불러오는 데 실패했습니다.")
        elif not recent_reports:
            st.info("상단의 버튼을 눌러 첫 번째 AI 뉴스 분석 리포트를 생성해 보세요.")

    @staticmethod
    def _display_report(report):
        """리포트 상세 렌더링"""
        st.divider()
        st.markdown(f"### 📑 {report['report_title']}")
        
        m1, m2 = st.columns(2)
        m1.metric("분석 기준일", report['date'])
        m2.metric("출처 수", f"{report.get('source_count', 0)}개")
        
        # 리포트 본문
        st.markdown(report['content'])
        
        # 출처 정보
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
