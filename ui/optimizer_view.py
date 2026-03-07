import streamlit as st
import json
import datetime
import pandas as pd
from core.optimizer import StrategyOptimizer
from core.llm_service import LLMService

class OptimizerView:
    """AI를 활용한 전략 파라미터 최적화 UI 구성 클래스"""
    
    @staticmethod
    def render(data_dict, fg_df, vix_df, news_df, current_params):
        st.markdown("### 🧠 지능형 전략 최적화 도구 (AI Optimizer)")
        
        # [Phase 3] Gemini API 설정 상시 노출 (접근성 개선)
        with st.sidebar:
            st.markdown("---")
            st.markdown("#### 🤖 AI 설정")
            llm = LLMService()
            if not llm.is_available():
                with st.expander("🔑 Gemini API Key 설정", expanded=False):
                    st.info("AI 분석을 위해 Google API 키가 필요합니다.")
                    u_key = st.text_input("API Key 입력", type="password", key="sidebar_key")
                    if st.button("키 저장", key="save_sidebar_key"):
                        if u_key:
                            with open(".env", "a") as f: f.write(f"\nGEMINI_API_KEY={u_key}")
                            st.success("API 키 저장 완료! (새로고침 하세요)")
                            st.rerun()
            else:
                st.success("✅ AI 브리핑 준비 완료")

        optimizer = StrategyOptimizer(data_dict, fg_df, vix_df, news_df)
        
        # [성능 최적화 1] 총 분석 횟수 캐싱 (10초 주기)
        if 'opt_refresh_key' not in st.session_state:
            st.session_state.opt_refresh_key = 0
            
        @st.cache_data(ttl=60)
        def get_total_trials_cached(_opt, refresh_key):
            return _opt.get_total_trials()
        
        total_trials = get_total_trials_cached(optimizer, st.session_state.opt_refresh_key)
        
        # [신규] 세션 상태의 오타 실시간 교정 (벨 -> 밸)
        if 'top_dashboard_targets' in st.session_state:
            st.session_state.top_dashboard_targets = [t.replace("벨", "밸") for t in st.session_state.top_dashboard_targets]
        
        st.markdown(f"""
        <div style="background: #e8f4f8; padding: 12px; border-radius: 8px; border-left: 5px solid #3498db; margin-bottom: 15px;">
            🤖 <b>자가 학습형 베이지안 엔진 가동 중</b><br>
            • 현재까지 축적된 분석 데이터: <span style="color:#2980b9; font-weight:bold;">{total_trials}개</span><br>
            • 무작위가 아닌 <b>과거 성과를 학습</b>하여 최적의 파라미터를 역추적합니다.<br>
            • ⚠️ <b>전 영역 최적화(Full)</b> 선택 시 조합의 수가 많아져 분석 시간이 길어질 수 있습니다.
        </div>
        """, unsafe_allow_html=True)

        # [속도 개선] 입력 폼을 사용하여 이벤트 핸들링 최적화
        with st.form("optimizer_settings_form"):
            st.markdown("#### ⚙️ 최적화 대상 및 탐색 설정")
            col_p1, col_p2, col_p3 = st.columns([2, 3, 2])
            with col_p1:
                target_period = st.selectbox("📅 분석 대상 기간 선택", ["전체 (2010~현재)", "최근 5년", "최근 3년"], index=0)
            with col_p2:
                opt_targets = st.multiselect(
                    "🎯 현재 필터링할 탐색 대상 (Targets)",
                    ["매수 시그널 (Full)", "매도 시그널 (Full)", "리밸런싱 및 ATR 변동성 (Full)", "패닉 가속 (MA/RSI)", "S3 보호 조건 (Gap/Acc)", "방어 및 대피 시스템 (VXN/손절)"],
                    default=["매수 시그널 (Full)", "매도 시그널 (Full)", "리밸런싱 및 ATR 변동성 (Full)"],
                    key="top_dashboard_targets"
                )
            with col_p3:
                st.write("") # 간격 조절
                use_wfa = st.checkbox("🧪 WFA(전진 분석) 모드", value=False, key="dash_use_wfa", help="데이터를 훈련(IS)과 검증(OOS)으로 분할하여 '과최적화' 여부를 테스트합니다.")
                wfa_ratio_input = st.slider("OOS 비중", 10, 50, 30, step=5, format="%d%%", key="dash_wfa_ratio")
                
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("탐색 횟수 (N)", 50, 1000, 200, step=50, help="탐색 횟수가 많을수록 더 정교한 결과를 얻지만 시간이 오래 걸립니다.")
            
            with col2:
                trade_at = st.selectbox("최적화용 매매 시점", ["종가", "익일 시가"], index=0 if current_params.get('trade_at') == '종가' else 1)
            
            submit_col1, submit_col2 = st.columns([1, 2])
            with submit_col1:
                update_dash = st.form_submit_button("🔄 설정 적용 및 대시보드 갱신", use_container_width=True)
            with submit_col2:
                start_opt = st.form_submit_button("🔥 AI 최적화 탐색 시작", type="primary", use_container_width=True)
            
        wfa_ratio = wfa_ratio_input / 100.0 if use_wfa else 0.0

        # 기간 설정
        s_date = datetime.date(2010, 1, 1)
        if target_period == "최근 5년": s_date = datetime.date.today() - datetime.timedelta(days=5*365)
        elif target_period == "최근 3년": s_date = datetime.date.today() - datetime.timedelta(days=3*365)
        e_date = datetime.date.today()

        # [Phase 4] 실시간/상시 성과 현황판 섹션
        if total_trials > 0:
            with st.expander(f"📊 {target_period} 최적 성과 대시보드", expanded=True):
                # [성능 최적화 2] 상위 결과 캐싱 및 수동 업데이트
                # 설정 슬라이더를 바꿀 때마다 리시뮬레이션되는 것을 방지
                @st.cache_data(ttl=600)
                def get_top_results_cached(_opt, _params, refresh_key, _period, _targets, _wfa_ratio):
                    # [수정] targets과 wfa_ratio를 넘겨서 컨텍스트 필터링 수행
                    return _opt.get_top_results(_params, n_top=3, target_period=_period, targets=_targets, wfa_ratio=_wfa_ratio)
                
                # [속도 개선] 폼 제출 시 자동으로 세션 키를 증가시켜 데이터 갱신 유도
                if update_dash:
                    st.session_state.opt_refresh_key += 1
                
                top_results = get_top_results_cached(optimizer, current_params, st.session_state.opt_refresh_key, target_period, opt_targets, wfa_ratio)
                if top_results:
                    col_h1, col_h2, col_h3 = st.columns(3)
                    for i, res in enumerate(top_results):
                        with [col_h1, col_h2, col_h3][i]:
                            st.metric(f"Rank {i+1}", f"Score {res['score']:.2f}", f"CAGR {res['cagr']:.1f}%")
                            # [Phase 1] 대시보드에도 주요 정형 지표 추가 노출
                            st.caption(f"Sharpe: {res.get('sharpe', 0):.2f} | PF: {res.get('pf', 0):.2f}")
                    
                    st.markdown("---")
                    llm = LLMService()
                    if llm.is_available():
                        if st.button("🖋️ 현재까지의 성과 AI 중간 브리핑 받기", key="mid_term_report", use_container_width=True, type="secondary"):
                            with st.spinner("현재까지의 데이터를 기반으로 중간 요약 리포트를 작성 중..."):
                                b_metrics = {
                                    'cagr': current_params.get('cagr', 0), 
                                    'mdd': current_params.get('mdd', 0),
                                    'sharpe': current_params.get('sharpe', 0)
                                }
                                report = llm.generate_briefing(top_results, b_metrics, "현재까지의 탐색 범위")
                                st.session_state.mid_report = report
                    else:
                        st.warning("⚠️ AI 브리핑을 위해 Gemini API 키 설정이 필요합니다. (하단 '설정' 참고)")
                    
                    if 'mid_report' in st.session_state:
                        st.info("💡 **AI 중간 분석 리포트**")
                        st.markdown(st.session_state.mid_report)
                        if st.button("브리핑 닫기", key="close_mid"):
                            del st.session_state.mid_report
                            st.rerun()
                else:
                    st.write("아직 충분한 분석 데이터가 축적되지 않았습니다.")
        
        if start_opt:
            optimizer = StrategyOptimizer(data_dict, fg_df, vix_df)
            
            # [수정] 베이스라인 데이터를 미리 계산하여 base_res 정의 (NameError 방지)
            base_res = optimizer.run_simulation(current_params, s_date, e_date, trade_at)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_cb(current, total, best):
                # [수정] 진행률이 1.0을 초과하지 않도록 제한 (StreamlitAPIException 방지)
                progress_val = min(current / total, 1.0)
                progress_bar.progress(progress_val)
                if current % 10 == 0 or current == total:
                    status_text.text(f"⏳ 탐색 중... ({current}/{total}) | 현재 최고 점수: {best:.2f}")

            with st.spinner("AI가 최적 조합을 심층 탐색 중입니다..." + (" (WFA 검증 포함)" if use_wfa else "")):
                results, b_score, b_cagr, b_mdd = optimizer.optimize(
                    current_params, n_iter, s_date, e_date, opt_targets, trade_at, 
                    target_period=target_period, progress_callback=progress_cb,
                    wfa_ratio=wfa_ratio
                )
                # [신규] 결과를 세션 상태에 저장 (버튼 클릭 후 리그림 시 유실 방지)
                st.session_state.opt_results = results
                st.session_state.opt_baseline = {
                    'score': b_score, 'cagr': b_cagr, 'mdd': b_mdd,
                    'sharpe': base_res.get('sharpe', 0),
                    'pf': base_res.get('pf', 0),
                    'win_rate': base_res.get('win_rate', 0)
                }

        # [신규] 세션 상태에 저장된 결과가 있으면 표시
        if 'opt_results' in st.session_state and st.session_state.opt_results:
            results = st.session_state.opt_results
            b = st.session_state.opt_baseline
            
            st.success("✅ 최적화 탐색 완료!")
            st.info(f"📊 기준 성과 (Baseline): Score {b['score']:.2f} (CAGR {b['cagr']:.1f}%, MDD -{b['mdd']:.1f}%)")
            
            # 결과 테이블 표시 (정형화 데이터 확장)
            top_results = results[:5]
            sum_data = []
            for idx, r in enumerate(top_results):
                row = {
                    "Rank": idx + 1,
                    "IS Score": f"{r['score']:.2f}",
                    "IS CAGR": f"{r['cagr']:.1f}%",
                    "IS MDD": f"-{r['mdd']:.1f}%",
                    "Sharpe": f"{r.get('sharpe', 0):.2f}",
                    "Win%": f"{r.get('win_rate', 0):.1f}%",
                    "PF": f"{r.get('pf', 0):.2f}",
                    "Diff": f"{r['score'] - b['score']:+.2f}"
                }
                # WFA 데이터가 있을 경우 추가 노출
                if r.get('wfa_mode'):
                    row.update({
                        "OOS CAGR": f"{r.get('oos_cagr', 0):.1f}%",
                        "OOS MDD": f"-{r.get('oos_mdd', 0):.1f}%",
                        "Robust(WFE)": f"{r.get('wfe', 0)*100:.1f}%"
                    })
                
                sum_data.append(row)
            
            st.dataframe(pd.DataFrame(sum_data).set_index("Rank"), use_container_width=True)
            
            # 최고 성과 적용 및 저장
            best = top_results[0]
            if best['score'] > b['score']:
                st.balloons()
                st.markdown(f"🏆 **최적 파라미터 발견!** (점수 개선: {best['score'] - b['score']:+.2f})")
                
                # [Phase 2] 몬테카를로 정밀 검증
                with st.expander("🔬 Rank 1 전략 정밀 검증 (Monte Carlo)", expanded=False):
                    st.write("가상의 노이즈(±0.1%)를 가격 데이터에 섞어 20회 반복 테스트를 수행합니다.")
                    if st.button("🧪 몬테카를로 생존 테스트 시작", key="run_mc_btn"):
                        with st.spinner("노이즈 상황에서의 생존력 테스트 중..."):
                            mc_res = optimizer.run_monte_carlo(best['cfg'], s_date, e_date, trade_at)
                            st.session_state.mc_res = mc_res
                    
                    if 'mc_res' in st.session_state:
                        mc = st.session_state.mc_res
                        col_mc1, col_mc2, col_mc3 = st.columns(3)
                        with col_mc1:
                            st.metric("생존 확률", f"{mc['survival_rate']:.1f}%")
                        with col_mc2:
                            st.metric("평균 CAGR", f"{mc['avg_cagr']:.1f}%")
                        with col_mc3:
                            st.metric("하위 10% CAGR", f"{mc['low_cagr']:.1f}%")
                        
                        if mc['survival_rate'] >= 90:
                            st.success("✅ 이 전략은 시장 노이즈에 매우 강합니다. (Robust)")
                        elif mc['survival_rate'] >= 70:
                            st.warning("⚠️ 평이한 수준의 안정성입니다. 실전에서 변동성이 더 클 수 있습니다.")
                        else:
                            st.error("❌ 안정성이 낮습니다. 과최적화 가능성이 매우 높으니 주의하세요.")
                
                # [신규] 저장 및 적용 옵션 UI 개선
                # [Phase 3] AI 리포트 브리핑 섹션 (결과 표시부)
                st.markdown("---")
                st.markdown("#### 🤖 AI 전략 인사이트 브리핑 (Powered by Gemini)")
                
                llm = LLMService()
                if not llm.is_available():
                    st.warning("⚠️ 사이드바에서 Gemini API 키를 먼저 설정해 주세요.")
                else:
                    period_text = "전체" if target_period == "전체 (2010~현재)" else target_period
                    if st.button("🖋️ AI에게 전략 분석 리포트 요청하기", type="secondary", use_container_width=True):
                        with st.spinner("AI가 성과 데이터를 정밀 분석하여 리포트를 작성 중입니다..."):
                            report = llm.generate_briefing(results, st.session_state.opt_baseline, period_text)
                            st.session_state.ai_report = report
                    
                    if 'ai_report' in st.session_state:
                        st.markdown(st.session_state.ai_report)
                        if st.button("🗑️ 리포트 닫기"):
                            del st.session_state.ai_report
                            st.rerun()

                st.markdown("---")
                c_act1, c_act2 = st.columns(2)
                
                with c_act1:
                    st.write("📋 **세션 관리**")
                    if st.button("✨ 최고 성과 세션에 바로 적용", use_container_width=True):
                        st.session_state.current_params = best['cfg']
                        st.success("세션에 적용되었습니다. 상단 탭에서 '분석 결과'를 확인하세요.")
                        st.rerun()
                
                with c_act2:
                    st.write("💾 **파일 영구 저장**")
                    # [신규] 저장할 전략 이름 입력 필드 추가
                    suggested_name = f"tqqq_strategy_Optimized_{datetime.datetime.now().strftime('%m%d_%H%M')}"
                    new_opt_name = st.text_input("저장할 전략 이름", value=suggested_name, help="입력한 이름으로 strategies 폴더에 저장됩니다.")
                    
                    if st.button("💾 입력한 이름으로 저장하기", use_container_width=True):
                        if new_opt_name.strip():
                            from core.storage import StrategyStorage
                            # 파일명에서 특수문자 및 확장자 제거 시도 (안정성)
                            clean_name = new_opt_name.replace(".json", "").strip()
                            StrategyStorage.save_strategy(clean_name, best['cfg'])
                            st.success(f"'{clean_name}.json' 파일이 성공적으로 저장되었습니다.")
                        else:
                            st.warning("경고: 저장할 전략 이름을 입력해 주세요.")
            else:
                st.warning("기존 베이스라인을 넘는 성과를 발견하지 못했습니다. 탐색 범위를 넓히거나 횟수를 늘려보세요.")
