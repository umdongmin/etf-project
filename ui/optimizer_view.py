import streamlit as st
import json
import datetime
import pandas as pd
from core.optimizer import StrategyOptimizer

class OptimizerView:
    """AI를 활용한 전략 파라미터 최적화 UI 구성 클래스"""
    
    @staticmethod
    def render(data_dict, fg_df, vix_df, current_params):
        st.markdown("### 🧠 지능형 전략 최적화 도구 (AI Optimizer)")
        
        optimizer = StrategyOptimizer(data_dict, fg_df, vix_df)
        total_trials = optimizer.get_total_trials()
        
        st.markdown(f"""
        <div style="background: #e8f4f8; padding: 12px; border-radius: 8px; border-left: 5px solid #3498db; margin-bottom: 15px;">
            🤖 <b>자가 학습형 베이지안 엔진 가동 중</b><br>
            • 현재까지 축적된 분석 데이터: <span style="color:#2980b9; font-weight:bold;">{total_trials}개</span><br>
            • 무작위가 아닌 <b>과거 성과를 학습</b>하여 최적의 파라미터를 역추적합니다.<br>
            • ⚠️ <b>전 영역 최적화(Full)</b> 선택 시 조합의 수가 많아져 분석 시간이 길어질 수 있습니다.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🛠️ 최적화 설정 및 탐색 범위", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("탐색 횟수 (N)", 50, 1000, 200, step=50, help="탐색 횟수가 많을수록 더 정교한 결과를 얻지만 시간이 오래 걸립니다.")
                target_period = st.radio("최적화 대상 기간", ["전체 (2010~현재)", "최근 5년", "최근 3년"], index=0)
            
            with col2:
                opt_targets = st.multiselect(
                    "최적화 대상 변수",
                    ["매수 시그널 (Full)", "매도 시그널 (Full)", "리밸런싱 및 ATR 변동성 (Full)", "패닉 가속 (MA/RSI)", "S3 보호 조건 (Gap/Acc)", "방어 및 대피 시스템 (VXN/손절)"],
                    default=["매수 시그널 (Full)", "매도 시그널 (Full)", "리밸런싱 및 ATR 변동성 (Full)"]
                )
                trade_at = st.selectbox("최적화용 매매 시점", ["종가", "익일 시가"], index=0 if current_params.get('trade_at') == '종가' else 1)

        # 기간 설정
        s_date = datetime.date(2010, 1, 1)
        if target_period == "최근 5년": s_date = datetime.date.today() - datetime.timedelta(days=5*365)
        elif target_period == "최근 3년": s_date = datetime.date.today() - datetime.timedelta(days=3*365)
        e_date = datetime.date.today()

        if st.button("🔥 AI 최적화 탐색 시작", type="primary", use_container_width=True):
            optimizer = StrategyOptimizer(data_dict, fg_df, vix_df)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_cb(current, total, best):
                # [수정] 진행률이 1.0을 초과하지 않도록 제한 (StreamlitAPIException 방지)
                progress_val = min(current / total, 1.0)
                progress_bar.progress(progress_val)
                if current % 10 == 0 or current == total:
                    status_text.text(f"⏳ 탐색 중... ({current}/{total}) | 현재 최고 점수: {best:.2f}")

            with st.spinner("AI가 최적 조합을 심층 탐색 중입니다..."):
                results, b_score, b_cagr, b_mdd = optimizer.optimize(
                    current_params, n_iter, s_date, e_date, opt_targets, trade_at, progress_callback=progress_cb
                )
                # [신규] 결과를 세션 상태에 저장 (버튼 클릭 후 리그림 시 유실 방지)
                st.session_state.opt_results = results
                st.session_state.opt_baseline = {
                    'score': b_score, 'cagr': b_cagr, 'mdd': b_mdd
                }

        # [신규] 세션 상태에 저장된 결과가 있으면 표시
        if 'opt_results' in st.session_state and st.session_state.opt_results:
            results = st.session_state.opt_results
            b = st.session_state.opt_baseline
            
            st.success("✅ 최적화 탐색 완료!")
            st.info(f"📊 기준 성과 (Baseline): Score {b['score']:.2f} (CAGR {b['cagr']:.1f}%, MDD -{b['mdd']:.1f}%)")
            
            # 결과 테이블 표시
            top_results = results[:5]
            sum_data = []
            for idx, r in enumerate(top_results):
                sum_data.append({
                    "Rank": idx + 1,
                    "Score": f"{r['score']:.2f}",
                    "CAGR": f"{r['cagr']:.1f}%",
                    "MDD": f"-{r['mdd']:.1f}%",
                    "Diff": f"{r['score'] - b['score']:+.2f}"
                })
            
            st.table(pd.DataFrame(sum_data).set_index("Rank"))
            
            # 최고 성과 적용 및 저장
            best = top_results[0]
            if best['score'] > b['score']:
                st.balloons()
                st.markdown(f"🏆 **최적 파라미터 발견!** (점수 개선: {best['score'] - b['score']:+.2f})")
                
                # [신규] 저장 및 적용 옵션 UI 개선
                st.write("---")
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
