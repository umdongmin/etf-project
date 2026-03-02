import streamlit as st
import json
import datetime
import pandas as pd
from core.optimizer import StrategyOptimizer

class OptimizerView:
    """AI를 활용한 전략 파라미터 최적화 UI 구성 클래스"""
    
    @staticmethod
    def render(data_dict, fg_df, vix_df, current_params):
        st.markdown("### 🚀 AI 전략 최적화 도구 (Optimizer)")
        st.caption("2:1 성과 점수(CAGR - 0.5 * |MDD|)를 기반으로 최상의 파라미터 조합을 탐색합니다.")
        
        with st.expander("🛠️ 최적화 설정 및 탐색 범위", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                n_iter = st.slider("탐색 횟수 (N)", 50, 1000, 200, step=50, help="탐색 횟수가 많을수록 더 정교한 결과를 얻지만 시간이 오래 걸립니다.")
                target_period = st.radio("최적화 대상 기간", ["전체 (2010~현재)", "최근 5년", "최근 3년"], index=0)
            
            with col2:
                opt_targets = st.multiselect(
                    "최적화 대상 변수",
                    ["매수/매도 시그널 (RSI)", "리밸런싱 비율 (Buy/Sell)", "패닉 가속 (MA/RSI)", "S3 보호 조건 (Gap/Acc)"],
                    default=["매수/매도 시그널 (RSI)", "리밸런싱 비율 (Buy/Sell)"]
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
                progress_bar.progress(current / total)
                if current % 10 == 0:
                    status_text.text(f"⏳ 탐색 중... ({current}/{total}) | 현재 최고 점수: {best:.2f}")

            with st.spinner("AI가 최적 조합을 심층 탐색 중입니다..."):
                results, b_score, b_cagr, b_mdd = optimizer.optimize(
                    current_params, n_iter, s_date, e_date, opt_targets, trade_at, progress_callback=progress_cb
                )

            if results:
                st.success("✅ 최적화 탐색 완료!")
                st.info(f"📊 기준 성과 (Baseline): Score {b_score:.2f} (CAGR {b_cagr:.1f}%, MDD -{b_mdd:.1f}%)")
                
                # 결과 테이블 표시
                top_results = results[:5]
                sum_data = []
                for idx, r in enumerate(top_results):
                    sum_data.append({
                        "Rank": idx + 1,
                        "Score": f"{r['score']:.2f}",
                        "CAGR": f"{r['cagr']:.1f}%",
                        "MDD": f"-{r['mdd']:.1f}%",
                        "Diff": f"{r['score'] - b_score:+.2f}"
                    })
                
                st.table(pd.DataFrame(sum_data).set_index("Rank"))
                
                # 최고 성과 적용 및 저장
                best = top_results[0]
                if best['score'] > b_score:
                    st.balloons()
                    st.markdown(f"🏆 **최적 파라미터 발견!** (점수 개선: {best['score'] - b_score:+.2f})")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✨ 최고 성과 세션에 바로 적용"):
                            st.session_state.current_params = best['cfg']
                            st.success("세션에 적용되었습니다. 상단 탭에서 '분석 결과'를 확인하세요.")
                            st.rerun()
                    with c2:
                        if st.button("💾 최고 성과 파일로 영구 저장 (Optimized)"):
                            with open("strategies/tqqq_strategy_Optimized.json", "w", encoding="utf-8") as f:
                                json.dump(best['cfg'], f, indent=4, ensure_ascii=False)
                            st.success("tqqq_strategy_Optimized.json 파일이 업데이트되었습니다.")
                else:
                    st.warning("기존 베이스라인을 넘는 성과를 발견하지 못했습니다. 탐색 범위를 넓히거나 횟수를 늘려보세요.")
            else:
                st.error("탐색 결과가 없습니다.")
