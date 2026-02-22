import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.tester_view import TesterView

class HistoryLabView:
    """역사적 마켓 랩 화면 UI 구성 클래스"""
    
    EVENTS = {
        2010: "플래시 크래시, 유럽 재정 위기 시작",
        2011: "미국 국가신용등급 강등, 그리스 디폴트 우려",
        2012: "유로존 붕괴 위기 고조, 페이스북 IPO",
        2013: "테이퍼 탠트럼 (긴축 발작), 양적완화 축소 논의",
        2014: "유가 급락, 알리바바 역대 최대 IPO",
        2015: "중국 증시 폭락, 미국 7년 만의 금리 인상",
        2016: "브렉시트 투표, 트럼프 대통령 당선",
        2017: "FAANG 주도 기술주 랠리, 비트코인 열풍",
        2018: "미중 무역 전쟁, 연준의 지속적 금리 인상",
        2019: "장단기 금리 역전, 레포 금리 급등",
        2020: "코로나19 팬데믹, 사상 최대 유동성 공급",
        2021: "포스트 팬데믹 인플레이션 시작, 공급망 병목",
        2022: "40년 만의 고물가, 공격적 금리 인상 (Bear Market)",
        2023: "SVB 은행 파산 위기, 생성형 AI (ChatGPT) 열풍",
        2024: "AI 반도체 잭팟, 금리 인하 기대감 형성"
    }

    @staticmethod
    def render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, params, smart_params=None, salt="v2.3 (CrashLogs)"):
        st.header("📜 매매 전략 설정 및 역사적 분석")
        st.caption("전략을 저장하거나 불러오고, 2010년부터 현재까지의 성과를 분석합니다. [v2.3]")

        st.divider()
        st.subheader("💾 전략 파일 관리")
        cp = st.session_state.current_params

        col1, col2 = st.columns(2)
        with col1:
            with st.form("save_strat_form_main"):
                st.write("📝 현재 전략 저장")
                new_strat_name = st.text_input("새 전략 이름 입력", key="new_strat_save_name_main")
                save_submitted = st.form_submit_button("💾 내 컴퓨터에 신규 저장", use_container_width=True)
                if save_submitted:
                    if new_strat_name:
                        StrategyStorage.save_strategy(new_strat_name, cp)
                        st.session_state.loaded_strat_name = new_strat_name
                        st.success(f"'{new_strat_name}' 저장 완료")
                        st.rerun()
                    else:
                        st.warning("전략 이름을 입력해 주세요.")
            
            if st.session_state.get('loaded_strat_name'):
                st.write(f"🔄 기존 전략: **{st.session_state.loaded_strat_name}**")
                if st.button(f"🚀 '{st.session_state.loaded_strat_name}' 업데이트 (덮어쓰기)", key="update_btn_main", use_container_width=True):
                    StrategyStorage.save_strategy(st.session_state.loaded_strat_name, cp)
                    st.success(f"'{st.session_state.loaded_strat_name}' 업데이트 완료!")

        with col2:
            stored_strats = StrategyStorage.list_strategies()
            if stored_strats:
                with st.form("load_strat_form_main"):
                    st.write("📂 저장된 전략 불러오기")
                    target_strat = st.selectbox("전략 선택 (목록)", ["선택 안 함"] + stored_strats, key="load_sel_main")
                    load_submitted = st.form_submit_button("✨ 선택한 전략 적용하기", use_container_width=True)
                    if load_submitted and target_strat != "선택 안 함":
                        config = StrategyStorage.load_strategy(target_strat)
                        if config:
                            # 1. 메인 파라미터 업데이트
                            st.session_state.current_params.update(config)
                            
                            # 2. UI 위젯 상태 강제 동기화 (Widget Keys)
                            # 리밸런싱 및 기본 설정 매핑
                            key_map = {
                                'buy_reb_up': 'r_b_u', 'buy_reb_down': 'r_b_d', 
                                'sell_reb_up': 'r_s_u', 'sell_reb_down': 'r_s_d',
                                'cash_ratio_pct': 'conf_cash', 'base_asset': 'conf_base',
                                'leverage_asset': 'conf_lev', 'trade_at': 'conf_trade_at'
                            }
                            for p_key, w_key in key_map.items():
                                if p_key in config:
                                    val = config[p_key]
                                    # 리밸런싱 비율들은 소수점(0.01)으로 저장되므로 UI(1%)를 위해 100을 곱함
                                    if p_key.endswith('_up') or p_key.endswith('_down'):
                                        st.session_state[w_key] = float(val) * 100.0
                                    # 현금 비중(cash_ratio_pct)은 이미 퍼센트(10)로 저장됨
                                    elif p_key == 'cash_ratio_pct':
                                        st.session_state[w_key] = float(val)
                                    else:
                                        st.session_state[w_key] = val
                            
                            # 시그널 조건 동기화
                            for i, bp in enumerate(config.get('buy_signals', []), 1):
                                st.session_state[f"b_rsi_{i}"] = bp.get('rsi_val', 0)
                                st.session_state[f"b_cross_{i}"] = bp.get('rsi_cross', False)
                                st.session_state[f"b_inc_{i}"] = bp.get('rsi_inc', False)
                                st.session_state[f"b_use_adx_{i}"] = bp.get('use_adx', False)
                                st.session_state[f"b_adx_op_{i}"] = bp.get('adx_op', "<=")
                                st.session_state[f"b_adx_{i}"] = bp.get('adx_val', 40)
                                st.session_state[f"b_m_inc_{i}"] = bp.get('macd_inc', False)
                                st.session_state[f"b_m_below_{i}"] = bp.get('macd_signal_below', False)
                                st.session_state[f"b_m_golden_{i}"] = bp.get('macd_golden', False)
                                st.session_state[f"b_bb_low_{i}"] = bp.get('bb_lower', False)

                            for i, sp in enumerate(config.get('sell_signals', []), 1):
                                st.session_state[f"s_rsi_{i}"] = sp.get('rsi_val', 0)
                                st.session_state[f"s_dead_{i}"] = sp.get('rsi_dead', False)
                                st.session_state[f"s_dec_{i}"] = sp.get('rsi_dec', False)
                                st.session_state[f"s_di_{i}"] = sp.get('di_minus_above', False)
                                st.session_state[f"s_m_dec_{i}"] = sp.get('macd_dec', False)
                                st.session_state[f"s_m_above_{i}"] = sp.get('macd_signal_above', False)
                                st.session_state[f"s_m_dead_{i}"] = sp.get('macd_dead', False)
                                st.session_state[f"s_bb_up_{i}"] = sp.get('bb_upper', False)

                            st.session_state.loaded_strat_name = target_strat
                            st.success(f"'{target_strat}' 모든 설정 및 UI 로드 완료!")
                            st.rerun()
            else:
                st.info("저장된 전략이 없습니다.")
        
        start_date = datetime.date(2010, 1, 1)
        end_date = datetime.date.today()
        
        with st.spinner('전 기간 역사적 데이터 시뮬레이션 중...'):
            golden_history, closed_trades = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, params['cash_ratio_pct']/100.0, start_date, end_date, params, trade_at, smart_params=smart_params, salt=salt)
            
            st.session_state.last_golden_result = {
                'history': golden_history.copy(),
                'base': base_asset,
                'lev': leverage_asset
            }
            
            semi_group = ['SOXX', 'USD', 'SOXL']
            is_semi = base_asset in semi_group or leverage_asset in semi_group
            b1, b2, b3 = ("SOXX", "USD", "SOXL") if is_semi else ("QQQ", "QLD", "TQQQ")
            
            bh_1 = StrategyEngine.run_benchmark(data_dict, b1, start_date, end_date)
            bh_2 = StrategyEngine.run_benchmark(data_dict, b2, start_date, end_date)
            bh_3 = StrategyEngine.run_benchmark(data_dict, b3, start_date, end_date)
            
        if golden_history.empty:
            st.warning("분석할 데이터가 충분하지 않습니다.")
            return

        st.divider()
        st.subheader("⚙️ 전략 매매 조건 설정")
        st.caption("여기서 설정을 변경한 후 아래 버튼을 클릭해야 모든 분석 결과에 반영됩니다.")
        with st.form("strategy_config_form"):
            ticker_map = {
                "QQQ": "나스닥 100 (QQQ)", "QLD": "나스닥 2배 (QLD)", "TQQQ": "나스닥 3배 (TQQQ)",
                "SOXX": "반도체 1배 (SOXX)", "USD": "반도체 2배 (USD)", "SOXL": "반도체 3배 (SOXL)",
                "TMF": "채권 3배 (TMF)", "TLT": "채권 1배 (TLT)"
            }
            new_params = TesterView.render_config(st.session_state.current_params, ticker_map=ticker_map)
            col_btn1, col_btn2 = st.columns(2)
            if col_btn1.form_submit_button("🚀 설정 반영 및 재시뮬레이션", use_container_width=True):
                st.session_state.current_params.update(new_params)
                st.rerun()
            
            if col_btn2.form_submit_button("🧼 시스템 데이터 완전 초기화 (캐시 삭제)", use_container_width=True):
                st.cache_data.clear()
                st.success("캐시가 성공적으로 삭제되었습니다.")
                st.rerun()

        st.divider()
        st.subheader("📊 전체 기간 요약")
        HistoryLabView.render_overall_summary(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3))
        
        st.divider()
        HistoryLabView.render_yearly_table(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3), smart_params=smart_params)

        # [신규] 레버리지 100% (S3) 구간 로그 추가
        st.divider()
        with st.expander("🚩 레버리지 100% (S3) 진입 구간 기록", expanded=False):
            st.caption("전략이 TQQQ/SOXL 등 레버리지 자산에 100% 집중되었던 상시 노출 구간입니다.")
            
            # 레버리지 100% (S3) 구간 추출 로직 정교화 (Double-Lock Safety)
            if 'Stage' in golden_history.columns and 'Lev_Weight' in golden_history.columns:
                # Stage 3이면서 실제 레버리지 비중이 높은 날(85% 초과)만 S3 구간으로 추출합니다.
                # 이는 엔진 규칙 개편 후에도 남아있을 수 있는 초기 단계(S1/S2)와의 혼선을 완벽히 차단합니다.
                is_100 = (golden_history['Stage'].astype(float).fillna(0).round().astype(int) == 3) & (golden_history['Lev_Weight'] > 0.85)
            elif 'Stage' in golden_history.columns:
                is_100 = (golden_history['Stage'] == 3)
            elif 'Lev_Weight' in golden_history.columns:
                is_100 = golden_history['Lev_Weight'] > 0.95
            else:
                lev_assets = ['QLD', 'TQQQ', 'USD', 'SOXL', 'TMF']
                weight_cols = [c for c in golden_history.columns if c.endswith('_Weight') and any(la in c for la in lev_assets)]
                is_100 = golden_history[weight_cols].sum(axis=1) > 0.95 if weight_cols else pd.Series(False, index=golden_history.index)
            
            periods = []
            start_p = None
            for date, val in is_100.items():
                if val:
                    if start_p is None: start_p = date
                else:
                    if start_p is not None:
                        periods.append((start_p, date)) # 비중이 줄어든 당일을 종료일로 설정
                        start_p = None
            if start_p is not None: periods.append((start_p, golden_history.index[-1]))
                
            if periods:
                p_data = []
                for s, e in reversed(periods): # 최신순
                    dur = (e - s).days + 1
                    # 해당 구간 수익률 계산
                    try:
                        v_start = golden_history.loc[s, 'Value']
                        v_end = golden_history.loc[e, 'Value']
                        ret = (v_end / v_start - 1) * 100
                        ret_str = f"{ret:+.1f}%"
                    except:
                        ret_str = "-"
                    
                    p_data.append({
                        "시작일": s.strftime('%Y-%m-%d'), 
                        "종료일": e.strftime('%Y-%m-%d'), 
                        "유지기간": f"{dur}일",
                        "구간 수익률": ret_str,
                        "Stage": f"S{int(golden_history.loc[s, 'Stage'])}" if 'Stage' in golden_history.columns else "-",
                        "최고 비중": f"{golden_history.loc[s:e, 'Lev_Weight'].max()*100:.1f}%" if 'Lev_Weight' in golden_history.columns else "-"
                    })
                
                df_p = pd.DataFrame(p_data)
                
                def style_p_ret(val):
                    if val == "-": return ""
                    try:
                        num = float(val.replace('%','').replace('+',''))
                        return f'color: {"#00c853" if num >= 0 else "#ff1744"}; font-weight: bold;'
                    except: return ''

                st.dataframe(df_p.style.applymap(style_p_ret, subset=["구간 수익률"]), use_container_width=True)
            else:
                st.info("레버리지 100% 구간이 없습니다.")

        # [신규] 급락 리스크 진단 로그 (분석용)
        st.divider()
        st.subheader("⚠️ 급락 리스크 진단 로그 (분석용)")
        st.caption("실제 로그와 별개로, 시장의 급격한 하락 신호가 포착된 날들을 나열합니다. (위험 관리 아이디어 검증용)")
        
        col_crash1, col_crash2, col_crash3 = st.columns(3)
        
        with col_crash1:
            with st.expander("🚩 급격한 하락 발생 기록 (-7% 이하)", expanded=False):
                st.caption("최근 2거래일간 지수(Base)가 -7% 이상 급급락한 날짜들입니다.")
                if 'Drop_Acc' in golden_history.columns:
                    crash_days = golden_history[golden_history['Drop_Acc'] <= -7.0].copy()
                    if not crash_days.empty:
                        crash_display = crash_days[['Drop_Acc', 'Asset', 'Stage']].copy()
                        crash_display.index = crash_display.index.strftime('%Y-%m-%d')
                        crash_display.columns = ['2일 누적', '보유자산', 'Stage']
                        st.dataframe(crash_display.style.format({'2일 누적': '{:+.2f}%'}), use_container_width=True)
                    else:
                        st.info("해당되는 급격한 하락 구간이 없습니다.")
                else:
                    st.warning("데이터가 부족하여 분석할 수 없습니다.")
        
        with col_crash2:
            with st.expander("🚩 패닉 시가 갭 발생 기록 (-3% 이하)", expanded=False):
                st.caption("전일 종가 대비 시가(Open)가 -3% 이상 낮게 시작한 날짜들입니다.")
                if 'Gap_Down' in golden_history.columns:
                    gap_days = golden_history[golden_history['Gap_Down'] <= -3.0].copy()
                    if not gap_days.empty:
                        gap_display = gap_days[['Gap_Down', 'Asset', 'Stage']].copy()
                        gap_display.index = gap_display.index.strftime('%Y-%m-%d')
                        gap_display.columns = ['시가 갭', '보유자산', 'Stage']
                        st.dataframe(gap_display.style.format({'시가 갭': '{:+.2f}%'}), use_container_width=True)
                    else:
                        st.info("해당되는 시가 갭 구간이 없습니다.")
                else:
                    st.warning("데이터가 부족하여 분석할 수 없습니다.")

        with col_crash3:
            with st.expander("🚩 S3 전용 복합 보호 작동 기록", expanded=False):
                st.caption("레버리지 100% 상태에서 설정된 복합 조건(일일/VIX/시가갭/가속도)에 의해 매도 보호가 작동한 날짜들입니다.")
                if 'S3_Drop_Limit' in golden_history.columns:
                    s3_days = golden_history[golden_history['S3_Drop_Limit'] != 0].copy()
                    if not s3_days.empty:
                        # [수정] 상세 정보가 있으면 상세 정보를, 없으면 기존 수치 표시
                        if 'S3_Detail' in s3_days.columns:
                            s3_display = s3_days[['S3_Detail', 'Asset', 'Stage']].copy()
                            s3_display.columns = ['상세 정보 (D/V/G/A)', '보유자산', 'Stage']
                        else:
                            s3_display = s3_days[['S3_Drop_Limit', 'Asset', 'Stage']].copy()
                            s3_display.columns = ['당일 변동', '보유자산', 'Stage']
                            
                        s3_display.index = s3_display.index.strftime('%Y-%m-%d')
                        st.dataframe(s3_display, use_container_width=True)
                    else:
                        st.info("S3 보호 로직이 작동한 기록이 없습니다.")
                else:
                    st.warning("데이터가 부족하거나 해당 기능이 비활성화 상태입니다.")

    @staticmethod
    def render_yearly_table(s_h, b1_h, b2_h, b3_h, b_names=("QQQ", "QLD", "TQQQ"), smart_params=None, selected_comparisons=[]):
        st.subheader(f"연도별 성과 및 리스크 분석 (vs {b_names[0]}군)")
        
        def calculate_mdd(series):
            s = series.dropna()
            if s.empty: return 0
            roll_max = s.cummax()
            dd = (s - roll_max) / roll_max
            return dd.min()

        years = sorted(s_h.index.year.unique())
        data = []
        
        for y in years:
            def safe_get_yearly(df, year):
                if df.empty or 'Value' not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
                    return pd.Series(dtype='float64')
                try: return df[df.index.year == year]['Value']
                except: return pd.Series(dtype='float64')

            s_y = safe_get_yearly(s_h, y)
            q_y = safe_get_yearly(b1_h, y)
            
            def get_ret(ser):
                s = ser.dropna()
                if s.empty or len(s) < 2: return 0
                return (s.iloc[-1] / s.iloc[0] - 1) * 100

            s_ret = get_ret(s_y)
            q_ret = get_ret(q_y)
            s_mdd = calculate_mdd(s_y) * 100
            q_mdd = calculate_mdd(q_y) * 100
            
            panic_days, safety_count = 0, 0
            if 'Trade_Label' in s_h.columns:
                s_y_full = s_h[s_h.index.year == y]
                panic_ma = smart_params.get('panic_ma', 200) if smart_params else 200
                panic_days = s_y_full['Trade_Label'].str.contains(f"지연\\(MA{panic_ma}", na=False).sum()
                safety_count = s_y_full['Trade_Label'].str.contains("대피", na=False).sum()
            
            event = HistoryLabView.EVENTS.get(y, "-")
            row = {"연도": f"{y}년", "내 전략 수익": f"{s_ret:+.1f}%", f"{b_names[0]} 수익": f"{q_ret:+.1f}%", "내 전략 MDD": f"{s_mdd:.1f}%", f"{b_names[0]} MDD": f"{q_mdd:.1f}%"}
            
            for target in selected_comparisons:
                t_hist = target['history'].copy()
                if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
                t_hist.index = pd.to_datetime(t_hist.index).normalize()
                t_hist['Value'] = pd.to_numeric(t_hist['Value'], errors='coerce')
                t_y_data = safe_get_yearly(t_hist, y)
                t_ret_val, t_mdd_val = get_ret(t_y_data), calculate_mdd(t_y_data) * 100
                row[f"{target['name']} 수익"] = f"{t_ret_val:+.1f}%"
                row[f"{target['name']} MDD"] = f"{t_mdd_val:.1f}%"

            row.update({"방어(일)": f"{panic_days}일" if panic_days > 0 else "-", "Safety(회)": f"{safety_count}회" if safety_count > 0 else "-", "주요 이벤트": event})
            data.append(row)
            
        df = pd.DataFrame(data).set_index("연도")
        
        def style_returns(val):
            try:
                num = float(val.replace('%','').replace('+',''))
                return f'color: {"#00c853" if num >= 0 else "#ff1744"}; font-weight: bold;'
            except: return ''

        def style_mdd(val):
            try:
                num = float(val.replace('%',''))
                alpha = min(abs(num)/50.0, 1.0) * 0.4
                return f'background-color: rgba(255, 23, 68, {alpha});'
            except: return ''

        st.dataframe(df.style.applymap(style_returns, subset=[c for c in df.columns if "수익" in c])\
                            .applymap(style_mdd, subset=[c for c in df.columns if "MDD" in c]), use_container_width=True, height=520)

    @staticmethod
    def render_overall_summary(s_h, b1_h, b2_h, b3_h, b_names=("QQQ", "QLD", "TQQQ"), selected_comparisons=[]):
        def calculate_mdd_ui(series):
            s = series.dropna()
            if s.empty: return 0, 0, 0, 0, 0, 0, 0
            roll_max = s.cummax()
            dd = (s - roll_max) / roll_max
            
            # 하락 구간 그룹화
            is_in_dd = s < roll_max
            dd_groups = (is_in_dd != is_in_dd.shift()).cumsum()
            
            # 평균 회복 일수 계산
            dd_durations = is_in_dd.groupby(dd_groups).sum()
            avg_rec = dd_durations[dd_durations > 0].mean() if not dd_durations[dd_durations > 0].empty else 0
            
            # 하락 이벤트별 최저점(Depth) 추출
            event_depths = dd.groupby(dd_groups).min()
            event_depths = event_depths[event_depths < 0]
            avg_dd_depth = event_depths.mean() if not event_depths.empty else 0
            med_dd_depth = event_depths.median() if not event_depths.empty else 0
            
            # 연평균 MDD 계산
            yr_mdds = []
            for y in s.index.year.unique():
                y_s = s[s.index.year == y]
                if len(y_s) > 1:
                    y_rm = y_s.cummax()
                    y_mdd = ((y_s - y_rm) / y_rm).min()
                    yr_mdds.append(y_mdd)
            avg_ann_mdd = np.mean(yr_mdds) if yr_mdds else 0
            
            # 고통 가성비 (복구 속도: |평균 하락폭| / 평균 회복 일수)
            # 1%를 회복하는 데 며칠이 걸리는지가 아니라, 하루에 몇 %를 회복하는지(가성비)로 계산
            pain_efficiency = (abs(avg_dd_depth) * 100) / avg_rec if avg_rec > 0 else 0
            
            return dd.min(), np.sqrt((dd**2).mean()) * 100, avg_ann_mdd, avg_rec, avg_dd_depth, med_dd_depth, pain_efficiency

        def get_summary(ser, name):
            if ser is None or ser.empty: return None
            total_ret = (ser.iloc[-1] / 10000.0 - 1) * 100
            years_count = (ser.index[-1] - ser.index[0]).days / 365.25
            cagr = ((ser.iloc[-1] / 10000.0) ** (1/years_count) - 1) * 100 if years_count > 0 else 0
            mdd_val, ui_val, avg_ann_mdd, avg_rec, avg_dd, med_dd, pain_eff = calculate_mdd_ui(ser)
            return {
                "구분": name, 
                "누적 수익률": f"{total_ret:+.1f}%", 
                "연평균(CAGR)": f"{cagr:.1f}%", 
                "최대낙폭(MDD)": f"{mdd_val*100:.1f}%",
                "연평균 MDD": f"{avg_ann_mdd*100:.1f}%",
                "평균 하락폭": f"{avg_dd*100:.1f}%",
                "중간 하락폭": f"{med_dd*100:.1f}%",
                "평균 회복일수": f"{avg_rec:.1f}일",
                "고통 가성비": f"{pain_eff:.2f}",
                "궤양 지수(UI)": f"{ui_val:.1f}%"
            }

        summary_data = [get_summary(s_h['Value'], "내 전략 🔥"), get_summary(b1_h['Value'] if not b1_h.empty else None, f"{b_names[0]}(1x)")]
        for target in selected_comparisons:
            t_hist = target['history'].copy()
            if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
            t_hist.index = pd.to_datetime(t_hist.index).normalize()
            t_hist['Value'] = pd.to_numeric(t_hist['Value'], errors='coerce')
            s_item = get_summary(t_hist['Value'], target['name'])
            if s_item: summary_data.append(s_item)
        
        st.dataframe(pd.DataFrame([s for s in summary_data if s]).set_index("구분"), use_container_width=True)

    @staticmethod
    def render_election_cycle(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"미 대통령 임기 주기별 평균 성과 (Strategy vs {b_name})")
        
        def get_cycle_data(hist, col_name):
            if hist.empty or 'Value' not in hist.columns: return pd.Series(dtype='float64')
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['YearType'] = [(year - 2009) % 4 + 1 for year in df.index.year]
                return df.groupby('YearType')[col_name].mean() * 100
            except: return pd.Series(dtype='float64')

        s_avg = get_cycle_data(strategy_hist, 'Strategy').reindex([1, 2, 3, 4], fill_value=0.0)
        b_avg = get_cycle_data(bench_hist, b_name).reindex([1, 2, 3, 4], fill_value=0.0)
        plot_df = pd.DataFrame({'내 전략': s_avg, f'{b_name}': b_avg}).fillna(0)
        plot_df.index = [f"{i}년차" for i in [1,2,3,4]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['내 전략'], name='내 전략', marker_color='#00c853', text=plot_df['내 전략'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df[f'{b_name}'], name=f'{b_name}', marker_color='#66b3ff', text=plot_df[f'{b_name}'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
        fig.update_layout(title="Strategy vs Benchmark by Election Cycle Year", template='plotly_white', barmode='group', height=450)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_midterm_analysis(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"중간선거 주기 집중 분석 (Strategy vs {b_name})")
        def process_midterm(hist, col_name):
            if hist.empty or 'Value' not in hist.columns: return pd.DataFrame(columns=[col_name, 'IsMidterm'])
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['IsMidterm'] = [((y - 2009) % 4 + 1) == 2 for y in df.index.year]
                return df
            except: return pd.DataFrame(columns=[col_name, 'IsMidterm'])

        s_df, b_df = process_midterm(strategy_hist, 'Strategy'), process_midterm(bench_hist, b_name)
        s_avg, b_avg = s_df.groupby('IsMidterm')['Strategy'].mean() * 100, b_df.groupby('IsMidterm')[b_name].mean() * 100
        avg_comp = pd.DataFrame({'내 전략': s_avg, f'시장({b_name})': b_avg}).reindex([False, True], fill_value=0.0)
        avg_comp.index = ["비중간선거", "중간선거"]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=avg_comp.index, y=avg_comp['내 전략'], name='내 전략', marker_color='#00c853', text=avg_comp['내 전략'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.add_trace(go.Bar(x=avg_comp.index, y=avg_comp[f'시장({b_name})'], name=f'시장({b_name})', marker_color='#66b3ff', text=avg_comp[f'시장({b_name})'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.update_layout(template='plotly_white', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            s_mid, b_mid = s_df[s_df['IsMidterm']].copy(), b_df[b_df['IsMidterm']].copy()
            s_mid['Year'], b_mid['Year'] = s_mid.index.year, b_mid.index.year
            mid_yearly = pd.merge(s_mid[['Year', 'Strategy']], b_mid[['Year', b_name]], on='Year').set_index('Year') * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mid_yearly.index.map(str), y=mid_yearly['Strategy'], name='내 전략', marker_color='#00c853', text=mid_yearly['Strategy'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.add_trace(go.Bar(x=mid_yearly.index.map(str), y=mid_yearly[b_name], name=f'시장({b_name})', marker_color='#66b3ff', text=mid_yearly[b_name].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.update_layout(template='plotly_white', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_seasonality(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"지난 15년간의 월별 평균 수익률 & 승률 (vs {b_name})")
        def get_monthly_stats(hist):
            full_m = range(1, 13)
            if hist.empty or 'Value' not in hist.columns: return pd.Series(index=full_m, data=0.0), pd.Series(index=full_m, data=0.0)
            try:
                rets = hist['Value'].pct_change().dropna()
                df_rets = rets.to_frame(name='Return')
                df_rets['Month'] = df_rets.index.month
                avg = df_rets.groupby('Month')['Return'].mean() * 100 * 21
                win = df_rets.groupby('Month')['Return'].apply(lambda x: (x > 0).mean()) * 100
                return avg.reindex(full_m, fill_value=0.0), win.reindex(full_m, fill_value=0.0)
            except: return pd.Series(index=full_m, data=0.0), pd.Series(index=full_m, data=0.0)

        s_avg, s_win = get_monthly_stats(strategy_hist)
        b_avg, _ = get_monthly_stats(bench_hist)
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=list(range(1,13)), y=s_avg, name='내 전략 (Avg Ret)', marker_color='#00c853', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Bar(x=list(range(1,13)), y=b_avg, name=f'{b_name} (Avg Ret)', marker_color='#66b3ff', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=list(range(1,13)), y=s_win, name='내 전략 승률', line=dict(color='#ff1744', width=3)), secondary_y=True)
        fig.update_layout(title="Seasonality Analysis", template='plotly_white', barmode='group', height=500, xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=[f"{m}월" for m in range(1,13)]))
        fig.update_yaxes(title_text="Avg Monthly Return (%)", secondary_y=False)
        fig.update_yaxes(title_text="Strategy Win Rate (%)", secondary_y=True, range=[40, 80])
        st.plotly_chart(fig, use_container_width=True)
