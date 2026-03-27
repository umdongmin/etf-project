import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # .env 파일에서 환경변수 로드 (FRED_API_KEY 등)

# [중요] Streamlit 설정은 모든 출력 및 다른 라이브러리 실행보다 먼저 호출되어야 함
st.set_page_config(page_title="ETF Golden Strategy", page_icon="📈", layout="wide")

import pandas as pd
import numpy as np
import datetime
import os
import time

# 커스텀 모듈 임포트
from core.data import DataService
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.backtest_view import BacktestView
from ui.history_view import HistoryLabView
from ui.intelligence_view import IntelligenceView
from ui.portfolio_view import PortfolioView
from ui.portfolio_realtime_view import PortfolioRealtimeView
from ui.asset_view import AssetView
from ui.tester_view import TesterView
from ui.simulation_view import SimulationView
from ui.log_view import LogView
from ui.trading_history_view import TradingHistoryView
from utils.metrics import calculate_metrics
from core.defaults import get_default_equity_params, get_default_bond_params, get_default_bond_lev_params, REBALANCE_PRESET_LABELS
from core.engine_kwargs_builder import extract_smart_params
from core.state import AppState
from core.session_keys import SK
# config_loader는 core/state.py 내부에서 호출됨 (AppState.init_equity/bond)

class GoldenStrategyApp:
    def __init__(self):
        self.ticker_map = {
            "QQQ": "나스닥 100 (QQQ)",
            "QLD": "나스닥 2배 (QLD)",
            "TQQQ": "나스닥 3배 (TQQQ)"
        }
        
    def run(self):
        # 사이드바 공통 메뉴
        st.sidebar.title("🚀 Golden Strategy v3.0")
        menu = st.sidebar.radio("메인 메뉴", [
            "🛰️ 실시간 모니터링",
            "💼 자산 관리",
            "📈 포트폴리오 매니저",
            "📜 주식 전략 설정",
            "📉 채권 전략 설정",
            "📋 로그 분석",
            "💰 모의투자 시뮬레이터",
            "🧠 AI 뉴스 분석 리포트",
            "🏦 거래 내역",
        ])
        
        
        # ── 세션 상태 초기화 (AppState 중앙 관리) ─────────────────────────────
        if 'trigger_recalc' not in st.session_state:
            st.session_state.trigger_recalc = False

        # 주식 전략: DB 우선 → defaults.py fallback (최초 1회)
        AppState.init_equity()

        # 채권 전략: DB 우선 → defaults.py fallback (최초 1회)
        _bond_needs_init = not st.session_state.get(SK.INITIAL_BOND_LOAD_V28, False)
        if _bond_needs_init:
            AppState.init_bond()
            TesterView.sync_bond_params_to_widgets(
                AppState.get_bond_params(),
                lev_params=AppState.get_bond_lev_params(),
                key_prefix="bond_"
            )
            st.session_state[SK.TRIGGER_RECALC_BOND] = True
            st.rerun()

        if SK.COMPARISON_LIST not in st.session_state:
            st.session_state[SK.COMPARISON_LIST] = []
            
        cp = AppState.get_equity_params()

        # 데이터 로딩 (모드 전환 감지 → 관련 캐시 자동 무효화)
        _data_mode = st.session_state.get('_data_mode_realtime', False)
        _prev_mode = st.session_state.get('_data_mode_prev', None)
        if _prev_mode is not None and _prev_mode != _data_mode:
            # 모드 전환됨 → 모니터링/자산관리/시그널 캐시 전부 삭제
            for _k in [SK.PORTFOLIO_REALTIME_RESULT, SK.CURRENT_SIGNALS,
                       SK.CURRENT_SIGNALS_PORTFOLIO, SK.LAST_HISTORY_RESULT]:
                st.session_state.pop(_k, None)
            # sig_cache 키들도 삭제 (패턴 기반)
            _to_del = [k for k in st.session_state if isinstance(k, str) and ('sig_cache' in k)]
            for _k in _to_del:
                st.session_state.pop(_k, None)
        st.session_state['_data_mode_prev'] = _data_mode

        if _data_mode:
            with st.spinner('실시간 데이터를 전량 다운로드 중... (1~2분 소요)'):
                data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time = DataService.load_all_data_realtime()
        else:
            with st.spinner('증분 데이터를 가져오는 중...'):
                data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, fetch_time = DataService.load_all_data()

        if not data_dict:
            st.error("데이터 로드 실패")
            st.stop()
            
        # [수정] 날짜 선택 및 파라미터 설정을 로직 최상단으로 이동 (UnboundLocalError 방지)
        start_d, end_d = None, None
        if 'QQQ' in data_dict:
            years = list(range(datetime.datetime.now().year, 2009, -1))
            min_d, max_d = data_dict['QQQ'].index.min().date(), data_dict['QQQ'].index.max().date()
            current_year_jan1 = datetime.date(datetime.datetime.now().year, 1, 1)

            if 'start_input' not in st.session_state:
                st.session_state.start_input = current_year_jan1 if min_d <= current_year_jan1 else min_d
            if 'end_input' not in st.session_state:
                st.session_state.end_input = datetime.date.today()

            def on_year_change():
                if st.session_state.q_year != "직접 선택":
                    yr = int(st.session_state.q_year)
                    st.session_state.start_input = datetime.date(yr, 1, 1)
                    st.session_state.end_input = datetime.date(yr, 12, 31)

            # 데이터 수집 상태 표시
            st.sidebar.divider()
            st.sidebar.subheader("📡 데이터 상태")
            st.sidebar.toggle(
                "실시간 데이터 모드",
                value=st.session_state.get('_data_mode_realtime', False),
                key='_data_mode_realtime',
                help="ON: yfinance 전량 다운로드 (시그널 정확, 느림)\nOFF: 증분 캐시 (빠름, 지표 오차 가능)"
            )
            _mode_label = "🔴 실시간" if st.session_state.get('_data_mode_realtime') else "🟢 증분"
            status_labels = {'FRED': 'FRED 매크로', 'VXN': 'VXN 변동성'}
            all_ok = all(v == 'Success' for v in fetch_status.values())
            data_warnings = fetch_status.get('DataWarnings', [])
            if all_ok and not data_warnings:
                st.sidebar.success(f"모든 데이터 정상 ({_mode_label}) · {fetch_time}")
            else:
                for key, label in status_labels.items():
                    s = fetch_status.get(key, 'Pending')
                    if s == 'Success':
                        st.sidebar.success(f"✅ {label}")
                    elif s == 'Error':
                        st.sidebar.error(f"❌ {label} 실패")
                    else:
                        st.sidebar.warning(f"⏳ {label} 대기중")
                for w in data_warnings:
                    st.sidebar.warning(f"⚠️ {w}")

            st.sidebar.divider()
            st.sidebar.subheader("💸 거래 수수료 설정")
            use_fee = st.sidebar.toggle("수수료 적용", value=st.session_state.get(SK.GLOBAL_USE_FEE, True), key=SK.GLOBAL_USE_FEE)
            if use_fee:
                fee_pct = st.sidebar.number_input("수수료율 (%)", min_value=0.0, max_value=1.0,
                    value=st.session_state.get(SK.GLOBAL_FEE_PCT, 0.10), step=0.01,
                    format="%.3f", help="0.10% = 한국 증권사 해외 ETF 기준 (수수료 0.07% + 슬리피지)", key=SK.GLOBAL_FEE_PCT)
                AppState.set_fee_rate(fee_pct / 100.0)
            else:
                AppState.set_fee_rate(0.0)

            st.sidebar.subheader("📦 주식 수량 설정")
            st.sidebar.toggle("정수 주식 수량 적용", value=st.session_state.get(SK.GLOBAL_INTEGER_SHARES, True), key=SK.GLOBAL_INTEGER_SHARES,
                help="활성화 시 소수점 주식 없이 정수 단위로만 매매 (잔여금은 현금으로 보존)")

            st.sidebar.divider()
            st.sidebar.subheader("📋 기본 전략 설정")

            _eq_strats = StrategyStorage.list_strategies(category='equity')
            _bond_strats = StrategyStorage.list_strategies(category='bond')
            _portfolio_list = StrategyStorage.list_portfolios()

            _cur_default_eq = StrategyStorage.get_default_strategy(category='equity')
            _cur_default_bond = StrategyStorage.get_default_strategy(category='bond')
            _cur_default_pf = StrategyStorage.get_default_portfolio()

            if _eq_strats:
                _eq_fmt = lambda x: f"⭐ {x}" if x == _cur_default_eq else x
                _eq_idx = _eq_strats.index(_cur_default_eq) if _cur_default_eq in _eq_strats else 0
                _sel_eq = st.sidebar.selectbox("주식 전략", _eq_strats, index=_eq_idx, format_func=_eq_fmt, key="sb_default_eq")
                if st.sidebar.button("⭐ 주식 기본 설정", key="btn_set_default_eq", use_container_width=True):
                    StrategyStorage.set_default_strategy(_sel_eq, category='equity')
                    AppState.init_equity(force=True)  # 세션에 즉시 반영
                    st.sidebar.success(f"⭐ '{_sel_eq}' 기본 전략 설정 및 로드 완료")
                    st.rerun()

            if _bond_strats:
                _bond_fmt = lambda x: f"⭐ {x}" if x == _cur_default_bond else x
                _bond_idx = _bond_strats.index(_cur_default_bond) if _cur_default_bond in _bond_strats else 0
                _sel_bond = st.sidebar.selectbox("채권 전략", _bond_strats, index=_bond_idx, format_func=_bond_fmt, key="sb_default_bond")
                if st.sidebar.button("⭐ 채권 기본 설정", key="btn_set_default_bond", use_container_width=True):
                    StrategyStorage.set_default_strategy(_sel_bond, category='bond')
                    AppState.init_bond(force=True)  # 세션에 즉시 반영
                    st.session_state[SK.TRIGGER_RECALC_BOND] = True
                    st.sidebar.success(f"⭐ '{_sel_bond}' 기본 전략 설정 및 로드 완료")
                    st.rerun()

            if _portfolio_list:
                _pf_fmt = lambda x: f"⭐ {x}" if x == _cur_default_pf else x
                _pf_idx = _portfolio_list.index(_cur_default_pf) if _cur_default_pf in _portfolio_list else 0
                _sel_pf = st.sidebar.selectbox("포트폴리오", _portfolio_list, index=_pf_idx, format_func=_pf_fmt, key="sb_default_pf")
                if st.sidebar.button("⭐ 포트폴리오 기본 설정", key="btn_set_default_pf", use_container_width=True):
                    StrategyStorage.set_default_portfolio(_sel_pf)
                    st.session_state.pop(SK.PORTFOLIO_INIT_VER, None)   # 포트폴리오 재초기화 허용
                    st.session_state.pop(SK.PORTFOLIO_RESULT, None)
                    # 각 뷰의 selectbox 위젯 키 초기화 → rerun 후 DB 기본값으로 재선택
                    for _wk in ('p_load_sel', 'signal_portfolio', SK.PORTFOLIO_SELECTED_NAME):
                        st.session_state.pop(_wk, None)
                    st.sidebar.success(f"⭐ '{_sel_pf}' 기본 포트폴리오 설정 완료")
                    st.rerun()

            st.sidebar.divider()
            st.sidebar.subheader("📅 분석 기간 설정")
            st.sidebar.selectbox(
                "연도 선택 (1/1 ~ 12/31 자동설정)", 
                ["직접 선택"] + years,
                key="q_year",
                on_change=on_year_change
            )
            
            start_d = st.sidebar.date_input("시작일", key="start_input")
            end_d = st.sidebar.date_input("종료일", key="end_input")

        # [수정] 중앙화된 extract_smart_params 사용
        smart_params = extract_smart_params(cp)
        cash_ratio = cp['cash_ratio_pct'] / 100.0

        # [신규] 실시간 시그널 프리뷰 처리 인터럽트
        if st.session_state.get('trigger_preview'):
            with st.spinner('실시간 신호를 계산 중입니다... (장마감 전 종가베팅 지원)'):
                # 1. 현재가 및 VIX/VXN 수집
                all_tickers = list(data_dict.keys()) + ['^VIX', '^VXN']
                cur_prices = DataService.fetch_current_prices(all_tickers)
                
                if cur_prices:
                    # 2. 가상 종가 주입 및 지표 재계산 (VIX/VXN 포함)
                    try:
                        v_dict, v_vxn_df, v_fg_df = DataService.inject_virtual_close(data_dict, cur_prices, vxn_df, fg_df)
                    except Exception as ve:
                        st.error(f"데이터 주입 오류: {ve}")
                        v_dict = None

                    if v_dict:
                        try:
                            # 3. 오늘자 신호 예측
                            pred = StrategyEngine.predict_today_signal(
                                v_dict, v_fg_df, v_vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], 
                                cash_ratio, start_d, end_d, cp, cp['trade_at'], smart_params
                            )
                        except Exception as e:
                            pred = None
                            st.error(f"예측 엔진 오류: {e}")
                    else:
                        pred = None
                    
                    if pred:
                        st.session_state.preview_result = pred
                        st.session_state.trigger_preview = False
                        st.toast("✅ 실시간 신호 계산 완료!", icon="🚀")
                    else:
                        st.error("신호 예측에 실패했습니다. (결과 데이터 없음 - 엔진 내부 필터링 확인 필요)")
                else:
                    st.error("실시간 가격을 전혀 가져올 수 없습니다. 네트워크 상태를 확인해 주세요.")
                
            st.session_state.trigger_preview = False
                
        # 메뉴별 렌더링
        # 사이드바 위젯 렌더링 전에도 start_date를 참조할 수 있도록 session_state에서 직접 조회
        _effective_start = start_d or st.session_state.get('start_input') or (
            data_dict['QQQ'].index.min().date() if 'QQQ' in data_dict else None
        )

        if menu == "💼 자산 관리":
            AssetView().render()
        elif menu == "🛰️ 실시간 모니터링":
            st.title("🛰️ 실시간 포트폴리오 모니터링")
            PortfolioView.render_realtime_tab(data_dict, fg_df, vxn_df, macro_df, news_df,
                                              _effective_start,
                                              datetime.date.today())
        elif menu == "📈 포트폴리오 매니저":
            PortfolioView.render(data_dict, fg_df, vxn_df, macro_df, news_df, start_d, end_d)
        elif menu == "📜 주식 전략 설정":
            HistoryLabView.render(data_dict, fg_df, vxn_df, news_df, cp['leverage_asset'], cp['base_asset'], cp['trade_at'], cp, smart_params=smart_params, salt="v2.3 (CrashLogs)")
        elif menu == "📉 채권 전략 설정":
            st.title("📉 Bond Strategy Lab (V7)")
            
            # [신규] 채권 전용 기간 프리셋 추가 (📜 주식 전략 설정과 동일)
            start_b, end_b = TesterView.render_period_selector(key_suffix="bond")
            st.divider()

            # [신규] 탭 시스템 적용 (분석 결과 & 상세 로그 + 전략 설정)
            b_tab1, b_tab2 = st.tabs(["📊 분석 결과 & 상세 로그", "⚙️ 채권 전략 설정"])
            
            with b_tab2:
                # [신규] 채권 전략 파일 관리 (메매 전략 설정과 동일 구조)
                st.subheader("💾 채권 전략 파일 관리")
                col_b1, col_b2 = st.columns(2)
                
                with col_b1:
                    with st.form("save_bond_strat_form"):
                        st.write("📝 현재 채권 전략 저장")
                        new_bond_name = st.text_input("새 전략 이름", value="Bond_V7_Custom")
                        if st.form_submit_button("💾 신규 저장", use_container_width=True):
                            # Layer1(params)와 Layer2(lev_params)를 strategies 테이블에 분리 저장
                            p, lp = TesterView.get_bond_params_from_widgets(key_prefix="bond_")
                            StrategyStorage.save_strategy(new_bond_name, p, category='bond', lev_params=lp)
                            st.success(f"'{new_bond_name}' 저장 완료 (UI 현재값 반영)")
                            st.rerun()

                with col_b2:
                    bond_list = StrategyStorage.list_strategies(category='bond')
                    if bond_list:
                        with st.form("load_bond_strat_form"):
                            st.write("📂 저장된 채권 전략 불러오기")
                            selected_bond = st.selectbox("전략 선택", bond_list)
                            if st.form_submit_button("✨ 적용하기", use_container_width=True):
                                loaded = StrategyStorage.load_strategy(selected_bond)
                                if loaded:
                                    lev_p = loaded.pop('lev_params', None)
                                    loaded.pop('category', None)
                                    AppState.set_bond_params(
                                        loaded,
                                        lev_params=lev_p or get_default_bond_lev_params(),
                                        strategy_name=selected_bond,
                                    )
                                    TesterView.sync_bond_params_to_widgets(
                                        AppState.get_bond_params(),
                                        lev_params=AppState.get_bond_lev_params(),
                                        key_prefix="bond_"
                                    )
                                    st.session_state[SK.TRIGGER_RECALC_BOND] = True
                                    st.success(f"'{selected_bond}' 로드 및 UI 동기화 완료")
                                    st.rerun()
                    else:
                        st.info("저장된 채권 전략이 없습니다.")

                st.divider()
                st.subheader("⚖️ 채권 전략 상세 설정")
                with st.form("bond_settings_form"):
                    # 기존 설정 UI 호출
                    new_bond_params, new_lev_params = TesterView.render_bond_settings(
                        AppState.get_bond_params(),
                        current_lev_params=AppState.get_bond_lev_params(),
                    )
                    st.write("---")
                    bond_submitted = st.form_submit_button("🚀 설정값 확정 및 재시뮬레이션 실행", use_container_width=True, type="primary")

                    if bond_submitted:
                        AppState.set_bond_params(new_bond_params, lev_params=new_lev_params)
                        st.session_state[SK.TRIGGER_RECALC_BOND] = True
                        st.rerun()
            
            with b_tab1:
                if macro_df.empty:
                    st.error("⚠️ 매크로 데이터(FRED)를 불러오지 못했습니다. API 키 설정을 확인하거나 잠시 후 다시 시도해 주세요.")
                    st.info("데이터가 없으면 채권 전략 분석을 수행할 수 없습니다.")
                
                # [수정] 성과 변경 감지 및 자동 계산 로직 (trigger_recalc_bond 또는 해시 변경 시)
                # [수정] 레버리지(lev_params) 설정 변경도 감지 대상에 추가
                _bond_params     = AppState.get_bond_params()
                _bond_lev_params = AppState.get_bond_lev_params()
                current_bond_hash = hash(
                    str(start_b) + str(end_b) +
                    str(_bond_params) +
                    str(_bond_lev_params) +
                    str(len(macro_df)) +
                    str(AppState.get_fee_rate()) +
                    str(AppState.get_integer_shares())
                )
                should_calc_bond = (
                    st.session_state.get(SK.TRIGGER_RECALC_BOND, False) or
                    st.session_state.get(SK.LAST_BOND_HASH) != current_bond_hash
                )

                if should_calc_bond:
                    with st.spinner('채권 분석 데이터 업데이트 중...'):
                        # 채권 전략 실행 (V7)
                        bond_bench = {'TLT': StrategyEngine.run_benchmark(data_dict, 'TLT', start_b, end_b)}
                        bond_history, b_closed, _ = StrategyEngine.run_bond_strategy(
                            data_dict, macro_df, start_b, end_b,
                            params=_bond_params,
                            lev_params=_bond_lev_params,
                            fee_rate=AppState.get_fee_rate(),
                            integer_shares=AppState.get_integer_shares(),
                        )

                        curr_res = st.session_state.get(SK.LAST_BT_RESULT, {})
                        curr_res['bond'] = {'history': bond_history, 'closed_trades': b_closed, 'benchmarks': bond_bench}
                        st.session_state[SK.LAST_BT_RESULT] = curr_res
                        st.session_state[SK.LAST_BOND_HASH] = current_bond_hash
                        st.session_state[SK.TRIGGER_RECALC_BOND] = False


                res = st.session_state.get(SK.LAST_BT_RESULT, {}).get('bond')
                if res and res['history'] is not None and not res['history'].empty:
                    # [신규] 주식 전략 설정과 동일한 요약 정보 추가
                    bh_bond = res['benchmarks'].get('TLT', pd.DataFrame())
                    empty_df = pd.DataFrame()
                    
                    st.subheader("📊 전체 기간 요약")
                    HistoryLabView.render_overall_summary(res['history'], bh_bond, empty_df, empty_df, b_names=("TLT", "", ""))
                    
                    st.divider()
                    HistoryLabView.render_yearly_table(res['history'], bh_bond, empty_df, empty_df, b_names=("TLT", "", ""))
 
                    st.divider()
                    # 상세 분석 결과 렌더링 (차트 및 로그) - 채권 전용 필터 적용
                    BacktestView.render_results(res['history'], res['benchmarks'], res['closed_trades'], 'TLT', 'TLT', smart_params=None, is_bond=True)
                    st.info("💡 상세 거래 로그는 '📋 로그 분석' 메뉴에서 확인하세요.")
                else:
                    st.info("시뮬레이션 데이터를 준비 중입니다... (파라미터를 변경해 보세요)")
                    # [신규] 진단 정보 표시 (결과 미출력 원인 파악용)
                    with st.expander("🔍 분석 진단 정보 (결과가 나오지 않을 경우 확인)", expanded=False):
                        st.write(f"- 분석 기간: {start_b} ~ {end_b}")
                        st.write(f"- 매크로 데이터 수: {len(macro_df)} 행")
                        if not macro_df.empty:
                            st.write("- 매크로 데이터 최신 날짜:", macro_df.index.max())
                            st.write("- 주요 지표 결측치: ", macro_df.isna().sum().to_dict())
                        tlt_len = len(data_dict.get('TLT', []))
                        st.write(f"- TLT 데이터 수: {tlt_len} 행")
                        if tlt_len == 0:
                            st.warning("⚠️ TLT 데이터 없음 — yfinance Rate Limit 가능성. 아래 버튼으로 캐시를 초기화하세요.")
                            if st.button("🔄 데이터 캐시 초기화 후 재시도", key="clear_cache_bond"):
                                DataService.load_all_data.clear()
                                st.rerun()
                        st.write(f"- 최적화 파라미터: {AppState.get_bond_params()}")

        elif menu == "📋 로그 분석":
            LogView.render()
        elif menu == "💰 모의투자 시뮬레이터":
            SimulationView.render(data_dict, fg_df, vxn_df, macro_df, news_df)
        elif menu == "🧠 AI 뉴스 분석 리포트":
            IntelligenceView.render()
        elif menu == "🏦 거래 내역":
            TradingHistoryView.render()

if __name__ == "__main__":
    try:
        app = GoldenStrategyApp()
        app.run()
    except Exception as e:
        st.error(f"❌ 애플리케이션 실행 중 오류가 발생했습니다: {e}")
        st.exception(e)
