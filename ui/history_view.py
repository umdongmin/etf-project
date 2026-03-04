import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.tester_view import TesterView
from ui.optimizer_view import OptimizerView

# [신규] 시뮬레이션 캐싱 래퍼 (성능 최적화 및 안정성 확보)
def deep_tuple(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, deep_tuple(v)) for k, v in obj.items()))
    if isinstance(obj, list):
        return tuple(deep_tuple(i) for i in obj)
    return obj

@st.cache_data
def get_cached_strategy_result(_data_dict, _fg_df, _vix_df, lev, base, cash, s_d, e_d, params_tuple, t_at, smart_tuple, salt):
    # 튜플을 다시 딕셔너리로 복구하여 엔진 실행 (재귀적으로 처리)
    def to_dict(tup):
        if not isinstance(tup, tuple): return tup
        # 모든 항목이 (문자열, 값) 형태의 튜플인 경우에만 딕셔너리로 간주
        if len(tup) > 0 and all(isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) for v in tup):
            return {k: to_dict(v) for k, v in tup}
        # 그 외에는 리스트로 복구
        return [to_dict(v) for v in tup]

    _p = to_dict(params_tuple)
    _s = to_dict(smart_tuple)
    # [수정] 엔진이 3개(history, closed_trades, sell4_events)의 값을 반환하므로 전체 반환
    return StrategyEngine.run_golden_strategy(_data_dict, _fg_df, _vix_df, lev, base, cash, s_d, e_d, _p, t_at, smart_params=_s, salt=salt)

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
                            # [신규] 로드 전 세션 상태 초기화 (이전 전략 오염 방지)
                            # app.py의 초기값을 복사하거나, 최소한 주요 리스트들은 비워야 함
                            # 여기서는 app.py의 기본 구조를 세션에 다시 설정함
                            st.session_state.current_params = {
                                'buy_signals': [], 'sell_signals': [], 's3_protection': [], 'panic_buy_signals': [],
                                'buy_reb_up': 0.018, 'buy_reb_down': -1.0, 'sell_reb_up': 0.03, 'sell_reb_down': -0.035,
                                'base_asset': 'QQQ', 'leverage_asset': 'TQQQ', 'cash_ratio_pct': 0.0, 'trade_at': '종가',
                                'use_fixed_reb': True, 'use_atr_reb': True,
                                'atr_mult_buy_up': 10.0, 'atr_mult_buy_down': 3.0, 'atr_mult_sell': 10.0,
                                'atr_period_buy': 20, 'atr_period_sell': 20,
                                'use_panic': True, 'panic_ma': 200, 
                                'use_vxn_safety': False, 'vxn_exit': 31,
                                'use_rsi_turbo': False, 'rsi_turbo': 31
                            }

                            # [수정] 중첩 데이터 평활화 (Flatten) - overwrite (update) 사용
                            if 'params' in config and isinstance(config['params'], dict):
                                inner_p = config.pop('params')
                                config.update(inner_p) # Overwrite top-level with specific params
                            if 'smart_params' in config and isinstance(config['smart_params'], dict):
                                inner_s = config.pop('smart_params')
                                config.update(inner_s) # Overwrite top-level with smart params
                            
                            # [신규] 레거시 키 호환성 (VIX -> VXN)
                            legacy_mapping = {
                                'use_vix_safety': 'use_vxn_safety',
                                'vix_exit': 'vxn_exit'
                            }
                            for old_k, new_k in legacy_mapping.items():
                                if old_k in config and new_k not in config:
                                    config[new_k] = config[old_k]

                            # 1. 메인 파라미터 업데이트
                            st.session_state.current_params.update(config)
                            
                            # 2. UI 위젯 상태 강제 동기화 (Widget Keys)
                            key_map = {
                                'buy_reb_up': 'r_b_u', 'buy_reb_down': 'r_b_d', 
                                'sell_reb_up': 'r_s_u', 'sell_reb_down': 'r_s_d',
                                'cash_ratio_pct': 'conf_cash', 'base_asset': 'conf_base',
                                'leverage_asset': 'conf_lev', 'trade_at': 'conf_trade_at',
                                # 스마트 필터 위젯 키 매핑 강화
                                'use_panic': 'use_panic_chk',
                                'panic_ma': 'panic_ma_sel',
                                'use_vxn_safety': 'use_vxn_safety_chk',
                                'vxn_exit': 'vxn_exit_sli',
                                'use_rsi_turbo': 'use_rsi_turbo_chk',
                                'rsi_turbo': 'rsi_turbo_sli',
                                'use_sl_control': 'use_sl_control_chk',
                                'sl_control_limit': 'sl_control_lim_sli',
                                'use_set_sl': 'use_set_sl_chk',
                                'set_sl_limit': 'set_sl_lim_sli'
                            }

                            for p_key, w_key in key_map.items():
                                if p_key in config:
                                    val = config[p_key]
                                    if p_key.endswith('_up') or p_key.endswith('_down'):
                                        st.session_state[w_key] = float(val) * 100.0
                                    elif p_key == 'cash_ratio_pct':
                                        st.session_state[w_key] = float(val)
                                    else:
                                        st.session_state[w_key] = val
                            
                            # 시그널 조건 동기화 (최대 3개)
                            buy_sig_list = config.get('buy_signals', [])
                            for i in range(1, 4):
                                bp = buy_sig_list[i-1] if i <= len(buy_sig_list) else {}
                                st.session_state[f"b_rsi_wait_use_{i}"] = bp.get('use_rsi_wait', False)
                                st.session_state[f"b_rsi_wait_val_{i}"] = bp.get('rsi_wait_val', 35)
                                st.session_state[f"b_rsi_{i}"] = bp.get('rsi_val', 35 if i==1 else 0)
                                st.session_state[f"b_cross_{i}"] = bp.get('rsi_cross', False if i==1 else True)
                                st.session_state[f"b_inc_{i}"] = bp.get('rsi_inc', False)
                                st.session_state[f"b_use_adx_{i}"] = bp.get('use_adx', False)
                                st.session_state[f"b_adx_op_{i}"] = bp.get('adx_op', "<=")
                                st.session_state[f"b_adx_{i}"] = bp.get('adx_val', 40)
                                st.session_state[f"b_m_inc_{i}"] = bp.get('macd_inc', False if i==1 else True)
                                st.session_state[f"b_m_below_{i}"] = bp.get('macd_signal_below', False if i==1 else True)
                                st.session_state[f"b_m_golden_{i}"] = bp.get('macd_golden', False)
                                st.session_state[f"b_bb_low_{i}"] = bp.get('bb_lower', False)
                                st.session_state[f"b_willr_use_{i}"] = bp.get('use_willr', False)
                                st.session_state[f"b_willr_val_{i}"] = bp.get('willr_val', -80)
                                st.session_state[f"b_di_plus_cross_{i}"] = bp.get('di_plus_cross', False)
                                st.session_state[f"b_di_minus_cross_{i}"] = bp.get('di_minus_cross', False)
                                st.session_state[f"b_sar_use_{i}"] = bp.get('use_sar', False)

                            # 매도 시그널 조건 동기화 (최대 4개)
                            sell_sig_list = config.get('sell_signals', [])
                            for i in range(1, 5):
                                sp = sell_sig_list[i-1] if i <= len(sell_sig_list) else {}
                                st.session_state[f"s_rsi_wait_use_{i}"] = sp.get('use_rsi_wait', False)
                                st.session_state[f"s_rsi_wait_val_{i}"] = sp.get('rsi_wait_val', 70)
                                st.session_state[f"s_rsi_{i}"] = sp.get('rsi_val', 70 if i==1 else 0)
                                st.session_state[f"s_dead_{i}"] = sp.get('rsi_dead', i==2)
                                st.session_state[f"s_dec_{i}"] = sp.get('rsi_dec', i==1)
                                st.session_state[f"s_di_{i}"] = sp.get('di_minus_above', False)
                                st.session_state[f"s_di_minus_cross_{i}"] = sp.get('di_minus_cross', False)
                                st.session_state[f"s_m_dec_{i}"] = sp.get('macd_dec', i==2)
                                st.session_state[f"s_m_above_{i}"] = sp.get('macd_signal_above', i==2)
                                st.session_state[f"s_m_dead_{i}"] = sp.get('macd_dead', i==3)
                                st.session_state[f"s_bb_up_{i}"] = sp.get('bb_upper', False)
                                st.session_state[f"s_chan_use_{i}"] = sp.get('use_chandelier', i==4)
                                st.session_state[f"s_chan_mult_{i}"] = float(sp.get('chandelier_mult', 3.0))
                                st.session_state[f"s_sar_use_{i}"] = sp.get('use_sar', False)
                                st.session_state[f"s_willr_use_{i}"] = sp.get('use_willr', False)
                                st.session_state[f"s_willr_val_{i}"] = sp.get('willr_val', -20)

                            # S3 보호 설정 동기화 (최대 3개 신호 블록 처리)
                            s3_prots = config.get('s3_protection', [])
                            for i in range(1, 4):
                                # 해당 순번의 설정이 있으면 불러오고, 없으면 기본값으로 초기화
                                s3p = s3_prots[i-1] if i <= len(s3_prots) else {}
                                
                                st.session_state[f"s_drop_use_{i}"] = s3p.get('use_daily_drop', False)
                                st.session_state[f"s_drop_lim_{i}"] = float(s3p.get('drop_limit', -3.0))
                                st.session_state[f"s_ma60_use_{i}"] = s3p.get('use_ma60', False)
                                st.session_state[f"s_ma60_lim_{i}"] = float(s3p.get('ma60_limit', 0.0))
                                st.session_state[f"s_ma200_use_{i}"] = s3p.get('use_ma200', False)
                                st.session_state[f"s_ma200_lim_{i}"] = float(s3p.get('ma200_limit', 0.0))
                                st.session_state[f"s_vj_use_{i}"] = s3p.get('use_vix_jump', False)
                                st.session_state[f"s_vj_{i}"] = float(s3p.get('vix_jump', 15.0))
                                st.session_state[f"s_gap_use_{i}"] = s3p.get('use_gap_down', False)
                                st.session_state[f"s_gap_lim_{i}"] = float(s3p.get('gap_limit', -3.0))
                                st.session_state[f"s_acc_use_{i}"] = s3p.get('use_drop_acc', False) # 파라미터명 수정: use_acc_down -> use_drop_acc
                                st.session_state[f"s_acc_lim_{i}"] = float(s3p.get('acc_limit', -7.0))
                                # [신규] S3 보호용 샹들리에 엑시트 파라미터 동기화
                                st.session_state[f"s3_chan_use_{i}"] = s3p.get('use_chandelier', False)
                                st.session_state[f"s3_chan_mult_{i}"] = float(s3p.get('chandelier_mult', 3.0))
                                st.session_state[f"s_exit_all_{i}"] = s3p.get('use_exit_all', False)

                            # [신규] 하락장 매수 신호(S1~S3) 동기화 (최대 3개)
                            pb_sig_list = config.get('panic_buy_signals', [])
                            for i in range(1, 4):
                                pb = pb_sig_list[i-1] if i <= len(pb_sig_list) else {}
                                st.session_state[f"pb_rsi_wait_use_{i}"] = pb.get('use_rsi_wait', False)
                                st.session_state[f"pb_rsi_wait_val_{i}"] = pb.get('rsi_wait_val', 35)
                                st.session_state[f"pb_rsi_{i}"] = pb.get('rsi_val', 27 if i==1 else (28 if i==2 else 30))
                                st.session_state[f"pb_cross_{i}"] = pb.get('rsi_cross', False)
                                st.session_state[f"pb_inc_{i}"] = pb.get('rsi_inc', False)
                                st.session_state[f"pb_use_adx_{i}"] = pb.get('use_adx', False)
                                st.session_state[f"pb_adx_op_{i}"] = pb.get('adx_op', "<=")
                                st.session_state[f"pb_adx_{i}"] = pb.get('adx_val', 40)
                                st.session_state[f"pb_m_inc_{i}"] = pb.get('macd_inc', False)
                                st.session_state[f"pb_m_below_{i}"] = pb.get('macd_signal_below', False)
                                st.session_state[f"pb_m_golden_{i}"] = pb.get('macd_golden', False)
                                st.session_state[f"pb_bb_low_{i}"] = pb.get('bb_lower', False)
                                st.session_state[f"pb_willr_use_{i}"] = pb.get('use_willr', False)
                                st.session_state[f"pb_willr_val_{i}"] = pb.get('willr_val', -80)
                                st.session_state[f"pb_di_plus_cross_{i}"] = pb.get('di_plus_cross', False)
                                st.session_state[f"pb_di_minus_cross_{i}"] = pb.get('di_minus_cross', False)
                                st.session_state[f"pb_sar_use_{i}"] = pb.get('use_sar', False)

                            # 리밸런싱 모드 및 ATR 설정 동기화
                            st.session_state.use_fixed_chk = config.get('use_fixed_reb', True)
                            st.session_state.use_atr_chk = config.get('use_atr_reb', False)
                            st.session_state.atr_m_buy_up = float(config.get('atr_mult_buy_up', 10.0))
                            st.session_state.atr_m_buy_down = float(config.get('atr_mult_buy_down', 1.5))
                            st.session_state.atr_m_sell = float(config.get('atr_mult_sell', 3.0))
                            
                            # [신규] 매수/매도 ATR 기준일 동기화 (과거 데이터 호환을 위해 get 기본값 사용)
                            st.session_state.atr_p_buy = int(config.get('atr_period_buy', 20))
                            st.session_state.atr_p_sell = int(config.get('atr_period_sell', 20))

                            st.session_state.loaded_strat_name = target_strat
                            st.success(f"'{target_strat}' 모든 설정 및 UI 로드 완료!")
                            st.rerun()
            else:
                st.info("저장된 전략이 없습니다.")
        
        start_date = datetime.date(2010, 1, 1)
        end_date = datetime.date.today()
        
        with st.spinner('전 기간 역사적 데이터 시뮬레이션 중...'):
            # 중첩된 설정(시그널 리스트 등)의 변경을 감지하기 위해 Deep Tuplification 적용
            params_tuple = deep_tuple(params)
            smart_tuple = deep_tuple(smart_params)
            
            golden_history, closed_trades, all_signal_events = get_cached_strategy_result(
                data_dict, fg_df, vix_df, leverage_asset, base_asset, params['cash_ratio_pct']/100.0, 
                start_date, end_date, params_tuple, trade_at, smart_tuple, salt
            )
            
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
            
        st.divider()
        
        # [수정] 탭 상태 유지를 위한 세션 상태 기반 탭 시스템 구현
        tab_list = ["📊 분석 결과 & 상세 로그", "📝 매매 전략 설정", "🚀 AI 최적화 탐색 (Optimizer)"]
        if 'history_active_tab' not in st.session_state:
            st.session_state.history_active_tab = tab_list[0]
            
        # 가로형 라디오 버튼을 탭처럼 활용 (label_visibility="collapsed"로 깔끔하게)
        active_tab = st.radio("Navigation", tab_list, 
                             index=tab_list.index(st.session_state.history_active_tab),
                             horizontal=True, label_visibility="collapsed")
        st.session_state.history_active_tab = active_tab
        
        st.write("") # 간격 조절

        if active_tab == tab_list[0]:
            # tab1 내용
            if golden_history.empty:
                st.warning("분석할 데이터가 충분하지 않습니다.")
            else:
                st.subheader("📊 전체 기간 요약")
                HistoryLabView.render_overall_summary(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3))
                
                st.divider()
                HistoryLabView.render_yearly_table(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3), smart_params=smart_params)

                # [신규] 하락장 매수 지연 해제 작동 기록
                st.divider()
                st.subheader("🛡️ 하락장 매수 지연 해제 기록")
                if all_signal_events:
                    panic_buy_list = [e for e in all_signal_events if str(e.get('type')).startswith('하락장매수해제')]
                    if panic_buy_list:
                        pb_df = pd.DataFrame(panic_buy_list)
                        pb_df = pb_df[['date', 'type', 'reason', 'price']].copy()
                        pb_df['date'] = pb_df['date'].dt.strftime('%Y-%m-%d')
                        pb_df.columns = ['발생일', '해제 단계', '충족 조건 (이유)', '기준가(종가)']
                        st.dataframe(pb_df.set_index('발생일'), use_container_width=True)
                    else:
                        st.info("시뮬레이션 기간 내 하락장 매수 지연 해제 기록이 없습니다.")
                
                # [신규] 손절매(Stop-Loss) 작동 기록
                st.divider()
                st.subheader("🛑 손절매(Stop-Loss) 작동 기록")
                if all_signal_events:
                    sl_events = [e for e in all_signal_events if '손절' in str(e.get('type'))]
                    if sl_events:
                        sl_df = pd.DataFrame(sl_events)
                        sl_df = sl_df[['date', 'type', 'reason', 'price', 'executed']].copy()
                        sl_df['date'] = sl_df['date'].dt.strftime('%Y-%m-%d')
                        sl_df['executed'] = sl_df['executed'].map({True: "✅ 체결(손절)", False: "🛡️ 차단(보호)"})
                        sl_df.columns = ['발생일', '항목', '상세 사유', '당시 가격', '처리 상태']
                        st.dataframe(sl_df.set_index('발생일'), use_container_width=True)
                    else:
                        st.info("시뮬레이션 기간 내 손절매 작동 기록이 없습니다.")

                # [수정] S3 보호 설정 및 매도 기록 통합 진단 로그
                st.divider()
                st.subheader("⚠️ S3 보호 및 긴급 매도 진단 로그 (분석용)")
                col_diag1, col_diag2 = st.columns(2)
                with col_diag1:
                    with st.expander("🚩 S3 보호(Emergency) 작동 기록", expanded=False):
                        if 'S3_Detail' in golden_history.columns:
                            s3_emerg = golden_history[golden_history['S3_Detail'] != ""].copy()
                            if not s3_emerg.empty:
                                s3_emerg_display = s3_emerg[['S3_Detail', 'Asset', 'RSI', 'VXN']].copy()
                                s3_emerg_display.index = s3_emerg_display.index.strftime('%Y-%m-%d')
                                s3_emerg_display.columns = ['작동 상세 원인', '보유자산', 'RSI', 'VXN']
                                st.dataframe(s3_emerg_display, use_container_width=True)
                            else:
                                st.info("기록된 S3 보호 작동 내역이 없습니다.")
                with col_diag2:
                    with st.expander("🚩 S3 + 매도신호3(MACD데드+매도우세) 기록", expanded=False):
                        if 'S3_Sell3_Event' in golden_history.columns:
                            sell3_days = golden_history[golden_history['S3_Sell3_Event'] == 1].copy()
                            if not sell3_days.empty:
                                s3_sell3_display = sell3_days[['RSI', 'MACD', 'VXN', 'ADX', 'DMP', 'DMN', 'Asset']].copy()
                                s3_sell3_display.index = s3_sell3_display.index.strftime('%Y-%m-%d')
                                s3_sell3_display.columns = ['RSI', 'MACD', 'VXN', 'ADX', 'DI+', 'DI-', '보유자산']
                                st.dataframe(s3_sell3_display, use_container_width=True)
                            else:
                                st.info("S3 + 매도신호3 작동 기록이 없습니다.")

        elif active_tab == tab_list[1]:
            st.subheader("⚖️ 리밸런싱 및 상세 시그널 설정")
            c_sel1, c_sel2, _ = st.columns(3)
            use_fixed = c_sel1.checkbox("🔹 고정 비율 리밸런싱 사용", value=st.session_state.current_params.get('use_fixed_reb', True), key="use_fixed_chk")
            use_atr = c_sel2.checkbox("🔸 ATR 변동성 리밸런싱 사용", value=st.session_state.current_params.get('use_atr_reb', False), key="use_atr_chk")
            
            st.session_state.current_params['use_fixed_reb'] = use_fixed
            st.session_state.current_params['use_atr_reb'] = use_atr

            ticker_map = {
                "QQQ": "나스닥 100 (QQQ)", "QLD": "나스닥 2배 (QLD)", "TQQQ": "나스닥 3배 (TQQQ)",
                "SOXX": "반도체 1배 (SOXX)", "USD": "반도체 2배 (USD)", "SOXL": "반도체 3배 (SOXL)",
                "TMF": "채권 3배 (TMF)", "TLT": "채권 1배 (TLT)"
            }

            with st.form("strategy_config_combined_form"):
                # 1. 상단 섹션 (기준 자산 및 매매 시그널)
                new_params_top = TesterView.render_top_part(st.session_state.current_params, ticker_map=ticker_map)
                # 2. 하위 섹션 (리밸런싱 구체 설정 및 스마트 필터)
                new_params_bottom = TesterView.render_bottom_part(st.session_state.current_params, ticker_map=ticker_map)
                
                st.write("---")
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.form_submit_button("🚀 전체 설정 반영 및 재시뮬레이션", use_container_width=True):
                    st.session_state.current_params.update(new_params_top)
                    st.session_state.current_params.update(new_params_bottom)
                    st.rerun()
                
                if col_btn2.form_submit_button("🧼 시스템 데이터 완전 초기화 (캐시 삭제)", use_container_width=True):
                    st.cache_data.clear()
                    st.success("캐시가 성공적으로 삭제되었습니다.")
                    st.rerun()

        elif active_tab == tab_list[2]:
            OptimizerView.render(data_dict, fg_df, vix_df, st.session_state.current_params)

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
                if s.empty or len(s) < 2 or s.iloc[0] == 0: return 0
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
