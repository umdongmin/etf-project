import streamlit as st
import pandas as pd
import datetime
import time
import copy
import plotly.graph_objects as go
from core.portfolio_manager import PortfolioManager
from core.storage import StrategyStorage
from ui.tester_view import TesterView
from ui.history_view import HistoryLabView
from ui.portfolio_realtime_view import PortfolioRealtimeView
from ui.backtest_view import BacktestView
from utils.metrics import calculate_metrics
from core.defaults import DEFAULT_BOND_PARAMS, REBALANCE_PRESET_LABELS, get_default_bond_lev_params
from core.engine_kwargs_builder import build_equity_kwargs, build_bond_kwargs, resolve_rebalance_preset, extract_smart_params
from core.config_loader import load_default_equity_params, load_default_bond_params, load_default_portfolio_name

class PortfolioView:
    @staticmethod
    def render(data_dict, fg_df, vxn_df, macro_df, news_df, start_date, end_date):
        st.header("📈 포트폴리오 매니저 (Multi-Strategy)")
        
        # [신규] 기간 선택기 추가 (사이드바 날짜보다 우선순위 가짐)
        local_start, local_end = TesterView.render_period_selector()
        st.session_state.portfolio_start_date = local_start
        st.session_state.portfolio_end_date = local_end
        
        st.caption(f"📅 분석 기간 (적용됨): **{local_start} ~ {local_end}**")

        # 세션 상태 초기화: 포트폴리오 전략 목록
        from core.session_keys import SK as _SK
        _PORTFOLIO_INIT_VER = 'v6'
        _needs_portfolio_init = (
            'portfolio_configs' not in st.session_state
            or st.session_state.get(_SK.PORTFOLIO_INIT_VER) != _PORTFOLIO_INIT_VER
            or not any(c.get('type') == 'bond' and 'lev_params' in c
                       for c in st.session_state.get('portfolio_configs', []))
        )
        if _needs_portfolio_init:
            # 1순위: DB에 저장된 기본 포트폴리오 자동 로드
            _default_portfolio_name = StrategyStorage.get_default_portfolio()
            _loaded_default = StrategyStorage.load_portfolio(_default_portfolio_name) if _default_portfolio_name else None

            if _loaded_default:
                # lev_params는 strategies 테이블에서 load_portfolio()가 직접 가져옴
                st.session_state.portfolio_configs = copy.deepcopy(_loaded_default['configs'])
                st.session_state.portfolio_total_capital = _loaded_default['total_capital']
                st.session_state.portfolio_rebalance_preset = _loaded_default.get('rebalance_preset', 'none')
            else:
                # 2순위: 하드코딩 기본값 (기본 포트폴리오 미지정 시)
                _default_bond_params = DEFAULT_BOND_PARAMS
                _default_bond_lev_params = copy.deepcopy(st.session_state.get('current_bond_lev_params')) or None
                _eq_strats = StrategyStorage.list_strategies(category='equity')
                _default_eq_name = next((n for n in _eq_strats if 'TQQQ' in n or 'tqqq' in n), _eq_strats[0] if _eq_strats else '전략 A (TQQQ 베이스)')
                _default_eq_params = StrategyStorage.load_strategy(_default_eq_name) or copy.deepcopy(st.session_state.current_params)
                _bond_strats = StrategyStorage.list_strategies(category='bond')
                _default_bond_name = next((n for n in _bond_strats if 'V8' in n), None) \
                                  or next((n for n in _bond_strats if 'V7' in n), None) \
                                  or (_bond_strats[0] if _bond_strats else '전략 B (TLT/TBF 채권)')
                _default_bond_loaded = StrategyStorage.load_strategy(_default_bond_name) if _bond_strats else None
                st.session_state.portfolio_configs = [
                    {'name': _default_eq_name, 'weight': 0.78, 'type': 'equity', 'params': copy.deepcopy(_default_eq_params)},
                    {'name': _default_bond_name, 'weight': 0.22, 'type': 'bond',
                     'params': copy.deepcopy(_default_bond_loaded or st.session_state.get('current_bond_params', _default_bond_params)),
                     'lev_params': _default_bond_lev_params},
                ]

            st.session_state[_SK.PORTFOLIO_INIT_VER] = _PORTFOLIO_INIT_VER
            st.session_state.pop('portfolio_result', None)  # 초기화 시 이전 결과 제거
        
        if 'portfolio_total_capital' not in st.session_state:
            st.session_state.portfolio_total_capital = 10000.0

        tab1, tab2 = st.tabs(["📊 통합 대시보드", "🧩 전략 구성 및 비중"])

        with tab2:
            st.subheader("💾 포트폴리오 관리 (DB)")
            
            p_management_expander = st.expander("📂 포트폴리오 불러오기 / 저장하기", expanded=True)
            with p_management_expander:
                col_load, col_save = st.columns(2)
                
                with col_load:
                    st.markdown("##### 📥 불러오기")
                    p_list = StrategyStorage.list_portfolios()
                    if p_list:
                        _default_p_name = StrategyStorage.get_default_portfolio()
                        _fmt_p = lambda x: f"⭐ {x} (기본)" if x == _default_p_name else x
                        _default_p_idx = (
                            p_list.index(_default_p_name)
                            if _default_p_name and _default_p_name in p_list
                            else 0
                        )
                        selected_p = st.selectbox("저장된 포트폴리오 선택", p_list, index=_default_p_idx, key="p_load_sel", format_func=_fmt_p)
                        c1, c2, c3 = st.columns(3)
                        if c1.button("📂 불러오기", use_container_width=True):
                            loaded = StrategyStorage.load_portfolio(selected_p)
                            if loaded:
                                # (1) 핵심 데이터 교체 (deepcopy 필수)
                                loaded_configs = copy.deepcopy(loaded['configs'])

                                # lev_params는 strategies 테이블에서 load_portfolio()가 직접 가져옴
                                st.session_state.portfolio_configs = loaded_configs
                                st.session_state.portfolio_total_capital = loaded['total_capital']
                                st.session_state.portfolio_rebalance_preset = loaded.get('rebalance_preset', 'none')
                                # portfolio_init_ver 동기화 → 재초기화 방지
                                st.session_state[_SK.PORTFOLIO_INIT_VER] = _PORTFOLIO_INIT_VER

                                # (3) 위젯 상태 초기화 (강력하게)
                                if 'p_load_version' not in st.session_state:
                                    st.session_state.p_load_version = 0
                                st.session_state.p_load_version += 1

                                prefixes_to_del = (
                                    "p_type_sel_", "p_strat_sel_", "p_weight_",
                                    "p_strat_tune_sel", "p_strat_swap_", "strat_detail_"
                                )
                                for k in list(st.session_state.keys()):
                                    if k.startswith(prefixes_to_del):
                                        del st.session_state[k]

                                st.success(f"✅ '**{selected_p}**' 구성을 성공적으로 불러왔습니다.")
                                st.rerun()
                        if c2.button("⭐ 기본으로 설정", use_container_width=True):
                            if StrategyStorage.set_default_portfolio(selected_p):
                                st.success(f"⭐ '**{selected_p}**'이(가) 기본 포트폴리오로 설정되었습니다.")
                                st.rerun()
                            else:
                                st.error("기본 포트폴리오 설정 중 오류가 발생했습니다.")
                        if c3.button("🗑️ 삭제", use_container_width=True):
                            if StrategyStorage.delete_portfolio(selected_p):
                                st.warning(f"🗑️ '**{selected_p}**' 구성을 삭제했습니다.")
                                st.rerun()
                    else:
                        st.info("저장된 포트폴리오가 없습니다.")
                
                with col_save:
                    st.markdown("##### 📤 현재 구성 저장")
                    save_name = st.text_input("새로 저장할 이름 입력", placeholder="예: 2024 하반기 공격형", key="p_save_name")
                    if st.button("💾 현재 포트폴리오 저장", use_container_width=True):
                        if not save_name.strip():
                            st.error("저장할 이름을 입력해 주세요.")
                        else:
                            # lev_params는 strategies 테이블에서 관리 — portfolio_configs 그대로 저장
                            saved_name = StrategyStorage.save_portfolio(
                                save_name,
                                st.session_state.portfolio_configs,
                                st.session_state.portfolio_total_capital,
                                st.session_state.get('portfolio_rebalance_preset', 'none')
                            )
                            if saved_name:
                                st.success(f"✅ 현재 배분 상태가 '**{saved_name}**'(으)로 저장되었습니다.")
                                st.session_state.pop('portfolio_result', None)  # 저장 후 재시뮬레이션 강제
                                st.rerun()

            st.write("---")
            st.subheader("🧩 포트폴리오 전략 구성")
            col_cap, col_rb, col_add = st.columns([2, 2, 1])
            total_cap = col_cap.number_input("총 투자 자산 ($)", min_value=100.0, value=st.session_state.portfolio_total_capital, step=100.0)
            st.session_state.portfolio_total_capital = total_cap


            PRESET_LABELS = REBALANCE_PRESET_LABELS
            preset_keys = list(PRESET_LABELS.keys())
            current_preset = st.session_state.get('portfolio_rebalance_preset', 'hybrid')
            preset_idx = preset_keys.index(current_preset) if current_preset in preset_keys else 1
            selected_preset = col_rb.selectbox(
                "🔄 리밸런싱 전략",
                options=preset_keys,
                format_func=lambda k: PRESET_LABELS[k],
                index=preset_idx,
                key='portfolio_rebalance_preset'
            )
            
            if col_add.button("➕ 전략 추가", use_container_width=True):
                from core.dispatcher import Dispatcher, Action
                from core.state import AppState
                new_idx = len(AppState.get_portfolio_configs()) + 1
                Dispatcher.dispatch(Action.ADD_PORTFOLIO_STRATEGY, {
                    'config': {
                        'name':   f'신규 전략 {new_idx}',
                        'weight': 0.0,
                        'type':   'equity',
                        'params': AppState.get_equity_params(),
                    }
                })
                st.rerun()

            st.write("---")
            configs = st.session_state.portfolio_configs
            
            # [최적화] 루프 밖에서 한 번만 목록 가져오기 (캐시 활용)
            all_equity_names = StrategyStorage.list_strategies(category='equity')
            all_bond_names = StrategyStorage.list_strategies(category='bond')
            
            total_weight = 0.0
            to_delete = -1
            
            p_v = st.session_state.get('p_load_version', 0)
            
            for i in range(len(configs)):
                config = configs[i]
                # [개선] 제목은 심플하게 고정 (내용 변경 시 헷갈리지 않게)
                with st.expander(f"전략 #{i+1}", expanded=True):
                    col_type, col_sel, col_w, col_del = st.columns([1, 2, 1, 1])
                    
                    # (1) 타입 선택 (주식/채권)
                    curr_type = config.get('type', 'equity')
                    new_type = col_type.selectbox(
                        f"타입 #{i+1}", 
                        ['equity', 'bond'], 
                        index=0 if curr_type == 'equity' else 1,
                        format_func=lambda x: "📈 주식" if x == 'equity' else "💰 채권",
                        key=f"p_type_sel_{i}_{p_v}"
                    )
                    
                    # (2) 필터링된 전략 목록 (루프 밖에서 가져온 리스트 활용)
                    strat_options = all_equity_names if new_type == 'equity' else all_bond_names
                    
                    if not strat_options:
                        col_sel.warning("저장된 전략 없음")
                        selected_name = config['name']
                    else:
                        try:
                            # 현재 선택된 이름이 목록에 있는지 확인
                            sel_idx = strat_options.index(config['name']) if config.get('type') == new_type else 0
                        except ValueError:
                            sel_idx = 0
                            
                        selected_name = col_sel.selectbox(
                            f"전략 선택 #{i+1}", 
                            strat_options, 
                            index=sel_idx,
                            key=f"p_strat_sel_{i}_{p_v}"
                        )
                    
                    # (3) 데이터 동기화 — Dispatcher를 통해 diff 비교 후 변경된 경우에만 DB 호출
                    if new_type != curr_type or (strat_options and selected_name != config['name']):
                        from core.dispatcher import Dispatcher, Action
                        changed = Dispatcher.dispatch(Action.UPDATE_PORTFOLIO_STRATEGY, {
                            'index': i,
                            'name':  selected_name,
                            'type':  new_type,
                        })
                        if changed:
                            st.rerun()

                    # (4) 비중 표시 (실시간 반영)
                    configs[i]['weight'] = col_w.number_input(f"비중 (0~1.0) #{i+1}", value=config['weight'], min_value=0.0, max_value=1.0, step=0.05, key=f"p_weight_{i}_{p_v}")
                    total_weight += configs[i]['weight']
                    
                    st.write(f"할당 자산: `${total_cap * configs[i]['weight']:,.2f}`")
                    
                    if col_del.button("🗑️ 삭제", key=f"btn_p_del_{i}", use_container_width=True):
                        to_delete = i
            
            if to_delete != -1:
                from core.dispatcher import Dispatcher, Action
                Dispatcher.dispatch(Action.REMOVE_PORTFOLIO_STRATEGY, {'index': to_delete})
                st.rerun()

            if abs(total_weight - 1.0) > 1e-5:
                st.warning(f"⚠️ 현재 총 비중 합계: **{total_weight:.2f}**. (정확히 1.0이 되어야 합니다.)")
            else:
                st.success(f"✅ 비중 합계: 1.0 (정상)")

            st.write("---")
            if st.button("🚀 포트폴리오 통합 시뮬레이션 실행", type="primary", use_container_width=True):
                if abs(total_weight - 1.0) > 1e-5:
                    st.error("비중 합계가 1.0이 아니면 실행할 수 없습니다.")
                else:
                    PortfolioView.run_simulation(data_dict, fg_df, vxn_df, macro_df, news_df, local_start, local_end)
                    from core.dispatcher import Dispatcher
                    Dispatcher.clear_portfolio_dirty()

        with tab1:
            from core.state import AppState
            from core.dispatcher import Dispatcher
            from core.session_keys import SK

            # ── 결과 stale 경고 (자동 재실행 대신 사용자에게 안내) ──────────────
            result_exists = SK.PORTFOLIO_RESULT in st.session_state
            if result_exists:
                saved_fee   = st.session_state[SK.PORTFOLIO_RESULT].get('fee_rate', 0.0)
                current_fee = AppState.get_fee_rate()
                if abs(saved_fee - current_fee) > 1e-8:
                    st.warning(
                        f"⚠️ 수수료가 변경되었습니다 "
                        f"({saved_fee*100:.3f}% → {current_fee*100:.3f}%). "
                        f"'전략 구성 및 비중' 탭에서 시뮬레이션을 재실행하세요."
                    )
            if Dispatcher.is_portfolio_dirty() and result_exists:
                st.info("⚡ 전략 구성이 변경되었습니다. '전략 구성 및 비중' 탭에서 시뮬레이션을 실행하세요.")

            # ── 결과 없으면 안내만 표시 (자동 실행 금지) ────────────────────────
            if not result_exists:
                st.info("📊 시뮬레이션 결과가 없습니다. '전략 구성 및 비중' 탭에서 실행해 주세요.")
            else:
                PortfolioView.render_dashboard()

    @staticmethod
    def render_realtime_tab(data_dict, fg_df, vxn_df, macro_df, news_df, start_date, end_date=None):
        """실시간 모니터링 탭: 현재 portfolio_configs 기반으로 PortfolioManager를 빌드하고 신호를 표시.

        Parameters:
        - end_date: None이면 자동으로 현재 날짜로 설정
        """
        import copy as _copy
        import datetime as _dt

        # end_date가 None이거나 미래일 경우 현재 시간으로 설정
        if end_date is None or (hasattr(end_date, 'date') and end_date.date() > _dt.date.today()) or (isinstance(end_date, _dt.date) and end_date > _dt.date.today()):
            end_date = _dt.date.today()

        # ── 전략 A/B/C/D 선택 드롭다운 (Supabase 기준) ────────────────────────
        st.markdown("#### 📊 포트폴리오 전략 선택")

        # Supabase portfolios 테이블에서 포트폴리오 목록 불러오기
        portfolio_names = StrategyStorage.list_portfolios()

        if not portfolio_names:
            st.error("⚠️ Supabase에서 포트폴리오를 불러올 수 없습니다. 포트폴리오 데이터를 확인하세요.")
            return

        if 'portfolio_selected_name' not in st.session_state:
            # DB 기본값 우선 → 없으면 첫 번째 항목
            _db_default = StrategyStorage.get_default_portfolio()
            default_portfolio = (
                _db_default if _db_default and _db_default in portfolio_names
                else portfolio_names[0]
            )
            st.session_state.portfolio_selected_name = default_portfolio

        def on_portfolio_change():
            """포트폴리오 변경 시 portfolio_realtime_result 초기화 (신호 재계산 트리거)"""
            st.session_state.portfolio_realtime_result = None

        selected_portfolio_name = st.selectbox(
            "다음 중 하나 선택:",
            portfolio_names,
            index=portfolio_names.index(st.session_state.portfolio_selected_name) if st.session_state.portfolio_selected_name in portfolio_names else 0,
            key='portfolio_selector',
            on_change=on_portfolio_change
        )
        st.session_state.portfolio_selected_name = selected_portfolio_name

        # ── 선택된 포트폴리오 로드 ──────────────────────────────────────────────
        portfolio_data = StrategyStorage.load_portfolio(selected_portfolio_name)
        if not portfolio_data:
            st.error(f"⚠️ 포트폴리오 '{selected_portfolio_name}'을 불러올 수 없습니다.")
            return

        _default_bond_params = DEFAULT_BOND_PARAMS
        _default_bond_lev    = _copy.deepcopy(st.session_state.get('current_bond_lev_params')) or None

        # 포트폴리오 configs에서 자산별 정보 추출
        _configs = portfolio_data.get('configs', [])
        _eq_config = next((c for c in _configs if c['type'] == 'equity'), None)
        _bond_config = next((c for c in _configs if c['type'] == 'bond'), None)

        _eq_name = _eq_config['name'] if _eq_config else 'TQQQ Strategy'
        _bond_name = _bond_config['name'] if _bond_config else 'Bond V8'
        _eq_params = _eq_config['params'] if _eq_config else _copy.deepcopy(st.session_state.get('current_params', {}))
        _bond_loaded = _bond_config['params'] if _bond_config else None

        _tqqq_w = _eq_config['weight'] if _eq_config else 0.78
        _bond_w = _bond_config['weight'] if _bond_config else 0.22
        _preset = portfolio_data.get('rebalance_preset', 'hybrid')

        # trend_defense는 portfolio_items에는 저장되지 않으므로, preset에 따라 결정
        _trend_def = None
        if _preset == 'trend':
            _trend_def = {'tqqq_strategy': 0.50, 'Bond_V7_Custom': 0.50}

        st.session_state.portfolio_configs = [
            {'name': _eq_name,   'weight': _tqqq_w, 'type': 'equity', 'params': _copy.deepcopy(_eq_params)},
            {'name': _bond_name, 'weight': _bond_w, 'type': 'bond',
             'params': _copy.deepcopy(_bond_loaded or st.session_state.get('current_bond_params', _default_bond_params)),
             'lev_params': _default_bond_lev},
        ]
        st.session_state.portfolio_total_capital   = st.session_state.get('portfolio_total_capital', 10000.0)
        st.session_state.portfolio_rebalance_preset = _preset
        st.session_state.portfolio_trend_defense_weights = _trend_def  # 전략 B의 trend 방어 비중

        configs   = st.session_state.get('portfolio_configs', [])
        total_cap = st.session_state.get('portfolio_total_capital', 10000.0)
        preset    = st.session_state.get('portfolio_rebalance_preset', 'hybrid')

        # ── PortfolioManager 빌드 ────────────────────────────────────────────
        # custom_preset_cfg가 필요한 preset (hybrid, trend, 밴드 등)을 처리
        std_preset, custom_cfg = resolve_rebalance_preset(preset)

        from core.state import AppState as _AppState
        fee_rate       = _AppState.get_fee_rate()
        integer_shares = _AppState.get_integer_shares()

        # ── 전략 B(trend) 전용: QQQ MA200 계산 ────────────────────────────────
        trend_data = None
        if preset == 'trend' and 'QQQ' in data_dict:
            try:
                qqq_close = data_dict['QQQ']['close']
                ma200 = qqq_close.rolling(window=200).mean()
                trend_data = qqq_close > ma200  # True=상승(공격), False=하강(방어)
            except Exception:
                trend_data = None

        trend_def_weights = st.session_state.get('portfolio_trend_defense_weights', None)

        pm = PortfolioManager(
            total_capital=total_cap,
            rebalance_preset=std_preset,
            custom_preset_cfg=custom_cfg,
            fee_rate=fee_rate,
            integer_shares=integer_shares,
            trend_data=trend_data,
            trend_defense_weights=trend_def_weights,
        )

        # 전략 등록
        for cfg in configs:
            name     = cfg['name']
            weight   = cfg['weight']
            stype    = cfg.get('type', 'equity')
            params   = cfg.get('params', {})
            lev_p    = cfg.get('lev_params', None)

            if stype == 'equity':
                pm.register_strategy(name, weight, build_equity_kwargs(
                    data_dict=data_dict, fg_df=fg_df, vxn_df=vxn_df, news_df=news_df,
                    params=params, start_date=start_date, end_date=end_date,
                    allocated_capital=total_cap * weight,
                    fee_rate=fee_rate, integer_shares=integer_shares, fast_mode=True,
                ), strat_type='equity')
            else:
                bond_params = params if params else DEFAULT_BOND_PARAMS
                pm.register_strategy(name, weight, build_bond_kwargs(
                    data_dict=data_dict, macro_df=macro_df,
                    params=bond_params, start_date=start_date, end_date=end_date,
                    lev_params=lev_p,
                    allocated_capital=total_cap * weight,
                    fee_rate=fee_rate, integer_shares=integer_shares,
                ), strat_type='bond')

        PortfolioRealtimeView.render(
            portfolio_manager=pm,
            portfolio_name=selected_portfolio_name,
            data_dict=data_dict, fg_df=fg_df, vxn_df=vxn_df,
            macro_df=macro_df, news_df=news_df,
            start_date=start_date,
            rebalance_preset=preset,
        )

    @staticmethod
    def run_simulation(data_dict, fg_df, vxn_df, macro_df, news_df, start_date, end_date):
        configs = st.session_state.portfolio_configs
        total_cap = st.session_state.portfolio_total_capital
        
        # standalone 백테스트 및 벤치마크용 슬라이싱
        filtered_data = {t: df.loc[start_date:end_date] for t, df in data_dict.items() if not df.empty}
        
        # Portfolio Manager 초기화
        rebalance_preset_key = st.session_state.get('portfolio_rebalance_preset', 'quarterly')
        std_preset, custom_cfg = resolve_rebalance_preset(rebalance_preset_key)
        from core.state import AppState as _AppState
        fee_rate = _AppState.get_fee_rate()
        integer_shares = _AppState.get_integer_shares()
        pm = PortfolioManager(total_capital=total_cap, rebalance_preset=std_preset, custom_preset_cfg=custom_cfg, fee_rate=fee_rate, integer_shares=integer_shares)
        
        # 전략 등록
        for config in configs:
            params = config['params']
            s_type = config.get('type', 'equity')

            kwargs = {
                'data_dict': data_dict,  # 전체 데이터 전달 (지표 워밍업 보장)
                'start_date': start_date,
                'end_date': end_date,
                'allocated_capital': total_cap * config['weight']
            }

            if s_type == 'equity':
                eq_params = config['params'] if config.get('params') else params
                kwargs.update(build_equity_kwargs(
                    data_dict=data_dict, fg_df=fg_df, vxn_df=vxn_df, news_df=news_df,
                    params=eq_params, start_date=start_date, end_date=end_date,
                    allocated_capital=total_cap * config['weight'],
                    fee_rate=fee_rate, integer_shares=integer_shares, fast_mode=True,
                ))
            else:
                bond_params = config.get('params', config.get('bond_params', DEFAULT_BOND_PARAMS))
                kwargs.update(build_bond_kwargs(
                    data_dict=data_dict, macro_df=macro_df,
                    params=bond_params, start_date=start_date, end_date=end_date,
                    lev_params=config.get('lev_params'),
                    allocated_capital=total_cap * config['weight'],
                    fee_rate=fee_rate, integer_shares=integer_shares,
                ))

            pm.register_strategy(name=config['name'], weight=config['weight'], engine_kwargs=kwargs, strat_type=s_type)
        
        with st.spinner("다중 전략 통합 시뮬레이션 구동 중..."):
            try:
                # 1. 통합 포트폴리오 시뮬레이션 (리밸런싱 포함)
                combined_equity, individual_results, rebalance_events = pm.run_portfolio()
                
                # 2. [추가] 전략별 독립(Standalone) 백테스트 실행 - 리밸런싱 영향 제거용
                from core.engine import StrategyEngine
                standalone_results = {}
                for config in configs:
                    name = config['name']
                    s_type = config.get('type', 'equity')
                    params = config['params']
                    
                    # 공통 인자
                    base_kwargs = {
                        'data_dict': filtered_data,
                        'start_date': start_date,
                        'end_date': end_date,
                        'allocated_capital': total_cap * config['weight'], # 동일 비중 시작
                        'fee_rate': fee_rate,
                        'integer_shares': integer_shares,
                    }
                    
                    if s_type == 'equity':
                        eq_params = config['params'] if config.get('params') else params
                        st_res, _, _ = StrategyEngine.run_golden_strategy(
                            **build_equity_kwargs(
                                data_dict=data_dict, fg_df=fg_df, vxn_df=vxn_df, news_df=news_df,
                                params=eq_params, start_date=start_date, end_date=end_date,
                                allocated_capital=base_kwargs['allocated_capital'],
                                fee_rate=fee_rate, integer_shares=integer_shares,
                            )
                        )
                    else:
                        bond_p = config.get('params', config.get('bond_params', DEFAULT_BOND_PARAMS))
                        bond_lev_p = config.get('lev_params')
                        st_res, _, _ = StrategyEngine.run_bond_strategy(
                            **build_bond_kwargs(
                                data_dict=data_dict, macro_df=macro_df,
                                params=bond_p, start_date=start_date, end_date=end_date,
                                lev_params=bond_lev_p,
                                allocated_capital=base_kwargs['allocated_capital'],
                                fee_rate=fee_rate, integer_shares=integer_shares,
                            )
                        )
                    
                    standalone_results[name] = st_res
                
                # QQQ 벤치마크 데이터 저장 (연도별 비교용)
                qqq_df = filtered_data.get('QQQ', pd.DataFrame())
                st.session_state.portfolio_result = {
                    'combined': combined_equity,
                    'individuals': individual_results,
                    'standalone': standalone_results,
                    'qqq_df': qqq_df,
                    'fee_rate': st.session_state.get('global_fee_rate', 0.0),
                    'rebalance_events': rebalance_events,
                }
                st.toast("✅ 통합 시뮬레이션 완료!", icon="📈")
            except Exception as e:
                st.error(f"시뮬레이션 중 오류 발생: {e}")
                st.exception(e)

    @staticmethod
    def render_dashboard():
        if 'portfolio_result' not in st.session_state:
            st.info("🧩 '전략 구성' 탭에서 시뮬레이션을 실행해 주세요.")
            return
        
        res = st.session_state.portfolio_result
        combined = res['combined']
        individuals = res['individuals']
        configs = st.session_state.portfolio_configs
        
        if combined.empty:
            st.error("결과 데이터가 비어 있습니다.")
            return

        # 1. 메트릭스 요약
        metrics = calculate_metrics(combined)
        
        st.subheader("📊 통합 포트폴리오 성과")
        m1, m2, m3, m4, m5 = st.columns(5)
        total_ret = (metrics['Final Value']/st.session_state.portfolio_total_capital - 1)*100
        calmar = metrics['CAGR'] / abs(metrics['MDD']) if metrics['MDD'] != 0 else 0
        m1.metric("최종 자산", f"${metrics['Final Value']:,.2f}")
        m2.metric("누적 수익률", f"{total_ret:+.1f}%")
        m3.metric("연평균 (CAGR)", f"{metrics['CAGR']*100:.1f}%")
        m4.metric("최대 낙폭 (MDD)", f"{metrics['MDD']*100:.1f}%")
        m5.metric("칼마 지수", f"{calmar:.2f}")

        # 2. 자산 곡선 차트
        st.write("---")
        st.subheader("📈 자산성장 곡선 (Equity Curve)")
        
        fig = go.Figure()
        
        # 통합 곡선
        fig.add_trace(go.Scatter(
            x=combined.index, y=combined['PortfolioValue'], # [수정] combined['Value'] -> 'PortfolioValue'
            name='통합 포트폴리오 (Total)', 
            line=dict(color='#00c853', width=4),
            fill='tozeroy',
            fillcolor='rgba(0, 200, 83, 0.1)'
        ))
        
        # 개별 전략 곡선
        colors = ['#2979ff', '#ff9100', '#f44336', '#9c27b0']
        for i, (name, data) in enumerate(individuals.items()):
            hist_df = data.get('history', pd.DataFrame())
            if hist_df.empty:
                continue
            
            # [수정] 성과 컬럼 유연하게 탐색 (KeyError 방지 강건성 강화)
            val_col = None
            for col in ['Value', 'PortfolioValue', 'Close']:
                if col in hist_df.columns:
                    val_col = col
                    break
            
            if val_col is None:
                continue
            
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df[val_col], 
                name=f"Strat: {name}", 
                line=dict(width=1.5, color=colors[i % len(colors)]),
                opacity=0.7
            ))
            
        fig.update_layout(
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=10, b=20),
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3. 전략별 비교 테이블
        st.write("---")
        st.subheader("📋 전략별 세부 지표 비교")
        
        comparison_data = []
        standalone_map = res.get('standalone', {})
        
        for i, (name, data) in enumerate(individuals.items()):
            # 리밸런싱된 결과가 아닌, 독립 실행(Standalone) 결과로 메트릭 계산
            st_df = standalone_map.get(name, data['history'])
            m = calculate_metrics(st_df)
            
            # 해당 전략의 비중 찾기
            strat_config = next((c for c in configs if c['name'] == name), None)
            weight = strat_config['weight'] if strat_config else 0.5
            allocated = st.session_state.portfolio_total_capital * weight
            
            comparison_data.append({
                "전략명": name,
                "타입": strat_config.get('type', 'equity') if strat_config else '-',
                "목표비중": f"{weight*100:.0f}%",
                "MDD": f"{m['MDD']*100:.1f}%",
                "Sharpe": f"{m.get('Sharpe', 0):.2f}",
                "Calmar": f"{m.get('MAR', 0):.3f}",
                "CAGR": f"{m['CAGR']*100:.1f}%",
            })

        st.table(pd.DataFrame(comparison_data))
        st.info("💡 위 지표는 **리밸런싱을 배제한** 개별 전략의 순수 성과(Standalone)입니다. 포트폴리오 효과는 상황 대시보드를 참고하세요.")

        # 4. 연도별 성과 및 리스크 분석
        PortfolioView.render_annual_analysis(combined, res.get('qqq_df', pd.DataFrame()), standalone_map)

        # 5. 리밸런싱 거래 내역
        rebalance_events = res.get('rebalance_events', [])
        if rebalance_events:
            st.write("---")
            st.subheader("🔄 리밸런싱 거래 내역")
            PortfolioView._render_rebalance_log(rebalance_events)

        # 6. 전략별 거래 체결 내역
        st.write("---")
        st.subheader("📝 전략별 거래 체결 내역")
        for name, data in individuals.items():
            strat_config = next((c for c in configs if c['name'] == name), None)
            s_type = strat_config.get('type', 'equity') if strat_config else 'equity'
            icon = "📈" if s_type == 'equity' else "💰"
            with st.expander(f"{icon} {name} 거래 내역", expanded=False):
                history_df = data.get('history', pd.DataFrame())
                if not history_df.empty and 'Trade_Label' in history_df.columns:
                    if s_type == 'bond':
                        BacktestView.render_execution_log(history_df, 'TLT', is_bond=True)
                    else:
                        base = strat_config.get('params', {}).get('base_asset', 'QQQ') if strat_config else 'QQQ'
                        BacktestView.render_execution_log(history_df, base)
                closed = data.get('closed_trades', pd.DataFrame())
                if isinstance(closed, pd.DataFrame) and not closed.empty:
                    BacktestView.render_closed_sets(closed)

    @staticmethod
    def _render_rebalance_log(events):
        """리밸런싱 이벤트를 테이블로 표시"""
        rows = []
        for evt in events:
            for detail in evt['Details']:
                rows.append({
                    'Date': evt['Date'],
                    'Strategy': detail['Strategy'],
                    'Before ($)': detail['Before_Value'],
                    'After ($)': detail['After_Value'],
                    'Delta ($)': detail['Delta'],
                    'Target (%)': detail['Target_Weight'],
                    'Actual (%)': detail['Actual_Weight'],
                    'Drift (%)': detail['Drift'],
                })
        if not rows:
            st.info("리밸런싱 이벤트가 없습니다.")
            return

        df = pd.DataFrame(rows)
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        formats = {
            'Before ($)': '${:,.0f}',
            'After ($)': '${:,.0f}',
            'Delta ($)': '${:+,.0f}',
            'Target (%)': '{:.1%}',
            'Actual (%)': '{:.1%}',
            'Drift (%)': '{:+.1%}',
        }

        def style_delta(row):
            try:
                delta = float(row['Delta ($)'])
                if delta > 0:
                    return ['background-color: rgba(231, 76, 60, 0.08)'] * len(row)
                elif delta < 0:
                    return ['background-color: rgba(52, 152, 219, 0.08)'] * len(row)
            except (ValueError, KeyError):
                pass
            return [''] * len(row)

        active_formats = {k: v for k, v in formats.items() if k in df.columns}
        st.caption(f"총 {len(events)}회 리밸런싱 발생")
        st.dataframe(
            df.style.apply(style_delta, axis=1).format(active_formats, na_rep='-'),
            use_container_width=True,
            height=min(80 + len(df) * 38, 600),
        )

    @staticmethod
    def render_annual_analysis(combined: pd.DataFrame, qqq_df: pd.DataFrame, standalone_map: dict):
        """연도별 포트폴리오 및 개별 전략 성과 비교"""
        st.write("---")
        st.subheader("📅 연도별 성과 및 리스크 분석 (Portfolio & Assets)")

        def calc_annual_metrics(df, val_col='Value'):
            if df.empty: return pd.DataFrame()
            if val_col not in df.columns and 'PortfolioValue' in df.columns:
                val_col = 'PortfolioValue'
            elif val_col not in df.columns and 'Close' in df.columns:
                val_col = 'Close'

            series = df[val_col].dropna()
            rows = []
            years = sorted(series.index.year.unique())
            for year in years:
                grp = series[series.index.year == year]
                if len(grp) < 5: continue
                # 전년도 마지막 값을 기준으로 수익률 계산 (리밸런싱 자금이동 왜곡 방지)
                prev_year = series[series.index.year == year - 1]
                base_val = prev_year.iloc[-1] if not prev_year.empty else grp.iloc[0]
                ret = grp.iloc[-1] / base_val - 1
                roll_max = grp.cummax()
                mdd = (grp / roll_max - 1).min()
                calmar = ret / abs(mdd) if mdd != 0 else 0
                rows.append({'year': year, 'Return': ret, 'MDD': mdd, 'Calmar': calmar})
            return pd.DataFrame(rows).set_index('year')

        # 1. 메인 데이터 계산
        port_annual = calc_annual_metrics(combined, 'PortfolioValue')
        qqq_annual = calc_annual_metrics(qqq_df, 'Close')
        
        asset_annuals = {}
        for name, df in standalone_map.items():
            asset_annuals[name] = calc_annual_metrics(df, 'Value')

        if port_annual.empty:
            st.info("분석할 데이터가 부족합니다.")
            return

        # 2. 통합 테이블 구성
        years = port_annual.index
        table_data = []
        for yr in years:
            row = {'연도': f"{yr}년"}
            # Portfolio
            row['포트폴리오 수익'] = port_annual.loc[yr, 'Return']
            row['포트폴리오 MDD'] = port_annual.loc[yr, 'MDD']
            row['포트폴리오 칼마'] = port_annual.loc[yr, 'Calmar']
            
            # Individual Assets
            for name, ann_df in asset_annuals.items():
                if yr in ann_df.index:
                    row[f'{name} 수익'] = ann_df.loc[yr, 'Return']
                    row[f'{name} MDD'] = ann_df.loc[yr, 'MDD']
                    row[f'{name} 칼마'] = ann_df.loc[yr, 'Calmar']
                else:
                    row[f'{name} 수익'] = row[f'{name} MDD'] = row[f'{name} 칼마'] = float('nan')
            table_data.append(row)

        df_final = pd.DataFrame(table_data).set_index('연도')

        # 3. 스타일링
        def style_metrics(df):
            def color_ret(val):
                if pd.isna(val): return ''
                color = '#c8f7c5' if val > 0 else '#ffc0c0'
                return f'background-color:{color}; font-weight:bold'
            
            def color_mdd(val):
                if pd.isna(val): return ''
                intensity = min(abs(val) / 0.5, 1.0)
                return f'background-color:rgba(255,0,0,{intensity*0.3})'

            def color_calmar(val):
                if pd.isna(val): return ''
                color = '#e3f2fd' if val > 1.0 else ''
                return f'background-color:{color}'

            ret_cols = [c for c in df.columns if '수익' in c]
            mdd_cols = [c for c in df.columns if 'MDD' in c]
            calmar_cols = [c for c in df.columns if '칼마' in c]

            return df.style \
                .map(color_ret, subset=ret_cols) \
                .map(color_mdd, subset=mdd_cols) \
                .map(color_calmar, subset=calmar_cols) \
                .format("{:+.1%}", subset=ret_cols + mdd_cols) \
                .format("{:.2f}", subset=calmar_cols, na_rep="-")

        st.dataframe(style_metrics(df_final), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════
    # 공유 유틸: 포트폴리오 신호 계산 (asset_view 등 외부에서도 사용)
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def compute_portfolio_signals(portfolio_name, data_dict, fg_df, vxn_df, macro_df, news_df, start_date,
                                   cur_prices=None):
        """사이드바 설정을 반영한 신호 계산.

        cur_prices 있음 → inject_virtual_close 후 실시간 신호
        cur_prices 없음 → 마지막 종가 기준 신호

        Returns:
            dict | None  — predict_portfolio_today() 결과, 실패 시 None
        """
        from core.signal_service import get_cached_signals

        fee_rate       = st.session_state.get('global_fee_rate', 0.001) or 0.001
        integer_shares = st.session_state.get('global_integer_shares', True)
        sidebar_start  = st.session_state.get('start_input', None)
        if sidebar_start:
            start_date = sidebar_start

        return get_cached_signals(
            portfolio_name, data_dict, fg_df, vxn_df, macro_df, news_df, start_date,
            cur_prices=cur_prices,
        )

