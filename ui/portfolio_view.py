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
from utils.metrics import calculate_metrics

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
        _portfolio_init_ver = 'v3'
        _needs_portfolio_init = (
            'portfolio_configs' not in st.session_state
            or st.session_state.get('portfolio_init_ver') != _portfolio_init_ver
            or not any(c.get('type') == 'bond' and 'lev_params' in c
                       for c in st.session_state.get('portfolio_configs', []))
        )
        if _needs_portfolio_init:
            _default_bond_params = {'ffv_lo': -1.0, 'ffv_hi': 1.5, 'yc_inv': -0.3, 'yc_steep': 0.4, 'tlt_rsi_max': 60}
            _default_bond_lev_params = copy.deepcopy(st.session_state.get('current_bond_lev_params')) or None
            st.session_state.portfolio_configs = [
                {'name': '전략 A (TQQQ 베이스)', 'weight': 0.7, 'type': 'equity', 'params': copy.deepcopy(st.session_state.current_params)},
                {'name': '전략 B (TLT/TBF 채권)', 'weight': 0.3, 'type': 'bond',
                 'params': copy.deepcopy(st.session_state.get('current_bond_params', _default_bond_params)),
                 'lev_params': _default_bond_lev_params},
            ]
            st.session_state.portfolio_init_ver = _portfolio_init_ver
        
        if 'portfolio_total_capital' not in st.session_state:
            st.session_state.portfolio_total_capital = 10000.0

        tab1, tab2, tab3 = st.tabs(["📊 통합 대시보드", "🧩 전략 구성 및 비중", "⚙️ 전략별 세부 설정"])

        with tab2:
            st.subheader("💾 포트폴리오 관리 (DB)")
            
            p_management_expander = st.expander("📂 포트폴리오 불러오기 / 저장하기", expanded=True)
            with p_management_expander:
                col_load, col_save = st.columns(2)
                
                with col_load:
                    st.markdown("##### 📥 불러오기")
                    p_list = StrategyStorage.list_portfolios()
                    if p_list:
                        selected_p = st.selectbox("저장된 포트폴리오 선택", p_list, key="p_load_sel")
                        c1, c2 = st.columns(2)
                        if c1.button("📂 불러오기", use_container_width=True):
                            loaded = StrategyStorage.load_portfolio(selected_p)
                            if loaded:
                                # (1) 핵심 데이터 교체 (deepcopy 필수)
                                loaded_configs = copy.deepcopy(loaded['configs'])

                                # (2) 불러온 configs에 lev_params 누락 시 현재 세션 값으로 보완
                                #     → 누락 시 _needs_portfolio_init=True 가 다시 덮어쓰는 버그 방지
                                for cfg in loaded_configs:
                                    if cfg.get('type') == 'bond' and 'lev_params' not in cfg:
                                        cfg['lev_params'] = copy.deepcopy(st.session_state.get('current_bond_lev_params'))

                                st.session_state.portfolio_configs = loaded_configs
                                st.session_state.portfolio_total_capital = loaded['total_capital']
                                st.session_state.portfolio_rebalance_preset = loaded.get('rebalance_preset', 'none')
                                # portfolio_init_ver 동기화 → 재초기화 방지
                                st.session_state.portfolio_init_ver = _portfolio_init_ver

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
                        if c2.button("🗑️ 삭제", use_container_width=True):
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
                            # 저장 직전 실시간 동기화
                            saved_name = StrategyStorage.save_portfolio(
                                save_name, 
                                st.session_state.portfolio_configs, 
                                st.session_state.portfolio_total_capital,
                                st.session_state.get('portfolio_rebalance_preset', 'none')
                            )
                            if saved_name:
                                st.success(f"✅ 현재 배분 상태가 '**{saved_name}**'(으)로 저장되었습니다.")
                                st.rerun()

            st.write("---")
            st.subheader("🧩 포트폴리오 전략 구성")
            col_cap, col_rb, col_add = st.columns([2, 2, 1])
            total_cap = col_cap.number_input("총 투자 자산 ($)", min_value=100.0, value=st.session_state.portfolio_total_capital, step=100.0)
            st.session_state.portfolio_total_capital = total_cap

            PRESET_LABELS = {
                'none':        '🚫 리밸런싱 없음',
                'annual':      '📅 연간 (1년)',
                'semi_annual': '📅 반기 (6개월)',
                'quarterly':   '📅 분기 (3개월)',
                'band_5pct':   '📊 밴드 ±5%',
                'band_10pct':  '📊 밴드 ±10%',
                'asymmetric':  '⚖️ 비대칭 (상 15% / 하 5%)',
            }
            preset_keys = list(PRESET_LABELS.keys())
            current_preset = st.session_state.get('portfolio_rebalance_preset', 'annual')
            preset_idx = preset_keys.index(current_preset) if current_preset in preset_keys else 1
            selected_preset = col_rb.selectbox(
                "🔄 리밸런싱 전략",
                options=preset_keys,
                format_func=lambda k: PRESET_LABELS[k],
                index=preset_idx,
                key='portfolio_rebalance_preset'
            )
            
            if col_add.button("➕ 전략 추가", use_container_width=True):
                new_idx = len(st.session_state.portfolio_configs) + 1
                st.session_state.portfolio_configs.append({
                    'name': f'신규 전략 {new_idx}',
                    'weight': 0.0,
                    'type': 'equity', # [신규] 기본값 주식
                    'params': copy.deepcopy(st.session_state.current_params)
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
                    
                    # (3) 데이터 동기화 (타입 또는 이름 변경 시에만 DB 호출 및 rerun)
                    if new_type != curr_type or (strat_options and selected_name != config['name']):
                        loaded_strat = StrategyStorage.load_strategy(selected_name)
                        if loaded_strat:
                            configs[i]['name'] = selected_name
                            configs[i]['type'] = new_type
                            configs[i]['params'] = copy.deepcopy(loaded_strat)
                            st.rerun()

                    # (4) 비중 표시 (실시간 반영)
                    configs[i]['weight'] = col_w.number_input(f"비중 (0~1.0) #{i+1}", value=config['weight'], min_value=0.0, max_value=1.0, step=0.05, key=f"p_weight_{i}")
                    total_weight += configs[i]['weight']
                    
                    st.write(f"할당 자산: `${total_cap * configs[i]['weight']:,.2f}`")
                    
                    if col_del.button("🗑️ 삭제", key=f"btn_p_del_{i}", use_container_width=True):
                        to_delete = i
            
            if to_delete != -1:
                st.session_state.portfolio_configs.pop(to_delete)
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

        with tab3:
            st.subheader("⚙️ 전략별 세부 파라미터 설정")
            configs = st.session_state.portfolio_configs
            
            if not configs:
                st.info("먼저 '전략 구성' 탭에서 전략을 추가해 주세요.")
            else:
                # [개선] 가로형 라디오 버튼으로 선택 (이름 정제)
                def config_label(i):
                    c = configs[i]
                    return f"{'📈 주식' if c.get('type')=='equity' else '💰 채권'} ({c['name']})"
                
                # 사용자가 단순 '주식', '채권'으로 원하므로 label만 변경하되 중복 방지 위해 인덱스 활용
                selected_idx = st.radio(
                    "🔧 튜닝할 전략 선택", 
                    range(len(configs)), 
                    horizontal=True, 
                    format_func=lambda i: f"{'📈 주식' if configs[i].get('type')=='equity' else '💰 채권'}",
                    key="p_strat_tune_sel_idx"
                )
                
                config = configs[selected_idx]
                idx = selected_idx
                
                # [추가] 글로벌 전략에서 다시 불러오기 (교체 기능)
                current_type = config.get('type', 'equity')
                global_strats = StrategyStorage.list_strategies(category=current_type)
                
                if global_strats:
                    # 현재 선택된 전략이 목록의 어디에 있는지 확인
                    try:
                        g_sel_idx = global_strats.index(config['name'])
                    except ValueError:
                        g_sel_idx = 0
                    
                    new_global_name = st.selectbox(
                        "🔄 글로벌 전략에서 다른 전략으로 교체", 
                        global_strats, 
                        index=g_sel_idx,
                        key=f"p_strat_swap_{idx}"
                    )
                    
                    # 전략 교체 시 파라미터 로드 및 갱신
                    if new_global_name != config['name']:
                        loaded = StrategyStorage.load_strategy(new_global_name)
                        if loaded:
                            configs[idx]['name'] = new_global_name
                            configs[idx]['params'] = copy.deepcopy(loaded)
                            st.rerun()

                # 선택된 전략에 대한 설정 화면 렌더링
                st.write("---")
                st.markdown(f"#### 🛠️ {config['name']} 파라미터 튜닝")
                prefix = f"strat_detail_{idx}_"
                
                st.info(f"💡 현재 설정은 글로벌 전략 **'{config['name']}'**에서 로드되었습니다. 아래에서 이 포트폴리오에 맞게 세부 조정이 가능합니다.")
                
                final_type = config.get('type', 'equity')
                
                if final_type == 'bond':
                    with st.form(f"{prefix}bond_form"):
                        new_params, new_lev_params = TesterView.render_bond_settings(
                            config.get('params', {}),
                            current_lev_params=config.get('lev_params'),
                            key_prefix=prefix,
                        )
                        if st.form_submit_button("⚙️ 채권 설정 업데이트 (Update)"):
                            config['params'] = new_params
                            config['lev_params'] = new_lev_params
                            st.success(f"✅ {config['name']}의 채권 설정이 반영되었습니다.")
                            st.rerun()
                else:
                    st.markdown("##### 📥 매매 시그널 및 필터")
                    # 주식 전략은 실시간 동기화 방식 유지
                    new_params_top = TesterView.render_top_part(config.get('params', {}), key_prefix=prefix)
                    new_params_bottom = TesterView.render_bottom_part(config.get('params', {}), key_prefix=prefix)
                    
                    config['params'].update(new_params_top)
                    config['params'].update(new_params_bottom)
                    st.caption(f"💡 {config['name']}의 주식 설정은 수정 즉시 반영됩니다.")

        with tab1:
            PortfolioView.render_dashboard()

    @staticmethod
    def run_simulation(data_dict, fg_df, vxn_df, macro_df, news_df, start_date, end_date):
        configs = st.session_state.portfolio_configs
        total_cap = st.session_state.portfolio_total_capital
        
        # 데이터 슬라이싱 (start_date ~ end_date)
        filtered_data = {t: df.loc[start_date:end_date] for t, df in data_dict.items() if not df.empty}
        f_fg = fg_df.loc[start_date:end_date] if not fg_df.empty else pd.DataFrame()
        f_vxn = vxn_df.loc[start_date:end_date] if not vxn_df.empty else pd.DataFrame()
        f_macro = macro_df.loc[start_date:end_date] if not macro_df.empty else pd.DataFrame()
        
        # Portfolio Manager 초기화
        rebalance_preset = st.session_state.get('portfolio_rebalance_preset', 'annual')
        pm = PortfolioManager(total_capital=total_cap, rebalance_preset=rebalance_preset)
        
        # 채권 전략 기본 파라미터 (V7)
        DEFAULT_BOND_PARAMS = {'ffv_lo': -1.0, 'ffv_hi': 1.5, 'yc_inv': -0.3, 'yc_steep': 0.4, 'tlt_rsi_max': 60}

        # 전략 등록
        for config in configs:
            params = config['params']
            s_type = config.get('type', 'equity')

            kwargs = {
                'data_dict': filtered_data,
                'start_date': start_date,
                'end_date': end_date,
                'allocated_capital': total_cap * config['weight']
            }

            if s_type == 'equity':
                # 항상 최신 current_params 사용 (config['params']는 초기화 시점 복사본으로 stale할 수 있음)
                live_eq_params = st.session_state.get('current_params', params)
                kwargs['params'] = live_eq_params
                kwargs.update({
                    'fg_df': f_fg,
                    'vxn_df': f_vxn,
                    'news_df': news_df.loc[start_date:end_date] if news_df is not None else None,
                    'leverage_asset': live_eq_params.get('leverage_asset', 'TQQQ'),
                    'base_asset': live_eq_params.get('base_asset', 'QQQ'),
                    'smart_params': live_eq_params
                })
            else:
                # 채권 전략: Layer1 params + Layer2 lev_params 모두 전달
                bond_params = config.get('params', config.get('bond_params', DEFAULT_BOND_PARAMS))
                kwargs['params'] = bond_params
                kwargs['lev_params'] = config.get('lev_params')
                kwargs.update({
                    'macro_df': f_macro
                })
                
            pm.register_strategy(name=config['name'], weight=config['weight'], engine_kwargs=kwargs, strat_type=s_type)
        
        with st.spinner("다중 전략 통합 시뮬레이션 구동 중..."):
            try:
                # 1. 통합 포트폴리오 시뮬레이션 (리밸런싱 포함)
                combined_equity, individual_results = pm.run_portfolio()
                
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
                        'allocated_capital': total_cap * config['weight'] # 동일 비중 시작
                    }
                    
                    if s_type == 'equity':
                        # 항상 최신 current_params 사용 (config['params']는 초기화 시점 복사본으로 stale할 수 있음)
                        live_eq_params = st.session_state.get('current_params', params)
                        # [중요] 지표 워밍업을 위해 전체 data_dict 전달 (filtered_data는 start_date부터 잘려 SMA200/MACD 등 부정확)
                        # engine 내부에서 start_date/end_date로 시뮬레이션 구간을 자체 필터링함
                        st_res, _, _ = StrategyEngine.run_golden_strategy(
                            data_dict=data_dict,
                            fg_df=fg_df, vxn_df=vxn_df,
                            news_df=news_df,
                            leverage_asset=live_eq_params.get('leverage_asset', 'TQQQ'),
                            base_asset=live_eq_params.get('base_asset', 'QQQ'),
                            cash_ratio=live_eq_params.get('cash_ratio', 0.0),
                            start_date=start_date, end_date=end_date,
                            params=live_eq_params, smart_params=live_eq_params,
                            trade_at=live_eq_params.get('trade_at', '종가'),
                            allocated_capital=base_kwargs['allocated_capital'],
                        )
                    else:
                        bond_p = config.get('params', config.get('bond_params', DEFAULT_BOND_PARAMS))
                        bond_lev_p = config.get('lev_params')
                        st_res, _, _ = StrategyEngine.run_bond_strategy(**base_kwargs, macro_df=f_macro, params=bond_p, lev_params=bond_lev_p)
                    
                    standalone_results[name] = st_res
                
                # QQQ 벤치마크 데이터 저장 (연도별 비교용)
                qqq_df = filtered_data.get('QQQ', pd.DataFrame())
                st.session_state.portfolio_result = {
                    'combined': combined_equity,
                    'individuals': individual_results,
                    'standalone': standalone_results,  # 독립 결과 추가
                    'qqq_df': qqq_df
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
        m1, m2, m3, m4 = st.columns(4)
        total_ret = (metrics['Final Value']/st.session_state.portfolio_total_capital - 1)*100
        m1.metric("최종 자산", f"${metrics['Final Value']:,.2f}")
        m2.metric("누적 수익률", f"{total_ret:+.1f}%")
        m3.metric("연평균 (CAGR)", f"{metrics['CAGR']*100:.1f}%")
        m4.metric("최대 낙폭 (MDD)", f"{metrics['MDD']*100:.1f}%")

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
            hist_df = data['history']
            
            # [수정] 성과 컬럼 유연하게 탐색
            val_col = 'Value'
            if 'Value' not in hist_df.columns:
                if 'PortfolioValue' in hist_df.columns:
                    val_col = 'PortfolioValue'
                elif 'Close' in hist_df.columns:
                    val_col = 'Close'
            
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
                .applymap(color_ret, subset=ret_cols) \
                .applymap(color_mdd, subset=mdd_cols) \
                .applymap(color_calmar, subset=calmar_cols) \
                .format("{:+.1%}", subset=ret_cols + mdd_cols) \
                .format("{:.2f}", subset=calmar_cols, na_rep="-")

        st.dataframe(style_metrics(df_final), use_container_width=True)

