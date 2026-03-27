"""
Streamlit session_state 키 중앙 관리

모든 session_state 키를 이 파일에서 상수로 정의합니다.
문자열 오타로 인한 버그를 방지하고, 키 목록을 한 곳에서 파악할 수 있습니다.

사용법:
    from core.session_keys import SK
    st.session_state[SK.SYNC_AUTO_QTY] = auto_qty
    val = st.session_state.get(SK.GLOBAL_FEE_RATE, 0.0)
"""


class SK:
    # ── 글로벌 설정 (app.py ↔ history_view / portfolio_view) ──────────
    GLOBAL_USE_FEE       = 'global_use_fee'
    GLOBAL_FEE_PCT       = 'global_fee_pct'
    GLOBAL_FEE_RATE      = 'global_fee_rate'
    GLOBAL_INTEGER_SHARES = 'global_integer_shares'

    # ── 전략 파라미터 (app.py ↔ history_view / portfolio_view) ────────
    CURRENT_PARAMS         = 'current_params'
    CURRENT_BOND_PARAMS    = 'current_bond_params'
    CURRENT_BOND_LEV_PARAMS = 'current_bond_lev_params'
    LOADED_STRAT_NAME      = 'loaded_strat_name'
    START_INPUT            = 'start_input'

    # ── 백테스트 결과 (app.py ↔ backtest_view / history_view) ─────────
    LAST_BT_RESULT         = 'last_bt_result'
    LAST_BOND_HASH         = 'last_bond_hash'
    LAST_RUN_HASH_HISTORY  = 'last_run_hash_history'
    LAST_HISTORY_RESULT    = 'last_history_result'
    PREVIEW_RESULT         = 'preview_result'

    # ── 트리거 플래그 (app.py → 각 뷰) ───────────────────────────────
    TRIGGER_PREVIEW         = 'trigger_preview'
    TRIGGER_RECALC_BOND     = 'trigger_recalc_bond'
    TRIGGER_RECALC_HISTORY  = 'trigger_recalc_history'

    # ── 포트폴리오 뷰 (portfolio_view ↔ portfolio_realtime_view) ──────
    PORTFOLIO_CONFIGS              = 'portfolio_configs'
    PORTFOLIO_TOTAL_CAPITAL        = 'portfolio_total_capital'
    PORTFOLIO_REBALANCE_PRESET     = 'portfolio_rebalance_preset'
    PORTFOLIO_RESULT               = 'portfolio_result'
    PORTFOLIO_INIT_VER             = 'portfolio_init_ver'
    PORTFOLIO_TREND_DEFENSE_WEIGHTS = 'portfolio_trend_defense_weights'
    PORTFOLIO_LOAD_VERSION         = 'p_load_version'
    PORTFOLIO_REALTIME_RESULT      = 'portfolio_realtime_result'
    PORTFOLIO_SELECTED_NAME        = 'portfolio_selected_name'
    PORTFOLIO_DRAWDOWN_BOOST       = 'portfolio_drawdown_boost'

    # ── 자산 관리 뷰 (asset_view) ─────────────────────────────────────
    CURRENT_SIGNALS           = 'current_signals'
    CURRENT_SIGNALS_PORTFOLIO = 'current_signals_portfolio'

    # ── 자산 동기화 뷰 (sync_view) ────────────────────────────────────
    SYNC_AUTO_QTY    = 'sync_auto_qty'
    SYNC_PRICES      = 'sync_prices'
    SYNC_WEIGHTS     = 'sync_weights'
    SYNC_CAPITAL     = 'sync_capital'
    SYNC_ACCOUNT_ID  = 'sync_account_id'
    SYNC_RESIDUAL    = 'sync_residual'
    SYNC_SIGNALS_INFO = 'sync_signals_info'

    # ── AI 인텔리전스 뷰 ──────────────────────────────────────────────
    LAST_GENERATED_DATE  = 'last_generated_date'
    MANUAL_ANALYST_INPUT = 'manual_analyst_input'

    # ── 차트 / 비교 뷰 ───────────────────────────────────────────────
    SHOW_COMP_MANAGER = 'show_comp_manager'
    COMPARISON_LIST   = 'comparison_list'

    # ── 모의투자 DCA (적립식) 설정 ───────────────────────────────────
    SIM_DCA_USE       = 'sim_dca_use'
    SIM_DCA_KRW       = 'sim_dca_krw'
    SIM_DCA_USD       = 'sim_dca_usd'
    SIM_DCA_THRESHOLD = 'sim_dca_threshold_pct'

    # ── 초기 로드 플래그 ─────────────────────────────────────────────
    INITIAL_BOND_LOAD_V28 = 'initial_bond_load_v28'

    # ── 신호 캐시 (signal_service / data) — 동적 키 생성 헬퍼 ─────────
    @staticmethod
    def sig_cache(portfolio_name: str, mode: str) -> str:
        """mode: 'closing' | 'realtime'"""
        return f'_sig_cache_{portfolio_name}_{mode}'

    @staticmethod
    def sig_cache_ts(portfolio_name: str, mode: str) -> str:
        return f'_sig_cache_{portfolio_name}_{mode}_ts'

    @staticmethod
    def cur_prices_cache(tickers_key: str, bucket: str) -> str:
        return f'_cur_prices_{tickers_key}_{bucket}'

    @staticmethod
    def cur_prices_cache_ts(tickers_key: str, bucket: str) -> str:
        return f'_cur_prices_{tickers_key}_{bucket}_ts'
