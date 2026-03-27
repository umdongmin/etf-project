"""
로그 분석 뷰 — 주식/채권/포트폴리오 전략 거래 로그를 통합해서 보여주는 뷰.
session_state에서 직접 결과를 읽으므로 엔진 재실행 없이 최신 결과를 표시합니다.
"""
import streamlit as st
import pandas as pd

from core.session_keys import SK
from ui.backtest_view import BacktestView
from ui.portfolio_view import PortfolioView


class LogView:

    @staticmethod
    def render():
        st.title("📋 로그 분석")
        st.caption("각 전략 메뉴에서 시뮬레이션을 실행한 후 결과 로그를 이곳에서 확인하세요.")

        tab_stock, tab_bond, tab_portfolio = st.tabs([
            "📈 주식 전략 로그",
            "💰 채권 전략 로그",
            "📂 포트폴리오 로그",
        ])

        with tab_stock:
            LogView._render_stock_log()

        with tab_bond:
            LogView._render_bond_log()

        with tab_portfolio:
            LogView._render_portfolio_log()

    # ── 주식 전략 로그 ────────────────────────────────────────────────────
    @staticmethod
    def _render_stock_log():
        _mode = "실시간" if st.session_state.get('_data_mode_realtime', False) else "증분"
        st.caption(f"📡 데이터 모드: **{_mode}** — 모드 변경 후 '주식 전략 설정'에서 시뮬레이션을 다시 실행하면 로그에 반영됩니다.")

        result = st.session_state.get(SK.LAST_HISTORY_RESULT)
        if not result:
            st.info("주식 전략 시뮬레이션 결과가 없습니다. '주식 전략 설정' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        golden_history = result.get('history', pd.DataFrame())
        closed_trades = result.get('closed_trades', pd.DataFrame())
        base_asset = result.get('base', 'QQQ')
        bh_histories = result.get('bh_histories', {})

        if golden_history.empty:
            st.info("주식 전략 시뮬레이션 결과가 없습니다. '주식 전략 설정' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        BacktestView.render_closed_sets(closed_trades)
        BacktestView.render_execution_log(golden_history, base_asset)

        st.divider()
        BacktestView.render_daily_log(golden_history, bh_histories, base_asset)

    # ── 채권 전략 로그 ────────────────────────────────────────────────────
    @staticmethod
    def _render_bond_log():
        bt_result = st.session_state.get(SK.LAST_BT_RESULT, {})
        bond_result = bt_result.get('bond') if bt_result else None

        if not bond_result:
            st.info("채권 전략 시뮬레이션 결과가 없습니다. '채권 전략 설정' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        history = bond_result.get('history', pd.DataFrame())
        closed_trades = bond_result.get('closed_trades', pd.DataFrame())
        benchmarks = bond_result.get('benchmarks', {})

        if history is None or history.empty:
            st.info("채권 전략 시뮬레이션 결과가 없습니다. '채권 전략 설정' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        BacktestView.render_closed_sets(closed_trades)
        BacktestView.render_execution_log(history, 'TLT', is_bond=True)

        st.divider()
        BacktestView.render_daily_log(history, benchmarks, 'TLT')

    # ── 포트폴리오 로그 ───────────────────────────────────────────────────
    @staticmethod
    def _render_portfolio_log():
        portfolio_result = st.session_state.get(SK.PORTFOLIO_RESULT, {})
        if not portfolio_result:
            st.info("포트폴리오 시뮬레이션 결과가 없습니다. '포트폴리오 매니저' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        individuals = portfolio_result.get('individuals', {})
        rebalance_events = portfolio_result.get('rebalance_events', [])
        # configs는 portfolio_result에 저장되지 않으므로 session_state에서 직접 읽음
        configs = st.session_state.get('portfolio_configs', [])

        if not individuals and not rebalance_events:
            st.info("포트폴리오 시뮬레이션 결과가 없습니다. '포트폴리오 매니저' 메뉴에서 먼저 시뮬레이션을 실행하세요.")
            return

        # 리밸런싱 거래 내역
        if rebalance_events:
            st.subheader("🔄 리밸런싱 거래 내역")
            PortfolioView._render_rebalance_log(rebalance_events)
            st.divider()

        # 전략별 거래 체결 내역
        if individuals:
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
