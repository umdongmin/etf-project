import copy
import streamlit as st
import pandas as pd

from core.engine import StrategyEngine
from core.storage import StrategyStorage
from ui.tester_view import TesterView
from utils.metrics import calculate_metrics
from utils.tax_calculator import calculate_tax, estimate_unrealized_tax
from core.defaults import DEFAULT_BOND_PARAMS
from core.engine_kwargs_builder import build_equity_kwargs, build_bond_kwargs, extract_smart_params


class SimulationView:
    """모의투자 시뮬레이터 UI — M-1 투자금 입력 + M-2 전략 선택 & 백테스트"""

    DEFAULT_EXCHANGE_RATE = 1450.0

    # ──────────────────────────────────────────────────────────────────────
    # 진입점
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def render(data_dict, fg_df, vxn_df, macro_df, news_df):
        st.title("💰 모의투자 시뮬레이터")
        st.caption("투자금·환율·전략·기간을 설정하면 세후 실수익을 계산합니다.")
        st.divider()

        # M-1: 투자금 입력
        invest_cfg = SimulationView._render_investment_section()

        st.divider()

        # M-2: 전략 선택 & 백테스트
        SimulationView._render_strategy_section(
            invest_cfg, data_dict, fg_df, vxn_df, macro_df, news_df
        )

        # M-3: 세금 계산 (시뮬레이션 결과가 있을 때만)
        if "sim_result" in st.session_state:
            st.divider()
            SimulationView._render_tax_section(invest_cfg)

        # M-4: 대시보드 (시뮬레이션 결과가 있을 때만)
        if "sim_result" in st.session_state:
            st.divider()
            SimulationView._render_dashboard_section(invest_cfg, data_dict)

    # ──────────────────────────────────────────────────────────────────────
    # M-1: 투자금 입력
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_investment_section() -> dict:
        """환율·투자금 입력 UI. 달러 기준 투자금 dict 반환."""

        # 세션 초기화
        if "sim_krw" not in st.session_state:
            st.session_state.sim_krw = 10_000_000.0
        if "sim_usd" not in st.session_state:
            st.session_state.sim_usd = round(10_000_000.0 / SimulationView.DEFAULT_EXCHANGE_RATE, 2)
        if "sim_exchange_rate" not in st.session_state:
            st.session_state.sim_exchange_rate = SimulationView.DEFAULT_EXCHANGE_RATE
        if "sim_input_mode" not in st.session_state:
            st.session_state.sim_input_mode = "원화 (KRW)"

        # ① 환율
        st.subheader("① 환율 설정")
        col_rate, col_hint = st.columns([2, 3])
        with col_rate:
            exchange_rate = st.number_input(
                "USD/KRW 환율 (원)", min_value=500.0, max_value=3000.0,
                value=st.session_state.sim_exchange_rate,
                step=1.0, format="%.1f", key="input_exchange_rate",
                help="매수 시점 기준 환율. 실시간 환율은 네이버 금융 또는 하나은행 고시환율 참고."
            )
        with col_hint:
            st.info(f"💡 현재 설정 환율: **{exchange_rate:,.1f} 원/달러**\n\n"
                    "실시간 환율은 네이버 금융 또는 하나은행 고시환율을 참고하세요.")

        if exchange_rate != st.session_state.sim_exchange_rate:
            st.session_state.sim_exchange_rate = exchange_rate
            if st.session_state.sim_input_mode == "원화 (KRW)":
                st.session_state.sim_usd = round(st.session_state.sim_krw / exchange_rate, 2)
            else:
                st.session_state.sim_krw = round(st.session_state.sim_usd * exchange_rate, 0)

        st.divider()

        # ② 투자금
        st.subheader("② 투자금 입력")
        input_mode = st.radio(
            "입력 기준 통화", ["원화 (KRW)", "달러 (USD)"],
            index=0 if st.session_state.sim_input_mode == "원화 (KRW)" else 1,
            horizontal=True, key="radio_input_mode"
        )
        st.session_state.sim_input_mode = input_mode

        col_krw, col_arrow, col_usd = st.columns([5, 1, 5])

        if input_mode == "원화 (KRW)":
            with col_krw:
                krw_input = st.number_input(
                    "투자 원금 (원화)", min_value=0.0, max_value=10_000_000_000.0,
                    value=st.session_state.sim_krw, step=100_000.0,
                    format="%.0f", key="input_krw"
                )
            st.session_state.sim_krw = krw_input
            st.session_state.sim_usd = round(krw_input / exchange_rate, 2)
            with col_arrow:
                st.markdown("<div style='text-align:center;padding-top:32px;font-size:24px;'>→</div>",
                            unsafe_allow_html=True)
            with col_usd:
                st.metric("달러 환산 투자금 (USD)", f"$ {st.session_state.sim_usd:,.2f}")
        else:
            with col_usd:
                usd_input = st.number_input(
                    "투자 원금 (달러)", min_value=0.0, max_value=10_000_000.0,
                    value=st.session_state.sim_usd, step=100.0,
                    format="%.2f", key="input_usd"
                )
            st.session_state.sim_usd = usd_input
            st.session_state.sim_krw = round(usd_input * exchange_rate, 0)
            with col_arrow:
                st.markdown("<div style='text-align:center;padding-top:32px;font-size:24px;'>←</div>",
                            unsafe_allow_html=True)
            with col_krw:
                st.metric("원화 환산 투자금 (KRW)", f"₩ {st.session_state.sim_krw:,.0f}")

        # ③ 요약
        st.divider()
        SimulationView._render_investment_summary(
            st.session_state.sim_krw, st.session_state.sim_usd, exchange_rate
        )

        return {
            "allocated_capital": st.session_state.sim_usd,
            "krw": st.session_state.sim_krw,
            "usd": st.session_state.sim_usd,
            "exchange_rate": exchange_rate,
        }

    @staticmethod
    def _render_investment_summary(krw: float, usd: float, rate: float):
        st.subheader("③ 투자금 설정 요약")
        if krw < 1_000_000:
            scale = "소액 투자"
        elif krw < 10_000_000:
            scale = "개인 투자"
        elif krw < 100_000_000:
            scale = "중형 투자"
        else:
            scale = "대형 투자"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("투자 원금 (KRW)", f"₩ {krw:,.0f}")
        c2.metric("투자 원금 (USD)", f"$ {usd:,.2f}")
        c3.metric("적용 환율", f"{rate:,.1f} 원/달러")
        c4.metric("투자 규모", scale)

        if krw >= 10_000:
            man = krw / 10_000
            label = f"{man/10000:,.1f}억원" if man >= 10_000 else f"{man:,.1f}만원"
            st.caption(f"📌 {label} 규모 | ${usd:,.0f} (환율 {rate:,.0f}원 기준)")

    # ──────────────────────────────────────────────────────────────────────
    # M-2: 전략 선택 & 백테스트 실행
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_strategy_section(invest_cfg, data_dict, fg_df, vxn_df, macro_df, news_df):
        st.subheader("④ 전략 선택")

        # ── 전략 모드 ──────────────────────────────────────────────────
        MODE_OPTIONS = {
            "포트폴리오 통합 (TQQQ + Bond)": "portfolio",
            "TQQQ 단독": "equity",
            "Bond 단독": "bond",
        }
        mode_label = st.radio(
            "운용 전략",
            list(MODE_OPTIONS.keys()),
            index=0,
            horizontal=True,
            key="sim_mode"
        )
        mode = MODE_OPTIONS[mode_label]

        st.divider()

        # ── 전략 파일 선택 ─────────────────────────────────────────────
        eq_names   = StrategyStorage.list_strategies(category='equity') or ["(없음)"]
        bond_names = StrategyStorage.list_strategies(category='bond')   or ["(없음)"]

        eq_params   = None
        bond_params = None
        eq_weight   = 1.0
        bond_weight = 0.0
        p_data      = None
        configs     = []

        if mode == "equity":
            col_s, _ = st.columns([3, 2])
            with col_s:
                eq_sel = st.selectbox("주식 전략 선택", eq_names, key="sim_eq_sel")
            eq_params = StrategyStorage.load_strategy(eq_sel) or copy.deepcopy(
                st.session_state.get("current_params", {})
            )
            SimulationView._render_strategy_badge("equity", eq_sel, 1.0)

        elif mode == "bond":
            col_s, _ = st.columns([3, 2])
            with col_s:
                bond_sel = st.selectbox("채권 전략 선택", bond_names, key="sim_bond_sel")
            bond_params = StrategyStorage.load_strategy(bond_sel) or {}
            SimulationView._render_strategy_badge("bond", bond_sel, 1.0)

        else:  # portfolio — Supabase 저장 포트폴리오 선택
            portfolio_names = StrategyStorage.list_portfolios()

            if not portfolio_names:
                st.warning("⚠️ 저장된 포트폴리오가 없습니다. 포트폴리오 매니저에서 먼저 저장해 주세요.")
                eq_params, bond_params, eq_weight, bond_weight = None, None, 0.78, 0.22
            else:
                # A-공격형 우선, 없으면 첫 번째
                _default_p = next((n for n in portfolio_names if "A" in n and "공격" in n), None) \
                          or next((n for n in portfolio_names if "A" in n), None) \
                          or portfolio_names[0]
                _default_idx = portfolio_names.index(_default_p)

                col_sel, _ = st.columns([3, 2])
                with col_sel:
                    p_sel = st.selectbox(
                        "포트폴리오 선택",
                        portfolio_names,
                        index=_default_idx,
                        key="sim_portfolio_sel",
                    )

                # 포트폴리오 로드
                p_data = StrategyStorage.load_portfolio(p_sel)

                if not p_data:
                    st.error(f"포트폴리오 '{p_sel}' 로드 실패")
                    eq_params, bond_params, eq_weight, bond_weight = None, None, 0.78, 0.22
                else:
                    configs = p_data.get("configs", [])
                    eq_cfg   = next((c for c in configs if c["type"] == "equity"), None)
                    bond_cfg = next((c for c in configs if c["type"] == "bond"),   None)

                    eq_weight   = eq_cfg["weight"]   if eq_cfg   else 0.78
                    bond_weight = bond_cfg["weight"]  if bond_cfg  else 0.22
                    eq_params   = eq_cfg["params"]   if eq_cfg   else copy.deepcopy(st.session_state.get("current_params", {}))
                    bond_params = bond_cfg["params"]  if bond_cfg  else {}

                    # 포트폴리오 구성 정보 표시
                    st.success(f"✅ 포트폴리오 로드 완료: **{p_sel}**")

                    info_cols = st.columns(len(configs))
                    for i, cfg in enumerate(configs):
                        icon  = "📈" if cfg["type"] == "equity" else "💰"
                        label = "주식" if cfg["type"] == "equity" else "채권"
                        info_cols[i].metric(
                            f"{icon} {label} 전략",
                            cfg["name"],
                            f"비중 {cfg['weight']:.0%}",
                        )

                    # 배분 금액 미리보기
                    total_usd = invest_cfg["allocated_capital"]
                    c1, c2 = st.columns(2)
                    c1.info(f"📈 주식 배분: **${total_usd * eq_weight:,.0f}**\n\n₩ {invest_cfg['krw'] * eq_weight:,.0f}")
                    c2.info(f"💰 채권 배분: **${total_usd * bond_weight:,.0f}**\n\n₩ {invest_cfg['krw'] * bond_weight:,.0f}")

        st.divider()

        # ── 기간 선택 ──────────────────────────────────────────────────
        st.subheader("⑤ 분석 기간")
        start_date, end_date = TesterView.render_period_selector(key_suffix="sim")
        st.caption(f"📅 선택 기간: **{start_date} ~ {end_date}**")

        st.divider()

        # ── 실행 버튼 ──────────────────────────────────────────────────
        col_btn, col_clear = st.columns([3, 1])
        run_clicked   = col_btn.button(
            "🚀 시뮬레이션 실행", type="primary", use_container_width=True, key="sim_run_btn"
        )
        clear_clicked = col_clear.button(
            "🗑️ 결과 초기화", use_container_width=True, key="sim_clear_btn"
        )

        if clear_clicked:
            st.session_state.pop("sim_result", None)
            st.rerun()

        if run_clicked:
            SimulationView._run_backtest(
                mode=mode,
                invest_cfg=invest_cfg,
                eq_params=eq_params,
                bond_params=bond_params,
                eq_weight=eq_weight,
                bond_weight=bond_weight,
                data_dict=data_dict,
                fg_df=fg_df,
                vxn_df=vxn_df,
                macro_df=macro_df,
                news_df=news_df,
                start_date=start_date,
                end_date=end_date,
                p_data=p_data if mode == "portfolio" else None,
                configs=configs if mode == "portfolio" else None,
            )

        # ── 결과 표시 ──────────────────────────────────────────────────
        if "sim_result" in st.session_state:
            SimulationView._render_result_summary(invest_cfg)

    # ──────────────────────────────────────────────────────────────────────
    # 백테스트 실행 로직
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _run_backtest(mode, invest_cfg, eq_params, bond_params,
                      eq_weight, bond_weight,
                      data_dict, fg_df, vxn_df, macro_df, news_df,
                      start_date, end_date,
                      p_data=None, configs=None):
        total_usd = invest_cfg["allocated_capital"]

        with st.spinner("시뮬레이션 실행 중..."):
            try:
                if mode == "equity":
                    history, closed_trades, _ = StrategyEngine.run_golden_strategy(
                        **build_equity_kwargs(
                            data_dict=data_dict, fg_df=fg_df, vxn_df=vxn_df, news_df=news_df,
                            params=eq_params, start_date=start_date, end_date=end_date,
                            allocated_capital=total_usd,
                            fee_rate=st.session_state.get("global_fee_rate", 0.0),
                            integer_shares=st.session_state.get("global_integer_shares", False),
                        )
                    )
                    metrics = calculate_metrics(history)
                    st.session_state.sim_result = {
                        "mode": mode,
                        "history": history,
                        "closed_trades": closed_trades,
                        "metrics": metrics,
                        "eq_weight": 1.0,
                        "bond_weight": 0.0,
                    }

                elif mode == "bond":
                    if macro_df is None or macro_df.empty:
                        st.error("⚠️ 채권 전략은 FRED 매크로 데이터가 필요합니다.")
                        return
                    history, closed_trades, _ = StrategyEngine.run_bond_strategy(
                        **build_bond_kwargs(
                            data_dict=data_dict, macro_df=macro_df,
                            params=bond_params, start_date=start_date, end_date=end_date,
                            lev_params=lev_params,
                            allocated_capital=total_usd,
                            fee_rate=st.session_state.get("global_fee_rate", 0.0),
                            integer_shares=st.session_state.get("global_integer_shares", False),
                        )
                    )
                    metrics = calculate_metrics(history)
                    st.session_state.sim_result = {
                        "mode": mode,
                        "history": history,
                        "closed_trades": closed_trades,
                        "metrics": metrics,
                        "eq_weight": 0.0,
                        "bond_weight": 1.0,
                    }

                else:  # portfolio — PortfolioManager로 포트폴리오 매니저와 동일한 결과 산출
                    if macro_df is None or macro_df.empty:
                        st.error("⚠️ 포트폴리오 통합 모드는 FRED 매크로 데이터가 필요합니다.")
                        return

                    from core.portfolio_manager import PortfolioManager

                    fee_rate       = st.session_state.get("global_fee_rate", 0.0)
                    integer_shares = st.session_state.get("global_integer_shares", False)
                    # 포트폴리오 매니저 탭과 동일하게 session_state 우선, 없으면 저장값 사용
                    rebalance_preset = st.session_state.get(
                        "portfolio_rebalance_preset",
                        p_data.get("rebalance_preset", "quarterly")
                    )

                    # 포트폴리오 매니저와 동일한 데이터 슬라이싱
                    filtered_data = {t: df.loc[start_date:end_date] for t, df in data_dict.items() if not df.empty}
                    f_fg    = fg_df.loc[start_date:end_date]    if not fg_df.empty    else pd.DataFrame()
                    f_vxn   = vxn_df.loc[start_date:end_date]   if not vxn_df.empty   else pd.DataFrame()
                    f_macro = macro_df.loc[start_date:end_date] if not macro_df.empty else pd.DataFrame()
                    f_news  = news_df.loc[start_date:end_date]  if news_df is not None and not news_df.empty else None


                    pm = PortfolioManager(
                        total_capital=total_usd,
                        rebalance_preset=rebalance_preset,
                        fee_rate=fee_rate,
                        integer_shares=integer_shares,
                    )

                    # 포트폴리오 매니저와 동일한 전략 등록 방식
                    for cfg in configs:
                        s_type = cfg.get("type", "equity")
                        kwargs = {
                            "data_dict": filtered_data,
                            "start_date": start_date,
                            "end_date":   end_date,
                            "allocated_capital": total_usd * cfg["weight"],
                        }
                        if s_type == "equity":
                            live_p = st.session_state.get("current_params", cfg["params"])
                            kwargs["params"] = live_p
                            kwargs.update({
                                "fg_df":          f_fg,
                                "vxn_df":         f_vxn,
                                "news_df":        f_news,
                                "leverage_asset": live_p.get("leverage_asset", "TQQQ"),
                                "base_asset":     live_p.get("base_asset", "QQQ"),
                                "smart_params":   live_p,
                            })
                        else:
                            bond_p     = cfg.get("params", DEFAULT_BOND_PARAMS)
                            bond_lev_p = cfg.get("lev_params")
                            kwargs["params"]     = bond_p
                            kwargs["lev_params"] = bond_lev_p
                            kwargs["macro_df"]   = f_macro

                        pm.register_strategy(
                            name=cfg["name"],
                            weight=cfg["weight"],
                            engine_kwargs=kwargs,
                            strat_type=s_type,
                        )

                    combined, individual_results, rebalance_events = pm.run_portfolio()
                    metrics = calculate_metrics(combined)

                    # 개별 전략 히스토리 추출 (차트 overlay용)
                    eq_cfg_name   = next((c["name"] for c in configs if c["type"] == "equity"), None)
                    bond_cfg_name = next((c["name"] for c in configs if c["type"] == "bond"),   None)
                    eq_hist   = individual_results.get(eq_cfg_name,   pd.DataFrame())
                    bond_hist = individual_results.get(bond_cfg_name, pd.DataFrame())

                    # 개별 전략 closed_trades 합산
                    all_trades = pd.concat(
                        [r["closed_trades"] for r in individual_results.values()
                         if isinstance(r.get("closed_trades"), pd.DataFrame) and not r["closed_trades"].empty],
                        ignore_index=True
                    ) if individual_results else pd.DataFrame()
                    if not all_trades.empty and "Exit_Date" in all_trades.columns:
                        all_trades = all_trades.sort_values("Exit_Date").reset_index(drop=True)

                    st.session_state.sim_result = {
                        "mode":              mode,
                        "history":           combined,
                        "eq_history":        eq_hist,
                        "bond_history":      bond_hist,
                        "closed_trades":     all_trades,
                        "metrics":           metrics,
                        "eq_weight":         eq_weight,
                        "bond_weight":       bond_weight,
                        "rebalance_preset":  rebalance_preset,
                        "fee_rate":          fee_rate,
                        "integer_shares":    integer_shares,
                        "rebalance_events":  rebalance_events,
                    }

                st.success("✅ 시뮬레이션 완료!")
                # 사용된 설정 확인용
                st.caption(
                    f"⚙️ 적용 설정 — 리밸런싱: **{rebalance_preset}** | "
                    f"수수료: **{fee_rate*100:.3f}%** | "
                    f"정수주식: **{'ON' if integer_shares else 'OFF'}**"
                )

            except Exception as e:
                st.error(f"❌ 시뮬레이션 오류: {e}")
                import traceback
                with st.expander("상세 오류 로그"):
                    st.code(traceback.format_exc())

    @staticmethod
    def _combine_histories(eq_hist: pd.DataFrame, bond_hist: pd.DataFrame, total_usd: float) -> pd.DataFrame:
        """주식·채권 히스토리를 날짜 기준으로 합산해 통합 히스토리 반환."""
        if eq_hist.empty and bond_hist.empty:
            return pd.DataFrame()
        if eq_hist.empty:
            return bond_hist.copy()
        if bond_hist.empty:
            return eq_hist.copy()

        # DatetimeIndex 기준으로 통일 (calculate_metrics는 DatetimeIndex 필요)
        def _to_series(df):
            if "Date" in df.columns:
                return df.set_index("Date")["Value"]
            return df["Value"]  # 이미 DatetimeIndex

        eq_val   = _to_series(eq_hist)
        bond_val = _to_series(bond_hist)

        eq_val.index   = pd.to_datetime(eq_val.index)
        bond_val.index = pd.to_datetime(bond_val.index)

        idx      = eq_val.index.union(bond_val.index)
        eq_val   = eq_val.reindex(idx).ffill()
        bond_val = bond_val.reindex(idx).ffill()

        combined = pd.DataFrame(index=idx)
        combined.index.name    = "Date"
        combined["Value"]      = eq_val + bond_val
        combined["Equity_Val"] = eq_val
        combined["Bond_Val"]   = bond_val
        # DatetimeIndex 유지 — reset_index 하지 않음 (calculate_metrics 호환)
        return combined

    # ──────────────────────────────────────────────────────────────────────
    # 결과 요약 (M-4 전 단계 — 핵심 지표만 표시)
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_result_summary(invest_cfg: dict):
        res     = st.session_state.sim_result
        metrics = res.get("metrics", {})
        history = res.get("history", pd.DataFrame())
        rate    = invest_cfg["exchange_rate"]
        usd_in  = invest_cfg["usd"]
        krw_in  = invest_cfg["krw"]

        if history.empty or metrics is None:
            st.warning("결과 데이터가 없습니다.")
            return

        # Value 컬럼 통일 — PortfolioManager는 PortfolioValue, 엔진은 Value 사용
        val_col = "Value" if "Value" in history.columns else \
                  "PortfolioValue" if "PortfolioValue" in history.columns else None
        if val_col is None:
            st.warning("결과 데이터에 자산 가치 컬럼이 없습니다.")
            return

        st.divider()
        st.subheader("⑥ 시뮬레이션 결과 요약")

        # 최종 자산 (USD)
        final_usd = history[val_col].iloc[-1]
        profit_usd = final_usd - usd_in
        final_krw  = final_usd * rate
        profit_krw = profit_usd * rate
        total_return_pct = (final_usd / usd_in - 1) * 100 if usd_in > 0 else 0.0

        # 핵심 지표 카드
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("투자 원금 (USD)",   f"$ {usd_in:,.0f}")
        c2.metric("최종 자산 (USD)",   f"$ {final_usd:,.0f}",
                  delta=f"$ {profit_usd:+,.0f}")
        c3.metric("누적 수익률",       f"{total_return_pct:+.1f}%")
        c4.metric("연평균 (CAGR)",     f"{metrics.get('CAGR', metrics.get('cagr', 0))*100:.1f}%")
        c5.metric("최대 낙폭 (MDD)",   f"{metrics.get('MDD', metrics.get('mdd', 0))*100:.1f}%")

        st.divider()

        # 원화 환산 결과
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("투자 원금 (KRW)",   f"₩ {krw_in:,.0f}")
        c2.metric("최종 자산 (KRW)",   f"₩ {final_krw:,.0f}",
                  delta=f"₩ {profit_krw:+,.0f}")
        c3.metric("칼마 지수",         f"{metrics.get('MAR', metrics.get('calmar', 0)):.3f}")
        c4.metric("샤프 지수",         f"{metrics.get('Sharpe', metrics.get('sharpe', 0)):.3f}")

        st.caption(f"📌 원화 환산 기준 환율: {rate:,.0f} 원/달러 (입력값 고정 — 실제 환차손익 미반영)")

        # 모드별 비중 정보
        if res["mode"] == "portfolio":
            st.info(
                f"포트폴리오 구성: 주식 **{res['eq_weight']:.0%}** "
                f"(${usd_in * res['eq_weight']:,.0f}) + "
                f"채권 **{res['bond_weight']:.0%}** "
                f"(${usd_in * res['bond_weight']:,.0f})"
            )

    # ──────────────────────────────────────────────────────────────────────
    # M-3: 양도세 계산
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_tax_section(invest_cfg: dict):
        res           = st.session_state.sim_result
        closed_trades = res.get("closed_trades", [])
        history       = res.get("history", pd.DataFrame())
        rate          = invest_cfg["exchange_rate"]
        usd_in        = invest_cfg["usd"]

        st.subheader("⑦ 양도소득세 계산")
        st.caption("한국 거주자 기준 — 연간 250만원 기본공제 후 초과분 과세")

        # ── 세율 입력 ─────────────────────────────────────────────────
        col_rate, col_ded, col_info = st.columns([2, 2, 3])
        with col_rate:
            tax_rate_pct = st.number_input(
                "양도세율 (%)",
                min_value=0.0, max_value=50.0,
                value=st.session_state.get("sim_tax_rate_pct", 22.0),
                step=0.1, format="%.1f",
                key="sim_tax_rate_pct",
                help="양도세 20% + 지방소득세 2% = 기본 22%"
            )
        with col_ded:
            deduction_man = st.number_input(
                "연간 기본공제 (만원)",
                min_value=0.0, max_value=1000.0,
                value=st.session_state.get("sim_deduction_man", 250.0),
                step=10.0, format="%.0f",
                key="sim_deduction_man",
                help="2025년 기준 연간 250만원 기본공제"
            )
        with col_info:
            st.info(
                f"📋 **과세 공식**\n\n"
                f"과세표준 = 연간 실현손익 - {deduction_man:.0f}만원\n\n"
                f"납부세액 = 과세표준 × {tax_rate_pct:.1f}%"
            )

        tax_rate       = tax_rate_pct / 100.0
        deduction_krw  = deduction_man * 10_000.0

        # ── 세금 계산 실행 ────────────────────────────────────────────
        mode = res.get("mode", "equity")
        is_portfolio_mode = (mode == "portfolio")

        _vc_h = "Value" if "Value" in history.columns else \
                "PortfolioValue" if "PortfolioValue" in history.columns else None

        # 포트폴리오 모드: closed_trades 없음 → 연간 포트폴리오 가치 변화로 추정
        if is_portfolio_mode and _vc_h:
            st.info("📌 포트폴리오 통합 모드: 개별 거래 내역 대신 **연간 평가손익** 기준으로 세금을 추정합니다.")
            hist_tax = history.copy()
            if "Date" in hist_tax.columns:
                hist_tax = hist_tax.set_index("Date")
            hist_tax.index = pd.to_datetime(hist_tax.index)
            val_series = hist_tax[_vc_h]

            # 연도별 시작/종료 가치 → 연간 손익 계산
            from utils.tax_calculator import YearlyTaxResult, TaxSummary
            yearly_results = []
            years = sorted(val_series.index.year.unique())
            prev_val = usd_in  # 첫 해 기준: 투자 원금

            for y in years:
                y_series = val_series[val_series.index.year == y]
                if y_series.empty:
                    continue
                end_val  = y_series.iloc[-1]
                gain_usd = end_val - prev_val
                gain_krw = gain_usd * rate
                deduction = min(deduction_krw, gain_krw) if gain_krw > 0 else 0.0
                taxable   = max(gain_krw - deduction, 0.0)
                tax_krw   = taxable * tax_rate
                tax_usd   = tax_krw / rate if rate > 0 else 0.0
                yearly_results.append(YearlyTaxResult(
                    year=int(y),
                    realized_gain_usd=gain_usd,
                    realized_gain_krw=gain_krw,
                    deduction_krw=deduction,
                    taxable_krw=taxable,
                    tax_krw=tax_krw,
                    tax_usd=tax_usd,
                    after_tax_gain_krw=gain_krw - tax_krw,
                    trade_count=0,
                    tax_rate=tax_rate,
                ))
                prev_val = end_val

            total_gain_krw_t = sum(r.realized_gain_krw for r in yearly_results)
            total_tax_krw_t  = sum(r.tax_krw for r in yearly_results)
            tax_summary = TaxSummary(
                yearly=yearly_results,
                total_realized_gain_usd=sum(r.realized_gain_usd for r in yearly_results),
                total_realized_gain_krw=total_gain_krw_t,
                total_tax_krw=total_tax_krw_t,
                total_tax_usd=sum(r.tax_usd for r in yearly_results),
                total_after_tax_gain_krw=sum(r.after_tax_gain_krw for r in yearly_results),
                effective_tax_rate=total_tax_krw_t / total_gain_krw_t if total_gain_krw_t > 0 else 0.0,
            )
        else:
            tax_summary = calculate_tax(
                closed_trades=closed_trades,
                exchange_rate=rate,
                tax_rate=tax_rate,
                basic_deduction_krw=deduction_krw,
            )

        if not tax_summary.yearly:
            st.info("체결된 거래가 없어 양도세 계산 대상이 없습니다.")
            return

        st.divider()

        # ── 연도별 세금 테이블 ────────────────────────────────────────
        st.markdown("##### 📅 연도별 손익 & 세금")
        trade_col_label = "평가손익 기준" if is_portfolio_mode else "체결 건수"

        rows = []
        for y in tax_summary.yearly:
            row = {
                "연도":           y.year,
                "실현손익 (USD)": f"$ {y.realized_gain_usd:+,.0f}",
                "실현손익 (KRW)": f"₩ {y.realized_gain_krw:+,.0f}",
                "기본공제":       f"₩ {y.deduction_krw:,.0f}",
                "과세표준":       f"₩ {y.taxable_krw:,.0f}",
                "납부세액":       f"₩ {y.tax_krw:,.0f}",
                "세후손익 (KRW)": f"₩ {y.after_tax_gain_krw:+,.0f}",
            }
            if not is_portfolio_mode:
                row["체결 건수"] = y.trade_count
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── 전체 합산 요약 카드 ───────────────────────────────────────
        st.markdown("##### 💳 전체 기간 세금 요약")

        _vc = "Value" if "Value" in history.columns else "PortfolioValue" if "PortfolioValue" in history.columns else None
        final_usd = history[_vc].iloc[-1] if (not history.empty and _vc) else usd_in
        total_gain_krw = tax_summary.total_realized_gain_krw
        total_tax_krw  = tax_summary.total_tax_krw
        total_tax_usd  = tax_summary.total_tax_usd
        after_tax_krw  = tax_summary.total_after_tax_gain_krw

        # 세후 최종 자산 (원화)
        after_tax_final_krw = final_usd * rate - total_tax_krw

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "총 실현손익 (KRW)",
            f"₩ {total_gain_krw:,.0f}",
            help="매도 완료 거래 합계 (보유 중 미실현 손익 제외)"
        )
        c2.metric(
            "총 납부세액 (KRW)",
            f"₩ {total_tax_krw:,.0f}",
            delta=f"$ {total_tax_usd:,.0f} (USD)",
            delta_color="inverse",
        )
        c3.metric(
            "실효세율",
            f"{tax_summary.effective_tax_rate*100:.1f}%",
            help="실제 납부세액 / 총 실현손익"
        )
        c4.metric(
            "세후 실현손익 (KRW)",
            f"₩ {after_tax_krw:,.0f}",
        )

        st.divider()

        # ── 세후 최종 자산 ────────────────────────────────────────────
        st.markdown("##### 🏦 세후 최종 자산")
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "세전 최종 자산 (KRW)",
            f"₩ {final_usd * rate:,.0f}",
        )
        c2.metric(
            "납부세액 차감",
            f"- ₩ {total_tax_krw:,.0f}",
            delta_color="inverse",
        )
        c3.metric(
            "세후 최종 자산 (KRW)",
            f"₩ {after_tax_final_krw:,.0f}",
            delta=f"₩ {after_tax_final_krw - invest_cfg['krw']:+,.0f} (투자원금 대비)",
        )

        # ── 미실현손익 추정 ───────────────────────────────────────────
        st.divider()
        st.markdown("##### 📊 미실현손익 예상 세금 (참고용)")
        st.caption("현재 보유 중인 포지션을 지금 전량 매도한다고 가정한 추정치입니다.")

        unrealized = estimate_unrealized_tax(
            current_value_usd=final_usd,
            cost_basis_usd=usd_in,
            exchange_rate=rate,
            tax_rate=tax_rate,
            basic_deduction_krw=deduction_krw,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "미실현손익 (KRW)",
            f"₩ {unrealized['unrealized_gain_krw']:+,.0f}",
        )
        c2.metric(
            "예상 세액 (KRW)",
            f"₩ {unrealized['estimated_tax_krw']:,.0f}",
            delta=f"$ {unrealized['estimated_tax_usd']:,.0f}",
            delta_color="inverse",
        )
        c3.metric(
            "세후 예상 손익 (KRW)",
            f"₩ {unrealized['after_tax_gain_krw']:+,.0f}",
        )
        st.caption("⚠️ 본 계산은 시뮬레이션용 추정치이며 실제 세금과 다를 수 있습니다. 세무사 확인을 권장합니다.")

    # ──────────────────────────────────────────────────────────────────────
    # M-4: 대시보드 — 차트 + 연도별 테이블 + 거래 로그
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_dashboard_section(invest_cfg: dict, data_dict: dict):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        res     = st.session_state.sim_result
        history = res.get("history", pd.DataFrame())
        rate    = invest_cfg["exchange_rate"]
        usd_in  = invest_cfg["usd"]

        # Value 컬럼 통일 (PortfolioManager → PortfolioValue, Engine → Value)
        _val_col = "Value" if "Value" in history.columns else \
                   "PortfolioValue" if "PortfolioValue" in history.columns else None
        if history.empty or _val_col is None:
            return

        st.subheader("⑧ 성과 대시보드")

        # ── 히스토리 인덱스 통일 ─────────────────────────────────────
        hist = history.copy()
        if "Date" in hist.columns:
            hist = hist.set_index("Date")
        hist.index = pd.to_datetime(hist.index)

        # 원화 환산 Value 시리즈
        strat_krw = hist[_val_col] * rate

        # ── QQQ 벤치마크 준비 ─────────────────────────────────────────
        qqq_df = data_dict.get("QQQ", pd.DataFrame())
        bench_krw = None
        if not qqq_df.empty and "Close" in qqq_df.columns:
            qqq = qqq_df["Close"].reindex(hist.index).ffill()
            qqq_norm = qqq / qqq.iloc[0] * (usd_in * rate)   # 동일 원금으로 정규화
            bench_krw = qqq_norm

        # ── Tab 레이아웃 ──────────────────────────────────────────────
        tab_chart, tab_yearly, tab_trades = st.tabs([
            "📈 자산성장 곡선", "📅 연도별 성과", "📋 거래 내역"
        ])

        # ════════════════════════════════════════════════════════════════
        # Tab 1: 자산성장 곡선 + 드로우다운
        # ════════════════════════════════════════════════════════════════
        with tab_chart:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.04,
                subplot_titles=("자산 성장 곡선 (원화)", "드로우다운 (Drawdown)")
            )

            # 전략 자산 곡선
            fig.add_trace(go.Scatter(
                x=strat_krw.index, y=strat_krw,
                name="내 전략",
                line=dict(color="#00c853", width=2),
                hovertemplate="%{x|%Y-%m-%d}<br>₩ %{y:,.0f}<extra>내 전략</extra>"
            ), row=1, col=1)

            # QQQ 벤치마크
            if bench_krw is not None:
                fig.add_trace(go.Scatter(
                    x=bench_krw.index, y=bench_krw,
                    name="QQQ (벤치마크)",
                    line=dict(color="#2196f3", width=1.5, dash="dot"),
                    hovertemplate="%{x|%Y-%m-%d}<br>₩ %{y:,.0f}<extra>QQQ</extra>"
                ), row=1, col=1)

            # 포트폴리오 모드: 주식/채권 분리 곡선
            if res["mode"] == "portfolio":
                for key, label, color in [
                    ("Equity_Val", f"주식({res['eq_weight']:.0%})", "#ff9800"),
                    ("Bond_Val",   f"채권({res['bond_weight']:.0%})", "#9c27b0"),
                ]:
                    if key in hist.columns:
                        fig.add_trace(go.Scatter(
                            x=hist.index, y=hist[key] * rate,
                            name=label,
                            line=dict(width=1, dash="dash"),
                            opacity=0.6,
                            hovertemplate=f"%{{x|%Y-%m-%d}}<br>₩ %{{y:,.0f}}<extra>{label}</extra>"
                        ), row=1, col=1)

            # 투자원금 기준선
            fig.add_hline(
                y=usd_in * rate, row=1, col=1,
                line=dict(color="gray", dash="dot", width=1),
                annotation_text=f"투자원금 ₩{usd_in*rate:,.0f}",
                annotation_position="top left",
                annotation_font_size=11
            )

            # 드로우다운
            roll_max  = strat_krw.cummax()
            drawdown  = (strat_krw / roll_max - 1) * 100
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown,
                name="드로우다운",
                fill="tozeroy",
                fillcolor="rgba(255,23,68,0.15)",
                line=dict(color="#ff1744", width=1),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>DD</extra>"
            ), row=2, col=1)

            fig.update_layout(
                height=600,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
            )
            fig.update_yaxes(tickprefix="₩", tickformat=",.0f", row=1, col=1,
                             gridcolor="rgba(255,255,255,0.07)")
            fig.update_yaxes(ticksuffix="%", row=2, col=1,
                             gridcolor="rgba(255,255,255,0.07)")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

            # 연도별 수익률 바 차트
            st.markdown("##### 📊 연도별 수익률")
            years  = sorted(hist.index.year.unique())
            y_rets, q_rets = [], []
            for y in years:
                s_y = strat_krw[strat_krw.index.year == y]
                ret = (s_y.iloc[-1] / s_y.iloc[0] - 1) * 100 if len(s_y) >= 2 else 0
                y_rets.append(ret)
                if bench_krw is not None:
                    b_y = bench_krw[bench_krw.index.year == y]
                    q_rets.append((b_y.iloc[-1] / b_y.iloc[0] - 1) * 100 if len(b_y) >= 2 else 0)

            bar_fig = go.Figure()
            bar_fig.add_trace(go.Bar(
                x=[str(y) for y in years], y=y_rets,
                name="내 전략",
                marker_color=["#00c853" if r >= 0 else "#ff1744" for r in y_rets],
                text=[f"{r:+.1f}%" for r in y_rets],
                textposition="outside",
            ))
            if q_rets:
                bar_fig.add_trace(go.Scatter(
                    x=[str(y) for y in years], y=q_rets,
                    name="QQQ",
                    mode="lines+markers",
                    line=dict(color="#2196f3", width=2),
                ))
            bar_fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#fafafa"),
                yaxis=dict(ticksuffix="%", gridcolor="rgba(255,255,255,0.07)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        # ════════════════════════════════════════════════════════════════
        # Tab 2: 연도별 성과 테이블
        # ════════════════════════════════════════════════════════════════
        with tab_yearly:
            # M-3 세금 데이터 연동
            from utils.tax_calculator import calculate_tax
            tax_rate_pct  = st.session_state.get("sim_tax_rate_pct", 22.0)
            deduction_man = st.session_state.get("sim_deduction_man", 250.0)
            tax_summary   = calculate_tax(
                closed_trades=res.get("closed_trades", []),
                exchange_rate=rate,
                tax_rate=tax_rate_pct / 100.0,
                basic_deduction_krw=deduction_man * 10_000,
            )
            tax_by_year = {r.year: r for r in tax_summary.yearly}

            def calc_mdd(ser):
                s = ser.dropna()
                if s.empty: return 0.0
                return ((s / s.cummax()) - 1).min() * 100

            rows = []
            for y in years:
                s_y = strat_krw[strat_krw.index.year == y]
                s_ret = (s_y.iloc[-1] / s_y.iloc[0] - 1) * 100 if len(s_y) >= 2 else 0
                s_mdd = calc_mdd(s_y)

                q_ret_str = "-"
                if bench_krw is not None:
                    b_y = bench_krw[bench_krw.index.year == y]
                    q_ret = (b_y.iloc[-1] / b_y.iloc[0] - 1) * 100 if len(b_y) >= 2 else 0
                    q_ret_str = f"{q_ret:+.1f}%"

                tx = tax_by_year.get(y)
                row = {
                    "연도":              y,
                    "전략 수익률":       f"{s_ret:+.1f}%",
                    "QQQ 수익률":        q_ret_str,
                    "전략 MDD":          f"{s_mdd:.1f}%",
                    "실현손익 (KRW)":    f"₩ {tx.realized_gain_krw:+,.0f}" if tx else "-",
                    "납부세액 (KRW)":    f"₩ {tx.tax_krw:,.0f}"            if tx else "-",
                    "세후 손익 (KRW)":   f"₩ {tx.after_tax_gain_krw:+,.0f}" if tx else "-",
                    "체결 건수":         tx.trade_count if tx else 0,
                }
                rows.append(row)

            df_yearly = pd.DataFrame(rows).set_index("연도")

            def color_ret(val):
                try:
                    n = float(str(val).replace("%", "").replace("+", ""))
                    return f'color: {"#00c853" if n >= 0 else "#ff1744"}; font-weight:bold'
                except:
                    return ""

            def color_mdd(val):
                try:
                    n = abs(float(str(val).replace("%", "")))
                    alpha = min(n / 50.0, 1.0) * 0.4
                    return f"background-color: rgba(255,23,68,{alpha:.2f})"
                except:
                    return ""

            ret_cols = [c for c in df_yearly.columns if "수익률" in c or "손익" in c]
            mdd_cols = [c for c in df_yearly.columns if "MDD" in c]

            st.dataframe(
                df_yearly.style
                    .map(color_ret, subset=ret_cols)
                    .map(color_mdd, subset=mdd_cols),
                use_container_width=True,
                height=min(80 + len(rows) * 38, 600),
            )

        # ════════════════════════════════════════════════════════════════
        # Tab 3: 거래 내역
        # ════════════════════════════════════════════════════════════════
        with tab_trades:
            closed = res.get("closed_trades", [])
            _closed_empty = closed.empty if isinstance(closed, pd.DataFrame) else not closed
            if _closed_empty:
                st.info("체결된 거래 내역이 없습니다.")
            else:
                ct_df = pd.DataFrame(closed) if isinstance(closed, list) else closed.copy()

                # 원화 환산 컬럼 추가
                if "Net_Gain" in ct_df.columns:
                    ct_df["손익_KRW"] = (ct_df["Net_Gain"] * rate).map(lambda x: f"₩ {x:+,.0f}")
                if "Total_Cost" in ct_df.columns:
                    ct_df["매수금액_KRW"] = (ct_df["Total_Cost"] * rate).map(lambda x: f"₩ {x:,.0f}")
                if "Total_Revenue" in ct_df.columns:
                    ct_df["매도금액_KRW"] = (ct_df["Total_Revenue"] * rate).map(lambda x: f"₩ {x:,.0f}")

                # 표시 컬럼 선택 (존재하는 것만)
                display_cols = [c for c in [
                    "Entry_Date", "Exit_Date", "Asset", "Status",
                    "Buy_Price", "Sell_Price", "Shares",
                    "매수금액_KRW", "매도금액_KRW", "손익_KRW",
                    "Return", "MDD",
                ] if c in ct_df.columns]

                ct_display = ct_df[display_cols].copy()

                # 수익 행 색상
                def color_trade(row):
                    if "Return" not in row.index:
                        return [""] * len(row)
                    try:
                        r = float(row["Return"])
                        c = "rgba(0,200,83,0.08)" if r >= 0 else "rgba(255,23,68,0.08)"
                        return [f"background-color:{c}"] * len(row)
                    except:
                        return [""] * len(row)

                total_trades = len(ct_display)
                win_count = 0
                if "Return" in ct_df.columns:
                    win_count = (pd.to_numeric(ct_df["Return"], errors="coerce") >= 0).sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("총 체결 건수",  f"{total_trades}건")
                c2.metric("수익 거래",     f"{win_count}건 ({win_count/total_trades*100:.0f}%)" if total_trades else "-")
                c3.metric("손실 거래",     f"{total_trades - win_count}건" if total_trades else "-")

                st.dataframe(
                    ct_display.style.apply(color_trade, axis=1),
                    use_container_width=True,
                    height=min(80 + total_trades * 38, 600),
                )

            # 리밸런싱 이벤트 표시 (Portfolio 모드일 때만)
            if res.get("mode") == "portfolio":
                rebalance_events = res.get("rebalance_events", [])
                if rebalance_events:
                    st.divider()
                    st.subheader("🔄 리밸런싱 이벤트")
                    from ui.portfolio_view import PortfolioView
                    PortfolioView._render_rebalance_log(rebalance_events)

    # ──────────────────────────────────────────────────────────────────────
    # 보조: 전략 뱃지
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _render_strategy_badge(strategy_type: str, name: str, weight: float):
        icon  = "📈" if strategy_type == "equity" else "💰"
        label = "주식" if strategy_type == "equity" else "채권"
        st.info(f"{icon} **{label} 전략**: `{name}` — 비중 {weight:.0%}")
