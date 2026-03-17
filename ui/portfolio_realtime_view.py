"""
portfolio_realtime_view.py — 포트폴리오 실시간 신호 프리뷰 UI

Phase 2: 포트폴리오 매니저 내 "실시간 모니터링" 탭
- 전략별 현재 신호 (TQQQ stage, Bond asset/lev_stage)
- 포트폴리오 목표 비중 vs 드리프트 시각화
- Trend 모드 표시 (QQQ MA200 기준)
- 리밸런싱 필요 여부 알림
"""

import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go

from core.portfolio_manager import PortfolioManager


# ── 색상/스타일 헬퍼 ─────────────────────────────────────────────────────────
def _safe_float(val, default=0.0) -> float:
    """안전한 float 변환 (문자열/None/etc 모두 처리)"""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _signal_color(signal: str) -> tuple[str, str]:
    """(bg_color, text_color) 반환"""
    s = str(signal)
    if s.startswith("매도"):
        return "rgba(52,152,219,0.15)", "#2980b9"
    if s.startswith("매수"):
        return "rgba(231,76,60,0.15)", "#c0392b"
    if s.startswith("진행중"):
        return "rgba(39,174,96,0.15)", "#27ae60"
    return "rgba(127,140,141,0.12)", "#555"


def _stage_color(stage: int) -> str:
    return {0: "#95a5a6", 1: "#f39c12", 2: "#e67e22", 3: "#e74c3c"}.get(stage, "#555")


def _bond_asset_color(asset: str) -> tuple[str, str]:
    a = str(asset).upper()
    if "TMF" in a or "TMV" in a:
        return "rgba(155,89,182,0.15)", "#8e44ad"
    if "TLT" in a:
        return "rgba(52,152,219,0.12)", "#2980b9"
    if "TBF" in a:
        return "rgba(231,76,60,0.12)", "#c0392b"
    return "rgba(127,140,141,0.12)", "#555"   # BIL / 현금


def _drift_bar_fig(target_weights: dict, drifts: dict, upper_band: float) -> go.Figure:
    names  = list(target_weights.keys())
    targets = [target_weights[n] * 100 for n in names]
    drift_pcts = [drifts.get(n, 0) * 100 for n in names]
    current = [t + d for t, d in zip(targets, drift_pcts)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="목표 비중",
        x=names, y=targets,
        marker_color=["#3498db" if d <= upper_band * 100 else "#e74c3c" for d in [abs(d) for d in drift_pcts]],
        opacity=0.6,
        text=[f"{t:.1f}%" for t in targets],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="추정 현재 비중",
        x=names, y=current,
        marker_color=["#27ae60" if abs(d) <= upper_band * 100 else "#e74c3c" for d in drift_pcts],
        opacity=0.85,
        text=[f"{c:.1f}%" for c in current],
        textposition="outside",
    ))
    band_pct = upper_band * 100
    for n, t in zip(names, targets):
        fig.add_shape(type="line", x0=n, x1=n,
                      y0=t - band_pct, y1=t + band_pct,
                      line=dict(color="#e74c3c", width=2, dash="dot"))
    fig.update_layout(
        barmode="group", height=280, margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=-0.15),
        yaxis=dict(title="비중 (%)", range=[0, max(max(current), max(targets)) * 1.25]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── 메인 렌더 함수 ────────────────────────────────────────────────────────────
class PortfolioRealtimeView:

    @staticmethod
    def render(portfolio_manager: PortfolioManager,
               portfolio_name: str,
               data_dict: dict, fg_df, vxn_df, macro_df, news_df,
               start_date: datetime.date,
               rebalance_preset: str = 'hybrid'):
        """
        포트폴리오 실시간 신호 프리뷰 탭 렌더링.

        Parameters
        ----------
        portfolio_manager : 전략이 register_strategy()로 등록된 PortfolioManager 인스턴스
        portfolio_name : 포트폴리오 이름 (UI 표시 용도)
        data_dict / fg_df / vxn_df / macro_df / news_df : DataService.load_all_data() 결과
        start_date : 백테스트 시작일 (신호 계산 기준)
        rebalance_preset : 현재 선택된 리밸런싱 프리셋 키
        """

        st.markdown("### 🛰️ 실시간 포트폴리오 모니터링 (Beta)")

        # ── 헤더 정보 바 ────────────────────────────────────────────────────
        now   = datetime.datetime.now()
        is_dst = datetime.datetime(now.year, 3, 8) <= now < datetime.datetime(now.year, 11, 1)
        dst_label = "☀️ 서머타임 (22:30~05:00 KST)" if is_dst else "❄️ 평시 (23:30~06:00 KST)"
        dst_color = "#e67e22" if is_dst else "#3498db"

        st.markdown(f"""
        <div style="background:#f8f9fa; border-left:4px solid {dst_color}; padding:10px 14px;
                    border-radius:8px; font-size:13px; color:#444; margin-bottom:12px;">
            🇺🇸 미국 정규장: <b style="color:{dst_color}">{dst_label}</b>
            &nbsp;|&nbsp; 분석 기준일: <b style="color:#d35400">{now.strftime('%Y-%m-%d')}</b>
            &nbsp;|&nbsp; 포트폴리오: <b>{portfolio_name}</b>
        </div>
        """, unsafe_allow_html=True)

        # ── 자동 신호 계산 (페이지 로드 시) ────────────────────────────────
        result = st.session_state.get("portfolio_realtime_result")

        # 신호가 없거나 새로운 페이지 진입이면 자동 계산
        if result is None:
            with st.spinner("포트폴리오 신호 계산 중... (현재가 조회 포함)"):
                try:
                    from core.data import DataService
                    from ui.portfolio_view import PortfolioView

                    # 현재가 조회 → compute_portfolio_signals에서 inject_virtual_close 처리
                    _all_tickers = list(data_dict.keys()) + ['^VIX', '^VXN']
                    _cur_prices  = DataService.fetch_current_prices(_all_tickers)

                    result = PortfolioView.compute_portfolio_signals(
                        portfolio_name, data_dict, fg_df, vxn_df, macro_df, news_df,
                        start_date, cur_prices=_cur_prices,
                    )
                    if result:
                        result['_calc_time'] = datetime.datetime.now().strftime('%H:%M:%S')
                        st.session_state["portfolio_realtime_result"] = result
                    else:
                        raise ValueError("신호 계산 결과 없음")
                except Exception as e:
                    st.error(f"신호 계산 오류: {e}")
                    import traceback; traceback.print_exc()
                    return

        # ── 새로고침 버튼 ──────────────────────────────────────────────────
        col_btn, col_status = st.columns([2, 5])
        with col_btn:
            if st.button("🔄 새로고침", use_container_width=True):
                st.session_state.pop("portfolio_realtime_result", None)
                st.rerun()

        with col_status:
            if result:
                st.caption(f"마지막 계산: {result.get('_calc_time', '?')}  |  기준일: {result.get('date', '?')}")

        if not result:
            st.warning("신호 계산 실패. 새로고침 버튼을 누르세요.")
            return

        # ── 섹션 1: 포트폴리오 건강도 요약 (항상 표시) ────────────────────
        st.markdown("---")
        st.markdown("#### 📊 포트폴리오 상태 요약")

        with st.container():
            rb = result.get("rebalance", {})
            trend = result.get("trend_mode")
            strategies = result.get("strategies", {})

            col1, col2, col3, col4 = st.columns(4)

            # 종합 상태
            all_signals = [str(sig.get('signal', '중립')) for sig in strategies.values() if sig]
            buy_count = sum(1 for s in all_signals if s.startswith('매수'))
            sell_count = sum(1 for s in all_signals if s.startswith('매도'))
            hold_count = sum(1 for s in all_signals if '진행중' in s)

            if sell_count > buy_count:
                overall_state = "⚠️ 매도 신호"
            elif buy_count > sell_count:
                overall_state = "🟢 매수 신호"
            else:
                overall_state = "⏸️ 진행중/중립"

            col1.metric("종합 상태", overall_state)
            col2.metric("리밸런싱", "필요 ⚠️" if rb.get('needed') else "정상 ✅")
            col3.metric("Drift", f"{max(abs(d) for d in rb.get('drifts', {}).values()) * 100:.1f}%")
            col4.metric("Trend", f"{'📈 상승' if trend == 'uptrend' else '📉 하락'}")

        # ── 수익률 요약 (신규) ────────────────────────────────────────────────
        returns = result.get("returns", {})
        portfolio_ret = returns.get("portfolio_return", 0.0)
        strategy_returns = returns.get("strategy_returns", {})

        with st.container():
            st.markdown("#### 📈 수익률 (YTD)")

            # 포트폴리오 + 개별 전략 수익률을 행으로 표시
            ret_cols = st.columns(len(strategies) + 1)

            # 포트폴리오 수익률 (첫 번째 열)
            with ret_cols[0]:
                ret_color = "#27ae60" if portfolio_ret >= 0 else "#e74c3c"
                ret_sign = "+" if portfolio_ret >= 0 else ""
                st.markdown(f"""
                <div style="text-align:center; padding:12px; background:rgba(52,152,219,0.1);
                            border-radius:8px; border:1px solid #3498db;">
                    <div style="font-size:12px; color:#666; margin-bottom:4px;"><b>포트폴리오</b></div>
                    <div style="font-size:22px; font-weight:bold; color:{ret_color};">{ret_sign}{portfolio_ret:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # 개별 전략 수익률
            for idx, (strat_name, strat_ret) in enumerate(strategy_returns.items(), 1):
                with ret_cols[idx]:
                    ret_color = "#27ae60" if strat_ret >= 0 else "#e74c3c"
                    ret_sign = "+" if strat_ret >= 0 else ""
                    bg_color = "rgba(39,174,96,0.1)" if strat_ret >= 0 else "rgba(231,76,60,0.1)"
                    st.markdown(f"""
                    <div style="text-align:center; padding:12px; background:{bg_color};
                                border-radius:8px; border:1px solid {ret_color}; opacity:0.9;">
                        <div style="font-size:12px; color:#666; margin-bottom:4px;"><b>{strat_name}</b></div>
                        <div style="font-size:22px; font-weight:bold; color:{ret_color};">{ret_sign}{strat_ret:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── 섹션 2: 전략별 신호 카드 (항상 표시) ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 전략별 현재 신호")

        strategies = result.get("strategies", {})
        n_strats   = len(strategies)
        if n_strats == 0:
            st.warning("등록된 전략이 없습니다.")
            return

        cols = st.columns(n_strats + 1)  # +1: 포트폴리오 종합 카드

        for idx, (name, sig) in enumerate(strategies.items()):
            with cols[idx]:
                PortfolioRealtimeView._render_strategy_card(name, sig, portfolio_manager, result)

        # 포트폴리오 종합 카드
        with cols[n_strats]:
            PortfolioRealtimeView._render_portfolio_summary_card(result, portfolio_manager)

        # ── 섹션 3: 리밸런싱 상태 (항상 표시) ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 리밸런싱 상태")

        rb = result.get("rebalance", {})
        PortfolioRealtimeView._render_rebalance_section(rb, rebalance_preset, portfolio_manager)

        # ── 섹션 4: Trend 모드 (항상 표시) ───────────────────────────────────────────────
        trend_mode = result.get("trend_mode")
        if trend_mode is not None:
            st.markdown("---")
            st.markdown("#### QQQ Trend Mode")
            PortfolioRealtimeView._render_trend_section(trend_mode, portfolio_manager)

    # ── 전략 신호 카드 ────────────────────────────────────────────────────────
    @staticmethod
    def _render_strategy_card(name: str, sig: dict | None, pm: PortfolioManager, result: dict = None):
        target_w = pm.strategies.get(name, {}).get('weight', 0)
        stype    = pm.strategies.get(name, {}).get('strat_type', 'equity')

        # 수익률 추출
        strategy_ret = 0.0
        if result and 'returns' in result:
            returns = result.get('returns', {})
            strategy_returns = returns.get('strategy_returns', {})
            strategy_ret = strategy_returns.get(name, 0.0)

        if sig is None:
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:10px; padding:14px; min-height:160px;">
                <div style="font-weight:bold; font-size:15px; margin-bottom:8px;">{name}</div>
                <div style="color:#999; font-size:13px;">신호 없음</div>
            </div>
            """, unsafe_allow_html=True)
            return

        if stype == 'equity':
            signal = str(sig.get('signal', '중립'))
            stage  = int(sig.get('stage', 0))
            price  = sig.get('price', 0)
            rsi    = sig.get('rsi', 0)
            vxn    = sig.get('vxn', 0)
            summary = str(sig.get('summary', ''))

            bg, tc  = _signal_color(signal)
            sc      = _stage_color(stage)
            lev_labels = {0: "현금/QQQ", 1: "TQQQ 30%", 2: "TQQQ 70%", 3: "TQQQ 100%"}
            lev_label  = lev_labels.get(stage, f"S{stage}")

            # 수익률 색상
            ret_color = "#27ae60" if strategy_ret >= 0 else "#e74c3c"
            ret_sign = "+" if strategy_ret >= 0 else ""

            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {tc}; border-radius:10px;
                        padding:14px; min-height:200px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-weight:bold; font-size:15px; color:#333;">{name}</span>
                    <span style="font-size:13px; background:#f0f0f0; padding:3px 10px;
                                 border-radius:12px; color:{ret_color}; font-weight:bold;">{ret_sign}{strategy_ret:.2f}%</span>
                </div>
                <div style="font-size:18px; font-weight:900; color:{tc}; margin-bottom:10px;">
                    {signal}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:12px; color:#555;">
                    <div><b>Stage:</b>
                        <span style="color:{sc}; font-weight:bold;">S{stage} ({lev_label})</span>
                    </div>
                    <div><b>가격:</b> ${_safe_float(price):,.2f}</div>
                    <div><b>RSI:</b> {_safe_float(rsi):.1f}</div>
                    <div><b>VXN:</b> {_safe_float(vxn):.1f}</div>
                </div>
                <div style="margin-top:8px; font-size:11px; color:#777; border-top:1px solid rgba(0,0,0,0.08);
                            padding-top:6px; overflow:hidden; max-height:48px;">{summary[:80]}</div>
            </div>
            """, unsafe_allow_html=True)

        else:  # bond
            asset      = str(sig.get('signal', 'BIL'))
            lev_stage  = int(sig.get('lev_stage', 0))
            lev_asset  = sig.get('lev_asset') or '없음'
            rsi        = sig.get('rsi', 0)
            yc         = sig.get('yc', 0)
            f_state    = sig.get('f_state', 0)

            bg, tc = _bond_asset_color(asset)
            lev_fracs = {0: "0%", 1: "30%", 2: "70%", 3: "100%"}
            lev_frac  = lev_fracs.get(lev_stage, f"S{lev_stage}")

            # 수익률 색상
            ret_color = "#27ae60" if strategy_ret >= 0 else "#e74c3c"
            ret_sign = "+" if strategy_ret >= 0 else ""

            st.markdown(f"""
            <div style="background:{bg}; border:2px solid {tc}; border-radius:10px;
                        padding:14px; min-height:200px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <span style="font-weight:bold; font-size:15px; color:#333;">{name}</span>
                    <span style="font-size:13px; background:#f0f0f0; padding:3px 10px;
                                 border-radius:12px; color:{ret_color}; font-weight:bold;">{ret_sign}{strategy_ret:.2f}%</span>
                </div>
                <div style="font-size:18px; font-weight:900; color:{tc}; margin-bottom:10px;">
                    {asset}
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:12px; color:#555;">
                    <div><b>레버리지:</b>
                        <span style="color:#8e44ad; font-weight:bold;">{lev_asset} ({lev_frac})</span>
                    </div>
                    <div><b>Lev Stage:</b> S{lev_stage}</div>
                    <div><b>TLT RSI:</b> {_safe_float(rsi):.1f}</div>
                    <div><b>10y-2y:</b> {_safe_float(yc):.2f}%</div>
                    <div><b>F-State:</b> {_safe_float(f_state):.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 포트폴리오 종합 카드 ──────────────────────────────────────────────────
    @staticmethod
    def _render_portfolio_summary_card(result: dict, pm: PortfolioManager):
        rb      = result.get("rebalance", {})
        needed  = rb.get("needed", False)
        reason  = rb.get("reason", "none")
        trend   = result.get("trend_mode")
        date    = result.get("date", "")

        rb_bg  = "rgba(231,76,60,0.15)"  if needed else "rgba(39,174,96,0.12)"
        rb_tc  = "#c0392b"               if needed else "#27ae60"
        rb_icon = "⚠️ 리밸런싱 필요"     if needed else "✅ 정상 범위"
        rb_reason_map = {
            "drift":       "비중 이탈",
            "calendar":    "정기 주기 도달",
            "within_band": "밴드 이내",
            "none":        "리밸런싱 없음 설정",
        }
        rb_reason_label = rb_reason_map.get(reason, reason)

        trend_html = ""
        if trend == "downtrend":
            trend_html = '<div style="margin-top:6px; font-size:12px; color:#c0392b; font-weight:bold;">📉 QQQ 하락추세 (MA200 하단) — 방어 모드</div>'
        elif trend == "uptrend":
            trend_html = '<div style="margin-top:6px; font-size:12px; color:#27ae60; font-weight:bold;">📈 QQQ 상승추세 (MA200 상단) — 공격 모드</div>'

        parts = [f"{n} {int(v.get('weight',0)*100)}%"
                 for n, v in pm.strategies.items()]
        label = " / ".join(parts)

        st.markdown(f"""
        <div style="background:{rb_bg}; border:2px solid {rb_tc}; border-radius:10px;
                    padding:14px; min-height:200px;">
            <div style="font-weight:bold; font-size:15px; color:#333; margin-bottom:10px;">
                포트폴리오 종합
            </div>
            <div style="font-size:18px; font-weight:900; color:{rb_tc}; margin-bottom:8px;">
                {rb_icon}
            </div>
            <div style="font-size:12px; color:#555;">
                <div><b>사유:</b> {rb_reason_label}</div>
                <div><b>구성:</b> {label}</div>
                <div><b>기준일:</b> {date}</div>
            </div>
            {trend_html}
        </div>
        """, unsafe_allow_html=True)

    # ── 리밸런싱 섹션 ────────────────────────────────────────────────────────
    @staticmethod
    def _render_rebalance_section(rb: dict, rebalance_preset: str, pm: PortfolioManager):
        needed  = rb.get("needed", False)
        reason  = rb.get("reason", "none")
        drifts  = rb.get("drifts", {})
        targets = rb.get("target_weights", {})
        est     = rb.get("current_weights_estimated", {})

        # 리밸런싱 방식에서 band 값 추출
        preset_cfg  = pm.custom_preset_cfg
        if preset_cfg is None:
            preset_cfg = pm.REBALANCE_PRESETS.get(rebalance_preset, {'method': 'none'})
        upper_band  = preset_cfg.get('upper_band', 0.07)

        col_info, col_chart = st.columns([1, 2])

        with col_info:
            if needed:
                st.error(f"⚠️ **리밸런싱 필요** — {reason}")
            else:
                st.success("✅ **리밸런싱 불필요** — 비중 정상 범위 내")

            st.markdown(f"**밴드 기준:** ±{upper_band*100:.0f}%")
            st.markdown("**전략별 드리프트 (목표 대비)**")

            rows = []
            for n in targets:
                tw  = targets[n]
                d   = drifts.get(n, 0)
                ew  = tw + d
                flag = " ⚠️" if abs(d) > upper_band else ""
                rows.append({
                    "전략": n,
                    "목표 (%)": f"{tw*100:.1f}%",
                    "추정 (%)": f"{ew*100:.1f}%",
                    "드리프트": f"{d*100:+.1f}%{flag}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("전략"),
                         use_container_width=True, hide_index=False)

        with col_chart:
            if targets:
                fig = _drift_bar_fig(targets, drifts, upper_band)
                st.plotly_chart(fig, use_container_width=True)

        # 실행 가이드
        if needed:
            st.markdown("##### 리밸런싱 실행 가이드")
            guide_rows = []
            total_cap = pm.total_capital
            for n in targets:
                tw  = targets[n]
                ew  = tw + drifts.get(n, 0)
                diff_pct = tw - ew
                diff_usd = diff_pct * total_cap
                action = "매수" if diff_usd > 0 else "매도"
                guide_rows.append({
                    "전략": n,
                    "목표 비중": f"{tw*100:.1f}%",
                    "현재 비중(추정)": f"{ew*100:.1f}%",
                    "조정": f"{action} ${abs(diff_usd):,.0f}",
                })
            st.dataframe(pd.DataFrame(guide_rows).set_index("전략"),
                         use_container_width=True, hide_index=False)

    # ── Trend 섹션 ───────────────────────────────────────────────────────────
    @staticmethod
    def _render_trend_section(trend_mode: str, pm: PortfolioManager):
        if trend_mode == "downtrend":
            bg, tc = "rgba(231,76,60,0.08)", "#c0392b"
            icon   = "📉"
            title  = "하락추세 — 방어 모드 (QQQ < MA200)"
            desc   = "QQQ가 200일 이동평균선 아래에 있습니다. Trend-Filtered 전략이 적용 중인 경우 방어 비중으로 자동 전환됩니다."
        else:
            bg, tc = "rgba(39,174,96,0.08)", "#27ae60"
            icon   = "📈"
            title  = "상승추세 — 공격 모드 (QQQ > MA200)"
            desc   = "QQQ가 200일 이동평균선 위에 있습니다. 설정된 공격 비중으로 운용 중입니다."

        defense_info = ""
        if pm.trend_defense_weights and trend_mode == "downtrend":
            parts = [f"{n}: {int(v*100)}%" for n, v in pm.trend_defense_weights.items()]
            defense_info = f"<br><b>방어 비중 전환:</b> {' / '.join(parts)}"

        st.markdown(f"""
        <div style="background:{bg}; border-left:4px solid {tc}; border-radius:8px;
                    padding:12px 16px; font-size:13px; color:#444;">
            <div style="font-size:15px; font-weight:bold; color:{tc}; margin-bottom:6px;">
                {icon} QQQ MA200 추세 — {title}
            </div>
            {desc}{defense_info}
        </div>
        """, unsafe_allow_html=True)
