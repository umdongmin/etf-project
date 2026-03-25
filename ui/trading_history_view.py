"""
ui/trading_history_view.py — KIS 실거래/모의투자 내역 모니터링 탭

데이터 소스:
  - 계좌 현황 / 주문 내역 / 거래 통계 : KIS API 직접 조회
  - 안전 체크 로그                    : Supabase order_log
    (안전장치 차단 이력은 KIS에 기록되지 않으므로 DB만 가능)

실거래 전환 시: TRADING_MODE=live + KIS_LIVE_* 키 설정만으로 동작
"""

import os
import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from collections import Counter, defaultdict

from core.storage import OrderStorage


# ── 상수 ─────────────────────────────────────────────────────────────────────
_MODE_LABEL = {'paper': '모의투자', 'live': '실거래'}
_MODE_COLOR = {'paper': '#2980b9', 'live': '#e74c3c'}
_SIDE_KO    = {'buy': ('매수', '#e74c3c'), 'sell': ('매도', '#2980b9')}
_STATUS_KO  = {'Y': ('체결', '#27ae60'), 'N': ('미체결', '#f39c12')}


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _badge(text: str, color: str) -> str:
    return (f'<span style="background:{color}22;color:{color};'
            f'padding:2px 8px;border-radius:4px;font-size:0.82em;'
            f'font-weight:600;">{text}</span>')


def _fmt_price(v) -> str:
    try:
        return f"${float(v):.2f}" if v else '-'
    except Exception:
        return '-'


def _fmt_amount(v) -> str:
    try:
        return f"${float(v):,.2f}" if v else '-'
    except Exception:
        return '-'


def _get_trading_mode() -> str | None:
    mode = os.environ.get('TRADING_MODE', 'disabled').lower().strip()
    return mode if mode in ('paper', 'live') else None


def _try_load_client(mode: str):
    """KISClient 로드. (client, error_msg) 튜플 반환."""
    try:
        from core.kis_client import KISClient
        return KISClient(mode=mode), None
    except Exception as e:
        return None, str(e)


# ── 메인 뷰 클래스 ────────────────────────────────────────────────────────────
class TradingHistoryView:

    @staticmethod
    def render():
        trading_mode = _get_trading_mode()

        # ── 헤더 ──────────────────────────────────────────────────────────
        col_title, col_badge = st.columns([5, 1])
        with col_title:
            st.subheader("KIS 거래 내역")
        with col_badge:
            color = _MODE_COLOR.get(trading_mode, '#95a5a6')
            label = _MODE_LABEL.get(trading_mode, '비활성')
            st.markdown(
                f'<div style="text-align:right;padding-top:8px;">'
                f'{_badge(label, color)}</div>',
                unsafe_allow_html=True,
            )

        # ── 미설정 안내 ───────────────────────────────────────────────────
        if not trading_mode:
            st.info(
                "KIS 거래가 비활성화되어 있습니다.\n\n"
                "`.env`에 `TRADING_MODE=paper` (모의투자) 또는 "
                "`TRADING_MODE=live` (실거래)를 설정하면 활성화됩니다.",
                icon="ℹ️",
            )
            return

        # ── KIS 클라이언트 초기화 ──────────────────────────────────────────
        client, err = _try_load_client(trading_mode)
        if client is None:
            st.error(f"KIS API 연결 실패: {err}\n\n`.env`의 앱 키 설정을 확인하세요.")
            return

        # ── 탭 구성 ───────────────────────────────────────────────────────
        tab_account, tab_orders, tab_chart, tab_safety = st.tabs([
            "💼 계좌 현황",
            "📋 주문 내역",
            "📈 거래 통계",
            "🛡️ 안전 체크 로그",
        ])

        with tab_account:
            TradingHistoryView._render_account(client)

        with tab_orders:
            TradingHistoryView._render_orders(client)

        with tab_chart:
            TradingHistoryView._render_chart(client)

        with tab_safety:
            TradingHistoryView._render_safety(trading_mode)

    # ── 탭 1: 계좌 현황 ───────────────────────────────────────────────────────
    @staticmethod
    def _render_account(client):
        st.markdown("#### 실시간 계좌 잔고")
        st.caption("KIS API를 통해 실시간으로 조회합니다.")

        if st.button("새로고침", key="kis_refresh_balance", use_container_width=False):
            st.cache_data.clear()

        with st.spinner("잔고 조회 중..."):
            try:
                balance = client.get_balance()
            except Exception as e:
                st.error(f"잔고 조회 오류: {e}")
                return

        if not balance:
            st.info("현재 보유 종목이 없습니다.")
        else:
            rows, total_eval, total_pnl = [], 0.0, 0.0
            for ticker, info in balance.items():
                side_html = _badge('보유', '#27ae60')
                pnl_color = '#e74c3c' if info['pnl'] >= 0 else '#2980b9'
                rows.append({
                    '종목':     ticker,
                    '수량':     info['qty'],
                    '평단가':   f"${info['avg_price']:.2f}",
                    '현재가':   f"${info['current_price']:.2f}",
                    '평가금액': f"${info['eval_amount']:,.2f}",
                    '평가손익': f"${info['pnl']:+,.2f}",
                    '수익률':   f"{info['pnl_rate']:+.2f}%",
                })
                total_eval += info['eval_amount']
                total_pnl  += info['pnl']

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("총 평가금액", f"${total_eval:,.2f}")
            pnl_sign = '+' if total_pnl >= 0 else ''
            c2.metric("총 평가손익", f"${pnl_sign}{total_pnl:,.2f}")
            c3.metric("보유 종목", f"{len(balance)}개")

        # 오늘 체결내역
        st.markdown("---")
        st.markdown("#### 오늘 체결내역")
        with st.spinner("체결내역 조회 중..."):
            try:
                today = datetime.date.today().strftime('%Y%m%d')
                history = client.get_order_history(start_date=today)
            except Exception as e:
                st.caption(f"체결내역 조회 오류: {e}")
                history = []

        if not history:
            st.caption("오늘 체결내역이 없습니다.")
        else:
            rows = []
            for h in history:
                side_ko, _ = _SIDE_KO.get(h.get('side', ''), (h.get('side', '-'), ''))
                status_ko, _ = _STATUS_KO.get(h.get('status', ''), (h.get('status', '-'), ''))
                rows.append({
                    '주문시간':   h.get('order_time', '-'),
                    '종목':       h.get('ticker', '-'),
                    '구분':       side_ko,
                    '주문수량':   h.get('order_qty', 0),
                    '체결수량':   h.get('filled_qty', 0),
                    '체결가':     _fmt_price(h.get('filled_price')),
                    '체결금액':   _fmt_amount(h.get('filled_amount')),
                    '상태':       status_ko,
                    '주문번호':   h.get('order_no', '-'),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 탭 2: 주문 내역 ───────────────────────────────────────────────────────
    @staticmethod
    def _render_orders(client):
        st.markdown("#### 주문 내역 (KIS API 직접 조회)")
        st.caption("KIS 서버에서 직접 체결내역을 가져옵니다.")

        col_days, col_side, col_ticker = st.columns(3)
        with col_days:
            days = st.selectbox(
                "조회 기간", [3, 5, 7, 14, 21],
                index=2, key="order_days",
                format_func=lambda x: f"최근 {x}일",
                help="기간이 길수록 API 호출 횟수가 증가합니다 (하루 1회 호출)"
            )
        with col_side:
            side_filter = st.selectbox(
                "매수/매도", ['전체', 'buy', 'sell'],
                key="order_side_filter",
                format_func=lambda x: {'buy': '매수', 'sell': '매도'}.get(x, x),
            )
        with col_ticker:
            ticker_filter = st.text_input("종목 필터", value="", key="order_ticker_filter",
                                          placeholder="예: TQQQ")

        with st.spinner(f"최근 {days}일 내역 조회 중... (날짜별 순차 조회)"):
            try:
                history = client.get_order_history_range(days=days)
            except Exception as e:
                st.error(f"주문 내역 조회 오류: {e}")
                return

        if not history:
            st.info("해당 기간의 체결내역이 없습니다.")
            return

        df = pd.DataFrame(history)

        # 필터 적용
        if side_filter != '전체':
            df = df[df['side'] == side_filter]
        if ticker_filter.strip():
            df = df[df['ticker'].str.upper() == ticker_filter.strip().upper()]

        if df.empty:
            st.info("해당 조건의 체결내역이 없습니다.")
            return

        rows = []
        for _, h in df.iterrows():
            side_ko, _ = _SIDE_KO.get(h.get('side', ''), (h.get('side', '-'), ''))
            status_ko, _ = _STATUS_KO.get(h.get('status', ''), (h.get('status', '-'), ''))
            rows.append({
                '날짜':       h.get('date', '-'),
                '주문시간':   h.get('order_time', '-'),
                '종목':       h.get('ticker', '-'),
                '구분':       side_ko,
                '주문수량':   h.get('order_qty', 0),
                '체결수량':   h.get('filled_qty', 0),
                '체결가':     _fmt_price(h.get('filled_price')),
                '체결금액':   _fmt_amount(h.get('filled_amount')),
                '상태':       status_ko,
                '주문번호':   h.get('order_no', '-'),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"총 {len(rows)}건")

    # ── 탭 3: 거래 통계 ───────────────────────────────────────────────────────
    @staticmethod
    def _render_chart(client):
        st.markdown("#### 거래 통계 (KIS API 직접 집계)")

        col_days2, _ = st.columns([1, 3])
        with col_days2:
            days = st.selectbox(
                "조회 기간", [7, 14, 21],
                index=0, key="chart_days",
                format_func=lambda x: f"최근 {x}일",
            )

        with st.spinner(f"최근 {days}일 집계 중..."):
            try:
                history = client.get_order_history_range(days=days)
            except Exception as e:
                st.error(f"조회 오류: {e}")
                return

        if not history:
            st.info("집계할 체결 데이터가 없습니다.")
            return

        # 요약 지표
        filled = [h for h in history if h.get('status') == 'Y']
        total_buy_amt  = sum(h.get('filled_amount', 0) for h in filled if h.get('side') == 'buy')
        total_sell_amt = sum(h.get('filled_amount', 0) for h in filled if h.get('side') == 'sell')
        tickers_traded = list({h['ticker'] for h in filled})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 주문", f"{len(history)}건")
        c2.metric("체결 완료", f"{len(filled)}건")
        c3.metric("총 매수금액",  f"${total_buy_amt:,.0f}")
        c4.metric("총 매도금액",  f"${total_sell_amt:,.0f}")

        # 날짜별 매수/매도 금액 집계
        daily_buy  = defaultdict(float)
        daily_sell = defaultdict(float)
        daily_cnt  = defaultdict(int)
        for h in filled:
            d = h.get('date', '')
            if not d:
                continue
            if h.get('side') == 'buy':
                daily_buy[d]  += h.get('filled_amount', 0)
            else:
                daily_sell[d] += h.get('filled_amount', 0)
            daily_cnt[d] += 1

        all_dates = sorted(set(list(daily_buy.keys()) + list(daily_sell.keys())))

        if all_dates:
            # 매수/매도 금액 바차트
            fig = go.Figure()
            fig.add_bar(
                x=all_dates,
                y=[daily_buy.get(d, 0) for d in all_dates],
                name='매수금액', marker_color='rgba(231,76,60,0.75)',
            )
            fig.add_bar(
                x=all_dates,
                y=[daily_sell.get(d, 0) for d in all_dates],
                name='매도금액', marker_color='rgba(41,128,185,0.75)',
            )
            fig.update_layout(
                title='날짜별 매수/매도 금액',
                barmode='group',
                xaxis_title='날짜', yaxis_title='금액 (USD)',
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation='h', y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 체결 건수 추이
            fig2 = go.Figure()
            fig2.add_scatter(
                x=all_dates,
                y=[daily_cnt.get(d, 0) for d in all_dates],
                mode='lines+markers', name='체결 건수',
                line=dict(color='#27ae60', width=2),
                marker=dict(size=7),
            )
            fig2.update_layout(
                title='날짜별 체결 건수',
                xaxis_title='날짜', yaxis_title='건수',
                height=220,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 종목별 거래 비중 (체결금액 기준)
        if tickers_traded:
            st.markdown("---")
            st.markdown("##### 종목별 체결금액 비중")
            ticker_amt = defaultdict(float)
            for h in filled:
                ticker_amt[h['ticker']] += h.get('filled_amount', 0)
            sorted_tickers = sorted(ticker_amt.items(), key=lambda x: x[1], reverse=True)
            fig3 = go.Figure(go.Pie(
                labels=[t for t, _ in sorted_tickers],
                values=[v for _, v in sorted_tickers],
                hole=0.4,
                marker_colors=['#e74c3c', '#2980b9', '#27ae60', '#f39c12', '#8e44ad'],
            ))
            fig3.update_layout(height=280, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig3, use_container_width=True)

    # ── 탭 4: 안전 체크 로그 (Supabase) ──────────────────────────────────────
    @staticmethod
    def _render_safety(trading_mode: str):
        st.markdown("#### TradingSafetyGuard 차단 이력")
        st.caption(
            "안전장치에 의해 차단된 주문 이력입니다.\n"
            "이 데이터는 KIS에 기록되지 않으므로 Supabase DB에서 조회합니다."
        )

        days = st.selectbox(
            "조회 기간", [7, 14, 30], index=1,
            key="safety_days",
            format_func=lambda x: f"최근 {x}일",
        )

        with st.spinner("로딩..."):
            try:
                logs = OrderStorage.list_order_logs(trading_mode=trading_mode, days=days)
            except Exception as e:
                st.warning(f"Supabase 조회 오류: {e}")
                return

        blocked = [r for r in logs if not r.get('safety_passed', True)]

        if not blocked:
            st.success(f"최근 {days}일 내 안전장치 차단 이력이 없습니다.")
        else:
            st.warning(f"최근 {days}일 내 {len(blocked)}건 차단됨")
            rows = []
            for r in blocked:
                created = r.get('created_at')
                if hasattr(created, 'strftime'):
                    created = created.strftime('%Y-%m-%d %H:%M')
                checks = r.get('safety_checks') or {}
                failed = [k for k, v in checks.items() if v is False] if isinstance(checks, dict) else []
                rows.append({
                    '일시':       created,
                    '종목':       r.get('ticker', '-'),
                    '구분':       _SIDE_KO.get(r.get('side', ''), (r.get('side', '-'), ''))[0],
                    '수량':       r.get('quantity', '-'),
                    '거부사유':   r.get('rejection_reason') or '-',
                    '실패항목':   ', '.join(failed) if failed else '-',
                    '포트폴리오': r.get('portfolio_name') or '-',
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # 차단 사유 분포
            reasons = [r['거부사유'] for r in rows if r['거부사유'] != '-']
            if reasons:
                cnt = Counter(reasons)
                fig = go.Figure(go.Bar(
                    x=list(cnt.values()),
                    y=list(cnt.keys()),
                    orientation='h',
                    marker_color='rgba(231,76,60,0.7)',
                ))
                fig.update_layout(
                    title='차단 사유 분포',
                    height=max(200, len(cnt) * 45 + 80),
                    margin=dict(l=0, r=0, t=40, b=0),
                    xaxis_title='건수',
                )
                st.plotly_chart(fig, use_container_width=True)
