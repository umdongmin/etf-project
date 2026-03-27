import math
import streamlit as st
import pandas as pd
from core.storage import AssetStorage, StrategyStorage
from core.constants import STAGE_RATIOS, STAGE_ICONS
from core.session_keys import SK


# ── 순수 함수: greedy-fill 정수 수량 계산 ──────────────────────────────
def _calc_target_qty_greedy(total_capital, weights, prices):
    """
    총자본 + 비중 + 현재가 → 정수 목표수량 (greedy-fill)

    Args:
        total_capital: 총 운용자본 (USD)
        weights: {'TQQQ': 0.5, 'QQQ': 0.35, 'TLT': 0.15}
        prices:  {'TQQQ': 55.2, 'QQQ': 420.1, 'TLT': 88.3}

    Returns:
        (result_dict, residual_cash)
        result_dict: {'TQQQ': 90, 'QQQ': 83, 'TLT': 16}
    """
    result = {}
    residual = 0.0

    # 1단계: floor 배분
    for ticker, weight in weights.items():
        px = prices.get(ticker, 0)
        if px <= 0:
            result[ticker] = 0
            continue
        alloc = total_capital * weight
        floored = math.floor(alloc / px)
        result[ticker] = floored
        residual += alloc - (floored * px)

    # 2단계: greedy-fill (고가 자산 우선)
    buyable = sorted(
        [(t, prices[t]) for t in weights if prices.get(t, 0) > 0],
        key=lambda x: x[1], reverse=True
    )
    changed = True
    while changed and residual > 0:
        changed = False
        for ticker, px in buyable:
            extra = math.floor(residual / px)
            if extra > 0:
                result[ticker] = result.get(ticker, 0) + extra
                residual -= extra * px
                changed = True

    return result, residual


def _fetch_prices(tickers):
    """티커 배치 현재가 조회 — DataService.get_prices() 위임"""
    from core.data import DataService
    if not tickers:
        return {}
    try:
        return DataService.get_prices(tickers)
    except Exception as e:
        st.warning(f"현재가 조회 실패: {e}")
        return {}


def _extract_weights(portfolio, signals=None):
    """
    포트폴리오 configs + 신호(선택) → {ticker: weight} 맵

    signals 있으면 stage 기반 실제 비중, 없으면 단순 균등 배분(base_asset만)
    """
    # STAGE_RATIOS imported from core.constants
    weights = {}
    configs = portfolio.get('configs', [])

    for cfg in configs:
        w = cfg.get('weight', 0)
        params = cfg.get('params', {})
        strat_type = cfg.get('type', 'equity')

        if signals and cfg['name'] in signals:
            sig = signals[cfg['name']]
            stage = sig.get('stage', 0)
            # base_asset: 신호에서 먼저, 없으면 params fallback (equity/bond 구분)
            default_base = params.get('base_asset', 'TLT' if strat_type == 'bond' else 'QQQ')
            default_lev  = params.get('leverage_asset', '')
            base = (sig.get('base_asset') or default_base).upper()
            lev  = (sig.get('lev_asset')  or default_lev).upper()
            br, lr = STAGE_RATIOS.get(stage, (1.0, 0.0))
            if br > 0 and base:
                weights[base] = weights.get(base, 0) + w * br
            if lr > 0 and lev:
                weights[lev]  = weights.get(lev, 0)  + w * lr
        else:
            # 신호 없으면 base_asset 100% 배분
            if strat_type == 'equity':
                base = params.get('base_asset', 'QQQ').upper()
                weights[base] = weights.get(base, 0) + w
            else:
                base = params.get('base_asset', 'TLT').upper()
                weights[base] = weights.get(base, 0) + w

    # 합계 정규화
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {t: v / total_w for t, v in weights.items()}
    return weights


# ══════════════════════════════════════════════════════════════════════════
class SyncView:
    """자산 동기화 UI — 총자본 기반 정수 수량 자동 계산 → holdings 저장"""

    def render(self, account_id=None):
        """
        account_id: 외부(asset_view 탭)에서 전달받으면 계좌 선택 UI 생략
        """
        if account_id is None:
            # 독립 메뉴로 호출될 때만 헤더 + 계좌 선택 표시
            st.header("🔀 자산 동기화")
            accounts = AssetStorage.list_accounts()
            if not accounts:
                st.warning("등록된 계좌가 없습니다.")
                return
            account_names = [a['name'] for a in accounts]
            default_idx = next((i for i, a in enumerate(accounts) if '엄동민' in a['name']), 0)
            selected_name = st.selectbox("계좌 선택", account_names, index=default_idx)
            account = next(a for a in accounts if a['name'] == selected_name)
            account_id = account['id']

        st.caption("총자본과 현재 신호를 기반으로 정수 목표수량을 계산하고 holdings에 저장합니다.")

        # ── 포트폴리오 선택 ──────────────────────────────────────────
        portfolio_names = StrategyStorage.list_portfolios()
        if not portfolio_names:
            st.warning("등록된 포트폴리오가 없습니다.")
            return

        default_p = next((i for i, n in enumerate(portfolio_names) if n.startswith('A')), 0)
        portfolio_name = st.selectbox("포트폴리오", portfolio_names, index=default_p)
        portfolio = StrategyStorage.load_portfolio(portfolio_name)
        if not portfolio:
            st.error("포트폴리오를 불러올 수 없습니다.")
            return

        st.divider()

        # ── 총자본 입력 ──────────────────────────────────────────────
        sync_cfg = AssetStorage.load_sync_config(account_id)
        default_capital = sync_cfg['total_capital'] if sync_cfg else 0.0

        col_cap, col_info = st.columns([2, 1])
        with col_cap:
            total_capital = st.number_input(
                "총 관리 자본 (USD)",
                value=float(default_capital),
                min_value=0.0,
                step=1000.0,
                format="%.2f",
                help="보유 주식 평가액 + 현금잔고의 합계를 입력하세요."
            )
        with col_info:
            if sync_cfg:
                st.metric("마지막 동기화", str(sync_cfg['last_synced_at'])[:10] if sync_cfg['last_synced_at'] else "-")

        st.divider()

        # ── 신호 조회 + 자동 계산 ────────────────────────────────────
        use_signal = st.toggle("현재 신호(Stage) 기반 비중 적용", value=True,
                               help="OFF 시 base_asset에 100% 배분")

        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            calc_clicked = st.button("📊 수량 계산", type="primary", use_container_width=True)

        if calc_clicked:
            if total_capital <= 0:
                st.error("총자본을 입력해주세요.")
                return

            with st.spinner("현재가 조회 중..."):
                signals = None
                if use_signal:
                    try:
                        from core.data import DataService
                        from core.signal_service import get_cached_signals, extract_signals
                        import datetime
                        # 사이드바 데이터 모드에 따라 증분/실시간 분기
                        if st.session_state.get('_data_mode_realtime', False):
                            data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data_realtime()
                        else:
                            data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data()
                        result = get_cached_signals(
                            portfolio_name, data_dict, fg_df, vxn_df, macro_df, news_df,
                            datetime.date(2026, 1, 1), cur_prices=None
                        )
                        if result:
                            signals = extract_signals(result, portfolio.get('configs', []))
                    except Exception as e:
                        st.warning(f"신호 조회 실패, 기본 비중 사용: {e}")

                weights = _extract_weights(portfolio, signals)
                tickers = list(weights.keys())
                prices = _fetch_prices(tickers)

            missing = [t for t in tickers if t not in prices]
            if missing:
                st.warning(f"가격 조회 실패: {', '.join(missing)}")

            if not prices:
                st.error("현재가를 조회할 수 없습니다.")
                return

            auto_qty, residual = _calc_target_qty_greedy(total_capital, weights, prices)

            st.session_state[SK.SYNC_AUTO_QTY]    = auto_qty
            st.session_state[SK.SYNC_PRICES]       = prices
            st.session_state[SK.SYNC_WEIGHTS]      = weights
            st.session_state[SK.SYNC_CAPITAL]      = total_capital
            st.session_state[SK.SYNC_ACCOUNT_ID]   = account_id
            st.session_state[SK.SYNC_RESIDUAL]     = residual
            st.session_state[SK.SYNC_SIGNALS_INFO] = signals

        # ── 계산 결과 + 수동 조정 폼 ────────────────────────────────
        if SK.SYNC_AUTO_QTY in st.session_state and st.session_state.get(SK.SYNC_ACCOUNT_ID) == account_id:
            auto_qty = st.session_state[SK.SYNC_AUTO_QTY]
            prices   = st.session_state[SK.SYNC_PRICES]
            weights  = st.session_state[SK.SYNC_WEIGHTS]
            capital  = st.session_state[SK.SYNC_CAPITAL]
            residual = st.session_state[SK.SYNC_RESIDUAL]
            signals_info = st.session_state.get(SK.SYNC_SIGNALS_INFO)

            # 신호 스테이지 표시
            if signals_info:
                stage_parts = []
                for name, sig in signals_info.items():
                    icon = STAGE_ICONS.get(sig['stage'], '⚪')
                    stage_parts.append(f"{icon} {name}: S{sig['stage']}")
                st.info("  |  ".join(stage_parts))

            st.subheader("목표 수량 조정")
            st.caption(f"기준 자본: ${capital:,.0f}  |  잔여 현금: ${residual:,.2f}")

            # 수동 조정 폼
            with st.form("sync_qty_form"):
                edited_qty = {}
                header_cols = st.columns([2, 2, 2, 2, 2])
                header_cols[0].write("**티커**")
                header_cols[1].write("**비중**")
                header_cols[2].write("**현재가**")
                header_cols[3].write("**자동 계산**")
                header_cols[4].write("**최종 수량**")

                for ticker, calc_q in sorted(auto_qty.items()):
                    px = prices.get(ticker, 0)
                    w  = weights.get(ticker, 0)
                    cols = st.columns([2, 2, 2, 2, 2])
                    cols[0].write(ticker)
                    cols[1].write(f"{w*100:.1f}%")
                    cols[2].write(f"${px:,.2f}" if px else "-")
                    cols[3].write(f"{calc_q}주")
                    edited_qty[ticker] = cols[4].number_input(
                        "", value=calc_q, min_value=0, step=1,
                        key=f"sync_qty_{ticker}", label_visibility="collapsed"
                    )

                notes = st.text_input("메모 (선택)", placeholder="동기화 사유 입력")
                save_clicked = st.form_submit_button("💾 holdings에 저장", type="primary")

            if save_clicked:
                # 자산 등록 확인 후 holdings upsert
                success_list = []
                fail_list = []
                assets = AssetStorage.list_assets()  # (id, name) 튜플 리스트
                asset_ids = {a[0].upper() for a in assets}
                for ticker, qty in edited_qty.items():
                    # asset 테이블에 없으면 자동 등록
                    if ticker.upper() not in asset_ids:
                        AssetStorage.save_asset(
                            ticker.upper(), ticker.upper(),
                            category='equity', currency='USD', ticker=ticker.upper()
                        )
                    ok = AssetStorage.upsert_holding_qty(account_id, ticker.upper(), qty)
                    if ok:
                        success_list.append(f"{ticker}: {qty}주")
                    else:
                        fail_list.append(ticker)

                # sync_config 저장
                AssetStorage.save_sync_config(account_id, capital, notes=notes or None)

                if success_list:
                    st.success(f"저장 완료: {', '.join(success_list)}")
                if fail_list:
                    st.error(f"저장 실패: {', '.join(fail_list)}")

                # 세션 초기화 후 재실행
                for k in [SK.SYNC_AUTO_QTY, SK.SYNC_PRICES, SK.SYNC_WEIGHTS, SK.SYNC_CAPITAL,
                          SK.SYNC_ACCOUNT_ID, SK.SYNC_RESIDUAL, SK.SYNC_SIGNALS_INFO]:
                    st.session_state.pop(k, None)
                st.rerun()

        # ── 현재 vs 계산 비교표 ──────────────────────────────────────
        st.divider()
        st.subheader("📋 현재 Holdings 현황")
        self._render_comparison(account_id, st.session_state.get(SK.SYNC_PRICES, {}))

    def _render_comparison(self, account_id, prices):
        """현재 holdings vs 계산 수량 비교표"""
        holdings = AssetStorage.list_holdings(account_id)
        if not holdings:
            st.info("보유 중인 자산이 없습니다. 위에서 수량을 계산하고 저장해주세요.")
            return

        auto_qty = st.session_state.get(SK.SYNC_AUTO_QTY, {})
        rows = []
        tickers_shown = set()

        for h in holdings:
            ticker = h['asset_id'].upper()
            tickers_shown.add(ticker)
            cur_qty = int(h['quantity'])
            calc_q  = auto_qty.get(ticker, None)
            px = prices.get(ticker, 0)
            delta = (calc_q - cur_qty) if calc_q is not None else None
            rows.append({
                '티커':       ticker,
                '현재 보유':  cur_qty,
                '계산 수량':  calc_q if calc_q is not None else '-',
                '델타':       f"{delta:+d}주" if delta is not None else '-',
                '델타 금액':  f"${delta * px:+,.0f}" if (delta is not None and px) else '-',
            })

        # 계산에만 있는 티커 추가
        for ticker, calc_q in auto_qty.items():
            if ticker not in tickers_shown:
                px = prices.get(ticker, 0)
                rows.append({
                    '티커':       ticker,
                    '현재 보유':  0,
                    '계산 수량':  calc_q,
                    '델타':       f"+{calc_q}주",
                    '델타 금액':  f"${calc_q * px:+,.0f}" if px else '-',
                })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        sync_cfg = AssetStorage.load_sync_config(account_id)
        if sync_cfg:
            st.caption(f"기준 자본: ${sync_cfg['total_capital']:,.0f}  |  마지막 동기화: {str(sync_cfg['last_synced_at'])[:16] if sync_cfg['last_synced_at'] else '-'}")
