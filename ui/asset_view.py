import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date
from core.storage import AssetStorage
from core.constants import STAGE_RATIOS, STAGE_LABELS, STAGE_ICONS
from core.session_keys import SK
from ui.sync_view import SyncView


class AssetView:
    """포트폴리오 연동 자산 관리 UI"""

    def render(self):
        accounts = AssetStorage.list_accounts()
        if not accounts:
            st.warning("등록된 계좌가 없습니다.")
            self._render_account_setup()
            return

        account_names = [a['name'] for a in accounts]
        default_acc_idx = next((i for i, a in enumerate(accounts) if '엄동민' in a['name']), 0)

        col1, col2 = st.columns([4, 1])
        with col1:
            selected_account_name = st.selectbox("계좌 선택", account_names, index=default_acc_idx, label_visibility="collapsed")

        selected_account = next((a for a in accounts if a['name'] == selected_account_name), None)
        account_id = selected_account['id']

        tab1, tab2, tab3 = st.tabs(["📡 신호 & 주문지시", "💼 보유 현황", "⚙️ 설정"])

        with tab1:
            self._render_signal_orders(account_id)
        with tab2:
            self._render_holdings_and_deposits(account_id)
        with tab3:
            self._render_settings(account_id)

    # ══════════════════════════════════════════════════════
    # Tab 1: 신호 & 주문지시 (핵심)
    # ══════════════════════════════════════════════════════
    def _render_signal_orders(self, account_id):
        from core.storage import StrategyStorage

        # ── 포트폴리오 선택 + 신호 갱신 ─────────────────────────────
        portfolio_names = StrategyStorage.list_portfolios()
        if not portfolio_names:
            st.warning("등록된 포트폴리오가 없습니다.")
            return

        # DB 기본 포트폴리오 우선, 없으면 첫 번째 항목
        _db_default_pf = StrategyStorage.get_default_portfolio()
        default_idx = (
            portfolio_names.index(_db_default_pf)
            if _db_default_pf and _db_default_pf in portfolio_names
            else 0
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            portfolio_name = st.selectbox("포트폴리오", portfolio_names, index=default_idx, key="signal_portfolio")
        with col2:
            st.write("")
            refresh = st.button("🔄 신호 갱신", key="btn_refresh_signal", use_container_width=True)

        portfolio = StrategyStorage.load_portfolio(portfolio_name)
        if not portfolio:
            st.error("포트폴리오를 불러올 수 없습니다.")
            return
        configs = portfolio.get('configs', [])

        # ── 신호 캐시 관리 ───────────────────────────────────────────
        cached_portfolio = st.session_state.get(SK.CURRENT_SIGNALS_PORTFOLIO)
        cached_signals = st.session_state.get(SK.CURRENT_SIGNALS, {})
        portfolio_changed = cached_portfolio != portfolio_name

        if refresh:
            for mode in ('closing', 'realtime'):
                st.session_state.pop(SK.sig_cache(portfolio_name, mode), None)
                st.session_state.pop(SK.sig_cache_ts(portfolio_name, mode), None)
            with st.spinner("신호 계산 중..."):
                signals = self._fetch_current_signals(portfolio_name, configs)
            st.session_state[SK.CURRENT_SIGNALS] = signals
            st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
        elif portfolio_changed or not cached_signals:
            rt_result = st.session_state.get(SK.PORTFOLIO_REALTIME_RESULT)
            rt_portfolio = st.session_state.get(SK.PORTFOLIO_SELECTED_NAME)
            if rt_result and rt_portfolio == portfolio_name:
                signals = self._extract_signals(rt_result, configs)
                st.session_state[SK.CURRENT_SIGNALS] = signals
                st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
            else:
                signals = {}
                st.session_state[SK.CURRENT_SIGNALS] = {}
                st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
        else:
            signals = cached_signals

        # ── 현재 신호 카드 ────────────────────────────────────────────
        if signals:
            sig_cols = st.columns(len(signals))
            for i, (name, sig) in enumerate(signals.items()):
                with sig_cols[i]:
                    s = sig['stage']
                    st.metric(
                        label=name,
                        value=f"{STAGE_ICONS.get(s, '⚪')} {STAGE_LABELS.get(s, f'S{s}')}",
                        delta=f"{sig['base_asset']} → {sig.get('lev_asset', '')}" if sig.get('lev_asset') else sig['base_asset']
                    )
        else:
            st.info("📡 신호 없음. **[🔄 신호 갱신]** 버튼을 눌러주세요.")

        # ── 신호 변경 감지 배너 ───────────────────────────────────────
        last_snapshot = AssetStorage.get_latest_signal_snapshot(account_id, portfolio_name)
        changed_strategies = self._detect_signal_changes(signals, last_snapshot)

        if changed_strategies:
            st.warning(f"⚠️ **신호 변경 감지** — {len(changed_strategies)}개 전략에서 변화가 있습니다.")
            chg_rows = [
                {
                    '전략': n,
                    '이전': f"S{c['prev_stage']} {c['prev_asset']}",
                    '현재': f"S{c['new_stage']} {c['new_asset']}",
                    '액션': c['action']
                }
                for n, c in changed_strategies.items()
            ]
            st.dataframe(pd.DataFrame(chg_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── 보유 현황 + 주문지시 ─────────────────────────────────────
        holdings = AssetStorage.list_holdings(account_id)
        if not holdings:
            st.info("보유 자산이 없습니다. **⚙️ 설정** 탭 → **수량 동기화**에서 초기 수량을 입력하세요.")
            return

        # 현재가 + 평가액 계산
        holdings_data = []
        total_value = 0
        for h in holdings:
            price = self._get_current_price(h['asset_id'], h['asset_name'])
            val = h['quantity'] * price if price else 0
            total_value += val
            holdings_data.append({**h, 'current_price': price, 'current_value': val})

        target_alloc = self._compute_target_allocation(configs, signals)
        rebal_orders = self._calculate_rebal_orders(holdings_data, target_alloc, total_value)
        target_map = {item['ticker'].upper(): item['weight'] for item in target_alloc}

        # ── 통합 현황 + 주문지시 테이블 ─────────────────────────────
        st.subheader("💼 보유 현황 & 주문지시")
        rows = []
        holdings_keys = {h['asset_id'].upper() for h in holdings_data}

        for h in holdings_data:
            key = h['asset_id'].upper()
            target_w = target_map.get(key, 0.0) * 100
            current_w = (h['current_value'] / total_value * 100) if total_value > 0 else 0

            order_info = next((o for o in rebal_orders if o['asset_id'] == key), None)
            if order_info:
                raw_qty = abs(order_info['suggested_qty'])
                qty = int(raw_qty) if raw_qty >= 1 else round(raw_qty, 4)
                amt = abs(order_info['order_amount'])
                order_str = f"🟢 매수 {qty}주 (${amt:,.0f})" if order_info['action'] == 'buy' else f"🔴 매도 {qty}주 (${amt:,.0f})"
            else:
                order_str = "✅ 유지"

            rows.append({
                '자산': h['asset_name'],
                '보유수량': f"{int(h['quantity']):,}",
                '현재가': f"${h['current_price']:,.2f}" if h['current_price'] else "—",
                '평가액': f"${h['current_value']:,.0f}",
                '현재비중': f"{current_w:.1f}%",
                '목표비중': f"{target_w:.1f}%",
                '📌 주문지시': order_str,
            })

        # 미보유 신규 매수 대상
        for o in rebal_orders:
            if o['asset_id'] not in holdings_keys:
                raw_qty = abs(o['suggested_qty'])
                qty = int(raw_qty) if raw_qty >= 1 else round(raw_qty, 4)
                amt = abs(o['order_amount'])
                rows.append({
                    '자산': o['asset_id'],
                    '보유수량': "0",
                    '현재가': f"${o['current_price']:,.2f}" if o['current_price'] else "—",
                    '평가액': "$0",
                    '현재비중': "0.0%",
                    '목표비중': f"{o['target_weight']:.1f}%",
                    '📌 주문지시': f"🟢 신규매수 {qty}주 (${amt:,.0f})",
                })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 순 입금액 (deposits 기준)
        deposits = AssetStorage.list_deposits(account_id)
        net_cash = sum(d['amount'] if d['type'] == 'deposit' else -d['amount'] for d in deposits if d['currency'] == 'USD')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 평가액", f"${total_value:,.0f}")
        with col2:
            st.metric("순 입금액 (USD)", f"${net_cash:,.0f}")
        with col3:
            if net_cash and net_cash != 0:
                pnl = total_value - net_cash
                pnl_pct = pnl / net_cash * 100
                st.metric("수익률", f"{pnl_pct:+.2f}%", delta=f"${pnl:+,.0f}")

        st.divider()

        # ── 거래 저장 영역 ────────────────────────────────────────────
        if not rebal_orders:
            st.success("✅ 모든 자산이 목표 비중 범위 내에 있습니다. (편차 < 5%)")
            return

        if changed_strategies:
            st.subheader("🔄 거래 저장 — ⚠️ 포지션 변경 반영 필요")
        else:
            st.subheader("🔄 거래 저장")

        col1, col2 = st.columns(2)
        with col1:
            exec_date = st.date_input("실행일", value=date.today(), key="rebal_exec_date")
        with col2:
            use_integer = st.checkbox("정수 수량 적용", value=True, key="rebal_integer")

        col_save, col_tg = st.columns(2)
        with col_save:
            save_clicked = st.button("✅ 거래 저장", key="btn_execute_rebal", type="primary", use_container_width=True)
        with col_tg:
            st.button(
                "📱 Telegram 승인 요청",
                key="btn_tg_request",
                use_container_width=True,
                disabled=True,
                help="추후 Telegram 봇 연동 시 활성화됩니다."
            )

        if save_clicked:
            success = 0
            holdings_map = {h['asset_id'].upper(): h for h in holdings_data}
            for o in rebal_orders:
                delta_qty = int(o['suggested_qty']) if use_integer else o['suggested_qty']
                if delta_qty <= 0:
                    continue
                AssetStorage.save_asset(o['asset_id'], o['asset_id'], 'etf', 'USD', o['asset_id'], 'yfinance')
                current_qty = holdings_map.get(o['asset_id'], {}).get('quantity', 0)
                if o['action'] == 'buy':
                    new_qty = current_qty + delta_qty
                else:
                    new_qty = max(0, current_qty - delta_qty)
                result = AssetStorage.upsert_holding_qty(account_id, o['asset_id'], new_qty)
                if result:
                    success += 1

            if signals:
                event_ids = self._record_signal_events(account_id, portfolio_name, signals, last_snapshot, total_value, exec_date)
                if success > 0 and event_ids:
                    AssetStorage.mark_signal_events_executed(event_ids)

            if success > 0:
                st.success(f"✅ {success}개 거래 저장 완료!")
                st.session_state.pop(SK.CURRENT_SIGNALS, None)
                st.rerun()

    # ══════════════════════════════════════════════════════
    # Tab 2: 보유 현황 & 입출금
    # ══════════════════════════════════════════════════════
    def _render_holdings_and_deposits(self, account_id):
        tab_hold, tab_dep, tab_event = st.tabs(["📊 보유 수량", "💰 입출금", "📡 신호 이벤트"])

        with tab_hold:
            self._render_holdings_simple(account_id)
        with tab_dep:
            self._render_deposits(account_id)
        with tab_event:
            self._render_signal_events(account_id)

    def _render_holdings_simple(self, account_id):
        """보유 현황 — 수량 + 종가/현재가 기반 평가액"""
        from core.data import DataService

        holdings = AssetStorage.list_holdings(account_id)

        # 가격 소스 토글
        price_mode = st.radio(
            "가격 기준",
            ["종가 기준", "현재가 기준"],
            horizontal=True,
            key="holdings_price_mode"
        )

        tickers = [h['asset_id'] for h in holdings]
        if tickers:
            if price_mode == "현재가 기준":
                price_map = DataService.fetch_current_prices(tickers)
            else:
                price_map = DataService.get_prices(tickers)

        rows = []
        total_value = 0
        for h in holdings:
            price = price_map.get(h['asset_id'].upper()) if tickers else None
            val = h['quantity'] * price if price else 0
            total_value += val
            rows.append({
                '티커': h['asset_id'],
                '보유수량': int(h['quantity']),
                '단가': f"${price:,.2f}" if price else "—",
                '평가액 (USD)': f"${val:,.0f}",
                '최종수정': str(h['last_updated'])[:10] if h['last_updated'] else "—"
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.metric("총 평가액 (USD)", f"${total_value:,.0f}")
        else:
            st.info("보유 자산이 없습니다.")

        st.divider()
        st.subheader("✏️ 수량 직접 수정")
        col1, col2, col3 = st.columns(3)
        with col1:
            edit_ticker = st.text_input("티커", placeholder="TQQQ", key="edit_hold_ticker")
        with col2:
            edit_qty = st.number_input("수량", min_value=0.0, step=1.0, key="edit_hold_qty")
        with col3:
            st.write("")
            st.write("")
            if st.button("수량 저장", key="btn_edit_hold", use_container_width=True):
                if not edit_ticker:
                    st.error("티커를 입력하세요.")
                else:
                    ticker = edit_ticker.upper()
                    AssetStorage.save_asset(ticker, ticker, 'etf', 'USD', ticker, 'yfinance')
                    result = AssetStorage.update_holding_quantity(account_id, ticker, edit_qty)
                    st.success("✅ 저장됨") if result else st.error("저장 실패")
                    if result:
                        st.rerun()

    def _render_signal_events(self, account_id):
        events = AssetStorage.list_signal_events(account_id)
        if not events:
            st.info("기록된 신호 이벤트가 없습니다.")
            return
        rows = []
        for e in events:
            stage_str = f"S{e['prev_stage']}→S{e['new_stage']}" if e['prev_stage'] != e['new_stage'] else f"S{e['new_stage']}"
            asset_str = f"{e['prev_asset']}→{e['new_asset']}" if e['prev_asset'] != e['new_asset'] else (e['new_asset'] or "—")
            rows.append({
                '날짜': e['event_date'],
                '포트폴리오': e['portfolio_name'],
                '전략': e['strategy_name'],
                '스테이지': stage_str,
                '자산': asset_str,
                '액션': e['action'] or "—",
                '실행': '✅' if e['is_executed'] else '기록됨'
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 이벤트", len(events))
        with col2:
            st.metric("실행 완료", sum(1 for e in events if e['is_executed']))

    # ══════════════════════════════════════════════════════
    # Tab 3: 설정
    # ══════════════════════════════════════════════════════
    def _render_settings(self, account_id):
        tab_sync, tab_account = st.tabs(["🔀 수량 동기화", "🏦 계좌 관리"])

        with tab_sync:
            SyncView().render(account_id=account_id)
        with tab_account:
            self._render_account_setup(account_id)

    def _render_deposits(self, account_id):
        accounts = AssetStorage.list_accounts()
        dep_account_names = [a['name'] for a in accounts]
        dep_default_idx = next((i for i, a in enumerate(accounts) if a['id'] == account_id), 0)
        dep_selected_name = st.selectbox("계좌", dep_account_names, index=dep_default_idx, key="dep_account")
        dep_account = next((a for a in accounts if a['name'] == dep_selected_name), None)
        dep_account_id = dep_account['id'] if dep_account else account_id

        deposits = AssetStorage.list_deposits(dep_account_id)
        if deposits:
            dep_data = [{
                '날짜': d['date'],
                '유형': '입금' if d['type'] == 'deposit' else '출금',
                '금액': f"{d['currency']} {d['amount']:,.0f}",
                '비고': d['notes'] or "—"
            } for d in deposits]
            st.dataframe(pd.DataFrame(dep_data), use_container_width=True, hide_index=True)

        st.subheader("➕ 입출금 기록")
        col1, col2, col3 = st.columns(3)
        with col1:
            dep_date = st.date_input("날짜", value=date.today(), key="dep_date")
        with col2:
            dep_type = st.selectbox("유형", ["deposit", "withdraw"],
                format_func=lambda x: "입금" if x == "deposit" else "출금", key="dep_type")
        with col3:
            dep_currency = st.selectbox("통화", ["KRW", "USD"], key="dep_currency")
        col1, col2 = st.columns(2)
        with col1:
            dep_amount = st.number_input("금액", min_value=0.0, step=1.0, key="dep_amount")
        with col2:
            dep_notes = st.text_input("비고", key="dep_notes")

        if st.button("기록", key="btn_save_dep"):
            if dep_amount <= 0:
                st.error("금액을 입력해주세요")
            else:
                result = AssetStorage.save_deposit(dep_account_id, dep_amount, dep_type, dep_date, dep_currency, dep_notes)
                st.success("✅ 기록됨") if result else st.error("기록 실패")

    def _render_account_setup(self, account_id=None):
        accounts = AssetStorage.list_accounts()
        if accounts:
            st.subheader("📋 등록된 계좌")
            for acc in accounts:
                st.write(f"• **{acc['name']}** — {acc['type'].upper()} ({acc['currency']})")
            st.divider()

        st.subheader("➕ 새 계좌 추가")
        col1, col2 = st.columns(2)
        with col1:
            account_name = st.text_input("계좌명", key="new_account_name")
        with col2:
            account_type = st.selectbox("유형", ["brokerage", "crypto", "bank"],
                format_func=lambda x: {"brokerage": "증권사", "crypto": "암호화폐", "bank": "은행"}[x],
                key="new_account_type")
        col1, col2 = st.columns(2)
        with col1:
            currency = st.selectbox("기본 통화", ["USD", "KRW"], key="new_currency")
        with col2:
            broker_code = st.text_input("증권사 코드 (선택)", key="new_broker_code")

        if st.button("계좌 생성", key="btn_new_account"):
            if account_name:
                result = AssetStorage.save_account(account_name, account_type, currency, broker_code or None)
                if result:
                    st.success(f"✅ {account_name} 생성됨")
                    st.rerun()
                else:
                    st.error("생성 실패 (중복 계좌명)")
            else:
                st.error("계좌명을 입력해주세요")

    # ══════════════════════════════════════════════════════
    # 핵심 로직
    # ══════════════════════════════════════════════════════
    @staticmethod
    def _compute_target_allocation(configs, signals):
        """신호 기반 목표 배분 계산 → [{ticker, weight, label}]"""
        result = []
        for config in configs:
            name = config['name']
            weight = config.get('weight', 0)
            params = config.get('params', {})
            sig = signals.get(name, {})

            stage = sig.get('stage', 1) if sig else 1
            base_asset = sig.get('base_asset') or params.get('base_asset', 'QQQ')
            lev_asset = sig.get('lev_asset') or params.get('leverage_asset', 'TQQQ')
            stype = config.get('type', 'equity')

            if stype == 'bond' or 'bond' in name.lower():
                base_asset = sig.get('base_asset', 'TLT') if sig else 'TLT'
                lev_asset = sig.get('lev_asset') or params.get('leverage_asset', 'TMF')

            base_ratio, lev_ratio = STAGE_RATIOS.get(stage, (0.7, 0.3))

            if base_ratio > 0 and base_asset:
                result.append({
                    'ticker': base_asset,
                    'weight': weight * base_ratio,
                    'label': f"{name} / {base_asset} (S{stage}-Base)"
                })
            if lev_ratio > 0 and lev_asset:
                result.append({
                    'ticker': lev_asset,
                    'weight': weight * lev_ratio,
                    'label': f"{name} / {lev_asset} (S{stage}-Lev)"
                })
            if base_ratio == 0 and lev_ratio == 0:
                result.append({
                    'ticker': base_asset,
                    'weight': weight,
                    'label': f"{name} / {base_asset} (S{stage})"
                })
        return result

    @staticmethod
    def _calculate_rebal_orders(holdings_data, target_alloc, total_value, threshold=5.0):
        """보유 현황 vs 목표 배분 비교 → 주문 리스트 (미보유 신규 자산 포함)"""
        target_map = {item['ticker'].upper(): item for item in target_alloc}
        holdings_map = {h['asset_id'].upper(): h for h in holdings_data}
        orders = []

        all_keys = set(holdings_map.keys()) | set(target_map.keys())

        for key in all_keys:
            h = holdings_map.get(key)
            t = target_map.get(key)

            current_val = h['current_value'] if h else 0.0
            current_price = (h['current_price'] or 0) if h else 0.0
            target_w = t['weight'] if t else 0.0
            target_val = total_value * target_w
            current_w = (current_val / total_value) if total_value > 0 else 0.0
            diff_pct = (current_w - target_w) * 100

            if abs(diff_pct) < threshold:
                continue

            if not current_price and t:
                from core.data import DataService
                current_price = DataService.get_price(t['ticker']) or 0

            order_amount = target_val - current_val
            suggested_qty = order_amount / current_price if current_price > 0 else 0

            orders.append({
                'asset_id': key,
                'asset_name': h['asset_name'] if h else (t['ticker'] if t else key),
                'action': 'buy' if order_amount > 0 else 'sell',
                'current_weight': current_w * 100,
                'target_weight': target_w * 100,
                'current_value': current_val,
                'target_value': target_val,
                'order_amount': order_amount,
                'suggested_qty': abs(suggested_qty),
                'current_price': current_price
            })

        return orders

    @staticmethod
    def _detect_signal_changes(current_signals, last_snapshot):
        """현재 신호 vs 마지막 스냅샷 비교 → 변화된 전략 목록"""
        changed = {}
        for name, sig in current_signals.items():
            prev = last_snapshot.get(name, {})
            prev_stage = prev.get('stage', -1)
            prev_asset = prev.get('asset', '')
            new_stage = sig.get('stage', 0)
            new_asset = sig.get('base_asset', '')

            if prev_stage == -1:
                continue
            if prev_stage != new_stage or prev_asset != new_asset:
                action = 'buy' if new_stage > prev_stage else ('sell' if new_stage < prev_stage else 'asset_change')
                changed[name] = {
                    'prev_stage': prev_stage, 'new_stage': new_stage,
                    'prev_asset': prev_asset, 'new_asset': new_asset,
                    'action': action
                }
        return changed

    @staticmethod
    def _record_signal_events(account_id, portfolio_name, signals, last_snapshot, total_value, event_date=None):
        """현재 신호를 signal_events 테이블에 기록 → 저장된 event_id 목록 반환"""
        if event_date is None:
            event_date = date.today()
        event_ids = []
        for name, sig in signals.items():
            prev = last_snapshot.get(name, {})
            prev_stage = prev.get('stage', 0)
            prev_asset = prev.get('asset', '')
            new_stage = sig.get('stage', 0)
            new_asset = sig.get('base_asset', '')
            stype = sig.get('strat_type', 'equity')

            if prev_stage != new_stage or prev_asset != new_asset or not prev:
                action = 'buy' if new_stage > prev_stage else ('sell' if new_stage < prev_stage else 'rebalance')
                eid = AssetStorage.save_signal_event(
                    account_id=account_id,
                    portfolio_name=portfolio_name,
                    strategy_name=name,
                    strat_type=stype,
                    prev_stage=prev_stage,
                    new_stage=new_stage,
                    prev_asset=prev_asset,
                    new_asset=new_asset,
                    action=action,
                    total_value=total_value,
                    event_date=event_date
                )
                if eid:
                    event_ids.append(eid)
        return event_ids

    @staticmethod
    def _extract_signals(result, configs):
        from core.signal_service import extract_signals
        return extract_signals(result, configs)

    @staticmethod
    def _fetch_current_signals(portfolio_name, configs):
        try:
            from core.data import DataService
            from ui.portfolio_view import PortfolioView

            # 사이드바 데이터 모드에 따라 증분/실시간 분기
            if st.session_state.get('_data_mode_realtime', False):
                data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data_realtime()
            else:
                data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data()
            if not data_dict:
                return {}

            _all_tickers = list(data_dict.keys()) + ['^VIX', '^VXN']
            _cur_prices = DataService.fetch_current_prices(_all_tickers)

            sidebar_start = st.session_state.get('start_input', None)
            start_date = sidebar_start or (data_dict['QQQ'].index.min().date() if 'QQQ' in data_dict else None)

            result = PortfolioView.compute_portfolio_signals(
                portfolio_name, data_dict, fg_df, vxn_df, macro_df, news_df,
                start_date, cur_prices=_cur_prices,
            )
            if result:
                st.session_state[SK.PORTFOLIO_REALTIME_RESULT] = result
                st.session_state[SK.PORTFOLIO_SELECTED_NAME] = portfolio_name
                return AssetView._extract_signals(result, configs)

        except Exception as e:
            print(f"[_fetch_current_signals] 오류: {e}")
            import traceback; traceback.print_exc()
        return {}

    @staticmethod
    def _get_current_price(asset_id, asset_name):
        from core.data import DataService
        price = DataService.get_price(asset_id)
        if price:
            return price
        for token in str(asset_name).split():
            if token.isupper() and 2 <= len(token) <= 6:
                price = DataService.get_price(token)
                if price:
                    return price
        return None

    @staticmethod
    def _get_historical_price(ticker, target_date):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=str(target_date), end=str(target_date + pd.Timedelta(days=5)))
            if not hist.empty and 'Close' in hist.columns:
                return float(hist['Close'].iloc[0])
        except Exception as e:
            print(f"Historical price error [{ticker}]: {e}")
        return None
