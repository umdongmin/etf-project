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
        selected_account_name = st.selectbox("계좌 선택", account_names, index=default_acc_idx)
        selected_account = next((a for a in accounts if a['name'] == selected_account_name), None)
        account_id = selected_account['id']

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 포트폴리오 현황", "🔔 신호 이벤트", "📋 거래 내역", "🔀 자산 동기화", "⚙️ 계좌 설정"])

        with tab1:
            self._render_portfolio_status(account_id)
        with tab2:
            self._render_signal_events(account_id)
        with tab3:
            self._render_transactions(account_id)
        with tab4:
            SyncView().render(account_id=account_id)
        with tab5:
            self._render_account_setup(account_id)

    # ══════════════════════════════════════════════════════
    # Tab 1: 포트폴리오 현황
    # ══════════════════════════════════════════════════════
    def _render_portfolio_status(self, account_id):
        from core.storage import StrategyStorage

        st.header("📊 포트폴리오 현황")

        # ── 포트폴리오 선택 ──────────────────────────────────────────
        portfolio_names = StrategyStorage.list_portfolios()
        if not portfolio_names:
            st.warning("등록된 포트폴리오가 없습니다.")
            return

        default_idx = next((i for i, n in enumerate(portfolio_names) if n.startswith('A')), 0)

        col1, col2 = st.columns([3, 1])
        with col1:
            portfolio_name = st.selectbox("포트폴리오", portfolio_names, index=default_idx, key="status_portfolio")
        with col2:
            st.write("")
            refresh = st.button("🔄 신호 갱신", key="btn_refresh_signal", use_container_width=True)

        portfolio = StrategyStorage.load_portfolio(portfolio_name)
        if not portfolio:
            st.error("포트폴리오를 불러올 수 없습니다.")
            return
        configs = portfolio.get('configs', [])

        # ── 현재 전략 신호 조회 ──────────────────────────────────────
        cached_portfolio = st.session_state.get(SK.CURRENT_SIGNALS_PORTFOLIO)
        cached_signals = st.session_state.get(SK.CURRENT_SIGNALS, {})
        portfolio_changed = cached_portfolio != portfolio_name

        if refresh:
            # 신호 갱신 버튼: signal 캐시 무효화 후 재계산
            for mode in ('closing', 'realtime'):
                st.session_state.pop(SK.sig_cache(portfolio_name, mode), None)
                st.session_state.pop(SK.sig_cache_ts(portfolio_name, mode), None)
            with st.spinner("신호 계산 중..."):
                signals = self._fetch_current_signals(portfolio_name, configs)
            st.session_state[SK.CURRENT_SIGNALS] = signals
            st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
        elif portfolio_changed or not cached_signals:
            # 포트폴리오 변경 또는 캐시 없음: 실시간 모니터링 캐시 재사용 시도
            rt_result = st.session_state.get(SK.PORTFOLIO_REALTIME_RESULT)
            rt_portfolio = st.session_state.get(SK.PORTFOLIO_SELECTED_NAME)
            if rt_result and rt_portfolio == portfolio_name:
                signals = self._extract_signals(rt_result, configs)
                st.session_state[SK.CURRENT_SIGNALS] = signals
                st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
            else:
                # 캐시 없음 → 신호 갱신 안내
                signals = {}
                st.session_state[SK.CURRENT_SIGNALS] = {}
                st.session_state[SK.CURRENT_SIGNALS_PORTFOLIO] = portfolio_name
        else:
            signals = cached_signals

        # ── 신호 변화 감지 ───────────────────────────────────────────
        last_snapshot = AssetStorage.get_latest_signal_snapshot(account_id, portfolio_name)
        changed_strategies = self._detect_signal_changes(signals, last_snapshot)

        if changed_strategies:
            st.warning(f"⚠️ **신호 변경 감지** — {len(changed_strategies)}개 전략에서 변화가 감지되었습니다.")
            change_rows = []
            for name, chg in changed_strategies.items():
                change_rows.append({
                    '전략': name,
                    '이전': f"S{chg['prev_stage']} / {chg['prev_asset']}",
                    '현재': f"S{chg['new_stage']} / {chg['new_asset']}",
                    '변화': chg['action']
                })
            st.dataframe(pd.DataFrame(change_rows), use_container_width=True, hide_index=True)

        # ── 신호 없을 때 안내 ─────────────────────────────────────────
        if not signals:
            st.info("📡 신호 데이터가 없습니다. **[🔄 신호 갱신]** 버튼을 클릭하거나 실시간 모니터링에서 먼저 신호를 계산해주세요.")

        # ── 현재 신호 요약 ───────────────────────────────────────────
        if signals:
            st.subheader("📡 현재 전략 신호")
            sig_cols = st.columns(len(signals))
            for i, (name, sig) in enumerate(signals.items()):
                with sig_cols[i]:
                    s = sig['stage']
                    st.metric(
                        label=name,
                        value=f"{STAGE_ICONS.get(s, '⚪')} {STAGE_LABELS.get(s, f'S{s}')}",
                        delta=f"{sig['base_asset']} → {sig.get('lev_asset','')}" if sig.get('lev_asset') else sig['base_asset']
                    )

        st.divider()

        # ── 보유 현황 vs 목표 비중 ───────────────────────────────────
        holdings = AssetStorage.list_holdings(account_id)

        if not holdings:
            st.info("보유 자산이 없습니다.")
            with st.expander("📊 포트폴리오 초기 설정", expanded=True):
                self._render_portfolio_init(account_id, portfolio_name, configs, signals)
            return

        # 현재 보유 현황 계산
        holdings_data = []
        total_value = 0
        for h in holdings:
            price = self._get_current_price(h['asset_id'], h['asset_name'])
            val = h['quantity'] * price if price else 0
            total_value += val
            holdings_data.append({**h, 'current_price': price, 'current_value': val})

        # 신호 기반 목표 비중 계산
        target_alloc = self._compute_target_allocation(configs, signals)
        # asset_id → target_weight 맵
        target_map = {item['ticker'].upper(): item['weight'] for item in target_alloc}

        # 비중 분석표
        st.subheader("💼 보유 현황 vs 목표")
        rows = []
        total_cost = 0
        for h in holdings_data:
            key = h['asset_id'].upper()
            target_w = target_map.get(key, 0.0) * 100
            current_w = (h['current_value'] / total_value * 100) if total_value > 0 else 0
            diff = current_w - target_w
            pnl = h['current_value'] - h['total_cost']
            pnl_pct = pnl / h['total_cost'] * 100 if h['total_cost'] > 0 else 0
            total_cost += h['total_cost']
            status = "⚠️" if abs(diff) >= 5 else "✅"
            rows.append({
                '상태': status,
                '자산': h['asset_name'],
                '수량': f"{h['quantity']:,.4f}",
                '현재가': f"${h['current_price']:,.2f}" if h['current_price'] else "—",
                '평가액': f"${h['current_value']:,.0f}",
                '현재비중': f"{current_w:.1f}%",
                '목표비중': f"{target_w:.1f}%",
                '편차': f"{diff:+.1f}%",
                '손익': f"${pnl:+,.0f} ({pnl_pct:+.1f}%)"
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 평가액", f"${total_value:,.0f}")
        with col2:
            st.metric("총 투입액", f"${total_cost:,.0f}")
        with col3:
            total_pnl_pct = (total_value - total_cost) / total_cost * 100 if total_cost > 0 else 0
            st.metric("수익률", f"{total_pnl_pct:+.2f}%", delta=f"${total_value - total_cost:+,.0f}")

        st.divider()

        # ── 리밸런싱 실행 영역 ───────────────────────────────────────
        if changed_strategies:
            st.subheader("🔄 리밸런싱 — ⚠️ 포지션 변경 반영 필요")
        else:
            st.subheader("🔄 리밸런싱")

        rebal_orders = self._calculate_rebal_orders(holdings_data, target_alloc, total_value)

        if not rebal_orders:
            st.success("✅ 현재 모든 자산이 목표 비중 범위 내에 있습니다. (편차 < 5%)")
        else:
            order_rows = []
            for o in rebal_orders:
                order_rows.append({
                    '자산': o['asset_id'],
                    '액션': '🟢 매수' if o['action'] == 'buy' else '🔴 매도',
                    '현재금액': f"${o['current_value']:,.0f}",
                    '목표금액': f"${o['target_value']:,.0f}",
                    '주문금액': f"${abs(o['order_amount']):,.0f}",
                    '현재비중': f"{o['current_weight']:.1f}%",
                    '목표비중': f"{o['target_weight']:.1f}%",
                    '제안수량': f"{abs(o['suggested_qty']):,.4f}",
                    '현재가': f"${o['current_price']:,.2f}"
                })
            st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                exec_date = st.date_input("실행일", value=date.today(), key="rebal_exec_date")
            with col2:
                use_integer = st.checkbox("정수 수량 적용", value=True, key="rebal_integer")

            if st.button("✅ 리밸런싱 실행 (거래 기록)", key="btn_execute_rebal", type="primary"):
                success = 0
                for o in rebal_orders:
                    qty = int(o['suggested_qty']) if use_integer else o['suggested_qty']
                    if qty <= 0:
                        continue
                    AssetStorage.save_asset(o['asset_id'], o['asset_id'], 'etf', 'USD', o['asset_id'], 'yfinance')
                    result = AssetStorage.save_transaction(
                        account_id, o['asset_id'], o['action'],
                        qty * o['current_price'],
                        quantity=float(qty), price=o['current_price'],
                        tx_date=exec_date,
                        notes=f"리밸런싱: {portfolio_name}"
                    )
                    if result:
                        success += 1

                # 신호 이벤트 기록
                if signals:
                    self._record_signal_events(account_id, portfolio_name, signals, last_snapshot, total_value)

                if success > 0:
                    st.success(f"✅ {success}개 거래 기록 완료!")
                    st.session_state.pop(SK.CURRENT_SIGNALS, None)

        st.info("💡 보유 수량 초기 설정은 **🔀 자산 동기화** 메뉴를 이용해주세요.", icon="🔀")

    # ══════════════════════════════════════════════════════
    # Tab 2: 신호 이벤트
    # ══════════════════════════════════════════════════════
    def _render_signal_events(self, account_id):
        st.header("🔔 신호 이벤트")

        events = AssetStorage.list_signal_events(account_id)

        if not events:
            st.info("기록된 신호 이벤트가 없습니다.\n\n포트폴리오 현황 탭에서 리밸런싱을 실행하면 신호 이벤트가 자동으로 기록됩니다.")
            return

        st.caption(f"총 {len(events)}개 이벤트")

        rows = []
        for e in events:
            stage_change = ""
            if e['prev_stage'] != e['new_stage']:
                stage_change = f"S{e['prev_stage']}→S{e['new_stage']}"
            asset_change = ""
            if e['prev_asset'] != e['new_asset']:
                asset_change = f"{e['prev_asset']}→{e['new_asset']}"

            rows.append({
                '날짜': e['event_date'],
                '포트폴리오': e['portfolio_name'],
                '전략': e['strategy_name'],
                '유형': e['strat_type'].upper(),
                '스테이지 변화': stage_change or f"S{e['new_stage']}",
                '자산 변화': asset_change or e['new_asset'] or "—",
                '액션': e['action'] or "—",
                '실행': '✅' if e['is_executed'] else '기록됨',
                '비고': e['notes'] or "—"
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 통계
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 이벤트", len(events))
        with col2:
            executed = sum(1 for e in events if e['is_executed'])
            st.metric("실행 완료", executed)
        with col3:
            portfolios = len(set(e['portfolio_name'] for e in events))
            st.metric("포트폴리오 수", portfolios)

    # ══════════════════════════════════════════════════════
    # Tab 3: 거래 내역 (조회 전용)
    # ══════════════════════════════════════════════════════
    def _render_transactions(self, account_id):
        st.header("📋 거래 내역")

        transactions = AssetStorage.list_transactions(account_id)

        if not transactions:
            st.info("거래 내역이 없습니다.")
            return

        data = [{
            '날짜': tx['tx_date'],
            '자산': tx['asset_name'],
            '유형': tx['type'].upper(),
            '수량': f"{tx['quantity']:,.4f}" if tx['quantity'] else "—",
            '단가': f"${tx['price']:,.2f}" if tx['price'] else "—",
            '금액': f"${tx['amount']:,.2f}",
            '수수료': f"${tx['fee']:,.2f}" if tx['fee'] > 0 else "—",
            '비고': tx['notes'] or "—"
        } for tx in transactions]

        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        st.caption(f"총 {len(transactions)}건")

    # ══════════════════════════════════════════════════════
    # Tab 4: 계좌 설정 + 입출금
    # ══════════════════════════════════════════════════════
    def _render_account_setup(self, account_id=None):
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

        st.divider()
        accounts = AssetStorage.list_accounts()
        if accounts:
            st.subheader("📋 등록된 계좌")
            for acc in accounts:
                st.write(f"• **{acc['name']}** — {acc['type'].upper()} ({acc['currency']})")

        if account_id is None:
            return

        st.divider()
        st.subheader("💰 입출금")

        # 계좌 선택 (현재 선택된 계좌가 기본값)
        dep_account_names = [a['name'] for a in accounts]
        dep_default_idx = next((i for i, a in enumerate(accounts) if a['id'] == account_id), 0)
        dep_selected_name = st.selectbox("계좌 선택", dep_account_names, index=dep_default_idx, key="dep_account")
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

        st.subheader("💳 입출금 기록")
        col1, col2 = st.columns(2)
        with col1:
            dep_date = st.date_input("날짜", value=date.today(), key="dep_date")
        with col2:
            dep_type = st.selectbox("유형", ["deposit", "withdraw"],
                format_func=lambda x: "입금" if x == "deposit" else "출금", key="dep_type")
        col1, col2 = st.columns(2)
        with col1:
            dep_currency = st.selectbox("통화", ["KRW", "USD"], key="dep_currency")
        with col2:
            dep_amount = st.number_input("금액", min_value=0.0, step=1.0, key="dep_amount")
        dep_notes = st.text_input("비고", key="dep_notes")

        if st.button("기록", key="btn_save_dep"):
            if dep_amount <= 0:
                st.error("금액을 입력해주세요")
            else:
                result = AssetStorage.save_deposit(dep_account_id, dep_amount, dep_type, dep_date, dep_currency, dep_notes)
                st.success(f"✅ [{dep_selected_name}] 기록됨") if result else st.error("기록 실패")

    # ══════════════════════════════════════════════════════
    # 핵심 로직
    # ══════════════════════════════════════════════════════
    @staticmethod
    def _compute_target_allocation(configs, signals):
        """신호 기반 목표 배분 계산 → [{ticker, weight, label}]"""
        # STAGE_RATIOS imported from core.constants

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

        # 보유 자산 + 목표에 있는 신규 자산 모두 처리
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

            # 신규 자산의 경우 현재가 조회
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
                # 최초 등록
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
        """현재 신호를 signal_events 테이블에 기록"""
        if event_date is None:
            event_date = date.today()
        for name, sig in signals.items():
            prev = last_snapshot.get(name, {})
            prev_stage = prev.get('stage', 0)
            prev_asset = prev.get('asset', '')
            new_stage = sig.get('stage', 0)
            new_asset = sig.get('base_asset', '')
            stype = sig.get('strat_type', 'equity')

            if prev_stage != new_stage or prev_asset != new_asset or not prev:
                action = 'buy' if new_stage > prev_stage else ('sell' if new_stage < prev_stage else 'rebalance')
                AssetStorage.save_signal_event(
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

    @staticmethod
    def _extract_signals(result, configs):
        """signal_service.extract_signals() 위임 래퍼"""
        from core.signal_service import extract_signals
        return extract_signals(result, configs)

    @staticmethod
    def _fetch_current_signals(portfolio_name, configs):
        """신호 갱신 버튼 클릭 시 호출.
        현재가 있으면 실시간 신호, 없으면 종가 기준 신호를 계산한다.
        compute_portfolio_signals → signal_service로 단일 경로 사용."""
        try:
            from core.data import DataService
            from ui.portfolio_view import PortfolioView

            data_dict, fg_df, vxn_df, macro_df, news_df, _, _ = DataService.load_all_data()
            if not data_dict:
                return {}

            # 현재가 조회 (없으면 None → 종가 기준)
            _all_tickers = list(data_dict.keys()) + ['^VIX', '^VXN']
            _cur_prices  = DataService.fetch_current_prices(_all_tickers)

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
        """DataService.get_price() 래퍼 — asset_name 파싱 fallback 포함"""
        from core.data import DataService
        price = DataService.get_price(asset_id)
        if price:
            return price
        # asset_name에서 대문자 토큰 추출하여 추가 시도
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
