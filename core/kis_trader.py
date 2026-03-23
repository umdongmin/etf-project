"""
KISTrader — 신호 → 주문 변환 + 파이프라인 오케스트레이션

신호 계산 결과를 받아서:
  1. 목표수량 vs 현재수량 차이 → OrderRequest 목록 생성
  2. TradingSafetyGuard 검증 (9개 체크)
  3. 검증 결과 DB 기록 (통과/거부 모두)
  4. 모의투자 자동승인 or Telegram 승인 요청
  5. 승인 시 KISClient.place_order() 실행
"""

import os
import json
import datetime
import secrets
import traceback

import requests

from core.kis_client import KISClient, KISError, TradingHaltedError, AuthenticationError
from core.kis_safety import OrderRequest, SafetyCheckResult, TradingSafetyGuard
from core.storage import OrderStorage


class KISTrader:
    """신호 → 주문 변환 + 파이프라인 실행"""

    def __init__(self):
        self.mode = self._resolve_mode()
        self._client = None  # 지연 초기화 (disabled면 생성 안 함)
        self._guard = None   # 계좌별로 생성

    # ── 초기화 ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_mode() -> str:
        """TRADING_MODE 환경변수 해석 — 불명확하면 'disabled'"""
        mode = os.environ.get('TRADING_MODE', 'disabled').lower().strip()
        if mode not in ('paper', 'live', 'disabled'):
            print(f"[KIS] Invalid TRADING_MODE='{mode}', forcing 'disabled'", flush=True)
            return 'disabled'
        return mode

    @property
    def client(self) -> KISClient:
        """KISClient 지연 초기화"""
        if self._client is None:
            self._client = KISClient(self.mode)
        return self._client

    def is_auto_approve(self, account_id: int = None) -> bool:
        """모의투자 자동승인 여부 확인"""
        if self.mode != 'paper':
            return False
        if account_id is not None:
            config = OrderStorage.load_trading_config(account_id)
            if config:
                return config.get('auto_approve_paper', False)
        return False

    # ── 메인 파이프라인 ──────────────────────────────────────────────────

    def process_signal_orders(self, account_id: int, realtime_signals: dict,
                               closing_signals: dict, target_qty: dict,
                               holdings_map: dict, cur_prices: dict,
                               portfolio_name: str = '',
                               signal_event_ids: dict = None) -> list:
        """
        신호 차이 → OrderRequest 생성 → SafetyGuard 검증

        Parameters
        ----------
        account_id : int — DB 계좌 ID
        realtime_signals : dict — 실시간 신호 {strategy_name: {...}}
        closing_signals : dict — 종가 신호
        target_qty : dict — {ticker: 목표수량}
        holdings_map : dict — {ticker: {qty, price, value}}
        cur_prices : dict — {ticker: 현재가}
        portfolio_name : str
        signal_event_ids : dict — {strategy_name: signal_event_id}

        Returns
        -------
        list[OrderRequest] — SafetyGuard 통과한 주문만 반환
        """
        if self.mode == 'disabled':
            print("[KIS] Trading disabled, skipping order generation", flush=True)
            return []

        self._guard = TradingSafetyGuard(self.mode, account_id)
        signal_event_ids = signal_event_ids or {}

        # 1. 주문 목록 생성
        orders = self._generate_orders(
            account_id=account_id,
            target_qty=target_qty,
            holdings_map=holdings_map,
            cur_prices=cur_prices,
            realtime_signals=realtime_signals,
            portfolio_name=portfolio_name,
            signal_event_ids=signal_event_ids,
        )

        if not orders:
            print("[KIS] No orders to process", flush=True)
            return []

        # 2. SafetyGuard 검증 + DB 기록
        validated = []
        for order in orders:
            result = self._guard.validate(order)

            # DB에 검증 결과 기록 (통과/거부 모두)
            order_data = {
                'account_id': order.account_id,
                'portfolio_name': order.portfolio_name,
                'strategy_name': order.strategy_name,
                'signal_event_id': order.signal_event_id,
                'ticker': order.ticker,
                'side': order.side,
                'quantity': order.quantity,
                'price': order.price,
                'order_type': order.order_type,
                'estimated_amount': order.estimated_amount,
                'safety_passed': result.passed,
                'safety_checks': result.checks,
                'rejection_reason': result.rejection_reason if not result.passed else None,
                'approval_status': 'pending' if result.passed else 'rejected',
                'execution_status': 'not_sent',
                'trading_mode': order.trading_mode,
            }
            order_log_id = OrderStorage.save_order_log(order_data)

            if result.passed:
                # order_log_id를 order에 첨부 (승인/실행 시 사용)
                order._order_log_id = order_log_id
                validated.append(order)
                passed_count = sum(1 for c in result.checks if c['passed'])
                print(f"[KIS PASSED] {order.ticker} {order.side} {order.quantity}주 "
                      f"(${order.estimated_amount:,.0f}) — {passed_count}/{len(result.checks)} checks", flush=True)
            else:
                print(f"[KIS REJECTED] {order.ticker} {order.side} {order.quantity}주: "
                      f"{result.rejection_reason}", flush=True)

        return validated

    # ── 주문 생성 ────────────────────────────────────────────────────────

    def _generate_orders(self, account_id: int, target_qty: dict,
                          holdings_map: dict, cur_prices: dict,
                          realtime_signals: dict, portfolio_name: str,
                          signal_event_ids: dict) -> list:
        """목표수량 - 현재수량 = 주문수량 변환"""
        orders = []

        # signal → ticker 매핑 구성
        ticker_to_strategy = {}
        ticker_to_event_id = {}
        for name, sig in realtime_signals.items():
            base = sig.get('base_asset', '').upper()
            lev = sig.get('lev_asset', '').upper()
            if base:
                ticker_to_strategy[base] = name
            if lev:
                ticker_to_strategy[lev] = name
            event_id = signal_event_ids.get(name)
            if event_id:
                if base:
                    ticker_to_event_id[base] = event_id
                if lev:
                    ticker_to_event_id[lev] = event_id

        for ticker, tgt_qty in target_qty.items():
            cur_qty = int(holdings_map.get(ticker, {}).get('qty', 0))
            diff = tgt_qty - cur_qty

            if diff == 0:
                continue

            price = 0
            if cur_prices:
                price = cur_prices.get(ticker, 0)
                if isinstance(price, dict):
                    price = price.get('price', 0)
            if price <= 0:
                # holdings_map에서 가격 참조
                price = holdings_map.get(ticker, {}).get('price', 0)
            if price <= 0:
                print(f"[KIS] {ticker} price unavailable, skipping order", flush=True)
                continue

            strategy_name = ticker_to_strategy.get(ticker, 'unknown')
            event_id = ticker_to_event_id.get(ticker)

            orders.append(OrderRequest(
                account_id=account_id,
                ticker=ticker,
                side='buy' if diff > 0 else 'sell',
                quantity=abs(diff),
                price=price,
                order_type='market',
                signal_event_id=event_id,
                portfolio_name=portfolio_name,
                strategy_name=strategy_name,
                reason='stage_change',
                estimated_amount=abs(diff) * price,
                trading_mode=self.mode,
            ))

        # 매도를 매수보다 먼저 (자금 확보)
        orders.sort(key=lambda o: (0 if o.side == 'sell' else 1))
        return orders

    # ── 주문 실행 ────────────────────────────────────────────────────────

    def execute_order_directly(self, order: OrderRequest) -> dict:
        """주문 즉시 실행 (모의투자 자동승인 전용)"""
        order_log_id = getattr(order, '_order_log_id', None)
        try:
            result = self.client.place_order(
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
            )

            # DB 업데이트
            if order_log_id:
                update = {
                    'approval_status': 'auto',
                    'approved_at': datetime.datetime.now(datetime.timezone.utc),
                    'approved_by': 'auto_paper',
                }
                if result['success']:
                    update.update({
                        'execution_status': 'sent',
                        'kis_order_no': result.get('order_no', ''),
                    })
                else:
                    update.update({
                        'execution_status': 'failed',
                        'error_message': result.get('message', 'Unknown error'),
                    })
                OrderStorage.update_order_execution(order_log_id, update)

            status = 'SUCCESS' if result['success'] else 'FAILED'
            print(f"[KIS {status}] {order.side} {order.ticker} x{order.quantity} "
                  f"— {result.get('message', '')}", flush=True)
            return result

        except (TradingHaltedError, AuthenticationError) as e:
            print(f"[KIS BLOCKED] {e}", flush=True)
            if order_log_id:
                OrderStorage.update_order_execution(order_log_id, {
                    'execution_status': 'failed',
                    'error_message': str(e),
                })
            return {'success': False, 'order_no': '', 'message': str(e), 'raw': {}}

        except Exception as e:
            print(f"[KIS ERROR] {e}", flush=True)
            traceback.print_exc()
            if order_log_id:
                OrderStorage.update_order_execution(order_log_id, {
                    'execution_status': 'failed',
                    'error_message': str(e),
                })
            return {'success': False, 'order_no': '', 'message': str(e), 'raw': {}}

    # ── Telegram 승인 요청 ───────────────────────────────────────────────

    def request_telegram_approval(self, token: str, chat_id: str,
                                   orders: list) -> bool:
        """
        승인 요청 메시지 발송 + pending_approvals DB 저장

        Parameters
        ----------
        token : str — Telegram Bot Token
        chat_id : str — Telegram Chat ID
        orders : list[OrderRequest] — SafetyGuard 통과한 주문 목록
        """
        if not token or not chat_id:
            print("[KIS] Telegram token/chat_id missing, cannot request approval", flush=True)
            return False

        expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=30)
        approval_lines = []

        for order in orders:
            # 6자리 승인 코드 생성
            approval_code = secrets.token_hex(3).upper()  # 예: 'A3F82K'
            order_log_id = getattr(order, '_order_log_id', None)

            if order_log_id:
                OrderStorage.save_pending_approval(order_log_id, approval_code, expires_at)

            icon = '🟢' if order.side == 'buy' else '🔴'
            action = '매수' if order.side == 'buy' else '매도'
            approval_lines.append(
                f"{icon} {action}: {order.ticker} {order.quantity}주 "
                f"x ${order.price:,.2f} = ${order.estimated_amount:,.2f}"
            )
            approval_lines.append(f"  /approve {approval_code} | /reject {approval_code}")

        total_amount = sum(o.estimated_amount for o in orders)
        mode_label = '모의투자' if self.mode == 'paper' else '실거래'

        msg_lines = [
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"🔔 *주문 승인 요청* [{mode_label}]",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📊 포트폴리오: *{orders[0].portfolio_name}*" if orders else "",
            "",
        ]
        msg_lines.extend(approval_lines)
        msg_lines.extend([
            "",
            f"💰 총 주문금액: ${total_amount:,.2f}",
            f"⏰ {expires_at.strftime('%H:%M UTC')} 이후 자동 거부",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ])

        msg = '\n'.join(msg_lines)

        try:
            send_url = f"https://api.telegram.org/bot{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
            r = requests.post(send_url, json=payload, timeout=10)
            print(f"[KIS] Telegram approval request sent: {r.status_code}", flush=True)
            return r.status_code == 200
        except Exception as e:
            print(f"[KIS] Telegram approval send error: {e}", flush=True)
            return False

    # ── 승인 후 실행 ─────────────────────────────────────────────────────

    def execute_approved_order(self, approval_code: str) -> dict:
        """
        Telegram /approve 명령에서 호출.

        1. pending_approvals에서 주문 정보 로드
        2. 타임아웃 체크 (30분 초과 → 거부)
        3. 이미 처리됨 확인
        4. 안전가드 재검증
        5. KISClient.place_order() 실행
        6. 결과 DB 기록

        Returns
        -------
        dict : {'success': bool, 'message': str}
        """
        # 1. 승인 레코드 조회
        approval = OrderStorage.get_pending_approval(approval_code)
        if not approval:
            return {'success': False, 'message': f'승인 코드 "{approval_code}" 를 찾을 수 없습니다.'}

        # 2. 이미 처리됨 확인
        if approval['status'] != 'pending':
            return {'success': False, 'message': f'이미 처리된 주문입니다 (status={approval["status"]}).'}

        # 3. 타임아웃 체크
        now = datetime.datetime.now(datetime.timezone.utc)
        expires_at = approval['expires_at']
        # timezone-aware 비교를 위해 처리
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=datetime.timezone.utc)
        if now > expires_at:
            OrderStorage.update_pending_approval(approval_code, 'expired')
            OrderStorage.update_order_execution(approval['order_log_id'], {
                'approval_status': 'expired',
                'execution_status': 'cancelled',
            })
            return {'success': False, 'message': '승인 시간이 만료되었습니다 (30분 초과).'}

        # 4. 안전가드 재검증
        order = OrderRequest(
            account_id=approval['account_id'],
            ticker=approval['ticker'],
            side=approval['side'],
            quantity=approval['quantity'],
            price=approval['price'],
            order_type=approval['order_type'],
            signal_event_id=None,  # 재검증 시 중복 체크 스킵
            portfolio_name=approval.get('portfolio_name', ''),
            strategy_name=approval.get('strategy_name', ''),
            reason='approved',
            estimated_amount=approval['quantity'] * approval['price'],
            trading_mode=approval['trading_mode'],
        )

        guard = TradingSafetyGuard(self.mode, approval['account_id'])
        recheck = guard.validate(order)

        if not recheck.passed:
            OrderStorage.update_pending_approval(approval_code, 'rejected')
            OrderStorage.update_order_execution(approval['order_log_id'], {
                'approval_status': 'rejected',
                'execution_status': 'cancelled',
                'error_message': f'재검증 실패: {recheck.rejection_reason}',
            })
            return {'success': False,
                    'message': f'안전가드 재검증 실패: {recheck.rejection_reason}'}

        # 5. 주문 실행
        try:
            result = self.client.place_order(
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                price=order.price,
                order_type=order.order_type,
            )

            # 6. DB 업데이트
            OrderStorage.update_pending_approval(approval_code, 'approved')
            now_ts = datetime.datetime.now(datetime.timezone.utc)

            if result['success']:
                OrderStorage.update_order_execution(approval['order_log_id'], {
                    'approval_status': 'approved',
                    'approved_at': now_ts,
                    'approved_by': 'telegram',
                    'execution_status': 'sent',
                    'kis_order_no': result.get('order_no', ''),
                })
                action = '매수' if order.side == 'buy' else '매도'
                return {'success': True,
                        'message': f'{order.ticker} {order.quantity}주 {action} 주문 완료 '
                                   f'(주문번호: {result.get("order_no", "N/A")})'}
            else:
                OrderStorage.update_order_execution(approval['order_log_id'], {
                    'approval_status': 'approved',
                    'approved_at': now_ts,
                    'approved_by': 'telegram',
                    'execution_status': 'failed',
                    'error_message': result.get('message', 'Unknown error'),
                })
                return {'success': False,
                        'message': f'주문 실행 실패: {result.get("message", "Unknown error")}'}

        except (TradingHaltedError, AuthenticationError) as e:
            OrderStorage.update_pending_approval(approval_code, 'rejected')
            OrderStorage.update_order_execution(approval['order_log_id'], {
                'execution_status': 'failed',
                'error_message': str(e),
            })
            return {'success': False, 'message': str(e)}

    # ── 주문 거부 ────────────────────────────────────────────────────────

    @staticmethod
    def reject_order(approval_code: str) -> dict:
        """Telegram /reject 명령에서 호출"""
        approval = OrderStorage.get_pending_approval(approval_code)
        if not approval:
            return {'success': False, 'message': f'승인 코드 "{approval_code}" 를 찾을 수 없습니다.'}

        if approval['status'] != 'pending':
            return {'success': False, 'message': f'이미 처리된 주문입니다 (status={approval["status"]}).'}

        OrderStorage.update_pending_approval(approval_code, 'rejected')
        OrderStorage.update_order_execution(approval['order_log_id'], {
            'approval_status': 'rejected',
            'execution_status': 'cancelled',
        })
        return {'success': True,
                'message': f'{approval["ticker"]} {approval["quantity"]}주 주문이 거부되었습니다.'}
