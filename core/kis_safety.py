"""
TradingSafetyGuard — 다중 레이어 주문 안전 검증

모든 주문은 반드시 이 모듈의 validate()를 통과해야 실행 가능.
하나라도 실패하면 주문 거부 (Fail-Closed 원칙).

9개 안전 체크:
  1. kill_switch      — 환경변수 + DB 긴급 중단
  2. mode_consistency — 코드/환경변수 모드 교차검증
  3. trading_hours    — 미국 정규장 시간 확인
  4. allowed_ticker   — 허용 종목 목록 확인
  5. max_single_order — 1회 주문 금액 상한
  6. daily_order_limit — 일일 주문 총액 상한
  7. daily_loss_limit  — 당일 손실 한도 (매수만 차단)
  8. duplicate_order   — 동일 signal_event + ticker 중복 방지
  9. cooldown          — 동일 종목 쿨다운 (최근 N분)
"""

import os
import datetime
from dataclasses import dataclass, field
from typing import List, Optional


# ── 데이터 클래스 ────────────────────────────────────────────────────────

@dataclass
class OrderRequest:
    """주문 요청 데이터 (불변 객체로 사용)"""
    account_id: int
    ticker: str
    side: str               # 'buy' | 'sell'
    quantity: int
    price: float
    order_type: str          # 'market' | 'limit'
    signal_event_id: Optional[int]
    portfolio_name: str
    strategy_name: str
    reason: str              # 'stage_change' | 'rebalance'
    estimated_amount: float  # quantity * price
    trading_mode: str        # 'paper' | 'live'
    created_at: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )


@dataclass
class SafetyCheckResult:
    """안전 검증 결과"""
    passed: bool
    checks: List[dict]       # [{'name': str, 'passed': bool, 'detail': str}]
    rejection_reason: str    # passed=False일 때 첫 번째 실패 사유


# ── 안전가드 ─────────────────────────────────────────────────────────────

class TradingSafetyGuard:
    """다중 레이어 안전 검증 — 모든 체크 통과해야 주문 허용"""

    # 모드별 기본 한도
    _DEFAULT_LIMITS = {
        'paper': {
            'max_single_order_usd': 10000,
            'daily_order_limit_usd': 50000,
            'daily_loss_limit_usd': 10000,
            'position_limit_pct': 1.0,
            'cooldown_minutes': 5,
            'allowed_tickers': ['TQQQ', 'QQQ', 'QLD', 'TLT', 'TMF', 'BIL', 'IEF'],
        },
        'live': {
            'max_single_order_usd': 5000,
            'daily_order_limit_usd': 20000,
            'daily_loss_limit_usd': 5000,
            'position_limit_pct': 0.50,
            'cooldown_minutes': 30,
            'allowed_tickers': ['TQQQ', 'QQQ', 'QLD', 'TLT', 'TMF', 'BIL', 'IEF'],
        },
    }

    def __init__(self, mode: str, account_id: int):
        self.mode = mode
        self.account_id = account_id
        self._config = self._load_config()

    def _load_config(self) -> dict:
        """DB에서 거래 설정 로드, 없으면 기본값 사용"""
        from core.storage import OrderStorage

        db_config = OrderStorage.load_trading_config(self.account_id)
        defaults = self._DEFAULT_LIMITS.get(self.mode, self._DEFAULT_LIMITS['live'])

        if db_config:
            # DB 설정이 기본값보다 높으면 경고
            for key in ('max_single_order_usd', 'daily_order_limit_usd'):
                if db_config.get(key, 0) > defaults[key]:
                    print(f"[SafetyGuard] WARNING: DB {key}=${db_config[key]:,.0f} "
                          f"> default ${defaults[key]:,.0f}", flush=True)
            return db_config

        return defaults

    # ── 메인 검증 ────────────────────────────────────────────────────────

    def validate(self, order: OrderRequest) -> SafetyCheckResult:
        """전체 안전 검증 실행 — 하나라도 실패하면 거부"""
        checks = [
            self._check_kill_switch(),
            self._check_mode_consistency(order),
            self._check_trading_hours(),
            self._check_allowed_ticker(order),
            self._check_max_single_order(order),
            self._check_daily_order_limit(order),
            self._check_daily_loss_limit(order),
            self._check_duplicate_order(order),
            self._check_cooldown(order),
        ]

        all_passed = all(c['passed'] for c in checks)
        rejection = next((c['detail'] for c in checks if not c['passed']), '')

        return SafetyCheckResult(
            passed=all_passed,
            checks=checks,
            rejection_reason=rejection,
        )

    # ── 개별 체크 ────────────────────────────────────────────────────────

    def _check_kill_switch(self) -> dict:
        """1. 환경변수 + DB kill switch 확인"""
        name = 'kill_switch'

        # 환경변수 kill switch
        env_kill = os.environ.get('TRADING_KILL_SWITCH', 'false').lower().strip() == 'true'
        if env_kill:
            return {'name': name, 'passed': False,
                    'detail': 'TRADING_KILL_SWITCH 환경변수가 활성화됨'}

        # DB kill switch
        is_halted = self._config.get('is_halted', True)  # 기본값 True = 안전
        if is_halted:
            return {'name': name, 'passed': False,
                    'detail': f'trading_config.is_halted=TRUE (account_id={self.account_id})'}

        return {'name': name, 'passed': True, 'detail': 'Kill switch 비활성'}

    def _check_mode_consistency(self, order: OrderRequest) -> dict:
        """2. 코드 내 mode와 환경변수 교차검증"""
        name = 'mode_consistency'

        env_mode = os.environ.get('TRADING_MODE', 'disabled').lower().strip()
        if order.trading_mode != env_mode:
            return {'name': name, 'passed': False,
                    'detail': f'주문 mode={order.trading_mode} != TRADING_MODE={env_mode}'}

        if self.mode != order.trading_mode:
            return {'name': name, 'passed': False,
                    'detail': f'Guard mode={self.mode} != 주문 mode={order.trading_mode}'}

        return {'name': name, 'passed': True, 'detail': f'mode={self.mode} 일치'}

    def _check_trading_hours(self) -> dict:
        """3. 미국 정규장 시간 확인 (EST 09:30~16:00)"""
        name = 'trading_hours'

        try:
            # UTC → EST 변환 (EST = UTC-5, EDT = UTC-4)
            now_utc = datetime.datetime.now(datetime.timezone.utc)

            # 간단한 DST 판단: 3월 둘째 일요일 ~ 11월 첫째 일요일
            year = now_utc.year
            # 3월 둘째 일요일
            mar1 = datetime.date(year, 3, 1)
            dst_start = mar1 + datetime.timedelta(days=(6 - mar1.weekday()) % 7 + 7)
            # 11월 첫째 일요일
            nov1 = datetime.date(year, 11, 1)
            dst_end = nov1 + datetime.timedelta(days=(6 - nov1.weekday()) % 7)

            is_dst = dst_start <= now_utc.date() < dst_end
            est_offset = datetime.timedelta(hours=-4 if is_dst else -5)
            now_est = now_utc + est_offset

            # 주말 체크
            if now_est.weekday() >= 5:
                return {'name': name, 'passed': False,
                        'detail': f'주말 ({now_est.strftime("%A")}) — 장 휴무'}

            # 정규장 시간 체크 (09:30 ~ 16:00 EST)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)

            if now_est < market_open or now_est > market_close:
                return {'name': name, 'passed': False,
                        'detail': f'장외 시간 (현재 EST {now_est.strftime("%H:%M")}, '
                                  f'정규장 09:30~16:00)'}

            return {'name': name, 'passed': True,
                    'detail': f'정규장 시간 내 (EST {now_est.strftime("%H:%M")})'}

        except Exception as e:
            # 시간 판단 실패 → Fail-Closed
            return {'name': name, 'passed': False,
                    'detail': f'시간 확인 실패: {e}'}

    def _check_allowed_ticker(self, order: OrderRequest) -> dict:
        """4. 허용 종목 목록 확인"""
        name = 'allowed_ticker'

        allowed = self._config.get('allowed_tickers', [])
        if not allowed:
            # 허용 목록이 비어있으면 기본값 사용
            allowed = self._DEFAULT_LIMITS.get(self.mode, {}).get('allowed_tickers', [])

        ticker = order.ticker.upper()
        if ticker not in [t.upper() for t in allowed]:
            return {'name': name, 'passed': False,
                    'detail': f'{ticker}은 허용 종목 목록에 없음 ({allowed})'}

        return {'name': name, 'passed': True, 'detail': f'{ticker} 허용됨'}

    def _check_max_single_order(self, order: OrderRequest) -> dict:
        """5. 1회 주문 금액 상한"""
        name = 'max_single_order'

        limit = self._config.get('max_single_order_usd',
                                  self._DEFAULT_LIMITS[self.mode]['max_single_order_usd'])

        if order.estimated_amount > limit:
            return {'name': name, 'passed': False,
                    'detail': f'주문금액 ${order.estimated_amount:,.0f} > 한도 ${limit:,.0f}'}

        return {'name': name, 'passed': True,
                'detail': f'${order.estimated_amount:,.0f} <= ${limit:,.0f}'}

    def _check_daily_order_limit(self, order: OrderRequest) -> dict:
        """6. 일일 주문 총액 상한"""
        name = 'daily_order_limit'

        from core.storage import OrderStorage
        daily_total = OrderStorage.get_daily_order_total(self.account_id, self.mode)
        limit = self._config.get('daily_order_limit_usd',
                                  self._DEFAULT_LIMITS[self.mode]['daily_order_limit_usd'])

        projected = daily_total + order.estimated_amount
        if projected > limit:
            return {'name': name, 'passed': False,
                    'detail': f'일일 총액 ${projected:,.0f} (기존 ${daily_total:,.0f} + '
                              f'신규 ${order.estimated_amount:,.0f}) > 한도 ${limit:,.0f}'}

        return {'name': name, 'passed': True,
                'detail': f'일일 누적 ${projected:,.0f} <= ${limit:,.0f}'}

    def _check_daily_loss_limit(self, order: OrderRequest) -> dict:
        """7. 당일 손실 한도 초과 시 매수 차단"""
        name = 'daily_loss_limit'

        # 매도는 손실 한도와 무관 (포지션 청산은 항상 허용)
        if order.side == 'sell':
            return {'name': name, 'passed': True, 'detail': '매도 주문 — 손실 한도 체크 불필요'}

        # 현재 구현에서는 KIS 잔고에서 실시간 손실을 조회해야 하므로
        # Phase 4에서 KIS 잔고 연동 후 활성화. 현재는 통과 처리.
        return {'name': name, 'passed': True, 'detail': '손실 한도 체크 (Phase 4에서 활성화 예정)'}

    def _check_duplicate_order(self, order: OrderRequest) -> dict:
        """8. 동일 signal_event + ticker 중복 방지 (멱등성)"""
        name = 'duplicate_order'

        if order.signal_event_id is None:
            return {'name': name, 'passed': True, 'detail': 'signal_event_id 없음 — 스킵'}

        from core.storage import OrderStorage
        is_dup = OrderStorage.check_duplicate_order(order.signal_event_id, order.ticker)
        if is_dup:
            return {'name': name, 'passed': False,
                    'detail': f'중복 주문: signal_event_id={order.signal_event_id}, '
                              f'ticker={order.ticker}'}

        return {'name': name, 'passed': True, 'detail': '중복 없음'}

    def _check_cooldown(self, order: OrderRequest) -> dict:
        """9. 동일 종목 쿨다운 (최근 N분 내 주문 이력)"""
        name = 'cooldown'

        cooldown_min = self._config.get('cooldown_minutes',
                                         self._DEFAULT_LIMITS[self.mode]['cooldown_minutes'])

        from core.storage import OrderStorage
        recent = OrderStorage.get_recent_order_for_ticker(
            self.account_id, order.ticker, cooldown_min
        )
        if recent:
            return {'name': name, 'passed': False,
                    'detail': f'{order.ticker} 쿨다운 중 — '
                              f'최근 주문: {recent["created_at"]} ({cooldown_min}분 제한)'}

        return {'name': name, 'passed': True,
                'detail': f'{order.ticker} 쿨다운 없음 ({cooldown_min}분)'}
