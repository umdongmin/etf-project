"""
한국투자증권 KIS Open API 클라이언트

- 모의투자/실거래 환경 완전 분리
- OAuth 토큰 자동 발급/갱신 (Supabase 캐시)
- Fail-Closed: 토큰 없음, 환경 불일치, kill switch → 모든 API 차단
"""

import os
import time
import datetime
import hashlib
import secrets

import requests


# ── 예외 클래스 ──────────────────────────────────────────────────────────

class KISError(Exception):
    """KIS API 관련 기본 예외"""


class TradingHaltedError(KISError):
    """Kill switch가 활성화되어 거래가 중단된 상태"""


class AuthenticationError(KISError):
    """KIS 인증 실패 (토큰 발급/갱신 불가)"""


class ModeInconsistencyError(KISError):
    """코드 내 mode와 환경변수 TRADING_MODE 불일치"""


# ── KIS API 클라이언트 ───────────────────────────────────────────────────

class KISClient:
    """
    KIS REST API 클라이언트 (모의투자/실거래 완전 분리)

    Usage:
        client = KISClient('paper')
        balance = client.get_balance()
    """

    # 환경별 Base URL (하드코딩 — 실수로 변경 불가)
    _BASE_URLS = {
        'paper': 'https://openapivts.koreainvestment.com:29443',
        'live':  'https://openapi.koreainvestment.com:9443',
    }

    # 환경별 tr_id 매핑
    _TR_IDS = {
        'paper': {
            'buy':       'VTTT1002U',
            'sell':      'VTTT1006U',
            'balance':   'VTTS3012R',
            'price':     'HHDFS00000300',
            'ccnl':      'VTTS3035R',
        },
        'live': {
            'buy':       'JTTT1002U',
            'sell':      'JTTT1006U',
            'balance':   'TTTS3012R',
            'price':     'HHDFS00000300',
            'ccnl':      'TTTS3035R',
        },
    }

    # 재시도 설정
    _MAX_RETRIES = 3
    _RETRY_DELAYS = [1, 2, 4]  # 초
    _REQUEST_TIMEOUT = 15       # 초

    def __init__(self, mode: str = None):
        """
        Parameters
        ----------
        mode : str
            'paper' 또는 'live'. None이면 TRADING_MODE 환경변수에서 결정.

        Raises
        ------
        ModeInconsistencyError
            mode='live'인데 TRADING_MODE!='live'이면 발생
        ValueError
            유효하지 않은 mode 또는 필수 환경변수 누락
        """
        self._mode = self._resolve_and_validate_mode(mode)
        self._base_url = self._BASE_URLS[self._mode]
        self._tr_ids = self._TR_IDS[self._mode]
        self._app_key, self._app_secret, self._account_no = self._load_credentials()

    # ── 초기화 헬퍼 ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_and_validate_mode(mode: str = None) -> str:
        """mode 결정 + 환경변수 교차검증"""
        env_mode = os.environ.get('TRADING_MODE', 'disabled').lower().strip()

        # 유효하지 않은 환경변수 → disabled 강제
        if env_mode not in ('paper', 'live', 'disabled'):
            print(f"[KIS] Invalid TRADING_MODE='{env_mode}', forcing 'disabled'", flush=True)
            env_mode = 'disabled'

        if mode is None:
            mode = env_mode

        if mode == 'disabled':
            raise ValueError("TRADING_MODE is 'disabled'. KIS client cannot be created.")

        if mode not in ('paper', 'live'):
            raise ValueError(f"Invalid mode: '{mode}'. Must be 'paper' or 'live'.")

        # 핵심 안전장치: live 모드 교차검증
        if mode == 'live' and env_mode != 'live':
            raise ModeInconsistencyError(
                f"mode='live' requested but TRADING_MODE='{env_mode}'. "
                f"실거래 모드는 환경변수 TRADING_MODE=live일 때만 허용됩니다."
            )

        return mode

    def _load_credentials(self) -> tuple:
        """환경변수에서 KIS API 자격증명 로드"""
        prefix = 'KIS_PAPER' if self._mode == 'paper' else 'KIS_LIVE'

        app_key = os.environ.get(f'{prefix}_APP_KEY', '').strip()
        app_secret = os.environ.get(f'{prefix}_APP_SECRET', '').strip()
        account_no = os.environ.get(f'{prefix}_ACCOUNT_NO', '').strip()

        if not app_key or not app_secret or not account_no:
            raise ValueError(
                f"KIS {self._mode} 자격증명 누락. "
                f"환경변수 {prefix}_APP_KEY, {prefix}_APP_SECRET, {prefix}_ACCOUNT_NO를 설정하세요."
            )

        # 계좌번호 형식 검증 (8자리-2자리)
        if '-' not in account_no or len(account_no.replace('-', '')) != 10:
            raise ValueError(
                f"계좌번호 형식 오류: '{account_no}'. 올바른 형식: '12345678-01'"
            )

        return app_key, app_secret, account_no

    # ── 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def account_no(self) -> str:
        return self._account_no

    @property
    def account_prefix(self) -> str:
        """계좌번호 앞 8자리"""
        return self._account_no.split('-')[0]

    @property
    def account_suffix(self) -> str:
        """계좌번호 뒤 2자리"""
        return self._account_no.split('-')[1]

    # ── 토큰 관리 ────────────────────────────────────────────────────────

    def _get_token(self) -> str:
        """
        유효한 OAuth 토큰 반환.
        1. DB 캐시 확인 → 유효하면 반환
        2. 만료 1시간 이내 또는 없으면 → 재발급 → DB 저장
        3. 실패 시 None 반환 (호출 측에서 차단)
        """
        # 1. DB에서 캐시된 토큰 조회
        cached = self._load_token_from_db()
        if cached:
            token, expires_at = cached
            now = datetime.datetime.now(datetime.timezone.utc)
            # 만료 1시간 전까지는 유효
            if expires_at > now + datetime.timedelta(hours=1):
                return token

        # 2. 토큰 재발급
        new_token, expires_at = self._issue_token()
        if new_token:
            self._save_token_to_db(new_token, expires_at)
            return new_token

        # 3. 재발급 실패 → 캐시 토큰이 아직 유효하면 사용
        if cached:
            token, expires_at = cached
            now = datetime.datetime.now(datetime.timezone.utc)
            if expires_at > now:
                print("[KIS] Token refresh failed, using cached token", flush=True)
                return token

        return None

    def _issue_token(self) -> tuple:
        """KIS OAuth 토큰 발급 API 호출"""
        url = f"{self._base_url}/oauth2/tokenP"
        body = {
            'grant_type': 'client_credentials',
            'appkey': self._app_key,
            'appsecret': self._app_secret,
        }
        try:
            resp = requests.post(url, json=body, timeout=self._REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                token = data.get('access_token')
                # KIS 토큰 유효기간: 약 24시간
                expires_str = data.get('access_token_token_expired', '')
                if expires_str:
                    try:
                        # KIS 형식: "2025-03-22 12:00:00"
                        expires_at = datetime.datetime.strptime(
                            expires_str, '%Y-%m-%d %H:%M:%S'
                        ).replace(tzinfo=datetime.timezone.utc)
                    except ValueError:
                        # 파싱 실패 시 23시간 후로 설정
                        expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=23)
                else:
                    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=23)

                if token:
                    print(f"[KIS] Token issued for {self._mode}, expires: {expires_at.isoformat()}", flush=True)
                    return token, expires_at

            print(f"[KIS] Token issue failed: HTTP {resp.status_code} - {resp.text[:200]}", flush=True)
        except requests.RequestException as e:
            print(f"[KIS] Token issue error: {e}", flush=True)

        return None, None

    def _load_token_from_db(self) -> tuple:
        """DB에서 캐시된 토큰 로드. 반환: (token, expires_at) 또는 None"""
        try:
            from core.storage import OrderStorage
            return OrderStorage.load_kis_token(self._mode)
        except Exception as e:
            print(f"[KIS] Token DB load error: {e}", flush=True)
            return None

    def _save_token_to_db(self, token: str, expires_at: datetime.datetime):
        """DB에 토큰 저장"""
        try:
            from core.storage import OrderStorage
            OrderStorage.save_kis_token(self._mode, token, expires_at)
        except Exception as e:
            print(f"[KIS] Token DB save error: {e}", flush=True)

    # ── Kill Switch 확인 ─────────────────────────────────────────────────

    @staticmethod
    def is_kill_switch_active() -> bool:
        """환경변수 기반 kill switch 확인"""
        return os.environ.get('TRADING_KILL_SWITCH', 'false').lower().strip() == 'true'

    # ── 공통 API 요청 ────────────────────────────────────────────────────

    def _api_request(self, method: str, path: str, body: dict = None,
                     params: dict = None, tr_id: str = None) -> dict:
        """
        KIS API 공통 요청 메서드.

        Raises
        ------
        TradingHaltedError : kill switch 활성
        AuthenticationError : 토큰 없음

        Returns
        -------
        dict : API 응답 데이터 (성공 시) 또는 None (실패 시)
        """
        # 1. Kill switch 확인
        if self.is_kill_switch_active():
            raise TradingHaltedError("Kill switch is active — all API calls blocked")

        # 2. 토큰 확보
        token = self._get_token()
        if not token:
            raise AuthenticationError("No valid KIS token available")

        # 3. 헤더 구성
        headers = {
            'authorization': f'Bearer {token}',
            'appkey': self._app_key,
            'appsecret': self._app_secret,
            'Content-Type': 'application/json; charset=utf-8',
        }
        if tr_id:
            headers['tr_id'] = tr_id

        # 4. 재시도 로직
        url = f"{self._base_url}{path}"
        last_error = None

        for attempt in range(self._MAX_RETRIES):
            try:
                resp = requests.request(
                    method, url, headers=headers,
                    json=body, params=params,
                    timeout=self._REQUEST_TIMEOUT
                )

                if resp.status_code == 200:
                    data = resp.json()
                    rt_cd = data.get('rt_cd', '')
                    if rt_cd == '0':
                        return data
                    else:
                        msg = data.get('msg1', 'Unknown error')
                        print(f"[KIS API Error] rt_cd={rt_cd}, msg={msg}", flush=True)
                        # 비즈니스 에러는 재시도하지 않음
                        return None

                elif resp.status_code == 429:
                    # Rate limit → 재시도
                    delay = self._RETRY_DELAYS[min(attempt, len(self._RETRY_DELAYS) - 1)]
                    print(f"[KIS] Rate limited, retry in {delay}s (attempt {attempt + 1}/{self._MAX_RETRIES})", flush=True)
                    time.sleep(delay)
                    continue

                else:
                    print(f"[KIS HTTP {resp.status_code}] {resp.text[:200]}", flush=True)
                    return None

            except requests.Timeout:
                delay = self._RETRY_DELAYS[min(attempt, len(self._RETRY_DELAYS) - 1)]
                print(f"[KIS Timeout] attempt {attempt + 1}/{self._MAX_RETRIES}, retry in {delay}s", flush=True)
                last_error = "Timeout"
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(delay)

            except requests.ConnectionError as e:
                print(f"[KIS Connection Error] {e}", flush=True)
                last_error = str(e)
                return None

        print(f"[KIS] All {self._MAX_RETRIES} attempts failed. Last error: {last_error}", flush=True)
        return None

    # ── 조회 API ─────────────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """
        해외주식 잔고 조회.

        Returns
        -------
        dict : {ticker: {qty, avg_price, current_price, eval_amount, pnl, ...}} 또는 빈 dict
        """
        params = {
            'CANO': self.account_prefix,
            'ACNT_PRDT_CD': self.account_suffix,
            'OVRS_EXCG_CD': 'NASD',
            'TR_CRCY_CD': 'USD',
            'CTX_AREA_FK200': '',
            'CTX_AREA_NK200': '',
        }
        data = self._api_request(
            'GET',
            '/uapi/overseas-stock/v1/trading/inquire-balance',
            params=params,
            tr_id=self._tr_ids['balance']
        )
        if not data:
            return {}

        result = {}
        items = data.get('output1', [])
        for item in items:
            ticker = item.get('ovrs_pdno', '').strip()
            qty = int(float(item.get('ovrs_cblc_qty', '0') or '0'))
            if ticker and qty > 0:
                result[ticker] = {
                    'qty': qty,
                    'avg_price': float(item.get('pchs_avg_pric', '0') or '0'),
                    'current_price': float(item.get('now_pric2', '0') or '0'),
                    'eval_amount': float(item.get('ovrs_stck_evlu_amt', '0') or '0'),
                    'pnl': float(item.get('frcr_evlu_pfls_amt', '0') or '0'),
                    'pnl_rate': float(item.get('evlu_pfls_rt', '0') or '0'),
                    'currency': item.get('tr_crcy_cd', 'USD'),
                }
        return result

    def get_current_price(self, ticker: str) -> dict:
        """
        해외주식 현재가 조회.

        Returns
        -------
        dict : {price, change, change_rate, volume, ...} 또는 None
        """
        params = {
            'AUTH': '',
            'EXCD': 'NAS',  # NASDAQ
            'SYMB': ticker.upper(),
        }
        data = self._api_request(
            'GET',
            '/uapi/overseas-price/v1/quotations/price',
            params=params,
            tr_id=self._tr_ids['price']
        )
        if not data:
            return None

        output = data.get('output', {})
        price = float(output.get('last', '0') or '0')
        if price <= 0:
            return None

        return {
            'price': price,
            'change': float(output.get('diff', '0') or '0'),
            'change_rate': float(output.get('rate', '0') or '0'),
            'volume': int(float(output.get('tvol', '0') or '0')),
            'high': float(output.get('high', '0') or '0'),
            'low': float(output.get('low', '0') or '0'),
            'open': float(output.get('open', '0') or '0'),
        }

    def get_order_history(self, start_date: str = None, end_date: str = None) -> list:
        """
        해외주식 체결내역 조회.

        Parameters
        ----------
        start_date, end_date : str
            YYYYMMDD 형식. None이면 오늘.

        Returns
        -------
        list[dict] : 체결 내역 리스트 또는 빈 list
        """
        today = datetime.date.today().strftime('%Y%m%d')
        params = {
            'CANO': self.account_prefix,
            'ACNT_PRDT_CD': self.account_suffix,
            'PDNO': '',
            'ORD_STRT_DT': start_date or today,
            'ORD_END_DT': end_date or today,
            'SLL_BUY_DVSN': '00',  # 전체
            'CCLD_NCCS_DVSN': '00',  # 전체
            'OVRS_EXCG_CD': 'NASD',
            'SORT_SQN': 'DS',  # 최신순
            'ORD_GNO_BRNO': '',
            'ODNO': '',
            'CTX_AREA_FK200': '',
            'CTX_AREA_NK200': '',
        }
        data = self._api_request(
            'GET',
            '/uapi/overseas-stock/v1/trading/inquire-ccnl',
            params=params,
            tr_id=self._tr_ids['ccnl']
        )
        if not data:
            return []

        result = []
        items = data.get('output', [])
        for item in items:
            order_no = item.get('odno', '').strip()
            if not order_no:
                continue
            result.append({
                'order_no': order_no,
                'ticker': item.get('pdno', '').strip(),
                'side': 'buy' if item.get('sll_buy_dvsn_cd') == '02' else 'sell',
                'order_qty': int(float(item.get('ft_ord_qty', '0') or '0')),
                'filled_qty': int(float(item.get('ft_ccld_qty', '0') or '0')),
                'order_price': float(item.get('ft_ord_unpr3', '0') or '0'),
                'filled_price': float(item.get('ft_ccld_unpr3', '0') or '0'),
                'filled_amount': float(item.get('ft_ccld_amt3', '0') or '0'),
                'order_time': item.get('ord_tmd', ''),
                'status': item.get('ccld_yn', ''),  # Y: 체결, N: 미체결
            })
        return result

    # ── 주문 API ─────────────────────────────────────────────────────────

    def place_order(self, ticker: str, side: str, quantity: int,
                    price: float = 0, order_type: str = 'market') -> dict:
        """
        해외주식 매수/매도 주문.

        Parameters
        ----------
        ticker : str — 종목코드 (예: 'TQQQ')
        side : str — 'buy' 또는 'sell'
        quantity : int — 주문 수량 (양의 정수)
        price : float — 주문 가격 (시장가면 0)
        order_type : str — 'market' 또는 'limit'

        Returns
        -------
        dict : {'success': bool, 'order_no': str, 'message': str, 'raw': dict}
        """
        if side not in ('buy', 'sell'):
            return {'success': False, 'order_no': '', 'message': f"Invalid side: {side}", 'raw': {}}

        if quantity <= 0:
            return {'success': False, 'order_no': '', 'message': f"Invalid quantity: {quantity}", 'raw': {}}

        tr_id = self._tr_ids[side]

        # 주문 유형: 00=지정가, 31=시장가 (KIS 해외주식)
        ord_dvsn = '00' if order_type == 'limit' else '31'

        body = {
            'CANO': self.account_prefix,
            'ACNT_PRDT_CD': self.account_suffix,
            'OVRS_EXCG_CD': 'NASD',
            'PDNO': ticker.upper(),
            'ORD_QTY': str(quantity),
            'OVRS_ORD_UNPR': f"{price:.2f}" if order_type == 'limit' else '0',
            'ORD_SVR_DVSN_CD': '0',
            'ORD_DVSN': ord_dvsn,
        }

        print(f"[KIS] Placing {side} order: {ticker} x{quantity} @ "
              f"{'market' if order_type == 'market' else f'${price:.2f}'} "
              f"(mode={self._mode})", flush=True)

        data = self._api_request(
            'POST',
            '/uapi/overseas-stock/v1/trading/order',
            body=body,
            tr_id=tr_id
        )

        if data:
            output = data.get('output', {})
            order_no = output.get('ODNO', output.get('odno', ''))
            return {
                'success': True,
                'order_no': order_no,
                'message': data.get('msg1', 'Order placed'),
                'raw': data,
            }

        return {
            'success': False,
            'order_no': '',
            'message': 'KIS API call failed',
            'raw': {},
        }

    # ── 유틸리티 ─────────────────────────────────────────────────────────

    def __repr__(self):
        masked_acct = f"{self.account_prefix[:4]}****-{self.account_suffix}"
        return f"KISClient(mode={self._mode}, account={masked_acct})"
