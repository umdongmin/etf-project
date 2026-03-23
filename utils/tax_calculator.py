"""
해외주식 양도소득세 계산 모듈 (한국 거주자 기준)

과세 규칙:
  - 과세 단위  : 연간 (1월 1일 ~ 12월 31일)
  - 기본공제   : 연간 250만원 (USD 환산은 호출 시점 환율 사용)
  - 세율       : 기본 22% (양도세 20% + 지방소득세 2%)
  - 과세표준   : 연간 실현손익 합계 - 기본공제
  - 납부세액   : max(과세표준, 0) × 세율
  - 결손       : 해당 연도 손실은 이월공제 없음 (단순 0 처리)

참고: 본 계산은 시뮬레이션용 추정치이며 실제 세금과 다를 수 있습니다.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field


# 연간 기본공제 (원화)
BASIC_DEDUCTION_KRW: float = 2_500_000.0


@dataclass
class YearlyTaxResult:
    year: int
    realized_gain_usd: float      # 연간 실현손익 (USD)
    realized_gain_krw: float      # 연간 실현손익 (KRW, 환율 적용)
    deduction_krw: float          # 적용된 기본공제 (KRW)
    taxable_krw: float            # 과세표준 (KRW)
    tax_krw: float                # 납부 세액 (KRW)
    tax_usd: float                # 납부 세액 (USD, 환율 역산)
    after_tax_gain_krw: float     # 세후 실현손익 (KRW)
    trade_count: int              # 해당 연도 체결 건수
    tax_rate: float               # 적용 세율


@dataclass
class TaxSummary:
    yearly: list[YearlyTaxResult] = field(default_factory=list)
    total_realized_gain_usd: float = 0.0
    total_realized_gain_krw: float = 0.0
    total_tax_krw: float = 0.0
    total_tax_usd: float = 0.0
    total_after_tax_gain_krw: float = 0.0
    effective_tax_rate: float = 0.0    # 실효세율 (세액 / 실현손익)


def calculate_tax(
    closed_trades: list[dict] | pd.DataFrame,
    exchange_rate: float,
    tax_rate: float = 0.22,
    basic_deduction_krw: float = BASIC_DEDUCTION_KRW,
) -> TaxSummary:
    """
    closed_trades 기반 연도별 양도세 계산.

    Parameters
    ----------
    closed_trades   : engine이 반환하는 거래 목록 (list[dict] 또는 DataFrame)
                      필수 컬럼: Exit_Date, Net_Gain
    exchange_rate   : KRW/USD 환율 (투자 시점 기준 단일 환율 사용)
    tax_rate        : 양도세율 (기본 0.22 = 22%)
    basic_deduction_krw : 연간 기본공제 (기본 250만원)

    Returns
    -------
    TaxSummary
    """
    if closed_trades is None:
        return TaxSummary()

    # DataFrame 통일
    if isinstance(closed_trades, list):
        if not closed_trades:
            return TaxSummary()
        df = pd.DataFrame(closed_trades)
    else:
        df = closed_trades.copy()

    if df.empty or "Net_Gain" not in df.columns:
        return TaxSummary()

    # Exit_Date → datetime
    date_col = "Exit_Date" if "Exit_Date" in df.columns else df.columns[0]
    df["_exit_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_exit_dt"])
    df["_year"] = df["_exit_dt"].dt.year

    yearly_results: list[YearlyTaxResult] = []

    for year, grp in df.groupby("_year"):
        gain_usd   = float(grp["Net_Gain"].sum())
        gain_krw   = gain_usd * exchange_rate
        count      = len(grp)

        # 기본공제: 수익이 있을 때만 공제 적용 (손실 연도는 0)
        deduction  = min(basic_deduction_krw, gain_krw) if gain_krw > 0 else 0.0
        taxable    = max(gain_krw - deduction, 0.0)
        tax_krw    = taxable * tax_rate
        tax_usd    = tax_krw / exchange_rate if exchange_rate > 0 else 0.0
        after_tax  = gain_krw - tax_krw

        yearly_results.append(YearlyTaxResult(
            year=int(year),
            realized_gain_usd=gain_usd,
            realized_gain_krw=gain_krw,
            deduction_krw=deduction,
            taxable_krw=taxable,
            tax_krw=tax_krw,
            tax_usd=tax_usd,
            after_tax_gain_krw=after_tax,
            trade_count=count,
            tax_rate=tax_rate,
        ))

    # 미실현 포지션은 closed_trades에 없으므로 별도 처리 불필요 (호출자가 처리)

    # 합계
    total_gain_usd     = sum(r.realized_gain_usd for r in yearly_results)
    total_gain_krw     = sum(r.realized_gain_krw for r in yearly_results)
    total_tax_krw      = sum(r.tax_krw for r in yearly_results)
    total_tax_usd      = sum(r.tax_usd for r in yearly_results)
    total_after_krw    = sum(r.after_tax_gain_krw for r in yearly_results)
    effective_rate     = total_tax_krw / total_gain_krw if total_gain_krw > 0 else 0.0

    return TaxSummary(
        yearly=yearly_results,
        total_realized_gain_usd=total_gain_usd,
        total_realized_gain_krw=total_gain_krw,
        total_tax_krw=total_tax_krw,
        total_tax_usd=total_tax_usd,
        total_after_tax_gain_krw=total_after_krw,
        effective_tax_rate=effective_rate,
    )


def estimate_unrealized_tax(
    current_value_usd: float,
    cost_basis_usd: float,
    exchange_rate: float,
    tax_rate: float = 0.22,
    basic_deduction_krw: float = BASIC_DEDUCTION_KRW,
) -> dict:
    """
    현재 보유 포지션의 미실현손익에 대한 예상 세금 추정.
    (실제 납부 대상 아님 — 참고용)
    """
    gain_usd   = current_value_usd - cost_basis_usd
    gain_krw   = gain_usd * exchange_rate
    deduction  = min(basic_deduction_krw, gain_krw) if gain_krw > 0 else 0.0
    taxable    = max(gain_krw - deduction, 0.0)
    tax_krw    = taxable * tax_rate
    tax_usd    = tax_krw / exchange_rate if exchange_rate > 0 else 0.0

    return {
        "unrealized_gain_usd": gain_usd,
        "unrealized_gain_krw": gain_krw,
        "estimated_tax_krw":   tax_krw,
        "estimated_tax_usd":   tax_usd,
        "after_tax_gain_krw":  gain_krw - tax_krw,
    }
