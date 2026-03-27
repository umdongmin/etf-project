import datetime
import pandas as pd
import numpy as np


def calculate_xirr(cashflows: list[dict]) -> float | None:
    """연속 현금흐름 기반 내부수익률(XIRR) 계산 — Newton-Raphson, scipy 불필요.

    Args:
        cashflows: [{'date': datetime.date | pd.Timestamp, 'amount': float}, ...]
                   음수 = 투자(outflow), 양수 = 회수(inflow / 최종 가치)

    Returns:
        연간 내부수익률 (decimal, e.g. 0.15 = 15%). 계산 불가 시 None.
    """
    if not cashflows or len(cashflows) < 2:
        return None

    amounts = [cf['amount'] for cf in cashflows]
    # 부호 변환 확인 — XIRR은 음수/양수가 모두 있어야 수렴
    has_neg = any(a < 0 for a in amounts)
    has_pos = any(a > 0 for a in amounts)
    if not (has_neg and has_pos):
        return None

    # 기준일 = 첫 번째 현금흐름 날짜
    base_date = cashflows[0]['date']
    if isinstance(base_date, pd.Timestamp):
        base_date = base_date.date()

    def _year_fraction(d) -> float:
        if isinstance(d, pd.Timestamp):
            d = d.date()
        return (d - base_date).days / 365.25

    def _npv(rate: float) -> float:
        return sum(
            cf['amount'] / ((1 + rate) ** _year_fraction(cf['date']))
            for cf in cashflows
        )

    def _dnpv(rate: float) -> float:
        """NPV의 rate에 대한 1차 도함수"""
        result = 0.0
        for cf in cashflows:
            t = _year_fraction(cf['date'])
            if t == 0:
                continue
            result -= t * cf['amount'] / ((1 + rate) ** (t + 1))
        return result

    # Newton-Raphson 반복
    rate = 0.1  # 초기 추정값 10%
    for _ in range(100):
        npv = _npv(rate)
        dnpv = _dnpv(rate)
        if abs(dnpv) < 1e-12:
            break
        new_rate = rate - npv / dnpv
        if abs(new_rate - rate) < 1e-7:
            return new_rate
        rate = new_rate
        if rate <= -1:  # 발산 방지
            return None

    # 최종 수렴 확인
    if abs(_npv(rate)) < 1e-3:
        return rate
    return None


def calculate_metrics(history):
    """성과 지표 계산 함수 (MDD, CAGR 등)"""
    if history.empty:
        return {'Final Value': 0, 'Cumulative Return': 0, 'CAGR': 0, 'MDD': 0}
    
    df = history.copy()
    if 'PortfolioValue' not in df.columns and 'Value' in df.columns:
        df.rename(columns={'Value': 'PortfolioValue'}, inplace=True)

    # DatetimeIndex 보장 — 정수/문자열 인덱스 방어
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index('Date')
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

    df['Daily_Return'] = df['PortfolioValue'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1

    # CAGR 계산
    days = (df.index[-1] - df.index[0]).days
    if days <= 0: return {'Final Value': df['PortfolioValue'].iloc[-1], 'Cumulative Return': 0, 'CAGR': 0, 'MDD': 0}
    
    years = days / 365.25
    final_value = df['PortfolioValue'].iloc[-1]
    initial_value = df.get('PortfolioValue', pd.Series([10000.0])).iloc[0]
    if initial_value == 0: initial_value = 10000.0
    
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    # MDD 및 UI(궤양 지수) 계산
    df['Rolling_Max'] = df['PortfolioValue'].cummax()
    df['Drawdown'] = df['PortfolioValue'] / df['Rolling_Max'] - 1
    mdd = df['Drawdown'].min()
    # UI = sqrt(mean(Drawdown^2)) * 100
    ui = np.sqrt((df['Drawdown']**2).mean()) * 100
    
    # 샤프 지수 및 변동성 계산 (연율화)
    daily_std = df['Daily_Return'].std()
    if daily_std > 0:
        ann_vol = daily_std * np.sqrt(252)
        sharpe = cagr / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0

    # MAR 지수 계산
    mar = cagr / abs(mdd) if mdd != 0 else 0
    
    # 승률 및 손익비 계산 (일별 수익률 기반)
    valid_returns = df['Daily_Return'].dropna()
    winning_days = valid_returns[valid_returns > 0]
    losing_days = valid_returns[valid_returns < 0]
    
    total_valid_days = len(valid_returns)
    win_rate = len(winning_days) / total_valid_days if total_valid_days > 0 else 0
    
    gross_profit = winning_days.sum()
    gross_loss = abs(losing_days.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (100.0 if gross_profit > 0 else 0)
    
    # [신규] 일별 수익률의 단순 합산 (산술 합계)
    return_sum = df['Daily_Return'].sum()
    
    # 최대 회복 기간 계산 (일 수)
    under_water = df['PortfolioValue'] < df['Rolling_Max']
    recovery_periods = []
    current_period = 0
    for is_down in under_water:
        if is_down:
            current_period += 1
        else:
            if current_period > 0:
                recovery_periods.append(current_period)
            current_period = 0
    if current_period > 0: recovery_periods.append(current_period)
    max_recovery = max(recovery_periods) if recovery_periods else 0
    
    return {
        'Final Value': final_value,
        'Cumulative Return': df['Cumulative_Return'].iloc[-1] if not df['Cumulative_Return'].empty else 0,
        'CAGR': cagr,
        'MDD': mdd,
        'UI': ui,
        'Sharpe': sharpe,
        'MAR': mar,
        'Win_Rate': win_rate,
        'Profit_Factor': profit_factor,
        'Return_Sum': return_sum,
        'Max_Recovery': max_recovery
    }
