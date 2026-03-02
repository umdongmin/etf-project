import pandas as pd
import numpy as np

def calculate_metrics(history):
    """성과 지표 계산 함수 (MDD, CAGR 등)"""
    if history.empty:
        return {'Final Value': 0, 'Cumulative Return': 0, 'CAGR': 0, 'MDD': 0}
    
    df = history.copy()
    if 'PortfolioValue' not in df.columns and 'Value' in df.columns:
        df.rename(columns={'Value': 'PortfolioValue'}, inplace=True)
        
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
