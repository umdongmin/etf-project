import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, data_dict, initial_capital=10000.0):
        self.data_dict = data_dict
        self.initial_capital = initial_capital
        
    def run_strategy(self, strategy_func):
        """
        매일 각 자산의 목표 비중을 반환하는 전략 함수를 실행합니다.
        strategy_func(date, current_prices, indicators) -> {'QQQ': weight1, 'QLD': weight2, 'TQQQ': weight3}
        """
        # 모든 데이터를 동일한 날짜로 정렬
        dates = self.data_dict['QQQ'].index
        
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        holdings = {ticker: 0.0 for ticker in self.data_dict.keys()}
        
        history = []
        
        current_asset = 'QQQ' # QQQ로 시작
        
        for date in dates:
            prices = {ticker: self.data_dict[ticker].loc[date, 'Close'] for ticker in self.data_dict.keys()}
            # 참고: 종가 데이터가 Series/DataFrame 형태이므로 멀티인덱스 등을 적절히 처리해야 할 수 있음
            
            # 단순 가정: 매일 종가에 리벨런싱 수행
            # 지표 기반의 대상 자산 선정 로직
            rsi = self.data_dict['QQQ'].loc[date, 'RSI']
            
            # 전략 로직 (예시: RSI 기반)
            # 추후 최적화를 위해 람다 함수나 별도 함수로 전달 가능
            target_asset = 'QQQ'
            if rsi < 30:
                target_asset = 'TQQQ' # 과매도 구간에서 공격적으로 대응
            elif rsi < 40:
                target_asset = 'QLD'  # 완만한 하락 구간에서 대응
            else:
                target_asset = 'QQQ'  # 고점/정상 구간에서 안정적 보유
            
            # 대상 자산이 변경된 경우 리벨런싱
            if target_asset != current_asset:
                # 현재 자산 매도
                cash = holdings[current_asset] * prices[current_asset]
                holdings[current_asset] = 0
                
                # 목표 자산 매수
                holdings[target_asset] = cash / prices[target_asset]
                cash = 0
                current_asset = target_asset
            
            # 포트폴리오 가치 업데이트
            daily_value = cash + sum(holdings[t] * prices[t] for t in self.data_dict.keys())
            history.append({
                'Date': date,
                'PortfolioValue': daily_value,
                'Asset': current_asset,
                'RSI': rsi
            })
            
        return pd.DataFrame(history).set_index('Date')

def calculate_metrics(history):
    df = history.copy()
    df['Daily_Return'] = df['PortfolioValue'].pct_change()
    df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
    
    # CAGR (연평균 성장률)
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    final_value = df['PortfolioValue'].iloc[-1]
    initial_value = df['PortfolioValue'].iloc[0]
    cagr = (final_value / initial_value) ** (1 / years) - 1
    
    # MDD (최대 낙폭)
    df['Rolling_Max'] = df['PortfolioValue'].cummax()
    df['Drawdown'] = df['PortfolioValue'] / df['Rolling_Max'] - 1
    mdd = df['Drawdown'].min()
    
    return {
        'Final Value': final_value,
        'Cumulative Return': df['Cumulative_Return'].iloc[-1],
        'CAGR': cagr,
        'MDD': mdd
    }
