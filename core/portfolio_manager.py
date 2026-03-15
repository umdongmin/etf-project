import pandas as pd
import numpy as np
from core.engine import StrategyEngine

class PortfolioManager:
    """여러 전략을 결합하여 포트폴리오를 구성하고 리밸런싱을 수행하는 클래스"""
    
    # 리밸런싱 프리셋 정의
    REBALANCE_PRESETS = {
        # UI 키와 완전 일치 (portfolio_view.py PRESET_LABELS 기준)
        'none':        {'method': 'none'},
        'annual':      {'method': 'calendar', 'interval_days': 252},
        'semi_annual': {'method': 'calendar', 'interval_days': 126},
        'quarterly':   {'method': 'calendar', 'interval_days': 63},
        'band_5pct':   {'method': 'threshold', 'band': 0.05, 'upper_band': 0.05, 'lower_band': 0.05, 'min_interval_days': 21},
        'band_10pct':  {'method': 'threshold', 'band': 0.10, 'upper_band': 0.10, 'lower_band': 0.10, 'min_interval_days': 21},
        'asymmetric':  {'method': 'threshold', 'upper_band': 0.15, 'lower_band': 0.05, 'min_interval_days': 21},
    }

    def __init__(self, strategies=None, rebalance_preset='none', total_capital=10000.0):
        """
        strategies: dict { 'strategy_name': { 'weight': 0.7, 'engine_kwargs': dict, 'strat_type': 'equity'/'bond' } }
        """
        self.strategies = strategies if strategies else {}
        self.rebalance_preset = rebalance_preset
        self.total_capital = total_capital
        self._last_rebalance_day_idx = 0
        self.history = []

    def register_strategy(self, name, weight, engine_kwargs, strat_type='equity'):
        """시뮬레이션에 참여할 전략 등록"""
        self.strategies[name] = {
            'weight': weight,
            'engine_kwargs': engine_kwargs,
            'strat_type': strat_type
        }

    def run_portfolio(self, start_date=None, end_date=None, total_capital=None):
        """모든 전략을 동기화하여 실행하고 리밸런싱 수행"""
        if total_capital is None:
            total_capital = self.total_capital
            
        if not self.strategies:
            return pd.DataFrame(), {}

        generators = {}
        states = {}
        individual_results = {}
        
        # 1. 제너레이터 초기화
        for name, spec in self.strategies.items():
            kwargs = spec['engine_kwargs'].copy()
            # PortfolioManager에서 할당된 자본으로 강제 덮어쓰기
            kwargs['allocated_capital'] = total_capital * spec['weight']
            
            if spec['strat_type'] == 'equity':
                gen = StrategyEngine._run_generator(**kwargs)
            else: # bond
                gen = StrategyEngine._run_bond_generator(**kwargs)
            
            try:
                state = next(gen)
                generators[name] = gen
                states[name] = state
            except StopIteration as e:
                ret_val = e.value
                if isinstance(ret_val, tuple) and len(ret_val) == 3:
                     h, c, s = ret_val
                else: h, c, s = ret_val, pd.DataFrame(), []
                individual_results[name] = {"history": h, "closed_trades": c, "signal_events": s}
                
        # [중요] 제너레이터 시작 날짜 동기화
        # 레버리지 ETF(TQQQ 등)는 상장일이 늦어 equity generator가 bond보다 늦게 시작할 수 있음
        # 예: TQQQ 상장 2010-02-11 → equity는 2월부터, bond는 1월부터 → 록스텝 불일치 → PortfolioValue 오염
        # 해결: 가장 늦게 시작하는 generator 날짜까지 나머지를 미리 전진
        if states and len(states) > 1:
            max_start_date = max(states[n]['date'] for n in states)
            for name in list(generators.keys()):
                while states[name]['date'] < max_start_date:
                    try:
                        states[name] = generators[name].send(None)
                    except StopIteration as e:
                        ret_val = e.value
                        if isinstance(ret_val, tuple) and len(ret_val) == 3:
                            h, c, s = ret_val
                        else:
                            h, c, s = ret_val, pd.DataFrame(), []
                        individual_results[name] = {"history": h, "closed_trades": c, "signal_events": s}
                        finished_strategies_final_values[name] = states[name]['portfolio_value']
                        del generators[name]
                        del states[name]
                        break

        # 리밸런싱 프리셋 파라미터 로드
        preset_cfg = self.REBALANCE_PRESETS.get(self.rebalance_preset, {'method': 'none'})
        rb_method = preset_cfg['method']

        # 2. 록스텝(Lockstep) 시뮬레이션 엔진 루프
        loop_day_idx = 0
        portfolio_history = []
        finished_strategies_final_values = {}

        while generators:
            active_names = list(generators.keys())
            
            # --- 포트폴리오 전체 가치 평가 ---
            current_total_portfolio_value = sum(states[n]['portfolio_value'] for n in active_names)
            current_total_portfolio_value += sum(finished_strategies_final_values.values())

            # 현재 날짜 (모든 활성 제너레이터는 동일 날짜여야 함)
            current_date = states[active_names[0]]['date']
            portfolio_history.append({'Date': current_date, 'PortfolioValue': current_total_portfolio_value})

            # --- [Hard Rebalancing 계산부] ---
            cash_injections = {n: 0.0 for n in active_names}
            
            if rb_method != 'none' and current_total_portfolio_value > 0 and len(active_names) > 1:
                do_rebalance = False
                if rb_method == 'calendar':
                    interval = preset_cfg.get('interval_days', 252)
                    if (loop_day_idx - self._last_rebalance_day_idx) >= interval:
                        do_rebalance = True
                elif rb_method == 'threshold':
                    min_interval = preset_cfg.get('min_interval_days', 21)
                    if (loop_day_idx - self._last_rebalance_day_idx) >= min_interval:
                        upper_band = preset_cfg.get('upper_band', preset_cfg.get('band', 0.10))
                        lower_band = preset_cfg.get('lower_band', preset_cfg.get('band', 0.10))
                        for name in active_names:
                            target_w = self.strategies[name]['weight']
                            curr_w = states[name]['portfolio_value'] / current_total_portfolio_value
                            drift = curr_w - target_w
                            if drift > upper_band or drift < -lower_band:
                                do_rebalance = True
                                break

                if do_rebalance:
                    self._last_rebalance_day_idx = loop_day_idx
                    for name in active_names:
                        target_w = self.strategies[name]['weight']
                        target_val = target_w * current_total_portfolio_value
                        curr_val = states[name]['portfolio_value']
                        cash_injections[name] = target_val - curr_val

            loop_day_idx += 1
            # --- 제너레이터 1일 단위 전진 ---
            for name in active_names:
                gen = generators[name]
                delta = cash_injections[name]
                try:
                    injection_data = {'cash_delta': delta} if delta != 0.0 else None
                    next_state = gen.send(injection_data)
                    states[name] = next_state
                except StopIteration as e:
                    ret_val = e.value
                    if isinstance(ret_val, tuple) and len(ret_val) == 3:
                        history_df, closed_trades_df, signal_events = ret_val
                    else:
                        history_df, closed_trades_df, signal_events = ret_val, pd.DataFrame(), []
                    
                    individual_results[name] = {
                        "history": history_df,
                        "closed_trades": closed_trades_df,
                        "signal_events": signal_events
                    }
                    # 종료 시점의 가치를 저장하여 포트폴리오 합산에 계속 반영
                    finished_strategies_final_values[name] = states[name]['portfolio_value']
                    del generators[name]
                    del states[name]
                    
        # 3. 결과 정리
        if not portfolio_history:
            return pd.DataFrame(), individual_results
            
        combined_equity = pd.DataFrame(portfolio_history).set_index('Date')
        return combined_equity, individual_results
