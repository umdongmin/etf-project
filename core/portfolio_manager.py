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
        # ── 신규 ──────────────────────────────────────────────────────────────────
        # hybrid   : calendar 강제 주기(quarterly) + band 이탈 즉시 트리거 (OR 조건)
        # trend    : QQQ MA200 기반 동적 비중 전환 + band 이탈 트리거
        'hybrid':      {'method': 'hybrid', 'interval_days': 63, 'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
        'trend':       {'method': 'trend',  'upper_band': 0.07, 'lower_band': 0.07, 'min_interval_days': 5},
    }

    def __init__(self, strategies=None, rebalance_preset='none', total_capital=10000.0,
                 regime_series=None, regime_weights=None, custom_preset_cfg=None, fee_rate=0.0, integer_shares=False,
                 trend_data=None, trend_defense_weights=None):
        """
        strategies           : dict { 'strategy_name': { 'weight': 0.7, 'engine_kwargs': dict, 'strat_type': 'equity'/'bond' } }
        regime_series        : pd.Series indexed by date → regime label ('bull'/'neutral'/'caution'/'bear')
                               None이면 각 전략의 고정 weight 사용 (기존 동작 유지)
        regime_weights       : dict { 'bull': {'strategy_name': 0.8, ...}, 'neutral': {...}, ... }
                               None이면 고정 weight 사용
        custom_preset_cfg    : dict — REBALANCE_PRESETS에 없는 임시 프리셋을 직접 주입
                               예: {'method': 'threshold', 'upper_band': 0.12, 'lower_band': 0.07, 'min_interval_days': 21}
                               설정 시 rebalance_preset 값은 무시됨
        trend_data           : pd.Series[date] → bool  (True = QQQ > MA200 = 공격 모드)
                               method='trend' 사용 시 필수. None이면 항상 공격 비중 사용.
        trend_defense_weights: dict { strategy_name: weight }  방어 모드 비중
                               예: {'TQQQ': 0.50, 'Bond': 0.50}
                               None이면 고정 비중 유지 (trend 효과 없음)
        """
        self.strategies = strategies if strategies else {}
        self.rebalance_preset = rebalance_preset
        self.total_capital = total_capital
        self._last_rebalance_day_idx = 0
        self.history = []

        # Dynamic Weight 설정 (None이면 기존 fixed weight 동작)
        self.regime_series         = regime_series
        self.regime_weights        = regime_weights
        self.custom_preset_cfg     = custom_preset_cfg
        self.fee_rate              = fee_rate
        self.integer_shares        = integer_shares
        # Trend-Filtered 설정
        self.trend_data            = trend_data             # pd.Series[date] → bool
        self.trend_defense_weights = trend_defense_weights  # dict: name → weight (방어 모드)

    def _get_target_weight(self, name: str, current_date) -> float:
        """현재 날짜의 Regime / Trend에 따라 target weight 반환.
        우선순위: regime_weights > trend_defense_weights > 고정 weight
        """
        lookup = current_date
        if hasattr(lookup, 'date'):
            lookup = lookup.date()

        # 1. Regime 기반 (기존 동작 유지)
        if self.regime_series is not None and self.regime_weights is not None:
            regime = self.regime_series.get(lookup, 'neutral')
            rw = self.regime_weights.get(regime, {})
            if name in rw:
                return rw[name]

        # 2. Trend-Filtered: QQQ MA200 기반 공격/방어 전환
        if self.trend_data is not None and self.trend_defense_weights is not None:
            is_uptrend = bool(self.trend_data.get(lookup, True))
            if not is_uptrend:
                defense_w = self.trend_defense_weights.get(name)
                if defense_w is not None:
                    return defense_w

        return self.strategies[name]['weight']

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
            
            kwargs['fee_rate'] = self.fee_rate
            kwargs['integer_shares'] = self.integer_shares
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

        # 리밸런싱 프리셋 파라미터 로드 (custom_preset_cfg 우선)
        if self.custom_preset_cfg is not None:
            preset_cfg = self.custom_preset_cfg
        else:
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
                days_since = loop_day_idx - self._last_rebalance_day_idx

                if rb_method == 'calendar':
                    interval = preset_cfg.get('interval_days', 252)
                    if days_since >= interval:
                        do_rebalance = True

                elif rb_method == 'threshold':
                    min_interval = preset_cfg.get('min_interval_days', 21)
                    if days_since >= min_interval:
                        upper_band = preset_cfg.get('upper_band', preset_cfg.get('band', 0.10))
                        lower_band = preset_cfg.get('lower_band', preset_cfg.get('band', 0.10))
                        for name in active_names:
                            target_w = self._get_target_weight(name, current_date)
                            curr_w = states[name]['portfolio_value'] / current_total_portfolio_value
                            drift = curr_w - target_w
                            if drift > upper_band or drift < -lower_band:
                                do_rebalance = True
                                break

                elif rb_method == 'hybrid':
                    # calendar(quarterly) 강제 주기 OR band 이탈 즉시 트리거
                    interval   = preset_cfg.get('interval_days', 63)
                    min_interval = preset_cfg.get('min_interval_days', 5)
                    upper_band = preset_cfg.get('upper_band', 0.07)
                    lower_band = preset_cfg.get('lower_band', 0.07)
                    if days_since >= interval:
                        do_rebalance = True
                    elif days_since >= min_interval:
                        for name in active_names:
                            target_w = self._get_target_weight(name, current_date)
                            curr_w = states[name]['portfolio_value'] / current_total_portfolio_value
                            drift = curr_w - target_w
                            if drift > upper_band or drift < -lower_band:
                                do_rebalance = True
                                break

                elif rb_method == 'trend':
                    # QQQ MA200 기반 동적 비중 전환 + band 이탈 트리거
                    # - 공격/방어 모드 전환 시 즉시 리밸런싱
                    # - 같은 모드에서도 band 이탈 시 트리거
                    min_interval = preset_cfg.get('min_interval_days', 5)
                    upper_band   = preset_cfg.get('upper_band', 0.07)
                    lower_band   = preset_cfg.get('lower_band', 0.07)
                    if days_since >= min_interval:
                        for name in active_names:
                            target_w = self._get_target_weight(name, current_date)
                            curr_w = states[name]['portfolio_value'] / current_total_portfolio_value
                            drift = curr_w - target_w
                            if drift > upper_band or drift < -lower_band:
                                do_rebalance = True
                                break

                if do_rebalance:
                    self._last_rebalance_day_idx = loop_day_idx
                    for name in active_names:
                        target_w = self._get_target_weight(name, current_date)
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

    def predict_portfolio_today(self, data_dict, fg_df, vxn_df, macro_df, news_df,
                                start_date=None):
        """
        포트폴리오 구성 전략들의 오늘 예상 신호를 통합하여 반환.

        Returns
        -------
        dict with keys:
            'date'              : 분석 기준일 (datetime.date)
            'strategies'        : { strategy_name: signal_dict }
                equity signal_dict keys: signal, stage, price, rsi, vxn, summary
                bond   signal_dict keys: signal(asset), f_state, gap, yc, rsi, lev_stage, lev_asset
            'rebalance'         : {
                'needed'        : bool  — 리밸런싱 필요 여부
                'reason'        : str   — 사유 ("drift", "calendar", "none")
                'drifts'        : { name: drift_pct }  — 목표 대비 실제 드리프트
                'target_weights': { name: target_w }
                'current_weights_estimated': { name: est_w }  — 신호 기반 추정 비중
            }
            'trend_mode'        : 'uptrend' | 'downtrend' | None
            'portfolio_label'   : str  — 포트폴리오 레이블 (요약용)
        """
        import datetime as _dt

        result = {
            'date': None,
            'strategies': {},
            'rebalance': {
                'needed': False,
                'reason': 'none',
                'drifts': {},
                'target_weights': {},
                'current_weights_estimated': {},
            },
            'trend_mode': None,
            'portfolio_label': '',
        }

        today = _dt.date.today()

        # ── 1. 각 전략별 신호 수집 ──────────────────────────────────────────────
        latest_date = None

        for name, spec in self.strategies.items():
            kwargs   = spec['engine_kwargs']
            stype    = spec.get('strat_type', 'equity')
            sig      = None

            try:
                if stype == 'equity':
                    sig = StrategyEngine.predict_today_signal(
                        data_dict=data_dict,
                        fg_df=fg_df,
                        vxn_df=vxn_df,
                        news_df=news_df,
                        leverage_asset=kwargs.get('leverage_asset', 'TQQQ'),
                        base_asset=kwargs.get('base_asset', 'QQQ'),
                        cash_ratio=kwargs.get('cash_ratio', 0.0),
                        start_date=kwargs.get('start_date', start_date),
                        end_date=kwargs.get('end_date', None),
                        params=kwargs.get('params', {}),
                        trade_at=kwargs.get('trade_at', '종가'),
                        smart_params=kwargs.get('smart_params', {}),
                    )
                    if sig:
                        sig['strat_type'] = 'equity'

                elif stype == 'bond':
                    bond_sig = StrategyEngine.predict_bond_signal(
                        data_dict=data_dict,
                        macro_df=macro_df,
                        params=kwargs.get('params', None),
                    )
                    if bond_sig:
                        # 레버리지 스테이지 추정: run_bond_strategy 마지막 행에서 추출
                        lev_stage  = 0
                        lev_asset  = None
                        try:
                            lev_params = kwargs.get('lev_params', None)
                            if lev_params and lev_params.get('use_leverage', False):
                                res_df, _, _ = StrategyEngine.run_bond_strategy(
                                    data_dict, macro_df, params=kwargs.get('params', None)
                                )
                                if not res_df.empty:
                                    row = res_df.iloc[-1]
                                    lev_stage = int(row.get('Stage', 0))
                                    # Trade_Label에서 레버리지 자산명 추출 (예: "진행중(S2): TMF")
                                    tl = str(row.get('Trade_Label', ''))
                                    lev_asset = tl.split(': ')[-1].strip() if ': ' in tl else row.get('Asset', None)
                        except Exception:
                            pass
                        bond_sig['lev_stage'] = lev_stage
                        bond_sig['lev_asset'] = lev_asset
                        bond_sig['strat_type'] = 'bond'
                        sig = bond_sig

            except Exception as e:
                print(f"[predict_portfolio_today] {name} 신호 오류: {e}")
                sig = None

            result['strategies'][name] = sig

            # 기준일 수집 (equity 우선)
            if sig and sig.get('date') and (latest_date is None or stype == 'equity'):
                d = sig['date']
                if hasattr(d, 'date'):
                    d = d.date()
                latest_date = d

        result['date'] = latest_date or today

        # ── 2. Trend 모드 판별 ──────────────────────────────────────────────────
        if self.trend_data is not None:
            lookup = result['date']
            is_up  = bool(self.trend_data.get(lookup, True))
            result['trend_mode'] = 'uptrend' if is_up else 'downtrend'

        # ── 3. 목표 비중 계산 ──────────────────────────────────────────────────
        target_weights = {}
        for name in self.strategies:
            target_weights[name] = self._get_target_weight(name, result['date'])
        result['rebalance']['target_weights'] = target_weights

        # ── 4. 현재 비중 추정 (신호 기반 stage → 레버리지 비중 반영) ──────────
        # equity: stage > 0 이면 레버리지 활성. stage==0 이면 QQQ/QLD 보유 (기본자산 비중 유지).
        # bond  : lev_stage > 0 이면 TMF/TMV 활성.
        # 현재 비중은 "target 비중 × stage 활성 여부"로 추정 (실제 잔고 없으므로 근사값).
        est_weights = {}
        for name, sig in result['strategies'].items():
            tw = target_weights.get(name, self.strategies[name]['weight'])
            if sig is None:
                est_weights[name] = tw  # 신호 없으면 목표 비중 유지 가정
            elif sig.get('strat_type') == 'equity':
                stage = sig.get('stage', 0)
                # stage==0 → 현금/QQQ 보유 → 레버리지 비중 0, 전략 자체 비중은 유지
                # 여기서는 "포트폴리오 내 비중"이 아닌 레버리지 활성도를 별도 표시
                est_weights[name] = tw  # 포트폴리오 비중은 리밸런싱으로 결정
            elif sig.get('strat_type') == 'bond':
                est_weights[name] = tw
        result['rebalance']['current_weights_estimated'] = est_weights

        # ── 5. 드리프트 계산 및 리밸런싱 필요 여부 판단 ─────────────────────────
        preset_cfg = self.custom_preset_cfg
        if preset_cfg is None:
            preset_cfg = self.REBALANCE_PRESETS.get(self.rebalance_preset, {'method': 'none'})

        rb_method   = preset_cfg.get('method', 'none')
        upper_band  = preset_cfg.get('upper_band', 0.07)
        lower_band  = preset_cfg.get('lower_band', 0.07)

        drifts = {}
        for name in self.strategies:
            tw  = target_weights.get(name, 0)
            ew  = est_weights.get(name, tw)
            drifts[name] = round(ew - tw, 4)
        result['rebalance']['drifts'] = drifts

        # 리밸런싱 필요 판정
        if rb_method == 'none':
            result['rebalance']['needed'] = False
            result['rebalance']['reason'] = 'none'
        else:
            max_drift = max(abs(d) for d in drifts.values()) if drifts else 0
            if max_drift > upper_band:
                result['rebalance']['needed'] = True
                result['rebalance']['reason'] = 'drift'
            else:
                result['rebalance']['needed'] = False
                result['rebalance']['reason'] = 'within_band'

        # ── 6. 누적 수익률 계산 ──────────────────────────────────────────────────
        # start_date부터 현재(result['date'])까지 포트폴리오 시뮬레이션 수행
        returns = {}
        try:
            calc_start = start_date or (result['date'] - _dt.timedelta(days=252))
            calc_end = result['date']

            print(f"[DEBUG] 수익률 계산 시작: start={calc_start}, end={calc_end}")
            print(f"[DEBUG] 등록된 전략: {list(self.strategies.keys())}")

            # 포트폴리오 전체 수익률
            portfolio_history, individual_results = self.run_portfolio(
                start_date=calc_start,
                end_date=calc_end,
                total_capital=self.total_capital
            )

            print(f"[DEBUG] portfolio_history 길이: {len(portfolio_history)}, 컬럼: {portfolio_history.columns.tolist() if not portfolio_history.empty else 'EMPTY'}")
            print(f"[DEBUG] individual_results 키: {individual_results.keys()}")

            if not portfolio_history.empty:
                start_val = portfolio_history.iloc[0].get('PortfolioValue', 10000.0) if len(portfolio_history) > 0 else 10000.0
                end_val = portfolio_history.iloc[-1].get('PortfolioValue', 10000.0) if len(portfolio_history) > 0 else 10000.0
                portfolio_return = (end_val - start_val) / start_val * 100 if start_val > 0 else 0.0
                returns['portfolio_return'] = round(portfolio_return, 2)
                print(f"[DEBUG] 포트폴리오 수익률: {returns['portfolio_return']}% (start={start_val:.2f}, end={end_val:.2f})")
            else:
                returns['portfolio_return'] = 0.0
                print(f"[DEBUG] portfolio_history가 비어있음")

            # 개별 전략 수익률
            returns['strategy_returns'] = {}
            for name in self.strategies:
                if name in individual_results:
                    hist = individual_results[name].get('history')
                    print(f"[DEBUG] {name} hist 타입: {type(hist)}, 길이: {len(hist) if hist is not None else 'None'}")
                    if hist is not None and not hist.empty:
                        print(f"[DEBUG] {name} hist 컬럼: {hist.columns.tolist()}")
                        # Series의 행 값에 접근 (개별 전략은 'Value' 컬럼 사용)
                        s_start = hist.iloc[0]['Value'] if 'Value' in hist.columns else 10000.0
                        s_end = hist.iloc[-1]['Value'] if 'Value' in hist.columns else 10000.0
                        s_return = (s_end - s_start) / s_start * 100 if s_start > 0 else 0.0
                        returns['strategy_returns'][name] = round(s_return, 2)
                        print(f"[DEBUG] {name} 수익률: {returns['strategy_returns'][name]}% (start={s_start:.2f}, end={s_end:.2f})")
                    else:
                        returns['strategy_returns'][name] = 0.0
                        print(f"[DEBUG] {name} 히스토리 비어있음")
                else:
                    returns['strategy_returns'][name] = 0.0
                    print(f"[DEBUG] {name} individual_results에 없음")
        except Exception as e:
            # 수익률 계산 실패 시 0으로 설정 + 오류 로그 출력
            import traceback
            print(f"[ERROR] 수익률 계산 오류: {e}")
            traceback.print_exc()
            returns = {'portfolio_return': 0.0, 'strategy_returns': {name: 0.0 for name in self.strategies}}

        result['returns'] = returns

        # ── 7. 포트폴리오 레이블 ────────────────────────────────────────────────
        parts = []
        for name, spec in self.strategies.items():
            pct = int(round(spec['weight'] * 100))
            parts.append(f"{name} {pct}%")
        result['portfolio_label'] = ' / '.join(parts) + f" [{self.rebalance_preset or 'custom'}]"

        return result
