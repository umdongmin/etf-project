import copy
import random
import json
import datetime
import os
import pandas as pd
import optuna
from core.engine import StrategyEngine
from utils.metrics import calculate_metrics

class StrategyOptimizer:
    """전략 파라미터 최적화를 수행하는 핵심 엔진 클래스 (Optuna + SQLite 학습형)"""
    
    def __init__(self, data_dict, fg_df, vix_df):
        self.data_dict = data_dict
        self.fg_df = fg_df
        self.vix_df = vix_df
        # 로컬 SQLite DB 경로 설정 (data 폴더 사용)
        db_dir = "data"
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
        self.db_path = f"sqlite:///{db_dir}/optimization_history.db"

    def run_simulation(self, cfg, s_date, e_date, trade_at):
        """단일 시뮬레이션 수행 및 점수 계산"""
        try:
            hist, _, _ = StrategyEngine.run_golden_strategy(
                self.data_dict, self.fg_df, self.vix_df, 
                cfg['leverage_asset'], cfg['base_asset'], 
                0.0, s_date, e_date, cfg, trade_at, smart_params=cfg
            )
            if hist.empty: return -999, 0, 0
            m = calculate_metrics(hist)
            cagr, mdd = m['CAGR']*100, abs(m['MDD']*100)
            score = cagr - 0.5 * mdd
            return score, cagr, mdd
        except Exception as e:
            return -999, 0, 0

    def optimize(self, base_cfg, n_iter, s_date, e_date, targets, trade_at, progress_callback=None):
        """Optuna 기반 베이지안 최적화 수행"""
        
        # 0. 현금 비중 강제 0% 고정 (사용자 요청)
        base_cfg['cash_ratio_pct'] = 0.0
        base_cfg['cash_ratio'] = 0.0
        
        # 1. 베이스라인 성과 측정
        base_score, b_cagr, b_mdd = self.run_simulation(base_cfg, s_date, e_date, trade_at)
        
        # 2. Optuna 스터디 생성 또는 불러오기 (SQLite 연동)
        study = optuna.create_study(
            study_name="tqqq_strategy_optimization",
            storage=self.db_path,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42) # 학습형 샘플러 (TPE)
        )

        # 3. 목표 함수 정의
        def objective(trial):
            cand = copy.deepcopy(base_cfg)
            
            # 탐색 범위 정의 (지능형 가이드)
            # [FSO 확장] 탐색 범위 정의 (구조 + 수치 전면 개방)
            # [FSO 확장] 탐색 범위 정의 (구조 + 수치 전면 개방)
            if "매수 시그널 (Full)" in targets:
                # 매수 신호 (1~3)
                for i in range(len(cand['buy_signals'])):
                    if i >= 3: break # 최대 3개까지만 탐색
                    prefix = f"b{i+1}_"
                    # 구조(ON/OFF) 탐색
                    cand['buy_signals'][i]['rsi_cross'] = trial.suggest_categorical(f"{prefix}rsi_cross", [True, False])
                    cand['buy_signals'][i]['rsi_inc'] = trial.suggest_categorical(f"{prefix}rsi_inc", [True, False])
                    cand['buy_signals'][i]['macd_inc'] = trial.suggest_categorical(f"{prefix}macd_inc", [True, False])
                    cand['buy_signals'][i]['macd_signal_below'] = trial.suggest_categorical(f"{prefix}macd_signal_below", [True, False])
                    cand['buy_signals'][i]['macd_golden'] = trial.suggest_categorical(f"{prefix}macd_golden", [True, False])
                    cand['buy_signals'][i]['use_adx'] = trial.suggest_categorical(f"{prefix}use_adx", [True, False])
                    cand['buy_signals'][i]['use_willr'] = trial.suggest_categorical(f"{prefix}use_willr", [True, False])
                    cand['buy_signals'][i]['di_plus_above'] = trial.suggest_categorical(f"{prefix}di_plus_above", [True, False])
                    cand['buy_signals'][i]['di_plus_cross'] = trial.suggest_categorical(f"{prefix}di_plus_cross", [True, False])
                    cand['buy_signals'][i]['use_sar'] = trial.suggest_categorical(f"{prefix}use_sar", [True, False])
                    
                    # 수치 탐색
                    cand['buy_signals'][i]['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", 25, 45)
                    if cand['buy_signals'][i]['use_willr']:
                        cand['buy_signals'][i]['willr_val'] = trial.suggest_int(f"{prefix}willr_val", -90, -70)
                    if cand['buy_signals'][i]['use_adx']:
                        cand['buy_signals'][i]['adx_val'] = trial.suggest_int(f"{prefix}adx_val", 20, 50)

            if "매도 시그널 (Full)" in targets:
                # 매도 신호 (1~4)
                for i in range(len(cand['sell_signals'])):
                    if i >= 4: break # 최대 4개까지만 탐색
                    prefix = f"s{i+1}_"
                    # 구조 탐색
                    cand['sell_signals'][i]['rsi_dead'] = trial.suggest_categorical(f"{prefix}rsi_dead", [True, False])
                    cand['sell_signals'][i]['rsi_dec'] = trial.suggest_categorical(f"{prefix}rsi_dec", [True, False])
                    cand['sell_signals'][i]['macd_dec'] = trial.suggest_categorical(f"{prefix}macd_dec", [True, False])
                    cand['sell_signals'][i]['macd_signal_above'] = trial.suggest_categorical(f"{prefix}macd_signal_above", [True, False])
                    cand['sell_signals'][i]['macd_dead'] = trial.suggest_categorical(f"{prefix}macd_dead", [True, False])
                    cand['sell_signals'][i]['di_minus_above'] = trial.suggest_categorical(f"{prefix}di_minus_above", [True, False])
                    cand['sell_signals'][i]['di_minus_cross'] = trial.suggest_categorical(f"{prefix}di_minus_cross", [True, False])
                    cand['sell_signals'][i]['use_chandelier'] = trial.suggest_categorical(f"{prefix}use_chan", [True, False])
                    cand['sell_signals'][i]['use_sar'] = trial.suggest_categorical(f"{prefix}use_sar", [True, False])
                    cand['sell_signals'][i]['use_willr'] = trial.suggest_categorical(f"{prefix}use_willr", [True, False])

                    # 수치 탐색
                    cand['sell_signals'][i]['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", 60, 85)
                    if cand['sell_signals'][i]['use_chandelier']:
                        cand['sell_signals'][i]['chandelier_mult'] = trial.suggest_float(f"{prefix}chan_mult", 2.0, 5.0)
                    if cand['sell_signals'][i]['use_willr']:
                        cand['sell_signals'][i]['willr_val'] = trial.suggest_int(f"{prefix}willr_val", -30, -10)

            if "리벨런싱 및 ATR 변동성 (Full)" in targets: # UI 레이블 변경 예정
                # 고정 비율
                cand['use_fixed_reb'] = trial.suggest_categorical("use_fixed", [True, False])
                if cand['use_fixed_reb']:
                    cand['buy_reb_up'] = trial.suggest_float("buy_reb_up", 0.01, 0.05)
                    cand['buy_reb_down'] = trial.suggest_float("buy_reb_down", -0.07, -0.02)
                    cand['sell_reb_up'] = trial.suggest_float("sell_reb_up", 0.01, 0.05)
                    cand['sell_reb_down'] = trial.suggest_float("sell_reb_down", -0.05, -0.01)

                # ATR 변동성
                cand['use_atr_reb'] = trial.suggest_categorical("use_atr", [True, False])
                if cand['use_atr_reb']:
                    cand['atr_period_buy'] = trial.suggest_categorical("atr_p_buy", [14, 20])
                    cand['atr_period_sell'] = trial.suggest_categorical("atr_p_sell", [14, 20])
                    cand['atr_mult_buy_up'] = trial.suggest_float("atr_m_buy_up", 5.0, 15.0)
                    cand['atr_mult_buy_down'] = trial.suggest_float("atr_m_buy_down", 1.0, 3.0)
                    cand['atr_mult_sell'] = trial.suggest_float("atr_m_sell", 1.5, 5.0)
            
            if "패닉 가속 (MA/RSI)" in targets:
                cand['use_panic'] = trial.suggest_categorical("use_panic", [True, False])
                if cand['use_panic']:
                    cand['panic_ma'] = trial.suggest_categorical("panic_ma", [120, 150, 200])
                    p_s1 = trial.suggest_int("p_rsi_s1", 20, 30)
                    p_s2 = trial.suggest_int("p_rsi_s2", 25, 35)
                    p_s3 = trial.suggest_int("p_rsi_s3", 30, 40)
                    cand['panic_rsi_s1'] = p_s1
                    cand['panic_rsi_s2'] = p_s2
                    cand['panic_rsi_s3'] = p_s3
                    if 'panic_buy_signals' in cand:
                        for i, rsi in enumerate([p_s1, p_s2, p_s3]):
                            if i < len(cand['panic_buy_signals']): 
                                cand['panic_buy_signals'][i]['rsi_val'] = rsi
            
            if "S3 보호 조건 (Gap/Acc)" in targets:
                for i in range(len(cand['s3_protection'])):
                    prefix = f"s3_{i+1}_"
                    cand['s3_protection'][i]['use_daily_drop'] = trial.suggest_categorical(f"{prefix}drop", [True, False])
                    cand['s3_protection'][i]['use_ma60'] = trial.suggest_categorical(f"{prefix}ma60", [True, False])
                    cand['s3_protection'][i]['use_ma200'] = trial.suggest_categorical(f"{prefix}ma200", [True, False])
                    cand['s3_protection'][i]['use_vxn_jump'] = trial.suggest_categorical(f"{prefix}vxn", [True, False])
                    cand['s3_protection'][i]['use_gap_down'] = trial.suggest_categorical(f"{prefix}gap", [True, False])
                    cand['s3_protection'][i]['use_drop_acc'] = trial.suggest_categorical(f"{prefix}acc", [True, False])

            if "방어 및 대피 시스템 (VXN/손절)" in targets:
                cand['use_vxn_safety'] = trial.suggest_categorical("use_vxn", [True, False])
                cand['vxn_exit'] = trial.suggest_int("vxn_exit", 25, 45)
                cand['use_sl_control'] = trial.suggest_categorical("use_sl_ctrl", [True, False])
                cand['sl_control_limit'] = trial.suggest_int("sl_ctrl_lim", -30, -10)
                cand['use_set_sl'] = trial.suggest_categorical("use_set_sl", [True, False])
                cand['set_sl_limit'] = trial.suggest_int("set_sl_lim", -20, -5)

            score, _, _ = self.run_simulation(cand, s_date, e_date, trade_at)
            
            if progress_callback:
                try:
                    # 완료된 trial이 없을 때 best_value 접근 시 ValueError 발생 대응
                    best_v = study.best_value
                except (ValueError, RuntimeError):
                    best_v = score
                progress_callback(trial.number + 1, n_iter, best_v)
                
            return score

        # 4. 베이스라인을 첫 번째 시도(Seed)로 주입 시도 (최초 1회만)
        # 생략 (Optuna sampler가 자동으로 보정하도록 설계)

        # 5. 최적화 실행
        study.optimize(objective, n_trials=n_iter)

        # 6. 결과 정리 및 반환
        results = []
        for t in study.trials:
            if t.value is not None and t.value > -900:
                cfg = copy.deepcopy(base_cfg)
                p = t.params
                
                # [복원 로직] 탐색된 파라미터를 cfg에 다시 매핑
                # 이 부분은 objective 함수와 동일한 로직으로 p 값을 cfg에 주입
                for key, val in p.items():
                    # 매수 신호 처리
                    if key.startswith('b') and '_rsi_val' in key:
                        idx = int(key.split('_')[0][1:]) - 1
                        cfg['buy_signals'][idx]['rsi_val'] = val
                    elif key.startswith('b') and '_' in key:
                        idx = int(key.split('_')[0][1:]) - 1
                        field = "_".join(key.split('_')[1:])
                        cfg['buy_signals'][idx][field] = val
                    # 매도 신호 처리
                    elif key.startswith('s') and not key.startswith('s3') and '_' in key:
                        idx = int(key.split('_')[0][1:]) - 1
                        field = "_".join(key.split('_')[1:])
                        if field == "chan_mult": field = "chandelier_mult"
                        cfg['sell_signals'][idx][field] = val
                    # S3 보호 처리
                    elif key.startswith('s3_'):
                        parts = key.split('_')
                        idx, field = int(parts[1]) - 1, parts[2]
                        field_map = {'drop':'use_daily_drop','ma60':'use_ma60','ma200':'use_ma200','vxn':'use_vxn_jump','gap':'use_gap_down','acc':'use_drop_acc'}
                        cfg['s3_protection'][idx][field_map[field]] = val
                    # 기타 공통 변수
                    else:
                        map_dict = {'p_rsi_s1':'panic_rsi_s1','p_rsi_s2':'panic_rsi_s2','p_rsi_s3':'panic_rsi_s3','sl_ctrl_lim':'sl_control_limit','sl_ctrl':'use_sl_control','set_sl_lim':'set_sl_limit'}
                        cfg[map_dict.get(key, key)] = val
                
                # 패닉 RSI 하위 시그널 동기화
                if 'panic_rsi_s1' in cfg and 'panic_buy_signals' in cfg:
                    cfg['panic_buy_signals'][0]['rsi_val'] = cfg['panic_rsi_s1']
                    cfg['panic_buy_signals'][1]['rsi_val'] = cfg['panic_rsi_s2']
                    cfg['panic_buy_signals'][2]['rsi_val'] = cfg['panic_rsi_s3']

                s, c, m = self.run_simulation(cfg, s_date, e_date, trade_at)
                results.append({'cfg': cfg, 'score': s, 'cagr': c, 'mdd': m})

        results.sort(key=lambda x: x['score'], reverse=True)
        return results, base_score, b_cagr, b_mdd

    def get_total_trials(self):
        """축적된 총 분석 횟수 반환"""
        try:
            study = optuna.load_study(study_name="tqqq_strategy_optimization", storage=self.db_path)
            return len(study.trials)
        except:
            return 0
