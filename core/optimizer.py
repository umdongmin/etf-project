import copy
import random
import json
import datetime
import pandas as pd
from core.engine import StrategyEngine
from utils.metrics import calculate_metrics

class StrategyOptimizer:
    """전략 파라미터 최적화를 수행하는 핵심 엔진 클래스"""
    
    def __init__(self, data_dict, fg_df, vix_df):
        self.data_dict = data_dict
        self.fg_df = fg_df
        self.vix_df = vix_df

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
        except:
            return -999, 0, 0

    def optimize(self, base_cfg, n_iter, s_date, e_date, targets, trade_at, progress_callback=None):
        """무작위 탐색 기반 최적화 수행"""
        results = []
        base_score, b_cagr, b_mdd = self.run_simulation(base_cfg, s_date, e_date, trade_at)
        
        best_score = base_score
        
        for i in range(n_iter):
            cand = copy.deepcopy(base_cfg)
            
            # 파라미터 변조 로직 (UI 독립적)
            if "매수/매도 시그널 (RSI)" in targets:
                if len(cand['buy_signals']) > 0:
                    cand['buy_signals'][0]['rsi_val'] = random.choice([32, 35, 38, 40, 42])
                if len(cand['sell_signals']) > 0:
                    cand['sell_signals'][0]['rsi_val'] = random.choice([65, 68, 70, 71, 72, 75, 78])

            if "리밸런싱 비율 (Buy/Sell)" in targets:
                cand['buy_reb_up'] = random.choice([0.015, 0.018, 0.02, 0.022, 0.025, 0.03])
                cand['sell_reb_down'] = random.choice([-0.025, -0.03, -0.035, -0.04, -0.05])
            
            if "패닉 가속 (MA/RSI)" in targets:
                cand['panic_ma'] = random.choice([150, 180, 200, 220, 250])
                p_s1 = random.choice([25, 26, 27, 28, 29, 30])
                cand['panic_rsi_s1'] = p_s1
                if 'panic_buy_signals' in cand and len(cand['panic_buy_signals']) > 0:
                    cand['panic_buy_signals'][0]['rsi_val'] = p_s1
            
            if "S3 보호 조건 (Gap/Acc)" in targets:
                if len(cand['s3_protection']) > 0:
                    cand['s3_protection'][0]['gap_limit'] = random.choice([-2.5, -3.0, -3.5, -4.0, -5.0])
                if len(cand['s3_protection']) > 1:
                    cand['s3_protection'][1]['acc_limit'] = random.choice([-5.0, -7.0, -9.0, -12.0])

            score, cagr, mdd = self.run_simulation(cand, s_date, e_date, trade_at)
            if score > -900:
                results.append({'cfg': cand, 'score': score, 'cagr': cagr, 'mdd': mdd})
                if score > best_score:
                    best_score = score
            
            if progress_callback:
                progress_callback(i + 1, n_iter, best_score)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results, base_score, b_cagr, b_mdd
