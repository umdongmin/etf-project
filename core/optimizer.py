import copy
import random
import json
import datetime
import os
import pandas as pd
import optuna
import time
import threading
from core.engine import StrategyEngine
from utils.metrics import calculate_metrics
try:
    from streamlit.runtime.scriptrunner import get_script_run_context, add_script_run_context
except ImportError:
    # Older streamlit versions or non-streamlit environment
    get_script_run_context = lambda: None
    add_script_run_context = lambda thread, ctx: None

class StrategyOptimizer:
    """전략 파라미터 최적화를 수행하는 핵심 엔진 클래스 (Optuna + SQLite 학습형)"""
    
    def __init__(self, data_dict, fg_df, vxn_df, news_df=None):
        self.data_dict = data_dict
        self.fg_df = fg_df
        self.vxn_df = vxn_df
        self.news_df = news_df
        # 데이터베이스 저장소 설정 (Supabase 클라우드 전용)
        supabase_url = os.getenv("SUPABASE_DB_URL")

        if supabase_url:
            # Supabase는 PostgreSQL 기반이므로 바로 사용 가능
            self.db_path = supabase_url
        else:
            raise ValueError("Optimizing을 위해 SUPABASE_DB_URL 환경 변수가 필요합니다. (SQLite 레거시는 더 이상 지원되지 않습니다)")

    def run_simulation(self, cfg, s_date, e_date, trade_at, fast_mode=False):
        """단일 시뮬레이션 수행 및 상세 성과 지표 계산"""
        try:
            hist, _, _ = StrategyEngine.run_golden_strategy(
                self.data_dict, self.fg_df, self.vxn_df, self.news_df, 
                cfg['leverage_asset'], cfg['base_asset'], 
                0.0, s_date, e_date, cfg, trade_at, smart_params=cfg, fast_mode=fast_mode
            )
            if hist.empty: 
                return {'score': -999, 'cagr': 0, 'mdd': 0}
                
            m = calculate_metrics(hist)
            cagr, mdd = m['CAGR']*100, abs(m['MDD']*100)
            
            # [정형화 데이터 설계] 다양한 지표를 조합한 커스텀 스코어링 (Calmar Ratio 기반)
            score = cagr - 0.5 * mdd
            
            return {
                'score': score,
                'cagr': cagr,
                'mdd': mdd,
                'sharpe': m.get('Sharpe', 0),
                'pf': m.get('Profit_Factor', 0),
                'win_rate': m.get('Win_Rate', 0) * 100,
                'max_rec': m.get('Max_Recovery', 0)
            }
        except Exception as e:
            print(f"Simulation Error: {e}")
            return {'score': -999, 'cagr': 0, 'mdd': 0}

    def _get_ctx_tag(self, leverage, base, targets, wfa_ratio=0.0):
        """최적화 컨텍스트 고유 태그 생성 (오타 정규화 및 수치 정밀도 보정)Log 확보)"""
        # [정규화] "벨" 오타를 "밸"로 통일하여 매칭 확률 극대화
        clean_targets = [t.replace("벨", "밸") for t in targets]
        tag = f"{leverage}/{base}|{','.join(sorted(clean_targets))}"
        if wfa_ratio > 0:
            # 소수점 2자리까지 고정하여 문자열 불일치 방지
            tag += f"|WFA:{wfa_ratio:.2f}"
        return tag

    def run_monte_carlo(self, cfg, s_date, e_date, trade_at, n_sim=20):
        """가격 데이터에 노이즈를 섞어 전략의 생존력을 확률적으로 검증 (몬테카를로)"""
        results = []
        import numpy as np
        
        for _ in range(n_sim):
            # 1. 데이터에 미세한 노이즈(±0.1%) 추가
            noisy_data = {}
            for asset, df in self.data_dict.items():
                noisy_df = df.copy()
                # Close 가격에 노이즈 추가
                noise = np.random.normal(0, 0.001, len(noisy_df)) # 0.1% 표준편차
                noisy_df['Close'] *= (1 + noise)
                # Open 가격도 동기화 (간소화)
                if 'Open' in noisy_df.columns:
                    noisy_df['Open'] *= (1 + noise)
                noisy_data[asset] = noisy_df
            
            # 2. 노이즈 데이터로 시뮬레이션 수행
            try:
                hist, _, _ = StrategyEngine.run_golden_strategy(
                    noisy_data, self.fg_df, self.vxn_df, self.news_df, 
                    cfg['leverage_asset'], cfg['base_asset'], 
                    0.0, s_date, e_date, cfg, trade_at, smart_params=cfg
                )
                if not hist.empty:
                    m = calculate_metrics(hist)
                    results.append(m['CAGR'] * 100)
            except:
                continue
        
        if not results:
            return {'survival_rate': 0, 'avg_cagr': 0, 'low_cagr': 0}
            
        results.sort()
        survival_rate = len([r for r in results if r > 0]) / n_sim * 100
        return {
            'survival_rate': survival_rate,
            'avg_cagr': np.mean(results),
            'low_cagr': results[int(len(results) * 0.1)] # 하위 10% 성과
        }

    def optimize(self, base_cfg, n_iter, s_date, e_date, targets, trade_at, target_period="전체", progress_callback=None, wfa_ratio=0.0, n_jobs=2, is_async=False):
        """
        Optuna 기반 베이지안 최적화 수행
        :param wfa_ratio: 전진 분석 비율 (0.0~0.5). 0보다 크면 s_date~e_date를 IS/OOS로 분할함.
        """
        # [신규] Streamlit 세션 컨텍스트 캡처
        ctx = get_script_run_context()
        self._iter_lock = threading.Lock()
        
        # [중요] 기존 설정을 직접 수정하지 않도록 딥카피 수행 (사이드 이펙트 방지)
        base_cfg = copy.deepcopy(base_cfg)
        
        print(f"\n[DIAG] optimize() called with targets: {targets}")
        opt_keys = [k for k in base_cfg.keys() if k.startswith('opt_')]
        print(f"[DIAG] Top-level opt_ keys: {opt_keys}")
        for k in opt_keys:
            if base_cfg[k]: print(f"  - {k}: {base_cfg[k]}")
            
        for i, b in enumerate(base_cfg.get('buy_signals', [])):
            b_opts = [k for k in b.keys() if k.startswith('opt_')]
            if any(b.get(k) for k in b_opts):
                print(f"[DIAG] buy_signals[{i}] active opts: {[k for k in b_opts if b.get(k)]}")
        for i, s in enumerate(base_cfg.get('sell_signals', [])):
            s_opts = [k for k in s.keys() if k.startswith('opt_')]
            if any(s.get(k) for k in s_opts):
                print(f"[DIAG] sell_signals[{i}] active opts: {[k for k in s_opts if s.get(k)]}")
        
        # [데이터 분할] WFA 모드인 경우 날짜 구간 계산
        is_s_date, is_e_date = s_date, e_date
        oos_s_date, oos_e_date = None, None
        
        if wfa_ratio > 0:
            full_s = pd.to_datetime(s_date)
            full_e = pd.to_datetime(e_date)
            delta = full_e - full_s
            split_point = full_s + delta * (1 - wfa_ratio)
            
            is_e_date = split_point.strftime('%Y-%m-%d')
            oos_s_date = (split_point + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            oos_e_date = e_date
            print(f"WFA Split: IS({is_s_date}~{is_e_date}), OOS({oos_s_date}~{oos_e_date})")

        # [수정] 통합된 태그 생성 메소드 사용
        ctx_tag = self._get_ctx_tag(base_cfg['leverage_asset'], base_cfg['base_asset'], targets, wfa_ratio)
        
        # 1. 베이스라인 성과 측정 (IS 기준, 베이스라인은 정확한 비교를 위해 fast_mode 사용)
        base_res = self.run_simulation(base_cfg, is_s_date, is_e_date, trade_at, fast_mode=True)
        base_score = base_res['score']
        b_cagr, b_mdd = base_res['cagr'], base_res['mdd']
        
        # 2. Optuna 스터디 생성 또는 불러오기 (SQLite 연동)
        study = optuna.create_study(
            study_name="tqqq_strategy_optimization",
            storage=self.db_path,
            direction="maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42) # 학습형 샘플러 (TPE)
        )
        
        # [신규] 스레드 간 공유 상태 관리 (진동 및 생존 확인 강화)
        shared_state = {
            'study': study, 
            'current': 0,
            'best_v': -999.0,
            'done': False,
            'error': None,
            'results': None,
            'base_info': None,
            'heartbeat': time.time(), # [신규] 생존 확인용
            'stop_requested': False,   # [신규] 중단 요청 플래그
            'thread': None             # [신규] 스레드 객체 저장
        }

        # [신규] 진행률 계산을 위한 시작 카운트 기록
        start_count = len(study.trials)

        # 4. 베이스라인을 첫 번째 시도(Seed)로 주입 시도 (공공 기록 및 비교용)
        try:
            base_params = self._extract_params(base_cfg, targets)
            if base_params:
                study.enqueue_trial(base_params)
        except Exception as e:
            print(f"Error enqueuing base trial: {e}")

        # 3. 목표 함수 정의
        def objective(trial):
            # [필수] 중단 요청 확인
            if shared_state.get('stop_requested'):
                trial.study.stop()
                return -999
                
            # [필수] 개별 작업 스레드에 Streamlit 컨텍스트 주입
            if ctx:
                add_script_run_context(threading.current_thread(), ctx)
            
            # 하트비트 업데이트
            shared_state['heartbeat'] = time.time()
                
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
                    
                    # 수치 탐색 (베이스라인 포용을 위해 범위 확장)
                    cand['buy_signals'][i]['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", 15, 50)
                    if cand['buy_signals'][i]['use_willr']:
                        cand['buy_signals'][i]['willr_val'] = trial.suggest_int(f"{prefix}willr_val", -95, -60)
                    if cand['buy_signals'][i]['use_adx']:
                        cand['buy_signals'][i]['adx_val'] = trial.suggest_int(f"{prefix}adx_val", 10, 60)
                    
                    # [신규] AI 심리 필터 탐색 주입
                    cand['buy_signals'][i]['use_news_q'] = trial.suggest_categorical(f"{prefix}use_news_q", [True, False])
                    if cand['buy_signals'][i]['use_news_q']:
                        cand['buy_signals'][i]['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", 0.75, 0.98)
                    
                    cand['buy_signals'][i]['use_news_z'] = trial.suggest_categorical(f"{prefix}use_news_z", [True, False])
                    if cand['buy_signals'][i]['use_news_z']:
                        cand['buy_signals'][i]['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", 0.5, 3.0)
                    
                    # [신규] 200일 이격도 탐색
                    cand['buy_signals'][i]['use_dev200'] = trial.suggest_categorical(f"{prefix}use_dev200", [True, False])
                    if cand['buy_signals'][i]['use_dev200']:
                        cand['buy_signals'][i]['dev200_val'] = trial.suggest_float(f"{prefix}dev200_val", 0.8, 1.2)

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

                    # 수치 탐색 (베이스라인 포용을 위해 범위 확장)
                    cand['sell_signals'][i]['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", 50, 95)
                    if cand['sell_signals'][i]['use_chandelier']:
                        cand['sell_signals'][i]['chandelier_mult'] = trial.suggest_float(f"{prefix}chan_mult", 1.5, 6.0)
                    if cand['sell_signals'][i]['use_willr']:
                        cand['sell_signals'][i]['willr_val'] = trial.suggest_int(f"{prefix}willr_val", -40, -5)

                    # [신규] 매도 시그널 AI 뉴스 필터 (Q, Z) 추가
                    cand['sell_signals'][i]['use_news_q'] = trial.suggest_categorical(f"{prefix}use_news_q", [True, False])
                    if cand['sell_signals'][i]['use_news_q']:
                        # 매도는 분위수가 '낮을 때' (공포/과매도) 파는 것이 일반적이므로 범위를 낮게 설정
                        cand['sell_signals'][i]['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", 0.05, 0.40)
                    
                    cand['sell_signals'][i]['use_news_z'] = trial.suggest_categorical(f"{prefix}use_news_z", [True, False])
                    if cand['sell_signals'][i]['use_news_z']:
                        # Z점수가 매우 낮을 때 (비정상적 투매) 대응
                        cand['sell_signals'][i]['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", -2.5, -0.5)

                    # [신규] 200일 이격도 탐색
                    cand['sell_signals'][i]['use_dev200'] = trial.suggest_categorical(f"{prefix}use_dev200", [True, False])
                    if cand['sell_signals'][i]['use_dev200']:
                        cand['sell_signals'][i]['dev200_val'] = trial.suggest_float(f"{prefix}dev200_val", 0.9, 1.3)

            if "리밸런싱 및 ATR 변동성 (Full)" in targets: # 오타 수정: 벨 -> 밸
                # 고정 비율
                cand['use_fixed_reb'] = trial.suggest_categorical("use_fixed", [True, False])
                if cand['use_fixed_reb']:
                    cand['buy_reb_up'] = trial.suggest_float("buy_reb_up", 0.01, 0.05)
                    cand['buy_reb_down'] = trial.suggest_float("buy_reb_down", -0.07, -0.02)
                    cand['sell_reb_up'] = trial.suggest_float("sell_reb_up", 0.01, 0.05)
                    cand['sell_reb_down'] = trial.suggest_float("sell_reb_down", -0.05, -0.01)
                    
                    # [신규] 리밸런싱 인터락 최적화
                    cand['use_reb_interlock'] = trial.suggest_categorical("use_reb_int", [True, False])
                    if cand['use_reb_interlock']:
                        cand['reb_interlock_dev'] = trial.suggest_float("reb_int_dev", 1.05, 1.25)
                        cand['reb_interlock_vxn'] = trial.suggest_float("reb_int_vxn", 15.0, 35.0)

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

            if "사용자 지정 (🎯 타겟 최적화)" in targets or "Custom Targeted Optimization" in targets or "AI 지표 지능형 탐색 (구조+수치)" in targets:
                print(f"[DEBUG] Trial {trial.number} - Targets: {targets}")
                # 상위 레벨 opt_ 확인
                top_opts = {k: v for k, v in cand.items() if k.startswith('opt_') and v}
                if top_opts: print(f"  [DEBUG] Active top-level opts: {top_opts}")
                
                # 1. 매수 시그널 타겟팅
                for i in range(len(cand['buy_signals'])):
                    if i >= 3: break
                    p = cand['buy_signals'][i]
                    prefix = f"custom_b{i+1}_"
                    
                    if p.get('opt_rsi_val'):
                        r = p.get('rng_rsi_val', [10, 60])
                        p['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}rsi_val: {p['rsi_val']}")
                    if p.get('opt_adx_val') and p.get('use_adx'):
                        r = p.get('rng_adx_val', [5, 80])
                        p['adx_val'] = trial.suggest_int(f"{prefix}adx_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}adx_val: {p['adx_val']}")
                    if p.get('opt_willr_val') and p.get('use_willr'):
                        r = p.get('rng_willr_val', [-95, -50])
                        p['willr_val'] = trial.suggest_int(f"{prefix}willr_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}willr_val: {p['willr_val']}")
                    # [2단계: Discovery] 구조적 최적화 (AI 필터 사용 여부 자체를 탐색)
                    if "AI 지표 지능형 탐색 (구조+수치)" in targets:
                        # Q 지표 구조 탐색
                        is_use_q = trial.suggest_categorical(f"{prefix}use_news_q", [True, False])
                        p['use_news_q'] = is_use_q
                        if is_use_q:
                            r = p.get('rng_news_q_val', [0.5, 0.99])
                            p['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", float(r[0]), float(r[1]))
                            print(f"  [SUGGEST][T{trial.number}] {prefix}news_q_val: {p['news_q_val']}")
                        
                        # Z 지표 구조 탐색
                        is_use_z = trial.suggest_categorical(f"{prefix}use_news_z", [True, False])
                        p['use_news_z'] = is_use_z
                        if is_use_z:
                            r = p.get('rng_news_z_val', [-1.0, 4.0])
                            p['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", float(r[0]), float(r[1]))
                            print(f"  [SUGGEST][T{trial.number}] {prefix}news_z_val: {p['news_z_val']}")
                    
                    # [일반: Tuning/Manual] 기존 타겟 최격화 (UI 또는 지정된 슬롯만 강제 활성화)
                    elif p.get('opt_news_q'):
                        p['use_news_q'] = True # Force enable if optimizing
                        r = p.get('rng_news_q_val', [0.5, 0.99])
                        p['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}news_q_val: {p['news_q_val']}")
                    
                    if not ("AI 지표 지능형 탐색 (구조+수치)" in targets) and p.get('opt_news_z'):
                        p['use_news_z'] = True # Force enable if optimizing
                        r = p.get('rng_news_z_val', [-1.0, 4.0])
                        p['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}news_z_val: {p['news_z_val']}")
                    
                    if p.get('opt_dev200_val'):
                        p['use_dev200'] = True
                        r = p.get('rng_dev200_val', [0.8, 1.3])
                        p['dev200_val'] = trial.suggest_float(f"{prefix}dev200_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}dev200_val: {p['dev200_val']}")

                # 2. 매도 시그널 타겟팅
                for i in range(len(cand['sell_signals'])):
                    if i >= 4: break
                    p = cand['sell_signals'][i]
                    prefix = f"custom_s{i+1}_"
                    
                    if p.get('opt_rsi_val'):
                        r = p.get('rng_rsi_val', [40, 95])
                        p['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}rsi_val: {p['rsi_val']}")
                    if p.get('opt_chandelier_mult') and p.get('use_chandelier'):
                        r = p.get('rng_chandelier_mult', [1.0, 5.0])
                        p['chandelier_mult'] = trial.suggest_float(f"{prefix}chan_mult", float(r[0]), float(r[1]))
                    if p.get('opt_willr_val') and p.get('use_willr'):
                        r = p.get('rng_willr_val', [-50, -5])
                        p['willr_val'] = trial.suggest_int(f"{prefix}willr_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}willr_val: {p['willr_val']}")
                    # [2단계: Discovery] 구조적 최적화
                    if "AI 지표 지능형 탐색 (구조+수치)" in targets:
                        # Q 지표 구조 탐색
                        is_use_q = trial.suggest_categorical(f"{prefix}use_news_q", [True, False])
                        p['use_news_q'] = is_use_q
                        if is_use_q:
                            r = p.get('rng_news_q_val', [0.01, 0.5])
                            p['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", float(r[0]), float(r[1]))
                            print(f"  [SUGGEST][T{trial.number}] {prefix}news_q_val: {p['news_q_val']}")
                        
                        # Z 지표 구조 탐색
                        is_use_z = trial.suggest_categorical(f"{prefix}use_news_z", [True, False])
                        p['use_news_z'] = is_use_z
                        if is_use_z:
                            r = p.get('rng_news_z_val', [-4.0, 1.0])
                            p['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", float(r[0]), float(r[1]))
                            print(f"  [SUGGEST][T{trial.number}] {prefix}news_z_val: {p['news_z_val']}")

                    elif p.get('opt_news_q'):
                        p['use_news_q'] = True # Force enable if optimizing
                        r = p.get('rng_news_q_val', [0.01, 0.5])
                        p['news_q_val'] = trial.suggest_float(f"{prefix}news_q_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}news_q_val: {p['news_q_val']}")
                    
                    if not ("AI 지표 지능형 탐색 (구조+수치)" in targets) and p.get('opt_news_z'):
                        p['use_news_z'] = True # Force enable if optimizing
                        r = p.get('rng_news_z_val', [-4.0, 1.0])
                        p['news_z_val'] = trial.suggest_float(f"{prefix}news_z_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}news_z_val: {p['news_z_val']}")

                    if p.get('opt_dev200_val'):
                        p['use_dev200'] = True
                        r = p.get('rng_dev200_val', [0.8, 1.3])
                        p['dev200_val'] = trial.suggest_float(f"{prefix}dev200_val", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}dev200_val: {p['dev200_val']}")

                # 3. 리밸런싱 타겟팅
                if cand.get('use_fixed_reb'):
                    if cand.get('opt_buy_reb_up'):
                        r = cand.get('rng_buy_reb_up', [0.1, 5.0])
                        cand['buy_reb_up'] = trial.suggest_float("custom_buy_reb_up", r[0], r[1], step=0.1) / 100.0
                        print(f"  [SUGGEST][T{trial.number}] buy_reb_up: {cand['buy_reb_up'] * 100.0}%")
                    if cand.get('opt_buy_reb_down'):
                        r = cand.get('rng_buy_reb_down', [-15.0, -1.0])
                        cand['buy_reb_down'] = trial.suggest_float("custom_buy_reb_down", r[0], r[1], step=0.1) / 100.0
                        print(f"  [SUGGEST][T{trial.number}] buy_reb_down: {cand['buy_reb_down'] * 100.0}%")
                    if cand.get('opt_sell_reb_up'):
                        r = cand.get('rng_sell_reb_up', [0.5, 10.0])
                        cand['sell_reb_up'] = trial.suggest_float("custom_sell_reb_up", r[0], r[1], step=0.1) / 100.0
                        print(f"  [SUGGEST][T{trial.number}] sell_reb_up: {cand['sell_reb_up'] * 100.0}%")
                    if cand.get('opt_sell_reb_down'):
                        r = cand.get('rng_sell_reb_down', [-10.0, -0.5])
                        cand['sell_reb_down'] = trial.suggest_float("custom_sell_reb_down", r[0], r[1], step=0.1) / 100.0
                        print(f"  [SUGGEST][T{trial.number}] sell_reb_down: {cand['sell_reb_down'] * 100.0}%")
                
                if cand.get('use_atr_reb'):
                    if cand.get('opt_atr_m_buy_up'):
                        r = cand.get('rng_atr_m_buy_up', [1.0, 20.0])
                        cand['atr_mult_buy_up'] = trial.suggest_float("custom_atr_m_buy_up", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] atr_mult_buy_up: {cand['atr_mult_buy_up']}")
                    if cand.get('opt_atr_m_buy_down'):
                        r = cand.get('rng_atr_m_buy_down', [0.5, 5.0])
                        cand['atr_mult_buy_down'] = trial.suggest_float("custom_atr_m_buy_down", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] atr_mult_buy_down: {cand['atr_mult_buy_down']}")
                    if cand.get('opt_atr_m_sell'):
                        r = cand.get('rng_atr_m_sell', [1.0, 6.0])
                        cand['atr_mult_sell'] = trial.suggest_float("custom_atr_m_sell", float(r[0]), float(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] atr_mult_sell: {cand['atr_mult_sell']}")

                # 4. 하락장 매수 지연 (Panic Buy Signals) 타겟팅
                for i in range(len(cand.get('panic_buy_signals', []))):
                    if i >= 3: break
                    p = cand['panic_buy_signals'][i]
                    prefix = f"custom_pb{i+1}_"
                    if p.get('opt_rsi_val'):
                        r = p.get('rng_rsi_val', [15, 50])
                        p['rsi_val'] = trial.suggest_int(f"{prefix}rsi_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}rsi_val: {p['rsi_val']}")
                    if p.get('opt_adx_val') and p.get('use_adx'):
                        r = p.get('rng_adx_val', [5, 80])
                        p['adx_val'] = trial.suggest_int(f"{prefix}adx_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}adx_val: {p['adx_val']}")
                    if p.get('opt_willr_val') and p.get('use_willr'):
                        r = p.get('rng_willr_val', [-95, -50])
                        p['willr_val'] = trial.suggest_int(f"{prefix}willr_val", int(r[0]), int(r[1]))
                        print(f"  [SUGGEST][T{trial.number}] {prefix}willr_val: {p['willr_val']}")

                # 5. 동적 현금 비중 (Dynamic Cash) 타겟팅
                dc = cand.get('dynamic_cash', {})
                if dc.get('use'):
                    if dc.get('opt_bull_vxn'):
                        r = dc.get('rng_bull_vxn', [10.0, 25.0])
                        dc['bull_vxn'] = trial.suggest_float("dc_bull_vxn", float(r[0]), float(r[1]), step=0.5)
                    if dc.get('opt_caution_vxn'):
                        r = dc.get('rng_caution_vxn', [25.0, 40.0])
                        dc['caution_vxn'] = trial.suggest_float("dc_caution_vxn", float(r[0]), float(r[1]), step=0.5)
                    if dc.get('opt_bear_vxn'):
                        r = dc.get('rng_bear_vxn', [35.0, 50.0])
                        dc['bear_vxn'] = trial.suggest_float("dc_bear_vxn", float(r[0]), float(r[1]), step=0.5)
                    if dc.get('opt_ma200_buffer'):
                        r = dc.get('rng_ma200_buffer', [0.0, 5.0])
                        dc['ma200_buffer'] = trial.suggest_float("dc_ma200_buffer", float(r[0]), float(r[1]), step=0.1)
                    
                    if dc.get('opt_cash_bull'):
                        r = dc.get('rng_cash_bull', [0.0, 30.0])
                        dc['cash_bull'] = trial.suggest_float("dc_cash_bull", float(r[0]), float(r[1]), step=1.0)
                    if dc.get('opt_cash_neutral'):
                        r = dc.get('rng_cash_neutral', [0.0, 60.0])
                        dc['cash_neutral'] = trial.suggest_float("dc_cash_neutral", float(r[0]), float(r[1]), step=1.0)
                    if dc.get('opt_cash_caution'):
                        r = dc.get('rng_cash_caution', [0.0, 90.0])
                        dc['cash_caution'] = trial.suggest_float("dc_cash_caution", float(r[0]), float(r[1]), step=1.0)
                    if dc.get('opt_cash_bear'):
                        r = dc.get('rng_cash_bear', [0.0, 100.0])
                        dc['cash_bear'] = trial.suggest_float("dc_cash_bear", float(r[0]), float(r[1]), step=1.0)
            
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

            # [Phase 2] 데이터 정형화: Trial 결과 상세 지표 산출 및 DB 저장
            # 훈련(IS) 구간 성과 측정 (빠른 탐색을 위해 fast_mode=True 적용)
            res = self.run_simulation(cand, is_s_date, is_e_date, trade_at, fast_mode=True)
            
            # [WFA] 검증(OOS) 구간 성과는 최적화 완료 후 Top 결과에 대해서만 수행하거나, 
            # 혹은 실시간으로 기록할지 결정. (여기서는 실시간 기록으로 설계)
            oos_res = None
            if wfa_ratio > 0:
                oos_res = self.run_simulation(cand, oos_s_date, oos_e_date, trade_at, fast_mode=True)
                # WFE (Walk Forward Efficiency) 계산: OOS CAGR / IS CAGR
                wfe = (oos_res['cagr'] / res['cagr']) if res['cagr'] > 0 else 0
                trial.set_user_attr("oos_cagr", oos_res['cagr'])
            # SQLite DB(Optuna)에 영구 기록 (Datafication)
            # [수정] 대시보드 필터링을 위한 기간 및 컨텍스트 정보 추가 저장
            # "전체 (2010~현재)" -> "전체", "최근 5년" -> "최근5년" 등으로 변별력 확보
            period_label = target_period.replace(" ", "").split("(")[0] 
            
            trial.set_user_attr("period", period_label)
            trial.set_user_attr("ctx_tag", ctx_tag)
            # 컨텍스트 저장
            trial.set_user_attr("cagr", res['cagr'])
            trial.set_user_attr("mdd", res['mdd'])
            trial.set_user_attr("sharpe", res.get('sharpe', 0))
            trial.set_user_attr("profit_factor", res.get('pf', 0))
            trial.set_user_attr("win_rate", res.get('win_rate', 0))
            trial.set_user_attr("max_recovery", res.get('max_rec', 0))
            if wfa_ratio > 0:
                trial.set_user_attr("wfa_mode", True)
            
            score = res['score']
            
            # [신규] 시행 결과 로그 출력 (사용자 요청: 최적 값을 찾지 못해도 점수 표시)
            print(f"  [RESULT][T{trial.number}] Score: {score:.2f} | CAGR: {res['cagr']:.1f}% | MDD: {res['mdd']:.1f}%")
            
            return score

        # [수정] 메인 스레드 폴링 방식으로 진행률 표시 (NoSessionContext 완전 방지)
        def opt_callback(study, trial):
            with self._iter_lock:
                shared_state['current'] += 1
                shared_state['heartbeat'] = time.time()
                try:
                    shared_state['best_v'] = study.best_value
                except:
                    shared_state['best_v'] = trial.value if trial.value is not None else -999.0

        def run_optimization_task():
            try:
                if ctx:
                    add_script_run_context(threading.current_thread(), ctx)
                
                # [중요] 최적화 실행
                study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs, callbacks=[opt_callback])
                
                if shared_state['stop_requested']:
                    return

                # 완료 후 결과 정리
                valid_t = [trial for trial in study.trials if trial.value is not None and trial.value > -900]
                # ... (이하 결과 정리 로직은 동일)
                p_f = target_period.split(' ')[0]
                f_trials = []
                for trial in valid_t:
                    t_p = trial.user_attrs.get("period")
                    t_c = trial.user_attrs.get("ctx_tag")
                    p_match = (p_f == "전체" and t_p in ["전체", None]) or (t_p == p_f)
                    c_match = (t_c == ctx_tag)
                    if p_match and c_match:
                        f_trials.append(trial)
                f_trials.sort(key=lambda x: x.value, reverse=True)
                
                res_list = []
                for trial in f_trials[:10]:
                    res_list.append(self._reconstruct_config(base_cfg, trial, s_date, e_date, trade_at, is_s_date, is_e_date, oos_s_date, oos_e_date))
                
                shared_state['results'] = res_list
                shared_state['base_info'] = (base_score, b_cagr, b_mdd)

            except Exception as e:
                shared_state['error'] = e
            finally:
                shared_state['done'] = True

        # 백그라운드에서 최적화 시작
        opt_thread = threading.Thread(target=run_optimization_task, daemon=True)
        shared_state['thread'] = opt_thread # 레퍼런스 유지
        opt_thread.start()

        # [수정] 비동기 모드일 경우 즉시 상태 객체 반환
        if is_async:
            return shared_state

        # 메인 스레드에서 UI 업데이트 폴링 (동기 모드 호환성 유지)
        while not shared_state['done']:
            if progress_callback:
                progress_callback(shared_state['current'], n_iter, shared_state['best_v'])
            time.sleep(0.1) # 0.1초 간격으로 UI 업데이트

        # 최적화 중 에러 발생 시 전파
        if shared_state['error']:
            raise shared_state['error']

        return shared_state['results'], base_score, b_cagr, b_mdd

        return shared_state['results'], base_score, b_cagr, b_mdd

    def get_top_results(self, base_cfg, n_top=5, s_date=None, e_date=None, trade_at='종가', target_period=None, targets=None, wfa_ratio=0.0):
        """DB에서 현재까지의 상위 성과를 추출 (탐색 중단/시작 전 용도)"""
        try:
            study = optuna.load_study(study_name="tqqq_strategy_optimization", storage=self.db_path)
            
            # [수정] 기간 + 컨텍스트 필터링 로직 추가
            valid_trials = [t for t in study.trials if t.value is not None and t.value > -900]
            
            # 컨텍스트 태그 생성 (targets가 제공된 경우에만 강화된 필터링 수행)
            ctx_tag = None
            if targets:
                ctx_tag = self._get_ctx_tag(base_cfg['leverage_asset'], base_cfg['base_asset'], targets, wfa_ratio)
            
            filtered_trials = []
            if target_period:
                # [수정] 동일한 레이블 생성 로직 적용
                p_filter = target_period.replace(" ", "").split("(")[0]
                for t in valid_trials:
                    t_period = t.user_attrs.get("period")
                    t_ctx = t.user_attrs.get("ctx_tag")
                    
                    # 레거시 호환: 이전 "최근" 태그도 허용 (단, "최근"일 때는 기간 매칭 완화)
                    period_match = (p_filter == "전체" and t_period in ["전체", None]) or \
                                   (t_period == p_filter) or \
                                   (p_filter.startswith("최근") and t_period == "최근")
                    # 컨텍스트 정규화 비교 (과거 오타 데이터 포함)
                    t_ctx_norm = t_ctx.replace("벨", "밸") if t_ctx else None
                    ctx_tag_norm = ctx_tag.replace("벨", "밸") if ctx_tag else None
                    
                    ctx_match = True if not ctx_tag else (t_ctx_norm == ctx_tag_norm)
                    
                    if period_match and ctx_match:
                        filtered_trials.append(t)
            else:
                filtered_trials = valid_trials
                
            filtered_trials.sort(key=lambda x: x.value, reverse=True)
            
            results = []
            for t in filtered_trials[:n_top]:
                results.append(self._reconstruct_config(base_cfg, t, s_date, e_date, trade_at))
            return results
        except Exception as e:
            print(f"Error getting top results: {e}")
            return []

    def _reconstruct_config(self, base_cfg, trial, s_date=None, e_date=None, trade_at='종가', is_s=None, is_e=None, oos_s=None, oos_e=None):
        """Trial 정보를 바탕으로 전체 전략 설정 복원 및 정밀 재시뮬레이션"""
        import copy
        cfg = copy.deepcopy(base_cfg)
        p = trial.params
        
        for key, val in p.items():
            # 1. 매수 신호 (b1_, b2_, ...)
            if key.startswith('b') and '_' in key and key[1].isdigit():
                parts = key.split('_')
                idx = int(parts[0][1:]) - 1
                field = "_".join(parts[1:])
                cfg['buy_signals'][idx][field] = val
                
            # 2. S3 보호 조건 (s3_1_, s3_2_, ...) 
            # 이 키는 s3_로 시작하고 두 번째 파트가 숫자여야 함 (예: s3_1_drop)
            elif key.startswith('s3_') and len(key.split('_')) >= 3 and key.split('_')[1].isdigit():
                parts = key.split('_')
                idx, field = int(parts[1]) - 1, parts[2]
                field_map = {'drop':'use_daily_drop','ma60':'use_ma60','ma200':'use_ma200','vxn':'use_vxn_jump','gap':'use_gap_down','acc':'use_drop_acc'}
                cfg['s3_protection'][idx][field_map[field]] = val
                
            # 3. 매도 신호 (s1_, s2_, s3_, ...)
            # s3_ 보호 조건에 걸리지 않은 s로 시작하는 시그널 키들
            elif key.startswith('s') and '_' in key and key[1].isdigit():
                parts = key.split('_')
                idx = int(parts[0][1:]) - 1
                field = "_".join(parts[1:])
                # [매핑 오류 수정] UI에서 사용하는 단축키와 실제 설정 키 매칭
                if field == "chan_mult": field = "chandelier_mult"
                if field == "use_chan": field = "use_chandelier"
                cfg['sell_signals'][idx][field] = val
            
            # 4. 기타 공통 변수 (p_rsi_s1, sl_ctrl_lim, 등)
            else:
                map_dict = {
                    'p_rsi_s1':'panic_rsi_s1',
                    'p_rsi_s2':'panic_rsi_s2',
                    'p_rsi_s3':'panic_rsi_s3',
                    'sl_ctrl_lim':'sl_control_limit',
                    'use_sl_ctrl':'use_sl_control',
                    'use_vxn':'use_vxn_safety',
                    'set_sl_limit':'set_sl_limit',
                    'use_set_sl':'use_set_sl',
                    # 동적 현금 비중 매핑
                    'dc_bull_vxn': 'bull_vxn',
                    'dc_caution_vxn': 'caution_vxn',
                    'dc_bear_vxn': 'bear_vxn',
                    'dc_ma200_buffer': 'ma200_buffer',
                    'dc_cash_bull': 'cash_bull',
                    'dc_cash_neutral': 'cash_neutral',
                    'dc_cash_caution': 'cash_caution',
                    'dc_cash_bear': 'cash_bear'
                }
                if key.startswith('dc_') and 'dynamic_cash' in cfg:
                    field = map_dict.get(key, key)
                    cfg['dynamic_cash'][field] = val
                else:
                    map_dict.get(key, key)
                    # 리밸런싱 인터락 매핑 추가
                    if key == 'use_reb_int': key = 'use_reb_interlock'
                    if key == 'reb_int_dev': key = 'reb_interlock_dev'
                    if key == 'reb_int_vxn': key = 'reb_interlock_vxn'
                    cfg[map_dict.get(key, key)] = val
        
        if 'panic_rsi_s1' in cfg and 'panic_buy_signals' in cfg:
            for i, rsi_val in enumerate([cfg.get('panic_rsi_s1'), cfg.get('panic_rsi_s2'), cfg.get('panic_rsi_s3')]):
                if i < len(cfg['panic_buy_signals']):
                    cfg['panic_buy_signals'][i]['rsi_val'] = rsi_val

        # 정밀 재시뮬레이션 수행
        res = self.run_simulation(cfg, s_date, e_date, trade_at)
        
        # [WFA] 추가 정보가 있다면 기록
        final_res = {
            'cfg': cfg,
            'score': res['score'],
            'cagr': trial.user_attrs.get('cagr', res['cagr']),
            'mdd': trial.user_attrs.get('mdd', res['mdd']),
            'sharpe': trial.user_attrs.get('sharpe', res.get('sharpe', 0)),
            'pf': trial.user_attrs.get('profit_factor', res.get('pf', 0)),
            'win_rate': trial.user_attrs.get('win_rate', res.get('win_rate', 0)),
            'max_rec': trial.user_attrs.get('max_recovery', res.get('max_rec', 0)),
            # WFA 관련 지표
            'oos_cagr': trial.user_attrs.get('oos_cagr', 0),
            'oos_mdd': trial.user_attrs.get('oos_mdd', 0),
            'wfe': trial.user_attrs.get('wfe', 0),
            'wfa_mode': trial.user_attrs.get('wfa_mode', False),
            # MC 관련 (필요 시 복원)
            'mc': trial.user_attrs.get('mc_survival', 0)
        }
        return final_res

    def get_total_trials(self):
        """축적된 총 분석 횟수 반환"""
        try:
            study = optuna.load_study(study_name="tqqq_strategy_optimization", storage=self.db_path)
            return len(study.trials)
        except:
            return 0
    def _extract_params(self, cfg, targets):
        """설정 딕셔너리에서 Optuna 파라미터 형식으로 추출"""
        params = {}
        
        if "매수 시그널 (Full)" in targets:
            for i, sig in enumerate(cfg['buy_signals'][:3]):
                p = f"b{i+1}_"
                params[f"{p}rsi_cross"] = sig['rsi_cross']
                params[f"{p}rsi_inc"] = sig['rsi_inc']
                params[f"{p}macd_inc"] = sig['macd_inc']
                params[f"{p}macd_signal_below"] = sig['macd_signal_below']
                params[f"{p}macd_golden"] = sig['macd_golden']
                params[f"{p}use_adx"] = sig['use_adx']
                params[f"{p}use_willr"] = sig['use_willr']
                params[f"{p}di_plus_above"] = sig['di_plus_above']
                params[f"{p}di_plus_cross"] = sig['di_plus_cross']
                params[f"{p}use_sar"] = sig['use_sar']
                params[f"{p}rsi_val"] = sig['rsi_val']
                if sig.get('use_willr'): params[f"{p}willr_val"] = sig['willr_val']
                if sig.get('use_adx'): params[f"{p}adx_val"] = sig['adx_val']
                
                # AI 필터 파라미터 추출 추가
                params[f"{p}use_news_q"] = sig.get('use_news_q', False)
                if sig.get('use_news_q'): params[f"{p}news_q_val"] = sig.get('news_q_val', 0.9)
                if sig.get('use_news_z'): params[f"{p}news_z_val"] = sig.get('news_z_val', 1.0)
                
                # [신규] 200일 이격도 추출
                params[f"{p}use_dev200"] = sig.get('use_dev200', False)
                if sig.get('use_dev200'): params[f"{p}dev200_val"] = sig.get('dev200_val', 1.0)

        if "매도 시그널 (Full)" in targets:
            for i, sig in enumerate(cfg['sell_signals'][:4]):
                p = f"s{i+1}_"
                params[f"{p}rsi_dead"] = sig['rsi_dead']
                params[f"{p}rsi_dec"] = sig['rsi_dec']
                params[f"{p}macd_dec"] = sig['macd_dec']
                params[f"{p}macd_signal_above"] = sig['macd_signal_above']
                params[f"{p}macd_dead"] = sig['macd_dead']
                params[f"{p}di_minus_above"] = sig['di_minus_above']
                params[f"{p}di_minus_cross"] = sig['di_minus_cross']
                params[f"{p}use_chan"] = sig['use_chandelier']
                params[f"{p}use_sar"] = sig['use_sar']
                params[f"{p}use_willr"] = sig['use_willr']
                params[f"{p}rsi_val"] = sig['rsi_val']
                if sig['use_willr']: params[f"{p}willr_val"] = sig['willr_val']
                
                # [신규] 200일 이격도 추출
                params[f"{p}use_dev200"] = sig.get('use_dev200', False)
                if sig.get('use_dev200'): params[f"{p}dev200_val"] = sig.get('dev200_val', 1.0)

        if "리밸런싱 및 ATR 변동성 (Full)" in targets: # 오타 수정: 벨 -> 밸
            params["use_fixed"] = cfg['use_fixed_reb']
            if cfg['use_fixed_reb']:
                params["buy_reb_up"] = cfg['buy_reb_up']
                params["buy_reb_down"] = cfg['buy_reb_down']
                params["sell_reb_up"] = cfg['sell_reb_up']
                params["sell_reb_down"] = cfg['sell_reb_down']
                # 리밸런싱 인터락 초기치 추출
                params["use_reb_int"] = cfg.get('use_reb_interlock', False)
                if cfg.get('use_reb_interlock'):
                    params["reb_int_dev"] = cfg.get('reb_interlock_dev', 1.13)
                    params["reb_int_vxn"] = cfg.get('reb_interlock_vxn', 22.0)
            params["use_atr"] = cfg['use_atr_reb']
            if cfg['use_atr_reb']:
                params["atr_p_buy"] = cfg['atr_period_buy']
                params["atr_p_sell"] = cfg['atr_period_sell']
                params["atr_m_buy_up"] = cfg['atr_mult_buy_up']
                params["atr_m_buy_down"] = cfg['atr_mult_buy_down']
                params["atr_m_sell"] = cfg['atr_mult_sell']

        if "패닉 가속 (MA/RSI)" in targets:
            params["use_panic"] = cfg['use_panic']
            if cfg['use_panic']:
                params["panic_ma"] = cfg['panic_ma']
                params["p_rsi_s1"] = cfg['panic_rsi_s1']
                params["p_rsi_s2"] = cfg['panic_rsi_s2']
                params["p_rsi_s3"] = cfg['panic_rsi_s3']

        if "S3 보호 조건 (Gap/Acc)" in targets:
            for i, prot in enumerate(cfg['s3_protection']):
                p = f"s3_{i+1}_"
                params[f"{p}drop"] = prot['use_daily_drop']
                params[f"{p}ma60"] = prot['use_ma60']
                params[f"{p}ma200"] = prot['use_ma200']
                params[f"{p}vxn"] = prot['use_vxn_jump']
                params[f"{p}gap"] = prot['use_gap_down']
                params[f"{p}acc"] = prot['use_drop_acc']

        if "방어 및 대피 시스템 (VXN/손절)" in targets:
            params["use_vxn"] = cfg['use_vxn_safety']
            params["vxn_exit"] = cfg['vxn_exit']
            params["use_sl_ctrl"] = cfg['use_sl_control']
            params["sl_ctrl_lim"] = cfg['sl_control_limit']
            params["use_set_sl"] = cfg['use_set_sl']
            params["set_sl_lim"] = cfg['set_sl_limit']

        # [신규] 동적 현금 비중 파라미터 추출
        if 'dynamic_cash' in cfg:
            dc = cfg['dynamic_cash']
            if dc.get('use'):
                if dc.get('opt_bull_vxn'): params["dc_bull_vxn"] = dc.get('bull_vxn', 18.0)
                if dc.get('opt_caution_vxn'): params["dc_caution_vxn"] = dc.get('caution_vxn', 35.0)
                if dc.get('opt_bear_vxn'): params["dc_bear_vxn"] = dc.get('bear_vxn', 40.0)
                if dc.get('opt_ma200_buffer'): params["dc_ma200_buffer"] = dc.get('ma200_buffer', 0.0)
                if dc.get('opt_cash_bull'): params["dc_cash_bull"] = dc.get('cash_bull', 0.0)
                if dc.get('opt_cash_neutral'): params["dc_cash_neutral"] = dc.get('cash_neutral', 20.0)
                if dc.get('opt_cash_caution'): params["dc_cash_caution"] = dc.get('cash_caution', 30.0)
                if dc.get('opt_cash_bear'): params["dc_cash_bear"] = dc.get('cash_bear', 80.0)

        return params
