import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import datetime
import math

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @st.cache_data(ttl=3600, show_spinner=False)
    @staticmethod
    def run_golden_strategy(data_dict, fg_df, vxn_df, news_df=None, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0.0, start_date=None, end_date=None, params=None, trade_at="종가", smart_params=None, salt=None, fast_mode=False, allocated_capital=10000.0, fee_rate=0.0, integer_shares=False):
        gen = StrategyEngine._run_generator(
            data_dict, fg_df, vxn_df, news_df, leverage_asset, base_asset, cash_ratio,
            start_date, end_date, params, trade_at, smart_params, salt, fast_mode, allocated_capital, fee_rate, integer_shares
        )
        try:
            state = next(gen)
            while True:
                state = gen.send(None)
        except StopIteration as e:
            return e.value

    @staticmethod
    def _run_generator(data_dict, fg_df, vxn_df, news_df=None, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0.0, start_date=None, end_date=None, params=None, trade_at="종가", smart_params=None, salt=None, fast_mode=False, allocated_capital=10000.0, fee_rate=0.0, integer_shares=False):
        if base_asset not in data_dict: return pd.DataFrame(), pd.DataFrame(), []
        
        # [신규] 파라미터 정규화 (JSON 로드 시 중첩된 params/smart_params 제거 및 최상위 병합)
        if params is not None:
            params = params.copy()
            # 1. 중첩된 'params'가 있으면 최상위로 덮어씀 (레거시 호환 및 명시적 설정 우선)
            if 'params' in params and isinstance(params['params'], dict):
                inner_p = params.pop('params')
                params.update(inner_p)
            
            # 2. 중첩된 'smart_params'가 있으면 최상위로 덮어씀 (일관성 확보)
            if 'smart_params' in params and isinstance(params['smart_params'], dict):
                inner_s = params.pop('smart_params')
                params.update(inner_s)
        
        base_df = data_dict[base_asset]
        # 필요한 컬럼만 선택 (존재하지 않으면 스킵)
        cols_to_use = [col for col in ['Close', 'RSI'] if col in base_df.columns]
        if not cols_to_use:
            # 기본값으로 모든 컬럼 사용
            combined = base_df.copy()
        else:
            combined = base_df[cols_to_use].copy()
        
        # [신규] 전역 날짜 동기화 - 데이터가 없는 기간은 초기 자본보존을 위해 Reindex
        if start_date or end_date:
            # 기준 인덱스 생성 (주말 제외 평일 기준 또는 데이터셋의 합집합)
            # 여기서는 편의상 base_df의 전체 인덱스를 사용하되, 범위만 제한함
            full_idx = base_df.index
            if start_date:
                if isinstance(start_date, (str, datetime.date)):
                    sd = pd.Timestamp(start_date)
                    full_idx = full_idx[full_idx >= sd]
            if end_date:
                if isinstance(end_date, (str, datetime.date)):
                    ed = pd.Timestamp(end_date)
                    full_idx = full_idx[full_idx <= ed]
            
            # Reindex하여 빈 날짜(데이터 시작 전) 확보
            combined = combined.reindex(full_idx)
        
        # [추가] 시가 데이터 확보
        if 'Open' in base_df.columns:
            combined['Open'] = base_df['Open']
        elif 'Close' in combined.columns:
            combined['Open'] = combined['Close'] # 시가 없으면 종가 대체
        
        fg_clean = fg_df[~fg_df.index.duplicated(keep='first')]
        vxn_clean = vxn_df[~vxn_df.index.duplicated(keep='first')]
        
        # [수정] 컬럼 존재 여부 체크 후 안정적으로 병합
        if not fg_clean.empty and 'FearGreed' in fg_clean.columns:
            combined = combined.join(fg_clean['FearGreed'].rename('FG'), how='left')
        
        if not vxn_clean.empty:
            # [수정] 인덱스 정렬 및 reindex를 통한 안정적 데이터 매칭
            vxn_clean = vxn_clean.sort_index()
            for col in ['VIX', 'VXN', 'VVIX']:
                if col in vxn_clean.columns:
                    # 데이터가 있는 날짜만 가져오고 빈 자리는 전방 채우기(ffill)
                    combined[col] = vxn_clean[col].reindex(combined.index, method='ffill')

        # [신규] AI 뉴스 심리 데이터 병합 (분위수 및 Z-Score 포함)
        if news_df is not None and not news_df.empty:
            news_clean = news_df[~news_df.index.duplicated(keep='first')].sort_index()
            # 원본 점수와 가공된 점수(q_score, z_score) 모두 reindex
            for col in ['sentiment', 'impact', 'q_score', 'z_score']:
                if col in news_clean.columns:
                    combined[f'News_{col.capitalize().replace("_score", "")}'] = news_clean[col].reindex(combined.index, method='ffill')
        
        # [추가] 필수 컬럼 부재 시 기본값으로 생성 (KeyError 방지)
        if 'FG' not in combined.columns: combined['FG'] = 50.0
        if 'VXN' not in combined.columns: combined['VXN'] = 20.0
        if 'VIX' not in combined.columns: combined['VIX'] = combined['VXN'] # 호환성 유지용 VIX
        if 'ADX' not in combined.columns: combined['ADX'] = 0.0
        if 'DMP' not in combined.columns: combined['DMP'] = 0.0
        if 'DMN' not in combined.columns: combined['DMN'] = 0.0
        if 'SAR' not in combined.columns: combined['SAR'] = 0.0
        # [신규] 뉴스 지표 기본값 (중립/평균)
        if 'News_Impact' not in combined.columns: combined['News_Impact'] = 0.1
        if 'News_Q' not in combined.columns: combined['News_Q'] = 0.5  # 분위수 중간
        if 'News_Z' not in combined.columns: combined['News_Z'] = 0.0  # 평균
        
        macd_col = [c for c in combined.columns if 'MACD_' in c and 'MACDs_' not in c and 'MACDh_' not in c]
        signal_col = [c for c in combined.columns if 'MACDs_' in c]
        if macd_col and signal_col:
            combined['MACD'], combined['Signal_Line'] = combined[macd_col[0]], combined[signal_col[0]]
        else:
            macd = ta.macd(combined['Close'])
            if macd is not None:
                combined = pd.concat([combined, macd], axis=1)
                combined['MACD'], combined['Signal_Line'] = combined['MACD_12_26_9'], combined['MACDs_12_26_9']
            else:
                combined['MACD'], combined['Signal_Line'] = 0.0, 0.0
        
        combined['MACD_Hist'] = combined['MACD'] - combined['Signal_Line']
        combined['Prev2_MACD_Hist'] = combined['MACD_Hist'].shift(2)
        
        combined['RSI_SMA'] = combined['RSI'].rolling(window=14).mean()
        combined['SMA200'] = combined['Close'].rolling(window=200).mean()
        combined['SMA60'] = combined['Close'].rolling(window=60).mean()
        
        # [신규] 사용자 정의 Panic MA 계산 (예: SMA120 등)
        if smart_params and 'panic_ma' in smart_params:
            p_ma = smart_params['panic_ma']
            if f'SMA{p_ma}' not in combined.columns:
                combined[f'SMA{p_ma}'] = combined['Close'].rolling(window=p_ma).mean()
        
        # [신규] Williams %R 데이터베이스에서 복사
        if 'WILLR' in base_df.columns:
            combined['WILLR'] = base_df['WILLR']

        # [신규 C-4B] RSI slope / StochRSI 데이터 복사
        for _col in ['RSI_slope3', 'RSI_slope5', 'StochRSI_K', 'StochRSI_D']:
            if _col in base_df.columns:
                combined[_col] = base_df[_col]
            
        # [신규] ADX 및 DI 지표 계산 추가 (시그널용) - 위에서 ADX 없을 때 0으로 채웠으므로 계산 시도
        if 'High' in base_df.columns and 'Low' in base_df.columns:
            adx_df = ta.adx(base_df['High'], base_df['Low'], base_df['Close'], length=14)
            if adx_df is not None:
                # 덮어쓰기 (0으로 채워진 경우 갱신)
                combined['ADX'] = adx_df['ADX_14']
                combined['DMP'] = adx_df['DMP_14']
                combined['DMN'] = adx_df['DMN_14']
        
        # [신규] 급락 진단용 지표 계산 (Base Asset 기준)
        combined['Drop_Acc'] = combined['Close'].pct_change(periods=2) * 100
        combined['Gap_Down'] = (combined['Open'] / combined['Close'].shift(1) - 1) * 100
        
        # [신규] ATR 지표 계산
        combined['ATR_14'] = ta.atr(base_df['High'], base_df['Low'], base_df['Close'], length=14)
        combined['ATR_14'] = combined['ATR_14'].ffill().fillna(0)
        combined['ATR_20'] = ta.atr(base_df['High'], base_df['Low'], base_df['Close'], length=20)
        combined['ATR_20'] = combined['ATR_20'].ffill().fillna(0)
        
        # [신규] 파라볼릭 SAR 계산
        sar_df = ta.psar(base_df['High'], base_df['Low'], base_df['Close'])
        if sar_df is not None and not sar_df.empty:
            sar_cols = [c for c in sar_df.columns if c.startswith('PSARl') or c.startswith('PSARs')]
            if sar_cols:
                combined['SAR'] = sar_df[sar_cols].bfill(axis=1).iloc[:, 0]
                combined['SAR'] = combined['SAR'].ffill().fillna(0)

        cols_to_shift = ['RSI', 'RSI_SMA', 'MACD', 'Signal_Line', 'SMA200', 'SMA60', 'MACD_Hist', 'ADX', 'DMP', 'DMN', 'VIX', 'Drop_Acc', 'Gap_Down', 'ATR_14', 'ATR_20', 'SAR']
        
        # 존재하는 컬럼만 shift 타겟으로 필터링 (보너스 안전장치)
        cols_to_shift = [c for c in cols_to_shift if c in combined.columns]
        
        if 'STOCH_K' in combined.columns:
            cols_to_shift += ['STOCH_K', 'STOCH_D']
        if 'WILLR' in combined.columns:
            cols_to_shift += ['WILLR']
        if 'VXN' in combined.columns:
            cols_to_shift += ['VXN']
            
        for col in cols_to_shift:
            combined[f'Prev_{col}'] = combined[col].shift(1)
        combined['Prev_Close'] = combined['Close'].shift(1)
        
        combined['FG'] = combined['FG'].ffill().fillna(50)
        combined['VIX'] = combined['VIX'].ffill().fillna(15)
        if 'VXN' in combined.columns: combined['VXN'] = combined['VXN'].ffill().fillna(20)
        # [신규] VVIX 결측치 처리
        if 'VVIX' in combined.columns: combined['VVIX'] = combined['VVIX'].ffill()
        combined = combined.fillna(0)
        
        # [신규] 자산 상장일 고려: 모든 자산의 실제 데이터가 존재하는 날부터 시작
        leverage_df = data_dict.get(leverage_asset, pd.DataFrame())
        
        # [수정] dropna()를 사용하여 실제 데이터가 있는 날짜 탐색 (NaT 방어)
        b_min = base_df.dropna(subset=['Close']).index.min()
        l_min = leverage_df.dropna(subset=['Close']).index.min() if not leverage_df.empty else b_min
        
        # 둘 중 하나라도 NaT이면 다른 쪽을 사용, 둘 다 NaT이면 에러 방지용으로 오늘 날짜 사용
        if pd.isna(b_min): b_min = pd.Timestamp.now().normalize()
        if pd.isna(l_min): l_min = b_min
        common_start = max(b_min, l_min)
        
        dates = combined.index
        dates = dates[dates >= common_start]
        
        # [신규 추가] start_date와 end_date를 문자열에서 datetime.date 객체로 변환하여 에러 방지
        if start_date and isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()

        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame(columns=['Value', 'Stage', 'Asset', 'Trade_Label']), pd.DataFrame(), []

        portfolio_value, cash = allocated_capital, allocated_capital * cash_ratio
        current_total_val = portfolio_value # Unify with other generators
        current_planned_asset, rebalance_stage, last_entry_price = base_asset, 0, 0
        peak_price, peak_macd = 0, 0
        holdings = {t: 0.0 for t in data_dict.keys()}
        
        initial_price = base_df['Close'].asof(dates[0])
        if pd.isna(initial_price) or initial_price <= 0:
            # [수정] 데이터 부재 시 백업 로직 강화
            valid_closes = base_df['Close'].dropna()
            initial_price = valid_closes.iloc[0] if not valid_closes.empty else 1.0
            
        _raw_shares = (portfolio_value * (1 - cash_ratio)) / initial_price if initial_price > 0 else 0
        if integer_shares and initial_price > 0:
            _int_shares = math.floor(_raw_shares)
            cash += (_raw_shares - _int_shares) * initial_price  # 잔여 현금 보존
            holdings[base_asset] = _int_shares
        else:
            holdings[base_asset] = _raw_shares
        
        def get_v_level(planned, stage):
            # Stage는 이제 항상 레버리지 노출 수준(0-3)을 의미합니다.
            return stage

        def apply_shares(new_h_vals, trade_px, cash_val):
            """new_h_vals(금액 기준) → holdings(수량 기준) 변환. integer_shares=True면 정수 주식 처리.
            Greedy Fill: 잔여 현금으로 목표 자산 추가 매수 (현금 최소화).
            우선순위: 레버리지 자산(비싼 것) 먼저 → 그 다음 기초 자산 순으로 매수.
            """
            new_holdings = {}
            residual = 0.0
            for t, val in new_h_vals.items():
                px = trade_px.get(t, 0)
                if px <= 0:
                    new_holdings[t] = 0.0
                    continue
                raw = val / px
                if integer_shares:
                    floored = math.floor(raw)
                    residual += (raw - floored) * px  # 잔여 현금
                    new_holdings[t] = float(floored)
                else:
                    new_holdings[t] = raw
            # Greedy Fill: 잔여 현금으로 추가 매수 (현금 최소화)
            if integer_shares and residual > 0:
                buyable = [(t, trade_px[t]) for t in new_h_vals
                           if trade_px.get(t, 0) > 0 and new_holdings.get(t, 0) >= 0]
                # 가격 내림차순 정렬: 레버리지(고가) → 기초자산(저가) 순
                buyable.sort(key=lambda x: x[1], reverse=True)
                changed = True
                while changed and residual > 0:
                    changed = False
                    for t, px in buyable:
                        if residual >= px:
                            extra = math.floor(residual / px)
                            if extra > 0:
                                new_holdings[t] = new_holdings.get(t, 0.0) + float(extra)
                                residual -= extra * px
                                changed = True
            return new_holdings, cash_val + residual

        history = []
        trade_slots = [] # list of {'buy_price': val, 'date': date}
        closed_trades = [] # list of {'entry_date': d, 'exit_date': d, 'asset': a, 'buy_price': p, 'sell_price': p, 'ret': r, 'status': s}
        # [신규] 모든 1~2매수, 1~4매도 신호 발생 기록 수집 배열
        signal_events = []
        
        # [신규] RSI 1차 대기(알람) 상태 저장 변수 (신호 번호별 독립적인 bool 값 가짐)
        buy_wait_flags = {i: False for i in range(1, 5)}
        sell_wait_flags = {i: False for i in range(1, 5)}
        panic_buy_wait_flags = {i: False for i in range(1, 4)}
        
        # [수정] 자산 그룹 결정 (나스닥 전용으로 단일화)
        target_group = ['QQQ', 'QLD', 'TQQQ']
        
        # [신규] 루프 속도 최적화를 위한 데이터 매트릭스 사전 추출 (Fast Mode)
        # combined DataFrame의 각 열을 Numpy 배열(또는 list)로 변환하여 O(1) 정수 인덱스 접근을 가능하게 함
        # [수정] 전체 행이 아닌 필터링된 dates 구간만 추출하여 인덱스 어긋남(Misalignment) 방지
        col_idx = {col: i for i, col in enumerate(combined.columns)}
        combined_matrix = combined.loc[dates].values
        
        # 타겟 자산들 가격 데이터 사전 정렬 배열화
        # 주의: check_targets는 루틴 내내 동일하므로 밖에서 미리 정의
        target_group = ['QQQ', 'QLD', 'TQQQ']
        base_targets = list(set([base_asset, leverage_asset] + target_group))
        
        price_arrays = {}
        for t in base_targets:
            if t in data_dict:
                # [수정] 데이터 유실 방지: 전체 인덱스에서 ffill을 먼저 수행한 뒤 dates 구간만 추출
                filled_df = data_dict[t].reindex(combined.index).ffill().loc[dates]
                price_arrays[t] = filled_df['Close'].values

        prev_dmp_idx = col_idx.get('Prev_DMP', -1)
        prev_dmn_idx = col_idx.get('Prev_DMN', -1)
        prev_close_idx = col_idx.get('Prev_Close', -1)
        prev_sar_idx = col_idx.get('Prev_SAR', -1)
        sar_idx = col_idx.get('SAR', -1)
        news_q_idx = col_idx.get('News_Q', -1)
        news_z_idx = col_idx.get('News_Z', -1)
        bb_l_idx = col_idx.get('BB_Lower', -1)
        bb_u_idx = col_idx.get('BB_Upper', -1)
        sma60_idx = col_idx.get('SMA60', -1)
        sma200_idx = col_idx.get('SMA200', -1)
        vxn_idx = col_idx.get('VXN', col_idx.get('VIX', -1))

        # [신규] 동적 현금 비중 파라미터 추출
        dc_params = (smart_params.get('dynamic_cash') or {}) if smart_params else {}
        use_dc = dc_params.get('use', False)

        # [신규] 루프 시작 전 전체 기간의 Regime 미리 계산 (벡터화 연산으로 속도 확보)
        if use_dc:
            # 보수적 접근: MA200과 VXN이 둘 다 있어야 함
            if sma200_idx >= 0 and vxn_idx >= 0:
                # [수정] VXN × Dev200 2차원 국면 판단 (use_dev200_regime=True 시 활성화)
                _use_dev200_regime = dc_params.get('use_dev200_regime', False)
                def get_regime(row):
                    p  = row['Close']
                    v  = row['VXN'] if 'VXN' in row else row['VIX']
                    ma = row['SMA200']
                    dev200 = (p / ma) if (ma and ma > 0) else 1.0

                    if _use_dev200_regime:
                        # ── VXN × Dev200 2차원 국면 ──────────────────────────────
                        # hot    : VXN 낮음 AND Dev200 높음 (저변동 강세장)
                        # caution: VXN 중간 AND Dev200 하락 (MDD 핵심 구간)
                        # bear   : VXN 급등 OR Dev200 급락 (패닉/붕괴)
                        # normal : 나머지
                        hot_vxn  = dc_params.get('hot_vxn',  20.0)
                        hot_dev  = dc_params.get('hot_dev',   1.10)
                        caut_vxn = dc_params.get('caution_vxn', 30.0)
                        caut_dev = dc_params.get('caution_dev',  1.05)
                        bear_vxn = dc_params.get('bear_vxn',  35.0)
                        bear_dev = dc_params.get('bear_dev',   0.95)

                        if v >= bear_vxn or dev200 <= bear_dev: return 'bear'
                        if v >= caut_vxn and dev200 < caut_dev: return 'caution'
                        if v < hot_vxn  and dev200 >= hot_dev:  return 'hot'
                        return 'normal'
                    else:
                        # ── 기존 VXN 1차원 국면 (하위 호환) ─────────────────────
                        b_vxn, c_vxn, bear_vxn = dc_params.get('bull_vxn', 18.0), dc_params.get('caution_vxn', 35.0), dc_params.get('bear_vxn', 40.0)
                        ma_buffer = dc_params.get('ma200_buffer', 0.0)
                        if p < ma * (1 - ma_buffer/100.0) or v >= bear_vxn: return 'bear'
                        if v >= c_vxn: return 'caution'
                        if v >= b_vxn: return 'neutral'
                        return 'bull'
                
                # combined 전체에서 계산 후 reindex
                regimes = combined.apply(get_regime, axis=1)
                
                # [수정] 문자열 rolling apply 오류 방지를 위한 숫자 매핑 방식 적용
                # use_dev200_regime 시 'hot/normal/caution/bear', 기존은 'bull/neutral/caution/bear'
                if _use_dev200_regime:
                    r_map = {'hot': 0, 'normal': 1, 'caution': 2, 'bear': 3}
                else:
                    r_map = {'bull': 0, 'neutral': 1, 'caution': 2, 'bear': 3}
                inv_r_map = {v: k for k, v in r_map.items()}
                num_regimes = regimes.map(r_map)
                
                # 3일 일관성(Rolling Mode) 적용: 3일 중 가장 많이 발생한 국면 선택 (동수 시 최신값)
                rolled_num = num_regimes.rolling(window=3).apply(
                    lambda x: pd.Series(x).value_counts().idxmax() if len(x)==3 else x[-1],
                    raw=False
                )
                combined['Regime'] = rolled_num.map(inv_r_map)
            else:
                combined['Regime'] = 'bull' # 기본값
        else:
            combined['Regime'] = 'bull'
        
        regime_idx = combined.columns.get_loc('Regime')
        # [수정] 행렬 데이터 재추출 (Regime 컬럼 포함)
        combined_matrix = combined.loc[dates].values
        col_idx = {col: i for i, col in enumerate(combined.columns)}

        for i in range(len(dates)):
            date = dates[i]
            row_arr = combined_matrix[i]
            # [수정] 기존 로직(row.get/row['...']) 호환성을 위해 Series 객체 복구 (단, 인덱스는 사전 필터링됨)
            row = pd.Series(row_arr, index=combined.columns)
            
            # [수정] 사전 추출된 NumPy 배열에서 가격 정보 가져오기 (pandas .loc 호출 제거)
            price = row_arr[col_idx['Close']]
            
            # [신규] 일별 로직 변수 초기화
            trade_ret_val = None
            trade_status_val = ""
            trade_entry_date_val = None
            s3_drop_triggered_val = 0
            s3_detail_val = ""
            s3_sell3_event_val = 0
            
            trade_label = ""
            rebalance_cause = ""
            # [긴급 수정] 목표 상태 변수 초기화 (현재 상태 유지가 기본 - UnboundLocalError 방지)
            new_planned = current_planned_asset
            new_stage = rebalance_stage
            target_lev_pct = (1.0 if new_stage == 3 else (0.70 if new_stage == 2 else (0.30 if new_stage == 1 else 0.0)))
            rebalance_needed = False

            is_delayed = False
            delay_reason = ""
            current_signal_ref = None
            
            effective_lev_asset = leverage_asset 
            try:
                currently_holding = [t for t, qty in holdings.items() if qty > 0]
                check_targets = list(set([base_asset, leverage_asset, effective_lev_asset] + currently_holding + target_group))
                
                prices = {}
                for t in check_targets:
                    if t in price_arrays:
                        prices[t] = price_arrays[t][i]
                    else:
                        prices[t] = price # Fallback (should not happen properly configured)
                
                if pd.isna(prices[base_asset]):
                    # 데이터 부재 시(상장 전 등) 가치 유지하며 Yield하여 동기화 유지
                    state_dict = {
                        'date': date,
                        'cash': cash,
                        'portfolio_value': portfolio_value,
                        'stage': int(rebalance_stage),
                        'history_row': None,
                        'holdings': dict(holdings)
                    }
                    injected = yield state_dict
                    if injected and isinstance(injected, dict) and 'cash_delta' in injected:
                        delta = injected['cash_delta']
                        fee_rb = abs(delta) * fee_rate
                        portfolio_value += delta - fee_rb
                        cash += delta - fee_rb
                    continue
            except Exception as e:
                pass
            
            # [중요] 일일 포트폴리오 가치 업데이트 (재계산)
            # 재밸런싱 신호가 없더라도 매일 시장 가격 변동을 반영해야 함
            portfolio_value = cash + sum(holdings[t] * prices.get(t, 0) for t in holdings if prices.get(t, 0) > 0)
            current_total_val = portfolio_value # 동기화
            
            rsi = row_arr[col_idx['RSI']]
            fg = row_arr[col_idx['FG']]
            vxn = row_arr[col_idx['VXN']] if 'VXN' in col_idx else row_arr[col_idx['VIX']]
            macd_val = row_arr[col_idx['MACD']]
            signal_line = row_arr[col_idx['Signal_Line']]
            rsi_sma = row_arr[col_idx['RSI_SMA']]
            macd_hist = row_arr[col_idx['MACD_Hist']]
            bb_l = row_arr[bb_l_idx] if bb_l_idx >= 0 else 0
            bb_u = row_arr[bb_u_idx] if bb_u_idx >= 0 else 0
            prev_rsi = row_arr[col_idx['Prev_RSI']]
            prev_rsi_sma = row_arr[col_idx['Prev_RSI_SMA']]
            prev_macd = row_arr[col_idx['Prev_MACD']]
            prev_macd_hist = row_arr[col_idx['Prev_MACD_Hist']]
            prev2_macd_hist = row_arr[col_idx['Prev2_MACD_Hist']]
            prev_vxn = row_arr[col_idx['Prev_VXN']] if 'Prev_VXN' in col_idx else row_arr[col_idx['Prev_VIX']] if 'Prev_VIX' in col_idx else 0
            vxn_jump_pct = ((vxn / prev_vxn - 1) * 100) if prev_vxn > 0 else 0
            
            # [신규] 200일 이격도 계산 (시그널용)
            sma200_v = row_arr[sma200_idx] if sma200_idx >= 0 and row_arr[sma200_idx] > 0 else 0
            cur_dev200 = price / sma200_v if sma200_v > 0 else 0

            # [신규] 다이버전스용 고점 추적
            if current_planned_asset != base_asset:
                if price > peak_price:
                    peak_price = price
                    peak_macd = macd_val
            else:
                peak_price, peak_macd = 0, 0
            
            # [신규] Williams %R 지표 확보
            willr_val = row.get('WILLR', 0)
            prev_willr = row.get('Prev_WILLR', 0)

            # [신규 C-4B] RSI slope / StochRSI 지표 확보
            rsi_slope3_val  = row.get('RSI_slope3', float('nan'))
            rsi_slope5_val  = row.get('RSI_slope5', float('nan'))
            stochrsi_k_val  = row.get('StochRSI_K', float('nan'))
            
            if params:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                is_macd_golden_cross = (prev_macd_hist < 0) and (macd_hist > 0)
                is_macd_dead_cross = (prev_macd_hist > 0) and (macd_hist < 0)
                
                # [신규] 지표 값 확보
                adx_val = row.get('ADX', 0)
                d_plus, d_minus = row.get('DMP', 0), row.get('DMN', 0)
                
                # [신규] 공통 매수 조건 평가 함수 (일반 매수 및 하락장 매수 지연 해제 공용)
                def check_buy_cond(bp, idx_offset, wait_dict):
                    c, a = True, False
                    rs = []
                    
                    if bp.get('use_rsi_wait'):
                        wait_val = bp.get('rsi_wait_val', 35)
                        if rsi <= wait_val: wait_dict[idx_offset] = True
                        c &= wait_dict.get(idx_offset, False)
                        if wait_dict.get(idx_offset, False): rs.append(f"RSI대기(<={wait_val})")
                        a = True
                    
                    if bp.get('rsi_val', 0) > 0: 
                        c &= (rsi < bp['rsi_val']); a = True
                        rs.append(f"RSI<{bp['rsi_val']}")

                    if bp.get('rsi_cross'): 
                        c &= is_rsi_golden_cross; a = True
                        rs.append("RSI 골든크로스")
                        
                    if bp.get('use_adx'):
                        op = bp.get('adx_op', '<=')
                        val = bp.get('adx_val', 40)
                        if op == '<=': c &= (adx_val <= val)
                        else: c &= (adx_val >= val)
                        a = True
                        rs.append(f"ADX{op}{val}")
                        
                    if bp.get('rsi_inc'): 
                        c &= (rsi > prev_rsi); a = True
                        rs.append("RSI 상승")
                    if bp.get('macd_inc'): 
                        c &= (macd_val > prev_macd); a = True
                        rs.append("MACD 상승")
                    if bp.get('macd_signal_below'): 
                        c &= (macd_hist < 0); a = True
                        rs.append("MACD 시그널 하단")
                    if bp.get('macd_golden'): 
                        c &= is_macd_golden_cross; a = True
                        rs.append("MACD 골든크로스")
                    if bp.get('bb_lower'): 
                        c &= (price <= bb_l); a = True
                        rs.append("BB 하단 터치")
                    
                    if bp.get('di_plus_above'):
                        c &= (d_plus > d_minus); a = True
                        rs.append("DI+우세(상태유지)")

                    if bp.get('di_plus_cross'):
                        c &= ((row.get('Prev_DMP', 0) < row.get('Prev_DMN', 0)) and (d_plus > d_minus))
                        a = True
                        rs.append("DI+골든크로스(순간)")
                    
                    if bp.get('use_willr'):
                        w_val = bp.get('willr_val', -80)
                        c &= ((prev_willr <= w_val) and (willr_val > w_val))
                        a = True
                        
                    if bp.get('use_sar'):
                        # 상향 돌파: 전날 종가는 SAR 아래, 오늘 종가는 SAR 위
                        c &= ((row.get('Prev_Close', 0) < row.get('Prev_SAR', 0)) and (price > row.get('SAR', 0)))
                        a = True
                        rs.append("파라볼릭SAR 상향돌파")

                    # [신규] AI 뉴스 심리 조건 추가 (분위수 Q, 표준점수 Z 연동)
                    # [신규] AI 뉴스 심리 분위수(Q) 및 Z-점수(Z) 필터로 정형화
                    if bp.get('use_news_q'):
                        op = bp.get('news_q_op', '>')
                        val = bp.get('news_q_val', 0.9)
                        n_q = row.get('News_Q', 0.5)
                        if op == '>': c &= (n_q > val)
                        elif op == '>=': c &= (n_q >= val)
                        elif op == '<': c &= (n_q < val)
                        elif op == '<=': c &= (n_q <= val)
                        a = True
                        rs.append(f"AI뉴스분위수{op}{val}")

                    if bp.get('use_news_z'):
                        op = bp.get('news_z_op', '>')
                        val = bp.get('news_z_val', 1.0)
                        n_z = row.get('News_Z', 0.0)
                        if op == '>': c &= (n_z > val)
                        elif op == '>=': c &= (n_z >= val)
                        elif op == '<': c &= (n_z < val)
                        elif op == '<=': c &= (n_z <= val)
                        a = True
                        rs.append(f"AI뉴스Z점수{op}{val}")

                    if bp.get('use_dev200') and sma200_v > 0:
                        op = bp.get('dev200_op', '>=')
                        val = bp.get('dev200_val', 1.0)
                        if op == '>': c &= (cur_dev200 > val)
                        elif op == '>=': c &= (cur_dev200 >= val)
                        elif op == '<': c &= (cur_dev200 < val)
                        elif op == '<=': c &= (cur_dev200 <= val)
                        a = True
                        rs.append(f"이격도(200){op}{val}")

                    # [신규 C-4B] RSI slope 하한 필터 (use=False → BASE 동일)
                    # RSI 상승 속도가 rsi_slope_val 미만이면 매수 차단
                    if bp.get('use_rsi_slope'):
                        period = bp.get('rsi_slope_period', 5)
                        val    = bp.get('rsi_slope_val', 3.0)
                        slope  = rsi_slope5_val if period == 5 else rsi_slope3_val
                        import math
                        if not math.isnan(slope):
                            c &= (slope >= val)
                        a = True
                        rs.append(f"RSI기울기({period}일)>={val:.1f}")

                    # [신규 C-4B] StochRSI_K 하한 필터 (use=False → BASE 동일)
                    # StochRSI_K가 stochrsi_val 미만이면 매수 차단
                    if bp.get('use_stochrsi'):
                        val   = bp.get('stochrsi_val', 30.0)
                        import math
                        if not math.isnan(stochrsi_k_val):
                            c &= (stochrsi_k_val >= val)
                        a = True
                        rs.append(f"StochRSI_K>={val:.1f}")

                    return a, c, rs

                # ── [국면 분기] VXN × Dev200 기반 국면 판단 ──────────────────────
                # regime 파라미터가 있고 use=True 일 때만 국면별 시그널 전환
                # hot  : VXN < hot_vxn_max AND Dev200 > hot_dev200_min  → 과열 구간
                # panic: VXN > panic_vxn_min                            → 패닉/공포 구간
                # normal: 그 외                                         → 기본 BASE 시그널
                _regime_cfg = (params.get('regime') or {})
                _use_regime = _regime_cfg.get('use', False)
                _active_buy_signals  = params['buy_signals']
                _active_sell_signals = params['sell_signals']
                if _use_regime and cur_dev200 > 0:
                    _hot_vxn   = _regime_cfg.get('hot_vxn_max', 20)
                    _hot_dev   = _regime_cfg.get('hot_dev200_min', 1.10)
                    _panic_vxn = _regime_cfg.get('panic_vxn_min', 30)
                    if vxn < _hot_vxn and cur_dev200 > _hot_dev:
                        # 과열 국면: hot_buy_signals / hot_sell_signals 로 교체
                        if 'hot_buy_signals' in _regime_cfg:
                            _active_buy_signals = _regime_cfg['hot_buy_signals']
                        if 'hot_sell_signals' in _regime_cfg:
                            _active_sell_signals = _regime_cfg['hot_sell_signals']
                    elif vxn > _panic_vxn:
                        # 패닉 국면: panic_buy_signals_regime / panic_sell_signals 로 교체
                        if 'panic_buy_signals_regime' in _regime_cfg:
                            _active_buy_signals = _regime_cfg['panic_buy_signals_regime']
                        if 'panic_sell_signals' in _regime_cfg:
                            _active_sell_signals = _regime_cfg['panic_sell_signals']
                # ─────────────────────────────────────────────────────────────

                is_buy_signal = False
                buy_signal_idx = -1
                signal_reason = ""
                for idx, bp in enumerate(_active_buy_signals):
                    active, cond, reasons = check_buy_cond(bp, idx + 1, buy_wait_flags)
                    if active and cond:
                        is_buy_signal = True
                        buy_signal_idx = idx + 1
                        signal_reason = " / ".join(reasons)
                        break

                is_sell_signal = False
                sell_signal_idx = -1 # [신규] 어떤 매도 신호가 발생했는지 추적
                for idx, sp in enumerate(_active_sell_signals):
                    cond, active = True, False
                    reasons = []
                    
                    # [신규] RSI 1차 대기(알람) 로직 적용 (매도는 이상일 때 켜짐)
                    if sp.get('use_rsi_wait'):
                        wait_val = sp.get('rsi_wait_val', 70)
                        if rsi >= wait_val:
                            sell_wait_flags[idx + 1] = True
                        
                        cond &= sell_wait_flags[idx + 1]
                        if sell_wait_flags[idx + 1]: 
                            reasons.append(f"RSI대기(>={wait_val})")
                        active = True

                    if sp.get('rsi_val', 0) > 0: 
                        cond &= (rsi >= sp['rsi_val']); active = True
                        reasons.append(f"RSI>={sp['rsi_val']}")

                    if sp.get('rsi_dead'): 
                        cond &= is_rsi_dead_cross; active = True

                        reasons.append("RSI 데드크로스")
                    if sp.get('rsi_dec'): 
                        cond &= (rsi < prev_rsi); active = True
                        reasons.append("RSI 하락")
                    if sp.get('macd_dec'): 
                        cond &= (macd_val < prev_macd); active = True
                        reasons.append("MACD 하락")
                    if sp.get('macd_signal_above'): 
                        cond &= (macd_hist > 0); active = True
                        reasons.append("MACD 시그널 상단")
                    if sp.get('macd_dead'): 
                        cond &= is_macd_dead_cross; active = True
                        reasons.append("MACD 데드크로스")
                    if sp.get('di_minus_above'):
                        cond &= (d_minus > d_plus); active = True
                        reasons.append("DI- 우세")
                    if sp.get('di_minus_cross'):
                        prev_dmp, prev_dmn = row.get('Prev_DMP', 0), row.get('Prev_DMN', 0)
                        is_di_minus_cross = (prev_dmn < prev_dmp) and (d_minus > d_plus)
                        cond &= is_di_minus_cross; active = True
                        reasons.append("DI-데드크로스")
                    if sp.get('bb_upper'): 
                        cond &= (price >= bb_u); active = True
                        reasons.append("BB 상단 터치")
                    
                    # [신규] 샹들리에 엑시트를 매도 시그널 조건으로 판단
                    if sp.get('use_chandelier'):
                        # 매도 신호용 설정(보통 20일)의 이전 ATR을 가져옵니다.
                        period_s = params.get('atr_period_sell', 20)
                        prev_atr_sell = row.get(f'Prev_ATR_{period_s}', 0)
                        c_mult = float(sp.get('chandelier_mult', 3.0))
                        
                        # 투자 상태일 때 고점 대비 하락분이 승수 범위 내를 벗어났는지 확인
                        hit_chandelier = False
                        if (current_planned_asset != base_asset) and peak_price > 0:
                            hit_chandelier = (prices[base_asset] <= peak_price - (c_mult * prev_atr_sell))
                        
                        cond &= hit_chandelier; active = True
                        reasons.append(f"샹들리에 엑시트(ATR*{c_mult})")

                    # [신규] 파라볼릭 SAR 데드크로스를 매도 시그널 조건으로 판단
                    if sp.get('use_sar'):
                        # 전날과 오늘의 가격 흐름이 SAR 점을 하향 돌파했는지 확인 (Trailing Stop)
                        prev_sar = row.get('Prev_SAR', 0)
                        curr_sar = row.get('SAR', 0)
                        # 이전 종가는 SAR보다 높았으나, 현재 종가가 SAR 아래로 떨어졌을 때 데드크로스로 간주
                        is_sar_dead_cross = (row.get('Prev_Close', 0) > prev_sar) and (price < curr_sar)
                        
                        cond &= is_sar_dead_cross; active = True
                        reasons.append("파라볼릭SAR 데드크로스")
                    
                    # [신규] Williams %R 하향 이탈
                    if sp.get('use_willr'):
                        w_val = sp.get('willr_val', -20)
                        is_willr_dead = (prev_willr >= w_val) and (willr_val < w_val)
                        cond &= is_willr_dead; active = True
                        reasons.append(f"W%R 하향({w_val})")
                    
                    # [신규] AI 뉴스 심리 분위수(Q) 및 Z-점수(Z) 필터 추가 (매도)
                    if sp.get('use_news_q'):
                        op = sp.get('news_q_op', '<')
                        val = sp.get('news_q_val', 0.2)
                        n_q = row.get('News_Q', 0.5)
                        if op == '>': cond &= (n_q > val)
                        elif op == '>=': cond &= (n_q >= val)
                        elif op == '<': cond &= (n_q < val)
                        elif op == '<=': cond &= (n_q <= val)
                        active = True
                        reasons.append(f"AI뉴스분위수{op}{val}")

                    if sp.get('use_news_z'):
                        op = sp.get('news_z_op', '<')
                        val = sp.get('news_z_val', -1.5)
                        n_z = row.get('News_Z', 0.0)
                        if op == '>': cond &= (n_z > val)
                        elif op == '>=': cond &= (n_z >= val)
                        elif op == '<': cond &= (n_z < val)
                        elif op == '<=': cond &= (n_z <= val)
                        active = True
                        reasons.append(f"AI뉴스Z점수{op}{val}")

                    # [신규] 200일 이격도 조건 추가 (매도)
                    if sp.get('use_dev200') and sma200_v > 0:
                        op = sp.get('dev200_op', '>=')
                        val = sp.get('dev200_val', 1.0)
                        if op == '>': cond &= (cur_dev200 > val)
                        elif op == '>=': cond &= (cur_dev200 >= val)
                        elif op == '<': cond &= (cur_dev200 < val)
                        elif op == '<=': cond &= (cur_dev200 <= val)
                        active = True
                        reasons.append(f"이격도(200){op}{val}")
                    
                    if active and cond: 
                        is_sell_signal = True
                        sell_signal_idx = idx + 1 # 1-based index
                        signal_reason = " / ".join(reasons)
                        break

                # [신규] 공통 진단 로그 수집: 매수 및 매도 신호 모두 기록
                if is_buy_signal and buy_signal_idx > 0:
                    current_signal_ref = {'date': date, 'type': f'매수신호{buy_signal_idx}', 'reason': signal_reason, 'price': price, 'executed': False}
                    if not fast_mode: signal_events.append(current_signal_ref)
                    # [수정] 신호 조건이 완전 충족되었으므로 다음 거래를 위해 대기 플래그 초기화
                    buy_wait_flags[buy_signal_idx] = False
                    
                if is_sell_signal and sell_signal_idx > 0:
                    current_signal_ref = {'date': date, 'type': f'매도신호{sell_signal_idx}', 'reason': signal_reason, 'price': price, 'executed': False}
                    if not fast_mode: signal_events.append(current_signal_ref)
                    # [수정] 신호 조건이 완전 충족되었으므로 다음 거래를 위해 대기 플래그 초기화
                    sell_wait_flags[sell_signal_idx] = False

                # [복구] S3(레버리지 3단계)에서 매도신호3 발생 시에만 기록
                if is_sell_signal and rebalance_stage == 3 and sell_signal_idx == 3:
                    s3_sell3_event_val = 1

                # [기존] 일반 매도 시그널 순회 끝
                pass
                
                b_up, b_down, s_up, s_down = params['buy_reb_up'], params['buy_reb_down'], params['sell_reb_up'], params['sell_reb_down']
            else:
                is_rsi_golden_cross = (prev_rsi < prev_rsi_sma) and (rsi > rsi_sma)
                is_buy_signal = False
                signal_reason = ""
                if is_rsi_golden_cross and (macd_val > prev_macd) and (macd_hist < 0) and (adx_val <= 40):
                    is_buy_signal = True
                    signal_reason = "기본(RSI골든크로스+MACD상승)"
                elif (rsi < 35) and (adx_val <= 40):
                    is_buy_signal = True
                    signal_reason = "기본(RSI과매도)"
                
                sell_cond_1 = (rsi >= 70) and (rsi < prev_rsi)
                is_rsi_dead_cross = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
                sell_cond_2 = (macd_hist > 0) and (macd_val < prev_macd) and is_rsi_dead_cross
                sell_cond_5 = (prev_macd_hist > 0) and (macd_hist < 0)
                
                is_sell_signal = False
                if sell_cond_1: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI꺾임)"
                elif sell_cond_2: 
                    is_sell_signal = True
                    signal_reason = "기본(RSI데드크로스+MACD)"
                elif sell_cond_5: 
                    is_sell_signal = True
                    signal_reason = "기본(MACD데드크로스)"
                
                b_up, b_down, s_up, s_down = 0.03, -0.03, 0.03, -0.03

            # [수정] 스마트 레버리지 필터링 (RSI Turbo)
            # VXN Safety는 아래 버그수정 블록에서 강제 청산으로 처리됨
            effective_lev_asset = leverage_asset
            smart_reason = ""

            if smart_params:
                if smart_params.get('use_rsi_turbo') and rsi < smart_params.get('rsi_turbo', 31):
                    effective_lev_asset = target_group[2]
                    smart_reason = f"가속(RSI:{rsi:.1f})"
                elif smart_params.get('use_qld_decel') and rebalance_stage == 2:
                    # [신규] QLD Decelerate: S2 + 위험 조건 시 QLD 감속
                    qd = (smart_params.get('qld_decel') or {})
                    decel_vxn = qd.get('vxn_min', 28.0)
                    decel_dev = qd.get('dev200_max', 1.05)
                    if vxn >= decel_vxn or cur_dev200 < decel_dev:
                        effective_lev_asset = target_group[1]  # QLD
                        smart_reason = f"감속(VXN:{vxn:.1f}/Dev:{cur_dev200:.3f})"
                    else:
                        effective_lev_asset = leverage_asset
                        smart_reason = "정상(Normal)"
                else:
                    effective_lev_asset = leverage_asset
                    if smart_params.get('use_rsi_turbo'):
                        smart_reason = "정상(Normal)"

            # (기존 prices 위치 삭제됨 - 상단으로 이동)
            
            current_total_val = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings if t in prices)
            
            # [신규 추가] 수익률 nan 방지를 위한 수치 정규화 (Defensive Check)
            if pd.isna(current_total_val) or current_total_val <= 0:
                current_total_val = history[-1]['Value'] if history else allocated_capital
                
            rebalance_needed = False
            new_planned = current_planned_asset
            new_stage = rebalance_stage

            # [버그수정] VXN Safety 강제 청산: VXN 초과 시 기존 레버리지 보유 포지션을 즉시 QQQ로 청산
            # 기존 코드는 effective_lev_asset만 교체하여 매수 신호 없으면 레버리지를 계속 보유하는 버그가 있었음
            if smart_params:
                use_vxn = smart_params.get('use_vxn_safety', smart_params.get('use_vix_safety', False))
                exit_val = smart_params.get('vxn_exit', smart_params.get('vix_exit', 31))
                # [Step A] VXN Safety Dev200 필터: 0.0이면 비활성(기존 동작), >0이면 Dev200 >= 값일 때만 청산
                vxn_dev_filter = float(smart_params.get('vxn_dev_filter', 0.0))
                dev_cond = True if vxn_dev_filter <= 0.0 else (cur_dev200 >= vxn_dev_filter)
                if use_vxn and vxn > exit_val and dev_cond and rebalance_stage > 0:
                    rebalance_needed = True
                    new_planned = base_asset
                    new_stage = 0
                    target_lev_pct = 0.0
                    dev_info = f" Dev200>={vxn_dev_filter:.2f}({cur_dev200:.2f})" if vxn_dev_filter > 0 else ""
                    rebalance_cause = f"VXN대피({vxn:.1f}>{exit_val}){dev_info}"
                    if not fast_mode:
                        signal_events.append({'date': date, 'type': 'VXN대피청산', 'reason': rebalance_cause, 'price': price, 'executed': False})

            # [신규] 모든 활성 세트(trade_slots)의 보유 기간 중 최저 수익률(MDD용) 갱신
            for slot in trade_slots:
                asset_to_check = slot.get('asset', leverage_asset)
                if asset_to_check in prices:
                    curr_slot_ret = (prices[asset_to_check] / slot['buy_price'] - 1) * 100
                    if 'worst_ret' not in slot or curr_slot_ret < slot['worst_ret']:
                        slot['worst_ret'] = curr_slot_ret
            
            # [신규] 개별 거래 세트 기준 손절 라인 (Set-based Stop Loss) 검사
            if smart_params and smart_params.get('use_set_sl', False):
                sl_limit = smart_params.get('set_sl_limit', -15.0)
                surviving_slots = []
                sl_hit_count = 0
                for slot in trade_slots:
                    asset_to_check = slot.get('asset', leverage_asset)
                    if asset_to_check in prices:
                        slot_ret = (prices[asset_to_check] / slot['buy_price'] - 1) * 100
                        if slot_ret <= sl_limit:
                            # 해당 세트 강제 청산 (손절)
                            shares_to_sell = slot.get('shares', 0)
                            if shares_to_sell > 0:
                                sell_revenue = shares_to_sell * prices[asset_to_check]
                                holdings[asset_to_check] = max(0.0, holdings.get(asset_to_check, 0.0) - shares_to_sell)
                                cash += sell_revenue
                                sl_hit_count += 1
                            
                            # 완료된 거래(closed_trades) 기록
                            sh_sl = slot.get('shares', 0.0)
                            sell_p_sl = prices[asset_to_check]
                            total_cost_sl = sh_sl * slot['buy_price']
                            total_rev_sl  = sh_sl * sell_p_sl
                            closed_trades.append({
                                'Entry_Date': slot['date'],
                                'Exit_Date': date,
                                'Asset': asset_to_check,
                                'Buy_Price': slot['buy_price'],
                                'Sell_Price': sell_p_sl,
                                'Shares': sh_sl, 'Total_Cost': total_cost_sl, 'Total_Revenue': total_rev_sl,
                                'Fee_Cost': (total_cost_sl + total_rev_sl) * fee_rate,
                                'Net_Gain': total_rev_sl - total_cost_sl - (total_cost_sl + total_rev_sl) * fee_rate,
                                'Return': slot_ret,
                                'MDD': slot.get('worst_ret', slot_ret),
                                'Status': '개별손절'
                            })
                            
                            # 시그널 이벤트에도 로깅 (UI 표시용)
                            if not fast_mode:
                                signal_events.append({
                                    'date': date,
                                    'type': '🛑개별손절',
                                    'reason': f"{asset_to_check} 수익률({slot_ret:.1f}%) < 손절라인({sl_limit:.1f}%)",
                                    'price': prices[asset_to_check],
                                    'executed': True
                                })
                        else:
                            surviving_slots.append(slot)
                    else:
                        surviving_slots.append(slot)
                
                trade_slots = surviving_slots
                # [수정] 손절 발생 시 스테이지 정보 동기화
                if sl_hit_count > 0:
                    rebalance_stage = max(0, rebalance_stage - sl_hit_count)
                    # stage가 변했으므로 target_lev_pct 등은 다음 루프에서 자동 갱신됨
                
                # 잔고가 변했으므로 스테이지 재계산 (current_lev_pct 등은 아래에서 다시 계산됨)
            
            if last_entry_price == 0:
                last_entry_price = prices[base_asset]

            # [신규] 동적 현금 비중 적용
            eff_cash_ratio = cash_ratio # 기본값 (고정 비중)
            if use_dc:
                cur_regime = row_arr[regime_idx]
                
                # [강화] 국면 전환 시 즉각 리밸런싱 강제 (지연 방지)
                if i > 0:
                    prev_regime = combined_matrix[i-1][regime_idx]
                    if cur_regime != prev_regime:
                        rebalance_needed = True
                        rebalance_cause = f"국면전환({prev_regime}->{cur_regime})"

                if _use_dev200_regime:
                    # VXN × Dev200 2차원 국면 현금비율 (hot/normal/caution/bear)
                    ratio_map = {
                        'hot':     dc_params.get('cash_hot',     0.0) / 100.0,
                        'normal':  dc_params.get('cash_normal',  0.0) / 100.0,
                        'caution': dc_params.get('cash_caution', 30.0) / 100.0,
                        'bear':    dc_params.get('cash_bear',    80.0) / 100.0,
                    }
                else:
                    # 기존 VXN 1차원 국면 현금비율 (bull/neutral/caution/bear)
                    ratio_map = {
                        'bull':    dc_params.get('cash_bull',    0.0) / 100.0,
                        'neutral': dc_params.get('cash_neutral', 20.0) / 100.0,
                        'caution': dc_params.get('cash_caution', 30.0) / 100.0,
                        'bear':    dc_params.get('cash_bear',    80.0) / 100.0,
                    }
                eff_cash_ratio = ratio_map.get(cur_regime, cash_ratio)

            etf_funds = current_total_val * (1 - eff_cash_ratio)
            lev_candidates = ['QLD', 'TQQQ']
            current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            current_lev_pct = current_lev_funds / etf_funds if etf_funds > 0 else 0

            # [신규] 국면별 Stage 비율 강제 조정 (regime_stage 파라미터)
            # current_lev_pct 계산 후 실행 → 현재 비중이 국면 허용치 초과 시 즉시 리밸런싱 트리거
            _rs_cfg = (params.get('regime_stage') or {}) if params else {}
            if _rs_cfg.get('use', False) and cur_dev200 > 0 and is_buy_process:
                _rs_c_vxn = _rs_cfg.get('caution_vxn', 30.0)
                _rs_c_dev = _rs_cfg.get('caution_dev', 1.05)
                _rs_b_vxn = _rs_cfg.get('bear_vxn', 35.0)
                _rs_b_dev = _rs_cfg.get('bear_dev', 0.95)
                if vxn >= _rs_b_vxn or cur_dev200 <= _rs_b_dev:
                    _rs_ratios = {3: _rs_cfg.get('bear_s3', 1.0), 2: _rs_cfg.get('bear_s2', 0.70), 1: _rs_cfg.get('bear_s1', 0.30)}
                elif vxn >= _rs_c_vxn and cur_dev200 < _rs_c_dev:
                    _rs_ratios = {3: _rs_cfg.get('caution_s3', 1.0), 2: _rs_cfg.get('caution_s2', 0.70), 1: _rs_cfg.get('caution_s1', 0.30)}
                else:
                    _rs_ratios = None
                if _rs_ratios and new_stage in _rs_ratios:
                    _rs_limit = _rs_ratios[new_stage]
                    if current_lev_pct > _rs_limit + 0.02:  # 현재 비중이 허용치 초과 시
                        target_lev_pct = _rs_limit
                        rebalance_needed = True

            # [리밸런싱 판정 우선순위 조정] 매입(신호/가격)이 매도(신호/가격)보다 우선함
            is_buy_process = (current_planned_asset != base_asset)
            is_mode_consistent = (current_planned_asset == effective_lev_asset)

            # [신규] ATR 샹들리에 엑시트를 위한 최고가(Peak Price) 실시간 갱신
            if is_buy_process:
                if peak_price == 0 or prices[base_asset] > peak_price:
                    peak_price = prices[base_asset]
            else:
                peak_price = 0 # 매도 프로세스 중에는 고점 추적 안함
            
            # [수정] 리밸런싱 트리거 판정 (하이브리드: 고정 % + ATR 변동성 병합)
            use_fixed = params.get('use_fixed_reb', True)
            use_atr = params.get('use_atr_reb', False)
            
            price_change = prices[base_asset] / last_entry_price
            price_diff_pct = (price_change - 1) * 100
            
            buy_price_hit = False
            sell_price_hit = False
            
            # (1) 시그널 매입 판정 (모드 공통)
            buy_signal_hit = is_buy_signal and not is_mode_consistent and effective_lev_asset in prices

            # [Step 1-4] 매수 인터락: VXN > X AND Dev200 > Y 동시 충족 시 매수 신호 차단
            # params: use_buy_interlock(bool), interlock_vxn(float), interlock_dev(float),
            #         interlock_min_stage(int, 1=전체/2=S2이상/3=S3만)
            if buy_signal_hit and params and params.get('use_buy_interlock', False):
                il_vxn = params.get('interlock_vxn', 25.0)
                il_dev = params.get('interlock_dev', 1.1)
                il_min_stage = params.get('interlock_min_stage', 1)
                if vxn > il_vxn and cur_dev200 > il_dev and rebalance_stage >= il_min_stage - 1:
                    buy_signal_hit = False
                    if not fast_mode:
                        signal_events.append({'date': date, 'type': '매수인터락', 'reason': f"VXN({vxn:.1f}>{il_vxn}) AND Dev200({cur_dev200:.3f}>{il_dev})", 'price': price, 'executed': False})

            # (2) 고정 비율 트리거 계산
            fixed_buy_hit = False
            fixed_sell_hit = False
            interlock_ok = True  # 기본값: 리밸런싱 허용
            
            if use_fixed:
                up_thresh = abs(b_up if current_planned_asset == leverage_asset else s_up)
                down_thresh = abs(b_down if current_planned_asset == leverage_asset else s_down)
                price_trigger = (price_change >= 1 + up_thresh or price_change <= 1 - down_thresh)
                
                # [신규] 리밸런싱 인터락 (이격도/VXN 조건)
                use_reb_interlock = params.get('use_reb_interlock', False)
                if use_reb_interlock:
                    reb_dev = params.get('reb_interlock_dev', 1.13)
                    reb_vxn = params.get('reb_interlock_vxn', 22.0)
                    # 이격도와 VXN이 동시에 과열(상향)/안정(하향) 임계값을 넘으면 추가 매수 차단
                    if cur_dev200 >= reb_dev and vxn >= reb_vxn:
                        interlock_ok = False

                fixed_buy_hit = (rebalance_stage in [1, 2] and is_buy_process and is_mode_consistent 
                                 and (price_change >= 1 + up_thresh) and interlock_ok)
                fixed_sell_hit = (rebalance_stage in [1, 2, 3] and not is_buy_process and price_trigger)

            # (3) ATR 트리거 계산
            atr_buy_hit = False
            atr_sell_hit = False
            if use_atr:
                # [수정] 매수/매도별 ATR 기준일 개별 적용 (기본값: 매수 20일, 매도 20일)
                period_b = params.get('atr_period_buy', 20)
                period_s = params.get('atr_period_sell', 20)
                
                prev_atr_buy = row.get(f'Prev_ATR_{period_b}', 0)
                prev_atr_sell = row.get(f'Prev_ATR_{period_s}', 0)
                
                k_up = params.get('atr_mult_buy_up', 10.0)
                k_down = params.get('atr_mult_buy_down', 1.5)
                k_sell = params.get('atr_mult_sell', 3.0)
                
                # ATR 매수 트리거 (상향/하향) - prev_atr_buy 사용
                hit_down = (prices[base_asset] <= last_entry_price - (k_down * prev_atr_buy))
                hit_up = (prices[base_asset] >= last_entry_price + (k_up * prev_atr_buy))
                atr_buy_hit = (rebalance_stage in [1, 2] and is_mode_consistent and (hit_down or hit_up))
                
                # ATR 샹들리에 엑시트 - prev_atr_sell 사용
                if is_buy_process and peak_price > 0:
                    atr_sell_hit = (prices[base_asset] <= peak_price - (k_sell * prev_atr_sell))
                atr_sell_hit = (rebalance_stage in [1, 2, 3] and atr_sell_hit)

            # (4) 최종 판정 (OR 결합)
            buy_price_hit = fixed_buy_hit or atr_buy_hit
            sell_price_hit = fixed_sell_hit or atr_sell_hit
            
            if buy_signal_hit or buy_price_hit:
                if buy_signal_hit:
                    new_planned = effective_lev_asset
                    if effective_lev_asset == base_asset:
                        target_lev_pct, new_stage = 0.0, 0
                    else:
                        if current_lev_pct < 0.25: target_lev_pct, new_stage = 0.30, 1
                        elif current_lev_pct < 0.65: target_lev_pct, new_stage = 0.70, 2
                        else:
                            if current_planned_asset != effective_lev_asset and effective_lev_asset == target_group[2] and current_lev_pct > 0.9:
                                curr_tqqq_w = (holdings.get(target_group[2], 0) * prices.get(target_group[2], 0)) / etf_funds if etf_funds > 0 else 0
                                if curr_tqqq_w < 0.25: target_lev_pct, new_stage = 1.0, 1
                                elif curr_tqqq_w < 0.65: target_lev_pct, new_stage = 1.0, 2
                                else: target_lev_pct, new_stage = 1.0, 3
                            else:
                                target_lev_pct, new_stage = 1.0, 3
                    rebalance_cause = "시그널"
                else:
                    # 가격 트리거에 의한 매입/상향 (1->2, 2->3)
                    new_stage = min(3, rebalance_stage + 1)
                    target_lev_pct = (0.70 if new_stage == 2 else 1.0)
                    if fixed_buy_hit and atr_buy_hit: rebalance_cause = "고정+ATR(매수)"
                    elif fixed_buy_hit: rebalance_cause = "고정(매수)"
                    else: rebalance_cause = "ATR(매수)"
                rebalance_needed = True
                
                if not buy_signal_hit and not fast_mode:
                    current_signal_ref = {'date': date, 'type': '가격변동', 'reason': rebalance_cause, 'price': price, 'executed': False}
                    signal_events.append(current_signal_ref)
                
            # (2) 매도 판정 (자산 교체 신호)
            elif is_sell_signal and current_planned_asset != base_asset:
                new_planned = base_asset
                # [수정] 점진적 매도 로직 복원 (3->2, 2->1, 1->0)
                if current_lev_pct > 0.75: target_lev_pct, new_stage = 0.70, 2
                elif current_lev_pct > 0.35: target_lev_pct, new_stage = 0.30, 1
                else: target_lev_pct, new_stage = 0.0, 0
                rebalance_cause = "시그널"
                rebalance_needed = True
                
            elif sell_price_hit:
                # 레버리지 비중 축소 (3->2, 2->1, 1->0)
                new_stage = max(0, rebalance_stage - 1)
                target_lev_pct = (0.70 if new_stage == 2 else (0.30 if new_stage == 1 else 0.0))
                
                if fixed_sell_hit and atr_sell_hit: rebalance_cause = "고정+ATR(매도)"
                elif fixed_sell_hit: rebalance_cause = "고정(매도)"
                else: rebalance_cause = "ATR(샹들리에)"
                rebalance_needed = True
                
                # [신규] 가격 변동에 의한 매도 리밸런싱 로깅
                if not fast_mode:
                    current_signal_ref = {'date': date, 'type': '가격변동', 'reason': rebalance_cause, 'price': price, 'executed': False}
                    signal_events.append(current_signal_ref)

            # [신규] S3(레버리지 100%) 전용 보호 조건 처리 (Emergency Defensive)
            # 여러 개의 신호 중 하나라도 충족되면(OR) 작동함
            s3_p_list = (params.get('s3_protection') or [])
            if not isinstance(s3_p_list, list): s3_p_list = [s3_p_list]
            
            if rebalance_stage == 3:
                for idx, s3_p in enumerate(s3_p_list, 1):
                    is_drop_ok = True
                    if s3_p.get('use_daily_drop'):
                        daily_ret = (price / row['Prev_Close'] - 1) * 100
                        is_drop_ok = (daily_ret <= s3_p.get('drop_limit', -3.0))
                    
                    is_ma60_ok = True
                    if s3_p.get('use_ma60'):
                        sma60 = row.get('SMA60', 0)
                        limit = s3_p.get('ma60_limit', 0.0)
                        is_ma60_ok = (price < sma60 * (1 + limit / 100.0))
                    
                    is_ma200_ok = True
                    if s3_p.get('use_ma200'):
                        sma200 = row.get('SMA200', 0)
                        limit = s3_p.get('ma200_limit', 0.0)
                        is_ma200_ok = (price < sma200 * (1 + limit / 100.0))

                    is_vix_ok = True
                    if s3_p.get('use_vxn_jump', s3_p.get('use_vix_jump', False)):
                        is_vix_ok = (vxn_jump_pct >= s3_p.get('vxn_jump', s3_p.get('vix_jump', 15.0)))
                    
                    is_gap_ok = True
                    if s3_p.get('use_gap_down'):
                        gap_down = row.get('Gap_Down', 0)
                        is_gap_ok = (gap_down <= s3_p.get('gap_limit', -3.0))
                    
                    is_acc_ok = True
                    if s3_p.get('use_drop_acc'):
                        drop_acc = row.get('Drop_Acc', 0)
                        is_acc_ok = (drop_acc <= s3_p.get('acc_limit', -7.0))
                    
                    # [신규] S3 보호 설정 전용: 샹들리에 엑시트 (고점 추적 매도)
                    is_s3_chan_ok = True
                    if s3_p.get('use_chandelier'):
                        period_s = params.get('atr_period_sell', 20)
                        prev_atr_sell = row.get(f'Prev_ATR_{period_s}', 0)
                        c_mult = float(s3_p.get('chandelier_mult', 3.0))
                        
                        hit_s3_chandelier = False
                        if (current_planned_asset != base_asset) and peak_price > 0:
                            hit_s3_chandelier = (prices[base_asset] <= peak_price - (c_mult * prev_atr_sell))
                        is_s3_chan_ok = hit_s3_chandelier

                    # S3 보호 조건 결합 방식 정교화: 
                    # 1. 보호 신호 1, 2, 3 '내부' 항목들은 모두 만족해야 함 (AND)
                    # 2. 보호 신호 1, 2, 3 '끼리'는 하나라도 만족하면 작동 (OR)
                    
                    active_conditions = []
                    if s3_p.get('use_daily_drop'): active_conditions.append(is_drop_ok)
                    if s3_p.get('use_ma60'): active_conditions.append(is_ma60_ok)
                    if s3_p.get('use_ma200'): active_conditions.append(is_ma200_ok)
                    if s3_p.get('use_vxn_jump', s3_p.get('use_vix_jump', False)): active_conditions.append(is_vix_ok)
                    if s3_p.get('use_gap_down'): active_conditions.append(is_gap_ok)
                    if s3_p.get('use_drop_acc'): active_conditions.append(is_acc_ok)
                    if s3_p.get('use_chandelier'): active_conditions.append(is_s3_chan_ok)
                    
                    # 하나라도 체크되어 있고, 체크된 모든 조건이 참(True)인 경우 작동
                    if len(active_conditions) > 0 and all(active_conditions):
                        dr = (price / row['Prev_Close'] - 1) * 100
                        rebalance_needed = True
                        new_planned = base_asset
                        
                        # 로그 상세화 (#N:신호번호, V:VIX, G:Gap, A:Acc, M60:MA60, M200:MA200, C:Chan)
                        detail_log = f"#{idx}"
                        if s3_p.get('use_daily_drop'): detail_log += f" D:{dr:+.1f}%"
                        if s3_p.get('use_ma60'): detail_log += f" M60({s3_p.get('ma60_limit', 0.0):+.1f}%)"
                        if s3_p.get('use_ma200'): detail_log += f" M200({s3_p.get('ma200_limit', 0.0):+.1f}%)"
                        if s3_p.get('use_vxn_jump', s3_p.get('use_vix_jump', False)): detail_log += f" VXN:{vxn_jump_pct:+.1f}%"
                        if s3_p.get('use_gap_down'): detail_log += f" G:{row.get('Gap_Down', 0):+.1f}%"
                        if s3_p.get('use_drop_acc'): detail_log += f" A:{row.get('Drop_Acc', 0):+.1f}%"
                        if s3_p.get('use_chandelier'): detail_log += f" C:ATR*{s3_p.get('chandelier_mult', 3.0)}"

                        s3_detail_val = detail_log
                        
                        if s3_p.get('use_exit_all', False):
                            new_stage = 0
                            target_lev_pct = 0.0
                            rebalance_cause = f"S3-전량매도({detail_log})"
                        else:
                            new_stage = 2
                            target_lev_pct = 0.70
                            rebalance_cause = f"S3-복합위기({detail_log})"
                        
                        s3_drop_triggered_val = dr 
                        
                        # [신규] S3 보호 작동 로깅
                        if not fast_mode:
                            current_signal_ref = {'date': date, 'type': '가격변동', 'reason': rebalance_cause, 'price': price, 'executed': False}
                            signal_events.append(current_signal_ref)
                        break # 신호 간에는 OR이므로 하나라도 충족되면 검사 종료

            passed_panic = False
            if rebalance_needed and new_planned != base_asset and smart_params and smart_params.get('use_panic', False):
                panic_ma = smart_params.get('panic_ma', 200)
                ma_val = combined.loc[date, f'SMA{panic_ma}']
                if prices.get(base_asset, price) < ma_val:
                    pb_list = smart_params.get('panic_buy_signals', [])
                    pb_idx = new_stage - 1 # 1->0, 2->1, 3->2
                    if pb_idx >= 0 and pb_idx < len(pb_list) and pb_list[pb_idx]:
                        # 독립적인 하락장 매수 시그널 (Panic Buy Signal S1, S2, S3) 검사
                        # [수정] Dev200 기반 RSI 조건 완화: dev200이 낮을수록(깊은 낙폭) panic_buy RSI 완화
                        # dev200_rsi_map = [{"dev200_max": 0.90, "rsi_val": 38}, ...] 형식
                        pb_cfg = pb_list[pb_idx]
                        dev200_rsi_map = pb_cfg.get('dev200_rsi_map', [])
                        if dev200_rsi_map and cur_dev200 > 0:
                            pb_cfg = dict(pb_cfg)  # 원본 불변 복사본 생성
                            for tier in sorted(dev200_rsi_map, key=lambda x: x['dev200_max'], reverse=True):
                                if cur_dev200 <= tier['dev200_max']:
                                    pb_cfg['rsi_val'] = tier['rsi_val']
                                    break
                        active, cond, pb_reasons = check_buy_cond(pb_cfg, pb_idx + 1, panic_buy_wait_flags)
                        if active and cond:
                            passed_panic = True
                            panic_buy_wait_flags[pb_idx + 1] = False # 충족 성공 시 대기(알람) 초기화
                            # [신규] 하락장 매수 지연 해제 로그 기록
                            current_signal_ref = {'date': date, 'type': f'하락장매수해제S{new_stage}', 'reason': " / ".join(pb_reasons) if pb_reasons else "기본조건충족", 'price': price, 'executed': False}
                            signal_events.append(current_signal_ref)
                        else:
                            rebalance_needed, is_delayed, delay_reason = False, True, f"하락장 대기(S{new_stage})"
                    else:
                        passed_panic = True
                
            # [신규] 손절제어 (Stop-Loss Control) 로직
            # 오직 '매도(비중 축소)' 상황에서만 손절제어를 체크함
            if rebalance_needed and not is_delayed and smart_params and smart_params.get('use_sl_control', False):
                # 1. 목표 레버리지 비중이 현재보다 낮거나(매도), 2. 안전자산(base_asset)으로의 완전 회귀 시 작동
                is_selling_move = (target_lev_pct < current_lev_pct - 0.01) or (new_planned == base_asset and current_planned_asset != base_asset)
                
                if is_selling_move:
                    if trade_slots:
                        # FIFO 기준으로 가장 오래된 물량의 수익률 체크
                        oldest_slot = trade_slots[0]
                        asset_to_check = oldest_slot.get('asset', leverage_asset)
                        if asset_to_check in prices:
                            curr_ret = (prices[asset_to_check] / oldest_slot['buy_price'] - 1) * 100
                            sl_limit = smart_params.get('sl_control_limit', -15)
                            if curr_ret <= sl_limit:
                                # 지연 사유에 원래의 리밸런싱 원인(시그널/가격조건)을 포함하여 투명성 강화
                                cause_info = f" [{rebalance_cause}]" if rebalance_cause else ""
                                rebalance_needed, is_delayed, delay_reason = False, True, f"손절제어{cause_info}({curr_ret:.1f}%)"
                                
                                # [신규] 손절제어 작동 로그 기록 (매도 차단됨)
                                if not fast_mode:
                                    signal_events.append({
                                        'date': date,
                                        'type': '🛡️손절제어(매도차단)',
                                        'reason': f"{asset_to_check} 수익률({curr_ret:.1f}%) <= 임계값({sl_limit:.1f}%){cause_info}",
                                        'price': prices[asset_to_check],
                                        'executed': False # 매도가 차단되었으므로 False
                                    })

            if rebalance_needed:
                if current_signal_ref:
                    current_signal_ref['executed'] = True

                h_vals = {t: holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates}
                curr_total_lev_val = sum(h_vals.values())

                # [수수료] 거래 발생 시 포트폴리오 총액에서 수수료 선차감
                if fee_rate > 0:
                    next_lev_val = etf_funds * target_lev_pct
                    trade_val = abs(next_lev_val - curr_total_lev_val)
                    fee_deduct = trade_val * fee_rate
                    current_total_val = max(0, current_total_val - fee_deduct)
                    cash = current_total_val * eff_cash_ratio
                    etf_funds = current_total_val * (1 - eff_cash_ratio)

                target_lev_val = etf_funds * target_lev_pct
                new_h_vals = {t: 0.0 for t in holdings}

                if curr_total_lev_val > target_lev_val + 0.01:
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                    rem = target_lev_val - new_h_vals[target_group[1]]
                    new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                else:
                    tqqq_cap = target_lev_val
                    if target_lev_val > etf_funds * 0.9 and effective_lev_asset == target_group[2]:
                        if new_stage == 1: tqqq_cap = 0.30 * target_lev_val
                        elif new_stage == 2: tqqq_cap = 0.70 * target_lev_val

                    is_upgrade_move = (target_lev_val > etf_funds * 0.9 and new_stage in [1, 2])
                    if effective_lev_asset == target_group[2] and is_upgrade_move:
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0) + h_vals.get(target_group[1], 0), tqqq_cap)
                    else:
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), tqqq_cap)

                    rem = target_lev_val - new_h_vals[target_group[2]]
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)

                    rem = target_lev_val - (new_h_vals[target_group[1]] + new_h_vals[target_group[2]])
                    if rem > 0:
                        new_h_vals[effective_lev_asset] += rem

                new_h_vals[base_asset] = max(0, etf_funds - sum(new_h_vals.values()))

                # [신규] 물량 추가량 추적을 위한 이전 잔고 캡처
                prev_h_counts = {t: holdings.get(t, 0) for t in check_targets}

                # 수량 업데이트 (정수 주식 처리 포함)
                updated, cash = apply_shares(
                    {t: new_h_vals.get(t, 0) for t in check_targets},
                    prices, cash
                )
                for t in check_targets:
                    holdings[t] = updated.get(t, 0.0)

                # [신규] 이번 매매로 실제 추가된 수량
                inc_h_counts = {t: max(0.0, holdings.get(t, 0) - prev_h_counts.get(t, 0)) for t in check_targets}

                old_v = get_v_level(current_planned_asset, rebalance_stage)
                new_v = get_v_level(new_planned, new_stage)

                if new_v > old_v:
                    num_new = new_v - old_v
                    shares_per_slot = inc_h_counts.get(new_planned, 0) / num_new if num_new > 0 else 0
                    for _ in range(num_new):
                        trade_slots.append({
                            'buy_price': prices[new_planned],
                            'date': date,
                            'asset': new_planned,
                            'shares': shares_per_slot,
                            'worst_ret': 0.0
                        })
                elif new_v < old_v:
                    diff = old_v - new_v
                    rets = []
                    entry_dates = []
                    for _ in range(diff):
                        if trade_slots:
                            slot = trade_slots.pop(0)
                            asset_sold = slot.get('asset', leverage_asset)
                            sell_p = prices[asset_sold]
                            r = (sell_p / slot['buy_price'] - 1) * 100
                            rets.append(r)
                            entry_dates.append(slot['date'])
                            sh = slot.get('shares', 0.0)
                            total_cost = sh * slot['buy_price']
                            total_rev  = sh * sell_p
                            closed_trades.append({
                                'Entry_Date': slot['date'], 'Exit_Date': date, 'Asset': asset_sold,
                                'Buy_Price': slot['buy_price'], 'Sell_Price': sell_p,
                                'Shares': sh, 'Total_Cost': total_cost, 'Total_Revenue': total_rev,
                                'Fee_Cost': (total_cost + total_rev) * fee_rate,
                                'Net_Gain': total_rev - total_cost - (total_cost + total_rev) * fee_rate,
                                'Return': r,
                                'MDD': slot.get('worst_ret', min(0.0, r)),
                                'Status': "익절" if r > 0 else "손절"
                            })
                    if rets:
                        trade_ret_val = np.mean(rets)
                        trade_status_val = "익절" if trade_ret_val > 0 else "손절"
                        trade_entry_date_val = ", ".join([d.strftime('%Y-%m-%d') for d in entry_dates])

                current_planned_asset = new_planned
                rebalance_stage = new_stage
                last_entry_price = prices[base_asset]
                peak_price = prices[base_asset]

                # [추가] 매매 후 현금 및 레버리지 자산 총액 재계산 (로그용)
                cash = current_total_val * eff_cash_ratio
                current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)

            if not fast_mode:
                # 시각적 가독성을 위해 trade_label 설정
                if rebalance_needed:
                    prefix = "매도" if new_planned == base_asset else "매수"
                    if rebalance_cause == "시그널":
                        condition_label = signal_reason
                    elif "ATR" in rebalance_cause:
                        if "매도" in rebalance_cause or "샹들리에" in rebalance_cause:
                            disp_atr = row_arr[col_idx[f"Prev_ATR_{params.get('atr_period_sell', 20)}"]] if f"Prev_ATR_{params.get('atr_period_sell', 20)}" in col_idx else 0
                        else:
                            disp_atr = row_arr[col_idx[f"Prev_ATR_{params.get('atr_period_buy', 20)}"]] if f"Prev_ATR_{params.get('atr_period_buy', 20)}" in col_idx else 0
                        condition_label = f"{rebalance_cause} (ATR:{disp_atr:.2f})"
                    elif "고정" in rebalance_cause:
                        condition_label = f"{rebalance_cause} ({price_diff_pct:+.1f}%)"
                    else:
                        condition_label = rebalance_cause
                    
                    if passed_panic:
                        # [수정] 상수값(27, 28, 30) 대신 실제 config에 설정된 rsi_val 사용
                        pb_cfg = pb_list[pb_idx]
                        limit = pb_cfg.get('rsi_val', 30)
                        condition_label = f"RSI조건(RSI<{limit})"

                    trade_label = f"{prefix}(S{new_stage}): {condition_label}"
                    if smart_params and smart_params.get('use_smart') and smart_reason:
                        trade_label += f" [{smart_reason}]"
                elif is_delayed:
                    trade_label = f"지연({delay_reason})"
                    if delay_reason.startswith("MA"):
                        trade_label = f"지연({delay_reason}:{rsi:.1f})"
                else:
                    state_label = f"진행중(S{rebalance_stage})" if 0 < rebalance_stage < 3 else "중립"
                    if is_buy_signal or is_sell_signal:
                        sig_type = "매수시그널" if is_buy_signal else "매도시그널"
                        trade_label = f"{state_label}: {sig_type}({signal_reason})"
                    else:
                        trade_label = state_label

            # [수정] Fast Mode일 경우 최적화 루틴용 최소 이력만 저장
            if fast_mode:
                history.append({
                    'Date': date,
                    'Value': current_total_val,
                    'Stage': int(rebalance_stage)
                })
            else:
                summary_p = []
                for t in target_group[:0:-1]: # TQQQ, QLD 순기록
                    w_val = (holdings.get(t, 0) * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
                    if w_val >= 0.01: summary_p.append(f"{t} {w_val*100:.0f}%")
                b_w = (holdings.get(base_asset, 0) * prices.get(base_asset, 0)) / current_total_val if current_total_val > 0 else 0
                if b_w >= 0.01: summary_p.append(f"{base_asset} {b_w*100:.0f}%")
                
                # [수정] 맵핑용 레버리지 자산 총액 및 현금 비중 실시간 재계산 (로그 정확도 향상)
                log_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
                log_cash_weight = cash / current_total_val if current_total_val > 0 else 0
                log_lev_weight = log_lev_funds / current_total_val if current_total_val > 0 else 0
    
                # [수정] 자산 명칭과 Stage를 로그 기록 시점에 동기화
                log_asset_name = current_planned_asset
                if 0 < rebalance_stage < 3:
                    log_asset_name = f"{current_planned_asset}(S{int(rebalance_stage)})"
                elif rebalance_stage == 3:
                    # S3는 레버리지 자산일 때만 명시적으로 표시 (base_asset이면 그냥 표시)
                    if current_planned_asset != base_asset:
                        log_asset_name = f"{current_planned_asset}(S3)"
    
                # [신규] 개별 지표 시그널 상태 진단 (로그용)
                rsi_sig = "-"
                if params:
                    if rsi <= params.get('rsi_lower', 35): rsi_sig = "과매도"
                    elif rsi >= params.get('rsi_upper', 70): rsi_sig = "과매수"
                    if is_rsi_golden_cross: rsi_sig = "골든크로스"
                    elif is_rsi_dead_cross: rsi_sig = "데드크로스"
                
                macd_sig = "Neutral"
                if is_macd_golden_cross: macd_sig = "골든크로스"
                elif is_macd_dead_cross: macd_sig = "데드크로스"
                elif macd_hist > 0: macd_sig = "양선"
                else: macd_sig = "음선"
                
                willr_sig = "-"
                if willr_val <= -80: willr_sig = "과매도"
                elif willr_val >= -20: willr_sig = "과매수"
                if (prev_willr <= -80) and (willr_val > -80): willr_sig = "과매도이탈"
                elif (prev_willr >= -20) and (willr_val < -20): willr_sig = "과매수이탈"
    
                history.append({
                    'Date': date,
                    'Value': current_total_val,
                    'Cash_Weight': log_cash_weight,
                    'Lev_Weight': log_lev_weight,
                    'Stage': int(rebalance_stage),
                    'Regime': row_arr[regime_idx] if use_dc else 'bull',
                    'Asset': log_asset_name,
                    'Drop_Acc': row_arr[col_idx['Drop_Acc']] if 'Drop_Acc' in col_idx else 0,
                    'Gap_Down': row_arr[col_idx['Gap_Down']] if 'Gap_Down' in col_idx else 0,
                    'S3_Drop_Limit': s3_drop_triggered_val, # [기록]
                    'S3_Sell3_Event': s3_sell3_event_val, # [기록]
                    'S3_Detail': s3_detail_val, # [신규 추가]
                    'Trade_Label': trade_label,
                    'Summary': ", ".join(summary_p),
                    'Trade_Ret': trade_ret_val,
                    'Trade_Status': trade_status_val,
                    'Trade_Entry_Date': trade_entry_date_val,
                    'Close': prices[base_asset],
                    'RSI': rsi,
                    'RSI_Sig': rsi_sig,
                    'MACD': macd_val,
                    'MACD_Sig': macd_sig,
                    'WILLR_Sig': willr_sig,
                    'FG': fg,
                    'VIX': vxn, # [수정] VIX 컬럼 요청에 따라 VXN 값으로 대체 (호환성)
                    'VXN': vxn,
                    'ADX': row_arr[col_idx['ADX']] if 'ADX' in col_idx else 0,
                    'DMP': row_arr[col_idx['DMP']] if 'DMP' in col_idx else 0,
                    'DMN': row_arr[col_idx['DMN']] if 'DMN' in col_idx else 0,
                    'WILLR': row_arr[col_idx['WILLR']] if 'WILLR' in col_idx else 0,
                    'SAR': row_arr[col_idx['SAR']] if 'SAR' in col_idx else 0,
                    'News_Impact': row_arr[col_idx['News_Impact']] if 'News_Impact' in col_idx else 0,
                    'News_Q': row_arr[news_q_idx] if news_q_idx >= 0 else 0.5,
                    'News_Z': row_arr[news_z_idx] if news_z_idx >= 0 else 0,
                    'Smart_Mode': smart_reason,
                    'SMA200': row_arr[sma200_idx] if sma200_idx >= 0 else 0,
                    'SMA60': row_arr[sma60_idx] if sma60_idx >= 0 else 0,
                    'Target': f"{new_planned}(S{new_stage})" if new_stage < 3 else new_planned,
                    'STOCH_K': row_arr[col_idx['STOCH_K']] if 'STOCH_K' in col_idx else 0,
                    'STOCH_D': row_arr[col_idx['STOCH_D']] if 'STOCH_D' in col_idx else 0,
                    'BB_Lower': bb_l,
                    'BB_Upper': bb_u,
                    'Signal_Line': signal_line,
                    'MACD_Hist': macd_hist,
                    'ATR_14': row_arr[col_idx['ATR_14']] if 'ATR_14' in col_idx else 0,
                    'ATR_20': row_arr[col_idx['ATR_20']] if 'ATR_20' in col_idx else 0,
                    'Peak_Price': peak_price
                })
                for t in lev_candidates:
                    if t in holdings:
                        history[-1][f'{t}_Weight'] = (holdings[t] * prices.get(t, 0)) / current_total_val if current_total_val > 0 else 0
                history[-1][f'{base_asset}_Weight'] = (holdings.get(base_asset, 0) * prices.get(base_asset, 0)) / current_total_val if current_total_val > 0 else 0

            # [신규] 하루 단위 진행 후 외부 자금 수신 대기 (Generator + send 패턴)
            state_dict = {
                'date': date,
                'cash': cash,
                'portfolio_value': current_total_val,
                'stage': int(rebalance_stage),
                'history_row': history[-1] if history else None,
                'holdings': dict(holdings)
            }
            injected = yield state_dict
            if injected and isinstance(injected, dict) and 'cash_delta' in injected:
                delta = injected['cash_delta']
                if delta >= 0:
                    # 자금 수혈: 그대로 현금 추가
                    cash += delta
                    current_total_val += delta
                else:
                    # 자금 환수: 현금 우선 차감, 부족분은 포지션 비례 강제 매도
                    withdraw = abs(delta)
                    if cash >= withdraw:
                        cash -= withdraw
                        current_total_val -= withdraw
                    else:
                        # 현금으로 충당 가능한 만큼 먼저 사용
                        remaining = withdraw - cash
                        cash = 0.0
                        # 보유 포지션 전체 평가액 계산
                        holdings_val = sum(holdings[t] * prices.get(t, 0) for t in holdings if prices.get(t, 0) > 0)
                        if holdings_val > 0:
                            sell_ratio = min(remaining / holdings_val, 1.0)
                            for t in list(holdings.keys()):
                                p = prices.get(t, 0)
                                if p > 0:
                                    shares_to_sell = holdings[t] * sell_ratio
                                    holdings[t] -= shares_to_sell
                                    cash += shares_to_sell * p  # 실제 매도금 중 remaining 초과분은 현금으로 잔류
                            cash -= remaining  # 환수 금액만큼 제거 (매도금 > remaining인 경우 잔류분 유지)
                            cash = max(cash, 0.0)
                        current_total_val = cash + sum(holdings[t] * prices.get(t, 0) for t in holdings if prices.get(t, 0) > 0)

        if not history:
            return pd.DataFrame(columns=['Value', 'Stage', 'Asset', 'Trade_Label']), pd.DataFrame(), []
        return pd.DataFrame(history).set_index('Date'), pd.DataFrame(closed_trades), signal_events

    @staticmethod
    def run_bond_strategy(data_dict, macro_df, start_date=None, end_date=None, params=None, allocated_capital=10000.0, lev_params=None, fee_rate=0.0, integer_shares=False, **kwargs):
        """Bond Strategy V7 구현 (TLT/TBF/BIL) - generator 기반 통합 호출"""
        gen = StrategyEngine._run_bond_generator(data_dict, macro_df, start_date, end_date, params, allocated_capital, lev_params=lev_params, fee_rate=fee_rate, integer_shares=integer_shares)
        try:
            state = next(gen)
            while True:
                # Standalone 실행 시에는 주입되는 자금이 없으므로 None 전달
                state = gen.send(None)
        except StopIteration as e:
            return e.value

    @staticmethod
    def _run_bond_generator(data_dict, macro_df, start_date=None, end_date=None, params=None, allocated_capital=10000.0, lev_params=None, fee_rate=0.0, integer_shares=False):
        """Bond Strategy V7+V8 구현 (Generator 버전)

        Layer 1 (V7): 매크로 신호로 기초자산 결정 (TLT / TBF / BIL)
        Layer 2 (V8): 기초자산 기준 buy/sell 신호 + 가격 트리거로 레버리지 스테이지 결정
                      TLT → TMF (S1:30% / S2:70% / S3:100%)
                      TBF → TMV (S1:30% / S2:70% / S3:100%)
        """
        # 1. 데이터 준비
        if 'TLT' not in data_dict or macro_df.empty:
            return pd.DataFrame(), pd.DataFrame(), []

        tlt_df = data_dict['TLT']
        tbf_df = data_dict.get('TBF', pd.DataFrame())
        bil_df = data_dict.get('BIL', pd.DataFrame())
        tmf_df = data_dict.get('TMF', pd.DataFrame())
        tmv_df = data_dict.get('TMV', pd.DataFrame())

        # 가격 데이터 병합 (Close 컬럼 존재 여부 확인)
        close_col = 'Close' if 'Close' in tlt_df.columns else tlt_df.columns[0] if len(tlt_df.columns) > 0 else 'Close'
        combined = tlt_df[[close_col]].copy().rename(columns={close_col: 'TLT'})

        if not tbf_df.empty:
            tbf_close = 'Close' if 'Close' in tbf_df.columns else tbf_df.columns[0] if len(tbf_df.columns) > 0 else 'Close'
            combined['TBF'] = tbf_df[tbf_close]
        if not bil_df.empty:
            bil_close = 'Close' if 'Close' in bil_df.columns else bil_df.columns[0] if len(bil_df.columns) > 0 else 'Close'
            combined['BIL'] = bil_df[bil_close]
        if not tmf_df.empty:
            tmf_close = 'Close' if 'Close' in tmf_df.columns else tmf_df.columns[0] if len(tmf_df.columns) > 0 else 'Close'
            combined['TMF'] = tmf_df[tmf_close]
        if not tmv_df.empty:
            tmv_close = 'Close' if 'Close' in tmv_df.columns else tmv_df.columns[0] if len(tmv_df.columns) > 0 else 'Close'
            combined['TMV'] = tmv_df[tmv_close]
        combined = combined.join(macro_df, how='left').ffill()

        # ── Layer 1 지표 (V7용) ──
        if 'tlt_rsi' not in combined.columns:
            combined['tlt_rsi'] = ta.rsi(combined['TLT'], length=14)

        # ── Layer 2 지표 (V8용): 기초자산(TLT/TBF) 기준 RSI·MACD ──
        if lev_params is not None:
            _tlt_rsi     = ta.rsi(combined['TLT'], length=14).fillna(50)
            _tlt_rsi_sma = _tlt_rsi.rolling(9).mean().fillna(50)
            _tlt_macd_df = ta.macd(combined['TLT'], fast=12, slow=26, signal=9)
            combined['_tlt_rsi']      = _tlt_rsi
            combined['_tlt_rsi_sma']  = _tlt_rsi_sma
            combined['_tlt_macd']     = _tlt_macd_df['MACD_12_26_9'] if _tlt_macd_df is not None else 0
            combined['_tlt_macd_sig'] = _tlt_macd_df['MACDs_12_26_9'] if _tlt_macd_df is not None else 0
            combined['_tlt_macd_h']   = _tlt_macd_df['MACDh_12_26_9'] if _tlt_macd_df is not None else 0

            if 'TBF' in combined.columns:
                _tbf_rsi     = ta.rsi(combined['TBF'], length=14).fillna(50)
                _tbf_rsi_sma = _tbf_rsi.rolling(9).mean().fillna(50)
                _tbf_macd_df = ta.macd(combined['TBF'], fast=12, slow=26, signal=9)
                combined['_tbf_rsi']      = _tbf_rsi
                combined['_tbf_rsi_sma']  = _tbf_rsi_sma
                combined['_tbf_macd']     = _tbf_macd_df['MACD_12_26_9'] if _tbf_macd_df is not None else 0
                combined['_tbf_macd_sig'] = _tbf_macd_df['MACDs_12_26_9'] if _tbf_macd_df is not None else 0
                combined['_tbf_macd_h']   = _tbf_macd_df['MACDh_12_26_9'] if _tbf_macd_df is not None else 0

        # 파라미터 로드
        _bond_defaults = {'ffv_lo': -1.0, 'ffv_hi': 1.5, 'yc_inv': -0.3, 'yc_steep': 0.4, 'tlt_rsi_max': 60,
                          'dot_hi': None, 'dot_lo': None,   # None = 비활성 (기존 동작 유지)
                          'cycle_use': False,                # F1v2 금리 사이클 감지 활성화
                          'cycle_cut_depth': -0.50,         # 6개월 누적 인하폭 기준 (%)
                          'cycle_hike_depth':  0.50,        # 6개월 누적 인상폭 기준 (%)
                          'cycle_streak':       60,         # 인하/인상 지속 최소 일수
                          'fomc_use': True,                  # FOMC 회의록 감성 분석 활성화 (그리드 스캔 검증 완료)
                          'fomc_hi':  0.5,                  # 매파 임계값: fomc_sentiment > fomc_hi → final_state 강제 RISING
                          'fomc_lo': -0.3}                  # 비둘기 임계값: fomc_sentiment < fomc_lo → final_state 강제 FALLING (그리드 최적값)
        if params is None or 'ffv_lo' not in params:
            params = {**_bond_defaults, **(params or {})}

        # ─ F1: FF금리 상태머신 ─
        if 'ff_rate' not in combined.columns:
            print("[Bond] macro_df에 ff_rate 없음 → Bond 전략 스킵 (FRED 데이터 필요)")
            return pd.DataFrame(), pd.DataFrame(), []
        combined['ff_1m_change'] = combined['ff_rate'].diff(21)
        f1_states = []
        current_f1, days_frozen = 'NEUTRAL', 0
        for i in range(len(combined)):
            if i > 0:
                days_frozen = (days_frozen + 1 if combined['ff_rate'].iloc[i] == combined['ff_rate'].iloc[i - 1] else 0)
            chg = combined['ff_1m_change'].iloc[i]
            if chg > 0.10: current_f1 = 'RISING'; days_frozen = 0
            elif chg < -0.10: current_f1 = 'FALLING'; days_frozen = 0
            elif current_f1 == 'RISING' and days_frozen >= 21: current_f1 = 'NEUTRAL'
            f1_states.append(current_f1)
        combined['f1_state'] = f1_states

        # ─ F1v2: 금리 인하/인상 사이클 감지 (cycle_use=True 일 때 f1_state 강화) ─
        # 판정 기준: ff_rate(현재) vs ff_rate(streak일 전) 비교
        #   현재 금리 < streak일 전 금리 + cut_thr  → 인하 사이클 확인 → FALLING
        #   현재 금리 > streak일 전 금리 + hike_thr → 인상 사이클 확인 → RISING
        # FOMC는 월 1~2회만 변경하므로 "전일 대비" streak가 아닌 "N일 전 대비" 방식 사용
        if params.get('cycle_use', False):
            _cut_thr   = params.get('cycle_cut_depth',  -0.50)
            _hike_thr  = params.get('cycle_hike_depth',  0.50)
            _streak_th = int(params.get('cycle_streak',  60))
            ff         = combined['ff_rate']
            ff_past    = ff.shift(_streak_th)          # streak일 전 금리
            ff_delta   = ff - ff_past                  # 현재 - N일 전
            falling_mask = ff_delta < _cut_thr         # 인하 사이클
            rising_mask  = ff_delta > _hike_thr        # 인상 사이클
            f1v2 = combined['f1_state'].copy()
            f1v2[falling_mask] = 'FALLING'
            f1v2[rising_mask]  = 'RISING'
            combined['f1_state'] = f1v2

        # ─ F2/F3 필수 컬럼 존재 확인 → 없으면 전략 중단 (잘못된 신호 방지) ─
        required_macro = ['core_cpi', 'ppi', 'unrate', 'nrou']
        missing = [c for c in required_macro if c not in combined.columns]
        if missing:
            print(f"[Bond] FRED 필수 데이터 누락 {missing} → Bond 전략 스킵 (FRED API 확인 필요)")
            return pd.DataFrame(), pd.DataFrame(), []

        # ─ F2: 인플레이션 ─
        combined['cpi_yoy'] = combined['core_cpi'].pct_change(252).ffill()
        combined['cpi_accel'] = combined['cpi_yoy'].diff(63)
        combined['ppi_3m_avg'] = combined['ppi'].rolling(window=63).mean()
        combined['ppi_dir'] = combined['ppi_3m_avg'].diff(1)
        combined['f2_state'] = np.where((combined['cpi_accel'] < 0) & (combined['ppi_dir'] < 0), 'FALLING',
                                        np.where((combined['cpi_accel'] > 0) & (combined['ppi_dir'] > 0), 'RISING', 'NEUTRAL'))

        # ─ F3: 고용 ─
        combined['emp_gap'] = combined['unrate'] - combined['nrou']
        combined['emp_gap_3m_chg'] = combined['emp_gap'].diff(63)
        combined['f3_state'] = np.where((combined['emp_gap'] > 0.3) | (combined['emp_gap_3m_chg'] > 0), 'FALLING',
                                        np.where(combined['emp_gap'] < -0.2, 'RISING', 'NEUTRAL'))

        # ─ 최종 상태 투표 ─
        def vote_state(row):
            votes = [row['f1_state'], row['f2_state'], row['f3_state']]
            # F4: 점도표 Dot Score (dot_hi/dot_lo 파라미터가 설정된 경우에만 활성)
            _dot_hi = params.get('dot_hi')
            _dot_lo = params.get('dot_lo')
            if _dot_hi is not None and _dot_lo is not None:
                ds = row.get('dot_score', np.nan)
                if not (ds != ds):  # not NaN
                    if ds > _dot_hi:   votes.append('RISING')
                    elif ds < _dot_lo: votes.append('FALLING')
                    else:              votes.append('NEUTRAL')
            from collections import Counter
            c = Counter(votes); top = c.most_common(1)[0]
            return top[0] if top[1] >= 2 else 'NEUTRAL'
        combined['final_state'] = combined.apply(vote_state, axis=1)

        # ─ FOMC 감성 외부 채널 (fomc_use=True 일 때만 활성, 투표 구조 외부 독립 override) ─
        # fomc_sentiment > fomc_hi  → 매파 확인 → RISING  강제 (채권 회피)
        # fomc_sentiment < fomc_lo  → 비둘기 확인 → FALLING 강제 (채권 선호)
        # 그 외 구간 → final_state 유지 (개입 없음)
        if params.get('fomc_use', False) and 'fomc_sentiment' in combined.columns:
            _fomc_hi = params.get('fomc_hi',  0.5)
            _fomc_lo = params.get('fomc_lo', -0.5)
            _sent = combined['fomc_sentiment'].ffill()
            hawkish_mask = _sent > _fomc_hi
            dovish_mask  = _sent < _fomc_lo
            combined.loc[hawkish_mask, 'final_state'] = 'RISING'
            combined.loc[dovish_mask,  'final_state'] = 'FALLING'

        if 'yc_mom' not in combined.columns and 't10y2y' in combined.columns:
            combined['yc_mom'] = combined['t10y2y'].diff(63)

        # ── 날짜 범위 적용 ──
        if start_date or end_date:
            full_idx = combined.index
            if start_date:
                sd = pd.Timestamp(start_date)
                full_idx = full_idx[full_idx >= sd]
            if end_date:
                ed = pd.Timestamp(end_date)
                full_idx = full_idx[full_idx <= ed]
            combined = combined.reindex(full_idx)

        dates = combined.index

        # ── V8 레버리지 파라미터 ──
        _use_lev   = lev_params is not None
        _reb_up    = abs(lev_params.get('buy_reb_up',    0.02)) if _use_lev else 0.02
        _reb_dn    = abs(lev_params.get('buy_reb_down',  0.02)) if _use_lev else 0.02
        _buy_sigs  = lev_params.get('buy_signals', [])          if _use_lev else []
        _sell_sigs = lev_params.get('sell_signals', [])         if _use_lev else []
        # sell_mode: 'instant'(기본) / 'step'(단계적) / 'floor_s1'(S1 후퇴 절충)
        _sell_mode = lev_params.get('sell_mode', 'instant')     if _use_lev else 'instant'

        # ── 헬퍼: buy_signals / sell_signals 평가 (기초자산 기준) ──
        def _check_buy(pfx, row, prev_row):
            """lev_params buy_signals 조건을 pfx(tlt/tbf) 지표로 평가"""
            rsi      = row.get(f'_{pfx}_rsi', 50)
            prev_rsi = prev_row.get(f'_{pfx}_rsi', 50)
            rsi_sma  = row.get(f'_{pfx}_rsi_sma', 50)
            prev_rsi_sma = prev_row.get(f'_{pfx}_rsi_sma', 50)
            macd_h   = row.get(f'_{pfx}_macd_h', 0)
            macd_val = row.get(f'_{pfx}_macd', 0)
            prev_macd= prev_row.get(f'_{pfx}_macd', 0)
            is_rsi_cross = (prev_rsi < prev_rsi_sma) and (rsi >= rsi_sma)
            for sig in _buy_sigs:
                hit = True
                if sig.get('rsi_val', 0) > 0 and not sig.get('rsi_cross', False):
                    if rsi > sig['rsi_val']: hit = False
                if sig.get('rsi_cross', False):
                    if not is_rsi_cross: hit = False
                if sig.get('macd_inc', False):
                    if macd_val <= prev_macd: hit = False
                if sig.get('macd_signal_below', False):
                    if macd_h >= 0: hit = False
                if hit:
                    return True
            return False

        def _check_sell(pfx, row, prev_row):
            """lev_params sell_signals 조건을 pfx(tlt/tbf) 지표로 평가"""
            rsi      = row.get(f'_{pfx}_rsi', 50)
            prev_rsi = prev_row.get(f'_{pfx}_rsi', 50)
            rsi_sma  = row.get(f'_{pfx}_rsi_sma', 50)
            prev_rsi_sma = prev_row.get(f'_{pfx}_rsi_sma', 50)
            macd_h   = row.get(f'_{pfx}_macd_h', 0)
            prev_macd_h = prev_row.get(f'_{pfx}_macd_h', 0)
            macd_val = row.get(f'_{pfx}_macd', 0)
            prev_macd = prev_row.get(f'_{pfx}_macd', 0)
            is_rsi_dead = (prev_rsi > prev_rsi_sma) and (rsi < rsi_sma)
            is_macd_dead = (prev_macd_h > 0) and (macd_h <= 0)
            for sig in _sell_sigs:
                hit = True
                if sig.get('rsi_val', 0) > 0:
                    if rsi < sig['rsi_val']: hit = False
                if sig.get('rsi_dec', False):
                    if rsi >= prev_rsi: hit = False
                if sig.get('rsi_dead', False):
                    if not is_rsi_dead: hit = False
                if sig.get('macd_dead', False):
                    if not is_macd_dead: hit = False
                if sig.get('macd_dec', False):
                    if macd_val >= prev_macd: hit = False
                if sig.get('macd_signal_above', False):
                    if macd_h <= 0: hit = False
                if hit:
                    return True
            return False

        # ── 시뮬레이션 상태 변수 ──
        results = []
        portfolio_value  = allocated_capital
        current_asset    = 'BIL'
        current_lev_frac = 0.0   # lev_frac 변경 감지용
        bond_holdings    = {'TLT': 0.0, 'TMF': 0.0, 'TBF': 0.0, 'TMV': 0.0, 'BIL': 0.0}
        bond_cash        = allocated_capital  # 초기엔 전액 현금 (BIL 상태)
        # 정수 주식 모드: 실제 투자분(시장 수익 적용)과 잔여 현금(수익 없음) 분리 추적
        bond_invest      = allocated_capital  # 시장 수익률이 적용되는 투자 원금
        bond_cash_reserve = 0.0              # 정수화 후 남은 잔여 현금 (수익률 없음)

        def _bond_rebalance(budget, target_asset, lev_fraction, lev_key, row_prices):
            """budget을 target_asset + lev_key 비율로 나눠 수량 재계산. (holdings, cash) 반환.
            Greedy Fill: integer_shares=True 시 잔여 현금으로 레버리지 자산 추가 매수 (현금 최소화).
            """
            new_h = {'TLT': 0.0, 'TMF': 0.0, 'TBF': 0.0, 'TMV': 0.0, 'BIL': 0.0}
            residual = 0.0
            active_assets = []  # Greedy Fill 대상 자산 (가격 내림차순)
            if target_asset == 'BIL':
                px = row_prices.get('BIL', np.nan)
                if not np.isnan(px) and px > 0:
                    raw = budget / px
                    sh = math.floor(raw) if integer_shares else raw
                    new_h['BIL'] = sh
                    residual = budget - sh * px
                    active_assets = [('BIL', px)]
                else:
                    residual = budget
            elif lev_key and lev_fraction > 0:
                base_key = 'TLT' if lev_key == 'TMF' else 'TBF'
                lev_budget  = budget * lev_fraction
                base_budget = budget * (1 - lev_fraction)
                px_lev  = row_prices.get(lev_key, np.nan)
                px_base = row_prices.get(base_key, np.nan)
                if not np.isnan(px_lev) and px_lev > 0:
                    raw_lev = lev_budget / px_lev
                    sh_lev = math.floor(raw_lev) if integer_shares else raw_lev
                    new_h[lev_key] = sh_lev
                    residual += lev_budget - sh_lev * px_lev
                    active_assets.append((lev_key, px_lev))
                else:
                    residual += lev_budget
                if not np.isnan(px_base) and px_base > 0:
                    raw_base = base_budget / px_base
                    sh_base = math.floor(raw_base) if integer_shares else raw_base
                    new_h[base_key] = sh_base
                    residual += base_budget - sh_base * px_base
                    active_assets.append((base_key, px_base))
                else:
                    residual += base_budget
            else:
                px = row_prices.get(target_asset, np.nan)
                if not np.isnan(px) and px > 0:
                    raw = budget / px
                    sh = math.floor(raw) if integer_shares else raw
                    new_h[target_asset] = sh
                    residual = budget - sh * px
                    active_assets = [(target_asset, px)]
                else:
                    residual = budget
            # Greedy Fill: 잔여 현금으로 추가 매수 (레버리지→기초자산 가격 내림차순)
            if integer_shares and residual > 0 and active_assets:
                active_assets.sort(key=lambda x: x[1], reverse=True)
                changed = True
                while changed and residual > 0:
                    changed = False
                    for t, px in active_assets:
                        if residual >= px:
                            extra = math.floor(residual / px)
                            if extra > 0:
                                new_h[t] += float(extra)
                                residual -= extra * px
                                changed = True
            return new_h, residual

        # Layer 2 스테이지 상태 (TLT/TBF 각각 독립)
        tlt_stage      = 0   # 0=없음 / 1=S1(30%) / 2=S2(70%) / 3=S3(100%)
        tbf_stage      = 0
        tlt_in_signal  = False  # True=신호 발생 후 가격 모드
        tbf_in_signal  = False
        # floor_s1 모드: 첫 번째 sell로 S1에 도달한 상태 추적 (다음 sell에서 S0)
        tlt_floor_hit  = False  # True = sell 후 S1 대기 중 → 다음 sell시 S0
        tbf_floor_hit  = False

        for i in range(len(dates)):
            date     = dates[i]
            row      = combined.loc[date]
            prev_row = combined.loc[dates[i - 1]] if i > 0 else row

            state   = row['final_state']
            f1      = row['f1_state']
            rsi     = row.get('tlt_rsi', 50)
            gap_val = row.get('gap', 0)
            yc_val  = row.get('t10y2y', 0)
            ycm     = row.get('yc_mom', 0)

            # ── Layer 1: V7 기초자산 결정 ──
            target = 'BIL'
            if state == 'RISING':
                target = 'TBF'
            elif state == 'FALLING':
                target = 'TLT' if (current_asset == 'TLT' or rsi <= 55) else 'BIL'
            elif state == 'NEUTRAL':
                target = 'TLT'

            # V6: FFV Filter
            if not np.isnan(gap_val):
                if gap_val < params['ffv_lo'] and target == 'TLT': target = 'BIL'
                elif gap_val > params['ffv_hi'] and target == 'TBF': target = 'BIL'

            # V7: YC Override
            if not np.isnan(yc_val) and not np.isnan(ycm):
                if yc_val < params['yc_inv']:
                    if target == 'TLT': target = 'BIL'
                    if target != 'BIL' or gap_val <= params['ffv_hi']: target = 'TBF'
                elif (ycm > params['yc_steep'] and yc_val < 1.0 and f1 == 'FALLING' and gap_val > params['ffv_lo']):
                    target = 'TLT'

            # ── Layer 2: V8 레버리지 스테이지 머신 ──
            lev_stage     = 0
            lev_frac      = 0.0
            lev_asset_key = None

            if _use_lev and i > 0:
                # TLT → TMF 스테이지 머신
                if target == 'TLT':
                    tlt_px_now  = row.get('TLT', np.nan)
                    tlt_px_prev = prev_row.get('TLT', np.nan)
                    tlt_chg = (tlt_px_now / tlt_px_prev) if (not np.isnan(tlt_px_now) and not np.isnan(tlt_px_prev) and tlt_px_prev > 0) else 1.0

                    buy_hit  = _check_buy('tlt', row, prev_row)
                    sell_hit = _check_sell('tlt', row, prev_row)

                    if sell_hit:
                        if _sell_mode == 'instant':
                            # 즉시 S0 (기존 동작)
                            tlt_stage     = 0
                            tlt_in_signal = False
                            tlt_floor_hit = False
                        elif _sell_mode == 'step':
                            # 단계적 청산: stage -= 1
                            tlt_stage = max(0, tlt_stage - 1)
                            if tlt_stage == 0:
                                tlt_in_signal = False
                        elif _sell_mode == 'floor_s1':
                            # S1 후퇴 절충: sell -> S1, S1 대기 중 추가 sell -> S0
                            if tlt_floor_hit or tlt_stage <= 1:
                                tlt_stage     = 0
                                tlt_in_signal = False
                                tlt_floor_hit = False
                            else:
                                tlt_stage     = 1
                                tlt_floor_hit = True
                    elif not tlt_in_signal:
                        # 신호 모드: buy_signal → S1 진입
                        if buy_hit and tlt_stage < 3:
                            tlt_stage     += 1
                            tlt_in_signal  = True
                            tlt_floor_hit  = False  # buy 진입 시 floor 상태 해제
                    else:
                        # 가격 모드: reb_up → 스테이지 업 / reb_down → 스테이지 다운
                        if tlt_chg >= 1.0 + _reb_up and tlt_stage < 3:
                            tlt_stage     += 1
                            tlt_floor_hit  = False  # 가격 상승 시 floor 상태 해제
                        elif tlt_chg <= 1.0 - _reb_dn and tlt_stage > 0:
                            tlt_stage -= 1
                            if tlt_stage == 0:
                                tlt_in_signal = False
                                tlt_floor_hit = False

                    lev_stage = tlt_stage
                    tbf_stage = 0; tbf_in_signal = False; tbf_floor_hit = False  # TLT 구간에서는 TBF 스테이지 초기화

                # TBF → TMV 스테이지 머신
                elif target == 'TBF':
                    tbf_px_now  = row.get('TBF', np.nan)
                    tbf_px_prev = prev_row.get('TBF', np.nan)
                    tbf_chg = (tbf_px_now / tbf_px_prev) if (not np.isnan(tbf_px_now) and not np.isnan(tbf_px_prev) and tbf_px_prev > 0) else 1.0

                    buy_hit  = _check_buy('tbf', row, prev_row)
                    sell_hit = _check_sell('tbf', row, prev_row)

                    if sell_hit:
                        if _sell_mode == 'instant':
                            tbf_stage     = 0
                            tbf_in_signal = False
                            tbf_floor_hit = False
                        elif _sell_mode == 'step':
                            tbf_stage = max(0, tbf_stage - 1)
                            if tbf_stage == 0:
                                tbf_in_signal = False
                        elif _sell_mode == 'floor_s1':
                            if tbf_floor_hit or tbf_stage <= 1:
                                tbf_stage     = 0
                                tbf_in_signal = False
                                tbf_floor_hit = False
                            else:
                                tbf_stage     = 1
                                tbf_floor_hit = True
                    elif not tbf_in_signal:
                        if buy_hit and tbf_stage < 3:
                            tbf_stage     += 1
                            tbf_in_signal  = True
                            tbf_floor_hit  = False
                    else:
                        if tbf_chg >= 1.0 + _reb_up and tbf_stage < 3:
                            tbf_stage     += 1
                            tbf_floor_hit  = False
                        elif tbf_chg <= 1.0 - _reb_dn and tbf_stage > 0:
                            tbf_stage -= 1
                            if tbf_stage == 0:
                                tbf_in_signal = False
                                tbf_floor_hit = False

                    lev_stage = tbf_stage
                    tlt_stage = 0; tlt_in_signal = False; tlt_floor_hit = False  # TBF 구간에서는 TLT 스테이지 초기화

                else:
                    # BIL 구간: 모든 스테이지 초기화
                    tlt_stage = 0; tlt_in_signal = False; tlt_floor_hit = False
                    tbf_stage = 0; tbf_in_signal = False; tbf_floor_hit = False
                    lev_stage = 0

                lev_frac = {1: 0.30, 2: 0.70, 3: 1.0}.get(lev_stage, 0.0)
                if lev_frac > 0:
                    lev_asset_key = 'TMF' if target == 'TLT' else ('TMV' if target == 'TBF' else None)

            # ── 일별 수익률 계산 ──
            row_px = {a: row.get(a, np.nan) for a in bond_holdings}
            if i > 0:
                if lev_asset_key and lev_frac > 0:
                    base_key  = 'TLT' if lev_asset_key == 'TMF' else 'TBF'
                    px_lev_c  = row.get(lev_asset_key, np.nan)
                    px_lev_p  = prev_row.get(lev_asset_key, np.nan)
                    px_bas_c  = row.get(base_key, np.nan)
                    px_bas_p  = prev_row.get(base_key, np.nan)
                    if (not np.isnan(px_lev_c) and not np.isnan(px_lev_p) and px_lev_p > 0
                            and not np.isnan(px_bas_c) and not np.isnan(px_bas_p) and px_bas_p > 0):
                        daily_ret = lev_frac * (px_lev_c / px_lev_p) + (1 - lev_frac) * (px_bas_c / px_bas_p)
                    else:
                        daily_ret = 1.0
                else:
                    asset_key = target if target in combined.columns else 'BIL'
                    px_c = row.get(asset_key, np.nan)
                    px_p = prev_row.get(asset_key, np.nan)
                    daily_ret = (px_c / px_p if not np.isnan(px_c) and not np.isnan(px_p) and px_p > 0 else 1.0)
                if integer_shares:
                    # 정수 주식: 투자분에만 수익률 적용, 잔여 현금은 고정
                    bond_invest *= daily_ret
                    portfolio_value = bond_invest + bond_cash_reserve
                else:
                    portfolio_value *= daily_ret

            # ── 리밸런싱 이벤트: 자산 변경 OR lev_frac 변경 시 수수료 + 정수 주식 처리 ──
            need_rebalance = (target != current_asset) or (abs(lev_frac - current_lev_frac) > 1e-6)
            if need_rebalance:
                # 수수료: 실제 매매된 비중(traded_fraction)에만 적용
                if fee_rate > 0:
                    if target != current_asset:
                        traded_frac = 1.0
                    else:
                        traded_frac = abs(lev_frac - current_lev_frac)
                    portfolio_value *= (1 - traded_frac * fee_rate * 2)
                # 수량 추적 업데이트 (Greedy Fill 포함)
                bond_holdings, bond_cash = _bond_rebalance(
                    portfolio_value, target, lev_frac, lev_asset_key, row_px
                )
                if integer_shares:
                    # 투자분/잔여 현금 분리: bond_invest = 실제 매수한 가치, bond_cash_reserve = 잔여
                    bond_invest = portfolio_value - bond_cash  # = floor_shares × price (Greedy Fill 후)
                    bond_cash_reserve = bond_cash
                    portfolio_value = bond_invest + bond_cash_reserve  # 명시적 동기화
                current_lev_frac = lev_frac

            summary_parts = [f"상태:{state}"]
            if not np.isnan(gap_val): summary_parts.append(f"FFV:{gap_val:+.2f}")
            if yc_val < params['yc_inv']: summary_parts.append("YC역전")
            elif (ycm > params['yc_steep'] and yc_val < 1.0): summary_parts.append("YC스티프닝")

            # ── 로그 및 결과 기록 ──
            prefix = "유지"
            if target != current_asset:
                prefix = "매수" if target != 'BIL' else "매도"
            elif _use_lev and abs(lev_frac - current_lev_frac) > 1e-6:
                prefix = "매수" if lev_frac > current_lev_frac else "매도"
            
            trade_label = f"{prefix}(S{lev_stage}): {target}"
            if target == current_asset and abs(lev_frac - current_lev_frac) <= 1e-6:
                trade_label = f"진행중(S{lev_stage}): {target}"

            # 비중 계산
            log_lev_weight = (sum(bond_holdings[t] * row.get(t, 0) for t in ['TMF', 'TMV'] if t in bond_holdings) / portfolio_value) if portfolio_value > 0 else 0
            log_base_weight = (sum(bond_holdings[t] * row.get(t, 0) for t in ['TLT', 'TBF', 'BIL'] if t in bond_holdings) / portfolio_value) if portfolio_value > 0 else 0

            history_row = {
                'Date': date, 'Value': portfolio_value, 'Asset': target,
                'Trade_Label': trade_label,
                'Summary': " | ".join(summary_parts), 'f_state': state, 'gap': gap_val,
                't10y2y': yc_val, 'yc_mom': ycm, 'tlt_rsi': rsi,
                'ff_rate': row.get('ff_rate', 0), 'core_cpi': row.get('core_cpi', 0),
                'Stage': lev_stage, 'Lev_Weight': log_lev_weight, 'Base_Weight': log_base_weight,
                'lev_frac': lev_frac,
                'lev_asset': lev_asset_key if lev_asset_key else target,
            }
            # 자산별 개별 비중 추가
            for t in bond_holdings:
                history_row[f'{t}_Weight'] = (bond_holdings[t] * row.get(t, 0) / portfolio_value) if portfolio_value > 0 else 0

            results.append(history_row)

            state_dict = {
                'date': date,
                'cash': 0,
                'portfolio_value': portfolio_value,
                'history_row': history_row
            }
            injected = yield state_dict
            if injected and isinstance(injected, dict) and 'cash_delta' in injected:
                delta = injected['cash_delta']
                net_delta = delta - abs(delta) * fee_rate
                portfolio_value += net_delta
                # PM 자금 수혈/환수 후 현재 자산 비율로 재편입
                if portfolio_value > 0:
                    bond_holdings, bond_cash = _bond_rebalance(
                        portfolio_value, target, lev_frac, lev_asset_key,
                        {a: row.get(a, np.nan) for a in bond_holdings}
                    )
                    if integer_shares:
                        bond_invest = portfolio_value - bond_cash
                        bond_cash_reserve = bond_cash
                        portfolio_value = bond_invest + bond_cash_reserve

            current_asset = target

        res_df = pd.DataFrame(results).set_index('Date')
        return res_df, pd.DataFrame(), []

    @staticmethod
    def predict_bond_signal(data_dict, macro_df, params=None):
        """본드 전략 예상 신호 추출"""
        try:
            res_df, _, _ = StrategyEngine.run_bond_strategy(data_dict, macro_df, params=params)
            if not res_df.empty:
                latest = res_df.iloc[-1]
                return {
                    'date': latest.name,
                    'signal': latest.get('Asset', 'BIL'),
                    'f_state': latest.get('f_state', 0),
                    'gap': latest.get('gap', 0),
                    'yc': latest.get('t10y2y', 0),
                    'rsi': latest.get('tlt_rsi', 0)
                }
        except Exception as e:
            print(f"Bond Signal Prediction Error: {e}")
        return None

    @staticmethod
    def predict_today_signal(data_dict, fg_df, vxn_df, news_df, leverage_asset, base_asset, cash_ratio, start_date, end_date, params, trade_at, smart_params):
        """가상 종가가 주입된 데이터를 바탕으로 '오늘' 발생할 예상 신호만 추출"""
        try:
            # [수정] 프리뷰 시에는 모든 가상 데이터를 시뮬레이션에 포함시키기 위해 종료일 필터를 제거(None)
            # 대신 salt를 이용해 분 단위로 새로운 계산을 유도함
            dyn_salt = f"preview_{datetime.datetime.now().strftime('%Y%m%d%H%M')}"
            
            golden_history, _, signal_events = StrategyEngine.run_golden_strategy(
                data_dict, fg_df, vxn_df, news_df, leverage_asset, base_asset, cash_ratio, 
                start_date, None, params, trade_at, smart_params, salt=dyn_salt
            )
            
            if not golden_history.empty:
                latest = golden_history.iloc[-1]
                print(f"PREVIEW SUCCESS: Date={latest.name}, Signal={latest.get('Trade_Label')}")
                return {
                    'date': latest.name,
                    'signal': latest.get('Trade_Label', '중립'),
                    'summary': latest.get('Summary', '특이사항 없음'),
                    'price': latest.get('Close', 0),
                    'rsi': latest.get('RSI', 0),
                    'vxn': latest.get('VXN', 0),
                    'stage': latest.get('Stage', 0)
                }
            else:
                print(f"PREVIEW FAILURE: golden_history is empty. Check data_dict keys={list(data_dict.keys())}")
                if base_asset in data_dict:
                    print(f"Base asset ({base_asset}) data range: {data_dict[base_asset].index[0]} ~ {data_dict[base_asset].index[-1]}")
        except Exception as e:
            import traceback
            print(f"신호 예측 오류: {e}")
            traceback.print_exc()
        return None

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def run_benchmark(data_dict, ticker, start_date, end_date):
        if ticker not in data_dict: return pd.DataFrame()
        df = data_dict[ticker].dropna(subset=['Close'])
        dates = df.index
        if start_date: dates = dates[dates.date >= start_date]
        if end_date: dates = dates[dates.date <= end_date]
        if len(dates) == 0: return pd.DataFrame()
        shares = 10000.0 / df.loc[dates[0], 'Close']
        return pd.DataFrame([{'Date': d, 'Value': shares * df.loc[d, 'Close']} for d in dates]).set_index('Date')

    @staticmethod
    def run_monte_carlo(history, days=252, n_sims=200):
        if history.empty or len(history) < 30: return None
        rets = history['Value'].pct_change().dropna()
        mu, sigma, last_v = rets.mean(), rets.std(), history['Value'].iloc[-1]
        sims = np.zeros((days, n_sims))
        for i in range(n_sims):
            sims[:, i] = last_v * np.exp(np.cumsum(np.random.normal(mu, sigma, days)))
        return sims
