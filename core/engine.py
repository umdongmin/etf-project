import pandas as pd
import numpy as np
import pandas_ta as ta
import streamlit as st
import datetime

class StrategyEngine:
    """백테스팅 및 벤치마크 계산을 담당하는 클래스"""
    @st.cache_data(ttl=3600, show_spinner=False)
    @staticmethod
    def run_golden_strategy(data_dict, fg_df, vxn_df, news_df=None, leverage_asset='QLD', base_asset='QQQ', cash_ratio=0.0, start_date=None, end_date=None, params=None, trade_at="종가", smart_params=None, salt=None, fast_mode=False):
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
        combined = base_df[['Close', 'RSI']].copy()
        
        # [추가] 시가 데이터 확보
        if 'Open' in base_df.columns:
            combined['Open'] = base_df['Open']
        else:
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

        portfolio_value, cash = 10000.0, 10000.0 * cash_ratio
        current_planned_asset, rebalance_stage, last_entry_price = base_asset, 0, 0
        peak_price, peak_macd = 0, 0
        holdings = {t: 0.0 for t in data_dict.keys()}
        
        initial_price = base_df['Close'].asof(dates[0])
        if pd.isna(initial_price) or initial_price <= 0:
            # [수정] 데이터 부재 시 백업 로직 강화
            valid_closes = base_df['Close'].dropna()
            initial_price = valid_closes.iloc[0] if not valid_closes.empty else 1.0
            
        holdings[base_asset] = (portfolio_value * (1 - cash_ratio)) / initial_price if initial_price > 0 else 0
        
        def get_v_level(planned, stage):
            # Stage는 이제 항상 레버리지 노출 수준(0-3)을 의미합니다.
            return stage

        history = []
        trade_slots = [] # list of {'buy_price': val, 'date': date}
        closed_trades = [] # list of {'entry_date': d, 'exit_date': d, 'asset': a, 'buy_price': p, 'sell_price': p, 'ret': r, 'status': s}
        # [신규] 모든 1~2매수, 1~4매도 신호 발생 기록 수집 배열
        signal_events = []
        
        # [신규] RSI 1차 대기(알람) 상태 저장 변수 (신호 번호별 독립적인 bool 값 가짐)
        buy_wait_flags = {i: False for i in range(1, 5)}
        sell_wait_flags = {i: False for i in range(1, 5)}
        panic_buy_wait_flags = {i: False for i in range(1, 4)}
        
        # [추가] 지연 실행용 상태 변수
        pending_rebalance = None # {'target_lev_pct': val, 'rebalance_stage': val, 'planned': asset}
        
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
        open_price_arrays = {}
        for t in base_targets:
            if t in data_dict:
                # [수정] 데이터 유실 방지: 전체 인덱스에서 ffill을 먼저 수행한 뒤 dates 구간만 추출
                filled_df = data_dict[t].reindex(combined.index).ffill().loc[dates]
                price_arrays[t] = filled_df['Close'].values
                if 'Open' in filled_df.columns:
                    open_price_arrays[t] = filled_df['Open'].values
                else:
                    open_price_arrays[t] = filled_df['Close'].values

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
                    continue
            except Exception as e:
                pass
            
            # (중략) 이후 로직 동일
            
            # [수정] 익일 시가 매매 처리: 전날 신호가 있었다면 오늘 시가로 먼저 체결
            if trade_at == "익일 시가" and pending_rebalance is not None:
                # 오늘 시가(Open)로 체결 (데이터가 없는 경우 종가로 대체, 그것도 없으면 건너뜀)
                try:
                    trade_prices = {}
                    for t in check_targets:
                        if t in open_price_arrays:
                            trade_prices[t] = open_price_arrays[t][i]
                        else:
                            trade_prices[t] = prices[t]
                    
                    required = currently_holding + [pending_rebalance['planned']]
                    if any(t not in trade_prices or pd.isna(trade_prices[t]) for t in required):
                        continue
                except:
                    continue
                
                # 거래 시점의 자산 가치 재계산
                current_total_val = cash + sum(holdings[t] * trade_prices[t] for t in holdings if t in trade_prices)
                cash = current_total_val * cash_ratio
                etf_funds = current_total_val * (1 - cash_ratio)
                
                t_lev_asset, t_lev_pct = pending_rebalance['planned'], pending_rebalance['target_lev_pct']
                rebalance_stage = pending_rebalance['rebalance_stage']
                
                # [수정] 순차적 포트폴리오 관리 (상시 활성화)
                # 1. 현재 자산 평가 (trade_prices 기준)
                h_vals = {t: holdings[t] * trade_prices[t] for t in holdings if t in trade_prices}
                curr_total_lev_val = sum(h_vals.get(t, 0) for t in target_group[1:]) # 현재 합산 레버리지
                
                # 2. 목표 레버리지 총액 및 업그레이드 Cap 계산 (30/70/100)
                normalized_pct = t_lev_pct / 100.0 if t_lev_pct > 1 else t_lev_pct
                target_lev_val = etf_funds * normalized_pct
                
                tqqq_cap = target_lev_val
                if target_lev_val > etf_funds * 0.9 and t_lev_asset == target_group[2]:
                    # 100% 레버리지 구간에서의 터보 업그레이드 단계 적용
                    if rebalance_stage == 1: tqqq_cap = 0.30 * target_lev_val
                    elif rebalance_stage == 2: tqqq_cap = 0.70 * target_lev_val

                new_h_vals = {t: 0.0 for t in holdings}
                
                if curr_total_lev_val > target_lev_val + 0.01:
                    # 감축(Sell) 상황: QLD부터 채우고 남은 자리를 TQQQ로 채움 (TQQQ 우선 매도)
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), target_lev_val)
                    rem = target_lev_val - new_h_vals[target_group[1]]
                    new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), rem)
                else:
                    # 증액(Buy) 상황: 기존 레버리지 자산들을 유지하며 부족분만 현재 타겟으로 충전
                    # 1. TQQQ 보존 및 업그레이드 (Cap 한도 내)
                    if t_lev_asset == target_group[2]:
                        # 가속 모드: QLD를 팔아서라도 tqqq_cap까지 채움 (업그레이드)
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0) + h_vals.get(target_group[1], 0), tqqq_cap)
                    else:
                        # 일반 모드: 기존 TQQQ만 유지
                        new_h_vals[target_group[2]] = min(h_vals.get(target_group[2], 0), tqqq_cap)
                    
                    # 2. QLD 보전 및 충전 (남은 비중 한도 내)
                    rem = target_lev_val - new_h_vals[target_group[2]]
                    new_h_vals[target_group[1]] = min(h_vals.get(target_group[1], 0), rem)
                    
                    # 3. 최종 부족분 충전 (현재 시장 타겟 자산으로)
                    rem = target_lev_val - (new_h_vals[target_group[1]] + new_h_vals[target_group[2]])
                    if rem > 0:
                        new_h_vals[t_lev_asset] += rem

                # 3. 나머지는 Base 자산
                allocated_lev = sum(new_h_vals.values())
                new_h_vals[base_asset] = max(0, etf_funds - allocated_lev)
                
                # 수량 업데이트
                for t in check_targets:
                    holdings[t] = new_h_vals.get(t, 0) / trade_prices[t] if t in trade_prices and trade_prices[t] > 0 else 0
                
                # [신규] FIFO 기반 가상 장부 업데이트 (익일 시가 체결 버전)
                new_stage_open = pending_rebalance['rebalance_stage']
                old_v = get_v_level(current_planned_asset, rebalance_stage)
                new_v = get_v_level(t_lev_asset, new_stage_open)

                if new_v > old_v:
                    # 비중 확대
                    for _ in range(new_v - old_v):
                        # 실제 매매된 가속 자산 가격 사용
                        trade_slots.append({'buy_price': trade_prices[t_lev_asset], 'date': date, 'asset': t_lev_asset})
                elif new_v < old_v:
                    # 비중 축소: 정산
                    diff = old_v - new_v
                    rets = []
                    entry_dates = []
                    for _ in range(diff):
                        if trade_slots:
                            slot = trade_slots.pop(0)
                            asset_sold = slot.get('asset', t_lev_asset)
                            sell_p = trade_prices[asset_sold]
                            r = (sell_p / slot['buy_price'] - 1) * 100
                            rets.append(r)
                            entry_dates.append(slot['date'])
                            closed_trades.append({
                                'Entry_Date': slot['date'], 'Exit_Date': date, 'Asset': asset_sold,
                                'Buy_Price': slot['buy_price'], 'Sell_Price': sell_p,
                                'Return': r, 'Status': "익절" if r > 0 else "손절"
                            })
                    if rets and history:
                        # 익일 시가는 어제 행에 기록 (대표 수익률은 평균값)
                        history[-1]['Trade_Ret'] = np.mean(rets)
                        history[-1]['Trade_Status'] = "익절" if history[-1]['Trade_Ret'] > 0 else "손절"
                        history[-1]['Trade_Entry_Date'] = ", ".join([d.strftime('%Y-%m-%d') for d in entry_dates])

                current_planned_asset = t_lev_asset
                rebalance_stage = new_stage_open
                last_entry_price = trade_prices[base_asset]
                
                # [추가] 예약된 매매가 실행되었으므로 해당 시그널 플래그 켜기
                if 'signal_ref' in pending_rebalance and pending_rebalance['signal_ref']:
                    pending_rebalance['signal_ref']['executed'] = True
                    
                pending_rebalance = None # 처리 완료
            
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

                    return a, c, rs

                is_buy_signal = False
                buy_signal_idx = -1
                signal_reason = ""
                for idx, bp in enumerate(params['buy_signals']):
                    active, cond, reasons = check_buy_cond(bp, idx + 1, buy_wait_flags)
                    if active and cond:
                        is_buy_signal = True
                        buy_signal_idx = idx + 1
                        signal_reason = " / ".join(reasons)
                        break
                
                is_sell_signal = False
                sell_signal_idx = -1 # [신규] 어떤 매도 신호가 발생했는지 추적
                for idx, sp in enumerate(params['sell_signals']):
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

            # [수정] 스마트 레버리지 필터링 (VIX Safety vs RSI Turbo 분리)
            effective_lev_asset = leverage_asset
            smart_reason = ""
            
            if smart_params:
                use_vxn = smart_params.get('use_vxn_safety', smart_params.get('use_vix_safety', False))
                exit_val = smart_params.get('vxn_exit', smart_params.get('vix_exit', 31))
                if use_vxn and vxn > exit_val:
                    effective_lev_asset = target_group[0]
                    smart_reason = f"대피(VXN:{vxn:.1f})"
                elif smart_params.get('use_rsi_turbo') and rsi < smart_params.get('rsi_turbo', 31):
                    effective_lev_asset = target_group[2]
                    smart_reason = f"가속(RSI:{rsi:.1f})"
                else:
                    effective_lev_asset = leverage_asset
                    if use_vxn or smart_params.get('use_rsi_turbo'):
                        smart_reason = "정상(Normal)"

            # (기존 prices 위치 삭제됨 - 상단으로 이동)
            
            current_total_val = cash + sum(holdings.get(t, 0) * prices.get(t, 0) for t in holdings if t in prices)
            
            # [신규 추가] 수익률 nan 방지를 위한 수치 정규화 (Defensive Check)
            if pd.isna(current_total_val) or current_total_val <= 0:
                current_total_val = history[-1]['Value'] if history else 10000.0
                
            rebalance_needed = False
            new_planned = current_planned_asset
            new_stage = rebalance_stage
            
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
                            closed_trades.append({
                                'Entry_Date': slot['date'],
                                'Exit_Date': date,
                                'Asset': asset_to_check,
                                'Buy_Price': slot['buy_price'],
                                'Sell_Price': prices[asset_to_check],
                                'Return': slot_ret,
                                'MDD': slot.get('worst_ret', slot_ret), # 보유 기간 중 최저 수익률
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

            etf_funds = current_total_val * (1 - cash_ratio)
            lev_candidates = ['QLD', 'TQQQ']
            current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
            current_lev_pct = current_lev_funds / etf_funds if etf_funds > 0 else 0

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

            # (2) 고정 비율 트리거 계산
            fixed_buy_hit = False
            fixed_sell_hit = False
            if use_fixed:
                up_thresh = abs(b_up if current_planned_asset == leverage_asset else s_up)
                down_thresh = abs(b_down if current_planned_asset == leverage_asset else s_down)
                price_trigger = (price_change >= 1 + up_thresh or price_change <= 1 - down_thresh)
                fixed_buy_hit = (rebalance_stage in [1, 2] and is_buy_process and is_mode_consistent and price_trigger)
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
            s3_p_list = params.get('s3_protection', [])
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
                        active, cond, pb_reasons = check_buy_cond(pb_list[pb_idx], pb_idx + 1, panic_buy_wait_flags)
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
                # [추가] 당일 체결 방식이므로 곧바로 실행 여부 업데이트
                if current_signal_ref:
                    current_signal_ref['executed'] = True
                    
                if trade_at == "종가":
                    h_vals = {t: holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates}
                    curr_total_lev_val = sum(h_vals.values())
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
                    
                    for t in check_targets:
                        holdings[t] = new_h_vals.get(t, 0) / prices[t] if t in prices and prices[t] > 0 else 0
                    
                    # [신규] 이번 매매로 실제 추가된 수량
                    inc_h_counts = {t: max(0.0, holdings.get(t, 0) - prev_h_counts.get(t, 0)) for t in check_targets}
                    
                    old_v = get_v_level(current_planned_asset, rebalance_stage)
                    new_v = get_v_level(new_planned, new_stage)

                    if new_v > old_v:
                        num_new = new_v - old_v
                        # 해당 자산의 증가분을 슬롯 수만큼 나눔 (평균치 할당)
                        shares_per_slot = inc_h_counts.get(new_planned, 0) / num_new if num_new > 0 else 0
                        
                        for _ in range(num_new):
                            trade_slots.append({
                                'buy_price': prices[new_planned], 
                                'date': date, 
                                'asset': new_planned,
                                'shares': shares_per_slot,
                                'worst_ret': 0.0 # [신규] 최저 수익률 초기화
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
                                closed_trades.append({
                                    'Entry_Date': slot['date'], 'Exit_Date': date, 'Asset': asset_sold,
                                    'Buy_Price': slot['buy_price'], 'Sell_Price': sell_p,
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
                    peak_price = prices[base_asset] # 리밸런싱 시 고점 초기화

                    # [추가] 매매 후 현금 및 레버리지 자산 총액 재계산 (로그용)
                    cash = current_total_val * cash_ratio
                    current_lev_funds = sum(holdings.get(t, 0) * prices.get(t, 0) for t in lev_candidates if t in prices)
                else:
                    pending_rebalance = {'target_lev_pct': target_lev_pct, 'rebalance_stage': new_stage, 'planned': new_planned, 'signal_ref': current_signal_ref}
                    # [신규] '익일 시가' 매매 시에도 기준가격 계산의 베이스는 '신호 발생일 종가'로 고속
                    last_entry_price = prices[base_asset]
                    peak_price = prices[base_asset]

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
                        if new_stage == 1: limit = smart_params.get('panic_rsi_s1', 27)
                        elif new_stage == 2: limit = smart_params.get('panic_rsi_s2', 28)
                        else: limit = smart_params.get('panic_rsi_s3', 30)
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
                    'Asset': log_asset_name,
                    'Drop_Acc': row_arr[col_idx['Drop_Acc']] if 'Drop_Acc' in col_idx else 0,
                    'Gap_Down': row_arr[col_idx['Gap_Down']] if 'Gap_Down' in col_idx else 0,
                    'S3_Drop_Limit': s3_drop_triggered_val, # [기록]
                    'S3_Sell3_Event': s3_sell3_event_val, # [복구]
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

        if not history:
            return pd.DataFrame(columns=['Value', 'Stage', 'Asset', 'Trade_Label']), pd.DataFrame(), []
        return pd.DataFrame(history).set_index('Date'), pd.DataFrame(closed_trades), signal_events

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
