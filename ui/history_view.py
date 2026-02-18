import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from core.storage import StrategyStorage
from core.engine import StrategyEngine
from ui.tester_view import TesterView

class HistoryLabView:
    """역사적 마켓 랩 화면 UI 구성 클래스"""
    
    EVENTS = {
        2010: "플래시 크래시, 유럽 재정 위기 시작",
        2011: "미국 국가신용등급 강등, 그리스 디폴트 우려",
        2012: "유로존 붕괴 위기 고조, 페이스북 IPO",
        2013: "테이퍼 탠트럼 (긴축 발작), 양적완화 축소 논의",
        2014: "유가 급락, 알리바바 역대 최대 IPO",
        2015: "중국 증시 폭락, 미국 7년 만의 금리 인상",
        2016: "브렉시트 투표, 트럼프 대통령 당선",
        2017: "FAANG 주도 기술주 랠리, 비트코인 열풍",
        2018: "미중 무역 전쟁, 연준의 지속적 금리 인상",
        2019: "장단기 금리 역전, 레포 금리 급등",
        2020: "코로나19 팬데믹, 사상 최대 유동성 공급",
        2021: "포스트 팬데믹 인플레이션 시작, 공급망 병목",
        2022: "40년 만의 고물가, 공격적 금리 인상 (Bear Market)",
        2023: "SVB 은행 파산 위기, 생성형 AI (ChatGPT) 열풍",
        2024: "AI 반도체 잭팟, 금리 인하 기대감 형성"
    }

    @staticmethod
    def render(data_dict, fg_df, vix_df, leverage_asset, base_asset, trade_at, params, smart_params=None):
        st.header("📜 매매 전략 설정 및 역사적 분석")
        st.caption("전략을 저장하거나 불러오고, 2010년부터 현재까지의 성과를 분석합니다.")

        st.divider()
        st.subheader("💾 전략 파일 관리")
        cp = st.session_state.current_params

        col1, col2 = st.columns(2)
        with col1:
            with st.form("save_strat_form_main"):
                st.write("📝 현재 전략 저장")
                new_strat_name = st.text_input("새 전략 이름 입력", key="new_strat_save_name_main")
                save_submitted = st.form_submit_button("💾 내 컴퓨터에 신규 저장", use_container_width=True)
                if save_submitted:
                    if new_strat_name:
                        StrategyStorage.save_strategy(new_strat_name, cp)
                        st.session_state.loaded_strat_name = new_strat_name
                        st.success(f"'{new_strat_name}' 저장 완료")
                        st.rerun()
                    else:
                        st.warning("전략 이름을 입력해 주세요.")
            
            if st.session_state.get('loaded_strat_name'):
                st.write(f"🔄 기존 전략: **{st.session_state.loaded_strat_name}**")
                if st.button(f"🚀 '{st.session_state.loaded_strat_name}' 업데이트 (덮어쓰기)", key="update_btn_main", use_container_width=True):
                    StrategyStorage.save_strategy(st.session_state.loaded_strat_name, cp)
                    st.success(f"'{st.session_state.loaded_strat_name}' 업데이트 완료!")

        with col2:
            stored_strats = StrategyStorage.list_strategies()
            if stored_strats:
                with st.form("load_strat_form_main"):
                    st.write("📂 저장된 전략 불러오기")
                    target_strat = st.selectbox("전략 선택 (목록)", ["선택 안 함"] + stored_strats, key="load_sel_main")
                    load_submitted = st.form_submit_button("✨ 선택한 전략 적용하기", use_container_width=True)
                    if load_submitted and target_strat != "선택 안 함":
                        config = StrategyStorage.load_strategy(target_strat)
                        if config:
                            st.session_state.current_params.update(config)
                            st.session_state.loaded_strat_name = target_strat
                            st.session_state.pending_sync = True
                            st.success(f"'{target_strat}' 모든 설정 로드 완료!")
                            st.rerun()
            else:
                st.info("저장된 전략이 없습니다.")
        
        start_date = datetime.date(2010, 1, 1)
        end_date = datetime.date.today()
        
        with st.spinner('전 기간 역사적 데이터 시뮬레이션 중...'):
            golden_history, closed_trades = StrategyEngine.run_golden_strategy(data_dict, fg_df, vix_df, leverage_asset, base_asset, params['cash_ratio_pct']/100.0, start_date, end_date, params, trade_at, smart_params=smart_params)
            
            st.session_state.last_golden_result = {
                'history': golden_history.copy(),
                'base': base_asset,
                'lev': leverage_asset
            }
            
            semi_group = ['SOXX', 'USD', 'SOXL']
            is_semi = base_asset in semi_group or leverage_asset in semi_group
            b1, b2, b3 = ("SOXX", "USD", "SOXL") if is_semi else ("QQQ", "QLD", "TQQQ")
            
            bh_1 = StrategyEngine.run_benchmark(data_dict, b1, start_date, end_date)
            bh_2 = StrategyEngine.run_benchmark(data_dict, b2, start_date, end_date)
            bh_3 = StrategyEngine.run_benchmark(data_dict, b3, start_date, end_date)
            
        if golden_history.empty:
            st.warning("분석할 데이터가 충분하지 않습니다.")
            return

        st.divider()
        st.subheader("📊 전체 기간 요약")
        HistoryLabView.render_overall_summary(golden_history, bh_1, bh_2, bh_3, b_names=(b1, b2, b3))
        
        st.divider()
        st.subheader("⚙️ 전략 매매 조건 설정")
        st.caption("여기서 설정을 변경한 후 아래 버튼을 클릭해야 모든 분석 결과에 반영됩니다.")
        with st.form("strategy_config_form"):
            # 앱의 ticker_map에 접근하기 위해 st.session_state나 직접 전달 필요
            # app.py에서 ticker_map을 넘겨주도록 HistoryLabView.render 시그니처 수정 고려
            ticker_map = {
                "QQQ": "나스닥 100 (QQQ)", "QLD": "나스닥 2배 (QLD)", "TQQQ": "나스닥 3배 (TQQQ)",
                "SOXX": "반도체 1배 (SOXX)", "USD": "반도체 2배 (USD)", "SOXL": "반도체 3배 (SOXL)",
                "TMF": "채권 3배 (TMF)", "TLT": "채권 1배 (TLT)"
            }
            new_params = TesterView.render_config(st.session_state.current_params, ticker_map=ticker_map)
            if st.form_submit_button("🚀 설정 반영 및 재시뮬레이션"):
                st.session_state.current_params.update(new_params)
                st.rerun()

    @staticmethod
    def render_yearly_table(s_h, b1_h, b2_h, b3_h, b_names=("QQQ", "QLD", "TQQQ"), smart_params=None, selected_comparisons=[]):
        st.subheader(f"연도별 성과 및 리스크 분석 (vs {b_names[0]}군)")
        
        def calculate_mdd(series):
            s = series.dropna()
            if s.empty: return 0
            roll_max = s.cummax()
            dd = (s - roll_max) / roll_max
            return dd.min()

        years = sorted(s_h.index.year.unique())
        data = []
        
        for y in years:
            def safe_get_yearly(df, year):
                if df.empty or 'Value' not in df.columns or not isinstance(df.index, pd.DatetimeIndex):
                    return pd.Series(dtype='float64')
                try: return df[df.index.year == year]['Value']
                except: return pd.Series(dtype='float64')

            s_y = safe_get_yearly(s_h, y)
            q_y = safe_get_yearly(b1_h, y)
            
            def get_ret(ser):
                s = ser.dropna()
                if s.empty or len(s) < 2: return 0
                return (s.iloc[-1] / s.iloc[0] - 1) * 100

            s_ret = get_ret(s_y)
            q_ret = get_ret(q_y)
            s_mdd = calculate_mdd(s_y) * 100
            q_mdd = calculate_mdd(q_y) * 100
            
            panic_days, safety_count = 0, 0
            if 'Trade_Label' in s_h.columns:
                s_y_full = s_h[s_h.index.year == y]
                panic_ma = smart_params.get('panic_ma', 200) if smart_params else 200
                panic_days = s_y_full['Trade_Label'].str.contains(f"지연\\(MA{panic_ma}", na=False).sum()
                safety_count = s_y_full['Trade_Label'].str.contains("대피", na=False).sum()
            
            event = HistoryLabView.EVENTS.get(y, "-")
            row = {"연도": f"{y}년", "내 전략 수익": f"{s_ret:+.1f}%", f"{b_names[0]} 수익": f"{q_ret:+.1f}%", "내 전략 MDD": f"{s_mdd:.1f}%", f"{b_names[0]} MDD": f"{q_mdd:.1f}%"}
            
            for target in selected_comparisons:
                t_hist = target['history'].copy()
                if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
                t_hist.index = pd.to_datetime(t_hist.index).normalize()
                t_hist['Value'] = pd.to_numeric(t_hist['Value'], errors='coerce')
                t_y_data = safe_get_yearly(t_hist, y)
                t_ret_val, t_mdd_val = get_ret(t_y_data), calculate_mdd(t_y_data) * 100
                row[f"{target['name']} 수익"] = f"{t_ret_val:+.1f}%"
                row[f"{target['name']} MDD"] = f"{t_mdd_val:.1f}%"

            row.update({"방어(일)": f"{panic_days}일" if panic_days > 0 else "-", "Safety(회)": f"{safety_count}회" if safety_count > 0 else "-", "주요 이벤트": event})
            data.append(row)
            
        df = pd.DataFrame(data).set_index("연도")
        
        def style_returns(val):
            try:
                num = float(val.replace('%','').replace('+',''))
                return f'color: {"#00c853" if num >= 0 else "#ff1744"}; font-weight: bold;'
            except: return ''

        def style_mdd(val):
            try:
                num = float(val.replace('%',''))
                alpha = min(abs(num)/50.0, 1.0) * 0.4
                return f'background-color: rgba(255, 23, 68, {alpha});'
            except: return ''

        st.dataframe(df.style.applymap(style_returns, subset=[c for c in df.columns if "수익" in c])\
                            .applymap(style_mdd, subset=[c for c in df.columns if "MDD" in c]), use_container_width=True, height=520)

    @staticmethod
    def render_overall_summary(s_h, b1_h, b2_h, b3_h, b_names=("QQQ", "QLD", "TQQQ"), selected_comparisons=[]):
        def calculate_mdd_ui(series):
            s = series.dropna()
            if s.empty: return 0, 0
            roll_max = s.cummax()
            dd = (s - roll_max) / roll_max
            return dd.min(), np.sqrt((dd**2).mean()) * 100

        def get_summary(ser, name):
            if ser is None or ser.empty: return None
            total_ret = (ser.iloc[-1] / 10000.0 - 1) * 100
            years_count = (ser.index[-1] - ser.index[0]).days / 365.25
            cagr = ((ser.iloc[-1] / 10000.0) ** (1/years_count) - 1) * 100 if years_count > 0 else 0
            mdd_val, ui_val = calculate_mdd_ui(ser)
            return {"구분": name, "누적 수익률": f"{total_ret:+.1f}%", "연평균(CAGR)": f"{cagr:.1f}%", "최대낙폭(MDD)": f"{mdd_val*100:.1f}%", "궤양 지수(UI)": f"{ui_val:.1f}%"}

        summary_data = [get_summary(s_h['Value'], "내 전략 🔥"), get_summary(b1_h['Value'] if not b1_h.empty else None, f"{b_names[0]}(1x)")]
        for target in selected_comparisons:
            t_hist = target['history'].copy()
            if 'PortfolioValue' in t_hist.columns and 'Value' not in t_hist.columns: t_hist.rename(columns={'PortfolioValue': 'Value'}, inplace=True)
            t_hist.index = pd.to_datetime(t_hist.index).normalize()
            t_hist['Value'] = pd.to_numeric(t_hist['Value'], errors='coerce')
            s_item = get_summary(t_hist['Value'], target['name'])
            if s_item: summary_data.append(s_item)
        
        st.dataframe(pd.DataFrame([s for s in summary_data if s]).set_index("구분"), use_container_width=True)

    @staticmethod
    def render_election_cycle(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"미 대통령 임기 주기별 평균 성과 (Strategy vs {b_name})")
        
        def get_cycle_data(hist, col_name):
            if hist.empty or 'Value' not in hist.columns: return pd.Series(dtype='float64')
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['YearType'] = [(year - 2009) % 4 + 1 for year in df.index.year]
                return df.groupby('YearType')[col_name].mean() * 100
            except: return pd.Series(dtype='float64')

        s_avg = get_cycle_data(strategy_hist, 'Strategy').reindex([1, 2, 3, 4], fill_value=0.0)
        b_avg = get_cycle_data(bench_hist, b_name).reindex([1, 2, 3, 4], fill_value=0.0)
        plot_df = pd.DataFrame({'내 전략': s_avg, f'{b_name}': b_avg}).fillna(0)
        plot_df.index = [f"{i}년차" for i in [1,2,3,4]]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['내 전략'], name='내 전략', marker_color='#00c853', text=plot_df['내 전략'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df[f'{b_name}'], name=f'{b_name}', marker_color='#66b3ff', text=plot_df[f'{b_name}'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
        fig.update_layout(title="Strategy vs Benchmark by Election Cycle Year", template='plotly_white', barmode='group', height=450)
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_midterm_analysis(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"중간선거 주기 집중 분석 (Strategy vs {b_name})")
        def process_midterm(hist, col_name):
            if hist.empty or 'Value' not in hist.columns: return pd.DataFrame(columns=[col_name, 'IsMidterm'])
            try:
                s_y = hist['Value'].resample('Y').last()
                s_ret = s_y.pct_change()
                if not s_y.empty: s_ret.iloc[0] = (s_y.iloc[0] / 10000.0 - 1)
                df = s_ret.to_frame(name=col_name)
                df['IsMidterm'] = [((y - 2009) % 4 + 1) == 2 for y in df.index.year]
                return df
            except: return pd.DataFrame(columns=[col_name, 'IsMidterm'])

        s_df, b_df = process_midterm(strategy_hist, 'Strategy'), process_midterm(bench_hist, b_name)
        s_avg, b_avg = s_df.groupby('IsMidterm')['Strategy'].mean() * 100, b_df.groupby('IsMidterm')[b_name].mean() * 100
        avg_comp = pd.DataFrame({'내 전략': s_avg, f'시장({b_name})': b_avg}).reindex([False, True], fill_value=0.0)
        avg_comp.index = ["비중간선거", "중간선거"]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=avg_comp.index, y=avg_comp['내 전략'], name='내 전략', marker_color='#00c853', text=avg_comp['내 전략'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.add_trace(go.Bar(x=avg_comp.index, y=avg_comp[f'시장({b_name})'], name=f'시장({b_name})', marker_color='#66b3ff', text=avg_comp[f'시장({b_name})'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.update_layout(template='plotly_white', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            s_mid, b_mid = s_df[s_df['IsMidterm']].copy(), b_df[b_df['IsMidterm']].copy()
            s_mid['Year'], b_mid['Year'] = s_mid.index.year, b_mid.index.year
            mid_yearly = pd.merge(s_mid[['Year', 'Strategy']], b_mid[['Year', b_name]], on='Year').set_index('Year') * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=mid_yearly.index.map(str), y=mid_yearly['Strategy'], name='내 전략', marker_color='#00c853', text=mid_yearly['Strategy'].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.add_trace(go.Bar(x=mid_yearly.index.map(str), y=mid_yearly[b_name], name=f'시장({b_name})', marker_color='#66b3ff', text=mid_yearly[b_name].apply(lambda x: f"{x:.1f}%"), textposition='auto'))
            fig.update_layout(template='plotly_white', barmode='group', height=400)
            st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_seasonality(strategy_hist, bench_hist, b_name="QQQ"):
        st.subheader(f"지난 15년간의 월별 평균 수익률 & 승률 (vs {b_name})")
        def get_monthly_stats(hist):
            full_m = range(1, 13)
            if hist.empty or 'Value' not in hist.columns: return pd.Series(index=full_m, data=0.0), pd.Series(index=full_m, data=0.0)
            try:
                rets = hist['Value'].pct_change().dropna()
                df_rets = rets.to_frame(name='Return')
                df_rets['Month'] = df_rets.index.month
                avg = df_rets.groupby('Month')['Return'].mean() * 100 * 21
                win = df_rets.groupby('Month')['Return'].apply(lambda x: (x > 0).mean()) * 100
                return avg.reindex(full_m, fill_value=0.0), win.reindex(full_m, fill_value=0.0)
            except: return pd.Series(index=full_m, data=0.0), pd.Series(index=full_m, data=0.0)

        s_avg, s_win = get_monthly_stats(strategy_hist)
        b_avg, _ = get_monthly_stats(bench_hist)
        from plotly.subplots import make_subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=list(range(1,13)), y=s_avg, name='내 전략 (Avg Ret)', marker_color='#00c853', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Bar(x=list(range(1,13)), y=b_avg, name=f'{b_name} (Avg Ret)', marker_color='#66b3ff', opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=list(range(1,13)), y=s_win, name='내 전략 승률', line=dict(color='#ff1744', width=3)), secondary_y=True)
        fig.update_layout(title="Seasonality Analysis", template='plotly_white', barmode='group', height=500, xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=[f"{m}월" for m in range(1,13)]))
        fig.update_yaxes(title_text="Avg Monthly Return (%)", secondary_y=False)
        fig.update_yaxes(title_text="Strategy Win Rate (%)", secondary_y=True, range=[40, 80])
        st.plotly_chart(fig, use_container_width=True)
