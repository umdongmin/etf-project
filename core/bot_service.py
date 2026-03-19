"""
Telegram 알림 봇 서비스 레이어

run_tqqq_bot(token, chat_id, dry_run=False)
  - 포트폴리오 신호 계산 → 계좌별 Telegram 알림 발송
  - Flask(main.py)와 완전히 분리된 순수 비즈니스 로직
"""

import os
import datetime
import json


def run_tqqq_bot(token, chat_id, dry_run=False):
    """포트폴리오 구조 기반 신호 계산 → 계좌별 Telegram 알림 발송"""
    try:
        import requests
        from core.data import DataService
        from core.storage import StrategyStorage, AssetStorage
        from core.signal_service import compute_signals, extract_signals, alloc_from_signals
        from core.constants import STAGE_RATIOS, STAGE_LABELS, STAGE_ICONS

        def _get_price(ticker, prices):
            return DataService.get_price(ticker, price_map=prices)

        # ── 1. 포트폴리오 로드 ────────────────────────────────────────
        PORTFOLIO_NAME = os.environ.get("TELEGRAM_PORTFOLIO", "A-공격형")
        portfolio_data = StrategyStorage.load_portfolio(PORTFOLIO_NAME)
        if not portfolio_data:
            print(f"❌ 포트폴리오 '{PORTFOLIO_NAME}' 로드 실패", flush=True)
            return
        configs  = portfolio_data.get('configs', [])
        start_date = datetime.date(2026, 1, 1)

        # ── 2. 데이터 수집 ────────────────────────────────────────────
        print("📡 데이터 수집 중...", flush=True)
        data_dict, fg_df, vxn_df, macro_df, news_df, fetch_status, _ = DataService.load_all_data()
        if not data_dict:
            print("❌ 데이터 로드 실패", flush=True)
            return

        cur_prices = DataService.fetch_current_prices(list(data_dict.keys()) + ['^VIX', '^VXN'])
        print(f"{'✅' if cur_prices else '⚠️'} 현재가 {'조회 완료' if cur_prices else '없음 — 종가 기준 사용'}", flush=True)

        # ── 3. 신호 계산: 종가(현재) + 실시간(변경) ──────────────────
        print("🔍 종가 기준 신호 계산 중...", flush=True)
        closing_result = compute_signals(
            PORTFOLIO_NAME, data_dict, fg_df, vxn_df, macro_df, news_df,
            start_date, cur_prices=None,
        )
        if not closing_result:
            print("❌ 종가 신호 계산 실패", flush=True)
            return

        print("🔍 실시간 신호 계산 중...", flush=True)
        realtime_result = compute_signals(
            PORTFOLIO_NAME, data_dict, fg_df, vxn_df, macro_df, news_df,
            start_date, cur_prices=cur_prices,
        )
        if not realtime_result:
            realtime_result = closing_result

        closing_signals  = extract_signals(closing_result,  configs)
        realtime_signals = extract_signals(realtime_result, configs)
        target_alloc     = alloc_from_signals(configs, realtime_signals)
        portfolio_sig    = realtime_result.get('portfolio', {})

        # ── 4. 계좌별 메시지 생성 및 전송 ────────────────────────────
        today_str = datetime.date.today().strftime('%Y-%m-%d')
        accounts  = AssetStorage.list_accounts()

        for acc in accounts:
            acc_id   = acc['id']
            acc_name = acc['name']

            holdings = AssetStorage.list_holdings(acc_id)

            holdings_map = {}
            total_value  = 0.0
            for h in holdings:
                ticker = h['asset_id'].upper()
                price  = _get_price(ticker, cur_prices or {})
                val    = h['quantity'] * price
                total_value += val
                holdings_map[ticker] = {'qty': h['quantity'], 'price': price, 'value': val}

            sync_cfg = AssetStorage.load_sync_config(acc_id)
            base_capital = (sync_cfg['total_capital']
                            if sync_cfg and sync_cfg['total_capital'] > 0
                            else total_value)
            target_qty = {}
            for ticker, weight in target_alloc.items():
                price = _get_price(ticker, cur_prices or {})
                target_qty[ticker] = round(base_capital * weight / price) if price > 0 else 0

            capital_note = f" (기준 ${sync_cfg['total_capital']:,.0f})" if sync_cfg else ""
            lines = [
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"👤 *{acc_name}*  |  📅 {today_str}",
                f"📊 포트폴리오: *{PORTFOLIO_NAME}*",
                f"💰 총 평가액: ${total_value:,.0f}{capital_note}",
                f"━━━━━━━━━━━━━━━━━━━━━━",
            ]
            action_statuses = []

            for name, new_sig in realtime_signals.items():
                stype     = new_sig['strat_type']
                new_stage = new_sig['stage']
                new_icon  = STAGE_ICONS.get(new_stage, '⚪')
                new_label = STAGE_LABELS.get(new_stage, f'S{new_stage}')
                alert     = ' ⚠️' if new_stage >= 2 else ''

                cur_sig   = closing_signals.get(name, new_sig)
                cur_stage = cur_sig['stage']
                cur_icon  = STAGE_ICONS.get(cur_stage, '⚪')
                cur_label = STAGE_LABELS.get(cur_stage, f'S{cur_stage}')

                if stype == 'equity':
                    base_t = new_sig['base_asset'].upper()
                    lev_t  = new_sig['lev_asset'].upper()
                    cur_base_qty = holdings_map.get(base_t, {}).get('qty', 0)
                    cur_lev_qty  = holdings_map.get(lev_t,  {}).get('qty', 0)
                    tgt_base_qty = target_qty.get(base_t, 0)
                    tgt_lev_qty  = target_qty.get(lev_t,  0)
                    diff_base = tgt_base_qty - int(cur_base_qty)
                    diff_lev  = tgt_lev_qty  - int(cur_lev_qty)

                    if new_stage != cur_stage:
                        status = '매수' if new_stage > cur_stage else '매도'
                    elif diff_base != 0 or diff_lev != 0:
                        status = '조정'
                    else:
                        status = '보유'

                    lines.append(f"\n📈 *{name}*{alert}  [{status}]")
                    lines.append(f"  현재: {cur_icon} {cur_label}")
                    lines.append(f"        {base_t}:{int(cur_base_qty)}주  {lev_t}:{int(cur_lev_qty)}주")
                    lines.append(f"  변경: {new_icon} {new_label}")
                    lines.append(f"        {base_t}:{tgt_base_qty}주  {lev_t}:{tgt_lev_qty}주")

                    result_parts = []
                    if diff_base > 0:  result_parts.append(f"🟢 {base_t} {diff_base}주 매수")
                    elif diff_base < 0: result_parts.append(f"🔴 {base_t} {abs(diff_base)}주 매도")
                    if diff_lev > 0:   result_parts.append(f"🟢 {lev_t} {diff_lev}주 매수")
                    elif diff_lev < 0:  result_parts.append(f"🔴 {lev_t} {abs(diff_lev)}주 매도")
                    action_statuses.append(status)
                    lines.append(f"  결과: {' / '.join(result_parts) if result_parts else '변동 없음'}")

                else:
                    base_t = new_sig['base_asset'].upper()
                    lev_t  = (new_sig['lev_asset'] or '').upper()
                    cur_base_qty = holdings_map.get(base_t, {}).get('qty', 0)
                    cur_lev_qty  = holdings_map.get(lev_t,  {}).get('qty', 0) if lev_t else 0
                    tgt_base_qty = target_qty.get(base_t, 0)
                    tgt_lev_qty  = target_qty.get(lev_t,  0) if lev_t else 0
                    diff_base = tgt_base_qty - int(cur_base_qty)
                    diff_lev  = tgt_lev_qty  - int(cur_lev_qty)

                    status = '보유' if diff_base == 0 and diff_lev == 0 else '조정'
                    lines.append(f"\n📉 *{name}*  [{status}]")
                    lines.append(f"  현재: {cur_icon} {cur_label}  {base_t}:{int(cur_base_qty)}주")
                    lines.append(f"  변경: {new_icon} {base_t}" + (f" → {lev_t}:{tgt_lev_qty}주" if lev_t and tgt_lev_qty > 0 else f":{tgt_base_qty}주"))
                    result_parts = []
                    if diff_base > 0:  result_parts.append(f"🟢 {base_t} {diff_base}주 매수")
                    elif diff_base < 0: result_parts.append(f"🔴 {base_t} {abs(diff_base)}주 매도")
                    if lev_t:
                        if diff_lev > 0:   result_parts.append(f"🟢 {lev_t} {diff_lev}주 매수")
                        elif diff_lev < 0:  result_parts.append(f"🔴 {lev_t} {abs(diff_lev)}주 매도")
                    action_statuses.append(status)
                    lines.append(f"  결과: {' / '.join(result_parts) if result_parts else '변동 없음'}")

            # ── 실잔고 기반 드리프트 계산 ──────────────────────────────
            PRESET_BANDS = {
                'hybrid':     0.07, 'trend':    0.07,
                'band_5pct':  0.05, 'band_10pct': 0.10,
                'asymmetric': 0.15,
            }
            preset_key  = portfolio_data.get('rebalance_preset', 'hybrid')
            upper_band  = PRESET_BANDS.get(preset_key, 0.07)
            rebal_method = 'none' if preset_key == 'none' else 'threshold'

            strat_drifts = {}
            needs_rebal  = False
            if total_value > 0:
                # asset → strategy 매핑
                asset_to_strat = {}
                for cfg in configs:
                    p = cfg.get('params', {})
                    lev = (p.get('leverage_asset') or '').upper()
                    base = (p.get('base_asset') or '').upper()
                    stype = cfg.get('type', 'equity')
                    related = ({lev, base, 'QLD', 'TQQQ', 'QQQ'} if stype == 'equity'
                               else {lev, base, 'TLT', 'TMF', 'TMV', 'TBF', 'BIL', 'IEF'})
                    for a in related:
                        if a:
                            asset_to_strat[a] = cfg['name']

                strat_values = {cfg['name']: 0.0 for cfg in configs}
                for ticker, hdata in holdings_map.items():
                    sname = asset_to_strat.get(ticker)
                    if sname and sname in strat_values:
                        strat_values[sname] += hdata['value']

                for cfg in configs:
                    sname   = cfg['name']
                    target  = cfg.get('weight', 0)
                    actual  = strat_values[sname] / total_value
                    drift_v = round(actual - target, 4)
                    strat_drifts[sname] = {'target': target, 'actual': actual, 'drift': drift_v}
                    if rebal_method != 'none' and abs(drift_v) > upper_band:
                        needs_rebal = True

            max_drift_pct = max((abs(v['drift']) * 100 for v in strat_drifts.values()), default=0)
            rebal = '필요 ⚠️' if needs_rebal else '정상 ✅'
            lines.append(f"\n🔄 Drift: `{max_drift_pct:.1f}%`  Rebalance: {rebal} (±{upper_band*100:.0f}%)")
            if strat_drifts:
                for sname, d in strat_drifts.items():
                    flag = ' ⚠️' if abs(d['drift']) > upper_band else ''
                    lines.append(f"  {sname}: 목표 {d['target']*100:.0f}% → 실제 {d['actual']*100:.1f}% ({d['drift']*100:+.1f}%{flag})")

            # 데이터 이상 경고
            data_warnings = []
            if fetch_status.get('FRED') != 'Success':
                data_warnings.append("FRED 실패 (매크로 지표 없음)")
            if fetch_status.get('VXN') != 'Success':
                data_warnings.append("VXN 실패 (변동성 지표 없음)")
            data_warnings.extend(fetch_status.get('DataWarnings', []))
            if data_warnings:
                lines.append(f"\n⚠️ 데이터 경고: {' / '.join(data_warnings)}")

            lines.append("━━━━━━━━━━━━━━━━━━━━━━")

            msg = '\n'.join(lines)

            print(f"\n{'='*50}", flush=True)
            print(f"📋 [{acc_name}] 메시지 미리보기", flush=True)
            print('='*50, flush=True)
            print(msg, flush=True)
            print('='*50 + '\n', flush=True)

            has_action  = any(s != '보유' for s in action_statuses)
            should_send = has_action or needs_rebal

            # ── 5. 신호 감사 로그 저장 (신호/리밸런싱 발생 시만) ─────────
            if should_send and not dry_run:
                # 핵심 지표 스냅샷 구성
                snapshot = {'fetch_status': {k: v for k, v in fetch_status.items() if k != 'DataWarnings'}}
                if 'QQQ' in data_dict:
                    qqq = data_dict['QQQ']
                    snapshot['QQQ'] = {
                        'price': round(float(qqq['Close'].iloc[-1]), 2),
                        'RSI':   round(float(qqq['RSI'].iloc[-1]), 1) if 'RSI' in qqq else None,
                        'MACD':  round(float(qqq['MACD'].iloc[-1]), 4) if 'MACD' in qqq else None,
                    }
                if 'TQQQ' in data_dict:
                    snapshot['TQQQ'] = {'price': round(float(data_dict['TQQQ']['Close'].iloc[-1]), 2)}
                if not vxn_df.empty and 'VXN' in vxn_df.columns:
                    snapshot['VXN'] = round(float(vxn_df['VXN'].iloc[-1]), 2)
                if not macro_df.empty and 'DFEDTARU' in macro_df.columns:
                    snapshot['FedRate'] = round(float(macro_df['DFEDTARU'].iloc[-1]), 2)
                notes_json = json.dumps(snapshot, ensure_ascii=False)

                for name, new_sig in realtime_signals.items():
                    cur_sig   = closing_signals.get(name, new_sig)
                    cur_stage = cur_sig['stage']
                    new_stage = new_sig['stage']
                    action    = action_statuses[list(realtime_signals.keys()).index(name)]
                    if action != '보유':
                        AssetStorage.save_signal_event(
                            account_id=acc_id,
                            portfolio_name=PORTFOLIO_NAME,
                            strategy_name=name,
                            strat_type=new_sig['strat_type'],
                            prev_stage=cur_stage,
                            new_stage=new_stage,
                            prev_asset=cur_sig['base_asset'],
                            new_asset=new_sig['base_asset'],
                            action=action,
                            total_value=total_value,
                            notes=notes_json,
                        )
                        print(f"📝 [{name}] 신호 감사 로그 저장 완료 (action={action})", flush=True)

            if not dry_run:
                if should_send:
                    send_url = f"https://api.telegram.org/bot{token}/sendMessage"
                    payload  = {"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}
                    r = requests.post(send_url, json=payload, timeout=10)
                    print(f"📡 [{acc_name}] Telegram Send: {r.status_code}", flush=True)
                else:
                    print(f"⏭️ [{acc_name}] 변동 없음 — 전송 생략", flush=True)
            else:
                print(f"🔕 [DRY RUN] {acc_name} {'전송 예정' if should_send else '전송 생략 (변동 없음)'}", flush=True)

    except Exception as e:
        print(f"❌ [Bot Service Error]: {e}", flush=True)
        import traceback
        traceback.print_exc()
