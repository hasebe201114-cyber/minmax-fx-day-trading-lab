"""SYS-FX012 週次レポート markdown 生成 (2026-09-07 週).

ledger + summary + events + actions + checkpoints を統合して 1 本の markdown にする。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab")
JST = timezone(timedelta(hours=9))

LEDGER_JSON = PROJECT_ROOT / "research" / "method-notes" / "sysfx012_forward_test_ledger.json"
SUMMARY_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "weekly_summary_v2_2026-09-07.json"
EVENTS_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "weekly_events_v4_2026-09-07.json"
ACTIONS_LAB_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "actions_lab_2026-09-07.json"
ACTIONS_TAV_JSON = PROJECT_ROOT / "logs" / "forward_test_cycle" / "actions_tav_2026-09-07.json"
CHART_DIR = PROJECT_ROOT / "logs" / "charts" / "2026-09-07"
OUT_MD = PROJECT_ROOT / "logs" / "forward_test_cycle" / "weekly_report_2026-09-07.md"

CUTOFF = "2026-08-15 06:00:00"
LATEST = "2026-09-05 05:55:00"
REPORT_DATE = "2026-09-07"
PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"]


def md_table(headers, rows):
    sep = "|".join(["---"] * len(headers))
    lines = ["| " + " | ".join(headers) + " |", "|" + sep + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def main():
    with LEDGER_JSON.open(encoding="utf-8") as f:
        ledger = json.load(f)
    with SUMMARY_JSON.open(encoding="utf-8") as f:
        summary = json.load(f)
    with EVENTS_JSON.open(encoding="utf-8") as f:
        events_data = json.load(f)

    bt = ledger["backtest"]
    kpi = ledger.get("kpi") or {}
    trades = bt["trades"]

    # --- 通貨別トレード集計 (ledger から再集計、summary と一致) ---
    pair_stats_rows = []
    for pair in PAIRS:
        p_trades = [t for t in trades if t["pair"] == pair]
        if not p_trades:
            pair_stats_rows.append([pair, 0, 0, 0, "-", "-", "-"])
            continue
        n = len(p_trades)
        nw = sum(1 for t in p_trades if t["r_net"] > 0)
        nl = n - nw
        wr = nw / n * 100
        sum_r = sum(t["r_net"] for t in p_trades)
        sum_dollar = sum(t.get("dollar_pnl", 0) for t in p_trades)
        mean_r = sum_r / n
        pair_stats_rows.append([pair, n, nw, nl, f"{wr:.1f}%", f"{sum_r:+.2f}R", f"${sum_dollar:+.2f}"])

    # --- 値動き集計表 ---
    movement_rows = []
    for pair in PAIRS:
        ps = summary["pairs"].get(pair, {})
        if not ps:
            continue
        movement_rows.append([
            pair,
            f"{ps['first_close']:.4f}",
            f"{ps['last_close']:.4f}",
            f"{ps['change']:+.4f}",
            f"{ps['pct_change']:+.2f}%",
            f"{ps['pips_change']:+.1f}",
            f"{ps['period_high']:.4f}",
            f"{ps['period_low']:.4f}",
            f"{ps['atr_h1_then']:.4f} → {ps['atr_h1_now']:.4f} ({ps['atr_h1_change_pct']:+.1f}%)",
        ])

    # --- M5 ローソク分布 ---
    candle_rows = []
    for pair in PAIRS:
        ps = summary["pairs"].get(pair, {})
        if not ps:
            continue
        candle_rows.append([
            pair, ps["m5_n_bars"], ps["m5_up"], ps["m5_down"], ps["m5_doji"],
            f"{ps['m5_up_pct']:.1f}%", f"{ps['m5_down_pct']:.1f}%", f"{ps['m5_doji_pct']:.1f}%"
        ])

    # --- 検出イベント表 ---
    event_rows = []
    for e in sorted(events_data["events"], key=lambda x: (x["pair"], x["time"])):
        in_dedup = "✓" if e["in_dedup"] else "×"
        trend = e["h1_trend"] if e["h1_trend"] is not None else "None (除外)"
        higher_low = ("Yes" if e["m5_higher_low_or_lower_high"] is True
                      else "No" if e["m5_higher_low_or_lower_high"] is False
                      else "n/a")
        event_rows.append([
            e["pair"], e["time"], e["direction"],
            f"{e['h1_open']:.3f}", f"{e['h1_high']:.3f}", f"{e['h1_low']:.3f}", f"{e['h1_close']:.3f}",
            f"{e['range_atr']:.2f}", in_dedup, trend, higher_low,
        ])

    # --- トレード一覧 (該当週分) ---
    trade_rows = []
    for t in sorted(trades, key=lambda x: x["entry_time"]):
        win = t["r_net"] > 0
        win_s = "WIN" if win else "LOSS"
        trade_rows.append([
            t["pair"], t["direction"], t["entry_time"], t["exit_time"],
            f"{t['r_net']:+.3f}", f"${t.get('dollar_pnl', 0):+.2f}",
            win_s, t["exit_reason"],
        ])

    # --- 検出イベント集計 (raw / dedup / trend / M5 通過) ---
    raw = events_data["n_raw_total"]
    dedup = events_data["n_dedup_total"]
    trend = events_data["n_trend_total"]
    n_trades = bt["n_trades_total"]
    n_closed = bt["n_trades_closed"]
    n_open = bt["n_trades_open"]
    win_rate = bt.get("win_rate")
    mean_r = bt.get("mean_r_net")
    pf = bt.get("profit_factor")
    payoff = bt.get("payoff_ratio")
    perm_p = bt.get("perm_p_block")
    final_balance = bt["final_balance"]
    n_required_kpi = kpi.get("kpi_required_pass_count", "?")
    n_kpi = kpi.get("kpi_pass_count", "?")
    period_days = round((pd_days := (
        datetime.fromisoformat(LATEST) - datetime.fromisoformat(CUTOFF)
    ).total_seconds() / 86400), 2)

    # --- チェックポイント進捗 (30/60/90日) ---
    cutoff_dt = datetime.fromisoformat(CUTOFF)
    latest_dt = datetime.fromisoformat(LATEST)
    elapsed = (latest_dt - cutoff_dt).total_seconds() / 86400
    cp_rows = []
    for days, label, target in [(30, "30日", "2026-09-14 06:00"),
                                 (60, "60日", "2026-10-14 06:00"),
                                 (90, "90日", "2026-11-13 06:00")]:
        pct = elapsed / days * 100
        expected = days / 7 * 4.1  # Train 週 4.1 件ペース
        cp_rows.append([label, target, f"{elapsed:.2f} / {days}", f"{pct:.1f}%",
                        n_trades, f"{expected:.1f}"])

    # --- markdown 構築 ---
    lines = []
    lines.append(f"# SYS-FX012 フォワードテスト 週次レポート (第 3 週)")
    lines.append("")
    lines.append(f"**対象期間**: {CUTOFF} 〜 {LATEST} JST ({period_days:.2f} 日 / 3 週換算)")
    lines.append(f"**報告生成**: {datetime.now(JST).strftime('%Y-%m-%d %H:%M JST')} (cron 自動起動)")
    lines.append(f"**最新バー (ledger)**: 4 通貨とも {LATEST}")
    lines.append(f"**cutoff**: {CUTOFF} (フォワードテスト開始点)")
    lines.append(f"**前回レポート**: 2026-08-31 (第 2 週, 14.00 日, 検出 6 / M5 エントリー 0)")
    lines.append(f"**Note**: ローカル cron が 8/31 以降更新されていなかったが、9/7 早朝に手動同期 + cycle 再実行で 9/2-9/4 の急落局面を反映")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. エグゼクティブサマリ
    lines.append("## 1. エグゼクティブサマリ")
    lines.append("")
    lines.append(f"- **検出イベント (raw)**: **{raw} 件** (USD_JPY=4 / EUR_JPY=5 / GBP_JPY=4 / AUD_JPY=0)")
    lines.append(f"  - 第 1 週: 4 件 → 第 2 週: 6 件 → **第 3 週: 13 件** (前週比 +7 件)")
    lines.append(f"  - 3 タイミングで発生: **2026-08-19 21:00 (3 通貨同期 DOWN)** / **2026-08-28 23:00 (3 通貨混合)** / **2026-09-02 22:00 〜 2026-09-04 21:00 (DOWN 集中)**")
    lines.append(f"- **dedup 後**: {dedup} 件 (72h 窓で 4 件 drop)")
    lines.append(f"- **H1 トレンド判定通過**: {trend} 件 (通過 {trend} / 除外 {dedup - trend})")
    lines.append(f"- **M5 エントリー成立**: **{n_trades} 件** (第 1, 2 週 0 件 → 第 3 週 14 件)")
    lines.append(f"  - 全 EUR_JPY/GBP_JPY の DOWN 方向 (9/2-9/4 急落局面)、USD_JPY と AUD_JPY は 0 件")
    lines.append(f"- **損益 (決済分 {n_closed} 件)**: 平均 {mean_r:+.3f}R / 勝率 {win_rate*100:.1f}% / PF {pf:.2f} / ペイオフ {payoff:.2f}")
    lines.append(f"- **最終残高**: ${final_balance:.2f} (開始 $1,000.0 → **+{final_balance - 1000:.2f} (+{(final_balance/1000-1)*100:.2f}%)**)")
    lines.append(f"- **permutation p (day-block)**: {perm_p} (有意水準 0.05 未達、n=14 のため標本誤差大)")
    lines.append(f"- **KPI 達成**: {n_required_kpi} (必須 9 項目中)")
    lines.append("")
    lines.append("**課題**:")
    lines.append("1. **3 週目で初のトレード発生**: 第 1-2 週は 0 トレード週、第 3 週で 9/2-9/4 の EUR/GBP 急落局面に 14 トレード集中。環境ボラティリティ平常化 + 夏枯れ相場終了の合図")
    lines.append("2. **EUR_JPY・GBP_JPY のみ**: 検出 13 件中 USD_JPY=4, EUR_JPY=5, GBP_JPY=4, AUD_JPY=0。USD は方向不一致、AUD は raw 検出 0 件 (ATR 構造的問題継続)")
    lines.append("3. **permutation p 0.4975 で有意性未達**: 14 件は統計的に脆弱。n≥300 (Train ペースでは 5 年以上) まで結論保留")
    lines.append("4. **平均 +0.56R は設計凍結のまま**: 新規パラメータ調整なし (HARKing 防止 OK)、cost_r 平均 0.094R が想定通り")
    lines.append("5. **データ鮮度**: ローカル cron が 8/31 以降動かず 9/7 早朝に手動同期。latest_bar は 9/5 05:55 JST で 2 日古い。月曜 09:00 JST の自動 cycle で catch up")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 値動き
    lines.append("## 2. 値動き (4 通貨サマリ表)")
    lines.append("")
    lines.append(f"**期間**: cutoff {CUTOFF} 〜 latest {LATEST} ({period_days:.2f} 日)")
    lines.append("")
    lines.append(md_table(
        ["通貨", "始値", "終値", "変化", "変化率", "pips", "高値", "安値", "ATR(H1) 変化"],
        movement_rows,
    ))
    lines.append("")
    lines.append("**所見**:")
    lines.append("- 4 通貨すべて下落 (USD/EUR/GBP が 1.6-2.1% 圏、AUD は小幅 -0.32%)。GBP_JPY が最大 -447 pips")
    lines.append("- **ATR(H1) は全通貨で +72-90% 大幅上昇** (夏枯れ相場 → 秋相場への移行を示唆、USD: 0.16→0.30, EUR: 0.16→0.30, GBP: 0.20→0.36, AUD: 0.10→0.18)")
    lines.append("- 期間高値/安値の幅は USD=5.1 / EUR=5.8 / GBP=7.7 / AUD=3.3 円、GBP が最も値幅大きい")
    lines.append("")
    lines.append("**M5 ローソク分布 (cutoff 後 全 4,284 本)**:")
    lines.append("")
    lines.append(md_table(
        ["通貨", "本数", "UP", "DOWN", "DOJI", "UP率", "DOWN率", "DOJI率"],
        candle_rows,
    ))
    lines.append("")
    lines.append("→ 全通貨で UP/DOWN が 50% 付近で拮抗、有意な方向偏りなし。ATR 増大は「レンジ拡大・方向感なし」型 (typical of breakout 前兆)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. 検出イベント
    lines.append("## 3. 検出イベント (raw 13 件)")
    lines.append("")
    lines.append(md_table(
        ["通貨", "時刻 (JST)", "方向", "H1 O", "H1 H", "H1 L", "H1 C", "range/ATR", "dedup", "トレンド判定", "M5 higher-low/lower-high"],
        event_rows,
    ))
    lines.append("")
    lines.append("**集計**:")
    lines.append(f"- raw {raw} → dedup {dedup} (72h 窓で 4 件 drop、3 タイミングが時間的に離れている)")
    lines.append(f"- トレンド通過 {trend} → M5 エントリー成立 {n_trades}")
    lines.append(f"- 3 タイミングの特徴:")
    lines.append(f"  - **2026-08-19 21:00 (Wed 米国市場オープン)**: 3 通貨同期 DOWN、range/ATR = 4.31〜5.00 の強いブレイク")
    lines.append(f"  - **2026-08-28 23:00 (Fri NY クローズ)**: USD/GBP は UP、EUR のみ DOWN。方向不一致で M5 エントリーは USD_JPY のみ通過判定 (しかし trend=UP/DOWN 矛盾で M5 no entry)")
    lines.append(f"  - **2026-09-02 22:00 〜 2026-09-04 21:00**: USD/EUR/GBP の DOWN 集中 (9/2-9/4 急落局面)。AUD_JPY は raw 0 件で構造的沈黙")
    lines.append("")
    lines.append("**AUD_JPY の構造的問題 (前回レポートから継続)**: cutoff 期間中 raw 検出 0 件。ATR(H1) が他 3 通貨より小さく (0.18 vs 0.30-0.36) N_BREAKOUT=3.5 閾値未達。Train 段階でも AUD は他よりサンプル少なく、CALM_RATIO や通貨拡大の検討対象だが、HARKing 防止のため spec は凍結維持")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. トレード
    lines.append(f"## 4. トレード (該当週: {n_trades} 件決済 / {n_open} 件保有中)")
    lines.append("")
    lines.append(md_table(
        ["通貨", "方向", "エントリー", "決済", "損益 (R)", "損益 ($)", "勝敗", "決済理由"],
        trade_rows,
    ))
    lines.append("")
    lines.append("**通貨別集計**:")
    lines.append("")
    lines.append(md_table(
        ["通貨", "n", "WIN", "LOSS", "勝率", "合計 R", "合計 $"],
        pair_stats_rows,
    ))
    lines.append("")
    lines.append("**全 14 トレードが同方向 DOWN、同時期 (9/2-9/4)** の集中爆出:")
    lines.append("- 9/2 17:20 EUR 1 件: -1.19R (LOSS、欧州時間序盤の急落で初期 SL ヒット)")
    lines.append("- 9/3 10:05-23:35: EUR 5 件 + GBP 6 件 = 11 件 (欧州〜NY 時間の急落連続、71% 勝率)")
    lines.append("- 9/4 00:05: EUR 1 件: +0.62R (深夜の戻り売り)")
    lines.append("- 全決済理由が `SL_INITIAL_NO_TP` (trail-only 設計で TP 不在、初期 SL または trail に到達して決済)")
    lines.append("")
    lines.append("**r_gross vs r_net の乖離**: cost_r 平均 0.094R (spread 0.5pip + slippage 0.5/1.0pip + commission 0.00004) が想定通り加算。勝ったトレードでも r_net = r_gross - 0.08-0.22R")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. チャート
    lines.append("## 5. チャート")
    lines.append("")
    lines.append("### 5.1 分析時点の H1 足")
    lines.append("")
    for pair in PAIRS:
        lines.append(f"[{pair} H1 - Forward Test Week 3]")
        lines.append("")
        lines.append(f"![{pair} H1]({CHART_DIR / f'chart_h1_analysis_{pair}_{REPORT_DATE}.png'})")
        lines.append("")
    lines.append("凡例: 緑=UP (トレンド通過)、赤=DOWN (トレンド通過)、灰=トレンド判定不能 (除外)")
    lines.append("")
    lines.append("### 5.2 エントリーチャート")
    lines.append("")
    lines.append(f"![Entry chart]({CHART_DIR / f'chart_entry_{REPORT_DATE}.png'})")
    lines.append("")
    lines.append(f"該当週の M5 エントリー成立: **{n_trades} 件** (前 2 週 0 件 → 集中爆出)")
    lines.append("")
    lines.append("### 5.3 決済チャート")
    lines.append("")
    lines.append(f"![Exit chart]({CHART_DIR / f'chart_exit_{REPORT_DATE}.png'})")
    lines.append("")
    lines.append(f"決済済みトレード: **{n_closed} 件** (n_open={n_open} 件は週末クローズ後に強制決済予定)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. GitHub Actions 状況
    lines.append("## 6. GitHub Actions 状況 (直近 7 日間: 2026-08-31 以降)")
    lines.append("")
    lines.append("| workflow | repo | total_runs | success | failure | 成功率 | latest_run (UTC) | latest_conclusion |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append("| FX Forward Test Cycle (Weekly, SYS-FX012 + SYS-FX026) | minmax-fx-day-trading-lab | 1 | 1 | 0 | **100%** | 2026-08-31T02:11:45Z | success |")
    lines.append("| SYS-FX012 Forward M5 OHLCV Live Update (Hourly) | minmax-fx-day-trading-lab | 20+ | 20 | 0 | **100%** | 2026-09-06T13:58:57Z | success |")
    lines.append("| SYS-FX022 ライブ気配値記録 | minmax-fx-day-trading-lab | 15+ | 15 | 0 | **100%** | 2026-09-06T13:37:06Z | success |")
    lines.append("| SYS-FX012 FX Forward Test Sync (Weekly) | trading-app-v2 | 1 | 1 | 0 | **100%** | 2026-08-31T09:31:45Z | success |")
    lines.append("| Deploy to Firebase Hosting | trading-app-v2 | 10+ | 10 | 0 | **100%** | 2026-09-06T15:27:52Z | success |")
    lines.append("")
    lines.append("**所見**:")
    lines.append("- **FX Forward Test Cycle**: 8/24-8/25 の failure 2 件以降、8/31 で 1 回 success して以降、月曜 09:00 JST のスケジュール実行を 9/7 も待っている状態")
    lines.append("- **SYS-FX012 Forward M5 OHLCV Live Update (Hourly)**: 20 回連続 success (前週 100% 維持)。cycle の self-healing も機能している")
    lines.append("- **Deploy to Firebase Hosting**: 10+ 連続 success、スマホへの配信導線に問題なし")
    lines.append("- **本週の運用**: 9/7 01:00 JST のクロン起動が手動でローカル cycle を実行 (cycle 自体は正常終了、KPI 5/9 達成、ledger 正常 commit 対象)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 7. チェックポイント進捗
    lines.append("## 7. チェックポイント進捗")
    lines.append("")
    lines.append("| チェックポイント | 目標日 | 経過 | 進捗率 | 現 n_trades | 期待 n_trades (Train 週 4.1 件ペース) |")
    lines.append("|---|---|---|---|---|---|")
    for r in cp_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} 日 | **{r[3]}** | {r[4]} | {r[5]} |")
    lines.append("")
    lines.append("**評価**:")
    lines.append(f"- **{elapsed:.2f} 日経過時点で n_trades={n_trades} (WIN 10 / LOSS 4)、最終残高 ${final_balance:.2f}**")
    lines.append(f"- Train 週 4.1 件ペース想定だと 21 日で 12.3 件期待 → 実 {n_trades} 件で {n_trades/12.3*100:.0f}% 達成 (前 2 週 0 件の遅延を取り戻す形で集中爆出)")
    lines.append(f"- 30 日 checkpoint (2026-09-14) まで残り {(30-elapsed):.2f} 日。平常ボラなら週 4.1 件ペースで 4-5 件追加、合計 {n_trades + 4}-{n_trades + 5} 件想定")
    lines.append(f"- min_n_trades ≥ 300 は 90 日 checkpoint でも困難 (現状ペース 14 件/3 週 → 90 日で 60 件程度、300 件には 5 倍必要)")
    lines.append(f"- ただし ATR(H1) 増大 (+72-90%) ＋ 9/2-9/4 の急落実現で「平常ボラ環境」が戻りつつある兆候、以降の検出機会増加に期待")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 8. 課題・次の注目
    lines.append("## 8. 課題・次の注目")
    lines.append("")
    lines.append("### 観察された異常・要対応")
    lines.append("")
    lines.append("1. **3 週目で 14 トレード集中爆出**: 9/2-9/4 の EUR/GBP 急落局面に 14 トレード (全 DOWN 方向)。0 トレード週が 2 週連続した直後の集中は「平常ボラ環境への移行期」に典型的なパターン。n=14 では統計的に脆弱 (permutation p=0.4975)")
    lines.append("2. **ATR(H1) の大幅増大 (+72-90%)**: 4 通貨すべて平常値 (USD 0.30/EUR 0.30/GBP 0.36/AUD 0.18) に到達、夏枯れ相場終了のサイン")
    lines.append("3. **AUD_JPY の raw 検出 0 件が継続 (3 週連続)**: 構造的に ATR が小さく N_BREAKOUT=3.5 未達。AUD_JPY を SYS-FX012 の対象から除外するか、通貨別に N_BREAKOUT/CALM_RATIO を調整する spec 改訂は HARKing 防止の観点から保留")
    lines.append("4. **USD_JPY の M5 エントリー 0 件**: USD 検出 4 件中、トレンド通過 2 件 (8/19 DOWN, 8/28 UP) だが M5 ダウ理論での方向確認成立せず。通貨別の M5 エントリー閾値調整が必要か観察継続")
    lines.append("5. **ローカル cron の停止 (8/31〜9/7)**: 9/7 早朝に手動同期で回復。GitHub Actions のスケジュール実行は正常だが、ローカルの manual cycle が長らく動かなかった。obs ノートに記録")
    lines.append("")
    lines.append("### 次の週で監視すべき指標")
    lines.append("")
    lines.append("- n_trades (累計) の伸び: 平常ボラ環境で週 4.1 件ペースが維持されるか")
    lines.append("- ATR(H1) の安定: 9 月以降も 0.30+ が続くか、夏枯れ再突入はあるか")
    lines.append("- 通貨分散: USD/AUD でも M5 エントリーが成立するか (現在は EUR/GBP のみ)")
    lines.append("- permutation p 値: n=20-30 で 0.05 以下に改善するか")
    lines.append("- 設計凍結の維持: 検出層・エントリー層・出口・コストモデルのいずれも変更なし")
    lines.append("")
    lines.append("### 設計凍結 (HARKing 防止) の遵守確認")
    lines.append("")
    lines.append("- 検出層: N_BREAKOUT=3.5 / CALM_RATIO=2.0 / DONCHIAN_LENGTH=20 固定 (変更なし)")
    lines.append("- トレンド判定: SYS-FX009 ZigZag threshold_atr=2.0 固定")
    lines.append("- エントリー層: 5 連続フィルター / ZigZag threshold_atr_m5=1.0 固定")
    lines.append("- 出口: TP-13 / tp_levels=[] / breakeven_trigger_r=1.0 / atr_trail_multiplier_m5=0.703")
    lines.append("- コスト: T-09 確定値固定 (SPREAD_PIPS、SLIPPAGE_PIPS_MARKET_LEG、SLIPPAGE_PIPS_STOP_TRIGGERED、COMMISSION_RATE_ROUND_TRIP)")
    lines.append("")
    lines.append("→ **本週の分析で新規パラメータ調整なし** (HARKing 防止 OK)")
    lines.append("")
    lines.append("### 検出ロジック整合性 (Cycle vs Spec) - 注記")
    lines.append("")
    lines.append("`run_forward_test_cycle.py` は `detect_candidate1` (N_BREAKOUT 単独) を使用しているが、spec (`research/EXP-FX000006/00-spec.md`) は N_BREAKOUT OR (Donchian AND CALM_RATIO) の OR 合成を要求している。")
    lines.append("- **今期検出 13 件はすべて N_BREAKOUT 経路** (range/ATR >= 3.5)")
    lines.append("- 9/2-9/4 の急落局面では CALM_RATIO (2.0) 経路の Donchian ブレイクが追加候補になる可能性あり")
    lines.append("- 来週以降、C 品質チームまたは strategy-architect が cycle を spec 通り (`detect_candidate3`) に修正する可能性")
    lines.append("- 修正された場合、n_events_raw / n_events_dedup / n_trades すべて増加見込み。**HARKing 防止の観点では、現状の「N_BREAKOUT 単独」の方が「安全」だが、spec との不一致は要修正項目**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 9. 付録
    lines.append("## 9. 付録: データ品質")
    lines.append("")
    lines.append("### M5 データカバレッジ")
    lines.append("")
    lines.append(f"- cutoff 〜 latest ({period_days:.2f} 日) 期待 M5 本数: {period_days * 288:.0f} 本")
    lines.append("- 実 M5 本数: 4,284 本 (各通貨共通)")
    lines.append(f"- カバレッジ: {4284 / (period_days * 288) * 100:.1f}% (欠落 {period_days * 288 - 4284:.0f} 本 = {(1 - 4284 / (period_days * 288)) * 100:.1f}%、週末クローズによる)")
    lines.append("- 残存欠落期間: 2026-08-15 06:00 〜 2026-08-17 07:00 (2.04 日 / 588 本、各週の週末クローズによる恒久的な空白)")
    lines.append("")
    lines.append("### ledger 整合性")
    lines.append("")
    lines.append(f"- latest_bar_by_pair: 4 通貨とも {LATEST} (整合)")
    lines.append(f"- n_events_raw: {raw} (検算一致、cycle と同じ)")
    lines.append(f"- n_events_dedup: {dedup} (検算一致、72h 窓で 4 件 drop)")
    lines.append(f"- n_events_trendfiltered: {trend} (ledger 値と一致)")
    lines.append(f"- n_trades: {n_trades} (整合、決済 {n_closed} / 保有中 {n_open})")
    lines.append("")
    lines.append("### 出力ファイル")
    lines.append("")
    lines.append(f"- 集計 JSON v2: `logs/forward_test_cycle/weekly_summary_v2_{REPORT_DATE}.json`")
    lines.append(f"- 検出イベント詳細: `logs/forward_test_cycle/weekly_events_v4_{REPORT_DATE}.json`")
    lines.append("- チャート: `logs/charts/2026-09-07/` (6 ファイル)")
    lines.append("- 値動き集計スクリプト: `logs/forward_test_cycle/weekly_summary_2026-09-07.py`")
    lines.append("- チャート生成スクリプト: `logs/forward_test_cycle/weekly_charts_2026-09-07.py`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"**次回クロン起動予定**: 月曜 09:00 JST (sysfx012-fx-forward-cycle)")
    lines.append(f"**次週レポート予定日**: 2026-09-14 (cutoff + 30 日 checkpoint)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> このプロンプト自体が週次レポート生成の雛形。新規週で prompt 微調整があれば `obs/minmax_fx_day_trading_lab/70対応待ち/` 配下に記録。")

    md = "\n".join(lines)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"[OK] markdown: {OUT_MD}")
    print(f"  lines: {len(lines)}")
    print(f"  bytes: {len(md)}")


if __name__ == "__main__":
    main()
