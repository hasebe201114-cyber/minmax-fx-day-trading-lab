# SYS-FX012 フォワードテスト 週次レポート (第 3 週)

**対象期間**: 2026-08-15 06:00:00 〜 2026-09-05 05:55:00 JST (21.00 日 / 3 週換算)
**報告生成**: 2026-09-07 01:34 JST (cron 自動起動)
**最新バー (ledger)**: 4 通貨とも 2026-09-05 05:55:00
**cutoff**: 2026-08-15 06:00:00 (フォワードテスト開始点)
**前回レポート**: 2026-08-31 (第 2 週, 14.00 日, 検出 6 / M5 エントリー 0)
**Note**: ローカル cron が 8/31 以降更新されていなかったが、9/7 早朝に手動同期 + cycle 再実行で 9/2-9/4 の急落局面を反映

---

## 1. エグゼクティブサマリ

- **検出イベント (raw)**: **13 件** (USD_JPY=4 / EUR_JPY=5 / GBP_JPY=4 / AUD_JPY=0)
  - 第 1 週: 4 件 → 第 2 週: 6 件 → **第 3 週: 13 件** (前週比 +7 件)
  - 3 タイミングで発生: **2026-08-19 21:00 (3 通貨同期 DOWN)** / **2026-08-28 23:00 (3 通貨混合)** / **2026-09-02 22:00 〜 2026-09-04 21:00 (DOWN 集中)**
- **dedup 後**: 9 件 (72h 窓で 4 件 drop)
- **H1 トレンド判定通過**: 5 件 (通過 5 / 除外 4)
- **M5 エントリー成立**: **14 件** (第 1, 2 週 0 件 → 第 3 週 14 件)
  - 全 EUR_JPY/GBP_JPY の DOWN 方向 (9/2-9/4 急落局面)、USD_JPY と AUD_JPY は 0 件
- **損益 (決済分 14 件)**: 平均 +0.563R / 勝率 71.4% / PF 2.73 / ペイオフ 1.09
- **最終残高**: $1079.81 (開始 $1,000.0 → **+79.81 (+7.98%)**)
- **permutation p (day-block)**: 0.4975 (有意水準 0.05 未達、n=14 のため標本誤差大)
- **KPI 達成**: 5/9 (必須 9 項目中)

**課題**:
1. **3 週目で初のトレード発生**: 第 1-2 週は 0 トレード週、第 3 週で 9/2-9/4 の EUR/GBP 急落局面に 14 トレード集中。環境ボラティリティ平常化 + 夏枯れ相場終了の合図
2. **EUR_JPY・GBP_JPY のみ**: 検出 13 件中 USD_JPY=4, EUR_JPY=5, GBP_JPY=4, AUD_JPY=0。USD は方向不一致、AUD は raw 検出 0 件 (ATR 構造的問題継続)
3. **permutation p 0.4975 で有意性未達**: 14 件は統計的に脆弱。n≥300 (Train ペースでは 5 年以上) まで結論保留
4. **平均 +0.56R は設計凍結のまま**: 新規パラメータ調整なし (HARKing 防止 OK)、cost_r 平均 0.094R が想定通り
5. **データ鮮度**: ローカル cron が 8/31 以降動かず 9/7 早朝に手動同期。latest_bar は 9/5 05:55 JST で 2 日古い。月曜 09:00 JST の自動 cycle で catch up

---

## 2. 値動き (4 通貨サマリ表)

**期間**: cutoff 2026-08-15 06:00:00 〜 latest 2026-09-05 05:55:00 (21.00 日)

| 通貨 | 始値 | 終値 | 変化 | 変化率 | pips | 高値 | 安値 | ATR(H1) 変化 |
|---|---|---|---|---|---|---|---|---|
| USD_JPY | 159.4350 | 156.3010 | -3.1340 | -1.97% | -313.4 | 160.3960 | 155.2990 | 0.1594 → 0.3038 (+90.5%) |
| EUR_JPY | 184.4110 | 181.5230 | -2.8880 | -1.57% | -288.8 | 186.0270 | 180.2490 | 0.1636 → 0.3011 (+84.0%) |
| GBP_JPY | 215.7460 | 211.2760 | -4.4700 | -2.07% | -447.0 | 217.4780 | 209.7910 | 0.1966 → 0.3605 (+83.4%) |
| AUD_JPY | 112.9610 | 112.6020 | -0.3590 | -0.32% | -35.9 | 114.9660 | 111.7030 | 0.1033 → 0.1780 (+72.3%) |

**所見**:
- 4 通貨すべて下落 (USD/EUR/GBP が 1.6-2.1% 圏、AUD は小幅 -0.32%)。GBP_JPY が最大 -447 pips
- **ATR(H1) は全通貨で +72-90% 大幅上昇** (夏枯れ相場 → 秋相場への移行を示唆、USD: 0.16→0.30, EUR: 0.16→0.30, GBP: 0.20→0.36, AUD: 0.10→0.18)
- 期間高値/安値の幅は USD=5.1 / EUR=5.8 / GBP=7.7 / AUD=3.3 円、GBP が最も値幅大きい

**M5 ローソク分布 (cutoff 後 全 4,284 本)**:

| 通貨 | 本数 | UP | DOWN | DOJI | UP率 | DOWN率 | DOJI率 |
|---|---|---|---|---|---|---|---|
| USD_JPY | 4284 | 2151 | 2043 | 90 | 50.2% | 47.7% | 2.1% |
| EUR_JPY | 4284 | 2126 | 2090 | 68 | 49.6% | 48.8% | 1.6% |
| GBP_JPY | 4284 | 2120 | 2099 | 65 | 49.5% | 49.0% | 1.5% |
| AUD_JPY | 4284 | 2108 | 2079 | 97 | 49.2% | 48.5% | 2.3% |

→ 全通貨で UP/DOWN が 50% 付近で拮抗、有意な方向偏りなし。ATR 増大は「レンジ拡大・方向感なし」型 (typical of breakout 前兆)

---

## 3. 検出イベント (raw 13 件)

| 通貨 | 時刻 (JST) | 方向 | H1 O | H1 H | H1 L | H1 C | range/ATR | dedup | トレンド判定 | M5 higher-low/lower-high |
|---|---|---|---|---|---|---|---|---|---|---|
| EUR_JPY | 2026-08-19 21:00:00 | DOWN | 184.654 | 184.667 | 183.942 | 184.540 | 4.43 | ✓ | None (除外) | No |
| EUR_JPY | 2026-08-28 23:00:00 | DOWN | 185.854 | 185.880 | 185.438 | 185.734 | 3.60 | ✓ | None (除外) | No |
| EUR_JPY | 2026-09-02 15:00:00 | DOWN | 185.080 | 185.080 | 184.555 | 184.618 | 3.65 | ✓ | DOWN | No |
| EUR_JPY | 2026-09-02 22:00:00 | DOWN | 184.786 | 184.848 | 183.695 | 184.016 | 4.36 | × | DOWN | No |
| EUR_JPY | 2026-09-04 21:00:00 | DOWN | 181.593 | 181.720 | 180.249 | 180.641 | 3.58 | × | UP | No |
| GBP_JPY | 2026-08-19 21:00:00 | DOWN | 215.688 | 215.727 | 214.854 | 215.615 | 4.31 | ✓ | DOWN | No |
| GBP_JPY | 2026-08-28 23:00:00 | UP | 216.736 | 217.072 | 216.352 | 216.964 | 4.19 | ✓ | None (除外) | No |
| GBP_JPY | 2026-09-02 22:00:00 | DOWN | 215.056 | 215.217 | 213.770 | 214.286 | 4.20 | ✓ | DOWN | No |
| GBP_JPY | 2026-09-04 21:00:00 | DOWN | 211.472 | 211.509 | 209.791 | 210.256 | 3.59 | × | UP | No |
| USD_JPY | 2026-08-19 21:00:00 | DOWN | 159.145 | 159.159 | 158.175 | 158.601 | 5.00 | ✓ | DOWN | No |
| USD_JPY | 2026-08-28 23:00:00 | UP | 159.699 | 159.949 | 159.429 | 159.875 | 4.26 | ✓ | UP | No |
| USD_JPY | 2026-09-02 22:00:00 | DOWN | 159.596 | 159.622 | 158.242 | 158.574 | 4.95 | ✓ | None (除外) | No |
| USD_JPY | 2026-09-04 21:00:00 | DOWN | 156.269 | 156.793 | 155.364 | 155.695 | 3.63 | × | None (除外) | No |

**集計**:
- raw 13 → dedup 9 (72h 窓で 4 件 drop、3 タイミングが時間的に離れている)
- トレンド通過 5 → M5 エントリー成立 14
- 3 タイミングの特徴:
  - **2026-08-19 21:00 (Wed 米国市場オープン)**: 3 通貨同期 DOWN、range/ATR = 4.31〜5.00 の強いブレイク
  - **2026-08-28 23:00 (Fri NY クローズ)**: USD/GBP は UP、EUR のみ DOWN。方向不一致で M5 エントリーは USD_JPY のみ通過判定 (しかし trend=UP/DOWN 矛盾で M5 no entry)
  - **2026-09-02 22:00 〜 2026-09-04 21:00**: USD/EUR/GBP の DOWN 集中 (9/2-9/4 急落局面)。AUD_JPY は raw 0 件で構造的沈黙

**AUD_JPY の構造的問題 (前回レポートから継続)**: cutoff 期間中 raw 検出 0 件。ATR(H1) が他 3 通貨より小さく (0.18 vs 0.30-0.36) N_BREAKOUT=3.5 閾値未達。Train 段階でも AUD は他よりサンプル少なく、CALM_RATIO や通貨拡大の検討対象だが、HARKing 防止のため spec は凍結維持

---

## 4. トレード (該当週: 14 件決済 / 0 件保有中)

| 通貨 | 方向 | エントリー | 決済 | 損益 (R) | 損益 ($) | 勝敗 | 決済理由 |
|---|---|---|---|---|---|---|---|
| EUR_JPY | DOWN | 2026-09-02 17:20:00 | 2026-09-02 18:00:00 | -1.194 | $-11.94 | LOSS | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 10:05:00 | 2026-09-03 10:45:00 | +1.992 | $+19.69 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 10:05:00 | 2026-09-03 10:45:00 | +1.994 | $+19.70 | WIN | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 12:05:00 | 2026-09-03 15:30:00 | +1.205 | $+12.38 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 12:05:00 | 2026-09-03 15:35:00 | +1.831 | $+18.81 | WIN | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 16:05:00 | 2026-09-03 16:35:00 | -1.041 | $-11.02 | LOSS | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 16:05:00 | 2026-09-03 16:35:00 | -1.040 | $-11.01 | LOSS | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 18:05:00 | 2026-09-03 18:55:00 | +1.009 | $+10.45 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 18:05:00 | 2026-09-03 18:55:00 | +1.404 | $+14.55 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 19:00:00 | 2026-09-03 20:25:00 | +0.879 | $+9.33 | WIN | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 20:10:00 | 2026-09-03 21:35:00 | +0.373 | $+3.96 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 21:05:00 | 2026-09-03 21:30:00 | +1.133 | $+12.13 | WIN | SL_INITIAL_NO_TP |
| EUR_JPY | DOWN | 2026-09-03 22:15:00 | 2026-09-04 00:05:00 | +0.619 | $+6.73 | WIN | SL_INITIAL_NO_TP |
| GBP_JPY | DOWN | 2026-09-03 23:25:00 | 2026-09-03 23:35:00 | -1.283 | $-13.95 | LOSS | SL_INITIAL_NO_TP |

**通貨別集計**:

| 通貨 | n | WIN | LOSS | 勝率 | 合計 R | 合計 $ |
|---|---|---|---|---|---|---|
| USD_JPY | 0 | 0 | 0 | - | - | - |
| EUR_JPY | 7 | 5 | 2 | 71.4% | +2.96R | $+30.25 |
| GBP_JPY | 7 | 5 | 2 | 71.4% | +4.92R | $+49.56 |
| AUD_JPY | 0 | 0 | 0 | - | - | - |

**全 14 トレードが同方向 DOWN、同時期 (9/2-9/4)** の集中爆出:
- 9/2 17:20 EUR 1 件: -1.19R (LOSS、欧州時間序盤の急落で初期 SL ヒット)
- 9/3 10:05-23:35: EUR 5 件 + GBP 6 件 = 11 件 (欧州〜NY 時間の急落連続、71% 勝率)
- 9/4 00:05: EUR 1 件: +0.62R (深夜の戻り売り)
- 全決済理由が `SL_INITIAL_NO_TP` (trail-only 設計で TP 不在、初期 SL または trail に到達して決済)

**r_gross vs r_net の乖離**: cost_r 平均 0.094R (spread 0.5pip + slippage 0.5/1.0pip + commission 0.00004) が想定通り加算。勝ったトレードでも r_net = r_gross - 0.08-0.22R

---

## 5. チャート

### 5.1 分析時点の H1 足

[USD_JPY H1 - Forward Test Week 3]

![USD_JPY H1](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_h1_analysis_USD_JPY_2026-09-07.png)

[EUR_JPY H1 - Forward Test Week 3]

![EUR_JPY H1](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_h1_analysis_EUR_JPY_2026-09-07.png)

[GBP_JPY H1 - Forward Test Week 3]

![GBP_JPY H1](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_h1_analysis_GBP_JPY_2026-09-07.png)

[AUD_JPY H1 - Forward Test Week 3]

![AUD_JPY H1](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_h1_analysis_AUD_JPY_2026-09-07.png)

凡例: 緑=UP (トレンド通過)、赤=DOWN (トレンド通過)、灰=トレンド判定不能 (除外)

### 5.2 エントリーチャート

![Entry chart](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_entry_2026-09-07.png)

該当週の M5 エントリー成立: **14 件** (前 2 週 0 件 → 集中爆出)

### 5.3 決済チャート

![Exit chart](C:\Users\Atsushi Hasebe\.minimax-agent\projects\minmax-fx-day-trading-lab\logs\charts\2026-09-07\chart_exit_2026-09-07.png)

決済済みトレード: **14 件** (n_open=0 件は週末クローズ後に強制決済予定)

---

## 6. GitHub Actions 状況 (直近 7 日間: 2026-08-31 以降)

| workflow | repo | total_runs | success | failure | 成功率 | latest_run (UTC) | latest_conclusion |
|---|---|---|---|---|---|---|---|
| FX Forward Test Cycle (Weekly, SYS-FX012 + SYS-FX026) | minmax-fx-day-trading-lab | 1 | 1 | 0 | **100%** | 2026-08-31T02:11:45Z | success |
| SYS-FX012 Forward M5 OHLCV Live Update (Hourly) | minmax-fx-day-trading-lab | 20+ | 20 | 0 | **100%** | 2026-09-06T13:58:57Z | success |
| SYS-FX022 ライブ気配値記録 | minmax-fx-day-trading-lab | 15+ | 15 | 0 | **100%** | 2026-09-06T13:37:06Z | success |
| SYS-FX012 FX Forward Test Sync (Weekly) | trading-app-v2 | 1 | 1 | 0 | **100%** | 2026-08-31T09:31:45Z | success |
| Deploy to Firebase Hosting | trading-app-v2 | 10+ | 10 | 0 | **100%** | 2026-09-06T15:27:52Z | success |

**所見**:
- **FX Forward Test Cycle**: 8/24-8/25 の failure 2 件以降、8/31 で 1 回 success して以降、月曜 09:00 JST のスケジュール実行を 9/7 も待っている状態
- **SYS-FX012 Forward M5 OHLCV Live Update (Hourly)**: 20 回連続 success (前週 100% 維持)。cycle の self-healing も機能している
- **Deploy to Firebase Hosting**: 10+ 連続 success、スマホへの配信導線に問題なし
- **本週の運用**: 9/7 01:00 JST のクロン起動が手動でローカル cycle を実行 (cycle 自体は正常終了、KPI 5/9 達成、ledger 正常 commit 対象)

---

## 7. チェックポイント進捗

| チェックポイント | 目標日 | 経過 | 進捗率 | 現 n_trades | 期待 n_trades (Train 週 4.1 件ペース) |
|---|---|---|---|---|---|
| 30日 | 2026-09-14 06:00 | 21.00 / 30 日 | **70.0%** | 14 | 17.6 |
| 60日 | 2026-10-14 06:00 | 21.00 / 60 日 | **35.0%** | 14 | 35.1 |
| 90日 | 2026-11-13 06:00 | 21.00 / 90 日 | **23.3%** | 14 | 52.7 |

**評価**:
- **21.00 日経過時点で n_trades=14 (WIN 10 / LOSS 4)、最終残高 $1079.81**
- Train 週 4.1 件ペース想定だと 21 日で 12.3 件期待 → 実 14 件で 114% 達成 (前 2 週 0 件の遅延を取り戻す形で集中爆出)
- 30 日 checkpoint (2026-09-14) まで残り 9.00 日。平常ボラなら週 4.1 件ペースで 4-5 件追加、合計 18-19 件想定
- min_n_trades ≥ 300 は 90 日 checkpoint でも困難 (現状ペース 14 件/3 週 → 90 日で 60 件程度、300 件には 5 倍必要)
- ただし ATR(H1) 増大 (+72-90%) ＋ 9/2-9/4 の急落実現で「平常ボラ環境」が戻りつつある兆候、以降の検出機会増加に期待

---

## 8. 課題・次の注目

### 観察された異常・要対応

1. **3 週目で 14 トレード集中爆出**: 9/2-9/4 の EUR/GBP 急落局面に 14 トレード (全 DOWN 方向)。0 トレード週が 2 週連続した直後の集中は「平常ボラ環境への移行期」に典型的なパターン。n=14 では統計的に脆弱 (permutation p=0.4975)
2. **ATR(H1) の大幅増大 (+72-90%)**: 4 通貨すべて平常値 (USD 0.30/EUR 0.30/GBP 0.36/AUD 0.18) に到達、夏枯れ相場終了のサイン
3. **AUD_JPY の raw 検出 0 件が継続 (3 週連続)**: 構造的に ATR が小さく N_BREAKOUT=3.5 未達。AUD_JPY を SYS-FX012 の対象から除外するか、通貨別に N_BREAKOUT/CALM_RATIO を調整する spec 改訂は HARKing 防止の観点から保留
4. **USD_JPY の M5 エントリー 0 件**: USD 検出 4 件中、トレンド通過 2 件 (8/19 DOWN, 8/28 UP) だが M5 ダウ理論での方向確認成立せず。通貨別の M5 エントリー閾値調整が必要か観察継続
5. **ローカル cron の停止 (8/31〜9/7)**: 9/7 早朝に手動同期で回復。GitHub Actions のスケジュール実行は正常だが、ローカルの manual cycle が長らく動かなかった。obs ノートに記録

### 次の週で監視すべき指標

- n_trades (累計) の伸び: 平常ボラ環境で週 4.1 件ペースが維持されるか
- ATR(H1) の安定: 9 月以降も 0.30+ が続くか、夏枯れ再突入はあるか
- 通貨分散: USD/AUD でも M5 エントリーが成立するか (現在は EUR/GBP のみ)
- permutation p 値: n=20-30 で 0.05 以下に改善するか
- 設計凍結の維持: 検出層・エントリー層・出口・コストモデルのいずれも変更なし

### 設計凍結 (HARKing 防止) の遵守確認

- 検出層: N_BREAKOUT=3.5 / CALM_RATIO=2.0 / DONCHIAN_LENGTH=20 固定 (変更なし)
- トレンド判定: SYS-FX009 ZigZag threshold_atr=2.0 固定
- エントリー層: 5 連続フィルター / ZigZag threshold_atr_m5=1.0 固定
- 出口: TP-13 / tp_levels=[] / breakeven_trigger_r=1.0 / atr_trail_multiplier_m5=0.703
- コスト: T-09 確定値固定 (SPREAD_PIPS、SLIPPAGE_PIPS_MARKET_LEG、SLIPPAGE_PIPS_STOP_TRIGGERED、COMMISSION_RATE_ROUND_TRIP)

→ **本週の分析で新規パラメータ調整なし** (HARKing 防止 OK)

### 検出ロジック整合性 (Cycle vs Spec) - 注記

`run_forward_test_cycle.py` は `detect_candidate1` (N_BREAKOUT 単独) を使用しているが、spec (`research/EXP-FX000006/00-spec.md`) は N_BREAKOUT OR (Donchian AND CALM_RATIO) の OR 合成を要求している。
- **今期検出 13 件はすべて N_BREAKOUT 経路** (range/ATR >= 3.5)
- 9/2-9/4 の急落局面では CALM_RATIO (2.0) 経路の Donchian ブレイクが追加候補になる可能性あり
- 来週以降、C 品質チームまたは strategy-architect が cycle を spec 通り (`detect_candidate3`) に修正する可能性
- 修正された場合、n_events_raw / n_events_dedup / n_trades すべて増加見込み。**HARKing 防止の観点では、現状の「N_BREAKOUT 単独」の方が「安全」だが、spec との不一致は要修正項目**

---

## 9. 付録: データ品質

### M5 データカバレッジ

- cutoff 〜 latest (21.00 日) 期待 M5 本数: 6048 本
- 実 M5 本数: 4,284 本 (各通貨共通)
- カバレッジ: 70.8% (欠落 1764 本 = 29.2%、週末クローズによる)
- 残存欠落期間: 2026-08-15 06:00 〜 2026-08-17 07:00 (2.04 日 / 588 本、各週の週末クローズによる恒久的な空白)

### ledger 整合性

- latest_bar_by_pair: 4 通貨とも 2026-09-05 05:55:00 (整合)
- n_events_raw: 13 (検算一致、cycle と同じ)
- n_events_dedup: 9 (検算一致、72h 窓で 4 件 drop)
- n_events_trendfiltered: 5 (ledger 値と一致)
- n_trades: 14 (整合、決済 14 / 保有中 0)

### 出力ファイル

- 集計 JSON v2: `logs/forward_test_cycle/weekly_summary_v2_2026-09-07.json`
- 検出イベント詳細: `logs/forward_test_cycle/weekly_events_v4_2026-09-07.json`
- チャート: `logs/charts/2026-09-07/` (6 ファイル)
- 値動き集計スクリプト: `logs/forward_test_cycle/weekly_summary_2026-09-07.py`
- チャート生成スクリプト: `logs/forward_test_cycle/weekly_charts_2026-09-07.py`

---

**次回クロン起動予定**: 月曜 09:00 JST (sysfx012-fx-forward-cycle)
**次週レポート予定日**: 2026-09-14 (cutoff + 30 日 checkpoint)

---

> このプロンプト自体が週次レポート生成の雛形。新規週で prompt 微調整があれば `obs/minmax_fx_day_trading_lab/70対応待ち/` 配下に記録。