# EXP-FX000012 再実行記録（先読み修正後）

> 実施: 2026-09-30（trading-app-v2 側セッションで司令塔の指示「進めて下さい」を受けて実行）
> 位置づけ: 生データの再取得のみ。採否の宣告は C 品質チーム／司令塔が行う
> 関連: `00-spec.md`（事前登録の選定ルール・旧結果）、OBS000009 不具合1（先読み、2026-08-28 修正）

## 変更履歴
| 日付 | 内容 |
|---|---|
| 2026-09-30 | 新規作成。先読み修正後の条件でスイープを再実行した結果を記録 |

## 再実行した理由

`00-spec.md` の結果表（2026-08-23）は、先読み修正（`bar_close_anchored=True`、2026-08-28）より前に生成された
`sysfx018_breakeven_sweep_trainonly_backtest.json` の数値だった。2026-08-29 にスクリプトへ修正は入ったが、
このスイープ自体は再実行されておらず、SYS-FX018 の「重要な近接候補」という記載は修正前の数値に依存していた。

## 実行条件

- スクリプト: `scripts/backtest_sysfx018_breakeven_sweep_trainonly.py`（origin/main e782293 時点、変更なし）
- データ: `data/curated/ds-1.json` を `data/raw/ds-1/*.csv` から `fetch_ds1_ohlcv.aggregate_to_json()` で再生成
- 環境: Python 3.12 / pandas 3.0.6 / numpy 2.2.6
- 出力: `research/method-notes/sysfx018_breakeven_sweep_trainonly_backtest.json`（上書き。修正前の版は git 履歴に残る）

## 再現性チェック

現行値 1.0 の結果が、先読み修正後の SYS-FX012 公式値（`vol_breakout_trendfilter_candidate1_validation_backtest.json` の train_reference）と一致した。

| 指標 | 公式値 | 今回 |
|---|---|---|
| トレード数（実効n） | 308 | 308 |
| ペイオフレシオ | 1.108 | 1.108 |
| permutation_p | 0.048 | 0.048 |
| 必須KPI | 7/9 | 7/9 |

## 結果（Train 2023-11-01〜2025-03-31、17か月）

| breakeven_trigger_r | トレード数 | 勝率 | 期待値/件 | 月平均R | ペイオフ | 月次Sharpe | 最大DD | 平均保有 | 必須KPI | 実効n | permutation_p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 357 | 51.3% | 0.139 | 2.93 | 1.355 | 1.84 | 10.86% | 0.81h | 5/9 | 357 | 0.1768 |
| 0.75 | 327 | 59.0% | 0.232 | 4.47 | 1.083 | 2.42 | 9.09% | 0.97h | 6/9 | 327 | 0.0959 |
| **1.0（現行）** | 308 | 60.7% | 0.322 | 5.84 | 1.108 | 2.66 | 8.78% | 1.16h | **7/9** | 308 | **0.048** |
| 1.5 | 264 | 57.6% | 0.398 | 6.18 | 1.261 | 2.61 | 9.73% | 1.61h | 6/9 | 264 | 0.044 |
| 2.0 | 232 | 50.9% | 0.367 | 5.01 | 1.511 | 2.31 | 10.10% | 2.07h | 5/9 | 232 | 0.0679 |

月平均R = r_net 合計 ÷ 17。平均保有はトレード明細の exit_time − entry_time から集計。

各候補の必須KPI未達項目:

- 0.5: max_dd_monthly_pct, max_consecutive_losses, payoff_ratio, spread_cost_multiplier, permutation_p_value
- 0.75: payoff_ratio, spread_cost_multiplier, permutation_p_value
- 1.0: payoff_ratio, spread_cost_multiplier
- 1.5: payoff_ratio, spread_cost_multiplier, min_n_trades_effective
- 2.0: max_dd_monthly_pct, spread_cost_multiplier, min_n_trades_effective, permutation_p_value

## 修正前との差（2.0 のみ）

| 指標 | 修正前（2026-08-23） | 修正後（2026-09-30） |
|---|---|---|
| トレード数 | 219 | 232 |
| ペイオフレシオ | 1.549 | 1.511 |
| 必須KPI | 7/9 | 5/9 |
| permutation_p | 0.044（有意） | 0.0679（非有意） |
| 月次最大DD基準 | 達成 | 未達 |

## 事前登録ルールの機械的適用

選定ルール（必須KPI≥7/9 かつ 実効n≥300）を満たすのは修正前と同じく 1.0 のみで、選定結果は「改善なし」。
`00-spec.md` と `SYSTEMS.md` にある「2.0 は KPI 7/9・有意性を維持した近接候補」という前提は、修正後の数値では成り立たない。

## C 品質チーム・司令塔への申し送り

- SYS-FX018 のステータス（司令塔判断待ち）をどう閉じるかの宣告は未実施。
- 1.5 は月平均R・Sharpe が現行を上回るが、5候補から結果を見て選ぶことになるため、事前登録外の採用根拠にはならない。
- 同じ 2.0 を組み込んだ SYS-FX019（不採用確定済み）、SYS-FX025 改善ループ第1試行（SYS-FX018版への差替、P0 未達）の結論には、この再実行は影響しない（いずれも不採用方向）。
