# 検証仕様書（Spec） - EXP-FX000008

> 担当: A 設計チーム（strategy-architect）
> 起票: 2026-08-23
> 前提: `00-prescreen.md`（GO）
> 原則: 評価基準は**試算前**に数値で確定する（HARKing防止）

## 対応 OBS 番号
（E進行チームが採番予定）

## 紐づく戦略系統 ID
SYS-FX014

## 仮説（1行）

**H1較正済みのN_BREAKOUT=3.5をM30へ無再導出で転用したことが、非公式診断（`explore_m30_trend_detection_trainonly.py`）でのTrain KPI大幅悪化(7/9→1/9)の主因であり、M30のレンジ/ATR分布から同一方法論で再導出したN_BREAKOUT値を使えば、H1版に近い成績が得られるのではないか。**

## 検証範囲

- **対象通貨**: USD_JPY・EUR_JPY・GBP_JPY・AUD_JPY（SYS-FX012凍結設計と同一の4通貨、JPYクロス）
- **変更する変数**: M30リサンプルバー上でのN_BREAKOUT閾値のみ
- **完全凍結（変更しない）**: H1ダウ理論判定不能除外フィルターのzigzag閾値(=2.0、M30バーへそのまま適用。**この値自体はM30向けに再導出しない**、既知の限界として明記する)、M5エントリー層(`stop_buffer_atr_m5=0.703`・`zigzag_threshold_atr_m5=1.0`)、出口設計（トレール専業）、コストモデル、検定方式
- **Train期間のみ**: 2023-11-01〜2025-03-31。Validationへ進むかどうかは下記「評価・選定基準」に従う

## N_BREAKOUT再導出手順（事前登録、結果を見る前に固定）

H1版の導出方法論（`scripts/analyze_vol_breakout_frequency.py`）と完全に同一のロジックをM30リサンプルバーに適用する:

1. 対象4通貨・Train期間のM5データをM30へリサンプル（`to_m30()`、`scripts/derive_vol_breakout_entry_params.py`に実装済み）
2. 各M30バーで `レンジ(高値-安値) ÷ ATR(M30,14,Wilder)` を計算
3. 候補 `N_CANDIDATES = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]`（H1版と同一の候補集合）それぞれについて、4通貨プールでの検出イベント数と「週あたり発生件数」を実測
4. **選定基準**: プール発生頻度が**週1回程度**（1.0 events/week に最も近い値）となる候補Nを採用する（H1版の選定基準「発生頻度が週1回程度のオーダーになる水準」をそのまま踏襲。機械的な単一パーセンタイル固定ではなく、頻度とのバランスで選ぶという方針も同じ）

**H1版との既知の差異**: H1版の頻度実測は5通貨プール（JPYクロス4通貨+EUR_USD）で行われたが、本EXPはSYS-FX012の凍結設計と対象を揃えるため4通貨プールで実施する。この差はNの絶対値に軽微な影響を与えうるが、選定ロジック自体は同一である。

## 評価・選定基準（事前登録、結果を見る前に固定）

再導出したN_BREAKOUTを用いてM30版パイプライン（`explore_m30_trend_detection_trainonly.py`ベース、`detect_candidate1`・`h1_dow_trend_direction`・`simulate_dow_theory_trend`・`select_non_overlapping_breakout_events`を変更せず流用、検出バーのみH1→M30）でTrain評価する。

比較対象（H1版candidate①、`research/method-notes/candidate3_cost_ratio_filter_trainonly_backtest.json`の`candidate1_reference`）:

| 指標 | H1版candidate①(Train) |
|---|---|
| 必須KPI達成数 | 7/9 |
| 月次シャープ | 2.397 |
| 最大DD(月間) | 8.69% |
| Profit Factor | 1.759 |
| ペイオフレシオ | 1.078 |
| permutation_p | 0.031 |
| n_trades | 300 |

**判定ルール**:
- **明確に上回る/同等**（必須KPI達成数がH1版の7/9以上、かつpermutation_pが有意(<0.05)を維持）→ Validationへ進める判断材料とし、司令塔へ具申する
- **明確に下回る**（必須KPI達成数が7/9未満、またはpermutation_pが非有意）→ Train単独で不採用と判定し、Validationへは進めない。非公式診断（無再導出N=3.5、KPI 1/9）と比較し、再導出による改善幅を正直に記録した上でEXPをクローズする
- 中間的な結果（判断が割れる場合）は司令塔に判断を仰ぐ

## 改善ループとの関係

本EXPはSYS-FX012の改善ループ（上限5回、2026-08-22に消化済み）とは**別枠の新規検討**であり、その試行回数には計上しない。本EXP自体は単発のTrain評価（N再導出→1回の評価）として実施し、Train結果が不採用の場合は多段階の改善ループを新設せず、そのままクローズする。

## 実行

- `scripts/analyze_vol_breakout_frequency_m30.py`（新規、H1版のM30移植）→ `research/method-notes/vol_breakout_frequency_m30.json`
- `scripts/backtest_m30_rederived_n_trainonly.py`（新規）→ `research/method-notes/m30_rederived_n_trainonly_backtest.json`

## 結果（2026-08-23実施）

### N_BREAKOUT再導出（頻度ベース、4通貨プール）

| N候補 | 総イベント数 | 週あたり(4通貨) |
|---|---|---|
| 2.0 | 3432 | 16.643 |
| 2.5 | 1403 | 6.804 |
| 3.0 | 654 | 3.171 |
| 3.5 | 342 | 1.658 |
| **4.0** | **197** | **0.955** |
| 4.5 | 124 | 0.601 |
| 5.0 | 77 | 0.373 |

選定基準（週1回程度に最も近い）により **N_BREAKOUT=4.0** を採用（H1版のN=3.5・pooled 1.013 events/weekと同じ選定ロジックによる自然な帰結）。詳細: `research/method-notes/vol_breakout_frequency_m30.json`

### Train評価（再導出N_BREAKOUT=4.0、M30トレンド判定層）

| 指標 | H1版candidate①(基準) | 非公式診断(N=3.5無再導出) | 本EXP(N=4.0再導出) |
|---|---|---|---|
| 必須KPI達成数 | 7/9 | 1/9 | **1/9** |
| 月次シャープ | 2.397 | -0.375 | -0.093 |
| 最大DD(月間) | 8.69% | 45.95% | 21.68% |
| Profit Factor | 1.759 | 0.968 | 1.039 |
| ペイオフレシオ | 1.078 | 0.883 | 0.904 |
| permutation_p | 0.031 | 0.5694 | 0.4296 |
| n_trades | 300 | 591 | 331 |
| 勝率 | 0.62 | 0.523 | 0.535 |

**結論・不採用**: 頻度ベースの正式なN再導出（N=3.5→4.0）により、非公式診断からは全指標が改善した（DD 45.95%→21.68%、Sharpe -0.375→-0.093、PF 0.968→1.039、perm_p 0.5694→0.4296）。しかし事前登録した選定基準（必須KPI達成数がH1版の7/9以上、かつpermutation_pが有意）を満たさず、依然としてH1版candidate①に遠く及ばない（KPI 1/9のまま、perm_pは非有意0.4296）。**Validationへは進めず、Train単独で不採用と判定してEXPをクローズする。**

**解釈**: N_BREAKOUTの再導出だけでは、M30トレンド判定層の成績をH1水準まで戻せなかった。残る要因として、(a) H1ダウ理論判定不能除外フィルターのzigzag閾値(=2.0)自体もH1較正値のままM30へ転用しており、フィルターの選別性が歪んでいる可能性、(b) M30はH1よりノイズが多い時間軸であり、頻度を揃えても個々のブレイクイベントの「その後のトレンド継続の信頼性」自体がH1より低い可能性、の2点が考えられる。前者(a)は本EXPのスコープ外（zigzag閾値の再導出は別途の検討が必要）、後者(b)はより本質的な限界であり、M30トレンド判定層の追加検討には慎重な費用対効果の見極めが必要と考える。詳細: `research/method-notes/m30_rederived_n_trainonly_backtest.json`
