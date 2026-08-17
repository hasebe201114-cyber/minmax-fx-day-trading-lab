# 検証仕様書（Spec） - EXP-FX000003

> 担当: 設計チーム（strategy-architect、司令塔依頼によりチャットセッションで代行起票）
> 起票: 2026-08-17
> バージョン: v1
> 原則: **評価基準は「試算前」に数値で確定**する（HARKing 防止）。結果を見てから基準を曲げることを禁止。
> **SYS-FX007/008から得た教訓を最初から適用する**: (1) パラメータは教科書慣例値ではなくTrainデータの実測分布から導出、(2) 判定エンジン・permutation test・停止条件は起票時点から接続、(3) フェーズゲート（`research/ACTIVE.md`）をバックテスト着手前に必ずクリアする

## 対応 OBS 番号
（未起票。必要になった時点でE進行チームが採番）

## 紐づく戦略系統 ID
SYS-FX009（`research/SYSTEMS.md` 参照。当初構想から全面再設計、v2 = 上位足トレンド+ダブルトップ/ボトム）

## 仮説（1 行）

`research/EXP-FX000003/00-prescreen.md` と同一。上位足（LT=D1）のトレンド方向に沿って、中位足（MT=H4）でダブルトップ/ダブルボトムパターンがネックライン割れで確定した場合にエントリーし、リスク幅と同じ値幅（1:1）に到達したらブレイクイーブン+ATRトレーリングへ切り替える戦略は、月次シャープ≥0.4・月間DD≤10%・PF≥1.2を達成できるのではないか。

## 採用 / 不採用の数値基準（試算前に確定）

CLAUDE.md §KPIの暫定定義（K1m〜K7m）をそのまま適用する。SYS-FX007/008と同一基準を使うことで、3戦略のポートフォリオ比較が可能になる。

### 採用条件（全て満たすこと）
- K1m: 月次Profit Factor > 1.2、月次シャープレシオ ≥ 0.4、月次期待値 > 0
- K2m: 最大DD（月間）≤ 証拠金の10%、最大DD（年間）≤ 証拠金の20%
- K3m: 最大連続損失 ≤ 5トレード
- K4m: ペイオフレシオ ≥ 1.5
- K5m: 1トレードあたり期待値 > スプレッド往復コスト × 3
- K6m: バックテストとフォワードテストのKPI乖離率 ≤ 30%
- K7m: **対象外**（SYS-FX009はLTフィルターにより常に片側方向のみ有効となる設計で、両建てを行わない。SYS-FX008と同じ扱い）
- **min_n_trades ≥ 66**（プール、SYS-FX007/008のTVT運用で使われてきた閾値を踏襲）
- **permutation_p_value < 0.05**（sign-flip permutation test、1000回）

### 不採用条件（1つでも該当）
- 上記のいずれかを満たさない
- Trainの時点でpooled n_trades < 66（サンプル不足で判定不能。この場合は不採用として記録する）

## 採用 / 確認プロトコル（過学習検証付き、SYS-FX007/008と同一分割を使用）

- Train: 2023-11-01 〜 2025-03-31（パラメータ導出・初期評価）
- Validation: 2025-04-01 〜 2025-11-30（1回のみ評価）
- Test: 2025-12-01 〜 2026-08-15（1回のみ評価）
- 各期間を再学習しない。Validation/Testで基準を満たさなければ不採用

## 改善ループの停止条件（起票時点で確定）

- 初期プリセットで選定基準を満たさなかった場合、追加で試行できるプリセット数は**最大3件**（Trainのみでの絞り込み）
- 3件消費後もTrain基準未達なら、SYS-FX009（現行構造）はパラメータ調整では改善不可能としてREJECT確定する
- パラメータの再設計は必ずTrainデータの実測分布からの導出に基づくこと。結果を見てから当てずっぽうで動かす「試行錯誤」は禁止

## 検証範囲

- 通貨ペア: USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY, EUR/USD（SYS-FX007/008と同一、OBS000002案A）
- 時間軸構造:
  - **LT（D1）**: SMA短期/長期のクロスで方向判定（UP/DOWN/NONE、トレンド強度フィルターなし。パターン確認自体がエントリー精度のフィルターとして機能する設計）
  - **MT（H4）**: ZigZag（ATR比例閾値2.0）の直近3転換点からダブルトップ/ダブルボトムを検出。ダブルトップはLT=DOWNの時のみ、ダブルボトムはLT=UPの時のみ有効（トレンド継続セットアップ、反転天井/底取りではない）
  - **エントリー**: ネックラインブレイク時に成行（ST=M5粒度で判定）
  - **ストップ**: パターン無効化水準（ダブルトップなら2つの山の高い方、ダブルボトムなら2つの谷の低い方）+ ATRバッファ
  - **利確**: 初期リスク幅（entry - stop）と同じ値幅（1:1）に到達したら、ストップをブレイクイーブンへ移動し、以降はATRトレーリングストップに切り替える
  - **決済**: 上記トレーリングストップ、または週末強制クローズ。SYS-FX008と異なりLT反転での手仕舞いは行わない（パターン自体が明確な無効化水準を持つため）
- 両建て: 対象外（LTフィルターにより常に片側方向のみ）
- ホールド期間: 数日〜数週間、週末持ち越し禁止（土曜06:00 JSTまでに全決済、CLAUDE.md準拠）
- コスト前提: `SPREAD_PIPS`辞書（SYS-FX007/008から流用）、API手数料0.002%、スリッページ0.5pip固定、スワップは`data/curated/ds-7.json`の概算値を接続

## パラメータ空間（v1、2026-08-17 導出完了）

`scripts/derive_double_pattern_params.py` で以下のルールを結果を見る前に確定し、Trainデータから導出した（`research/EXP-FX000003/10-result/double_pattern_params.json`）:

```
zigzag_threshold_atr        = 2.0                                  # OBS000006既存手法を流用（再導出なし）
pattern_tolerance_atr       = pooled(|P(t)-P(t+2)|/ATR_neckline) の p25   # 交互ZigZag3点組、全ペア pooled
stop_buffer_atr             = pooled((H4のhigh-low)/ATR) の p25    # 通常バーの値幅ノイズでの早期損切りを避けるため
max_bars_since_second_pivot = round_to_standard(
                                 pooled(tolerance内で一致した3点組の
                                 ネックライン割れまでの遅延) の p90,
                                 候補=[10,15,20,30,40]
                               )
atr_trail_multiplier        = pooled(LT方向一致シグナルの、ブレイク後30本4hでのMFE/ATR) の中央値
lt_sma_short, lt_sma_long   = 10, 20   # EXP-FX000002 (SYS-FX008) で導出済み・全ペア収束、再利用
```

| 項目 | 値 | 根拠 |
|---|---|---|
| zigzag_threshold_atr | 2.0 | OBS000006既存手法の流用 |
| pattern_tolerance_atr | 0.587 | pooled n=1748交互3点組のΔP/ATR分布、p25 |
| stop_buffer_atr | 0.636 | pooled n=11235 H4バーの値幅/ATR分布、p25 |
| max_bars_since_second_pivot | 30 | pooled n=354ブレイク済み一致3点組の遅延分布、p90=28.0本→候補中最近接 |
| atr_trail_multiplier | 2.44 | pooled n=171 LT一致シグナルの30本4hフォロースルー(MFE/ATR)分布、中央値 |
| lt_sma_short / lt_sma_long | 10 / 20 | EXP-FX000002導出済み値の再利用（全5ペアで同一値に収束済み） |

**初期の暗黙デフォルト値（`pattern_detection.DoublePatternConfig`のダミー値: tolerance=0.5, buffer=0.1, staleness=20）との比較で、特にstop_buffer_atrが実測値(0.636)よりデフォルト(0.1)がかなり小さいことが判明した。** デフォルトのまま検証していれば、通常のH4バーの値幅ノイズだけでパターン無効化水準の外側を頻繁に突かれ、意図しない早期損切りが多発していた可能性が高い。データ駆動導出のHARKing防止プロセスがこのリスクを事前に検出した一例として記録する。

この値はTrain/Validation/Testいずれの結果を見た後も変更しない。

## 使用スクリプト / 再現方法

- パラメータ導出: `scripts/derive_double_pattern_params.py`
- 戦略ロジック: `src/minmax_fx_dt/strategy/pattern_detection.py`（パターン検出）、`src/minmax_fx_dt/strategy/double_pattern_strategy.py`（LT統合・エントリー評価）
- バックテスト実行: `src/minmax_fx_dt/backtest/double_pattern_runner.py`
- Train/Validation/Test実行: `scripts/run_train_val_test_fx009.py`（予定、SYS-FX008の`run_train_val_test_fx008.py`と同一構造）
- 判定エンジン: `decision/criteria.py`（`KPI_THRESHOLDS["SYS-FX009"]`として登録済み）・`permutation.py`をそのまま再利用

## DATA チケット（SYS-FX007/008で整備済みのものを流用）

- DS-1（M5 OHLCV、5通貨、2023-11〜2026-08）
- DS-7（スワップポイント概算）
- フェーズゲート1（データ基盤）・フェーズゲート2（パラメータ導出）は上記の通りクリア済み

## 変更履歴
- v1 (2026-08-17): 初版。パラメータ導出完了。
