# 検証仕様書（Spec） - EXP-FX000007

> 担当: A 設計チーム（strategy-architect）
> 起票: 2026-08-22
> 前提: `00-prescreen.md`（GO）
> 原則: 評価基準は**試算前**に数値で確定する（HARKing防止）

## 対応 OBS 番号
（E進行チームが採番予定）

## 紐づく戦略系統 ID
SYS-FX013

## 仮説（1行）

**SYS-FX012で確立した検出層（H1のN_BREAKOUTブレイク検出+H1ダウ理論トレンド判定可能性フィルター）とM5ダウ理論連続追跡は、JPYクロス通貨に固有の構造ではなく、他の主要非JPY通貨ペア（GBP/USD・AUD/USD・NZD/USD）でも同様のエッジを示すのではないか。**

## 検証範囲（既存設計を完全に凍結して流用）

- **対象通貨**: GBP_USD・AUD_USD・NZD_USD（EUR_USDは検証済み、除外）
- **戦略設計**: SYS-FX012の最良候補（`research/EXP-FX000006/00-spec.md`改善ループ第2試行）を一切変更せず流用
  - 検出層: `N_BREAKOUT=3.5`のH1ブレイク検出 + H1ダウ理論トレンド判定不能イベント除外フィルター（`zigzag threshold_atr_h1=2.0`）
  - エントリー層: M5ダウ理論連続追跡（`stop_buffer_atr_m5=0.703`、`zigzag_threshold_atr_m5=1.0`、いずれもJPYクロス4通貨のTrainデータからの導出値をそのまま流用、再導出しない）
  - 出口: トレール専業（`tp_levels=[]`・`breakeven_trigger_r=1.0`・`atr_trail_multiplier_m5=stop_buffer_atr_m5×1.0`）
  - 検定: `permutation_test_block()`・`compute_n_trades_effective()`・`compute_k3m_scale_invariant()`
- **Train/Validation/Test期間**: SYS-FX011/012と同一（Train: 2023-11-01〜2025-03-31 / Validation: 2025-04-01〜2025-11-30 / Test: 2025-12-01〜2026-08-15、Testは本specでは未取得・対象外）
- **保有期間**: 週末持ち越し禁止（本PJ共通ルール）

## コストモデル（新規通貨、未確認の暫定値）

GBP_USD・AUD_USD・NZD_USDはGMOコイン公式の手数料ページ（`https://coin.z.com/jp/corp/guide/fees/`、2026-08-22 WebFetch確認）に**原則固定スプレッドの記載がなく**、既存5通貨（USD/JPY・EUR/JPY・GBP/JPY・AUD/JPY・EUR/USD）のみが掲載されている。このため以下は一般的な業界水準を参考にした**保守的な暫定値**であり、正式な意思決定には未確認である旨を明示する:

| 通貨ペア | 暫定スプレッド(pips) | 根拠 |
|---|---|---|
| GBP_USD | 1.0 | メジャー通貨ペアの一般的な実勢水準を参考にした保守的な仮定（GMOコインの公式確認は取れていない） |
| AUD_USD | 0.8 | 同上 |
| NZD_USD | 1.5 | 同上（相対的に流動性が低い通貨のため他2ペアより保守的に設定） |

スリッページ・手数料率はSYS-FX011/012のT-09確定版（`SLIPPAGE_PIPS_MARKET_LEG=0.5`・`SLIPPAGE_PIPS_STOP_TRIGGERED=1.0`・`COMMISSION_RATE_ROUND_TRIP=0.00004`）をそのまま流用する。

## KPI閾値（SYS-FX011/012と完全に同一枠組みを流用、結果を見る前に固定）

| KPI | 閾値 | 必須/参考 |
|---|---|---|
| K1m 月次シャープ | ≥ 0.4 | 必須 |
| K1m PF | ≥ 1.2 | 必須 |
| K1m 月次期待値 | > 0円 | 必須 |
| K2m 月間DD | ≤ 10% | 必須 |
| K2m 年間DD | ≤ 20% | 必須 |
| K3m 最大連続損失 | i.i.d.帰無分布パーセンタイル判定 | 参考 |
| K4m ペイオフレシオ | ≥ 1.5 | 必須 |
| K5m スプレッドコスト倍率 | ≥ 3倍 | 必須 |
| min_n_trades（実効値） | ≥ 300 | 必須 |
| permutation_p_value | < 0.05 | 必須 |

## 検証プロトコル（第1段階: 個別通貨でのエッジ確認）

**まず各通貨ペアを単体でTrain評価し、mean_r_net・勝率でエッジの有無を確認する（EUR_USD検証と同じ形式）。** JPYクロス4通貨のプールへ追加するか、非JPY単体でのポートフォリオ化を検討するかは、この結果を見てから判断する。個別通貨のTrain単体評価で明確なマイナスが出た通貨は、以降の統合検証（プール化・Validation確認）の対象から除外する。

- Validationは、Train単体で有望と判断された通貨についてのみ、後続で参照する（HARKing防止）
- Testは参照しない（凍結方針をEXP起票時から適用）

## 改善ループの停止条件（`PJ000004`「Q10」準拠）

改善ループ上限5回。本EXPは「既存の凍結済み設計を新市場でテストする」検証であり、パラメータ・設計変更を伴う「試行」は基本的に発生しない想定（対象通貨の追加検討を除く）。

## 変更履歴
- 2026-08-22: 初版作成。GBP_USD・AUD_USD・NZD_USDを対象に、SYS-FX012の凍結済み設計をそのまま適用する検証を事前登録
