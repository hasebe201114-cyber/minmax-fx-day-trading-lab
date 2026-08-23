# 検証仕様書（Spec） - EXP-FX000010

> 担当: A 設計チーム（strategy-architect）
> 起票: 2026-08-23
> 前提: `00-prescreen.md`（GO）
> 原則: 評価基準は**試算前**に数値で確定する（HARKing防止）

## 対応 OBS 番号
（E進行チームが採番予定）

## 紐づく戦略系統 ID
SYS-FX016

## 仮説（1行）

**SYS-FX012の凍結済み設計（H1のN_BREAKOUTブレイク検出+H1ダウ理論トレンド判定可能性フィルター+M5ダウ理論連続追跡）は、既存4通貨(USD/EUR/GBP/AUD_JPY)以外のJPYクロス(NZD/CAD/CHF_JPY)でも同様のエッジを示すのではないか。**

## 検証範囲（既存設計を完全に凍結して流用）

- **対象通貨**: NZD_JPY・CAD_JPY・CHF_JPY（既存4通貨に追加、パラメータ再導出なし）
- **戦略設計**: SYS-FX012の最良候補を一切変更せず流用（N_BREAKOUT=3.5、zigzag閾値=2.0、stop_buffer_atr_m5=0.703、atr_trail_multiplier_m5=0.703、breakeven_trigger_r=1.0、tp_levels=[]）
- **Train期間のみ**: 2023-11-01〜2025-03-31。個別通貨のTrain評価で妥当性を確認した上で、7通貨版としてのValidation確認まで進めるかを判定する

## 評価・選定基準（事前登録、結果を見る前に固定）

1. **個別通貨評価**: NZD_JPY・CAD_JPY・CHF_JPYそれぞれ単体でTrain評価し、mean_r_netが正であることを確認する（SYS-FX013と同じ判定方法）。負の通貨は7通貨統合版から除外する
2. **7通貨統合版評価（個別評価で正だった通貨のみ組み込み）**: 既存4通貨+新規通貨（正だったもののみ）をプールしてTrain評価し、以下と比較する
   - 実効n（既存4通貨版Train n=300）
   - 必須KPI達成数（既存4通貨版Train 7/9）
   - permutation_p（既存4通貨版Train 0.031）
3. **判定ルール**:
   - 実効nが明確に増加し、必須KPI達成数・permutation_pが既存4通貨版と同等以上 → Validation確認へ進める
   - 実効nは増加するが質的指標(KPI達成数・PF・DD)が明確に悪化 → トレードオフを正直に記録し司令塔に判断を仰ぐ
   - 個別通貨がすべて負、または統合しても実効nが有意に改善しない → 不採用と判定してクローズ

## 実行

- `scripts/backtest_sysfx016_new_jpy_pairs_trainonly.py`（新規）→ `research/method-notes/sysfx016_new_jpy_pairs_trainonly_backtest.json`
- `scripts/backtest_sysfx016_pooled_6pairs_trainonly.py`（新規）→ `research/method-notes/sysfx016_pooled_6pairs_trainonly_backtest.json`

## 結果（2026-08-23実施）

### 個別通貨評価

| 通貨 | トレード数 | mean_r_net | 勝率 | 判定 |
|---|---|---|---|---|
| NZD_JPY | 65 | -0.2739 | 43.1% | **エッジなし（除外）** |
| CAD_JPY | 120 | +0.0383 | 55.0% | エッジあり（薄い） |
| CHF_JPY | 79 | +0.2877 | 57.0% | エッジあり |

事前登録ルールに従い、NZD_JPYを除外しCAD_JPY・CHF_JPYを既存4通貨に統合した6通貨プールで評価。

### 6通貨プール評価（4通貨基準との比較）

| 指標 | 4通貨(基準) | 6通貨 |
|---|---|---|
| 実効n | 300 | **427**(明確に増加) |
| 必須KPI達成数 | 7/9 | **5/9**(悪化) |
| 月次シャープ | 2.397 | 2.094(悪化) |
| Profit Factor | 1.759 | 1.564(悪化) |
| ペイオフレシオ | 1.078 | 1.034(悪化) |
| 最大DD | 8.69% | 13.75%(悪化) |
| スプレッドコスト倍率 | 2.35 | 1.99(悪化) |
| permutation_p | 0.031(有意) | **0.0649**(非有意化) |

**結果**: 実効nは明確に増加(300→427、目標300を突破)したが、質的指標(Sharpe・PF・ペイオフ・DD・スプレッド倍率)がすべて悪化し、KPI達成数は7/9→5/9に悪化。permutation_pも有意→非有意に転じた。事前登録した判定ルール「実効nは増加するが質的指標が明確に悪化」に該当し、**単独では結論を出さず司令塔に判断を仰ぐ**。
