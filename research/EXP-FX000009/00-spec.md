# 検証仕様書（Spec） - EXP-FX000009

> 担当: A 設計チーム（strategy-architect）
> 起票: 2026-08-23
> 前提: `00-prescreen.md`（GO）
> 原則: 評価基準は**試算前**に数値で確定する（HARKing防止）

## 対応 OBS 番号
（E進行チームが採番予定）

## 紐づく戦略系統 ID
SYS-FX015

## 仮説（1行）

**M5較正済みのstop_buffer_atr_m5=0.703をM3へ無再導出で転用した非公式診断（Train KPI 7/9→6/9、permutation_p 0.031→0.0699で非有意化）に対し、M3のレンジ/ATR分布から同一方法論(p25パーセンタイル)で再導出したstop_buffer_atr_m3を使えば、H1版に一段と近い、あるいはそれを上回る成績が得られるのではないか。**

## 検証範囲

- **対象通貨**: USD_JPY・EUR_JPY・GBP_JPY・AUD_JPY（SYS-FX012凍結設計と同一の4通貨、JPYクロス）
- **変更する変数**: `stop_buffer_atr_m3`（p25パーセンタイル再導出）、およびこれに機械的連動する`atr_trail_multiplier_m3`
- **完全凍結（変更しない）**: トレンド判定層（H1、N_BREAKOUT=3.5・zigzag閾値=2.0・判定不能除外フィルター）、エントリー層の`zigzag_threshold_atr_m3`(=1.0、既知の限界としてprescreenに明記済み、再導出の方法論が存在しないため据え置く)、出口設計（トレール専業、breakeven_trigger_r=1.0）、コストモデル、検定方式
- **Train期間のみ**: 2023-11-01〜2025-03-31。Validationへ進むかどうかは下記「評価・選定基準」に従う

## stop_buffer_atr_m3再導出手順（事前登録、結果を見る前に固定）

M5版の導出方法論（`scripts/backtest_vol_breakout_dow_theory.py`、`stop_buffer_atr_m5 = round(percentile(pooled_bar_range_atr_m5, 25), 3)`）と完全に同一のロジックをM3リサンプルバーに適用する:

1. 対象4通貨・Train期間のM3データ（`data/curated/ds-1-m3-train-4pairs.json`、EXP-FX000008の非公式診断向けに既に取得済み）を使用
2. 各M3バーで `レンジ(高値-安値) ÷ ATR(M3,14,Wilder)` を計算
3. 4通貨プールでこの比率分布のp25パーセンタイルを`stop_buffer_atr_m3`として採用（機械的な計算、選定の余地なし）
4. `atr_trail_multiplier_m3 = stop_buffer_atr_m3 × 1.0`（T-02で確立済みの公式をそのまま適用、新たな選択の余地なし）

## 評価・選定基準（事前登録、結果を見る前に固定）

再導出したパラメータを用いてM3版パイプライン（`explore_m3_entry_trainonly.py`ベース、検出層・トレンド判定フィルター・M5→M3差し替えの構造を変更せず、`stop_buffer_atr_m5`/`atr_trail_multiplier_m5`の値だけ再導出値に差し替え）でTrain評価する。

比較対象1（H1版candidate①、`research/method-notes/candidate3_cost_ratio_filter_trainonly_backtest.json`の`candidate1_reference`）:

| 指標 | H1版candidate①(Train) |
|---|---|
| 必須KPI達成数 | 7/9 |
| 月次シャープ | 2.397 |
| 最大DD(月間) | 8.69% |
| ペイオフレシオ | 1.078 |
| permutation_p | 0.031 |
| n_trades | 300 |

比較対象2（非公式診断、無再導出M3、`research/method-notes/explore_m3_entry_trainonly.json`）:

| 指標 | 非公式診断(無再導出) |
|---|---|
| 必須KPI達成数 | 6/9 |
| 月次シャープ | 3.121 |
| 最大DD(月間) | 8.09% |
| ペイオフレシオ | 1.194 |
| permutation_p | 0.0699 |
| n_trades | 388 |

**判定ルール**:
- **H1版を明確に上回る/同等**（必須KPI達成数がH1版の7/9以上、かつpermutation_pが有意(<0.05)）→ Validationへ進める判断材料とし、司令塔へ具申する
- **非公式診断は上回るがH1版には届かない**（KPI達成数が6/9のまま、またはpermutation_pが依然非有意だが非公式診断より改善）→ 「再導出による方向性の改善は確認できたが採用水準には届かない」と正直に記録し、Train単独で不採用と判定してVaildationへは進めない
- **非公式診断からも悪化**→ 再導出手順自体の妥当性を疑い、原因を分析した上で不採用と判定する
- 中間的な結果（判断が割れる場合）は司令塔に判断を仰ぐ

## 改善ループとの関係

本EXPはSYS-FX012の改善ループ（上限5回、2026-08-22に消化済み）とは**別枠の新規検討**であり、その試行回数には計上しない。単発のTrain評価（再導出→1回の評価）として実施し、不採用の場合は多段階の改善ループを新設せず、そのままクローズする。

## 実行

- `scripts/derive_m3_entry_params_trainonly.py`（新規）→ `research/method-notes/m3_entry_params_trainonly.json`
- `scripts/backtest_m3_rederived_params_trainonly.py`（新規）→ `research/method-notes/m3_rederived_params_trainonly_backtest.json`

## 結果（2026-08-23実施）

### stop_buffer_atr_m3再導出（p25パーセンタイル、4通貨プール）

| 通貨 | M3バー数 | レンジ/ATR比 p25(通貨別) |
|---|---|---|
| USD_JPY | 173,220 | 0.679 |
| EUR_JPY | 173,220 | 0.703 |
| GBP_JPY | 173,220 | 0.704 |
| AUD_JPY | 173,220 | 0.714 |

プール(4通貨、n=692,828件)のp25 = **stop_buffer_atr_m3 = 0.7**（参考: M5版=0.703、ほぼ同一）。`atr_trail_multiplier_m3 = 0.7 × 1.0 = 0.7`。詳細: `research/method-notes/m3_entry_params_trainonly.json`

### Train評価（再導出stop_buffer_atr_m3=0.7、atr_trail_multiplier_m3=0.7）

| 指標 | H1版candidate①(基準) | 非公式診断(M5値=0.703無再導出) | 本EXP(M3再導出=0.7) |
|---|---|---|---|
| 必須KPI達成数 | 7/9 | 6/9 | **6/9** |
| 月次シャープ | 2.397 | 3.121 | 3.068 |
| 最大DD(月間) | 8.69% | 8.09% | 8.08% |
| ペイオフレシオ | 1.078 | 1.194 | 1.11(KPI内部値) |
| permutation_p | 0.031(有意) | 0.0699 | 0.0709 |
| n_trades | 300 | 388 | 390 |

**結論・不採用**: stop_buffer_atr_m3の正式な再導出は**0.7**となり、非公式診断で無再導出のまま流用していたM5版の値(0.703)とほぼ完全に一致した。結果として、Train評価もほぼ変化なし（KPI 6/9のまま、permutation_p 0.0699→0.0709とわずかに悪化、他の指標も誤差範囲内）。事前登録した3パターンの判定ルールのうち「非公式診断からも悪化」に技術的には該当するが、差は無視できる範囲（サンプルノイズ相当）であり、実質的には**「再導出しても変わらなかった」**と解釈するのが正確である。

**解釈**: これは偶然ではなく合理的な結果である。`stop_buffer_atr`はバーレンジをそのバーのATRで正規化した比率(スケール不変)であり、M5とM3という近接した時間軸では、この正規化比率の分布形状(特にp25という下側の分位点)がほぼ同じになることは自然。**「M3較正パラメータが未知だったから非公式診断の結果が偏っていた」という仮説は棄却される**。非公式診断とほぼ同一の結果(H1版に近いが、KPI 6/9・permutation_p非有意)が、M3エントリー層設計の実力値と考えられる。

事前登録した選定基準（H1版の7/9以上・perm_p有意）を満たさず、**Train単独で不採用と判定、Validationへは進めない**。SYS-FX015不採用確定、EXP-FX000009をクローズする。残る改善余地として、`zigzag_threshold_atr_m3`(据え置いた1.0)自体の再導出が考えられるが、この値はM5版でも「暫定固定値」であり再導出の方法論が存在しないため、本EXPのスコープ外として次の判断材料に留める。
