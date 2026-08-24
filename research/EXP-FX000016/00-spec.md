# 検証仕様書（Spec） - EXP-FX000016

> 担当: A 設計チーム（strategy-architect）
> 起票: 2026-08-24
> 前提: `00-prescreen.md`（GO）
> 原則: 実装の受け入れ基準は**着手前**に固定する（統計的HARKing防止ではなく、スコープの肥大化・後付け正当化の防止が目的）

## 対応 OBS 番号
（E進行チームが採番予定）

## 紐づく戦略系統 ID
SYS-FX022

## 目的（再掲）

SYS-FX012の実運用移行に向け、実発注を一切行わずに以下3点を満たすライブ監視レイヤーを整備する:
1. トレード・判定が正しく行われているかをリアルに検証・確認できる
2. バックテストとの乖離状況を判断できる
3. リアルでしか拾えないデータの蓄積

## 段階分割（重要: 過大な約束をしないための事前区切り）

一度に全てを実装せず、2段階に分ける。**本EXPで実装するのはStage 1のみ**。Stage 2は着手前に別途、Stage 1で得られた知見をもとに要否を判断する。

### Stage 1（本EXPのスコープ）: 毎時スナップショット + 乖離レコンサイル
### Stage 2（将来検討、本EXPのスコープ外）: 高頻度ポーリングによる独立シャドー検出

理由: 本セッションのスケジューリング機構(Routine)は最短でも1時間間隔でしか定期実行できない制約がある。M5レベルの独立検出(要件1の完全な実現)には少なくともM5相当の粒度のライブデータが必要だが、毎時1サンプルではそれを満たせない。まず毎時サンプリングで「モデル仮定と実測の乖離がどの程度か」を数週間観測し、乖離が無視できない規模だと分かった場合にのみ、Stage 2としてポーリング頻度を上げる設計拡張に着手する。事前に不要な高頻度実装をしない。

## コンポーネント設計（Stage 1）

### A. ライブ気配値記録
- 新規: `scripts/live_monitor/poll_ticker.py`
- `GMOClient("", "").get_ticker()`（認証不要、公開エンドポイント）を1回呼び出し、対象4通貨(USD/EUR/GBP/AUD_JPY)のbid/ask/timestampを取得
- 出力: `data/raw/live-ticker/YYYY-MM.csv`（月次ローテーション）に追記。列: `polled_at, pair, bid, ask, spread_pips, api_timestamp, market_status`
- 実行: 新規Routine(毎時)から起動。1回の呼び出しはAPI 1コールのみで軽量、レート制限リスクは無視できる水準

### B. バックテストとの乖離レコンサイル（要件2に対応）
- 新規: `scripts/live_monitor/reconcile_divergence.py`
- 入力: `research/method-notes/sysfx012_forward_test_ledger.json`(週次バッチが検出した各トレードのentry_time/exit_time/pair)+ Aで蓄積したライブ気配値
- 各トレードのentry_time/exit_timeに**時間的に最も近い**ライブ気配値レコードを突き合わせ、以下を算出:
  - 実測スプレッド(ask-bid) vs コストモデルの`SPREAD_PIPS`定数との乖離率
  - 突き合わせ可能なレコードが無い場合(記録密度不足)は「判定不能」として明示、無理に埋めない
- 出力: `research/method-notes/live_cost_divergence.json`。通貨別・時間帯別(DS-4のセッション区分を再利用)に集計

### C. 判定タイミングの整合性チェック（要件1の一次的な実現、Stage 1の範囲内）
- Bのレコンサイル時に、各トレードのentry_time/exit_timeの`market_status`(OPEN/CLOSE等)を突き合わせ、**週末クローズ中や市場休止中に検出されたイベントが無いか**を機械的に検証する
- これは「M5レベルの独立再検出」ではなく、「バッチが出した判定が、実際に市場が開いていた時間帯のものか」という基礎的な整合性チェック。要件1の完全な実現(独立シャドー検出)はStage 2に持ち越す

### D. データ蓄積（要件3に対応）
- Aの`data/raw/live-ticker/`自体が蓄積データ。Bの集計(時間帯別実測スプレッド分布)を月次でACTIVE.mdに記録する運用とする
- 将来、十分なサンプルが溜まった時点で、コストモデルの`SPREAD_PIPS`定数を実測ベースの値へ較正し直す判断材料として使う(本EXPでは較正自体は行わない、あくまでデータ収集と乖離の可視化まで)

## Stage 1 受け入れ基準（事前登録）

- [ ] A: 毎時Routineが正常に稼働し、最低1週間、対象4通貨のティックが記録される
- [ ] B: `reconcile_divergence.py`が既存のフォワードテスト台帳と突き合わせ、乖離レポートを出力する（対応するトレードが無い期間は「データ不足」として明示、エラーにしない）
- [ ] C: 市場休止中の誤検出が無いことを確認する（もしあれば別途バグとして扱う）
- [ ] D: 蓄積データが`data/raw/live-ticker/`にコミットされ、再分析可能な形式で残る

## 完全に対象外（本EXPでは扱わない）

- 認証が必要な発注API・実発注（CLAUDE.md方針により採用GO後の別フェーズ）
- Stage 2（高頻度ポーリングによる独立シャドー検出）
- コストモデル(`SPREAD_PIPS`等)の実際の書き換え（データ収集と可視化までが本EXPの範囲）

## 実行

- `scripts/live_monitor/poll_ticker.py`（新規、Routineから毎時起動）
- `scripts/live_monitor/reconcile_divergence.py`（新規、週次または任意タイミングで実行）
- 出力: `data/raw/live-ticker/`, `research/method-notes/live_cost_divergence.json`
