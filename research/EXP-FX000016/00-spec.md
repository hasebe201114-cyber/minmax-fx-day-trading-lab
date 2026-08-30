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
- **実行方式(2026-08-24改訂、2度目の変更)**: 当初Claude Code Remoteの使い捨てセッションRoutine(`create_new_session_on_fire`)で毎時実行する設計だったが、実測したところ発火は記録されるがgit pushが完了しないという再現性のある問題が判明した(既存のSYS-FX012週次フォワードテストRoutineでも同様の問題が確認され、単発の不具合ではないと判明。原因未特定のまま)。LLMエージェントセッションを毎回起動する必要は無いという指摘を受け、GitHub Actions(`.github/workflows/live-ticker-poll.yml`、毎時cron)へ切り替えたが、**手動`workflow_dispatch`でテスト実行したところ開始2秒で失敗・ログ取得も404**。司令塔確認により**GitHub Actionsの実行時間上限(無料枠)を超過しており、課金プラン契約が必要**と判明。ワークフローファイル自体は正しく登録されている(将来課金プランへ移行した場合はそのまま使える)ため削除せず残すが、**現時点では自動実行の目処が立っていない**。Stage 1は当面「必要な時に司令塔またはセッションから手動実行」の運用に切り替える(下記「現状の運用」参照)

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

## 現状の運用（2026-08-26時点、司令塔側の対応により解決）

自動実行を試した経緯:

| 方式 | 結果 |
|---|---|
| Claude使い捨てセッションRoutine | 発火はするがgit pushが完了しない(原因未特定のまま棚上げ) |
| GitHub Actions(2026-08-24時点) | 無料枠超過で即時失敗 |
| Firebase Cloud Functions(2026-08-24、司令塔提案で着手) | コード実装のみでデプロイ未実施のまま、下記の解決で不要になった |
| **GitHub Actions(2026-08-26、司令塔が課金/設定を解決)** | **稼働中**。`.github/workflows/live-ticker-poll.yml`が実際に`github-actions[bot]`名義で毎時コミットを生成していることを確認済み(例: コミット`9bc9616`) |

司令塔がGitHub Actions側の制約(無料枠超過)を解消し、当初設計した`live-ticker-poll.yml`がそのまま稼働を始めた。**Firebase Cloud Functions版(`functions/`)はこの時点で不要と判明したため削除した**(2026-08-26)。GitHub ActionsとFirebaseを二重に運用する理由が無く、放置すると「どちらが実際に動いているか」で将来混乱するため。

さらに司令塔が以下を追加で構築(このセッションの外で実施、2026-08-26):

- `.github/workflows/update-ds1-forward.yml`(毎時5分): `scripts/live_monitor/fetch_m5_ohlcv.py`(新設)でM5 OHLCVを取得し`data/curated/ds-1-forward.json`を更新。`live-ticker-poll.yml`はbid/ask気配値のみでM5足を更新しないため、`sysfx012_forward_test_ledger.json`の`latest_bar_by_pair`が2026-08-24 16:25 JSTで止まっていた問題への対応。`ds-1-forward.json`は.gitignore対象化(毎時1.4MB+の更新でコミット履歴が膨らむのを回避、Actions runner上で直接読み書き)
- `.github/workflows/sysfx012-fx-forward-cycle.yml`(週次月曜00:00 UTC): Claude使い捨てセッションRoutineの不具合を受け、SYS-FX012週次フォワードテストサイクル自体もGitHub Actionsへ移管。`data/curated/ds-1.json`(479MB、.gitignore対象)を`data/raw/ds-1/*.csv`から都度再生成してから`run_forward_test_cycle.py`を実行し、ledgerをcommit/push
- `trading-app-v2`側に`sysfx012-fx-forward-sync.yml`(月曜12:00 JST)があり、本リポジトリがcommitしたledgerを公開Web UIへ同期する(クロスリポジトリ連携)

### フォワードデータ消失バグ（2026-08-29 発見・修正）

上記の`update-ds1-forward.yml`には、**取得したM5バーが1件も残らない**という致命的な欠陥があった。

**内容**: `data/curated/ds-1-forward.json`を.gitignore対象にした一方で、workflowは「既存JSONへ追記マージ」した結果を**何もコミットしていなかった**。GitHub Actions runnerは毎回使い捨てのため、毎時の追記結果はジョブ終了とともに消える。結果として、週次cycleがフォワード区間として読めるのは常に「実行直前の`--lookback-days 2`分」だけとなり、cutoff（2026-08-15 06:00 JST）以降の大半が恒久的な空白になっていた。

**影響**: `sysfx012_forward_test_ledger.json`は`n_events_raw=0`（イベントゼロ）を報告し続けていたが、実データでの真値は**6件**だった。ledgerは「イベントが無かった」のか「データが無かった」のかを区別できる情報を持たないため、**フォワードテストが実質的に何も観測していない状態が2週間気づかれずに続いた**。「Routineは成功と記録されるのに実データが更新されない」（2026-08-24に発覚したClaude Routineの不具合）と同型の、成功したように見える無音の失敗である。

**修正**:

1. バーの正本を git 管理の追記型CSV `data/raw/ds-1-forward/*.csv` に移し、`ds-1-forward.json`はそこから何度でも再生成できる派生物として扱う（`data/raw/ds-1/`＋`ds-1.json`と同じ「raw CSVが正本・curated JSONは派生」構造に統一）。既存の`weekly_cycle.py`が元々この構造で書かれていたのに対し、2026-08-26に追加した2つのworkflowだけがそこを迂回していたのが混入経路
2. `fetch_m5_ohlcv.py`: CSVの読み書き（`load_forward_csv`/`write_forward_csv`）を追加し、JSONが存在しない環境でもCSVから全期間を復元する（`load_existing()`はJSONとCSVの和集合を返すため、どちらか一方しか無い環境でも欠落しない）
3. `run_forward_test_cycle.py`: `load_m5_forward()`が正本CSVを直接読むよう変更。fetchステップの実行順序に依存しなくなった
4. `update-ds1-forward.yml`: `permissions: contents: write`に変更し、毎時CSVをcommit/push
5. `sysfx012-fx-forward-cycle.yml`: lookbackを2日→**9日**（週次7日＋余裕2日）に拡大し、毎時workflowが数日停止しても週次で穴が自動的に埋まるようにした（self-healing）。ledgerと同時にCSVもcommit
6. 回帰テスト`tests/test_forward_csv_persistence.py`（4件）を新設。「JSON不在時にCSVから全期間が復元されること」を固定した

**再計測結果**: cutoff以降の全期間（2026-08-15 06:00 〜 08-29 05:55 JST、10営業日）をバックフィルして再計算したところ、イベントは0件→**6件**（dedup後6、H1トレンド判定不能除外後3）。トレード数は0件のまま（内訳は下記）。

**受け入れ基準Dへの反映**: 「蓄積データが再分析可能な形でコミットされる」という基準は、live-tickerについては満たされていたが、M5 OHLCVについては満たされていなかった。本修正で両方が満たされる。

つまり、EXP-FX000016 Stage 1で目指していた「決定的な作業にLLMエージェントセッションを使わない」自動化は、最終的に**GitHub Actions 3ワークフローの組み合わせ**で実現された。`reconcile_divergence.py`(コンポーネントB/C)は未統合のまま、セッション起動時の手動実行を継続する(Firebase版への移植構想も含めて不要になったため、Python版のまま`update-ds1-forward.yml`等への統合を今後検討)。

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
