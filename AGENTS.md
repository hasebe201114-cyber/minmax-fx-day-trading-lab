# AGENTS.md

このプロジェクト（`minmax-fx-day-trading-lab`）の作業を開始する AI エージェント（Claude / Mavis / OpenCode 等）は、以下を順守すること。

## 必読

セッション開始時に**必ず**以下の順で読むこと：

1. `CLAUDE.md`（プロジェクト規約・スコープ・マルチエージェント体制）
2. `obs/minmax_fx_day_trading_lab/引き継ぎ/01進行中/` の最新引き継ぎノート
3. `research/ACTIVE.md`（機械的進行信号機）
4. `research/STRATEGY-BRIEF.md`（戦略選定根拠）
5. `research/SYSTEMS.md`（戦略ポートフォリオ台帳）
6. `obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000003-プロジェクト進捗サマリ.md`

## 重要ルール

- **デイトレード FX 検証のスコープ**: 当日クローズ。持ち越し禁止。
- **DSR 関数の扱い**: `minmax_fx_dt.statistics.dsr.deflated_sharpe_ratio()` は参考値。`decision/criteria.py` の必須ゲートには未組込（Phase 1 マージ・2026-08-29）。`portfolio-ledger.md` の各戦略 DSR 値と `research/method-notes/dsr_for_ledger.json` を参照。
- **検証の閾値**: バックテスト結果を見る前に spec で数値固定（HARKing 防止）
- **B 実装 / C 品質 は別エージェント**で実行（同じ実装者が自分の結果を評価しない）
- **本番採用 GO は司令塔（ユーザー）の明示判断**。C の「採用可」は参考意見
- **GMO コイン 外国為替FX** を主ブローカーとして扱う
- **API キー / `.env.local` は読まない**。シークレットはコミットしない
- **日本語で回答**。途中英語になる場合は注意
- **マルチエージェント体制**: 6体（chief-strategist / strategy-architect / quant-researcher / adversarial-reviewer / integration-deploy / archivist-pm）。Mavis 環境では物理的実行は単一 LLM のため、叩き台・並列実行・Web 検索・ファイル操作を中心に活用。本番品質検証は claude code 環境へ

## ディレクトリ規約

- `research/EXP-FXxxxxx/`（検証単位、`xx` は 5桁連番）
  - `00-prescreen.md`（S 戦略チーム）
  - `00-spec.md`（A 設計チーム）
  - `10-result/`（B 実装チーム、生データのみ）
  - `20-c-review.md`（C 品質チーム）
  - `30-decision.md`（D デプロイチーム / E 進行チーム）
- `research/DATA-FXxxx/`（データチケット）
  - `00-data-spec.md`（DS 取得仕様）
  - `10-build/`（取得・検証スクリプト）
  - `20-acceptance.md`（受入判定）
- `research/_templates/`（テンプレ）
- `obs/minmax_fx_day_trading_lab/`（Obsidian ナレッジベース）
  - `00プロジェクト方針/`（プロジェクト方針文書）
  - `01開発アイデア/`（アイデア・構想）
  - `02対応検討中/`（進行中の検討）
  - `70対応待ち/`（TODO・待機中）
  - `80採用/`（採用された戦略・決定）
  - `85外部レビュー/`（外部 AI / レビュアーからのレビュー記録）
  - `90不採用/`（不採用の決定・理由）
  - `引き継ぎ/01進行中/`（アクティブ）
  - `引き継ぎ/02済み/`（過去）

## 変更履歴
- 2026-08-13: 初版作成
