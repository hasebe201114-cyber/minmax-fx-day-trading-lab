# minmax-fx-day-trading-lab

FX（外国為替証拠金取引）の**マルチタイムフレーム・中期スイング戦略**を、ティック/分足レベルの時系列データで定量的に検証するラボです。

`minmax-trading-pilot`（FX 自動売買の親プロジェクト）の検証ラボ層（バックテスト・フォワードテストの定量評価レイヤ）として位置付けられます。lp-strategy-lab（流動性プール戦略）・crypto-strategy-lab（多戦略探索）・trading-app-v2（BTC キャリ実運用）とは別の、独立したリポジトリ・プロジェクトです。マルチエージェント体制と検証プロセスは lp-strategy-lab の規約をそのまま踏襲しています。

`minmax` は minmax チーム（ユーザー）のクオンツ系プロジェクト共通の**識別子プレフィクス**で、GitHub リポジトリは `minmax-<topic>` の形式で統一します。

## 開発言語

**Python 3.11+ エコシステム**で構築。親プロジェクト `minmax-trading-pilot` の Python 資産を流用:

- `data/gmo_fx_client.py` — GMO Coin FX API クライアント（HMAC-SHA256 認証、klines / ticker / symbols / order 全エンドポイント）
- `risk/sizing.py` — ポジションサイズ決定（SYS-FX007 用に拡張）
- `decision/criteria.py` — 撤退/採用判定（K1m〜K7m 7 指標で評価）
- `notify/discord.py` — Discord Webhook 通知

## 詳細
- プロジェクト規約: [CLAUDE.md](./CLAUDE.md)
- ドキュメント配置（A/B/C/D 区分）: [docs/README.md](./docs/README.md)
- プロジェクトの目的・ゴール: [obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000001-minmax-fx-day-trading-lab.md](./obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000001-minmax-fx-day-trading-lab.md)
- マルチエージェント体制: [obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000002-マルチエージェント体制定義.md](./obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000002-マルチエージェント体制定義.md)
- 進捗サマリ: [obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000003-プロジェクト進捗サマリ.md](./obs/minmax_fx_day_trading_lab/00プロジェクト方針/PJ000003-プロジェクト進捗サマリ.md)
- 戦略コンセプト: [obs/minmax_fx_day_trading_lab/01開発アイデア/OBS000001-レンジブレイク・プルバック戦略.md](./obs/minmax_fx_day_trading_lab/01開発アイデア/OBS000001-レンジブレイク・プルバック戦略.md)
- 通貨選定: [obs/minmax_fx_day_trading_lab/01開発アイデア/OBS000002-対象通貨5通貨選定.md](./obs/minmax_fx_day_trading_lab/01開発アイデア/OBS000002-対象通貨5通貨選定.md)
- 進行表: [research/ACTIVE.md](./research/ACTIVE.md)
- 戦略選定根拠: [research/STRATEGY-BRIEF.md](./research/STRATEGY-BRIEF.md)
- 戦略ポートフォリオ台帳: [research/SYSTEMS.md](./research/SYSTEMS.md)

## クイックスタート

### 1. 開発環境セットアップ

```powershell
# Python 3.11+ 仮想環境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 依存インストール
pip install -r requirements-dev.txt

# 環境変数 (.env を作成)
Copy-Item .env.example .env
# .env を編集して GMO_FX_API_KEY / GMO_FX_API_SECRET を設定
```

### 2. ドキュメント確認

1. セッション開始時: `obs/minmax_fx_day_trading_lab/引き継ぎ/01進行中/` の最新引き継ぎノートを読む
2. `research/ACTIVE.md` で現在の進行中の検証を把握
3. `research/STRATEGY-BRIEF.md` で戦略選定の現状と未確定事項を確認

### 3. 動作確認

```powershell
# GMO FX API クライアントの動作確認 (Public API のみ、API キー不要)
python -c "from minmax_fx_dt.data import GMOClient; c = GMOClient('', ''); print(c.get_symbols()[:3])"

# リスクサイズ決定のサンプル実行
python -m minmax_fx_dt.risk.sizing

# 撤退/採用判定のサンプル実行
python -m minmax_fx_dt.decision.criteria
```

## 想定ブローカー

**GMO コイン 外国為替FX** を前提として検証します。

- API 手数料: 約定金額 × 0.002%（30日間無料トライアル中は実質ゼロ）
- USD/JPY スプレッド: 0.3 銭
- 21 通貨ペア、レバレッジ最大25倍
- 暗号資産も同一口座・同一 API でアクセス可能（将来拡張余地）
- API エンドポイント: `https://forex-api.coin.z.com/public` / `/private`（HMAC-SHA256 認証）

将来、別のブローカー（OANDA Japan 個人アカウントでの Gold 会員昇格後、IBKR 等）へ移行する可能性は残します。
