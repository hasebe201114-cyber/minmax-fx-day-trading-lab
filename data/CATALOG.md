# データカタログ（minmax-fx-day-trading-lab）

> 担当: A 設計 + E 進行
> 取得済みデータ系列の一覧。各系列は `data/curated/<ds-id>.json` に保存する。
> 取得日と取得ソースは必ず `MANIFEST.json` に記録する。
> 2026-08-15 更新: Phase -1 完了 (OBS000008 §4 / OBS000007 §8 対応)、DS-1 を 5 通貨 × 2020-2025 に拡張

## 取得済み

| DS ID | 通貨ペア / 種別 | 期間 | 粒度 | 取得日 | 取得ソース | 検証状況 |
|---|---|---|---|---|---|---|
| **DS-1** | USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY, EUR/USD | 2020-01-01 〜 2025-12-31 (実質) | M5 (5min) | 2026-08-15 集約 | dukascopy (〜2024-06-30) + GMO Coin 外国為替FX Public API (2024-07-01〜) | ✅ 集約スクリプト実行済、mtf_cache 20 ファイル生成 |
| **DS-7** | 5 通貨 (USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY) のスワップ概算値 | 2024-01-01 〜 2024-12-31 (固定値) | 1 lot/日 | 2026-08-13 | 概算 (OBS000004 差し戻し 2 対応) | △ 概算値、GMO 公式 API 未接続、EUR_USD は 0.0 |

### DS-1 詳細 (2026-08-15 Phase -1 完了)

- **集約元**: `data/raw/ds-1/` 配下 16 ファイル (約 154 MB)
  - 5 通貨 × 3 期間構造 (dukascopy 2020-2024H1 + 既存 2024 単年 + GMO 2024H2-2025)
  - USD_JPY のみ 4 ファイル (2024-01-02_2024-01-04 のテスト 1 ファイル追加)
- **集約スクリプト**: `scripts/data/fetch_ds1_ohlcv.py:aggregate_to_json` (改修版、OBS000008 §4.4)
- **集約結果**: `data/curated/ds-1.json` (457 MB, 5 通貨 × 447k bars/通貨, 合計 2,238,752 bars)
- **MTF キャッシュ**: `data/curated/mtf_cache/` (20 ファイル, 59.4 MB, parquet 形式, 5 通貨 × M5/M15/H4/D1)
- **バックアップ**: `data/curated/ds-1.json.bak_20260815` (旧 2024 単年版, 65.9 MB)
- **再生成**: `python scripts/precompute_mtf.py --force` で MTF キャッシュ再構築可能

## 取得予定

| DS ID | 内容 | 主ソース | 想定サイズ | 優先度 |
|---|---|---|---|---|
| DS-1b | GBP/JPY, AUD/JPY, GBP/USD, AUD/USD の OHLCV 1年分 M1 足 | 同上 | 各ペア約 0.5M 行 | 2 |
| DS-2 | GMO スプレッド時系列 | GMO 公開値 or 1分粒度の取得 | 6ヶ月 × 1分 = 約 25万行 | 1 |
| DS-3 | 経済指標カレンダー | investing.com / ForexFactory | 3年 × 30 イベント/年 = 約 100 行 | 2 |
| DS-4 | セッション境界時刻 + 出来高/ボラ集計 | 自前（DS-1 から派生） | 3年 × 1時間 = 約 26,000 行 | 2 |
| DS-5 | スリッページ代表性データ | 0.5 pip 固定フォールバック | 1 行（固定値） | 1 |
| DS-6 | GMO コイン API 仕様 | 公式ドキュメント | 構造化 JSON 1 ファイル | 1 |
| **DS-8** | 注文板/センチメント (MT-3 拡張用) | 未定 | 未定 | 4 (DS-1 完了後再評価) |

## 命名規則

- `data/raw/<ds-id>/<source>__<pair>__<granularity>__<start>__<end>.csv`（生データ）
- `data/curated/<ds-id>.json`（検証済み・正規化済みデータ）
- `data/MANIFEST.json`（全データ系列のメタ情報一覧）

## 変更履歴

- 2026-08-15: DS-1 を 5 通貨 × 2020-2025 に拡張 (Phase -1, Mavis)。OBS000008 §4 / OBS000007 §8 対応。DS-7 を取得済みに追加。DS-8 を取得予定に追加 (OBS000004 論点 A 対応)
- (旧履歴): 2026-08-13: DS-1 (2024 単年), DS-7 (概算スワップ) を取得済み化
