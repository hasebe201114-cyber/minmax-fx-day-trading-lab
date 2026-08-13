# データカタログ（minmax-fx-day-trading-lab）

> 担当: A 設計 + E 進行
> 取得済みデータ系列の一覧。各系列は `data/curated/<ds-id>.json` に保存する。
> 取得日と取得ソースは必ず `MANIFEST.json` に記録する。

## 取得済み（none yet）

| DS ID | 通貨ペア / 種別 | 期間 | 粒度 | 取得日 | 取得ソース | 検証状況 |
|---|---|---|---|---|---|---|
| （なし） | - | - | - | - | - | - |

## 取得予定

| DS ID | 内容 | 主ソース | 想定サイズ | 優先度 |
|---|---|---|---|---|
| DS-1 | USD/JPY, EUR/JPY, EUR/USD の OHLCV 3年分 M1 足 | dukascopy / histdata.com | 各ペア約 1.5M 行 × 5列 = 7.5M 行 | 1 |
| DS-1b | GBP/JPY, AUD/JPY, GBP/USD, AUD/USD の OHLCV 1年分 M1 足 | 同上 | 各ペア約 0.5M 行 | 2 |
| DS-2 | GMO スプレッド時系列 | GMO 公開値 or 1分粒度の取得 | 6ヶ月 × 1分 = 約 25万行 | 1 |
| DS-3 | 経済指標カレンダー | investing.com / ForexFactory | 3年 × 30 イベント/年 = 約 100 行 | 2 |
| DS-4 | セッション境界時刻 + 出来高/ボラ集計 | 自前（DS-1 から派生） | 3年 × 1時間 = 約 26,000 行 | 2 |
| DS-5 | スリッページ代表性データ | 0.5 pip 固定フォールバック | 1 行（固定値） | 1 |
| DS-6 | GMO コイン API 仕様 | 公式ドキュメント | 構造化 JSON 1 ファイル | 1 |

## 命名規則

- `data/raw/<ds-id>/<source>__<pair>__<granularity>__<start>__<end>.csv`（生データ）
- `data/curated/<ds-id>.json`（検証済み・正規化済みデータ）
- `data/MANIFEST.json`（全データ系列のメタ情報一覧）
