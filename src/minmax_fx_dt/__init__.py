"""minmax-fx-day-trading-lab パッケージ.

FX 中期スイング戦略（レンジブレイク・プルバック）の検証ラボ。
親プロジェクト minmax-trading-pilot から GMO FX クライアント等のインフラを流用。

構造:
    data/      - データ取得（GMO FX API、dukascopy 等）
    risk/      - ポジションサイズ決定
    decision/  - 撤退/採用判定基準
    notify/    - 通知（Discord）
    strategy/  - 戦略ロジック（指標 / S/R 検出 / ステートマシン / MTF）
    backtest/  - バックテストエンジン
    shared/    - 共通ユーティリティ
"""

__version__ = "0.1.0"
