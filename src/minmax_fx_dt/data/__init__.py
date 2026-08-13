"""データアクセス層.

親PJ (minmax-trading-pilot) の `minmax_trading.data` を踏襲。
GMO Coin 外国為替FX API クライアントを中核に、dukascopy 等を追加可能。
"""

from minmax_fx_dt.data.gmo_fx_client import GMOClient, GMOClientError

__all__ = ["GMOClient", "GMOClientError"]
