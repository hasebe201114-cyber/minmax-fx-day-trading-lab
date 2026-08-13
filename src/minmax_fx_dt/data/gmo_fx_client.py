"""GMO Coin 外国為替FX API クライアント.

Public API (認証不要):
    GET  /v1/ticker
    GET  /v1/klines
    GET  /v1/symbols

Private API (HMAC-SHA256 認証):
    GET  /v1/account/assets         # 資産残高（dict 形式）
    GET  /v1/activeOrders           # 有効注文一覧
    GET  /v1/orders                 # 注文情報取得
    GET  /v1/executions             # 約定情報取得
    GET  /v1/latestExecutions       # 最新の約定一覧
    GET  /v1/openPositions          # 建玉一覧
    GET  /v1/positionSummary        # 建玉サマリー
    POST /v1/order                  # 新規注文
    POST /v1/speedOrder             # スピード注文
    POST /v1/closeOrder             # 決済注文
    POST /v1/cancelOrders           # 注文の複数キャンセル
    POST /v1/cancelBulkOrder        # 注文の一括キャンセル
    PUT  /v1/changeOrder            # 注文変更

エンドポイント:
    Public:  https://forex-api.coin.z.com/public
    Private: https://forex-api.coin.z.com/private

API 手数料: 約定金額 × 0.002%（外国為替FX 初回 API キー作成から 30 日間無料）

署名方式:
    text = timestamp + method + apiPath + bodyJson
    signature = HMAC-SHA256(secret, text)
    ここで apiPath は ``/v1/account/assets`` のように ``/private`` プレフィックスを含めない。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime
from typing import Any, ClassVar, cast

import requests


class GMOClientError(Exception):
    """GMO Coin API で発生したエラー."""


class GMOClient:
    """GMO Coin 外国為替FX API クライアント."""

    BASE_URL_PUBLIC: ClassVar[str] = "https://forex-api.coin.z.com/public"
    BASE_URL_PRIVATE: ClassVar[str] = "https://forex-api.coin.z.com/private"

    def __init__(self, api_key: str, api_secret: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self._session = requests.Session()

    # ----- 認証 -----

    def _sign(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Private API 用の HMAC-SHA256 署名ヘッダを生成."""
        timestamp = f"{int(time.mktime(datetime.now().timetuple()))}000"
        body_str = json.dumps(body) if body else ""
        text = timestamp + method.upper() + path + body_str
        sign = hmac.new(
            bytes(self.api_secret.encode("ascii")),
            bytes(text.encode("ascii")),
            hashlib.sha256,
        ).hexdigest()
        return {
            "API-KEY": self.api_key,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        base: str,
        signed: bool = False,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """HTTP リクエストを実行し、JSON を返す."""
        url = f"{base}{path}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if signed:
            headers.update(self._sign(method, path, body))

        resp = self._session.request(
            method=method,
            url=url,
            params=params,
            json=body,
            headers=headers,
            timeout=self.timeout,
        )
        try:
            data = resp.json()
        except ValueError as exc:
            raise GMOClientError(
                f"invalid JSON response: status={resp.status_code} body={resp.text[:500]}"
            ) from exc

        if resp.status_code >= 400 or (
            isinstance(data, dict) and data.get("status") not in (0, None)
        ):
            messages = data.get("messages", []) if isinstance(data, dict) else []
            raise GMOClientError(f"API error: status={resp.status_code} messages={messages}")
        return cast(dict[str, Any], data)

    def _public(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(method, path, base=self.BASE_URL_PUBLIC, params=params)

    def _private(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(method, path, base=self.BASE_URL_PRIVATE, signed=True, body=body)

    # ----- Public API -----

    def get_ticker(self, symbol: str = "USD_JPY") -> dict[str, Any]:
        """最新ティッカー（bid/ask/last）を取得."""
        return self._public("GET", "/v1/ticker", params={"symbol": symbol})

    def get_symbols(self) -> list[dict[str, Any]]:
        """取扱通貨ペア一覧を取得."""
        data = self._public("GET", "/v1/symbols")
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_klines(
        self,
        symbol: str,
        interval: str,
        date: str,
        price_type: str = "ASK",
    ) -> list[dict[str, Any]]:
        """過去のローソク足（K線）を取得.

        Args:
            symbol: 通貨ペア.
            interval: 1min / 5min / 10min / 15min / 30min / 1hour.
            date: YYYYMMDD 形式の日付.
            price_type: ``ASK`` または ``BID``. デフォルトは ``ASK``.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "date": date,
            "priceType": price_type,
        }
        data = self._public("GET", "/v1/klines", params=params)
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_status(self) -> dict[str, Any]:
        """外国為替FXステータスを取得."""
        return self._public("GET", "/v1/status")

    # ----- Private API -----

    def get_assets(self) -> dict[str, Any]:
        """口座残高を取得.

        Returns:
            ``data`` フィールド以下の dict（equity, availableAmount, balance, ...）.
        """
        data = self._private("GET", "/v1/account/assets")
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
            return data["data"]
        return {}

    def get_active_orders(self) -> list[dict[str, Any]]:
        """有効注文（未約定）一覧を取得."""
        data = self._private("GET", "/v1/activeOrders")
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_orders(
        self,
        symbol: str | None = None,
        order_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """注文情報を取得.

        Args:
            symbol: 通貨ペアでフィルタ.
            order_id: 注文 ID でフィルタ.
        """
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        if order_id is not None:
            params["orderId"] = order_id
        # _private は GET なら params を渡せないので、直接 _request を使う
        url = f"{self.BASE_URL_PRIVATE}/v1/orders"
        headers = {"Content-Type": "application/json"}
        headers.update(self._sign("GET", "/v1/orders", None))
        resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") not in (0, None):
            raise GMOClientError(
                f"API error: status={resp.status_code} messages={data.get('messages', [])}"
            )
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_executions(
        self,
        symbol: str | None = None,
    ) -> list[dict[str, Any]]:
        """約定情報を取得."""
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        url = f"{self.BASE_URL_PRIVATE}/v1/executions"
        headers = {"Content-Type": "application/json"}
        headers.update(self._sign("GET", "/v1/executions", None))
        resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") not in (0, None):
            raise GMOClientError(
                f"API error: status={resp.status_code} messages={data.get('messages', [])}"
            )
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_open_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """建玉（未決済ポジション）一覧を取得.

        Args:
            symbol: 通貨ペアでフィルタ（オプション）.
        """
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = symbol
        url = f"{self.BASE_URL_PRIVATE}/v1/openPositions"
        headers = {"Content-Type": "application/json"}
        headers.update(self._sign("GET", "/v1/openPositions", None))
        resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") not in (0, None):
            raise GMOClientError(
                f"API error: status={resp.status_code} messages={data.get('messages', [])}"
            )
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def get_position_summary(self) -> list[dict[str, Any]]:
        """建玉サマリーを取得."""
        data = self._private("GET", "/v1/positionSummary")
        return cast(list[dict[str, Any]], data.get("data", []) if isinstance(data, dict) else [])

    def send_order(
        self,
        symbol: str,
        side: str,
        size: int,
        execution_type: str = "MARKET",
        limit_price: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """新規注文を発注.

        Args:
            symbol: 通貨ペア.
            side: "BUY" / "SELL".
            size: 注文数量.
            execution_type: "MARKET" / "LIMIT" / "STOP".
            limit_price: 指値価格（LIMIT/STOP 時に必須）.
            client_order_id: 任意のクライアント ID.
        """
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "size": str(size),
            "executionType": execution_type,
        }
        if limit_price is not None:
            body["limitPrice"] = str(limit_price)
        if client_order_id is not None:
            body["clientOrderId"] = client_order_id

        return self._private("POST", "/v1/order", body=body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """注文取消."""
        return self._private("POST", "/v1/cancelOrders", body={"orderId": order_id})


