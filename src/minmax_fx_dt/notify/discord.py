"""Discord Webhook 通知 (シンプル POST、HMAC なし).

設計判断 (2026-08-10):
- trading-app-v2 は `standardwebhooks` ^1.0.0 + `DISCORD_WEBHOOK_DAILY` Secrets を使用
- ただし Discord Webhook は HMAC 署名検証をしないため、依存追加せず生 POST が最簡
- minmax-trading-pilot は Python 3.11+ で `requests` が既にある → 追加依存ゼロ

使い方 (Python から):
    from minmax_fx_dt.notify.discord import send, info, warn, error
    send(webhook_url, "Hello")
    info(webhook_url, "Daily digest ready", fields={"失敗率": "5%", "連続失敗": 0})

使い方 (CLI):
    python scripts/notify_discord.py --content "Hello"
    python scripts/notify_discord.py --level info --title "Digest" --content "..." --field k=v --field k2=v2

GitHub Actions:
    env:
      DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
    run: python scripts/notify_discord.py --level info --title "..." --content "..."

制約 (Discord 仕様):
- content: 最大 2000 文字
- embed.description / embed.title: 最大 4096 文字 / 256 文字
- embeds: 最大 10 個
- 429 (Rate Limited): `Retry-After` で指数バックオフ
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests

# structlog は optional (未インストール時は標準 logging にフォールバック)
try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)

# Discord Webhook 制約
CONTENT_MAX = 2000
EMBED_TITLE_MAX = 256
EMBED_DESC_MAX = 4096
EMBED_FIELD_NAME_MAX = 256
EMBED_FIELD_VALUE_MAX = 1024
EMBED_FOOTER_MAX = 2048
EMBEDS_MAX = 10
RATE_LIMIT_RETRIES = 3


def _truncate(text: str, limit: int) -> str:
    """Discord の文字数制限に合わせて切り詰める.

    切り詰めた場合は `...` マーカーを末尾に付与する (3 文字消費)。
    """
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def build_embed(
    title: str,
    content: str | None = None,
    fields: dict[str, Any] | list[tuple[str, Any, bool]] | None = None,
    color: int | None = None,
    level: str = "info",
) -> dict[str, Any]:
    """Discord Embed を構築.

    Args:
        title: 見出し (256 文字以内)
        content: 本文 (4096 文字以内、embeds[0].description に格納)
        fields: フィールド dict {name: value} または list[(name, value, inline)]
        color: 0xRRGGBB (10 進 int)
        level: info / warn / error (省略時 info と同色)

    Returns:
        Discord Embed dict
    """
    embed: dict[str, Any] = {
        "title": _truncate(title, EMBED_TITLE_MAX),
    }
    if content is not None:
        embed["description"] = _truncate(content, EMBED_DESC_MAX)

    if fields is not None:
        field_list: list[dict[str, Any]] = []
        if isinstance(fields, dict):
            items: list[tuple[str, Any, bool]] = [(k, v, False) for k, v in fields.items()]
        else:
            items = [(name, val, bool(inline)) for name, val, inline in fields]
        for name, value, inline in items[:25]:  # Discord は 25 フィールド上限
            field_list.append(
                {
                    "name": _truncate(str(name), EMBED_FIELD_NAME_MAX),
                    "value": _truncate(str(value), EMBED_FIELD_VALUE_MAX),
                    "inline": inline,
                }
            )
        embed["fields"] = field_list

    # level → color 自動マッピング (該当しなければ color を使う)
    level_color = {
        "info": 0x3498DB,     # 青
        "warn": 0xF39C12,     # 黄
        "warning": 0xF39C12,
        "error": 0xE74C3C,    # 赤
        "critical": 0xC0392B,
        "success": 0x2ECC71,  # 緑
    }.get(level.lower())
    embed["color"] = color if color is not None else (level_color or 0x3498DB)

    embed["footer"] = {"text": _truncate("minmax-trading-pilot", EMBED_FOOTER_MAX)}
    embed["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return embed


def send(
    webhook_url: str,
    content: str | None = None,
    embeds: list[dict[str, Any]] | None = None,
    username: str = "minmax-trading-pilot",
    timeout: float = 10.0,
) -> bool:
    """Discord Webhook に POST.

    Args:
        webhook_url: Discord で発行した Webhook URL
        content: テキスト (2000 文字以内、embeds と同時に送れる)
        embeds: Embed dict のリスト (最大 10 個)
        username: 送信者名 (Discord 側で上書きされる場合あり)
        timeout: HTTP タイムアウト秒

    Returns:
        True = 成功, False = 失敗
    """
    if not webhook_url:
        logger.warning("discord.send.empty_webhook_url")
        return False
    if not content and not embeds:
        logger.warning("discord.send.empty_payload")
        return False
    if content:
        content = _truncate(content, CONTENT_MAX)
    if embeds:
        embeds = embeds[:EMBEDS_MAX]

    payload: dict[str, Any] = {"username": username}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds

    body = json.dumps(payload, ensure_ascii=False)
    headers = {"Content-Type": "application/json"}

    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        try:
            resp = requests.post(
                webhook_url,
                data=body.encode("utf-8"),
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.error(f"discord.send.http_error attempt={attempt} error={exc}")
            if attempt == RATE_LIMIT_RETRIES:
                return False
            time.sleep(2**attempt)
            continue

        # 429 = レート制限
        if resp.status_code == 429:
            retry_after = float(resp.json().get("retry_after", 1.0)) if resp.headers.get("content-type", "").startswith("application/json") else 1.0
            logger.warning(f"discord.send.rate_limited retry_after={retry_after} attempt={attempt}")
            time.sleep(retry_after + 0.1)
            continue

        if 200 <= resp.status_code < 300:
            logger.info(f"discord.send.ok status={resp.status_code} attempt={attempt}")
            return True

        # 4xx/5xx
        logger.error(
            f"discord.send.http_status status={resp.status_code} "
            f"body={resp.text[:500]!r} attempt={attempt}"
        )
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            # 4xx は再送しても直らない (URL 誤り・payload 異常)
            return False
        if attempt == RATE_LIMIT_RETRIES:
            return False
        time.sleep(2**attempt)

    return False


def info(
    webhook_url: str,
    title: str,
    content: str | None = None,
    fields: dict[str, Any] | None = None,
) -> bool:
    """INFO レベル通知 (青)."""
    return send(webhook_url, embeds=[build_embed(title, content, fields, level="info")])


def warn(
    webhook_url: str,
    title: str,
    content: str | None = None,
    fields: dict[str, Any] | None = None,
) -> bool:
    """WARN レベル通知 (黄)."""
    return send(webhook_url, embeds=[build_embed(title, content, fields, level="warn")])


def error(
    webhook_url: str,
    title: str,
    content: str | None = None,
    fields: dict[str, Any] | None = None,
) -> bool:
    """ERROR レベル通知 (赤)."""
    return send(webhook_url, embeds=[build_embed(title, content, fields, level="error")])


def _env_webhook_url() -> str:
    """環境変数 `DISCORD_WEBHOOK_URL` から取得."""
    return os.environ.get("DISCORD_WEBHOOK_URL", "").strip()


def _parse_field(items: list[str]) -> dict[str, str]:
    """`--field k=v` のリストを dict に変換."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            print(f"WARN: --field '{item}' does not contain '='. Skipped.", file=sys.stderr)
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント.

    Examples:
        python scripts/notify_discord.py --content "Hello"
        python scripts/notify_discord.py --level info --title "Daily Digest" \\
            --content "Hourly CI 24h" --field 失敗率=0% --field 連続失敗=0
        python scripts/notify_discord.py --level error --title "🚨 GMO 障害" \\
            --content "Hourly Signal Update が失敗しました" \\
            --field RunURL=https://github.com/... --field 戦略=EXP-044
        # テスト: 実際の送信なしで payload をダンプ
        python scripts/notify_discord.py --dry-run --level info --title "Test"
    """
    parser = argparse.ArgumentParser(description="Discord webhook notifier")
    parser.add_argument("--webhook-url", default=_env_webhook_url(),
                        help="Discord webhook URL (default: $DISCORD_WEBHOOK_URL)")
    parser.add_argument("--content", default=None, help="Plain text message (2000 chars max)")
    parser.add_argument("--title", default=None, help="Embed title")
    parser.add_argument("--level", default="info", choices=["info", "warn", "warning", "error", "critical", "success"])
    parser.add_argument("--field", action="append", default=[],
                        help="Embed field as key=value (can be repeated)")
    parser.add_argument("--username", default="minmax-trading-pilot")
    parser.add_argument("--dry-run", action="store_true",
                        help="実際には送信せず payload を標準出力にダンプ (テスト用)")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.webhook_url:
        print("ERROR: --webhook-url or DISCORD_WEBHOOK_URL env is required "
              "(or use --dry-run).", file=sys.stderr)
        return 2

    embeds: list[dict[str, Any]] = []
    if args.title:
        embeds.append(
            build_embed(
                title=args.title,
                content=args.content,
                fields=_parse_field(args.field) if args.field else None,
                level=args.level,
            )
        )
    elif args.content and not args.field:
        # title なし + field なしなら content を plain で送る
        pass
    elif args.field:
        print("ERROR: --field requires --title.", file=sys.stderr)
        return 2

    if not embeds and not args.content:
        print("ERROR: --title or --content is required.", file=sys.stderr)
        return 2

    payload: dict[str, Any] = {"username": args.username}
    if args.content:
        payload["content"] = _truncate(args.content, CONTENT_MAX)
    if embeds:
        payload["embeds"] = embeds[:EMBEDS_MAX]

    if args.dry_run:
        # テスト用: payload をダンプして exit 0
        print("[DRY-RUN] Would POST to Discord webhook with the following payload:")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"[DRY-RUN] Payload size: {len(json.dumps(payload, ensure_ascii=False))} bytes")
        return 0

    return 0 if send(args.webhook_url, content=args.content, embeds=embeds, username=args.username) else 1


if __name__ == "__main__":
    raise SystemExit(main())


