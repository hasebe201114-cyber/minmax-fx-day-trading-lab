"""資金管理(ポジションサイズ決定と決済時の残高反映)の共通実装。OBS000015 対応(2026-09-30)。

従来は各スクリプトに同じ処理が複製され、レバレッジ25倍の上限を1ポジションごとにしか
適用していなかった。GMOコイン外国為替FXの上限は口座全体(保有中の建玉合計)にかかるため、
複数通貨を同時に保有すると実際には建てられない規模を含んでいた(SYS-FX012 Train で合計最大59.5倍)。

本モジュールは従来の処理(エントリー時に残高×リスク率で1Rの金額を決め、1ポジションの
レバレッジ上限でリスク率を下げ、決済時に r_net×1R金額 を残高へ反映、残高0で破産)を
そのまま踏襲し、保有中の建玉合計の上限(aggregate_leverage_cap)を加える。
aggregate_leverage_cap=None なら従来の処理と完全に一致する。
"""
from __future__ import annotations

from typing import Any

GMO_MAX_LEVERAGE = 25.0
# 公式評価の既定値。修正前の数値を再現する比較用に None へ差し替えられるよう、呼び出し時に参照する
DEFAULT_AGGREGATE_LEVERAGE_CAP: float | None = GMO_MAX_LEVERAGE
_DEFAULT = object()


def is_active(trade: dict) -> bool:
    """実際に建てた取引か(破産後の見送り・合計レバレッジ上限による見送りを除く)。"""
    return not (trade.get("skipped_ruin") or trade.get("skipped_cap"))


def settle_trades(all_trades: list[dict], *, initial_capital: float, risk_pct: float, start_label: str,
                  per_position_leverage_cap: float = GMO_MAX_LEVERAGE,
                  aggregate_leverage_cap: Any = _DEFAULT,
                  cap_mode: str = "shrink") -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    """取引リストに資金管理を適用し、各取引の辞書に結果を書き込む。

    各取引には entry_time, exit_time, leverage_ratio(=エントリー価格/初期リスク幅), r_net が必要。
    書き込むキー: risk_dollars, effective_risk_pct, leverage_capped, skipped_ruin, skipped_cap,
    aggregate_cap_action("none"|"shrink"|"skip"), notional_usd, dollar_pnl, balance_after。

    aggregate_leverage_cap: 保有中の建玉金額の合計を「確定残高 × 上限」以内に収める。
      cap_mode="shrink" は残り枠まで数量を縮小(枠がなければ見送り)、"skip" は全量が入らなければ見送り。
    戻り値: (最終残高, 資産曲線[{time, balance}], 集計情報)
    """
    if aggregate_leverage_cap is _DEFAULT:
        aggregate_leverage_cap = DEFAULT_AGGREGATE_LEVERAGE_CAP
    if cap_mode not in ("shrink", "skip"):
        raise ValueError(f"cap_mode must be 'shrink' or 'skip', got {cap_mode!r}")
    events = []
    for idx, t in enumerate(all_trades):
        events.append((t["entry_time"], 0, idx, "ENTRY"))
        events.append((t["exit_time"], 1, idx, "EXIT"))
    events.sort(key=lambda e: (e[0], e[1]))

    balance = initial_capital
    ruined = False
    open_notional = 0.0
    max_aggregate_leverage = 0.0
    n_shrunk = n_skipped = 0
    equity_curve: list[dict[str, Any]] = [{"time": start_label, "balance": balance}]
    for time_, _order, idx, kind in events:
        t = all_trades[idx]
        if kind == "ENTRY":
            t["skipped_cap"] = False
            t["aggregate_cap_action"] = "none"
            t["notional_usd"] = 0.0
            if ruined:
                t["risk_dollars"] = 0.0
                t["skipped_ruin"] = True
                continue
            lev = t["leverage_ratio"]
            max_risk_pct = per_position_leverage_cap / lev if lev > 0 else risk_pct
            effective_risk_pct = min(risk_pct, max_risk_pct)
            risk_dollars = balance * effective_risk_pct
            if aggregate_leverage_cap is not None and lev > 0:
                room = aggregate_leverage_cap * balance - open_notional
                if risk_dollars * lev > room + 1e-9:
                    if cap_mode == "shrink" and room > 1e-9:
                        risk_dollars = room / lev
                        t["aggregate_cap_action"] = "shrink"
                        n_shrunk += 1
                    else:
                        risk_dollars = 0.0
                        t["aggregate_cap_action"] = "skip"
                        t["skipped_cap"] = True
                        n_skipped += 1
            t["risk_dollars"] = risk_dollars
            t["effective_risk_pct"] = risk_dollars / balance if balance > 0 else 0.0
            t["leverage_capped"] = effective_risk_pct < risk_pct
            t["skipped_ruin"] = False
            t["notional_usd"] = risk_dollars * lev
            open_notional += t["notional_usd"]
            if balance > 0:
                max_aggregate_leverage = max(max_aggregate_leverage, open_notional / balance)
        else:
            open_notional -= t.get("notional_usd", 0.0)
            if not is_active(t):
                t["dollar_pnl"] = 0.0
            else:
                t["dollar_pnl"] = t["r_net"] * t["risk_dollars"]
                balance += t["dollar_pnl"]
                if balance <= 0:
                    balance = 0.0
                    ruined = True
            t["balance_after"] = balance
            equity_curve.append({"time": str(time_), "balance": balance})

    info = {
        "per_position_leverage_cap": per_position_leverage_cap,
        "aggregate_leverage_cap": aggregate_leverage_cap,
        "cap_mode": cap_mode,
        "n_aggregate_cap_shrunk": n_shrunk,
        "n_aggregate_cap_skipped": n_skipped,
        "max_aggregate_leverage": round(max_aggregate_leverage, 3),
    }
    return balance, equity_curve, info


__all__ = ["DEFAULT_AGGREGATE_LEVERAGE_CAP", "GMO_MAX_LEVERAGE", "is_active", "settle_trades"]
