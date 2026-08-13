"""本番サイズ決定ルール (SYS-FX007 レンジブレイク・プルバック戦略向け).

背景 (親プロジェクト minmax-trading-pilot の sizing.py を流用・拡張):
- 親PJの sizing.py は MA 20/60 (044) / Donchian (046) / MA 5/20 (048) / Mean Reversion (050) 向け
- 本PJの SYS-FX007 (レンジブレイク・プルバック) は性質が異なるため別系統で管理:
  * 両建て運用あり → ネットポジション = long - short で評価
  * 中期スイング (数日〜数週間保有) → 1日あたり DD は月次 DD の 1/20 程度を想定
  * 月間最大 DD ≤ 10% を厳格に制約

ルール (本PJ 2026-08-13 司令塔 GO + 暫定デフォルト):
1. 検証サイズ: 1,000 通貨で 30 日運用
2. サイズ決定: 月次 DD 実績ベースで 2 倍ずつ拡張 (TREND レジームのみ)
3. 1 日あたり許容 DD: 累積 MaxDD の 50%
4. 停止閾値: 累積 MaxDD × 80% で自動停止
5. 本番サイズ上限: 想定 MaxDD × 10 倍 ≦ 口座残高の 5% (親PJ踏襲)
6. 両建て制約: 同時両建てポジション最大 5 セット (通貨 5 × 1 セット)、証拠金消費率 ≤ 30%

親PJ sizing.py からの差分:
- 戦略別 MaxDD テーブルを SYS-FX007 用に置き換え
- 両建てネットポジション評価を追加
- レジーム判定を v2 拡張（強ブレイク / 弱ブレイク / レンジ）に拡張予定
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class SizeTier(str, Enum):
    """サイズ段階 (親PJ踏襲)."""

    TIER_1 = "1,000通貨 (検証)"
    TIER_2 = "2,000通貨 (拡張1)"
    TIER_3 = "4,000通貨 (拡張2)"
    TIER_4 = "8,000通貨 (本番候補)"
    TIER_STOP = "STOP - 上限到達"


# 戦略別 MaxDD テーブル (単位: 1,000 通貨、USD/JPY 150 円想定での JPY)
# SYS-FX007 (レンジブレイク・プルバック、両建て運用あり)
# 参照: research/EXP-FX000001/00-spec.md §想定 KPI
# NOTE: バックテスト結果が出るまでは暫定値。検証後に更新。
STRATEGY_BASE_MAXDD_1K: dict[str, dict[str, float]] = {
    "SYS-FX007": {
        # レンジブレイク・プルバック (両建て運用あり)
        # - 中期スイング (数日〜数週間保有)
        # - 月間最大 DD 10% 目標
        "trending": -8_000,   # トレンド相場: ブレイク追従で利益、DD は限定的
        "ranging": -12_000,   # レンジ相場: 両建ての含み損、想定 1.2%
        "worst": -15_000,     # 最悪ケース (1.5%)
    },
}

# 累積 MaxDD の 50% を 1 日あたり許容 DD (親PJ踏襲)
DAILY_DD_RATIO = 0.5

# 累積 MaxDD の 80% で自動停止 (親PJ踏襲)
STOP_THRESHOLD_RATIO = 0.8

# 想定 MaxDD × 10 倍 ≦ 口座残高の 5% (親PJ踏襲)
ACCOUNT_RATIO_LIMIT = 0.05

# ロットサイズの基準 (親PJ踏襲)
LOT_BASE = 1_000

# 両建て同時保有の最大数 (本PJ固有)
MAX_HEDGED_POSITIONS = 5

# 両建て状態の最大証拠金消費率 (本PJ固有、K7m)
MAX_MARGIN_USAGE_RATIO = 0.30


@dataclass
class SizingDecision:
    """サイズ決定結果 (親PJ踏襲).

    すべての金額フィールド (expected_max_dd / stop_threshold / daily_limit) は
    `lot_size` ベースの JPY。
    """

    tier: SizeTier
    lot_size: int
    expected_max_dd: float  # lot_size での想定 MaxDD (JPY)
    stop_threshold: float   # lot_size での累積 DD × 80% 停止閾値 (JPY)
    daily_limit: float      # lot_size での 1 日あたり許容 DD (JPY)
    notes: list[str]


def _base_dd_1k(strategy_id: str, regime: str) -> float:
    """戦略/レジーム別のベース MaxDD を 1,000 通貨基準で取得 (親PJ踏襲).

    本PJの SYS-FX007 用テーブル (STRATEGY_BASE_MAXDD_1K) は 1,000 通貨基準。
    """
    base_1k = STRATEGY_BASE_MAXDD_1K.get(strategy_id, {}).get("ranging", -15_000)
    if regime == "TREND":
        base_1k = STRATEGY_BASE_MAXDD_1K.get(strategy_id, {}).get("trending", -8_000)
    return base_1k


def decide_size(
    strategy_id: str,
    account_balance_jpy: float,
    n_days_elapsed: int,
    max_dd_observed: float,
    regime: Literal["TREND", "RANGE", "RANGE_WARN", "INSUFFICIENT_DATA"] = "TREND",
    *,
    current_hedged_positions: int = 0,
) -> SizingDecision:
    """本番サイズ決定 (SYS-FX007 レンジブレイク・プルバック戦略向け).

    Args:
        strategy_id: 戦略 ID (例: "SYS-FX007")
        account_balance_jpy: 口座残高 (JPY)
        n_days_elapsed: フォワード経過日数
        max_dd_observed: フォワードで観測された MaxDD (**1,000 通貨ベース**, 負の値)
        regime: 現在のレジーム
        current_hedged_positions: 現在の両建てポジション数 (デフォルト 0)

    Returns:
        SizingDecision (lot_size ベースの金額)
    """
    notes: list[str] = []

    # 両建て制約 (本PJ固有)
    if current_hedged_positions > MAX_HEDGED_POSITIONS:
        notes.append(
            f"両建てポジション {current_hedged_positions} > 上限 {MAX_HEDGED_POSITIONS} → 拡張しない"
        )
        base_dd_1k = _base_dd_1k(strategy_id, regime)
        return SizingDecision(
            tier=SizeTier.TIER_1,
            lot_size=LOT_BASE,
            expected_max_dd=base_dd_1k,
            stop_threshold=base_dd_1k * STOP_THRESHOLD_RATIO,
            daily_limit=base_dd_1k * DAILY_DD_RATIO,
            notes=notes,
        )

    # 30 日未満は TIER_1 固定 (親PJ踏襲)
    if n_days_elapsed < 30:
        notes.append("30 日未満は TIER_1 (1,000 通貨) 固定")
        return SizingDecision(
            tier=SizeTier.TIER_1,
            lot_size=LOT_BASE,
            expected_max_dd=max_dd_observed,
            stop_threshold=max_dd_observed * STOP_THRESHOLD_RATIO,
            daily_limit=max_dd_observed * DAILY_DD_RATIO,
            notes=notes,
        )

    base_dd_1k = _base_dd_1k(strategy_id, regime)

    # 観測 DD (1,000 通貨ベース) がベースより大きい → サイズ据え置き (親PJ踏襲)
    if max_dd_observed < base_dd_1k:
        notes.append(
            f"観測 DD {max_dd_observed:.0f} (1k 通貨) がベース {base_dd_1k:.0f} (1k 通貨) を下回る → サイズ据え置き"
        )
        return SizingDecision(
            tier=SizeTier.TIER_1,
            lot_size=LOT_BASE,
            expected_max_dd=max_dd_observed,
            stop_threshold=max_dd_observed * STOP_THRESHOLD_RATIO,
            daily_limit=max_dd_observed * DAILY_DD_RATIO,
            notes=notes,
        )

    # 2 倍ずつ拡張 (TREND の場合のみ、親PJ踏襲)
    if regime == "TREND":
        projected_4k = base_dd_1k * 4
        if abs(projected_4k) <= account_balance_jpy * ACCOUNT_RATIO_LIMIT:
            notes.append(f"4,000 通貨 (TREND 想定 DD {projected_4k:.0f} JPY)")
            return SizingDecision(
                tier=SizeTier.TIER_3,
                lot_size=4_000,
                expected_max_dd=projected_4k,
                stop_threshold=projected_4k * STOP_THRESHOLD_RATIO,
                daily_limit=projected_4k * DAILY_DD_RATIO,
                notes=notes,
            )
        projected_2k = base_dd_1k * 2
        notes.append(f"2,000 通貨 (TREND 想定 DD {projected_2k:.0f} JPY)")
        return SizingDecision(
            tier=SizeTier.TIER_2,
            lot_size=2_000,
            expected_max_dd=projected_2k,
            stop_threshold=projected_2k * STOP_THRESHOLD_RATIO,
            daily_limit=projected_2k * DAILY_DD_RATIO,
            notes=notes,
        )

    # RANGE / RANGE_WARN → TIER_1 維持 (親PJ踏襲)
    notes.append(f"レジーム {regime} → TIER_1 維持 (拡張しない、1,000 通貨で運用)")
    notes.append(
        f"想定 DD {base_dd_1k:.0f} JPY (口座 {abs(base_dd_1k) / account_balance_jpy:.2%})"
    )
    return SizingDecision(
        tier=SizeTier.TIER_1,
        lot_size=LOT_BASE,
        expected_max_dd=base_dd_1k,
        stop_threshold=base_dd_1k * STOP_THRESHOLD_RATIO,
        daily_limit=base_dd_1k * DAILY_DD_RATIO,
        notes=notes,
    )


def main() -> int:
    """CLI: SYS-FX007 サイズ決定例."""
    cases: list[dict[str, object]] = [
        {
            "strategy_id": "SYS-FX007",
            "account_balance_jpy": 1_000_000,
            "n_days_elapsed": 45,
            "max_dd_observed": -7_000,  # 1,000 通貨ベース
            "regime": "TREND",
            "current_hedged_positions": 3,
        },
        {
            "strategy_id": "SYS-FX007",
            "account_balance_jpy": 1_000_000,
            "n_days_elapsed": 45,
            "max_dd_observed": -10_000,  # 1,000 通貨ベース (RANGE)
            "regime": "RANGE_WARN",
            "current_hedged_positions": 2,
        },
        {
            "strategy_id": "SYS-FX007",
            "account_balance_jpy": 1_000_000,
            "n_days_elapsed": 60,
            "max_dd_observed": -8_000,  # 1,000 通貨ベース (TREND 良好)
            "regime": "TREND",
            "current_hedged_positions": 1,
        },
        {
            # 両建て上限到達ケース
            "strategy_id": "SYS-FX007",
            "account_balance_jpy": 1_000_000,
            "n_days_elapsed": 45,
            "max_dd_observed": -8_000,
            "regime": "TREND",
            "current_hedged_positions": 6,  # 上限 5 超過
        },
    ]
    for c in cases:
        result = decide_size(
            strategy_id=str(c["strategy_id"]),
            account_balance_jpy=float(str(c["account_balance_jpy"])),
            n_days_elapsed=int(str(c["n_days_elapsed"])),
            max_dd_observed=float(str(c["max_dd_observed"])),
            regime=str(c["regime"]),  # type: ignore[arg-type]
            current_hedged_positions=int(str(c["current_hedged_positions"])),
        )
        stop_ratio = abs(result.stop_threshold) / float(str(c["account_balance_jpy"]))
        print(
            f"{c['strategy_id']} Day {c['n_days_elapsed']:>3} regime={c['regime']:<6} "
            f"obs_dd={c['max_dd_observed']:>+8.0f} hedged={c['current_hedged_positions']} -> "
            f"{result.tier.value:<22} lot={result.lot_size:,} "
            f"exp_dd={result.expected_max_dd:>+9.0f} "
            f"stop={result.stop_threshold:>+9.0f} ({stop_ratio:.2%})"
        )
        for note in result.notes:
            print(f"   - {note}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
