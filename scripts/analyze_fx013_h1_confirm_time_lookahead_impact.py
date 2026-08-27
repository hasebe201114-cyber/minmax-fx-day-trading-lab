"""SYS-FX013(EXP-FX000007)REJECT再検証: H1確定時刻先読みバグの影響を定量化する.

## 背景

`research/method-notes/h1_confirm_time_lookahead_impact.json` で発見された
「h1.index[break_idx]をH1バー確定時刻として扱っているが実際は開始時刻」
という先読みバグ(`scripts/backtest_vol_breakout_dow_theory.py`
`simulate_dow_theory_trend()` 345-346行目: `break_time = h1.index[break_idx]`;
`start_time = break_time + pd.Timedelta(minutes=WINDOW_START_MIN)`)は、
SYS-FX012の現行凍結設計(候補①)についてのみ定量化済み。

SYS-FX013(EXP-FX000007)は、この凍結済み設計(候補①=N_BREAKOUT単独+
H1トレンド判定不能除外フィルター)を**一切変更せず**GBP_USD・AUD_USD・
NZD_USDに適用したもの(`scripts/backtest_sysfx013_new_pairs_trainonly.py`)
であり、検出層・エントリー層のコードは完全に同一(通貨とスプレッド仮定のみ
変更)。したがって同一の先読み経路・同一メカニズムが適用される。

## 検証方法

`backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly.find_trades_trendfiltered()`
(共有本番コード、無変更)を、`to_h1`をモンキーパッチして一時的に
「+1時間シフトしたh1」を返すようにした状態で呼び出す(既存の
`analyze_h1_confirm_time_lookahead_impact.py`と同じ自然実験手法)。
モジュールファイル自体は書き換えない(呼び出し前後でオリジナルの
`to_h1`に復元する)。

バグ再現版(パッチなし)が`research/method-notes/sysfx013_new_pairs_trainonly_backtest.json`
の既存値と一致することを確認した上で、シフト修正版と比較する。

出力: research/method-notes/fx013_h1_confirm_time_lookahead_impact.json
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import backtest_vol_continuation_candidates_trendfilter_4pairs_trainonly as engine_mod  # noqa: E402
import derive_vol_breakout_entry_params as h1_mod  # noqa: E402
from backtest_sysfx013_new_pairs_trainonly import (  # noqa: E402
    NEW_PAIR_SPREAD_PIPS, evaluate_single_pair,
)
from backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd import PERIODS  # noqa: E402

CONFIRM_SHIFT = pd.Timedelta(hours=1)
PAIRS = ["GBP_USD", "AUD_USD", "NZD_USD"]


@contextlib.contextmanager
def patched_to_h1(shift: bool):
    """to_h1()を一時的に差し替える(ファイルは書き換えない、プロセス内限定).

    shift=Trueの間、derive_vol_breakout_entry_params.to_h1 (engine_mod内で
    `from derive_vol_breakout_entry_params import to_h1`されて参照される
    ローカル名 `to_h1` も同時に差し替える必要がある)を、
    「本来のto_h1の結果のインデックスを+1時間シフトしたもの」に置き換える。
    """
    if not shift:
        yield
        return
    original = h1_mod.to_h1

    def shifted_to_h1(m5: pd.DataFrame) -> pd.DataFrame:
        h1 = original(m5)
        h1c = h1.copy()
        h1c.index = h1.index + CONFIRM_SHIFT
        return h1c

    h1_mod.to_h1 = shifted_to_h1
    engine_mod.to_h1 = shifted_to_h1
    try:
        yield
    finally:
        h1_mod.to_h1 = original
        engine_mod.to_h1 = original


def run_variant(use_confirm_time: bool) -> dict:
    start, end = PERIODS["train"]
    results = {}
    with patched_to_h1(use_confirm_time):
        for pair in PAIRS:
            results[pair] = evaluate_single_pair(pair, start, end)
    return results


def main() -> int:
    print("=== SYS-FX013: H1確定時刻先読みバグの影響定量化(GBP_USD/AUD_USD/NZD_USD) ===\n")

    print("[1/2] バグ再現版(パッチなし)を実行中...")
    bug = run_variant(use_confirm_time=False)
    for pair in PAIRS:
        r = bug[pair]
        print(f"  [{pair}] n_trades={r['n_trades']}  mean_r_net={r['mean_r_net']}  win_rate={r['win_rate']}")

    official_path = ROOT / "research" / "method-notes" / "sysfx013_new_pairs_trainonly_backtest.json"
    with official_path.open(encoding="utf-8") as f:
        official = json.load(f)["results"]
    reproduction_matches = all(
        bug[pair]["n_trades"] == official[pair]["n_trades"]
        and bug[pair]["mean_r_net"] == official[pair]["mean_r_net"]
        for pair in PAIRS
    )
    print(f"  過去の公式結果との一致: {'OK' if reproduction_matches else 'NG(要確認)'}")

    print("\n[2/2] 確定時刻修正版(h1を+1時間シフト)を実行中...")
    fixed = run_variant(use_confirm_time=True)
    for pair in PAIRS:
        r = fixed[pair]
        print(f"  [{pair}] n_trades={r['n_trades']}  mean_r_net={r['mean_r_net']}  win_rate={r['win_rate']}")

    print("\n=== 比較サマリ (mean_r_net: バグ版 → 修正版) ===")
    comparison = {}
    for pair in PAIRS:
        b, fx = bug[pair], fixed[pair]
        comparison[pair] = {
            "n_trades_bug": b["n_trades"], "n_trades_fixed": fx["n_trades"],
            "mean_r_net_bug": b["mean_r_net"], "mean_r_net_fixed": fx["mean_r_net"],
            "win_rate_bug": b["win_rate"], "win_rate_fixed": fx["win_rate"],
            "verdict_bug": "エッジあり(正)" if (b["mean_r_net"] or 0) > 0 else "エッジなし(負)",
            "verdict_fixed": "エッジあり(正)" if (fx["mean_r_net"] or 0) > 0 else "エッジなし(負)",
        }
        print(f"  {pair}: {b['mean_r_net']} → {fx['mean_r_net']}  "
              f"({comparison[pair]['verdict_bug']} → {comparison[pair]['verdict_fixed']})")

    verdict_flips = any(comparison[p]["verdict_bug"] != comparison[p]["verdict_fixed"] for p in PAIRS)

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "system": "SYS-FX013 (EXP-FX000007)",
        "purpose": "SYS-FX012で確認済みのH1確定時刻先読みバグが、同一エンジンを流用した"
                   "SYS-FX013(非JPY通貨個別評価)にも同一メカニズムで影響しているかを実測で確認する",
        "lookahead_path_found": True,
        "lookahead_path_location": "scripts/backtest_vol_breakout_dow_theory.py:345-346 "
                                    "(simulate_dow_theory_trend、SYS-FX012と共有)",
        "shared_engine_confirmation": (
            "backtest_sysfx013_new_pairs_trainonly.pyはdetect_candidate1・"
            "find_trades_trendfiltered・to_h1をSYS-FX012検証と完全に同一importで"
            "使用しており、通貨ペアとスプレッド仮定以外のロジック変更はゼロ"
        ),
        "reproduction_check": {
            "official": official,
            "reproduced_bug": bug,
            "reproduction_matches_official": reproduction_matches,
        },
        "confirm_time_fixed": fixed,
        "comparison": comparison,
        "reject_verdict_flips": verdict_flips,
        "conclusion": (
            "先読み修正後もmean_r_netの符号(エッジの有無判定)は変化" +
            ("した(要詳細確認)" if verdict_flips else "しなかった") +
            "。SYS-FX013のREJECT根拠(3通貨とも明確にmean_r_netマイナス)は"
            + ("覆らない" if not verdict_flips else "一部影響を受けている可能性がある")
            + "。SYS-FX012本体(候補①)への影響幅は`h1_confirm_time_lookahead_impact.json`"
              "の定量値(Train PF 1.874→1.043、Validation PF 2.318→0.813)を参照。"
              "SYS-FX013はSYS-FX012の凍結設計をそのまま流用しているため、同種・同程度の"
              "パフォーマンス低下(先読み除去でエッジがさらに弱まる方向)が本実測でも"
              "確認できれば、REJECT判定はより強固に裏付けられる(元々マイナスだったmean_r_net"
              "が先読み除去でさらにマイナス方向に振れるか、あるいは横ばいであれば結論は不変)。"
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "fx013_h1_confirm_time_lookahead_impact.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
