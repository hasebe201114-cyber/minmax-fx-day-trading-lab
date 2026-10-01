"""OBS000020 C品質チーム検証(2026-10-01): フォワード台帳(SYS-FX012、cutoff 2026-08-15 06:00 以降)を
実時間版の継続確認・急変フィルターで再計算する。公式台帳ファイルは上書きしない(出力先を差し替える)。

注: Test 期間(2025-12-01〜2026-08-15)のデータは公式フォワードと同じく ATR・ZigZag 等の文脈計算(warmup)
にのみ使われ、Test 期間の取引は生成・集計しない。
使い方: python scripts/obs000020/review_c_forward_causal.py <出力ディレクトリ>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "obs000020"))

import forward_test.run_forward_test_cycle as rf  # noqa: E402
import review_c_causal_reimpl as R  # noqa: E402

ORIG = (rf.simulate_dow_theory_trend, rf.make_price_shock_check, rf.load_m5_forward)


def _load(pair):
    m5 = ORIG[2](pair)
    R._M5_CACHE[pair] = m5
    return m5


def run(cm, sm, outdir: Path):
    rf.load_m5_forward = _load
    rf.simulate_dow_theory_trend = R.make_sim(cm)
    rf.make_price_shock_check = R.make_shock_builder(sm)
    try:
        cfg = dict(rf.STRATEGIES["sysfx012"])
        cfg["out_path"] = outdir / f"forward_{cm}_{sm}.json"
        out = rf.run_forward_cycle(cfg)
    finally:
        rf.simulate_dow_theory_trend, rf.make_price_shock_check, rf.load_m5_forward = ORIG
    bt = out["backtest"]
    closed = [t for t in bt["trades"] if "dollar_pnl" in t]
    r = np.array([t["r_net"] for t in closed]) if closed else np.array([])
    return {"n_closed": bt["n_trades_closed"], "n_open": bt["n_trades_open"], "sum_r": round(float(r.sum()), 2),
            "final_balance": bt["final_balance"], "perm_p_block": bt["perm_p_block"],
            "latest_bar": out["latest_bar_by_pair"]}


def main() -> int:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}
    for cm, sm in (("official", "official"), ("running", "closed"), ("running", "running"),
                   ("closed", "closed"), ("closed_strict", "closed")):
        res[f"confirm={cm}|shock={sm}"] = run(cm, sm, outdir)
        print(f"confirm={cm}|shock={sm}", json.dumps(res[f'confirm={cm}|shock={sm}'], ensure_ascii=False), flush=True)
    (ROOT / "scripts" / "obs000020" / "review_c_forward_causal_result.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
