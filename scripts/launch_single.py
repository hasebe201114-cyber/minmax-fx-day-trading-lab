"""1 セル (1 通貨 × 1 期間) を detached で起動する補助ランチャー.

Usage:
  python scripts/launch_single.py --pair EUR_USD --period train --preset A1_A2_combined
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_train_val_test.py"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
PERIODS = ["train", "validation", "test"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", required=True, choices=PAIRS)
    p.add_argument("--period", required=True, choices=PERIODS)
    p.add_argument("--preset", default="A1_A2_combined")
    args = p.parse_args()

    log = LOG_DIR / f"tvt_{args.pair}_{args.period}.log"
    cmd = [sys.executable, "-u", str(SCRIPT),
           "--pair", args.pair, "--period", args.period, "--preset", args.preset]
    f = open(log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=f,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    print(f"[START] pair={args.pair} period={args.period} preset={args.preset} PID={proc.pid} log={log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
