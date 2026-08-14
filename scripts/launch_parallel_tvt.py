"""15 セル (5 通貨 × 3 期間) を並列起動するランチャー.

各セルを detached プロセスとして起動し、launcher は即終了する。
各プロセスの stdout/stderr は logs/tvt_{pair}_{period}.log に保存。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_train_val_test.py"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

PAIRS = ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_USD"]
ALL_PERIODS = ["train", "validation", "test"]
PRESET = "A1_A2_combined"

# Windows で親プロセスから完全に切り離す
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def launch(pair: str, period: str) -> tuple[int, Path]:
    log = LOG_DIR / f"tvt_{pair}_{period}.log"
    f = open(log, "w", encoding="utf-8")
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--pair", pair, "--period", period, "--preset", PRESET,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=f,
        stderr=subprocess.STDOUT,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    return proc.pid, log


def main() -> int:
    p = argparse.ArgumentParser(description="5 通貨 × 期間 を並列起動")
    p.add_argument("--period", choices=ALL_PERIODS, help="単一期間のみ (省略時は 15 セル全起動)")
    args = p.parse_args()

    periods = [args.period] if args.period else ALL_PERIODS
    total = len(PAIRS) * len(periods)

    print(f"=== {total} セル並列起動 (preset={PRESET}, period={periods}) ===")
    print(f"ログ先: {LOG_DIR}")
    print(f"出力先: {ROOT / 'research' / 'EXP-FX000001' / '10-result' / 'train_val_test'}")
    print()

    started = []
    for p_sym in PAIRS:
        for per in periods:
            pid, log = launch(p_sym, per)
            started.append((p_sym, per, pid, log))
            print(f"  [START] {p_sym:<10} {per:<12} PID={pid:<8} {log.name}")

    print()
    print(f"=== {len(started)} プロセス起動完了。launcher 終了。各プロセスは独立に実行。 ===")
    print()
    print("状態確認コマンド (PowerShell):")
    print("  Get-Process python -ErrorAction SilentlyContinue | Where-Object Id -in (<pid 群>)")
    print()
    print("完了ファイル確認:")
    print("  Get-ChildItem research/EXP-FX000001/10-result/train_val_test/ -Filter 'tvt_*.json'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
