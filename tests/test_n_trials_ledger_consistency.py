"""scripts/calc_dsr_for_ledger.py のリグレッションテスト.

目的:
    - 6 ローダーの sys_id が全て KNOWN_STRATEGY_N_TRIALS に登録されていること
    - リテラル n_trials 値がローダーに残っていないこと (KNOWN_STRATEGY_N_TRIALS 一元管理)
    - breakdown dict が必ず返ること (n_improvement_loops 等の情報源)

起源:
    Phase 2 マージ (2026-08-30) v0.3 必須ゲート化。
    親 PJ の portfolio 台帳 DSR 値が v0.3 厳密 n_trials と整合していることを担保。
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from minmax_fx_dt.statistics.n_trials_counter import KNOWN_STRATEGY_N_TRIALS


# ============================================================
# モジュールロード
# ============================================================

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "calc_dsr_for_ledger.py"
)


def _load_ledger_module():
    """calc_dsr_for_ledger.py を spec でロード (scripts ディレクトリなので
    通常の import では src/ が見えない)."""
    spec = importlib.util.spec_from_file_location("calc_dsr_for_ledger", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# 1. 登録確認
# ============================================================


@pytest.mark.parametrize(
    "sys_id",
    [
        "SYS-FX007",
        "SYS-FX008",
        "SYS-FX009 v2",
        "SYS-FX010",
        "SYS-FX011 v7",
        "SYS-FX011 T-13",
    ],
)
def test_known_loader_sys_ids_are_registered(sys_id: str) -> None:
    """台帳対象 6 戦略が KNOWN_STRATEGY_N_TRIALS に登録されていること."""
    assert sys_id in KNOWN_STRATEGY_N_TRIALS, (
        f"sys_id={sys_id!r} が KNOWN_STRATEGY_N_TRIALS に未登録。"
        "n_trials_counter.py にエントリを追加してください。"
    )
    entry = KNOWN_STRATEGY_N_TRIALS[sys_id]
    assert entry.conservative >= 1, f"conservative n_trials must be >= 1 for {sys_id}"
    assert entry.liberal >= 1, f"liberal n_trials must be >= 1 for {sys_id}"


def test_specific_conservative_values() -> None:
    """戦略別の conservative n_trials 期待値.

    これらは v0.3 厳密カウントの Single Source of Truth.
    値が変わるときは v0.3 spec 改訂とセットで実施すること.
    """
    expected = {
        "SYS-FX007": 6,
        "SYS-FX008": 3,
        "SYS-FX009 v2": 1,
        "SYS-FX010": 5,
        "SYS-FX011 v7": 28,
        "SYS-FX011 T-13": 28,
    }
    for sys_id, want in expected.items():
        got = KNOWN_STRATEGY_N_TRIALS[sys_id].conservative
        assert got == want, f"{sys_id}: conservative n_trials expected {want}, got {got}"


# ============================================================
# 2. リテラル n_trials 値がローダーに残っていないこと
# ============================================================

# 検出パターン: `"n_trials": <リテラル整数>` または `'n_trials': <リテラル整数>`
# - 許容: dict 参照 (`data["n_trials"]` / `data.get("n_trials")` / 変数経由)
# - 拒否: `"n_trials": 7` のような定数直書き
RE_LITERAL_N_TRIALS = re.compile(
    r"""['"]n_trials['"]\s*:\s*(\d+)\b""",
)


def test_loaders_have_no_literal_n_trials() -> None:
    """6 ローダー関数に "n_trials": <整数> の直書きが残っていないこと.

    v0.3 改訂 (M-R2) で n_trials は KNOWN_STRATEGY_N_TRIALS 一元管理.
    リテラルが残っていると新旧 PJ で乖離するリスクがある.
    """
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    # 関数の ``return { ... "n_trials": ... }`` ブロックを抽出して走査.
    # 行ごとに "n_trials": 整数 が直接書かれていれば fail.
    bad_lines: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        # 変数経由は OK
        if 'data["n_trials"]' in line or "data['n_trials']" in line:
            continue
        # 整数直書き検出
        m = RE_LITERAL_N_TRIALS.search(line)
        if m is not None:
            bad_lines.append((lineno, line.strip()))

    assert not bad_lines, (
        "リテラル n_trials 値が calc_dsr_for_ledger.py に残っています。"
        "KNOWN_STRATEGY_N_TRIALS 経由に置き換えてください:\n"
        + "\n".join(f"  L{n}: {s}" for n, s in bad_lines)
    )


# ============================================================
# 3. breakdown dict が必ず返ること
# ============================================================


@pytest.mark.parametrize(
    "loader_name",
    [
        "load_sysfx007_backtest",
        "load_sysfx008_backtest",
        "load_sysfx009_backtest",
        "load_sysfx010_carry",
        "load_sysfx011_v7_ledger",
        "load_sysfx011_t13_backtest",
    ],
)
def test_loaders_return_breakdown(loader_name: str) -> None:
    """各ローダーが 'n_trials_breakdown' dict を返すこと.

    DSR 評価スクリプトは breakdown を根拠に conservative / liberal を切替するため、
    breakdown が無いと監査証跡が追えない.

    注: 実 data ファイルが無い環境では pytest.skip する代わりに、
    生成済み dsr_for_ledger.json の構造を直接検証する形に集約する.
    """
    import json

    json_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "method-notes"
        / "dsr_for_ledger.json"
    )
    if not json_path.exists():
        pytest.skip(f"{json_path.name} 未生成。`python scripts/calc_dsr_for_ledger.py` で生成してください。")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    by_id = {r["sys_id"]: r for r in results}

    # loader_name → sys_id マップ
    name_to_id = {
        "load_sysfx007_backtest": "SYS-FX007",
        "load_sysfx008_backtest": "SYS-FX008",
        "load_sysfx009_backtest": "SYS-FX009 v2",
        "load_sysfx010_carry": "SYS-FX010",
        "load_sysfx011_v7_ledger": "SYS-FX011 v7",
        "load_sysfx011_t13_backtest": "SYS-FX011 T-13",
    }
    sys_id = name_to_id[loader_name]
    assert sys_id in by_id, f"{sys_id} が {json_path.name} に無い"
    entry = by_id[sys_id]
    assert "n_trials_breakdown" in entry, f"{sys_id}: n_trials_breakdown が無い"
    bd = entry["n_trials_breakdown"]
    for key in [
        "n_improvement_loops",
        "n_grid_search_combinations",
        "n_currency_choices",
        "n_period_choices",
        "n_threshold_choices",
        "n_trials_conservative",
        "n_trials_liberal",
    ]:
        assert key in bd, f"{sys_id}: breakdown に {key} が無い"
    assert bd["n_trials_conservative"] == KNOWN_STRATEGY_N_TRIALS[sys_id].conservative
    assert bd["n_trials_liberal"] == KNOWN_STRATEGY_N_TRIALS[sys_id].liberal


# ============================================================
# 4. 内部ヘルパー _trials_breakdown の単体検証
# ============================================================


def test_internal_trials_breakdown_known() -> None:
    """_trials_breakdown() が known sys_id について dict を返すこと."""
    mod = _load_ledger_module()
    out = mod._trials_breakdown("SYS-FX011 T-13")
    assert isinstance(out, dict)
    assert out["n_trials_conservative"] == 28
    assert out["n_trials_liberal"] == 13
    assert out["n_improvement_loops"] == 7
    assert out["n_currency_choices"] == 2
    assert out["n_threshold_choices"] == 2


def test_internal_trials_breakdown_unknown_raises() -> None:
    """_trials_breakdown() が未登録 sys_id で KeyError を送出すること."""
    mod = _load_ledger_module()
    with pytest.raises(KeyError, match="SYS-FX999"):
        mod._trials_breakdown("SYS-FX999")
