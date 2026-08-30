"""SYS-FX022/SYS-FX012 フォワードバー永続化(2026-08-29 修正)の回帰テスト.

【修正した不具合】
`data/curated/ds-1-forward.json` は .gitignore 対象であり、GitHub Actions の
runner は毎回使い捨てである。それにもかかわらず毎時 workflow
(`update-ds1-forward.yml`) は「既存 JSON に追記マージ」するだけで何も
コミットしていなかったため、追記結果が次回実行に一切残らなかった。
結果として週次 cycle が読めるフォワード区間は毎回「直前の lookback 日数分」
だけになり、cutoff (2026-08-15 06:00 JST) 以降の大半が恒久的な空白になっていた
(ledger は n_events_raw=0 と報告し続けたが、実データでの真値は 6 件)。

【修正】
バーの正本を git 管理の追記型 CSV `data/raw/ds-1-forward/*.csv` に置き、
ds-1-forward.json はそこから再生成される派生物として扱う。

本テストは以下を固定する:
1. write_forward_csv → load_forward_csv のラウンドトリップでバーが欠けない
2. ds-1-forward.json が存在しない環境 (= 使い捨て runner) でも
   load_existing() が CSV から全期間を復元する
3. 既存 JSON と CSV の双方にバーがある場合は和集合になる (どちらも失わない)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

fetch_m5 = pytest.importorskip("live_monitor.fetch_m5_ohlcv")


def _records(times: list[str]) -> list[dict]:
    return [
        {"timestamp": t, "open": 150.0, "high": 150.5, "low": 149.5, "close": 150.2}
        for t in times
    ]


@pytest.fixture()
def isolated_paths(tmp_path, monkeypatch):
    """本物の data/ を触らないよう、CSV ディレクトリと JSON を tmp へ差し替える."""
    csv_dir = tmp_path / "ds-1-forward"
    monkeypatch.setattr(fetch_m5, "FORWARD_CSV_DIR", csv_dir)
    monkeypatch.setattr(fetch_m5, "FORWARD_JSON", tmp_path / "ds-1-forward.json")
    return tmp_path


def test_write_then_load_csv_roundtrip(isolated_paths):
    times = ["2026-08-15T06:00:00+09:00", "2026-08-15T06:05:00+09:00"]
    fetch_m5.write_forward_csv("USD_JPY", _records(times))

    loaded = fetch_m5.load_forward_csv("USD_JPY")

    assert [r["timestamp"] for r in loaded] == times
    assert loaded[0]["open"] == 150.0
    assert loaded[0]["close"] == 150.2


def test_load_existing_recovers_all_bars_without_json(isolated_paths):
    """使い捨て runner の再現: JSON が無くても CSV から全期間が戻ること."""
    times = [f"2026-08-{d:02d}T06:00:00+09:00" for d in range(15, 29)]
    fetch_m5.write_forward_csv("USD_JPY", _records(times))
    assert not fetch_m5.FORWARD_JSON.exists()

    out = fetch_m5.load_existing(["USD_JPY"])

    pair = out["pairs"]["USD_JPY"]
    assert pair["n_bars"] == len(times), "CSV にあるバーが JSON 不在時に失われている"
    assert pair["start"] == times[0]
    assert pair["end"] == times[-1]


def test_load_existing_unions_json_and_csv(isolated_paths):
    """JSON と CSV で保持しているバーが違っても、どちらも失われないこと."""
    csv_times = ["2026-08-15T06:00:00+09:00", "2026-08-15T06:05:00+09:00"]
    json_times = ["2026-08-15T06:05:00+09:00", "2026-08-15T06:10:00+09:00"]
    fetch_m5.write_forward_csv("USD_JPY", _records(csv_times))
    fetch_m5.FORWARD_JSON.write_text(
        json.dumps({
            "schema_version": "1.0",
            "interval": "5min",
            "pairs": {"USD_JPY": fetch_m5.pair_dict_from_records(_records(json_times))},
        }),
        encoding="utf-8",
    )

    out = fetch_m5.load_existing(["USD_JPY"])

    got = [r["timestamp"] for r in out["pairs"]["USD_JPY"]["data"]]
    assert got == sorted(set(csv_times) | set(json_times))


def test_merge_pair_is_idempotent():
    """同じバーを二重に取り込んでも増えないこと (毎時実行で重複追記しない)."""
    times = ["2026-08-15T06:00:00+09:00", "2026-08-15T06:05:00+09:00"]
    pair, added = fetch_m5.merge_pair({"data": []}, _records(times))
    assert added == 2

    pair2, added2 = fetch_m5.merge_pair(pair, _records(times))
    assert added2 == 0
    assert pair2["n_bars"] == 2
