"""EXP-FX000020 §2.1: 拡張期間の実スプレッドを実測し、保守的コストモデルを導出する.

司令塔の許可条件「保守的コスト仮定を条件に」に対応する。**当時のスプレッドを推測で
決めるのではなく、Dukascopy が配信している BID/ASK 双方のローソクから実測して決める。**

導出（`00-spec.md` §2.1、拡張期間の損益を見る前に固定済み）:

  1. BID/ASK の H1 を、拡張期間(2021-11〜2023-10)と GMO基準期間(2023-11〜2024-10)の
     両方について取得する
  2. 各H1バーで spread_t = ask_close - bid_close を pips 換算し、通貨ごと・期間ごとの
     中央値を取る
  3. era_ratio[pair] = median_spread(拡張期間) / median_spread(GMO基準期間)
  4. CONSERVATIVE_SPREAD[pair] = GMO_SPREAD[pair] * max(1.0, era_ratio[pair])
     ※ max(1.0, ...) により、実測が「当時の方が狭かった」と出ても現行値より安くしない
       （楽観側に倒さない）
  5. 感度分析用に x1.5 / x2.0 の水準も併記する

**損益・勝率等の成績指標は一切参照しない。**

出力: research/EXP-FX000020/10-result/conservative_cost_model.json
"""

from __future__ import annotations

import json
import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

SYMBOL_MAP = {"USD_JPY": "USDJPY", "EUR_JPY": "EURJPY", "GBP_JPY": "GBPJPY", "AUD_JPY": "AUDJPY"}
POINT_DIVISOR = {"USD_JPY": 1000.0, "EUR_JPY": 1000.0, "GBP_JPY": 1000.0, "AUD_JPY": 1000.0}
# T-09 で確定した GMO 現行スプレッド（本PJ全戦略で使用中の値）
GMO_SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6}

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
MAX_RETRIES = 4
WORKERS = 4  # M5取得と並走するため控えめ

ERAS = {
    "extension_2021_11_to_2023_10": [(y, m) for y in (2021, 2022, 2023)
                                     for m in range(1, 13)
                                     if (y, m) >= (2021, 11) and (y, m) <= (2023, 10)],
    "gmo_baseline_2023_11_to_2024_10": [(y, m) for y in (2023, 2024)
                                        for m in range(1, 13)
                                        if (y, m) >= (2023, 11) and (y, m) <= (2024, 10)],
}


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def fetch_month(symbol: str, year: int, month: int, side: str) -> bytes | None:
    url = f"{BASE_URL}/{symbol}/{year}/{month - 1:02d}/{side}_candles_hour_1.bi5"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionResetError, OSError):
            time.sleep(1.5 ** attempt)
    return None


def decode_month(raw: bytes | None, year: int, month: int, divisor: float) -> dict[int, float]:
    """{offset_sec: close} を返す（volume<=0 の停滞バーは除外）."""
    if not raw:
        return {}
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return {}
    out: dict[int, float] = {}
    for i in range(len(data) // 24):
        off, o, c, lo, hi, vol = struct.unpack(">IIIIIf", data[i * 24:(i + 1) * 24])
        if vol <= 0.0:
            continue
        out[off] = c / divisor
    return out


def main() -> int:
    print("=== EXP-FX000020 §2.1: 拡張期間の実スプレッド測定（損益非依存） ===")
    print("Dukascopy の BID/ASK H1 から、当時の実スプレッドを実測する\n")

    results: dict[str, dict] = {}
    for pair, symbol in SYMBOL_MAP.items():
        divisor, pip = POINT_DIVISOR[pair], pip_size(pair)
        results[pair] = {}
        for era, months in ERAS.items():
            tasks = [(y, m, side) for (y, m) in months for side in ("BID", "ASK")]

            def work(t):
                y, m, side = t
                return (y, m, side, decode_month(fetch_month(symbol, y, m, side), y, m, divisor))

            bid: dict[tuple, float] = {}
            ask: dict[tuple, float] = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for y, m, side, closes in ex.map(work, tasks):
                    target = bid if side == "BID" else ask
                    for off, c in closes.items():
                        target[(y, m, off)] = c
            common = sorted(set(bid) & set(ask))
            if not common:
                print(f"  [{pair}] {era}: データ取得失敗")
                continue
            spreads = np.array([(ask[k] - bid[k]) / pip for k in common])
            spreads = spreads[np.isfinite(spreads) & (spreads >= 0)]
            results[pair][era] = {
                "n_bars": int(len(spreads)),
                "median_pips": round(float(np.median(spreads)), 4),
                "p25_pips": round(float(np.percentile(spreads, 25)), 4),
                "p75_pips": round(float(np.percentile(spreads, 75)), 4),
                "mean_pips": round(float(spreads.mean()), 4),
            }
            d = results[pair][era]
            print(f"  [{pair}] {era}: n={d['n_bars']:,}  中央値={d['median_pips']}pips  "
                  f"(p25={d['p25_pips']} / p75={d['p75_pips']})")

    print("\n--- era_ratio と保守的スプレッド（spec §2.1） ---")
    conservative: dict[str, float] = {}
    era_ratios: dict[str, float] = {}
    for pair in SYMBOL_MAP:
        r = results.get(pair, {})
        ext = r.get("extension_2021_11_to_2023_10", {}).get("median_pips")
        gmo = r.get("gmo_baseline_2023_11_to_2024_10", {}).get("median_pips")
        if not ext or not gmo:
            print(f"  [{pair}] 測定不能、現行スプレッドをそのまま使用")
            era_ratios[pair], conservative[pair] = 1.0, GMO_SPREAD_PIPS[pair]
            continue
        ratio = ext / gmo
        era_ratios[pair] = round(ratio, 4)
        conservative[pair] = round(GMO_SPREAD_PIPS[pair] * max(1.0, ratio), 4)
        print(f"  [{pair}] 拡張期間={ext}pips / GMO基準期間={gmo}pips → era_ratio={ratio:.3f} "
              f"→ 保守的スプレッド = {GMO_SPREAD_PIPS[pair]} × max(1.0, {ratio:.3f}) "
              f"= **{conservative[pair]}pips**")

    levels = {
        "base_era_ratio": conservative,
        "sensitivity_x1.5": {p: round(v * 1.5, 4) for p, v in conservative.items()},
        "sensitivity_x2.0": {p: round(v * 2.0, 4) for p, v in conservative.items()},
    }
    print("\n  感度分析の3水準（spec §2.2、いずれも結果を併記し、結論が反転する場合は×2.0を採用）:")
    for name, d in levels.items():
        print(f"    {name}: {d}")

    out_dir = ROOT / "research" / "EXP-FX000020" / "10-result"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conservative_cost_model.json"
    path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exp_id": "EXP-FX000020",
        "spec_ref": "research/EXP-FX000020/00-spec.md §2.1・§2.2",
        "method": {
            "source": "Dukascopy BID/ASK H1 candles (認証不要の公開フィード)",
            "spread_definition": "spread_t = ask_close - bid_close（pips換算）、通貨・期間ごとの中央値",
            "era_ratio": "median_spread(拡張期間) / median_spread(GMO基準期間)",
            "conservative_spread": "GMO_SPREAD × max(1.0, era_ratio)（楽観側に倒さない）",
            "pnl_independence": "損益・勝率等の成績指標は本導出で一切参照していない",
        },
        "gmo_spread_pips": GMO_SPREAD_PIPS,
        "measurements": results,
        "era_ratio": era_ratios,
        "conservative_spread_pips": conservative,
        "sensitivity_levels": levels,
        "caveats": [
            "Dukascopy は GMO とは別ブローカーであり、その BID/ASK スプレッドが GMO の"
            "当時のスプレッドと一致する保証はない。era_ratio は『市況としてスプレッドが"
            "どれだけ広かったか』の代理指標として使う",
            "出来高0のバーを除外しているため、低流動性帯（スプレッドが最も広がる時間帯）が"
            "測定から抜けている可能性がある。この方向のバイアスは楽観側であり、"
            "感度分析(x1.5/x2.0)で補う",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[出力]: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
