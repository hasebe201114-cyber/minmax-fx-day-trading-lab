"""EXP-FX000018 / SYS-FX024: 両建て・複数同時保有に対応したグリッド戦略シミュレータ.

既存の`run_period()`系(`backtest_vol_breakout_dow_theory_*`)は「1通貨1ポジション・
1トレード=1リスク単位」を前提としており、以下の点でグリッド戦略に流用できない:

1. 同一通貨で買い・売りを**同時に**保有する(両建て)
2. 同一通貨・同一サイドで**複数段**を同時に保有する
3. エクイティカーブが**確定損益のみ**で構成されており、含み損を抱えたまま積み増す
   構造のDDを大幅に過小評価する(本エンジンは全H4バー終値で**時価評価(MTM)**する)
4. 証拠金消費率(K7m)がポートフォリオ横断で追跡されていない
5. スワップポイント(DS-7)が計上されていない(両建てでは正味計算が必須)

本モジュールは`research/EXP-FX000018/00-spec.md` §3〜§5 の事前登録仕様をそのまま
実装する。**パラメータ(N・k・R)は`10-result/grid_params.json`(フェーズゲート2、
損益非依存の導出)から読み込み、本モジュール内では一切チューニングしない。**

## 主要な設計判断(すべてspecで事前登録済み)

- グリッド生成はH4足、**執行判定はM5足**(同一バー内の順序依存性を構造的に排除)
- M5バー内で損切りと利確が両方成立しうる場合は**損切り優先**(保守側)
- 再アンカー時の既存ポジションの扱いは2候補: `carry_over=False`(G0/MARK方式) /
  `carry_over=True`(G1/持ち越し方式)
- 週内最終H4バーで全決済(週末持ち越し禁止)、グリッドは翌週に持ち越さない
- サイジング: 片側グリッド全段約定→最外段ストップで残高の1.0%を失う全段共通ロット
- 証拠金: 両建て分を**単純合算**(保守側)、分母は**その時点のMTM equity**
- 証拠金ガード: 新規建てで合算方式K7mが30%を超えるならそのエントリーを見送る
"""

from __future__ import annotations

import glob
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from minmax_fx_dt.strategy.indicators import atr as atr_ind

# --- spec §1.1 コストモデル (T-09確定版) ---
SPREAD_PIPS = {"USD_JPY": 0.3, "EUR_JPY": 0.5, "GBP_JPY": 0.7, "AUD_JPY": 0.6}
SLIPPAGE_PIPS_LIMIT = 0.0     # 指値レグ (グリッド新規建て・1刻み利確)
SLIPPAGE_PIPS_STOP = 1.0      # 逆指値レグ (最外段ストップ)
SLIPPAGE_PIPS_MARKET = 0.5    # 成行レグ (週末強制決済・MARK決済)
COMMISSION_RATE_ROUND_TRIP = 0.00004

# --- spec §4/§5 ---
INITIAL_CAPITAL_USD = 1000.0
RISK_PCT_PER_GRID = 0.01
LEVERAGE = 25.0
MARGIN_GUARD_PCT = 30.0
ATR_LENGTH = 14

# --- amendment-01 §3 (司令塔判断「緩和しても良い。リスク対策は必要」への対応、事前登録済み) ---
WEEKEND_GAP_LOSS_BUDGET_PCT = 10.0   # W2: 週末を跨ぐ想定窓開け損失の上限 (MTM equity比%)
LIQUIDATION_MAINTENANCE_PCT = 100.0  # W4: 証拠金維持率がこれを下回ると全ポジション強制決済
ALERT_MAINTENANCE_PCT = 125.0        # W4: これを下回ると新規エントリー停止 (ロスカットアラート)


def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001


def load_m5(pair: str, start: str, end: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "ds-1" / f"ohlcv_{pair}_5min_*.csv")))
    if not files:
        raise FileNotFoundError(f"DS-1のM5 CSVが見つかりません: {pair}")
    frames = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(frames).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    return df[(df.index >= start) & (df.index <= end)]


def to_h4(m5: pd.DataFrame) -> pd.DataFrame:
    agg = [("open", "first"), ("high", "max"), ("low", "min"), ("close", "last")]
    return pd.DataFrame({c: m5[c].resample("4h").agg(a) for c, a in agg}).dropna()


def load_swap_table(pairs: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """DS-7 から {pair: {date_str: (swap_long_jpy, swap_short_jpy)}} を作る (1,000通貨/日あたりJPY)."""
    import json
    with (ROOT / "data" / "curated" / "ds-7.json").open(encoding="utf-8") as f:
        ds7 = json.load(f)
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for pair in pairs:
        series = ds7["pairs"][pair]["daily_series"]
        table[pair] = {row["date"]: (float(row["swap_long_jpy"]), float(row["swap_short_jpy"])) for row in series}
    return table


@dataclass
class Position:
    pair: str
    side: str            # "buy" / "sell"
    gen_id: int
    level_idx: int
    entry_price: float
    tp_price: float
    stop_price: float
    initial_risk: float  # 価格単位のストップまでの距離 (= (N+1-j)*step)
    units: float
    entry_time: pd.Timestamp
    entry_bar: int
    swap_usd: float = 0.0


@dataclass
class Generation:
    gen_id: int
    anchor_time: pd.Timestamp
    center: float
    step: float
    units: float
    buy_levels: np.ndarray
    sell_levels: np.ndarray
    buy_stop: float
    sell_stop: float
    buy_occupied: list[bool] = field(default_factory=list)
    sell_occupied: list[bool] = field(default_factory=list)
    buy_intent: list[bool] = field(default_factory=list)
    sell_intent: list[bool] = field(default_factory=list)
    buy_disabled: bool = False
    sell_disabled: bool = False


def simulate(
    pairs: list[str],
    start: str,
    end: str,
    *,
    n_levels: int,
    grid_step_atr_mult: float,
    reanchor_bars: int,
    carry_over: bool,
    margin_guard: bool = True,
    apply_swap: bool = True,
    initial_capital: float = INITIAL_CAPITAL_USD,
    weekend_carry: bool = False,
    max_hold_h4_bars: int | None = None,
    rel_gap_p99: dict[str, float] | None = None,
    weekend_gap_budget_pct: float = WEEKEND_GAP_LOSS_BUDGET_PCT,
    verbose: bool = True,
) -> dict:
    """グリッド戦略のポートフォリオシミュレーション (4通貨同時、両建て、MTMエクイティ).

    Args:
        carry_over: False = G0(MARK方式、再アンカー時に既存ポジションを時価決済)、
            True = G1(持ち越し方式、既存ポジションは生まれた世代の利確幅・ストップを保持)
        margin_guard: 証拠金ガード(合算方式K7m>30%となる新規建てを見送る)を有効にするか。
            False は spec §5.2 の反実仮想(K7m_unguarded)用。
        weekend_carry: True で週末持ち越しを許可する (amendment-01、司令塔判断 2026-08-28)。
            False は本PJ従来の週末フラット (週内最終H4バーで全決済・グリッド破棄)。
        max_hold_h4_bars: 最大保有期間 (H4本数、amendment-01 §3 W3)。超過分は成行決済
            (outcome=MAX_HOLD)。None で無制限。weekend_carry=True のときは必須。
        rel_gap_p99: 通貨別の週明け相対窓開け99パーセンタイル (amendment-01 §3 W2、
            `10-result/weekend_gap_risk.json`)。weekend_carry=True のときのみ使用。
        weekend_gap_budget_pct: 週末を跨ぐ想定窓開け損失の上限 (MTM equity比%)。
    """
    if weekend_carry and (rel_gap_p99 is None or max_hold_h4_bars is None):
        raise ValueError("weekend_carry=True には rel_gap_p99 と max_hold_h4_bars が必須です "
                         "(amendment-01 §3 W2/W3)")
    raw_m5 = {pair: load_m5(pair, start, end) for pair in pairs}

    # 全通貨で共通のM5タイムラインを先に確定させ、H4足・ATRはその共通軸から生成する
    # (通貨ごとにH4の本数がずれると h4_pos による横断参照が壊れるため)
    index = raw_m5[pairs[0]].index
    for pair in pairs[1:]:
        index = index.intersection(raw_m5[pair].index)
    index = index.sort_values()
    n_bars = len(index)

    m5_by_pair: dict[str, pd.DataFrame] = {p: raw_m5[p].reindex(index) for p in pairs}
    h4_by_pair: dict[str, pd.DataFrame] = {}
    atr_by_pair: dict[str, np.ndarray] = {}
    for pair in pairs:
        h4 = to_h4(m5_by_pair[pair])
        h4_by_pair[pair] = h4
        atr_by_pair[pair] = atr_ind(h4["high"], h4["low"], h4["close"], length=ATR_LENGTH).to_numpy(dtype=float)
    ref_h4_index = h4_by_pair[pairs[0]].index
    for pair in pairs[1:]:
        if not h4_by_pair[pair].index.equals(ref_h4_index):
            raise ValueError(f"H4足の時刻軸が通貨間で一致しません: {pair}")
    if verbose:
        print(f"  共通M5タイムライン: {n_bars:,}bars ({index[0]} 〜 {index[-1]})")

    opens = {p: m5_by_pair[p]["open"].reindex(index).to_numpy(dtype=float) for p in pairs}
    lows = {p: m5_by_pair[p]["low"].reindex(index).to_numpy(dtype=float) for p in pairs}
    highs = {p: m5_by_pair[p]["high"].reindex(index).to_numpy(dtype=float) for p in pairs}
    closes = {p: m5_by_pair[p]["close"].reindex(index).to_numpy(dtype=float) for p in pairs}
    if "USD_JPY" not in pairs:
        raise ValueError("USD/JPYはJPY→USD換算に必須です")
    usdjpy = closes["USD_JPY"]

    # --- H4バーの終了バー(=そのH4足が確定するM5バー)位置を求める ---
    h4_bucket = index.floor("4h")
    is_h4_close = np.zeros(n_bars, dtype=bool)
    is_h4_close[:-1] = h4_bucket[1:] != h4_bucket[:-1]
    is_h4_close[-1] = True
    # M5バー位置 -> そのバーで確定したH4バーの整数位置 (pairごとにH4indexが同一なので共通化)
    h4_pos_of_bar = np.full(n_bars, -1, dtype=int)
    h4_index_ref = h4_by_pair[pairs[0]].index
    h4_lookup = {ts: i for i, ts in enumerate(h4_index_ref)}
    for i in np.flatnonzero(is_h4_close):
        h4_pos_of_bar[i] = h4_lookup.get(h4_bucket[i], -1)

    # --- 週内最終M5バー (次バーとISO週番号が変わるバー) ---
    iso = index.isocalendar()
    week_key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy())
    is_week_last = np.zeros(n_bars, dtype=bool)
    is_week_last[:-1] = week_key[1:] != week_key[:-1]
    is_week_last[-1] = True

    # --- スワップのロールオーバー境界 (JST 06:00 起点の取引日) ---
    trading_day = (index - pd.Timedelta(hours=6)).normalize()
    is_rollover = np.zeros(n_bars, dtype=bool)
    is_rollover[1:] = trading_day[1:] != trading_day[:-1]
    trading_day_str = np.array([d.strftime("%Y-%m-%d") for d in trading_day])
    trading_day_weekday = np.array([d.weekday() for d in trading_day])
    swap_table = load_swap_table(pairs) if apply_swap else {}

    # --- 状態 ---
    balance = float(initial_capital)   # 確定損益ベースの残高 (USD)
    ruined = False
    gens: dict[str, Generation | None] = {p: None for p in pairs}
    open_positions: dict[str, list[Position]] = {p: [] for p in pairs}
    next_gen_id = 0
    bars_since_anchor: dict[str, int] = {p: 0 for p in pairs}

    trades: list[dict] = []
    equity_samples: list[dict] = []
    margin_samples: list[dict] = []
    n_entry_attempts = 0
    n_entry_blocked_margin = 0
    n_entry_blocked_cap = 0
    both_side_stop_events = 0
    max_concurrent = 0
    max_margin_sum_pct = 0.0
    max_margin_max_pct = 0.0
    n_adverse_gap_fills = 0
    adverse_gap_slippage_usd = 0.0
    n_entry_blocked_alert = 0
    n_alert_bars = 0
    min_maintenance_pct = float("inf")
    liquidation_events = 0
    max_weekend_gap_loss_ratio = 0.0
    n_weekend_carry_positions = 0
    max_hold_bars_m5 = (max_hold_h4_bars * 48) if max_hold_h4_bars else None

    def unrealized_usd(bar: int) -> float:
        total = 0.0
        rate = usdjpy[bar]
        for pair, poss in open_positions.items():
            if not poss:
                continue
            price = closes[pair][bar]
            pip = pip_size(pair)
            cost_price = (2 * SPREAD_PIPS[pair] + SLIPPAGE_PIPS_MARKET) * pip
            for pos in poss:
                gross = (price - pos.entry_price) if pos.side == "buy" else (pos.entry_price - price)
                commission = COMMISSION_RATE_ROUND_TRIP * pos.entry_price
                total += (gross - cost_price - commission) * pos.units / rate + pos.swap_usd
        return total

    def margin_pcts(bar: int, equity: float) -> tuple[float, float]:
        """(合算方式, MAX方式) の証拠金消費率 (% of MTM equity)."""
        if equity <= 0:
            return (float("inf"), float("inf"))
        rate = usdjpy[bar]
        total_sum = 0.0
        total_max = 0.0
        for pair, poss in open_positions.items():
            if not poss:
                continue
            price = closes[pair][bar]
            long_m = sum(p.units * price / LEVERAGE / rate for p in poss if p.side == "buy")
            short_m = sum(p.units * price / LEVERAGE / rate for p in poss if p.side == "sell")
            total_sum += long_m + short_m
            total_max += max(long_m, short_m)
        return (total_sum / equity * 100.0, total_max / equity * 100.0)

    def gap_fill_price(pos: Position, bar: int, trigger_price: float, is_stop: bool) -> tuple[float, bool]:
        """amendment-01 §3 W1: 窓開けで水準を飛び越えた場合は寄り値で約定させる.

        Returns: (約定価格, 不利側の窓開けだったか)
        """
        op = opens[pos.pair][bar]
        if not np.isfinite(op):
            return trigger_price, False
        # 逆指値(買い=下方向/売り=上方向)は水準を飛び越えた分だけ不利、
        # 指値利確(買い=上方向/売り=下方向)は飛び越えた分だけ有利に約定する。
        if is_stop:
            gapped = (op <= trigger_price) if pos.side == "buy" else (op >= trigger_price)
        else:
            gapped = (op >= trigger_price) if pos.side == "buy" else (op <= trigger_price)
        if not gapped:
            return trigger_price, False
        return float(op), is_stop

    def close_position(pos: Position, bar: int, exit_price: float, reason: str, slippage_pips: float) -> None:
        nonlocal balance, ruined
        pip = pip_size(pos.pair)
        eff_exit = exit_price - slippage_pips * pip if pos.side == "buy" else exit_price + slippage_pips * pip
        gross = (eff_exit - pos.entry_price) if pos.side == "buy" else (pos.entry_price - eff_exit)
        cost_price = (2 * SPREAD_PIPS[pos.pair] + SLIPPAGE_PIPS_LIMIT + slippage_pips) * pip
        commission_price = COMMISSION_RATE_ROUND_TRIP * pos.entry_price
        net_price = gross - cost_price - commission_price
        rate = usdjpy[bar]
        dollar_pnl = net_price * pos.units / rate + pos.swap_usd
        balance += dollar_pnl
        if balance <= 0:
            balance = 0.0
            ruined = True
        trades.append({
            "pair": pos.pair, "side": pos.side, "gen_id": pos.gen_id, "level_idx": pos.level_idx,
            "entry_time": pos.entry_time, "exit_time": index[bar],
            "entry_price": pos.entry_price, "exit_price": eff_exit,
            "initial_risk": pos.initial_risk, "units": pos.units,
            "outcome": reason,
            "r_gross": gross / pos.initial_risk,
            "cost_r": cost_price / pos.initial_risk,
            "commission_r": commission_price / pos.initial_risk,
            "r_net": net_price / pos.initial_risk,
            "swap_usd": pos.swap_usd,
            "dollar_pnl": dollar_pnl,
            "balance_after": balance,
            "hold_bars_m5": bar - pos.entry_bar,
        })

    def required_margin_usd(bar: int) -> float:
        rate = usdjpy[bar]
        total = 0.0
        for pair, poss in open_positions.items():
            if not poss:
                continue
            price = closes[pair][bar]
            total += sum(pos.units * price / LEVERAGE / rate for pos in poss)
        return total

    def est_weekend_gap_loss(bar: int, rate: float) -> float:
        """amendment-01 §3 W2: 週末を跨ぐ想定窓開け損失 (通貨間は分散効果を認めず単純合算)."""
        total = 0.0
        for pr, poss in open_positions.items():
            if not poss:
                continue
            price = closes[pr][bar]
            net = sum((pos.units if pos.side == "buy" else -pos.units) for pos in poss)
            total += abs(net) * price / rate * (rel_gap_p99 or {}).get(pr, 0.0)
        return total

    def close_all(pair: str, bar: int, reason: str, price: float, slippage: float) -> None:
        for pos in list(open_positions[pair]):
            close_position(pos, bar, price, reason, slippage)
            open_positions[pair].remove(pos)
        g = gens[pair]
        if g is not None:
            g.buy_occupied = [False] * n_levels
            g.sell_occupied = [False] * n_levels

    def make_generation(pair: str, h4_pos: int, bar: int) -> Generation | None:
        nonlocal next_gen_id
        a = atr_by_pair[pair][h4_pos]
        if not np.isfinite(a) or a <= 0:
            return None
        center = float(h4_by_pair[pair]["close"].iloc[h4_pos])
        step = float(a) * grid_step_atr_mult
        # spec §4.1: 片側全段約定→最外段ストップの損失が残高の RISK_PCT_PER_GRID になるロット
        denom = step * n_levels * (n_levels + 1) / 2.0
        units = RISK_PCT_PER_GRID * balance * usdjpy[bar] / denom if denom > 0 else 0.0
        next_gen_id += 1
        return Generation(
            gen_id=next_gen_id,
            anchor_time=index[bar],
            center=center,
            step=step,
            units=units,
            buy_levels=np.array([center - (j + 1) * step for j in range(n_levels)]),
            sell_levels=np.array([center + (j + 1) * step for j in range(n_levels)]),
            buy_stop=center - (n_levels + 1) * step,
            sell_stop=center + (n_levels + 1) * step,
            buy_occupied=[False] * n_levels,
            sell_occupied=[False] * n_levels,
            buy_intent=[False] * n_levels,
            sell_intent=[False] * n_levels,
        )

    # --- メインループ (M5) ---
    for bar in range(n_bars):
        # 1) スワップのロールオーバー
        if apply_swap and is_rollover[bar]:
            mult = 3.0 if trading_day_weekday[bar] == 3 else 1.0  # 水曜→木曜のロールオーバーは3日分
            dstr = trading_day_str[bar]
            rate = usdjpy[bar]
            for pair, poss in open_positions.items():
                if not poss:
                    continue
                row = swap_table[pair].get(dstr)
                if row is None:
                    continue
                swap_long, swap_short = row
                for pos in poss:
                    jpy = (swap_long if pos.side == "buy" else swap_short) * (pos.units / 1000.0) * mult
                    pos.swap_usd += jpy / rate

        # 2) 決済フェーズ (全通貨。証拠金は口座横断のため、決済 → 維持率判定 → 新規建て の順に分ける)
        for pair in pairs:
            g = gens[pair]
            lo, hi = lows[pair][bar], highs[pair][bar]
            if not np.isfinite(lo):
                continue

            # 2a) 損切り優先 (spec §3.4)。窓開けは寄り値約定 (amendment-01 §3 W1)
            stopped_sides = set()
            for pos in list(open_positions[pair]):
                hit = (pos.side == "buy" and lo <= pos.stop_price) or (pos.side == "sell" and hi >= pos.stop_price)
                if hit:
                    fill, adverse = gap_fill_price(pos, bar, pos.stop_price, is_stop=True)
                    if adverse:
                        n_adverse_gap_fills += 1
                        adverse_gap_slippage_usd += abs(fill - pos.stop_price) * pos.units / usdjpy[bar]
                    close_position(pos, bar, fill, "STOP", SLIPPAGE_PIPS_STOP)
                    open_positions[pair].remove(pos)
                    if g is not None and pos.gen_id == g.gen_id:
                        if pos.side == "buy":
                            g.buy_disabled = True
                            g.buy_occupied[pos.level_idx] = False
                        else:
                            g.sell_disabled = True
                            g.sell_occupied[pos.level_idx] = False
                    stopped_sides.add(pos.side)
            if len(stopped_sides) == 2:
                both_side_stop_events += 1

            # 2b) 利確。窓開けは寄り値約定 (有利側に働く、amendment-01 §3 W1)
            for pos in list(open_positions[pair]):
                hit = (pos.side == "buy" and hi >= pos.tp_price) or (pos.side == "sell" and lo <= pos.tp_price)
                if hit:
                    fill, _ = gap_fill_price(pos, bar, pos.tp_price, is_stop=False)
                    close_position(pos, bar, fill, "TP", SLIPPAGE_PIPS_LIMIT)
                    open_positions[pair].remove(pos)
                    if g is not None and pos.gen_id == g.gen_id:
                        if pos.side == "buy":
                            g.buy_occupied[pos.level_idx] = False
                        else:
                            g.sell_occupied[pos.level_idx] = False

            # 2c) 最大保有期間 (amendment-01 §3 W3)
            if max_hold_bars_m5 is not None:
                for pos in list(open_positions[pair]):
                    if bar - pos.entry_bar >= max_hold_bars_m5:
                        close_position(pos, bar, closes[pair][bar], "MAX_HOLD", SLIPPAGE_PIPS_MARKET)
                        open_positions[pair].remove(pos)
                        if g is not None and pos.gen_id == g.gen_id:
                            if pos.side == "buy":
                                g.buy_occupied[pos.level_idx] = False
                            else:
                                g.sell_occupied[pos.level_idx] = False

        # 2d) 証拠金維持率の判定 (amendment-01 §3 W4、口座横断・全M5バー)
        entry_blocked_by_alert = False
        if any(open_positions.values()):
            equity_bar = balance + unrealized_usd(bar)
            req = required_margin_usd(bar)
            if req > 0:
                maintenance = equity_bar / req * 100.0
                min_maintenance_pct = min(min_maintenance_pct, maintenance)
                if maintenance < LIQUIDATION_MAINTENANCE_PCT:
                    liquidation_events += 1
                    for pair in pairs:
                        close_all(pair, bar, "LIQUIDATION", closes[pair][bar], SLIPPAGE_PIPS_MARKET)
                        gens[pair] = None
                        bars_since_anchor[pair] = 0
                    entry_blocked_by_alert = True
                elif maintenance < ALERT_MAINTENANCE_PCT:
                    n_alert_bars += 1
                    entry_blocked_by_alert = True

        # 3) 新規建てフェーズ (全通貨)
        for pair in pairs:
            g = gens[pair]
            lo, hi = lows[pair][bar], highs[pair][bar]
            if not np.isfinite(lo):
                continue
            if g is None or ruined:
                continue
            if entry_blocked_by_alert:
                n_entry_blocked_alert += 1
            else:
                n_buy_open = sum(1 for p in open_positions[pair] if p.side == "buy")
                n_sell_open = sum(1 for p in open_positions[pair] if p.side == "sell")
                for side in ("buy", "sell"):
                    occupied = g.buy_occupied if side == "buy" else g.sell_occupied
                    levels = g.buy_levels if side == "buy" else g.sell_levels
                    stop_px = g.buy_stop if side == "buy" else g.sell_stop
                    intent = g.buy_intent if side == "buy" else g.sell_intent
                    disabled = g.buy_disabled if side == "buy" else g.sell_disabled
                    for j in range(n_levels):
                        touched = (lo <= levels[j]) if side == "buy" else (hi >= levels[j])
                        # 「意図エピソード」の管理: 1回の接触あたり1件だけ試行としてカウントする
                        # (約定できずレベル付近に張り付いた場合、M5バーごとに二重計上しないため)
                        if not touched or occupied[j] or disabled:
                            intent[j] = False
                            continue
                        first_touch = not intent[j]
                        intent[j] = True
                        if first_touch:
                            n_entry_attempts += 1
                        n_side_open = n_buy_open if side == "buy" else n_sell_open
                        if n_side_open >= n_levels:
                            if first_touch:
                                n_entry_blocked_cap += 1
                            continue
                        if margin_guard:
                            equity_now = balance + unrealized_usd(bar)
                            if equity_now <= 0:
                                if first_touch:
                                    n_entry_blocked_margin += 1
                                continue
                            add_margin = g.units * float(levels[j]) / LEVERAGE / usdjpy[bar]
                            cur_sum, _ = margin_pcts(bar, equity_now)
                            if cur_sum + add_margin / equity_now * 100.0 > MARGIN_GUARD_PCT:
                                if first_touch:
                                    n_entry_blocked_margin += 1
                                continue
                        entry_px = float(levels[j])
                        risk = (n_levels + 1 - (j + 1)) * g.step
                        tp = entry_px + g.step if side == "buy" else entry_px - g.step
                        open_positions[pair].append(Position(
                            pair=pair, side=side, gen_id=g.gen_id, level_idx=j,
                            entry_price=entry_px, tp_price=tp, stop_price=stop_px,
                            initial_risk=risk, units=g.units,
                            entry_time=index[bar], entry_bar=bar,
                        ))
                        occupied[j] = True
                        intent[j] = False
                        if side == "buy":
                            n_buy_open += 1
                        else:
                            n_sell_open += 1

        # 4) 週末の扱い
        if is_week_last[bar]:
            if not weekend_carry:
                # 従来: 週内最終バーで全決済・グリッド破棄 (本PJ共通ルール)
                for pair in pairs:
                    close_all(pair, bar, "WEEKEND", closes[pair][bar], SLIPPAGE_PIPS_MARKET)
                    gens[pair] = None
                    bars_since_anchor[pair] = 0
            else:
                # amendment-01 §3 W2: 週末ネットエクスポージャー上限。
                # 想定窓開け損失が MTM equity の budget% を超える分だけ、含み損の大きい
                # ポジションから順に成行決済して持ち越し量を削る。
                equity_ws = balance + unrealized_usd(bar)
                rate = usdjpy[bar]

                budget = weekend_gap_budget_pct / 100.0 * equity_ws
                est = est_weekend_gap_loss(bar, rate)
                if equity_ws > 0:
                    max_weekend_gap_loss_ratio = max(max_weekend_gap_loss_ratio, est / equity_ws)
                while est > budget and any(open_positions.values()):
                    worst_pair, worst_pos, worst_pnl = None, None, None
                    for pr, poss in open_positions.items():
                        price = closes[pr][bar]
                        for pos in poss:
                            gross = ((price - pos.entry_price) if pos.side == "buy"
                                     else (pos.entry_price - price))
                            pnl = gross * pos.units / rate
                            if worst_pnl is None or pnl < worst_pnl:
                                worst_pair, worst_pos, worst_pnl = pr, pos, pnl
                    if worst_pos is None:
                        break
                    g_w = gens[worst_pair]
                    close_position(worst_pos, bar, closes[worst_pair][bar],
                                   "WEEKEND_TRIM", SLIPPAGE_PIPS_MARKET)
                    open_positions[worst_pair].remove(worst_pos)
                    if g_w is not None and worst_pos.gen_id == g_w.gen_id:
                        if worst_pos.side == "buy":
                            g_w.buy_occupied[worst_pos.level_idx] = False
                        else:
                            g_w.sell_occupied[worst_pos.level_idx] = False
                    est = est_weekend_gap_loss(bar, rate)
                n_weekend_carry_positions += sum(len(v) for v in open_positions.values())

        # 5) H4バー確定時: 再アンカー判定 + MTMサンプリング
        if is_h4_close[bar]:
            h4_pos = h4_pos_of_bar[bar]
            # 週末フラット時は週内最終バーでグリッドを破棄するため再アンカーしない。
            # 週末持ち越し時 (weekend_carry) は週境界と無関係に R 本ごとに再アンカーする。
            if h4_pos >= 0 and (weekend_carry or not is_week_last[bar]):
                for pair in pairs:
                    g = gens[pair]
                    if g is None:
                        newg = make_generation(pair, h4_pos, bar)
                        if newg is not None:
                            gens[pair] = newg
                            bars_since_anchor[pair] = 0
                    else:
                        bars_since_anchor[pair] += 1
                        if bars_since_anchor[pair] >= reanchor_bars:
                            if not carry_over:
                                close_all(pair, bar, "MARK", closes[pair][bar], SLIPPAGE_PIPS_MARKET)
                            newg = make_generation(pair, h4_pos, bar)
                            if newg is not None:
                                gens[pair] = newg
                                bars_since_anchor[pair] = 0
            equity = balance + unrealized_usd(bar)
            m_sum, m_max = margin_pcts(bar, equity)
            if np.isfinite(m_sum):
                max_margin_sum_pct = max(max_margin_sum_pct, m_sum)
                max_margin_max_pct = max(max_margin_max_pct, m_max)
            n_open_now = sum(len(v) for v in open_positions.values())
            max_concurrent = max(max_concurrent, n_open_now)
            equity_samples.append({"time": str(index[bar]), "balance": equity, "realized_balance": balance})
            margin_samples.append({
                "time": str(index[bar]), "margin_sum_pct": m_sum if np.isfinite(m_sum) else None,
                "margin_max_pct": m_max if np.isfinite(m_max) else None, "n_open": n_open_now,
            })

    # 期間末に残っているポジションを時価決済 (期間境界のアーティファクト、outcomeを分けて記録)
    last = n_bars - 1
    for pair in pairs:
        close_all(pair, last, "PERIOD_END", closes[pair][last], SLIPPAGE_PIPS_MARKET)

    n_generations = next_gen_id
    return {
        "params": {
            "n_levels": n_levels, "grid_step_atr_mult": grid_step_atr_mult,
            "reanchor_bars": reanchor_bars, "carry_over": carry_over,
            "margin_guard": margin_guard, "apply_swap": apply_swap,
            "initial_capital_usd": initial_capital, "risk_pct_per_grid": RISK_PCT_PER_GRID,
            "leverage": LEVERAGE, "margin_guard_pct": MARGIN_GUARD_PCT,
        },
        "start": start, "end": end, "pairs": pairs,
        "trades": trades,
        "equity_curve": equity_samples,
        "margin_curve": margin_samples,
        "final_balance_usd": balance,
        "ruined": ruined,
        "n_generations": n_generations,
        "n_entry_attempts": n_entry_attempts,
        "n_entry_blocked_margin": n_entry_blocked_margin,
        "n_entry_blocked_cap": n_entry_blocked_cap,
        "guard_block_rate": (n_entry_blocked_margin / n_entry_attempts) if n_entry_attempts else 0.0,
        "both_side_stop_events": both_side_stop_events,
        "max_concurrent_positions": max_concurrent,
        "max_margin_sum_pct": max_margin_sum_pct,
        "max_margin_max_pct": max_margin_max_pct,
        # --- amendment-01 §5.1 の必須報告項目 ---
        "weekend_carry": weekend_carry,
        "max_hold_h4_bars": max_hold_h4_bars,
        "n_adverse_gap_fills": n_adverse_gap_fills,
        "adverse_gap_slippage_usd": adverse_gap_slippage_usd,
        "n_entry_blocked_alert": n_entry_blocked_alert,
        "n_alert_bars": n_alert_bars,
        "min_maintenance_pct": (None if min_maintenance_pct == float("inf")
                                else min_maintenance_pct),
        "liquidation_events": liquidation_events,
        "max_weekend_gap_loss_ratio_pct": max_weekend_gap_loss_ratio * 100.0,
        "n_weekend_carry_position_slots": n_weekend_carry_positions,
    }
