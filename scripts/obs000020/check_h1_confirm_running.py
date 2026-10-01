"""一時検証2: H1 継続確認を「形成途中の H1 高値（5分足ごとの実時間の最高値）」で行う、実時間でも再現できる版。
confirm_bars に5分足そのものを渡す＝各5分足の確定時点までの高値・安値で判定する。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/home/user/minmax-fx-day-trading-lab')
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
import backtest_sysfx018_breakeven_sweep_trainonly as fx018
import price_shock_filter as psf
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7
orig_sim = fx018.simulate_dow_theory_trend
SHIFT = pd.Timedelta(minutes=55)
def sim(m5, atr_m5, h1, atr_h1, pos, direction, *a, **k):
    return orig_sim(m5, atr_m5, h1, atr_h1, pos, direction, *a, confirm_bars=m5, **k)
def mk(h1_by_pair, atr_by_pair):
    s = psf.build_shock_suppression_series(h1_by_pair, atr_by_pair); s.index = s.index + SHIFT; idx = s.index
    def check(ts):
        p = idx.searchsorted(ts, side='right') - 1
        return False if p < 0 else bool(s.iloc[p])
    return check
out = {}
for name, use_shock in (('confirm_m5_running', False), ('confirm_m5_running+shock_causal', True)):
    fx018.simulate_dow_theory_trend = sim
    fx018.make_price_shock_check = mk if use_shock else psf.make_price_shock_check
    out[name] = {}
    for per in ('train', 'validation'):
        s, e = v7.PERIODS[per]
        p = fx018.run_period(1.0, s, e)
        r = np.array([t['r_net'] for t in p['trades']])
        out[name][per] = {'n': len(r), 'sum_r': round(float(r.sum()), 2), 'mean_r': round(float(r.mean()), 4),
                          'final': round(float(p['equity_curve'][-1]['balance']), 2)}
print(json.dumps(out, ensure_ascii=False, indent=1))
