"""一時検証: SYS-FX012 の H1 継続確認・急変フィルターが、形成途中の H1 足を参照している影響を測る。
causal 版: H1 足は「その5分足の確定時刻までに確定した足」だけを使う（H1 開始+55分 ≤ 5分足の開始時刻）。"""
import sys, json
from pathlib import Path
import pandas as pd, numpy as np
ROOT = Path('/home/user/minmax-fx-day-trading-lab')
sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
import backtest_sysfx018_breakeven_sweep_trainonly as fx018
import price_shock_filter as psf
import backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd as v7

SHIFT = pd.Timedelta(minutes=55)
orig_sim = fx018.simulate_dow_theory_trend
orig_shock = fx018.make_price_shock_check

def run(mode):
    if mode in ('confirm', 'both'):
        def sim(m5, atr_m5, h1, atr_h1, pos, direction, *a, **k):
            hc = h1.copy(); hc.index = hc.index + SHIFT
            return orig_sim(m5, atr_m5, h1, atr_h1, pos, direction, *a, confirm_bars=hc, **k)
        fx018.simulate_dow_theory_trend = sim
    else:
        fx018.simulate_dow_theory_trend = orig_sim
    if mode in ('shock', 'both'):
        def mk(h1_by_pair, atr_by_pair):
            s = psf.build_shock_suppression_series(h1_by_pair, atr_by_pair)
            s.index = s.index + SHIFT
            idx = s.index
            def check(ts):
                p = idx.searchsorted(ts, side='right') - 1
                return False if p < 0 else bool(s.iloc[p])
            return check
        fx018.make_price_shock_check = mk
    else:
        fx018.make_price_shock_check = orig_shock
    out = {}
    for per in ('train', 'validation'):
        s, e = v7.PERIODS[per]
        p = fx018.run_period(1.0, s, e)
        r = np.array([t['r_net'] for t in p['trades']])
        out[per] = {'n': len(r), 'sum_r': round(float(r.sum()), 2), 'mean_r': round(float(r.mean()), 4),
                    'final': round(float(p['equity_curve'][-1]['balance']), 2)}
    return out

res = {m: run(m) for m in ('official', 'confirm', 'shock', 'both')}
print(json.dumps(res, ensure_ascii=False, indent=1))
