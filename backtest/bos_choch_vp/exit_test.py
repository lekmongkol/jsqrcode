"""
Hold longer for bigger R?  Compare exits on the same M30 signals.

Setups:
  A = H4 struct trend + VP-leg premium/discount 50%   (best entry filter so far)
  B = H4 struct trend only                             (more trades)
Exits: fixed TP 2R..5R, trailing stop (ATR chandelier or last-N-bar swing),
       50% at 2R then trail the rest.
"""
import os

import numpy as np
import pandas as pd

import backtest as b
from filters_test import prepare

OUT = os.path.join(b.HERE, "results")
SETUPS = {"A: H4 trend + leg P/D": dict(trend="with", pd_mode="leg", pd_level=0.5),
          "B: H4 trend only": dict(trend="with")}
EXITS = {
    "TP 2R": dict(rr=2.0),
    "TP 2R + BE at OB": dict(rr=2.0, ob="be_at_ob"),
    "TP 3R": dict(rr=3.0),
    "TP 4R": dict(rr=4.0),
    "TP 5R": dict(rr=5.0),
    "trail ATRx3 from 1R": dict(exit="trail_atr", trail_start=1.0, trail_atr=3.0),
    "trail ATRx3 from 2R": dict(exit="trail_atr", trail_start=2.0, trail_atr=3.0),
    "trail ATRx5 from 2R": dict(exit="trail_atr", trail_start=2.0, trail_atr=5.0),
    "trail swing6 from 1R": dict(exit="trail_swing", trail_start=1.0, trail_bars=6),
    "trail swing12 from 2R": dict(exit="trail_swing", trail_start=2.0, trail_bars=12),
    "50% at 2R + trail swing12": dict(exit="partial_trail", partial_r=2.0, trail_start=2.0, trail_bars=12),
}


def stats(f):
    r = f.r.to_numpy()
    eq = np.r_[0, np.cumsum(r)]
    gl = -r[r < 0].sum()
    hold = (f.exit_time - f.entry_time).dt.total_seconds() / 3600
    return dict(trades=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3),
                PF=round(r[r > 0].sum() / gl, 2) if gl else np.inf,
                maxDD=round((np.maximum.accumulate(eq) - eq).max(), 1),
                avgWinR=round(r[r > 0].mean(), 2) if (r > 0).any() else 0, maxR=round(r.max(), 1),
                hold_h=round(hold.median(), 1),
                t=round(r.mean() / r.std(ddof=1) * np.sqrt(len(r)), 2) if len(r) > 2 else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    periods = prepare()
    rows = []
    for sname, sp_kw in SETUPS.items():
        for ename, ex_kw in EXITS.items():
            frames = []
            for label, df, sp in periods:
                p = b.Params(pivot=3, spread=sp, min_risk_atr=0.5, **sp_kw, **ex_kw)
                f = b.to_frame(b.run(df.assign(htf=df.htf_struct), p))
                f["period"] = label
                frames.append(f)
                rows.append(dict(setup=sname, exit=ename, period=label, **stats(f)))
            rows.append(dict(setup=sname, exit=ename, period="ALL", **stats(pd.concat(frames))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "exit_m30_summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    for sname in SETUPS:
        q = res[res.setup == sname]
        pv = q.pivot_table(index="exit", columns="period", values="PF", sort=False)
        allp = q[q.period == "ALL"].set_index("exit")[["trades", "win", "avgR", "PF", "maxDD", "avgWinR",
                                                        "maxR", "hold_h", "t"]]
        print(f"\n=== {sname}")
        print(pv[["2012-2017", "2018-2022", "2026"]].join(allp).to_string())


if __name__ == "__main__":
    main()
