"""
Does moving SL to breakeven when price reaches the opposing Order Block improve PF?

Compares, on the same signals (PA = all):
  ignore        : hold to SL / TP
  be_at_ob      : SL -> entry when price touches the OB edge (net = -spread)
  be_at_ob+lock : SL -> entry - spread (true breakeven after cost)
and explains the difference on trades whose path had an OB.
"""
import itertools
import os

import numpy as np
import pandas as pd

import backtest as b

OUT = os.path.join(b.HERE, "results")


def main():
    os.makedirs(OUT, exist_ok=True)
    d12, d26 = b.load_2012(), b.load_2026()
    rules = {"M15": None, "M30": "30min", "H1": "1h"}
    rows, detail = [], []
    for tf, piv, rr in itertools.product(rules, [3, 5], [2.0, 3.0]):
        old = d12 if rules[tf] is None else b.resample(d12, rules[tf])
        new = d26 if rules[tf] is None else b.resample(d26, rules[tf])
        periods = (("2012-2017", old[old.time < "2018-01-01"], 0.30),
                   ("2018-2022", old[old.time >= "2018-01-01"], 0.30),
                   ("2026", new, 0.50))
        for label, df, sp in periods:
            df = df.reset_index(drop=True)
            for name, ob, lock in (("ignore", "ignore", 0.0), ("be_at_ob", "be_at_ob", 0.0),
                                   ("be_at_ob+lock", "be_at_ob", sp)):
                p = b.Params(pivot=piv, rr=rr, ob=ob, be_lock=lock, spread=sp, min_risk_atr=0.5)
                f = b.to_frame(b.run(df, p))
                rows.append(dict(tf=tf, pivot=piv, rr=rr, period=label, policy=name, **stats_df(f)))
                if name == "ignore" and len(f):
                    hit = f[f.ob_in_path & f.ob_hit]
                    detail.append(dict(tf=tf, pivot=piv, rr=rr, period=label, trades=len(f),
                                       ob_in_path=int(f.ob_in_path.sum()), reached_ob=len(hit),
                                       then_TP=int((hit.exit == "TP").sum()),
                                       then_SL=int((hit.exit == "SL").sum())))
    res = pd.DataFrame(rows)
    det = pd.DataFrame(detail)
    res.to_csv(os.path.join(OUT, "be_at_ob_summary.csv"), index=False)
    det.to_csv(os.path.join(OUT, "be_at_ob_detail.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    pv = res.pivot_table(index=["tf", "pivot", "rr", "period"], columns="policy", values=["PF", "avgR", "maxDD"])
    print(pv.round(3).to_string())
    print(det.to_string(index=False))


def stats_df(f):
    if f.empty:
        return dict(trades=0, win=np.nan, avgR=np.nan, totR=0.0, PF=np.nan, maxDD=0.0)
    r = f.r.to_numpy()
    eq = np.r_[0, np.cumsum(r)]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3),
                totR=round(r.sum(), 1), PF=round(gp / gl, 2) if gl > 0 else np.inf,
                maxDD=round((np.maximum.accumulate(eq) - eq).max(), 1))


if __name__ == "__main__":
    main()
