"""
M30: premium / discount entry filter (on top of the H4 trend filter).

SELL only when the entry is in the PREMIUM of the range, BUY only in the DISCOUNT:
  leg         : range = the BOS-high -> CHoCH-low leg used for the volume profile
  h4          : range = last confirmed H4 swing high / swing low (dealing range)
  h4_opposite : literal reverse (SELL in discount / BUY in premium), as a control
pd_level 0.5 = equilibrium, 0.618 = deeper (OTE-style) premium/discount.
Pick on 2012-2017; read 2018-2022 and 2026 as out-of-sample.
"""
import itertools
import os

import pandas as pd

import backtest as b
from be_at_ob_test import stats_df
from filters_test import prepare

OUT = os.path.join(b.HERE, "results")
TRENDS = {"none": ("none", None), "H4 struct with": ("with", "htf_struct"), "H4 EMA50 with": ("with", "htf_ema")}
PD = [("none", 0.5), ("leg", 0.5), ("leg", 0.618), ("h4", 0.5), ("h4", 0.618), ("h4_opposite", 0.5)]


def main():
    os.makedirs(OUT, exist_ok=True)
    periods = prepare()
    rows = []
    for (tname, (mode, col)), (pdm, lvl), piv, ob in itertools.product(
            TRENDS.items(), PD, [3, 5], ["ignore", "be_at_ob"]):
        for label, df, sp in periods:
            d = df.assign(htf=df[col]) if col else df
            p = b.Params(pivot=piv, rr=2.0, ob=ob, trend=mode, pd_mode=pdm, pd_level=lvl,
                         spread=sp, min_risk_atr=0.5)
            f = b.to_frame(b.run(d, p))
            rows.append(dict(trend=tname, pd=f"{pdm} {lvl}" if pdm != "none" else "none",
                             pivot=piv, ob=ob, period=label, **stats_df(f)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "pd_m30_summary.csv"), index=False)
    pv = res.pivot_table(index=["trend", "pd", "pivot", "ob"], columns="period",
                         values=["trades", "win", "PF", "maxDD"])
    pv.columns = [f"{a}_{c}" for a, c in pv.columns]
    cols = [f"{m}_{p}" for p in ("2012-2017", "2018-2022", "2026") for m in ("trades", "win", "PF", "maxDD")]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(pv[cols].round(2).to_string())


if __name__ == "__main__":
    main()
