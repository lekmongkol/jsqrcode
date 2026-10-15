"""
M30 + H4 trend filter + London/NY session filter.

Rule of the test: pick on 2012-2017 only, then read 2018-2022 and 2026 as out-of-sample.
"""
import itertools
import os

import pandas as pd

import backtest as b
import filters as fl
from be_at_ob_test import stats_df

OUT = os.path.join(b.HERE, "results")
TRENDS = {"none": ("none", None), "H4 EMA50 with": ("with", "htf_ema"),
          "H4 struct with": ("with", "htf_struct"), "H4 struct against": ("against", "htf_struct")}
SESSIONS = ["all", "ldn", "ny", "both"]


def prepare():
    d12, d26 = b.load_2012(), b.load_2026()
    out = {}
    for name, base, clock in (("old", d12, "server"), ("new", d26, "utc")):
        m30 = b.resample(base, "30min")
        m30 = fl.add_session_cols(m30, clock, 30)
        m30 = fl.add_htf_cols(m30, base, 30)
        out[name] = m30
    old, new = out["old"], out["new"]
    return (("2012-2017", old[old.time < "2018-01-01"].reset_index(drop=True), 0.30),
            ("2018-2022", old[old.time >= "2018-01-01"].reset_index(drop=True), 0.30),
            ("2026", new, 0.50))


def main():
    os.makedirs(OUT, exist_ok=True)
    periods = prepare()
    rows = []
    for (tname, (mode, col)), sess, piv, rr, ob in itertools.product(
            TRENDS.items(), SESSIONS, [3, 5], [2.0, 3.0], ["ignore", "be_at_ob"]):
        for label, df, sp in periods:
            d = df.assign(htf=df[col]) if col else df
            p = b.Params(pivot=piv, rr=rr, ob=ob, trend=mode, session=sess, spread=sp, min_risk_atr=0.5)
            f = b.to_frame(b.run(d, p))
            rows.append(dict(trend=tname, session=sess, pivot=piv, rr=rr, ob=ob, period=label, **stats_df(f)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "filters_m30_summary.csv"), index=False)
    pv = res.pivot_table(index=["trend", "session", "pivot", "rr", "ob"], columns="period",
                         values=["trades", "PF", "avgR"])
    pv.columns = [f"{a}_{c}" for a, c in pv.columns]
    cols = [f"{m}_{p}" for p in ("2012-2017", "2018-2022", "2026") for m in ("trades", "PF", "avgR")]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(pv[cols].round(2).to_string())


if __name__ == "__main__":
    main()
