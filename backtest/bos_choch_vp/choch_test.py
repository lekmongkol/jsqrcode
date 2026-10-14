"""
Enter right at the CHoCH close instead of waiting for the POC pullback (M30).

  choch      : trade WITH the CHoCH (sell a bearish CHoCH), SL = BOS swing high, TP = rr x risk
  choch_fade : flip side - BUY a bearish CHoCH (bet it is a fake break), SL options:
               mirror (same distance as the sell SL) | bar (below CHoCH bar low) | atr (1 ATR)
'H4 with' = only when the trade direction matches the H4 structure trend.
"""
import os

import pandas as pd

import backtest as b
from exit_test import stats
from filters_test import prepare

OUT = os.path.join(b.HERE, "results")
CONFIGS = {
    "POC pullback (reference) 2R": dict(rr=2.0),
    "CHoCH sell 1R": dict(entry_mode="choch", rr=1.0),
    "CHoCH sell 2R": dict(entry_mode="choch", rr=2.0),
    "CHoCH sell 3R": dict(entry_mode="choch", rr=3.0),
    "fade BUY, SL mirror, 1R": dict(entry_mode="choch_fade", fade_sl="mirror", rr=1.0),
    "fade BUY, SL mirror, 2R": dict(entry_mode="choch_fade", fade_sl="mirror", rr=2.0),
    "fade BUY, SL bar low, 2R": dict(entry_mode="choch_fade", fade_sl="bar", rr=2.0),
    "fade BUY, SL 1 ATR, 1R": dict(entry_mode="choch_fade", fade_sl="atr", rr=1.0),
    "fade BUY, SL 1 ATR, 2R": dict(entry_mode="choch_fade", fade_sl="atr", rr=2.0),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    periods = prepare()
    rows = []
    for trend in ("none", "with"):
        for name, kw in CONFIGS.items():
            frames = []
            for label, df, sp in periods:
                p = b.Params(pivot=3, spread=sp, min_risk_atr=0.5, trend=trend, **kw)
                f = b.to_frame(b.run(df.assign(htf=df.htf_struct), p))
                frames.append(f)
                rows.append(dict(trend=trend, config=name, period=label, **stats(f)))
            rows.append(dict(trend=trend, config=name, period="ALL", **stats(pd.concat(frames))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "choch_m30_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    for trend in ("none", "with"):
        q = res[res.trend == trend]
        pv = q.pivot_table(index="config", columns="period", values="PF", sort=False)
        allp = q[q.period == "ALL"].set_index("config")[["trades", "win", "avgR", "PF", "maxDD", "hold_h", "t"]]
        print(f"\n=== H4 trend filter: {trend}")
        print(pv[["2012-2017", "2018-2022", "2026"]].join(allp).to_string())


if __name__ == "__main__":
    main()
