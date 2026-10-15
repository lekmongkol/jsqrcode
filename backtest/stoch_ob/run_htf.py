"""
Stochastic + OB with the H4 trend filter (spec 9) on XAUUSD M30.

  python3 run_htf.py

The H4 trend is computed on each full data set (2012-2022, 2026) before the in-sample /
out-of-sample split; it only uses H4 bars that closed before each M30 bar's close.
"""
import os

import pandas as pd

import stoch_ob as so

OUT = os.path.join(so.HERE, "results")
SPLIT = pd.Timestamp("2018-01-01")
FILTERS = {"no filter": "none", "H4 structure": "struct", "H4 EMA50": "ema"}
VARIANTS = {
    "default": {},
    "entry: K/D cross": dict(entry_mode="cross"),
    "entry: K leaves zone": dict(entry_mode="exit_zone"),
    "stoch (21,5,5)": dict(k_period=21, d_period=5, slowing=5),
    "zones 90/10": dict(upper=90.0, lower=10.0),
    "SL lookback 20": dict(sl_lookback=20),
    "TP: OB only": dict(tp_mode="ob_only"),
    "TP: Stoch only": dict(tp_mode="stoch_only"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    old = so.resample(so.load_2012(), "30min")
    new = so.resample(so.load_2026(), "30min")
    rows, trades = [], []
    for fname, mode in FILTERS.items():
        hp = so.Params(htf_trend=mode)
        old_h, new_h = so.add_htf(old, hp), so.add_htf(new, hp)
        periods = (("2012-2017", old_h[old_h.time < SPLIT], 0.30), ("2018-2022", old_h[old_h.time >= SPLIT], 0.30),
                   ("2026", new_h, 0.50))
        for vname, kw in VARIANTS.items():
            for label, df, spread in periods:
                df = df.reset_index(drop=True)
                years = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
                f = so.run(df, so.Params(htf_trend=mode, spread=spread, **kw))
                f["filter"], f["variant"], f["period"] = fname, vname, label
                trades.append(f)
                for side in ("ALL", "SELL", "BUY"):
                    sub = f if side == "ALL" or not len(f) else f[f.side == side]
                    rows.append(dict(filter=fname, variant=vname, period=label, side=side, **so.stats(sub, years)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "htf_summary.csv"), index=False)
    pd.concat(trades, ignore_index=True).to_csv(os.path.join(OUT, "htf_trades.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["trades", "per_yr", "win", "avgR", "PF", "maxDD", "t", "%SL", "%TP_OB", "%STOCH"]
    for label in ("2012-2017", "2018-2022", "2026"):
        tag = "IN-SAMPLE" if label == "2012-2017" else "OUT-OF-SAMPLE"
        q = res[(res.period == label) & (res.side == "ALL")].set_index(["variant", "filter"])
        q = q.sort_index(level=0, sort_remaining=False).reindex(columns=cols)
        print(f"\n=== {label} ({tag})")
        print(q.to_string())
    d = res[(res.variant == "default") & (res.filter != "no filter")]
    print("\nDefault with filter, by side:")
    print(d.reindex(columns=["filter", "period", "side"] + cols).to_string(index=False))


if __name__ == "__main__":
    main()
