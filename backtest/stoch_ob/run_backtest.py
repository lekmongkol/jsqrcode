"""
Backtest runner for the Stochastic + OB strategy on XAUUSD M30 (spec: DESIGN.md sections 6-7).

  python3 run_backtest.py            # default + one-factor sensitivity table
"""
import os

import pandas as pd

import stoch_ob as so

OUT = os.path.join(so.HERE, "results")

# one-factor-at-a-time changes from the default Params()  (spec 7.2)
SENSITIVITY = {
    "default (14,3,3) 80/20 hook, SL 10 bars, OB>=1R": {},
    "stoch (5,3,3)": dict(k_period=5, d_period=3, slowing=3),
    "stoch (21,5,5)": dict(k_period=21, d_period=5, slowing=5),
    "zones 90/10": dict(upper=90.0, lower=10.0),
    "entry: K/D cross": dict(entry_mode="cross"),
    "entry: K leaves zone": dict(entry_mode="exit_zone"),
    "SL lookback 5": dict(sl_lookback=5),
    "SL lookback 20": dict(sl_lookback=20),
    "SL at last fractal swing": dict(sl_mode="fractal"),
    "min_rr 0 (any OB)": dict(min_rr=0.0),
    "min_rr 2": dict(min_rr=2.0),
    "OB too close -> next OB": dict(ob_too_close="next"),
    "TP: OB only": dict(tp_mode="ob_only"),
    "TP: Stoch only": dict(tp_mode="stoch_only"),
}


def periods():
    old = so.resample(so.load_2012(), "30min")
    new = so.resample(so.load_2026(), "30min")
    split = pd.Timestamp("2018-01-01")
    out = []
    for label, df, spread in (("2012-2017", old[old.time < split], 0.30),
                              ("2018-2022", old[old.time >= split], 0.30),
                              ("2026", new, 0.50)):
        df = df.reset_index(drop=True)
        years = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
        out.append((label, df, spread, years))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = periods()
    rows, all_trades = [], []
    for name, kw in SENSITIVITY.items():
        frames = []
        for label, df, spread, years in data:
            p = so.Params(spread=spread, **kw)
            f = so.run(df, p)
            f["period"], f["config"] = label, name
            frames.append(f)
            rows.append(dict(config=name, period=label, **so.stats(f, years)))
            for side in ("SELL", "BUY"):
                rows.append(dict(config=name, period=f"{label} {side}",
                                 **so.stats(f[f.side == side] if len(f) else f, years)))
        f = pd.concat(frames, ignore_index=True)
        all_trades.append(f)
        rows.append(dict(config=name, period="ALL", **so.stats(f, sum(x[3] for x in data))))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(os.path.join(OUT, "trades.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    main_p = res[res.period.isin(["2012-2017", "2018-2022", "2026"])]
    pf = main_p.pivot_table(index="config", columns="period", values="PF", sort=False)
    allp = res[res.period == "ALL"].set_index("config")[
        ["trades", "per_yr", "win", "avgR", "PF", "maxDD", "t", "hold_h", "%SL", "%TP_OB", "%STOCH"]]
    print(pf.join(allp).to_string())
    d = res[res.config == next(iter(SENSITIVITY))]
    print("\nDefault by period / side:")
    print(d.drop(columns="config").to_string(index=False))


if __name__ == "__main__":
    main()
