"""
Stochastic + OB on higher chart timeframes (H1, H4), with and without a higher-timeframe trend filter.

  python3 run_tf.py            # H1 (trend filter on H4) and H4 (trend filter on D1)

Same engine and rules as M30 (spec sections 1-5, 9); only the chart timeframe and the trend
timeframe change.  Costs stay in USD per trade (0.30 for 2012-2022, 0.50 for 2026), so they are
a smaller fraction of risk on bigger bars.
"""
import os

import pandas as pd

import stoch_ob as so

OUT = os.path.join(so.HERE, "results")
SPLIT = pd.Timestamp("2018-01-01")
CHARTS = {"H1": ("1h", "4h"), "H4": ("4h", "1D")}          # chart rule -> trend-filter rule
FILTERS = {"no filter": "none", "HTF structure": "struct", "HTF EMA50": "ema"}
VARIANTS = {
    "default": {},
    "stoch (5,3,3)": dict(k_period=5, d_period=3, slowing=3),
    "stoch (21,5,5)": dict(k_period=21, d_period=5, slowing=5),
    "zones 90/10": dict(upper=90.0, lower=10.0),
    "entry: K/D cross": dict(entry_mode="cross"),
    "entry: K leaves zone": dict(entry_mode="exit_zone"),
    "TP: OB only": dict(tp_mode="ob_only"),
    "TP: Stoch only": dict(tp_mode="stoch_only"),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    base_old, base_new = so.load_2012(), so.load_2026()
    rows, trades = [], []
    for chart, (rule, htf_rule) in CHARTS.items():
        old, new = so.resample(base_old, rule), so.resample(base_new, rule)
        for fname, mode in FILTERS.items():
            hp = so.Params(htf_trend=mode, htf_rule=htf_rule)
            old_h, new_h = so.add_htf(old, hp, bar=rule), so.add_htf(new, hp, bar=rule)
            periods = (("2012-2017", old_h[old_h.time < SPLIT], 0.30),
                       ("2018-2022", old_h[old_h.time >= SPLIT], 0.30), ("2026", new_h, 0.50))
            for vname, kw in VARIANTS.items():
                for label, df, spread in periods:
                    df = df.reset_index(drop=True)
                    years = (df.time.iloc[-1] - df.time.iloc[0]).days / 365.25
                    f = so.run(df, so.Params(htf_trend=mode, htf_rule=htf_rule, spread=spread, **kw))
                    f["chart"], f["filter"], f["variant"], f["period"] = chart, fname, vname, label
                    trades.append(f)
                    for side in ("ALL", "SELL", "BUY"):
                        sub = f if side == "ALL" or not len(f) else f[f.side == side]
                        rows.append(dict(chart=chart, filter=fname, variant=vname, period=label, side=side,
                                         **so.stats(sub, years)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "tf_summary.csv"), index=False)
    pd.concat(trades, ignore_index=True).to_csv(os.path.join(OUT, "tf_trades.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    cols = ["trades", "per_yr", "win", "avgR", "PF", "maxDD", "t", "%SL", "%TP_OB", "%STOCH"]
    for chart in CHARTS:
        for label in ("2012-2017", "2018-2022", "2026"):
            tag = "IN-SAMPLE" if label == "2012-2017" else "OUT-OF-SAMPLE"
            q = res[(res.chart == chart) & (res.period == label) & (res.side == "ALL")]
            q = q.set_index(["variant", "filter"]).reindex(columns=cols)
            print(f"\n=== {chart} {label} ({tag})")
            print(q.to_string())


if __name__ == "__main__":
    main()
