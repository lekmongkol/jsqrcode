"""
Same rules on other markets, M30, 2026-03-26 .. 2026-09-25 (free getdata-finance samples, UTC).
Longer free intraday history for XAGUSD / US30 / NAS100 was not reachable, so this is a
6-month check only.  XAUUSD over the same window is included for comparison.
"""
import os

import pandas as pd

import backtest as b
import filters as fl
from exit_test import stats

OUT = os.path.join(b.HERE, "results")
# symbol -> (file, spread in price units)
MARKETS = {
    "XAUUSD": (None, 0.50),
    "XAGUSD": ("XAGUSD_30m.csv", 0.03),
    "US30": ("US30_30m.csv", 3.0),
    "NAS100": ("NAS100_30m.csv", 2.0),
}
CONFIGS = {
    "no filter, 2R": dict(),
    "H4 trend, 2R + BE at OB": dict(trend="with", ob="be_at_ob"),
    "H4 trend + leg P/D, 2R + BE at OB (best)": dict(trend="with", pd_mode="leg", pd_level=0.5, ob="be_at_ob"),
}


def load(sym, fname):
    if fname is None:
        return b.resample(b.load_2026(), "30min")
    df = pd.read_csv(os.path.join(b.HERE, "data", fname))
    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df[["time", "open", "high", "low", "close", "volume"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, pooled = [], {k: [] for k in CONFIGS}
    for sym, (fname, spread) in MARKETS.items():
        m30 = load(sym, fname)
        m30 = fl.add_htf_cols(fl.add_session_cols(m30, "utc", 30), m30, 30)
        for name, kw in CONFIGS.items():
            p = b.Params(pivot=3, rr=2.0, spread=spread, min_risk_atr=0.5, **kw)
            f = b.to_frame(b.run(m30.assign(htf=m30.htf_struct), p))
            f["symbol"] = sym
            pooled[name].append(f)
            s = stats(f) if len(f) > 1 else dict(trades=len(f))
            sell = f[f.side == "SELL"].r.sum() if len(f) else 0
            buy = f[f.side == "BUY"].r.sum() if len(f) else 0
            rows.append(dict(symbol=sym, config=name, **s, sellR=round(sell, 1), buyR=round(buy, 1)))
    for name, fs in pooled.items():
        f = pd.concat(fs)
        rows.append(dict(symbol="ALL 4", config=name, **stats(f),
                         sellR=round(f[f.side == "SELL"].r.sum(), 1), buyR=round(f[f.side == "BUY"].r.sum(), 1)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "multi_asset_m30_2026.csv"), index=False)
    pd.set_option("display.width", 250)
    print(res[["symbol", "config", "trades", "win", "avgR", "PF", "maxDD", "sellR", "buyR", "t"]].to_string(index=False))


if __name__ == "__main__":
    main()
