"""
BOS -> CHoCH -> Volume-Profile POC pullback strategy (XAUUSD backtest)

SELL setup (uptrend reversing):
  1. Uptrend with at least one BOS up (close above the last swing high).
  2. CHoCH down: close below the last swing low (the higher low of the uptrend).
  3. Leg = top of the BOS move (highest high since the last BOS) -> the first
     confirmed swing low after the CHoCH.
  4. Volume profile of that leg (tick volume spread over price rows) -> POC.
  5. Wait for price to come back up to the POC; enter SELL at the close of a
     bearish price-action candle (engulfing / pin bar / evening star) whose high
     reached the POC and that closed back below it.
  6. SL = highest high of the pullback (the swing high that touched the POC)
     plus a spread buffer.  TP = RR x risk (default 1:2).
  7. Order-block handling: a fresh (untouched) bullish OB lying between entry and
     TP is treated with one of several policies (ignore / skip / TP at OB /
     move to BE at OB / take 50% at OB).

BUY setup is the exact mirror, done by running the same engine on the
price-inverted series.

All structure is confirmed on closed bars only (pivot needs `n` bars on the
right), so there is no look-ahead. If a bar touches both SL and TP the SL is
assumed to have been hit first.
"""
from __future__ import annotations

import argparse
import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- data
def load_2012() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(HERE, "data", "XAUUSDm15.csv"))
    df["time"] = pd.to_datetime(df["Date"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] / 100.0
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]]


def load_2026() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(HERE, "data", "XAUUSD_15m_2026.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df[["time", "open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("time").resample(rule, label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    }).dropna().reset_index()
    return out


# --------------------------------------------------------------- params / result
@dataclass
class Params:
    pivot: int = 3            # swing = extreme of +-pivot bars
    rows: int = 24            # volume-profile rows (TradingView default 24)
    rr: float = 2.0           # TP = rr x risk
    expiry: int = 100         # bars to wait for the POC pullback + signal
    pa: str = "all"           # all | engulf | pin | star | none (no PA filter)
    ob: str = "ignore"        # ignore | skip | tp_at_ob | be_at_ob | partial
    spread: float = 0.30      # USD round-trip cost
    min_risk_atr: float = 0.0 # skip if risk < x * ATR14
    min_ob_rr: float = 1.0    # tp_at_ob: need at least this R to the OB


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    risk: float
    r: float
    ob_in_path: bool
    poc: float
    leg_hi: float
    leg_lo: float
    pattern: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- helpers
def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def volume_profile_poc(h, l, v, rows):
    lo, hi = l.min(), h.max()
    if hi <= lo:
        return lo
    edges = np.linspace(lo, hi, rows + 1)
    b_lo, b_hi = edges[:-1], edges[1:]
    rng = np.maximum(h - l, 1e-9)
    ov = np.clip(np.minimum(h[:, None], b_hi[None, :]) - np.maximum(l[:, None], b_lo[None, :]), 0, None)
    # doji bars (h == l): put the whole volume in their row
    flat = (h - l) <= 1e-9
    if flat.any():
        idx = np.clip(np.searchsorted(edges, h[flat], side="right") - 1, 0, rows - 1)
        ov[flat] = 0
        ov[np.where(flat)[0], idx] = 1e-9
    prof = (ov / rng[:, None] * v[:, None]).sum(axis=0)
    k = int(prof.argmax())
    return (b_lo[k] + b_hi[k]) / 2


def bearish_pa(o, h, l, c, t, kind):
    """Return pattern name if bar t completes a bearish reversal pattern."""
    rng = h[t] - l[t]
    body = abs(c[t] - o[t])
    if kind in ("all", "engulf") and t >= 1:
        if c[t - 1] > o[t - 1] and c[t] < o[t] and o[t] >= c[t - 1] and c[t] <= o[t - 1] \
                and body > abs(c[t - 1] - o[t - 1]):
            return "engulf"
    if kind in ("all", "pin") and rng > 0:
        upper = h[t] - max(o[t], c[t])
        if upper >= 0.6 * rng and body <= 0.3 * rng:
            return "pin"
    if kind in ("all", "star") and t >= 2:
        b2 = c[t - 2] - o[t - 2]
        b1 = abs(c[t - 1] - o[t - 1])
        if b2 > 0 and b1 <= 0.5 * b2 and c[t] < o[t] and c[t] < o[t - 2] + b2 / 2:
            return "star"
    if kind == "none":
        return "close"
    return None


# ------------------------------------------------------------------------ engine
def run_sell_engine(df: pd.DataFrame, p: Params, side="SELL"):
    """Run the SELL logic.  For BUY call it on the inverted frame."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float); tm = df["time"].to_numpy()
    n = len(df); k = p.pivot
    a = atr(h, l, c)

    last_sh = None   # (idx, price, broken)
    last_sl = None
    trend = 0
    last_bos_bar = None  # bar of the latest BOS-up in the current uptrend
    # bullish OBs (demand) : arrays of [top, bottom, created_bar, touched]
    obs = []
    setup = None
    in_trade_until = -1
    trades: list[Trade] = []

    for t in range(n):
        # ---- update fresh/touched state of OBs (only matters for sell path)
        if obs:
            for ob in obs:
                if not ob[3] and t > ob[2] and l[t] <= ob[0]:
                    ob[3] = True
            if t % 500 == 0:
                obs = [ob for ob in obs if not ob[3]]

        # ---- confirm pivots formed at t-k (needs k closed bars on the right)
        i = t - k
        if i - k >= 0:
            win_h = h[i - k:t + 1]; win_l = l[i - k:t + 1]
            if h[i] == win_h.max() and h[i] > h[i - k:i].max():
                last_sh = [i, h[i], False]
            if l[i] == win_l.min() and l[i] < l[i - k:i].min():
                last_sl = [i, l[i], False]
                # leg-low confirmation for an active setup
                if setup and setup["state"] == "await_low" and i >= setup["choch_bar"] \
                        and l[i] <= l[setup["h_idx"]:t + 1].min():
                    setup["l_idx"] = i
                    setup["L"] = l[i]
                    hh, ll, vv = h[setup["h_idx"]:i + 1], l[setup["h_idx"]:i + 1], v[setup["h_idx"]:i + 1]
                    setup["poc"] = volume_profile_poc(hh, ll, vv, p.rows)
                    setup["state"] = "await_touch"
                    setup["since"] = t

        # ---- structure breaks on close
        if last_sh is not None and not last_sh[2] and c[t] > last_sh[1]:
            last_sh[2] = True
            # bullish OB = last down candle before the up-move that broke structure
            j0 = last_sh[0]
            lo_bar = j0 + int(np.argmin(l[j0:t + 1]))
            for j in range(t, max(lo_bar - 3, 0) - 1, -1):
                if c[j] < o[j]:
                    obs.append([h[j], l[j], t, False])
                    break
            if trend == 1:
                last_bos_bar = t
            else:
                last_bos_bar = None   # this is a CHoCH up / first break
            trend = 1
        if last_sl is not None and not last_sl[2] and c[t] < last_sl[1]:
            last_sl[2] = True
            if trend == 1 and last_bos_bar is not None:
                # BOS ... then CHoCH down -> new sell setup
                hb = last_bos_bar + int(np.argmax(h[last_bos_bar:t + 1]))
                setup = {"state": "await_low", "h_idx": hb, "H": h[hb], "choch_bar": t, "start": t}
            trend = -1
            last_bos_bar = None

        if setup is None:
            continue

        # ---- setup management
        if c[t] > setup["H"]:
            setup = None
            continue
        if setup["state"] == "await_low":
            if t - setup["start"] > p.expiry:
                setup = None
            continue
        if setup["state"] == "await_touch" and t > setup["since"]:
            if l[t] < setup["L"] or t - setup["since"] > p.expiry:
                setup = None
                continue
            poc = setup["poc"]
            if h[t] < poc or c[t] >= poc or t <= in_trade_until:
                continue
            pat = bearish_pa(o, h, l, c, t, p.pa)
            if pat is None:
                continue
            entry = c[t]
            sl = h[setup["l_idx"] + 1:t + 1].max() + p.spread
            risk = sl - entry
            if risk <= 0 or risk < p.min_risk_atr * a[t]:
                continue
            tp = entry - p.rr * risk
            # fresh bullish OB between entry and TP?
            ob_tops = [ob[0] for ob in obs if not ob[3] and ob[2] < t and tp < ob[0] < entry]
            ob_edge = max(ob_tops) if ob_tops else None
            res = simulate(o, h, l, c, tm, t, entry, sl, tp, risk, ob_edge, p)
            if res is None:
                setup = None
                continue
            r, exit_bar, extra = res
            trades.append(Trade(side, pd.Timestamp(tm[t]), pd.Timestamp(tm[exit_bar]), entry, sl, tp,
                                risk, r, ob_edge is not None, poc, setup["H"], setup["L"], pat, extra))
            in_trade_until = exit_bar
            setup = None   # one trade per setup
    return trades


def simulate(o, h, l, c, tm, t, entry, sl, tp, risk, ob_edge, p: Params):
    """Simulate a short from close of bar t.  Returns (R, exit_bar, extra)."""
    n = len(c)
    cost = p.spread / risk
    pol = p.ob if ob_edge is not None else "ignore"
    if pol == "skip":
        return None
    if pol == "tp_at_ob":
        new_tp = ob_edge + p.spread
        if (entry - new_tp) / risk < p.min_ob_rr:
            return None
        tp = new_tp
    stop = sl
    half_done = False
    realized = 0.0
    for j in range(t + 1, n):
        if h[j] >= stop:   # stop first (conservative)
            part = (entry - stop) / risk
            r = realized + (0.5 if half_done else 1.0) * part
            return r - cost, j, {"exit": "SL" if stop == sl else "BE"}
        if pol in ("be_at_ob", "partial") and not half_done and l[j] <= ob_edge:
            if pol == "partial":
                realized += 0.5 * (entry - ob_edge) / risk
            half_done = True
            stop = entry
        if l[j] <= tp:
            part = (entry - tp) / risk
            w = 0.5 if (half_done and pol == "partial") else 1.0
            return realized + w * part - cost, j, {"exit": "TP"}
    # end of data -> close at last price
    part = (entry - c[-1]) / risk
    w = 0.5 if (half_done and pol == "partial") else 1.0
    return realized + w * part - cost, n - 1, {"exit": "EOD"}


def invert(df):
    return pd.DataFrame({"time": df["time"], "open": -df["open"], "high": -df["low"],
                         "low": -df["high"], "close": -df["close"], "volume": df["volume"]})


def run(df, p: Params, sides=("SELL", "BUY")):
    out = []
    if "SELL" in sides:
        out += run_sell_engine(df, p, "SELL")
    if "BUY" in sides:
        out += run_sell_engine(invert(df), p, "BUY")   # prices negated; R is sign-free
    out.sort(key=lambda x: x.entry_time)
    return out


# ------------------------------------------------------------------------ stats
def stats(trades):
    if not trades:
        return dict(trades=0, win=np.nan, avgR=np.nan, totR=0.0, PF=np.nan, maxDD=0.0)
    r = np.array([t.r for t in trades])
    eq = np.cumsum(r)
    dd = (np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max()
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    return dict(trades=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3),
                totR=round(r.sum(), 1), PF=round(gp / gl, 2) if gl > 0 else np.inf, maxDD=round(dd, 1))


def to_frame(trades):
    return pd.DataFrame([{**{k: v for k, v in t.__dict__.items() if k != "extra"}, **t.extra}
                         for t in trades])


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tf", nargs="+", default=["M15", "M30", "H1"], help="timeframes to test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d12 = load_2012()
    d26 = load_2026()
    rules = {"M15": None, "M30": "30min", "H1": "1h"}
    datasets = {tf: (d12, d26) if rules[tf] is None else (resample(d12, rules[tf]), resample(d26, rules[tf]))
                for tf in args.tf}
    spreads = {"old": 0.30, "new": 0.50}   # gold ~1500-2000 vs ~4500 USD
    rows = []
    keep = {}
    grid = itertools.product(args.tf, [3, 5], ["all", "none"],
                             ["ignore", "skip", "tp_at_ob", "be_at_ob", "partial"], [2.0, 3.0])
    for tf, piv, pa, ob, rr in grid:
        old, new = datasets[tf]
        for label, df, sp in (("2012-2017 (IS)", old[old.time < "2018-01-01"], spreads["old"]),
                              ("2018-2022 (OOS)", old[old.time >= "2018-01-01"], spreads["old"]),
                              ("2026 Mar-Sep (OOS)", new, spreads["new"])):
            p = Params(pivot=piv, pa=pa, ob=ob, rr=rr, spread=sp, min_risk_atr=0.5)
            tr = run(df.reset_index(drop=True), p)
            for side in ("SELL", "BUY", "ALL"):
                sub = tr if side == "ALL" else [x for x in tr if x.side == side]
                rows.append(dict(tf=tf, pivot=piv, pa=pa, ob=ob, rr=rr, period=label, side=side, **stats(sub)))
            keep[(tf, piv, pa, ob, rr, label)] = tr
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print(res[res.side == "ALL"].to_string(index=False))

    # trade lists for the base configuration
    for key, tr in keep.items():
        tf, piv, pa, ob, rr, label = key
        if pa == "all" and rr == 2.0 and ob in ("ignore", "tp_at_ob"):
            name = f"trades_{tf}_p{piv}_{ob}_{label.split()[0]}.csv"
            to_frame(tr).to_csv(os.path.join(args.out, name), index=False)


if __name__ == "__main__":
    main()
