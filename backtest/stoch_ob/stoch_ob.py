"""
Stochastic overbought/oversold + Order Block take-profit strategy (XAUUSD M30).

Full specification: DESIGN.md (section numbers referenced below as "spec x.y").

SELL: %K was in the upper zone (>= upper) and turns down  -> sell at the bar close.
BUY : %K was in the lower zone (<= lower) and turns up    -> buy at the bar close.
SL  : beyond the swing just before entry (highest high / lowest low of the last N bars).
TP  : nearest fresh Order Block in the trade direction, or %K reaching the opposite zone,
      whichever comes first.

Everything is decided on closed bars; the trade outcome is simulated from the next bar.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "bos_choch_vp", "data")


# ----------------------------------------------------------------------------- data
def load_2012() -> pd.DataFrame:
    """XAUUSD M15 2012-05 .. 2022-03 (MT5 export, server time = New York + 7h, prices x100)."""
    df = pd.read_csv(os.path.join(DATA, "XAUUSDm15.csv"))
    df["time"] = pd.to_datetime(df["Date"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] / 100.0
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]]


def load_2026() -> pd.DataFrame:
    """XAUUSD M15 2026-03 .. 2026-09 (UTC)."""
    df = pd.read_csv(os.path.join(DATA, "XAUUSD_15m_2026.csv"))
    df["time"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df[["time", "open", "high", "low", "close", "volume"]]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = df.set_index("time").resample(rule, label="left", closed="left")
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.dropna().reset_index()


# --------------------------------------------------------------------------- params
@dataclass
class Params:
    # Stochastic (spec 2.1)
    k_period: int = 14
    d_period: int = 3
    slowing: int = 3
    upper: float = 80.0
    lower: float = 20.0
    # entry (spec 1)
    entry_mode: str = "hook"      # hook | cross | exit_zone
    one_per_zone: bool = True     # max one entry per visit to the zone
    # stop loss (spec 1)
    sl_mode: str = "lookback"     # lookback | fractal
    sl_lookback: int = 10
    min_risk_atr: float = 0.0     # skip if risk < x * ATR(14)
    # take profit (spec 1, 2.3)
    tp_mode: str = "first"        # first | ob_only | stoch_only
    min_rr: float = 1.0           # nearest OB must give >= min_rr * risk (0 = no check)
    ob_too_close: str = "skip"    # skip | next
    ob_pivot: int = 3             # fractal size for swings / structure breaks
    # costs
    spread: float = 0.30          # price units, charged once per trade


# ----------------------------------------------------------------------- indicators
def stochastic(h, l, c, k_period, d_period, slowing):
    """MT5 Stochastic (MODE_SMA, STO_LOWHIGH). Returns (K, D) with NaN during warm-up."""
    hh = pd.Series(h).rolling(k_period).max().to_numpy()
    ll = pd.Series(l).rolling(k_period).min().to_numpy()
    num = pd.Series(c - ll).rolling(slowing).sum().to_numpy()
    den = pd.Series(hh - ll).rolling(slowing).sum().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(den == 0, 100.0, 100.0 * num / den)
    k[np.isnan(num) | np.isnan(den)] = np.nan
    d = pd.Series(k).rolling(d_period).mean().to_numpy()
    return k, d


def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def entry_signals(k, d, p: Params):
    """Raw sell/buy signal at each bar close (spec 1). Zone membership is judged on bar t-1."""
    k1, d1 = np.r_[np.nan, k[:-1]], np.r_[np.nan, d[:-1]]
    in_up, in_dn = k1 >= p.upper, k1 <= p.lower
    if p.entry_mode == "hook":
        sell, buy = in_up & (k < k1), in_dn & (k > k1)
    elif p.entry_mode == "cross":
        sell = in_up & (k1 >= d1) & (k < d)
        buy = in_dn & (k1 <= d1) & (k > d)
    elif p.entry_mode == "exit_zone":
        sell, buy = in_up & (k < p.upper), in_dn & (k > p.lower)
    else:
        raise ValueError(p.entry_mode)
    return sell, buy


# ---------------------------------------------------------------------------- engine
def run(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    tm = df["time"].to_numpy()
    n, q = len(df), p.ob_pivot
    k, d = stochastic(h, l, c, p.k_period, p.d_period, p.slowing)
    a = atr(h, l, c)
    sell_sig, buy_sig = entry_signals(k, d, p)

    last_sh = last_sl = None          # [idx, price, broken]
    piv_hi, piv_lo = [], []           # confirmed pivots (idx, price) for sl_mode="fractal"
    bull_obs, bear_obs = [], []       # [top, bottom, created_bar, touched]
    up_ep = dn_ep = 0                 # zone-visit counters
    used_up = used_dn = -1            # last zone visit that produced an entry
    free_from = 0                     # first bar a new trade may be opened
    trades = []

    for t in range(n):
        # 1. freshness of existing OBs (spec 2.3)
        for ob in bull_obs:
            if not ob[3] and t > ob[2] and l[t] <= ob[0]:
                ob[3] = True
        for ob in bear_obs:
            if not ob[3] and t > ob[2] and h[t] >= ob[1]:
                ob[3] = True
        if t % 500 == 0:
            bull_obs = [ob for ob in bull_obs if not ob[3]]
            bear_obs = [ob for ob in bear_obs if not ob[3]]

        # 2. pivot at t-q is confirmed now (spec 2.2)
        i = t - q
        if i - q >= 0:
            if h[i] == h[i - q:t + 1].max() and h[i] > h[i - q:i].max():
                last_sh = [i, h[i], False]
                piv_hi.append((i, h[i]))
            if l[i] == l[i - q:t + 1].min() and l[i] < l[i - q:i].min():
                last_sl = [i, l[i], False]
                piv_lo.append((i, l[i]))

        # 3. structure breaks -> new OBs (spec 2.3)
        if last_sh is not None and not last_sh[2] and c[t] > last_sh[1]:
            last_sh[2] = True
            lo_bar = last_sh[0] + 1 + int(np.argmin(l[last_sh[0] + 1:t + 1]))
            for j in range(lo_bar, max(lo_bar - 5, last_sh[0] + 1) - 1, -1):   # never before the swing
                if c[j] < o[j]:
                    bull_obs.append([h[j], l[j], t, False])
                    break
        if last_sl is not None and not last_sl[2] and c[t] < last_sl[1]:
            last_sl[2] = True
            hi_bar = last_sl[0] + 1 + int(np.argmax(h[last_sl[0] + 1:t + 1]))
            for j in range(hi_bar, max(hi_bar - 5, last_sl[0] + 1) - 1, -1):
                if c[j] > o[j]:
                    bear_obs.append([h[j], l[j], t, False])
                    break

        # zone-visit bookkeeping (one_per_zone)
        if t >= 1:
            if k[t] >= p.upper and not (k[t - 1] >= p.upper):
                up_ep += 1
            if k[t] <= p.lower and not (k[t - 1] <= p.lower):
                dn_ep += 1

        # 4/5. entries - only the first signal of a zone visit counts (one_per_zone), even if it is
        # skipped by a filter or arrives while a trade is still open (spec 1)
        side = 0
        if sell_sig[t] and not (p.one_per_zone and used_up == up_ep):
            side, used_up = -1, up_ep
        elif buy_sig[t] and not (p.one_per_zone and used_dn == dn_ep):
            side, used_dn = 1, dn_ep
        if side == 0 or t < free_from or t == n - 1:
            continue
        tr = open_trade(t, side, o, h, l, c, k, a, tm, piv_hi, piv_lo, bull_obs, bear_obs, p)
        if tr is None:
            continue
        trades.append(tr)
        free_from = tr["exit_bar"] + 1
    return pd.DataFrame(trades)


def open_trade(t, side, o, h, l, c, k, a, tm, piv_hi, piv_lo, bull_obs, bear_obs, p: Params):
    entry = c[t]
    # ---- stop loss
    lb = h[max(t - p.sl_lookback + 1, 0):t + 1].max() if side == -1 else l[max(t - p.sl_lookback + 1, 0):t + 1].min()
    swing = lb
    if p.sl_mode == "fractal":
        # latest confirmed pivot beyond the entry; no pivot -> keep the lookback swing
        piv = piv_hi if side == -1 else piv_lo
        for _, price in reversed(piv):
            if (side == -1 and price > entry) or (side == 1 and price < entry):
                recent = h[max(t - p.ob_pivot, 0):t + 1] if side == -1 else l[max(t - p.ob_pivot, 0):t + 1]
                swing = max(price, recent.max()) if side == -1 else min(price, recent.min())
                break
    sl = swing + p.spread if side == -1 else swing - p.spread
    risk = abs(entry - sl)
    if risk <= 0 or risk < p.min_risk_atr * a[t]:
        return None

    # ---- take profit at the nearest fresh OB in the trade direction
    tp = None
    if p.tp_mode != "stoch_only":
        if side == -1:
            levels = sorted((ob[0] + p.spread for ob in bull_obs if not ob[3] and ob[0] < entry), reverse=True)
        else:
            levels = sorted(ob[1] - p.spread for ob in bear_obs if not ob[3] and ob[1] > entry)
        if levels:
            need = max(p.min_rr * risk, p.spread + 1e-9)   # TP must be in profit after the spread
            if side * (levels[0] - entry) >= need:
                tp = levels[0]
            elif p.ob_too_close == "skip":
                return None
            else:  # "next": first OB far enough away
                far = [x for x in levels if side * (x - entry) >= need]
                tp = far[0] if far else None
    use_stoch_exit = p.tp_mode in ("first", "stoch_only") or tp is None

    # ---- manage the trade from the next bar (spec 4)
    n = len(c)
    exit_px, reason, j = c[-1], "EOD", n - 1
    for j in range(t + 1, n):
        if side == -1:
            if h[j] >= sl:
                exit_px, reason = max(o[j], sl), "SL"
                break
            if tp is not None and l[j] <= tp:
                exit_px, reason = tp, "TP_OB"
                break
            if use_stoch_exit and k[j] <= p.lower:
                exit_px, reason = c[j], "STOCH"
                break
        else:
            if l[j] <= sl:
                exit_px, reason = min(o[j], sl), "SL"
                break
            if tp is not None and h[j] >= tp:
                exit_px, reason = tp, "TP_OB"
                break
            if use_stoch_exit and k[j] >= p.upper:
                exit_px, reason = c[j], "STOCH"
                break
    else:
        exit_px, reason, j = c[-1], "EOD", n - 1

    r = side * (exit_px - entry) / risk - p.spread / risk
    return dict(side="SELL" if side == -1 else "BUY", entry_time=pd.Timestamp(tm[t]),
                exit_time=pd.Timestamp(tm[j]), entry_bar=t, exit_bar=j, entry=entry, sl=sl,
                tp=np.nan if tp is None else tp, risk=risk, exit=exit_px, reason=reason, r=r,
                k_entry=k[t], rr_target=np.nan if tp is None else side * (tp - entry) / risk)


# ----------------------------------------------------------------------------- stats
def stats(f: pd.DataFrame, years: float | None = None) -> dict:
    if f is None or len(f) == 0:
        return dict(trades=0)
    r = f["r"].to_numpy()
    eq = np.r_[0, np.cumsum(r)]
    gl = -r[r < 0].sum()
    out = dict(trades=len(r), win=round((r > 0).mean() * 100, 1), avgR=round(r.mean(), 3),
               PF=round(r[r > 0].sum() / gl, 2) if gl > 0 else np.inf,
               maxDD=round((np.maximum.accumulate(eq) - eq).max(), 1),
               t=round(r.mean() / r.std(ddof=1) * np.sqrt(len(r)), 2) if len(r) > 2 else np.nan,
               hold_h=round(((f.exit_time - f.entry_time).dt.total_seconds() / 3600).mean(), 1))
    for reason in ("SL", "TP_OB", "STOCH", "EOD"):
        out[f"%{reason}"] = round((f.reason == reason).mean() * 100)
    if years:
        out["per_yr"] = round(len(r) / years, 1)
    return out
