"""
Extra columns for the M30 filters test:
  htf_ema / htf_struct : H4 trend (+1 up, -1 down) known at the close of each bar
  sess_ldn / sess_ny / sess_both : entry-bar close falls in the London / New York session

Clock of the data sets (checked from the daily gold break):
  2012-2022 MT5 export : server time = New York time + 7h (resumes 01:00 every day)
  2026 sample          : UTC
"""
import numpy as np
import pandas as pd

import backtest as b

LDN = (7, 12)   # London local hours [start, end)
NY = (8, 12)    # New York local hours [start, end)


def to_utc(times: pd.Series, clock: str) -> pd.Series:
    if clock == "utc":
        return times.dt.tz_localize("UTC")
    ny = (times - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT",
                                                        nonexistent="shift_forward")
    return ny.dt.tz_convert("UTC")


def add_session_cols(df: pd.DataFrame, clock: str, bar_minutes: int) -> pd.DataFrame:
    close_utc = to_utc(df["time"] + pd.Timedelta(minutes=bar_minutes), clock)
    hl = close_utc.dt.tz_convert("Europe/London").dt.hour
    hn = close_utc.dt.tz_convert("America/New_York").dt.hour
    out = df.copy()
    out["sess_ldn"] = ((hl >= LDN[0]) & (hl < LDN[1])).to_numpy()
    out["sess_ny"] = ((hn >= NY[0]) & (hn < NY[1])).to_numpy()
    out["sess_both"] = out["sess_ldn"] | out["sess_ny"]
    return out


def h4_struct_trend(h4: pd.DataFrame, k: int = 3) -> np.ndarray:
    """Market-structure trend on H4: flips when a close breaks the last confirmed swing."""
    h, l, c = (h4[x].to_numpy(float) for x in ("high", "low", "close"))
    n = len(h4)
    trend = np.zeros(n)
    sh = sl = None
    cur = 0
    for t in range(n):
        i = t - k
        if i - k >= 0:
            if h[i] == h[i - k:t + 1].max():
                sh = h[i]
            if l[i] == l[i - k:t + 1].min():
                sl = l[i]
        if sh is not None and c[t] > sh:
            cur, sh = 1, None
        elif sl is not None and c[t] < sl:
            cur, sl = -1, None
        trend[t] = cur
    return trend


def h4_range(h4: pd.DataFrame, k: int = 3):
    """Last confirmed H4 swing high / low (the dealing range) at the close of each H4 bar."""
    h, l = h4["high"].to_numpy(float), h4["low"].to_numpy(float)
    n = len(h4)
    hi, lo = np.full(n, np.nan), np.full(n, np.nan)
    sh = sl = np.nan
    for t in range(n):
        i = t - k
        if i - k >= 0:
            if h[i] == h[i - k:t + 1].max():
                sh = h[i]
            if l[i] == l[i - k:t + 1].min():
                sl = l[i]
        hi[t], lo[t] = sh, sl
    return hi, lo


def add_htf_cols(df: pd.DataFrame, base: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    h4 = b.resample(base, "4h")
    ema = h4["close"].ewm(span=50, adjust=False).mean()
    h4["htf_ema"] = np.sign(h4["close"] - ema)
    h4["htf_struct"] = h4_struct_trend(h4)
    h4["rng_hi"], h4["rng_lo"] = h4_range(h4)
    h4["known_at"] = h4["time"] + pd.Timedelta(hours=4)          # H4 bar is closed
    left = df.copy()
    left["known_at"] = left["time"] + pd.Timedelta(minutes=bar_minutes)  # signal bar close
    m = pd.merge_asof(left.sort_values("known_at"), h4[["known_at", "htf_ema", "htf_struct", "rng_hi", "rng_lo"]],
                      on="known_at", direction="backward")
    m[["htf_ema", "htf_struct"]] = m[["htf_ema", "htf_struct"]].fillna(0)
    return m.drop(columns="known_at")
