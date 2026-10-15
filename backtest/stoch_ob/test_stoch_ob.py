"""
Unit tests for stoch_ob.py (spec: DESIGN.md section 8).
Run:  python3 test_stoch_ob.py      (or: python3 -m pytest test_stoch_ob.py)
"""
import numpy as np
import pandas as pd

import stoch_ob as so


def random_bars(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    c = 1500 + np.cumsum(rng.normal(0, 1.5, n))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + rng.exponential(0.8, n)
    l = np.minimum(o, c) - rng.exponential(0.8, n)
    t = pd.date_range("2015-01-01", periods=n, freq="30min")
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": 1.0})


# ------------------------------------------------------------------ indicator formula
def test_stochastic_matches_hand_calculation():
    df = random_bars(200)
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    k, d = so.stochastic(h, l, c, 14, 3, 3)
    for t in (20, 57, 133, 199):
        num = den = 0.0
        for s in range(t - 2, t + 1):                       # slowing = 3
            hh, ll = h[s - 13:s + 1].max(), l[s - 13:s + 1].min()   # k = 14
            num += c[s] - ll
            den += hh - ll
        assert abs(k[t] - 100 * num / den) < 1e-9
        assert abs(d[t] - np.mean(k[t - 2:t + 1])) < 1e-9
    # warm-up: first valid K at index k+slowing-2 = 15, first D at 17
    assert np.isnan(k[14]) and not np.isnan(k[15])
    assert np.isnan(d[16]) and not np.isnan(d[17])


def test_stochastic_flat_market_is_100():
    h = l = c = np.full(30, 10.0)
    k, _ = so.stochastic(h, l, c, 5, 3, 3)
    assert k[10] == 100.0


# -------------------------------------------------------------------------- signals
def test_entry_signal_modes():
    k = np.array([50, 85, 90, 88, 70, 15, 10, 12, 30.0])
    d = np.array([50, 60, 80, 89, 82, 40, 18, 11, 20.0])
    p = so.Params(entry_mode="hook")
    s, b = so.entry_signals(k, d, p)
    # raw signals: every down-tick whose previous bar is in the zone (one_per_zone is applied in run())
    assert list(np.where(s)[0]) == [3, 4]        # 90 -> 88, 88 -> 70
    assert list(np.where(b)[0]) == [7, 8]        # 10 -> 12, 12 -> 30
    p.entry_mode = "exit_zone"
    s, b = so.entry_signals(k, d, p)
    assert list(np.where(s)[0]) == [4] and list(np.where(b)[0]) == [8]
    p.entry_mode = "cross"
    s, b = so.entry_signals(k, d, p)
    assert list(np.where(s)[0]) == [3]           # K 90>=D 80 then K 88 < D 89
    assert list(np.where(b)[0]) == [7]           # K 10<=D 18 then K 12 > D 11


# -------------------------------------------------------------------- trade handling
def _arrays(rows):
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def _open(rows, side, k, p, bull=(), bear=()):
    o, h, l, c = _arrays(rows)
    n = len(c)
    tm = pd.date_range("2020-01-01", periods=n, freq="30min").to_numpy()
    a = np.full(n, 1.0)
    return so.open_trade(4, side, o, h, l, c, np.asarray(k, float), a, tm, [], [],
                         [list(x) for x in bull], [list(x) for x in bear], p)


BASE = [  # open, high, low, close ; entry bar = 4 (close 100)
    (98, 99, 97, 98), (98, 101, 97, 100), (100, 103, 99, 102), (102, 105, 101, 101), (101, 102, 99, 100),
]


def test_sell_sl_lookback_and_r():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0, tp_mode="stoch_only")
    rows = BASE + [(100, 106, 99, 104)]                 # bar 5 hits SL
    tr = _open(rows, -1, [50] * 6, p)
    assert tr["sl"] == 105.5                            # max high of bars 2..4 = 105 + spread
    assert tr["reason"] == "SL" and tr["exit"] == 105.5
    assert abs(tr["r"] - (-(105.5 - 100) / 5.5 - 0.5 / 5.5)) < 1e-12


def test_sl_gap_fills_at_open():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0, tp_mode="stoch_only")
    rows = BASE + [(107, 108, 106, 107)]                # gaps above SL
    tr = _open(rows, -1, [50] * 6, p)
    assert tr["exit"] == 107


def test_tp_at_nearest_fresh_ob_and_sl_first():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=1.0)
    bull = [(90, 88, 0, False), (93, 91, 0, False), (95, 94, 0, True)]   # last one already touched
    rows = BASE + [(100, 101, 92, 93)]
    tr = _open(rows, -1, [50] * 6, p, bull=bull)
    assert tr["tp"] == 93.5 and tr["reason"] == "TP_OB" and tr["exit"] == 93.5
    rows = BASE + [(100, 106, 92, 93)]                   # same bar touches SL and TP -> SL
    tr = _open(rows, -1, [50] * 6, p, bull=bull)
    assert tr["reason"] == "SL"


def test_ob_too_close_skip_and_next():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=1.0, ob_too_close="skip")
    bull = [(97, 96, 0, False), (90, 88, 0, False)]     # 97.5 gives 2.5 < 5.5 risk
    assert _open(BASE + [(100, 101, 99, 100)], -1, [50] * 6, p, bull=bull) is None
    p.ob_too_close = "next"
    tr = _open(BASE + [(100, 101, 89, 90)], -1, [50] * 6, p, bull=bull)
    assert tr["tp"] == 90.5 and tr["reason"] == "TP_OB"


def test_stoch_exit_and_tp_modes():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0)
    k = [50, 50, 50, 50, 50, 15]
    tr = _open(BASE + [(100, 101, 98, 99)], -1, k, p)             # no OB -> exit on K<=20
    assert tr["reason"] == "STOCH" and tr["exit"] == 99
    bull = [(80, 79, 0, False)]
    p.tp_mode = "ob_only"
    tr = _open(BASE + [(100, 101, 98, 99)], -1, k, p, bull=bull)  # OB exists -> ignore stoch
    assert tr["reason"] == "EOD"


def test_buy_side_mirror():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0, tp_mode="stoch_only")
    rows = [(102, 103, 101, 102), (102, 103, 99, 100), (100, 101, 97, 98), (98, 99, 95, 99), (99, 101, 98, 100),
            (100, 101, 94, 96)]
    tr = _open(rows, 1, [50] * 6, p)
    assert tr["sl"] == 94.5 and tr["reason"] == "SL" and tr["exit"] == 94.5


# ------------------------------------------------------------------ whole-engine checks
def test_no_lookahead():
    """Trades closed before bar m must be identical when the data after m is removed."""
    df = random_bars(3000, seed=7)
    p = so.Params()
    full = so.run(df, p)
    m = 2000
    cut = so.run(df.iloc[:m].reset_index(drop=True), p)
    a = full[full.exit_bar < m - 1].reset_index(drop=True)
    b = cut.iloc[:len(a)].reset_index(drop=True)
    assert len(a) > 20
    pd.testing.assert_frame_equal(a, b)


def test_buy_sell_symmetry():
    """Negating prices must turn every SELL into an identical BUY (and vice versa)."""
    df = random_bars(3000, seed=3)
    inv = df.copy()
    inv["open"], inv["high"], inv["low"], inv["close"] = -df.open, -df.low, -df.high, -df.close
    for mode in ("hook", "cross", "exit_zone"):
        p = so.Params(entry_mode=mode)
        a, b = so.run(df, p), so.run(inv, p)
        assert len(a) == len(b) > 20
        assert (a.side.map({"SELL": "BUY", "BUY": "SELL"}) == b.side).all()
        assert np.allclose(a.r, b.r) and (a.reason == b.reason).all()


def test_one_position_at_a_time():
    df = random_bars(3000, seed=11)
    f = so.run(df, so.Params(min_rr=0))
    assert (f.entry_bar.iloc[1:].to_numpy() > f.exit_bar.iloc[:-1].to_numpy()).all()


if __name__ == "__main__":
    fns = [v for k_, v in dict(globals()).items() if k_.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
