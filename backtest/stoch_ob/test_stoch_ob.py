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
    """Every entry decision must be reproducible from data[:entry_bar + 1]: keep one extra bar (entries on a
    data set's last bar are not allowed) but replace it with a far-away bar, so any peek at it shows up."""
    df = random_bars(1500, seed=21)
    for kw in (dict(), dict(sl_mode="fractal"), dict(entry_mode="cross"), dict(entry_mode="exit_zone"),
               dict(ob_too_close="next"), dict(min_rr=0)):
        p = so.Params(**kw)
        full = so.run(df, p)
        assert len(full) > 20
        for _, tr in full.iloc[::4].iterrows():
            e = int(tr.entry_bar)
            for shift in (50.0, -50.0):
                cut = df.iloc[:e + 2].copy().reset_index(drop=True)
                px = cut.loc[e, "close"] + shift
                cut.loc[e + 1, ["open", "high", "low", "close"]] = [px, px + 1, px - 1, px]
                res = so.run(cut, p)
                row = res[res.entry_bar == e] if len(res) else res
                assert len(row) == 1, (kw, e, shift)
                row = row.iloc[0]
                assert row.side == tr.side and np.isclose(row.entry, tr.entry) and np.isclose(row.sl, tr.sl)
                assert (np.isnan(row.tp) and np.isnan(tr.tp)) or np.isclose(row.tp, tr.tp)
                assert set(res.entry_bar[res.entry_bar < e]) <= set(full.entry_bar)


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


# ------------------------------------------------------- order blocks / pivots / zones
class patched:
    """Temporarily replace attributes of the stoch_ob module."""
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k_, v in self.kw.items():
            self.old[k_] = getattr(so, k_)
            setattr(so, k_, v)

    def __exit__(self, *exc):
        for k_, v in self.old.items():
            setattr(so, k_, v)


OB_ROWS = [  # open, high, low, close
    (10.0, 11.0, 9.0, 10.5),    # 0
    (10.5, 12.0, 10.0, 11.5),   # 1
    (11.5, 13.0, 11.0, 12.5),   # 2
    (12.5, 15.0, 13.0, 14.0),   # 3  swing high 15, confirmed at t=6 (ob_pivot 3)
    (14.0, 14.0, 12.0, 13.0),   # 4  red
    (13.0, 13.0, 11.5, 12.0),   # 5  red  <- expected bullish OB [11.5, 13]
    (10.5, 12.5, 10.0, 12.0),   # 6  green, lowest low after the swing
    (12.0, 13.0, 11.0, 12.8),   # 7
    (12.8, 14.0, 12.0, 13.9),   # 8
    (13.9, 16.0, 13.0, 15.5),   # 9  close 15.5 > 15 -> structure break, OB created
    (15.5, 16.5, 14.5, 15.0),   # 10 forced SELL signal
    (15.0, 15.2, 13.0, 13.2),   # 11 low 13.0 <= TP 13.3
]


def _ob_run(rows, signal_bar=10, **kw):
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df["time"] = pd.date_range("2020-01-01", periods=len(df), freq="30min")
    seen = {}
    real_open = so.open_trade

    def fake_signals(k, d, p):
        s_ = np.zeros(len(k), bool)
        s_[signal_bar] = True
        return s_, np.zeros(len(k), bool)

    def spy(t, side, *a):
        seen[t] = [list(x) for x in a[-3]]          # bull_obs at the entry bar
        return real_open(t, side, *a)

    p = so.Params(spread=0.3, sl_lookback=3, min_rr=0.5, one_per_zone=False, **kw)
    with patched(entry_signals=fake_signals, open_trade=spy):
        f = so.run(df, p)
    return f, seen.get(signal_bar)


def test_order_block_created_on_break_and_used_as_tp():
    f, obs = _ob_run(OB_ROWS)
    assert obs == [[13.0, 11.5, 9, False]]               # bar 5 zone, created at t=9, fresh
    tr = f.iloc[0]
    assert tr.sl == 16.8 and tr.tp == 13.3 and tr.reason == "TP_OB" and tr.exit_bar == 11


def test_touched_order_block_is_not_a_target():
    rows = list(OB_ROWS)
    rows[10] = (15.5, 16.5, 12.9, 15.0)                  # bar 10 dips into the OB (low <= 13)
    f, obs = _ob_run(rows)
    assert obs[0][3] is True and np.isnan(f.iloc[0].tp)


def test_pivot_needs_right_bars_before_a_break_counts():
    rows = list(OB_ROWS)
    rows[5] = (13.0, 15.6, 12.9, 15.5)                   # close > 15 at t=5, before the pivot is confirmed
    f, obs = _ob_run(rows)
    assert obs == []                                     # bar 3 never became a pivot -> no OB


def test_pivot_confirmed_only_after_full_right_side():
    rows = [(10.0, 11.0, 9.0, 10.5), (10.5, 12.0, 10.0, 11.5), (11.5, 13.0, 11.0, 12.5),
            (12.5, 15.0, 13.0, 14.0),      # 3  candidate swing high 15 ...
            (14.0, 14.0, 12.0, 13.0), (13.0, 13.0, 11.5, 12.0),
            (12.0, 15.5, 11.8, 14.5),      # 6  ... killed by a higher high on the 3rd right bar
            (14.5, 15.45, 14.0, 15.4),     # 7  close > 15 would be a "break" only if bar 3 were a pivot
            (15.4, 15.4, 14.5, 14.8), (14.8, 15.0, 14.2, 14.5), (14.5, 15.0, 14.0, 14.2),
            (14.2, 14.3, 13.0, 13.2)]
    _, obs = _ob_run(rows)
    assert obs == []


def test_one_signal_per_zone_visit_even_when_skipped():
    n = 60
    df = random_bars(n, seed=5)
    k = np.full(n, 50.0)
    k[30:36] = [85, 90, 88, 86, 84, 70]                  # one visit, raw hooks at 32, 33, 34, 35
    calls = []

    def fake_stoch(h, l, c, kp, dp, sp):
        return k.copy(), k.copy()

    def refuse(t, side, *a):
        calls.append(t)
        return None

    with patched(stochastic=fake_stoch, open_trade=refuse):
        so.run(df, so.Params(one_per_zone=True))
        assert calls == [32]
        calls.clear()
        so.run(df, so.Params(one_per_zone=False))
        assert calls == [32, 33, 34, 35]


def test_no_entry_on_last_bar():
    df = random_bars(40, seed=2)
    k = np.full(40, 50.0)
    k[37:40] = [85, 90, 70]                              # hook on the last bar only
    with patched(stochastic=lambda *a: (k.copy(), k.copy())):
        assert len(so.run(df, so.Params(min_rr=0))) == 0


# ---------------------------------------------------------------- more trade checks
def test_sl_lookback_includes_entry_bar():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0, tp_mode="stoch_only")
    rows = BASE[:4] + [(101, 107, 99, 100), (100, 101, 99, 100)]   # entry bar has the highest high
    tr = _open(rows, -1, [50] * 6, p)
    assert tr["sl"] == 107.5


def test_fractal_sl_not_inside_recent_unconfirmed_swing():
    o, h, l, c = _arrays(BASE + [(100, 101, 99, 100)])
    tm = pd.date_range("2020-01-01", periods=6, freq="30min").to_numpy()
    p = so.Params(sl_mode="fractal", spread=0.5, min_rr=0, tp_mode="stoch_only", ob_pivot=3)
    tr = so.open_trade(4, -1, o, h, l, c, np.full(6, 50.0), np.ones(6), tm,
                       [(0, 104.0)], [], [], [], p)     # confirmed pivot 104 < recent high 105 (bar 3)
    assert tr["sl"] == 105.5


def test_tp_gap_fills_at_tp_and_tp_beats_stoch():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=1.0)
    bull = [(90, 88, 0, False)]
    tr = _open(BASE + [(85, 86, 84, 85)], -1, [50, 50, 50, 50, 50, 10], p, bull=bull)  # gap through TP
    assert tr["reason"] == "TP_OB" and tr["exit"] == 90.5


def test_nearest_ob_not_in_profit_is_too_close():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=1.0, ob_too_close="skip")
    bull = [(99.9, 99.0, 0, False), (90, 88, 0, False)]  # nearest TP 100.4 is not even in profit
    assert _open(BASE + [(100, 101, 89, 90)], -1, [50] * 6, p, bull=bull) is None


def test_zone_visit_used_while_a_trade_is_open():
    n = 80
    df = random_bars(n, seed=6)
    k = np.full(n, 50.0)
    k[30:33] = [85, 90, 88]                              # sell hook at 32, opens a trade
    k[40:46] = [10, 5, 8, 6, 9, 30]                      # buy hooks at 42, 44, 45 while the sell is still open
    calls = []

    def fake_stoch(h, l, c, kp, dp, sp):
        return k.copy(), k.copy()

    def open_until_43(t, side, *a):
        calls.append(t)
        return dict(exit_bar=43) if t == 32 else None

    with patched(stochastic=fake_stoch, open_trade=open_until_43):
        so.run(df, so.Params())
    assert calls == [32]                                 # the visit's first buy signal (42) came during the trade


def test_tp_closer_than_spread_is_too_close_even_with_min_rr_0():
    p = so.Params(sl_lookback=3, spread=0.5, min_rr=0.0)
    bull = [(99.2, 99.0, 0, False)]                      # TP 99.7: +0.3 gross, -0.2 after spread
    assert _open(BASE + [(100, 101, 99, 100)], -1, [50] * 6, p, bull=bull) is None


if __name__ == "__main__":
    fns = [v for k_, v in dict(globals()).items() if k_.startswith("test_")]
    for fn in fns:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(fns)} tests passed")
