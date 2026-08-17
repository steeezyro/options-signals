import numpy as np
import pytest

from optionsmarkets.pricing import black
from optionsmarkets.pricing.american import (Dividend, bjerksund_stensland_2002,
                                             de_americanise, early_exercise_threshold,
                                             leisen_reimer)
from optionsmarkets.pricing.forward import implied_forward
from optionsmarkets.pricing.implied import implied_vol_scalar

S, K, T, R, Q, SIG = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20


def test_put_call_parity_exact():
    c = float(black.price(S, K, T, R, Q, SIG, 1))
    p = float(black.price(S, K, T, R, Q, SIG, -1))
    assert c - p == pytest.approx(S * np.exp(-Q * T) - K * np.exp(-R * T), abs=1e-12)


@pytest.mark.parametrize("name,bump,order", [
    ("delta", "S", 1), ("gamma", "S", 2), ("vega", "sig", 1),
    ("rho", "r", 1), ("theta", "T", 1),
])
def test_greeks_match_finite_difference(name, bump, order):
    h = 1e-4
    def px(**kw):
        a = dict(S=S, K=K, T=T, r=R, q=Q, sig=SIG)
        a.update(kw)
        return float(black.price(a["S"], a["K"], a["T"], a["r"], a["q"], a["sig"], 1))
    base = {"S": S, "sig": SIG, "r": R, "T": T}[bump]
    up, dn = px(**{bump: base + h}), px(**{bump: base - h})
    if order == 1:
        fd = (up - dn) / (2 * h)
        if bump == "T":       # theta is d/dt, and dt = -dT
            fd = -fd
    else:
        fd = (up - 2 * px() + dn) / h**2
    assert float(black.greeks(S, K, T, R, Q, SIG, 1)[name]) == pytest.approx(fd, rel=2e-4, abs=1e-6)


def test_time_greek_sign_convention():
    """Long ATM option: value falls, vega falls, gamma rises as time passes."""
    g = black.greeks(S, K, T, R, Q, SIG, 1)
    assert float(g.theta) < 0
    assert float(g.veta) < 0
    assert float(g.color) > 0


def test_scale_for_trader_units():
    g = black.greeks(S, K, T, R, Q, SIG, 1)
    t = black.scale_for_trader(g)
    assert float(t.vega) == pytest.approx(float(g.vega) / 100.0)
    assert float(t.theta) == pytest.approx(float(g.theta) / 365.0)


def test_iv_roundtrip_tradeable_regime():
    """Every quote with >= 1 cent of time value must invert to 1e-9 or better."""
    worst, n = 0.0, 0
    for T_ in (1 / 365, 7 / 365, 30 / 365, 0.25, 1.0):
        for mny in np.linspace(0.6, 1.6, 41):
            for sv in (0.08, 0.15, 0.25, 0.45, 0.9, 1.5):
                for cp in (1, -1):
                    k = 100.0 * mny
                    F = float(black.forward(100.0, R, Q, T_))
                    df = float(np.exp(-R * T_))
                    px = float(black.price(100.0, k, T_, R, Q, sv, cp))
                    lo, _ = black.no_arb_bounds(100.0, k, T_, R, Q, cp)
                    if px - float(lo) < 0.01:
                        continue
                    res = implied_vol_scalar(px, F, k, T_, cp, df)
                    assert res.ok, (T_, mny, sv, cp, res.status)
                    worst = max(worst, abs(res.sigma - sv)); n += 1
    assert n > 800
    assert worst < 1e-9, worst


def test_iv_rejects_rather_than_hallucinates():
    """Deep-ITM quote whose time value is below double precision must be
    rejected, not inverted into a confident fictitious vol."""
    T_ = 1 / 365
    F = float(black.forward(100.0, R, Q, T_))
    px = float(black.price(100.0, 40.0, T_, R, Q, 0.03, 1))
    res = implied_vol_scalar(px, F, 40.0, T_, 1, float(np.exp(-R * T_)))
    assert not res.ok
    assert res.status in ("UNDERFLOW", "NO_ARBITRAGE")


def test_iv_rejects_zero_vega_quote():
    T_, k = 30 / 365, 200.0
    F = float(black.forward(100.0, R, Q, T_))
    px = float(black.price(100.0, k, T_, R, Q, 0.20, 1))
    res = implied_vol_scalar(px, F, k, T_, 1, 1.0, spread=0.10, max_spread_vols=0.05)
    assert res.status in ("NO_VEGA", "UNDERFLOW", "NO_ARBITRAGE")


def test_lr_converges_to_black_scholes_for_european():
    exact = float(black.price(S, K, T, R, Q, SIG, 1))
    errs = [abs(leisen_reimer(S, K, T, R, Q, SIG, 1, n=n, american=False) - exact)
            for n in (51, 101, 201, 401)]
    assert errs == sorted(errs, reverse=True)      # monotone convergence
    assert errs[-1] < 1e-5


def test_lr_forces_odd_steps():
    a = leisen_reimer(S, K, T, R, Q, SIG, 1, n=100, american=False)
    b = leisen_reimer(S, K, T, R, Q, SIG, 1, n=101, american=False)
    assert a == pytest.approx(b, abs=1e-12)


def test_american_put_dominates_european():
    am = leisen_reimer(S, K, T, R, 0.0, SIG, -1, n=201, american=True)
    eu = float(black.price(S, K, T, R, 0.0, SIG, -1))
    assert am > eu
    assert am >= max(K - S, 0.0)


def test_american_call_no_dividend_equals_european():
    am = leisen_reimer(100.0, 95.0, 1.0, 0.05, 0.0, 0.25, 1, n=301, american=True)
    eu = float(black.price(100.0, 95.0, 1.0, 0.05, 0.0, 0.25, 1))
    assert am == pytest.approx(eu, abs=1e-4)


def test_bs2002_matches_haug_reference():
    """Haug's published table: S=42 K=40 T=0.75 r=4% q=8% sig=35% -> 5.2704."""
    v = bjerksund_stensland_2002(42.0, 40.0, 0.75, 0.04, 0.08, 0.35, 1)
    assert v == pytest.approx(5.2704, abs=5e-3)


def test_bs2002_below_lattice_as_documented():
    lr = leisen_reimer(42.0, 40.0, 0.75, 0.04, 0.08, 0.35, 1, n=1001, american=True)
    bs = bjerksund_stensland_2002(42.0, 40.0, 0.75, 0.04, 0.08, 0.35, 1)
    assert bs <= lr + 1e-9
    assert lr - bs < 0.10


def test_dividend_lowers_call_raises_put():
    d = [Dividend(t=0.5, amount=2.0)]
    c0 = leisen_reimer(S, K, T, R, 0.0, SIG, 1, n=201, american=True)
    c1 = leisen_reimer(S, K, T, R, 0.0, SIG, 1, n=201, american=True, dividends=d)
    p0 = leisen_reimer(S, K, T, R, 0.0, SIG, -1, n=201, american=True)
    p1 = leisen_reimer(S, K, T, R, 0.0, SIG, -1, n=201, american=True, dividends=d)
    assert c1 < c0 and p1 > p0


def test_early_exercise_threshold():
    assert early_exercise_threshold(100.0, 0.05, 0.25) == pytest.approx(100 * (1 - np.exp(-0.0125)))
    assert early_exercise_threshold(100.0, 0.0, 0.25) == pytest.approx(0.0)


def test_de_americanisation_reduces_implied_vol():
    """American price > European at the same vol, so the vol implied by
    treating an American quote as European is biased HIGH.  De-Americanising
    must bring it back down."""
    mkt = leisen_reimer(100.0, 110.0, 1.0, 0.05, 0.0, 0.30, -1, n=201, american=True)
    F = float(black.forward(100.0, 0.05, 0.0, 1.0))
    naive = implied_vol_scalar(mkt, F, 110.0, 1.0, -1, float(np.exp(-0.05)))
    res = de_americanise(mkt, 100.0, 110.0, 1.0, 0.05, 0.0, -1, n=201)
    assert res.ok
    assert res.sigma_american == pytest.approx(0.30, abs=1e-4)
    assert res.sigma_european < naive.sigma


def test_forward_from_parity_recovers_truth():
    T_ = 0.09
    F = float(black.forward(778.54, 0.043, 0.012, T_))
    df = float(np.exp(-0.043 * T_))
    Ks = np.arange(740, 820, 5.0)
    C = black.price(778.54, Ks, T_, 0.043, 0.012, 0.15, 1)
    P = black.price(778.54, Ks, T_, 0.043, 0.012, 0.15, -1)
    fit = implied_forward(Ks, C, P, T_, S_ref=778.54)
    assert fit.ok
    assert fit.F == pytest.approx(F, abs=1e-6)
    assert fit.DF == pytest.approx(df, abs=1e-9)
