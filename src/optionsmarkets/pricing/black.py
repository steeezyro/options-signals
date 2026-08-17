"""Black-Scholes-Merton pricing and the full second/third-order Greek set.

All functions are vectorised over numpy arrays and return values in *natural*
units (per 1.00 of vol, per year of time).  Trader-facing scaling (per vol
point, per calendar day) is applied once, at the reporting boundary, by
:func:`scale_for_trader` -- never inside the math.

Conventions
-----------
S   spot of the underlying
K   strike
T   year fraction to expiry (ACT/365F)
r   continuously-compounded risk-free rate
q   continuous dividend yield (or borrow-adjusted carry); b = r - q is carry
sig Black-Scholes volatility, annualised, as a decimal (0.18 == 18 vol)
cp  +1 for a call, -1 for a put

The forward is F = S * exp((r - q) * T) and the discount factor DF = exp(-r*T).
Every routine here is expressible in forward terms; we keep the spot form
because the Greeks a trader hedges with (delta vs. shares) are spot Greeks.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "price",
    "forward",
    "greeks",
    "Greeks",
    "scale_for_trader",
    "intrinsic",
    "no_arb_bounds",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_TINY = 1e-12


def _pdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def _cdf(x):
    return norm.cdf(x)


def forward(S, r, q, T):
    """F = S e^{(r-q)T}."""
    return np.asarray(S, float) * np.exp((np.asarray(r, float) - np.asarray(q, float)) * np.asarray(T, float))


def d1_d2(S, K, T, r, q, sig):
    """Return (d1, d2).  Degenerate T or sig is clamped, not allowed to divide by zero."""
    S, K, T, r, q, sig = map(lambda x: np.asarray(x, dtype=float), (S, K, T, r, q, sig))
    v = np.maximum(sig, _TINY) * np.sqrt(np.maximum(T, _TINY))
    d1 = (np.log(np.maximum(S, _TINY) / np.maximum(K, _TINY)) + (r - q + 0.5 * sig**2) * T) / v
    return d1, d1 - v


def intrinsic(S, K, cp):
    cp = np.asarray(cp, float)
    return np.maximum(cp * (np.asarray(S, float) - np.asarray(K, float)), 0.0)


def no_arb_bounds(S, K, T, r, q, cp):
    """European no-arbitrage price bounds (lower, upper).

    Any quote outside these is unpriceable -- IV inversion must reject it
    rather than returning a garbage root.  Lower bound uses the *discounted
    forward* intrinsic, which is what actually binds; the undiscounted
    ``max(S-K,0)`` bound is looser and lets bad quotes through.
    """
    S, K, T, r, q = map(lambda x: np.asarray(x, float), (S, K, T, r, q))
    cp = np.asarray(cp, float)
    df, dq = np.exp(-r * T), np.exp(-q * T)
    lower = np.maximum(cp * (S * dq - K * df), 0.0)
    upper = np.where(cp > 0, S * dq, K * df)
    return lower, upper


def price(S, K, T, r, q, sig, cp):
    """Black-Scholes-Merton European price."""
    S, K, T, r, q, cp = map(lambda x: np.asarray(x, float), (S, K, T, r, q, cp))
    d1, d2 = d1_d2(S, K, T, r, q, sig)
    df, dq = np.exp(-r * T), np.exp(-q * T)
    return cp * (S * dq * _cdf(cp * d1) - K * df * _cdf(cp * d2))


class Greeks(dict):
    """Attribute-accessible Greek bundle.  A dict so it serialises for free."""

    __getattr__ = dict.__getitem__

    def __repr__(self):  # pragma: no cover - cosmetic
        return "Greeks(" + ", ".join(f"{k}={float(np.ravel(v)[0]):.6g}" for k, v in self.items()) + ")"


def greeks(S, K, T, r, q, sig, cp) -> Greeks:
    """Full Greek set in natural units.

    Returns
    -------
    delta   dV/dS
    gamma   d2V/dS2
    vega    dV/dsigma            (per 1.00 of vol)
    theta   dV/dt                (per YEAR, negative for long premium)
    rho     dV/dr                (per 1.00 of rate)
    vanna   d2V/dS dsigma
    volga   d2V/dsigma2          (vomma)
    charm   d(delta)/dt          (per YEAR)
    veta    d(vega)/dt           (per YEAR)
    speed   d3V/dS3
    zomma   d(gamma)/dsigma
    color   d(gamma)/dt          (per YEAR)
    ultima  d3V/dsigma3
    dual_delta  dV/dK  == -DF * N(cp*d2) * cp  ->  risk-neutral CDF, used for PoP
    """
    S, K, T, r, q, sig, cp = map(lambda x: np.asarray(x, float), (S, K, T, r, q, sig, cp))
    Tc = np.maximum(T, _TINY)
    sc = np.maximum(sig, _TINY)
    srt = sc * np.sqrt(Tc)
    d1, d2 = d1_d2(S, K, Tc, r, q, sc)
    nd1 = _pdf(d1)
    df, dq = np.exp(-r * Tc), np.exp(-q * Tc)
    Nc1, Nc2 = _cdf(cp * d1), _cdf(cp * d2)

    delta = cp * dq * Nc1
    gamma = dq * nd1 / (S * srt)
    vega = S * dq * nd1 * np.sqrt(Tc)
    theta = (
        -S * dq * nd1 * sc / (2.0 * np.sqrt(Tc))
        + cp * q * S * dq * Nc1
        - cp * r * K * df * Nc2
    )
    rho = cp * K * Tc * df * Nc2
    vanna = -dq * nd1 * d2 / sc
    volga = vega * d1 * d2 / sc
    # Time Greeks are ALL expressed as d/dt with t = calendar time moving
    # forward (so dT = -dt).  Sanity check for a long ATM option: value falls
    # (theta < 0), vega falls (veta < 0), gamma rises (color > 0).
    charm = cp * q * dq * Nc1 - dq * nd1 * (2.0 * (r - q) * Tc - d2 * srt) / (2.0 * Tc * srt)
    veta = S * dq * nd1 * np.sqrt(Tc) * (
        q + (r - q) * d1 / srt - (1.0 + d1 * d2) / (2.0 * Tc)
    )
    speed = -gamma / S * (d1 / srt + 1.0)
    zomma = gamma * (d1 * d2 - 1.0) / sc
    color = dq * nd1 / (2.0 * S * Tc * srt) * (
        2.0 * q * Tc + 1.0 + (2.0 * (r - q) * Tc - d2 * srt) / srt * d1
    )
    ultima = -vega / sc**2 * (d1 * d2 * (1.0 - d1 * d2) + d1**2 + d2**2)
    dual_delta = -cp * df * Nc2

    return Greeks(
        price=cp * (S * dq * Nc1 - K * df * Nc2),
        delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
        vanna=vanna, volga=volga, charm=charm, veta=veta,
        speed=speed, zomma=zomma, color=color, ultima=ultima,
        dual_delta=dual_delta, d1=d1, d2=d2,
    )


def scale_for_trader(g: Greeks, days_per_year: float = 365.0) -> Greeks:
    """Convert natural-unit Greeks to the units on a broker screen.

    vega, vanna, volga  -> per 1 vol POINT   (divide by 100)
    theta, charm, veta, color -> per CALENDAR DAY (divide by days_per_year)

    Doing this exactly once, here, is deliberate: the single most common
    source of silent sizing errors in an options system is a vega that is
    100x off because two modules disagreed about the unit.
    """
    out = Greeks(g)
    for k in ("vega", "vanna", "ultima"):
        out[k] = g[k] / 100.0
    out["volga"] = g["volga"] / 10000.0
    for k in ("theta", "charm", "veta", "color"):
        out[k] = g[k] / days_per_year
    return out
