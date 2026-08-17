"""American exercise: Leisen-Reimer lattice and the de-Americanisation loop.

US single-name and ETF options are American-style on underlyings that pay
*discrete cash* dividends, and they are not dividend-protected.  Inverting a
European formula on an American quote produces a biased volatility -- the bias
is largest exactly where the early-exercise premium is largest (deep ITM,
long-dated, high-yield names), which is where a naive system will hallucinate
skew that is not there.

Engine choice, in the order this system uses them:

  1. Leisen-Reimer (odd n, 101-201 steps) -- the production mark.  LR chooses
     u/d so the terminal binomial distribution is a high-order Peizer-Pratt
     approximation to the lognormal with the strike at its centre.  European
     error is O(1/n^2) and, critically, *smooth and monotone* in n rather than
     CRR's sawtooth -- which is what makes Richardson extrapolation legitimate.
  2. Bjerksund-Stensland 2002 -- microsecond closed form, used for the initial
     screen and for Jacobians.  Not used as a final mark: typical deviation is
     1-3 cents per contract.

Discrete dividends use Vellekoop-Nieuwenhuis: the lattice stays recombining
and the post-dividend value at an ex-date is obtained by interpolating the
continuation function at S - D.  Subtracting D from the node price directly
destroys recombination and makes the tree O(2^m) in the dividend count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from . import black
from .implied import implied_vol_scalar

__all__ = ["leisen_reimer", "bjerksund_stensland_2002", "de_americanise", "Dividend"]


@dataclass(frozen=True)
class Dividend:
    """A discrete cash dividend."""
    t: float      # year fraction from valuation to ex-date
    amount: float


# ----------------------------------------------------------------------------
# Leisen-Reimer
# ----------------------------------------------------------------------------

def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion, method 2.  Maps a normal deviate to a binomial p."""
    a = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    inner = (z / a) ** 2 * (n + 1.0 / 6.0)
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-inner))


def leisen_reimer(
    S: float, K: float, T: float, r: float, q: float, sig: float, cp: int,
    n: int = 151,
    american: bool = True,
    dividends: list[Dividend] | None = None,
) -> float:
    """Price a (European or American) option on an LR lattice.

    ``n`` is forced ODD.  The Peizer-Pratt correction is a normal-to-binomial
    approximation that is only centred when the terminal node count ``n+1`` is
    even; with even ``n`` the strike lands on a node, the correction misaligns,
    and the O(1/n^2) convergence is destroyed.
    """
    if T <= 0:
        return float(max(cp * (S - K), 0.0))
    n = int(n) | 1                                   # force odd
    dt = T / n
    b = r - q

    d1, d2 = black.d1_d2(S, K, T, r, q, sig)
    p = _peizer_pratt(float(d2), n)
    p_dash = _peizer_pratt(float(d1), n)
    p = min(max(p, 1e-12), 1 - 1e-12)

    u = np.exp(b * dt) * p_dash / p
    d = (np.exp(b * dt) - p * u) / (1.0 - p)
    disc = np.exp(-r * dt)

    divs = sorted(dividends or [], key=lambda x: x.t)
    div_steps = {min(int(np.floor(dv.t / dt)), n - 1): dv.amount for dv in divs if 0 <= dv.t < T}

    j = np.arange(n + 1)
    ST = S * (u ** j) * (d ** (n - j))
    V = np.maximum(cp * (ST - K), 0.0)

    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        Si = S * (u ** j) * (d ** (i - j))
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            V = np.maximum(V, cp * (Si - K))
        if i in div_steps:
            # Vellekoop-Nieuwenhuis: evaluate the continuation function at S-D
            # by interpolation on the (log-spaced) surviving lattice.
            D = div_steps[i]
            Sd = np.maximum(Si - D, 1e-8)
            order = np.argsort(Si)
            V = np.interp(Sd, Si[order], V[order])
            if american:
                V = np.maximum(V, cp * (Si - K))
    return float(V[0])


# ----------------------------------------------------------------------------
# Bjerksund-Stensland 2002
# ----------------------------------------------------------------------------

def _phi(S, tau, gamma, H, Z, r, b, sig):
    lam = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sig**2
    kappa = 2.0 * b / sig**2 + (2.0 * gamma - 1.0)
    srt = sig * np.sqrt(tau)
    dd = -(np.log(S / H) + (b + (gamma - 0.5) * sig**2) * tau) / srt
    return np.exp(lam * tau) * S**gamma * (
        norm.cdf(dd) - (Z / S) ** kappa * norm.cdf(dd - 2.0 * np.log(Z / S) / srt)
    )


def bjerksund_stensland_2002(S, K, T, r, q, sig, cp=1) -> float:
    """Closed-form American approximation (single-boundary variant of BS2002).

    Uses the flat-boundary trigger with the golden-section split collapsed to a
    single boundary -- the two-boundary refinement buys ~1 cent and is not
    worth the branch complexity here, because this engine is only ever used as
    a *screen*.  The production mark is :func:`leisen_reimer`.

    Puts by the Bjerksund-Stensland transform
    ``P(S, K, T, r, b, sig) = C(K, S, T, r - b, -b, sig)``.  With ``b = r - q``
    this is ``C(K, S, T, q, q - r, sig)``.
    """
    b = r - q
    if cp < 0:
        return _bs2002_call(K, S, T, r - b, -b, sig)
    return _bs2002_call(S, K, T, r, b, sig)


def _bs2002_call(S, K, T, r, b, sig) -> float:
    if T <= 0:
        return float(max(S - K, 0.0))
    if b >= r:                       # q <= 0: never optimal to exercise early
        return float(black.price(S, K, T, r, r - b, sig, 1))
    s2 = sig * sig
    beta = (0.5 - b / s2) + np.sqrt((b / s2 - 0.5) ** 2 + 2.0 * r / s2)
    if not np.isfinite(beta) or beta <= 1.0:
        return float(black.price(S, K, T, r, r - b, sig, 1))
    B_inf = beta / (beta - 1.0) * K
    denom = r - b
    B_0 = max(K, (r / denom) * K) if denom > 1e-12 else K
    if B_inf <= B_0:
        return float(black.price(S, K, T, r, r - b, sig, 1))
    h = -(b * T + 2.0 * sig * np.sqrt(T)) * K * K / ((B_inf - B_0) * B_0)
    X = B_0 + (B_inf - B_0) * (1.0 - np.exp(h))
    if S >= X:
        return float(S - K)
    alpha = (X - K) * X ** (-beta)
    val = (
        alpha * S**beta
        - alpha * _phi(S, T, beta, X, X, r, b, sig)
        + _phi(S, T, 1.0, X, X, r, b, sig)
        - _phi(S, T, 1.0, K, X, r, b, sig)
        - K * _phi(S, T, 0.0, X, X, r, b, sig)
        + K * _phi(S, T, 0.0, K, X, r, b, sig)
    )
    euro = float(black.price(S, K, T, r, r - b, sig, 1))
    return float(max(val, euro, S - K, 0.0))


# ----------------------------------------------------------------------------
# de-Americanisation
# ----------------------------------------------------------------------------

@dataclass
class DeAmResult:
    sigma_american: float
    european_price: float
    sigma_european: float
    ok: bool
    detail: str = ""


def de_americanise(
    market_price: float, S: float, K: float, T: float, r: float, q: float, cp: int,
    dividends: list[Dividend] | None = None,
    n: int = 151,
    lo: float = 1e-3, hi: float = 5.0,
) -> DeAmResult:
    """Convert an American market quote into the *synthetic European* price
    that a Black-76 surface fit can legitimately consume.

    1. solve sigma_A such that LR_american(sigma_A) == market_price
    2. reprice the *European* option at that same sigma_A, same dividends
    3. that synthetic European price is what gets inverted by Let's Be
       Rational and handed to the SVI fit

    Model-dependent but internally consistent -- which is exactly what makes
    the resulting surface arbitrage-checkable.  Feeding raw American IVs into
    an SVI fit produces a surface whose butterfly test fails for reasons that
    have nothing to do with the market.
    """
    def f(sig):
        return leisen_reimer(S, K, T, r, q, sig, cp, n=n, american=True, dividends=dividends) - market_price

    try:
        if f(lo) > 0 or f(hi) < 0:
            return DeAmResult(np.nan, np.nan, np.nan, False, "price outside lattice-attainable range")
        sig_a = brentq(f, lo, hi, xtol=1e-10, rtol=1e-12, maxiter=200)
    except (ValueError, RuntimeError) as exc:  # pragma: no cover
        return DeAmResult(np.nan, np.nan, np.nan, False, f"root-find failed: {exc}")

    euro = leisen_reimer(S, K, T, r, q, sig_a, cp, n=n, american=False, dividends=dividends)
    F = float(black.forward(S, r, q, T))
    if dividends:
        pv = sum(dv.amount * np.exp(-r * dv.t) for dv in dividends if 0 <= dv.t <= T)
        F = float(black.forward(S - pv, r, 0.0, T))
    res = implied_vol_scalar(euro, F, K, T, cp, df=float(np.exp(-r * T)))
    return DeAmResult(float(sig_a), float(euro), float(res.sigma), res.ok, res.status)


def early_exercise_threshold(K: float, r: float, dt: float) -> float:
    """K * (1 - exp(-r*dt)).

    A dividend smaller than this can *never* make early exercise of an American
    call optimal at that ex-date -- the interest earned on the strike over the
    remaining sub-period dominates.  Skipping the comparison when the test
    fails removes most of the assignment-risk scan.  Note the regime
    dependence: at r ~ 4-5% this binds frequently; at ZIRP it is ~0 and almost
    every dividend triggers the check.
    """
    return float(K * (1.0 - np.exp(-r * dt)))
