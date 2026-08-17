"""Implied-volatility inversion.

Production systems should link Jaeckel's *Let's Be Rational* (via
``py_lets_be_rational``) which attains full double precision in two
iterations for every representable input.  This module provides a
self-contained fallback with the same structural ideas -- normalised Black
function, Manaster-Koehler seed at the vega-maximising vol, a log-transformed
objective in the low-price regime where the price is exponentially flat in
sigma, third-order Householder steps, and a maintained bracket that
guarantees convergence -- so the repo has no hard dependency and the two can
be cross-validated against each other.

The known failure modes are handled *explicitly* rather than being allowed to
return a plausible-looking number:

    NO_ARBITRAGE   price below discounted intrinsic or above the upper bound
    NO_VEGA        (ask-bid)/vega exceeds `max_spread_vols` -- the market is
                   not quoting a volatility at this strike, and any number we
                   return is an artefact of the tick size
    NOT_CONVERGED  bracket exhausted (should be unreachable)

A rejected quote is *dropped from the surface fit*, never imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcx

__all__ = ["implied_vol", "implied_vol_scalar", "IVResult", "IVStatus"]

IVStatus = Literal["OK", "NO_ARBITRAGE", "NO_VEGA", "UNDERFLOW", "NOT_CONVERGED"]

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_INV_SQRT2 = 1.0 / np.sqrt(2.0)


@dataclass(frozen=True)
class IVResult:
    sigma: float
    status: IVStatus
    iterations: int = 0
    vega: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "OK"


# ----------------------------------------------------------------------------
# normalised Black machinery (Jaeckel's parameterisation)
# ----------------------------------------------------------------------------

def _normalised_black(x: float, s: float) -> float:
    """b(x,s) = N(x/s + s/2) e^{x/2} - N(x/s - s/2) e^{-x/2}.

    x = ln(F/K), s = sigma*sqrt(T).  Undiscounted price of the *call* on the
    normalised scale beta = price / (sqrt(F) sqrt(K)).

    Evaluated through the *scaled* complementary error function so that only
    ONE exponential appears and the difference of two CDFs never cancels.
    Using ``N(z) = exp(-z^2/2) * erfcx(-z/sqrt2) / 2``::

        b(x,s) = 0.5 * exp(-(h^2 + t^2)/2) * [ erfcx(-(h+t)/sqrt2)
                                             - erfcx(-(h-t)/sqrt2) ]

    The naive ``norm.cdf(h+t)*exp(h*t) - norm.cdf(h-t)*exp(-h*t)`` form loses
    every significant digit once the option is more than ~4 standard
    deviations out of the money, which silently poisons the wings of the
    surface fit -- precisely the strikes a short-premium book lives on.
    """
    if s <= 0.0:
        return max(np.exp(x / 2.0) - np.exp(-x / 2.0), 0.0)
    h, t = x / s, s / 2.0
    scale = -0.5 * (h * h + t * t)
    if scale < -708.0:                       # exp underflows: price is not representable
        return 0.0
    return 0.5 * np.exp(scale) * (
        erfcx(-(h + t) * _INV_SQRT2) - erfcx(-(h - t) * _INV_SQRT2)
    )


def _normalised_vega(x: float, s: float) -> float:
    """db/ds -- the normalised vega.  Single exponential, no cancellation."""
    if s <= 0.0:
        return 0.0
    return np.exp(-0.5 * ((x / s) ** 2 + (s / 2.0) ** 2)) / _SQRT_2PI


def implied_vol_scalar(
    price: float,
    F: float,
    K: float,
    T: float,
    cp: int,
    df: float = 1.0,
    *,
    spread: float | None = None,
    max_spread_vols: float = 0.05,
    tol: float = 1e-12,
    max_iter: int = 32,
) -> IVResult:
    """Invert one Black-76 price to a volatility.

    Parameters
    ----------
    price   option price as quoted (i.e. *discounted*)
    F, K    forward and strike
    T       year fraction
    cp      +1 call, -1 put
    df      discount factor; ``price/df`` is the undiscounted price
    spread  ask-bid in price terms.  If given, the quote is rejected when the
            implied vol uncertainty ``spread/(2*vega)`` exceeds
            ``max_spread_vols``.  This is the single most valuable filter in
            the whole pipeline: it removes the strikes where the surface fit
            would otherwise be chasing tick-size noise.
    """
    if not (np.isfinite(price) and np.isfinite(F) and np.isfinite(K)) or T <= 0 or F <= 0 or K <= 0:
        return IVResult(np.nan, "NO_ARBITRAGE")

    undisc = price / df
    x = float(np.log(F / K))

    # Map ITM -> OTM.  The OTM branch is where price is a well-conditioned
    # function of vol; on the ITM branch nearly all the price is intrinsic and
    # the inversion is dominated by cancellation error.
    q = float(cp)
    if q * x > 0.0:
        parity = q * (F - K)
        residual = undisc - parity
        # Catastrophic-cancellation guard.  For a deep-ITM quote nearly all of
        # the price is parity; if what survives the subtraction is within a few
        # ulps of the operands, it is rounding noise, not time value, and
        # inverting it produces a confident, completely fictitious volatility.
        # (Empirically this is the single worst silent failure in a naive
        # inverter: a 1-day 40-delta-wide ITM call reports a 200 vol.)
        if residual <= 1e-13 * max(abs(parity), abs(undisc), 1.0):
            return IVResult(np.nan, "UNDERFLOW")
        undisc = residual
        q = -q

    beta = undisc / np.sqrt(F * K)
    if q < 0:  # normalise everything to a call in x -> -x
        x = -x
    # beta is now the normalised OTM call price at log-moneyness x <= 0

    b_max = np.exp(x / 2.0)
    if beta <= 0.0 or beta >= b_max:
        return IVResult(np.nan, "NO_ARBITRAGE")

    # ---- seed: Manaster-Koehler, i.e. the vega-maximising / inflection vol --
    # s_c = sqrt(2|x|) is exactly where db/ds peaks, so Newton from here is
    # guaranteed to converge monotonically on the OTM branch.
    s_c = max(np.sqrt(2.0 * abs(x)), 1e-4) if x != 0.0 else 0.2
    b_c = _normalised_black(x, s_c)
    if b_c <= 0.0:
        # exp(-(h^2+t^2)/2) underflowed even at the vega-maximising vol: the
        # true price is below the smallest representable double.  There is no
        # volatility to recover -- say so rather than returning the midpoint
        # of whatever bracket we happen to be holding.
        return IVResult(np.nan, "UNDERFLOW")

    v_c = _normalised_vega(x, s_c)
    s = float(np.clip(s_c + (beta - b_c) / max(v_c, 1e-300), 1e-6, 12.0))

    # Maintained bracket.  Every function evaluation tightens it, so the
    # fallback is always available and always valid.
    lo, hi = 1e-9, 20.0
    log_beta = np.log(beta)
    it = 0
    converged = False

    for it in range(1, max_iter + 1):
        b = _normalised_black(x, s)
        v = _normalised_vega(x, s)
        if b < beta:
            lo = max(lo, s)
        else:
            hi = min(hi, s)
        if b <= 0.0 or v <= 0.0:
            s = 0.5 * (lo + hi)
            continue

        # Work on ln b, not b.  Below the inflection point the price is
        # exponentially flat in s and the linear objective is hopelessly
        # ill-conditioned; ln b is close to linear in 1/s there and merely
        # well-behaved above it, so one objective covers both regimes.
        g = np.log(b) - log_beta
        gp = v / b
        newton = -g / gp
        h2 = (x * x / s**3 - s / 4.0) - gp                       # g''/g'
        h3 = h2 * h2 - 3.0 * (x / s**2) ** 2 - 0.25              # ~ g'''/g'
        denom = 1.0 + newton * (h2 + h3 * newton / 6.0)
        step = newton * (1.0 + 0.5 * h2 * newton) / denom if abs(denom) > 1e-14 else newton

        s_new = s + step
        if not np.isfinite(s_new) or s_new <= lo or s_new >= hi:
            s_new = 0.5 * (lo + hi)                              # safeguarded bisection
        if abs(s_new - s) <= tol * max(s, 1.0):
            s, converged = s_new, True
            break
        s = s_new

    if not converged:
        # Guaranteed backstop on the bracket we have maintained throughout.
        try:
            s = brentq(lambda z: np.log(max(_normalised_black(x, z), 1e-320)) - log_beta,
                       lo, hi, xtol=1e-14, rtol=8.9e-16, maxiter=200)
        except (ValueError, RuntimeError):
            return IVResult(np.nan, "NOT_CONVERGED", it)

    sigma = s / np.sqrt(T)
    # vega in *price* units, for the spread filter and for downstream weights
    vega_price = df * np.sqrt(F * K) * _normalised_vega(x, s) * np.sqrt(T)

    if not np.isfinite(sigma) or sigma <= 0:
        return IVResult(np.nan, "NOT_CONVERGED", it, vega_price)
    if spread is not None:
        if vega_price <= 1e-10 or (0.5 * spread) / vega_price > max_spread_vols:
            return IVResult(sigma, "NO_VEGA", it, vega_price)
    return IVResult(float(sigma), "OK", it, float(vega_price))


def implied_vol(prices, F, K, T, cp, df=1.0, **kw):
    """Vectorised wrapper.  Returns (sigma, status_code, vega) arrays.

    status_code: 0 OK, 1 NO_ARBITRAGE, 2 NO_VEGA, 3 NOT_CONVERGED
    """
    codes = {"OK": 0, "NO_ARBITRAGE": 1, "NO_VEGA": 2, "NOT_CONVERGED": 3}
    prices, K, cp = np.atleast_1d(prices), np.atleast_1d(K), np.atleast_1d(cp)
    n = max(len(prices), len(K), len(cp))
    F_ = np.broadcast_to(np.atleast_1d(F), (n,))
    T_ = np.broadcast_to(np.atleast_1d(T), (n,))
    df_ = np.broadcast_to(np.atleast_1d(df), (n,))
    spread = kw.pop("spread", None)
    sp_ = np.broadcast_to(np.atleast_1d(spread), (n,)) if spread is not None else [None] * n

    sig = np.empty(n)
    sts = np.empty(n, dtype=int)
    veg = np.empty(n)
    for i in range(n):
        r = implied_vol_scalar(
            float(prices[i]), float(F_[i]), float(K[i]), float(T_[i]),
            int(cp[i]), float(df_[i]), spread=sp_[i], **kw
        )
        sig[i], sts[i], veg[i] = r.sigma, codes[r.status], r.vega
    return sig, sts, veg
