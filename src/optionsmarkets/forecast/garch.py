"""GJR-GARCH: the asymmetry cross-check on the HAR forecast.

BLUEPRINT.md section 6.2 is unambiguous about why this is here: **for equity
index volatility, a model with no asymmetry term is wrong.**  Representative SPX
estimates are alpha ~ 0.02, xi ~ 0.12, beta ~ 0.90 -- the leverage effect
outweighs the symmetric ARCH effect by roughly six to one.  A symmetric GARCH(1,1)
fitted to the same data absorbs the leverage effect into a larger alpha and then
over-forecasts after rallies and under-forecasts after selloffs, which for a
short-premium book is the wrong error in the more expensive direction.

    sigma^2_t = omega + (alpha + xi * 1[r_{t-1} < 0]) * r^2_{t-1} + beta * sigma^2_{t-1}

Two things this module is careful about, both of which are easy to get wrong:

**The h-step forecast uses the right persistence.**  Iterating the recursion
forward needs ``E[(alpha + xi*1[r<0]) r^2]``, and under a distribution symmetric
about zero the indicator fires half the time, so the effective one-step
persistence is ``alpha + xi/2 + beta`` -- not ``alpha + xi + beta``.  Using the
latter overstates persistence by ~0.06 for SPX-like parameters, which compounds
over a 30-day horizon into a materially too-slow mean reversion.

**The forecast is horizon-matched and returned in the same units as HAR** -- the
annualised average volatility over the next h days -- so the two are directly
comparable and can be run against each other by the section 13.3 evaluation
suite rather than merely coexisting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from .realized import ANNUALISER

__all__ = ["GJRGarchFit", "GJRGarch"]


@dataclass
class GJRGarchFit:
    omega: float
    alpha: float
    xi: float                        # the asymmetry (leverage) term
    beta: float
    nu: float                        # Student-t dof; inf for normal errors
    loglik: float
    n: int
    dist: str
    sigma2: np.ndarray = field(repr=False, default=None)

    @property
    def persistence(self) -> float:
        """alpha + xi/2 + beta -- the FORECASTING persistence.

        The indicator fires on half the observations under a symmetric error
        distribution, so half of xi enters the expected recursion.  Using
        alpha + xi + beta here is the standard error and it makes the model look
        near-unit-root when it is not.
        """
        return float(self.alpha + 0.5 * self.xi + self.beta)

    @property
    def half_life_days(self) -> float:
        p = self.persistence
        if not (0 < p < 1):
            return np.inf
        return float(np.log(0.5) / np.log(p))

    @property
    def long_run_variance(self) -> float:
        p = self.persistence
        return float(self.omega / (1.0 - p)) if 0 < p < 1 else np.nan

    @property
    def leverage_dominates(self) -> bool:
        """True when the asymmetry term exceeds the symmetric ARCH term.

        The expected state for an equity index.  If this comes out False on SPX
        data, suspect the return series before believing the model.
        """
        return bool(self.xi > self.alpha)

    def summary(self) -> dict:
        return {
            "omega": self.omega, "alpha": self.alpha, "xi": self.xi, "beta": self.beta,
            "nu": self.nu, "dist": self.dist, "loglik": self.loglik, "n": self.n,
            "persistence": self.persistence, "half_life_days": self.half_life_days,
            "long_run_vol_annualised": float(np.sqrt(max(self.long_run_variance, 0.0)
                                                     * ANNUALISER)),
            "leverage_dominates_arch": self.leverage_dominates,
        }


class GJRGarch:
    """GJR-GARCH(1,1) by maximum likelihood, on DAILY LOG RETURNS.

    Parameters
    ----------
    dist
        ``'normal'`` or ``'t'``.  Student-t is the better description of daily
        equity returns and it matters for the *tails* the sizing layer cares
        about, but the conditional-variance path is similar either way; fit both
        and compare the likelihood.

    Note on what this is for.  It is a **cross-check**, not the production
    forecast.  HAR is fitted on realised variance and directly targets the
    h-day average; GARCH is fitted on returns and gets there by iterating a
    recursion.  When they disagree materially, that disagreement is information
    about the regime -- and it is a better signal than either model's own
    standard error, which only knows about its own specification.
    """

    def __init__(self, dist: str = "normal", mean: str = "zero"):
        if dist not in ("normal", "t"):
            raise ValueError("dist must be 'normal' or 't'")
        self.dist = dist
        self.mean = mean
        self.fit_: GJRGarchFit | None = None
        self._r: np.ndarray | None = None

    # ---- likelihood ---------------------------------------------------
    @staticmethod
    def _recurse(r: np.ndarray, omega, alpha, xi, beta) -> np.ndarray:
        n = r.size
        s2 = np.empty(n)
        s2[0] = max(float(np.var(r)), 1e-12)
        for t in range(1, n):
            neg = 1.0 if r[t - 1] < 0.0 else 0.0
            s2[t] = omega + (alpha + xi * neg) * r[t - 1] ** 2 + beta * s2[t - 1]
            if not np.isfinite(s2[t]) or s2[t] <= 0:
                s2[t] = 1e-12
        return s2

    def _nll(self, theta: np.ndarray, r: np.ndarray) -> float:
        if self.dist == "t":
            omega, alpha, xi, beta, log_nu_m2 = theta
            nu = 2.0 + float(np.exp(log_nu_m2))          # keeps nu > 2 so variance exists
        else:
            omega, alpha, xi, beta = theta
            nu = np.inf
        # Constraints as hard rejections rather than penalties: a negative
        # variance parameter is not a worse fit, it is not a model.
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + xi) < 0:
            return 1e12
        if alpha + 0.5 * xi + beta >= 0.999:             # stationarity
            return 1e12
        s2 = self._recurse(r, omega, alpha, xi, beta)
        if self.dist == "normal":
            return 0.5 * float(np.sum(np.log(2 * np.pi * s2) + r**2 / s2))
        z2 = r**2 / s2
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
        return -float(np.sum(c - 0.5 * np.log(s2)
                             - (nu + 1) / 2 * np.log1p(z2 / (nu - 2))))

    # ---- fit ----------------------------------------------------------
    def fit(self, returns) -> GJRGarchFit:
        r = np.asarray(pd.Series(returns).astype(float).dropna(), float)
        r = r[np.isfinite(r)]
        if r.size < 250:
            raise ValueError(f"GJR-GARCH needs a few hundred returns to identify the "
                             f"asymmetry term; got {r.size}")
        if self.mean == "demean":
            r = r - r.mean()
        self._r = r
        v = float(np.var(r))

        # Start at the SPX-like values the blueprint cites, plus a couple of
        # alternatives: the likelihood is not concave and a single start on a
        # 5-parameter problem is how you get a plausible wrong answer.
        starts = [
            [v * 0.05, 0.02, 0.12, 0.90],
            [v * 0.10, 0.05, 0.08, 0.85],
            [v * 0.02, 0.01, 0.18, 0.88],
        ]
        best, best_nll = None, np.inf
        for s0 in starts:
            th0 = np.array(s0 + ([np.log(6.0)] if self.dist == "t" else []), float)
            res = minimize(self._nll, th0, args=(r,), method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-12})
            if res.fun < best_nll:
                best_nll, best = float(res.fun), res.x
        if best is None or not np.isfinite(best_nll):        # pragma: no cover
            raise RuntimeError("GJR-GARCH did not converge from any start")

        if self.dist == "t":
            omega, alpha, xi, beta, log_nu = best
            nu = 2.0 + float(np.exp(log_nu))
        else:
            omega, alpha, xi, beta = best
            nu = np.inf
        s2 = self._recurse(r, omega, alpha, xi, beta)
        self.fit_ = GJRGarchFit(float(omega), float(alpha), float(xi), float(beta),
                                float(nu), -best_nll, int(r.size), self.dist, s2)
        return self.fit_

    # ---- forecast -----------------------------------------------------
    def forecast_variance_path(self, h: int) -> np.ndarray:
        """One-step-through-h-step conditional variances, per day."""
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        f, r = self.fit_, self._r
        neg = 1.0 if r[-1] < 0.0 else 0.0
        nxt = f.omega + (f.alpha + f.xi * neg) * r[-1] ** 2 + f.beta * f.sigma2[-1]
        path = [max(float(nxt), 1e-16)]
        p = f.persistence
        lr = f.omega
        for _ in range(1, max(int(h), 1)):
            # E[sigma^2_{t+k}] = omega + (alpha + xi/2 + beta) E[sigma^2_{t+k-1}]
            path.append(float(lr + p * path[-1]))
        return np.asarray(path, float)

    def forecast(self, h: int = 21) -> float:
        """Annualised average volatility over the next ``h`` days.

        Same units and same horizon convention as
        :meth:`optionsmarkets.forecast.har.HARModel.predict`, so the two are
        directly substitutable in the pipeline and directly comparable in the
        section 13.3 evaluation.
        """
        path = self.forecast_variance_path(h)
        return float(np.sqrt(max(float(np.mean(path)), 0.0) * ANNUALISER))

    def conditional_vol(self) -> np.ndarray:
        """In-sample annualised conditional volatility, for diagnostics."""
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        return np.sqrt(self.fit_.sigma2 * ANNUALISER)

    def standardised_residuals(self) -> np.ndarray:
        """z_t = r_t / sigma_t.  Should be iid with unit variance.

        Worth checking rather than assuming: leftover ARCH in these is the
        clearest sign the (1,1) order is too low, and leftover skew says the
        asymmetry term is carrying more than it should.
        """
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        return self._r / np.sqrt(self.fit_.sigma2)
