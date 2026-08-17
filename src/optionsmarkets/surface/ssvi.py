"""SSVI surface backbone with closed-form no-arbitrage conditions.

    w(k, theta) = (theta/2) * { 1 + rho*phi(theta)*k
                                + sqrt( (phi(theta)*k + rho)^2 + (1 - rho^2) ) }

where ``theta_T = w(0,T)`` is the ATM total-variance term structure, read
straight off the market, and ``phi`` is a one- or two-parameter function of
theta.  The whole surface is therefore two global numbers ``(rho, eta)`` plus
an observed curve.

We use the **modified power law**::

    phi(theta) = eta / ( theta^gamma * (1 + theta)^(1-gamma) ),  gamma = 1/2 default

because it is the only common choice that is butterfly-arbitrage-free at
*every* maturity, under the single parameter inequality ``eta*(1+|rho|) <= 2``.
The plain power law ``eta*theta^-gamma`` fails calendar-freedom beyond a finite
maturity, which is a nasty failure mode: the surface looks fine on the front
months and silently goes arbitrageable in the back.

Why a backbone at all, when per-slice SVI fits the data better: independent
slice fits develop local pathologies (crossing, negative density in a wing)
that are invisible until you try to price a calendar or a butterfly off them.
The production architecture is SSVI as the arbitrage-free skeleton, then
per-slice raw-SVI refinement *constrained to stay inside the SSVI envelope*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ["SSVIParams", "ssvi_w", "ssvi_phi", "no_butterfly", "no_calendar", "fit_ssvi"]


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float = 0.5

    def ok(self) -> bool:
        return abs(self.rho) < 1 and self.eta > 0 and 0 < self.gamma < 1 \
            and self.eta * (1.0 + abs(self.rho)) <= 2.0 + 1e-12


def ssvi_phi(theta, p: SSVIParams):
    theta = np.maximum(np.asarray(theta, float), 1e-12)
    return p.eta / (theta**p.gamma * (1.0 + theta) ** (1.0 - p.gamma))


def ssvi_w(k, theta, p: SSVIParams):
    k = np.asarray(k, float)
    theta = np.asarray(theta, float)
    ph = ssvi_phi(theta, p)
    z = ph * k
    return 0.5 * theta * (1.0 + p.rho * z + np.sqrt((z + p.rho) ** 2 + (1.0 - p.rho**2)))


def no_butterfly(theta, p: SSVIParams) -> tuple[bool, float, float]:
    """Gatheral-Jacquier Theorem 4.2 (sufficient).

        theta*phi*(1+|rho|)   <  4
        theta*phi^2*(1+|rho|) <= 4
    """
    ph = ssvi_phi(theta, p)
    c1 = float(np.max(theta * ph * (1.0 + abs(p.rho))))
    c2 = float(np.max(theta * ph**2 * (1.0 + abs(p.rho))))
    return (c1 < 4.0 and c2 <= 4.0), c1, c2


def no_calendar(theta_curve, p: SSVIParams) -> tuple[bool, float]:
    """GJ Theorem 4.1: theta_T non-decreasing, and 0 <= d(theta*phi)/dtheta
    <= (1/rho^2)(1 + sqrt(1-rho^2)) * phi.

    Returns (ok, worst_violation).  Violation is reported in absolute units so
    it can be logged and trended -- a surface that is drifting toward the
    boundary is a leading indicator of a data problem upstream.
    """
    th = np.asarray(theta_curve, float)
    th = th[np.isfinite(th)]
    if th.size < 2:
        return True, 0.0
    mono = float(np.min(np.diff(th)))
    psi = th * ssvi_phi(th, p)
    dpsi = np.diff(psi) / np.maximum(np.diff(th), 1e-12)
    ub = (1.0 + np.sqrt(max(1.0 - p.rho**2, 0.0))) / max(p.rho**2, 1e-12) * ssvi_phi(th[:-1], p)
    worst = float(max(-mono, float(np.max(-dpsi)) if dpsi.size else 0.0,
                      float(np.max(dpsi - ub)) if dpsi.size else 0.0))
    return worst <= 1e-10, worst


def fit_ssvi(k_by_slice, w_by_slice, theta_curve, weights_by_slice=None,
             gamma: float = 0.5, n_rho: int = 81) -> tuple[SSVIParams, float]:
    """Calibrate (rho, eta) globally against all slices at once.

    Follows Corbetta et al.: sample ``rho`` on a grid, and for each rho run a
    1-D bounded minimisation on ``eta`` inside its feasible interval
    ``(0, 2/(1+|rho|)]``.  Because the feasible set is enforced by the search
    bounds rather than by a penalty, an infeasible surface is not merely
    discouraged -- it is unreachable.
    """
    theta_curve = np.asarray(theta_curve, float)
    if weights_by_slice is None:
        weights_by_slice = [np.ones_like(np.asarray(kk, float)) for kk in k_by_slice]

    def sse(rho, eta):
        p = SSVIParams(float(rho), float(eta), gamma)
        tot = 0.0
        for kk, ww, th, wt in zip(k_by_slice, w_by_slice, theta_curve,
                                  weights_by_slice, strict=True):
            kk = np.asarray(kk, float); ww = np.asarray(ww, float); wt = np.asarray(wt, float)
            tot += float(np.sum(wt * (ssvi_w(kk, th, p) - ww) ** 2))
        return tot

    best, best_val = None, np.inf
    for rho in np.linspace(-0.999, 0.999, n_rho):
        eta_max = 2.0 / (1.0 + abs(rho)) - 1e-9
        # rho bound as a default argument: the lambda is consumed inside this
        # iteration today, but late binding here would be a silent trap.
        res = minimize_scalar(lambda e, r=rho: sse(r, e), bounds=(1e-6, eta_max),
                              method="bounded",
                              options={"xatol": 1e-10})
        if res.fun < best_val:
            best_val, best = res.fun, SSVIParams(float(rho), float(res.x), gamma)
    return best, float(best_val)
