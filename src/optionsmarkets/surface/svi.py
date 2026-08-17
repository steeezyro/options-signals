"""Raw SVI slice: parameterisation, arbitrage diagnostics, quasi-explicit fit.

Total implied variance as a function of log-moneyness ``k = ln(K/F)``::

    w(k) = a + b * ( rho*(k - m) + sqrt( (k - m)^2 + sigma^2 ) )

with ``b >= 0``, ``|rho| < 1``, ``sigma > 0`` and ``a + b*sigma*sqrt(1-rho^2) >= 0``.

Calibration follows the Zeliade quasi-explicit scheme.  The key observation is
that after the change of variable ``y = (k - m)/sigma`` the model is *linear*
in three of its five parameters::

    w~(y) = a~ + d*y + c*sqrt(y^2 + 1),   c = b*sigma,  d = rho*b*sigma

so for fixed ``(m, sigma)`` the inner problem is a 3-parameter convex
least-squares over a polytope, solvable exactly, and the outer problem is only
a 2-D search.  Naive 5-parameter nonlinear least squares on the same data
falls into a local minimum often enough that you cannot run it unattended.

Fitting is done against *bid-ask-normalised* residuals with an
epsilon-insensitive core: any model vol that lands inside the quoted spread
contributes exactly zero loss.  Chasing the mid of a 40-vol-wide quote is how
a surface acquires a phantom smile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = ["SVIParams", "svi_w", "svi_g", "svi_jw", "fit_svi_slice", "SliceFit"]


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self):
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    @property
    def w_min(self) -> float:
        return self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2)

    @property
    def lee_slope(self) -> float:
        """b(1+|rho|); Roger Lee's moment formula caps this at 2 in total-variance units."""
        return self.b * (1.0 + abs(self.rho))

    def domain_ok(self) -> bool:
        return (
            self.b >= 0 and abs(self.rho) < 1 and self.sigma > 0
            and self.w_min >= 0 and self.lee_slope <= 2.0 + 1e-9
        )


def svi_w(k, p: SVIParams):
    k = np.asarray(k, float)
    z = k - p.m
    return p.a + p.b * (p.rho * z + np.sqrt(z * z + p.sigma**2))


def svi_dw(k, p: SVIParams):
    z = np.asarray(k, float) - p.m
    return p.b * (p.rho + z / np.sqrt(z * z + p.sigma**2))


def svi_d2w(k, p: SVIParams):
    z = np.asarray(k, float) - p.m
    return p.b * p.sigma**2 / (z * z + p.sigma**2) ** 1.5


def svi_g(k, p: SVIParams):
    """Gatheral-Jacquier ``g(k)``.  The slice is butterfly-arbitrage-free iff
    ``g(k) >= 0`` everywhere (plus the tail condition, implied by the Lee bound).

    ``p(k) = g(k) / sqrt(2 pi w) * exp(-d_-^2/2)`` is the risk-neutral density,
    so ``g < 0`` literally means negative probability mass -- a butterfly you
    could buy for a negative price.
    """
    k = np.asarray(k, float)
    w = svi_w(k, p)
    wp = svi_dw(k, p)
    wpp = svi_d2w(k, p)
    w = np.maximum(w, 1e-12)
    return (1.0 - k * wp / (2.0 * w)) ** 2 - (wp**2 / 4.0) * (1.0 / w + 0.25) + wpp / 2.0


def svi_density(k, p: SVIParams):
    """Risk-neutral density in log-moneyness (Breeden-Litzenberger, in SVI form)."""
    k = np.asarray(k, float)
    w = np.maximum(svi_w(k, p), 1e-12)
    dm = -k / np.sqrt(w) - np.sqrt(w) / 2.0
    return svi_g(k, p) / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * dm**2)


def svi_jw(p: SVIParams, t: float) -> dict:
    """Trader-interpretable SVI-JW coordinates: ATM variance, ATM skew, wings."""
    w_t = svi_w(0.0, p)
    v_t = w_t / t
    root = np.sqrt(p.m**2 + p.sigma**2)
    return {
        "v": float(v_t),
        "psi": float(0.5 * p.b / np.sqrt(w_t) * (p.rho - p.m / root)),
        "p_wing": float(p.b * (1.0 - p.rho) / np.sqrt(w_t)),
        "c_wing": float(p.b * (1.0 + p.rho) / np.sqrt(w_t)),
        "v_min": float(p.w_min / t),
    }


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------

@dataclass
class SliceFit:
    params: SVIParams
    rmse_vol: float
    n_used: int
    min_g: float
    inside_spread_frac: float
    ok: bool
    detail: str = ""


def _inner_linear(y, w_obs, weights, sigma):
    """Exact solve of the 3-parameter convex problem for fixed (m, sigma).

    minimise  sum_i weight_i * ( a~ + d*y_i + c*sqrt(y_i^2+1) - w_i )^2
    subject to the Zeliade polytope
        0 <= c <= 4*sigma,   |d| <= c,   |d| <= 4*sigma - c,   0 <= a~ <= max w_i

    Implemented as a bounded NNLS-style projection: solve the unconstrained
    normal equations, and if the solution leaves the polytope, fall back to a
    small projected solve.  For the sizes involved (3 unknowns) this is
    microseconds and is far more reliable than handing the whole thing to a
    general nonlinear optimiser.
    """
    X = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1.0)])
    W = np.sqrt(weights)
    Xw, yw = X * W[:, None], w_obs * W
    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:                                # pragma: no cover
        return None
    a_t, d, c = beta
    c_max = 4.0 * sigma
    a_max = float(np.max(w_obs))
    if 0 <= c <= c_max and abs(d) <= c and abs(d) <= c_max - c and 0 <= a_t <= a_max:
        return a_t, d, c

    # Projected fallback: grid the two binding coordinates (c, d), solve for a~
    # exactly at each grid point, keep the best feasible triple.  Convex
    # objective on a convex set, so a modest grid is sufficient.
    #
    # Fully vectorised over the grid.  This is not a micro-optimisation: the
    # outer (m, sigma) search evaluates this inner problem thousands of times per
    # slice, and the fallback fires whenever the unconstrained solution leaves
    # the polytope -- which near the constraint boundary is most of the time.
    # As a Python double loop it dominated the entire pipeline's runtime by an
    # order of magnitude. The grid, the objective and the tie-breaking order are
    # unchanged, so the result is identical.
    cc = np.linspace(1e-8, c_max, 40)                      # (nc,)
    d_hi = np.minimum(cc, c_max - cc)                      # (nc,)
    lin = np.linspace(-1.0, 1.0, 25)                       # (nd,)
    dd = d_hi[:, None] * lin[None, :]                      # (nc, nd) == linspace(-d_hi, d_hi)

    # resid[i, j, :] = yw - Xw[:,1]*dd[i,j] - Xw[:,2]*cc[i]
    resid = (yw[None, None, :]
             - Xw[None, None, :, 1] * dd[:, :, None]
             - Xw[None, None, :, 2] * cc[:, None, None])
    denom = max(float(np.sum(W * W)), 1e-300)
    # Xw[:, 0] is exactly W, since the first design column is all ones.
    aa = np.clip(np.sum(W[None, None, :] * resid, axis=-1) / denom, 0.0, a_max)
    obj = np.sum((resid - W[None, None, :] * aa[:, :, None]) ** 2, axis=-1)

    flat = int(np.argmin(obj))
    i, j = divmod(flat, dd.shape[1])
    return float(aa[i, j]), float(dd[i, j]), float(cc[i])


def fit_svi_slice(
    k, iv, t: float,
    iv_spread=None,
    vega=None,
    g_grid_halfwidth: float = 5.0,
    n_starts: int = 5,
) -> SliceFit:
    """Fit one expiry slice.

    Parameters
    ----------
    k          log-moneyness relative to the slice's OWN forward
    iv         implied volatilities (decimals)
    t          year fraction
    iv_spread  half-width of the quoted spread in vol terms.  Residuals are
               divided by this, and residuals smaller than it are zeroed
               (epsilon-insensitive loss).
    vega       optional additional weight; use when spreads are unavailable.
    """
    k = np.asarray(k, float)
    iv = np.asarray(iv, float)
    good = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[good], iv[good]
    if k.size < 5:
        return SliceFit(SVIParams(np.nan, np.nan, np.nan, np.nan, np.nan),
                        np.nan, int(k.size), np.nan, np.nan, False, "too few quotes")

    w_obs = iv**2 * t
    if iv_spread is not None:
        s = np.asarray(iv_spread, float)[good]
        s = np.where(np.isfinite(s) & (s > 1e-4), s, np.nanmedian(s[np.isfinite(s)]) or 0.01)
        weights = 1.0 / s**2
    elif vega is not None:
        v = np.asarray(vega, float)[good]
        weights = np.maximum(v, 1e-8) ** 2
    else:
        weights = np.ones_like(k)
    weights = weights / np.mean(weights)

    def unpack(m, sig):
        y = (k - m) / sig
        sol = _inner_linear(y, w_obs, weights, sig)
        if sol is None:
            return None, np.inf
        a_t, d, c = sol
        b = c / sig
        rho = float(np.clip(d / c, -0.999999, 0.999999)) if c > 1e-12 else 0.0
        p = SVIParams(float(a_t), float(b), rho, float(m), float(sig))
        w_hat = svi_w(k, p)
        resid = (w_hat - w_obs) * np.sqrt(weights)
        return p, float(np.sum(resid**2))

    def obj(theta):
        m, log_sig = theta
        sig = float(np.exp(log_sig))
        if not (1e-4 < sig < 5.0):
            return 1e12
        _, val = unpack(m, sig)
        return val

    atm_var = float(np.interp(0.0, np.sort(k), w_obs[np.argsort(k)]))
    starts = [
        (0.0, np.log(0.1)), (0.0, np.log(0.4)),
        (-0.05, np.log(0.2)), (0.05, np.log(0.2)),
        (float(np.mean(k)), np.log(max(np.std(k), 0.05))),
    ][:max(n_starts, 1)]

    best_p, best_val = None, np.inf
    for s0 in starts:
        res = minimize(obj, np.array(s0), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
        if res.fun < best_val:
            p, val = unpack(res.x[0], float(np.exp(res.x[1])))
            if p is not None and val < best_val:
                best_p, best_val = p, val

    if best_p is None:
        return SliceFit(SVIParams(*[np.nan] * 5), np.nan, int(k.size), np.nan, np.nan, False, "no fit")

    w_hat = svi_w(k, best_p)
    iv_hat = np.sqrt(np.maximum(w_hat, 0.0) / t)
    rmse = float(np.sqrt(np.mean((iv_hat - iv) ** 2)))
    inside = float(np.mean(np.abs(iv_hat - iv) <= np.asarray(iv_spread, float)[good])) \
        if iv_spread is not None else np.nan

    span = g_grid_halfwidth * np.sqrt(max(atm_var, 1e-6))
    kg = np.linspace(-span, span, 801)
    min_g = float(np.min(svi_g(kg, best_p)))

    ok = bool(best_p.domain_ok() and min_g > -1e-8 and np.isfinite(rmse))
    detail = "" if ok else (
        f"domain_ok={best_p.domain_ok()} min_g={min_g:.3e} lee={best_p.lee_slope:.3f}"
    )
    return SliceFit(best_p, rmse, int(k.size), min_g, inside, ok, detail)


def crossedness(p_near: SVIParams, p_far: SVIParams, k_grid) -> float:
    """max_k (w_near(k) - w_far(k))+ -- the calendar-arbitrage diagnostic.

    Total variance must be non-decreasing in maturity at every fixed k.
    Geometrically: plotted as w vs k, slices must never cross.  Any positive
    value here is a calendar spread with a negative price.
    """
    d = svi_w(k_grid, p_near) - svi_w(k_grid, p_far)
    return float(np.max(np.maximum(d, 0.0)))
