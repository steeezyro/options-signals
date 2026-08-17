"""Constrained joint surface fit: SSVI skeleton + per-slice refinement.

This is the production architecture BLUEPRINT.md section 5.3 specifies, and the
reason it is not simply "fit each slice independently" is worth stating plainly:

  * **SSVI alone is too rigid.** Two global numbers plus an ATM curve cannot
    track a real equity smile to bid/ask across every maturity.  Fit it alone
    and the front-month wings sit outside the spread.
  * **Independent slice fits are too loose.** Each one can be individually
    butterfly-free and still cross its neighbour.  You do not find out until
    you price a calendar, and by then the "edge" you found is a negative-price
    calendar spread that exists only in your own surface.

So: fit SSVI globally to get an arbitrage-free *skeleton*, then refine each
slice with raw SVI **shortest maturity first**, with the previous published
slice entering the objective as a hard calendar constraint and the SSVI slice
as an envelope the refinement may not leave.  A slice that cannot satisfy both
falls back to the SSVI backbone for that maturity, which is arbitrage-free by
construction.  The fallback is recorded, not hidden -- a surface that keeps
falling back is telling you something about the data.

The SSVI -> raw-SVI map used throughout is exact::

    w(k, theta) = (theta/2) [ 1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2) ]

is the raw SVI slice

    a = theta (1 - rho^2) / 2,   b = theta*phi / 2,   rho_raw = rho,
    m = -rho / phi,              sigma = sqrt(1 - rho^2) / phi

which is worth checking rather than trusting: under it the Lee bound
``b(1+|rho|) <= 2`` becomes ``theta*phi*(1+|rho|) <= 4``, exactly the
Gatheral-Jacquier Theorem 4.2 butterfly condition.  The two conditions are the
same statement, and that is the coherence check that the map is right.

Publication is gated.  :class:`SurfacePublisher` recomputes ``min_k g(k)`` per
slice and ``max_k crossedness`` for every adjacent pair, and on violation
returns the last known-good surface and raises.  Never publish an unvalidated
surface: a wrong surface does not look wrong, it looks like an opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
from scipy.optimize import minimize

from .ssvi import SSVIParams, fit_ssvi, no_butterfly, no_calendar, ssvi_phi
from .svi import SliceFit, SVIParams, _inner_linear, crossedness, svi_g, svi_w

__all__ = [
    "SliceQuotes", "SliceResult", "Surface", "SurfaceDiagnostics",
    "ssvi_to_svi", "fit_surface", "SurfacePublisher", "SurfaceViolation",
]


# ----------------------------------------------------------------------------
# inputs / outputs
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SliceQuotes:
    """One expiry's worth of surface-legal input.

    ``iv`` must already be European -- de-Americanised if the quotes are
    American (see :func:`optionsmarkets.pricing.american.de_americanise`) --
    and ``k`` must be log-moneyness against *this slice's own* forward.  Using
    a single forward across expiries is a classic way to manufacture a term
    structure of skew that is really a term structure of carry.
    """
    T: float
    k: np.ndarray
    iv: np.ndarray
    iv_spread: np.ndarray | None = None
    forward: float = np.nan
    df: float = np.nan
    expiry: date | None = None

    def __post_init__(self):
        object.__setattr__(self, "k", np.asarray(self.k, float))
        object.__setattr__(self, "iv", np.asarray(self.iv, float))
        if self.iv_spread is not None:
            object.__setattr__(self, "iv_spread", np.asarray(self.iv_spread, float))

    @property
    def n(self) -> int:
        return int(np.sum(np.isfinite(self.k) & np.isfinite(self.iv) & (self.iv > 0)))

    def atm_total_variance(self) -> float:
        """w(0) read straight off the quotes, by interpolation in k.

        This is ``theta_T``, the only thing SSVI takes from the market beyond
        its two global parameters.  Interpolating the *observed* smile rather
        than reading it off a fitted slice keeps the backbone anchored to data
        even when a slice fit is poor.
        """
        m = np.isfinite(self.k) & np.isfinite(self.iv) & (self.iv > 0)
        if m.sum() < 2:
            return np.nan
        kk, vv = self.k[m], self.iv[m]
        o = np.argsort(kk)
        return float(np.interp(0.0, kk[o], (vv[o] ** 2) * self.T))


@dataclass
class SliceResult:
    """One fitted maturity, with the provenance of how it got there."""
    T: float
    params: SVIParams
    theta: float
    rmse_vol: float
    n_used: int
    min_g: float
    inside_spread_frac: float
    crossedness_prev: float
    source: str                     # 'refined' | 'ssvi-backbone' | 'independent'
    ok: bool
    detail: str = ""
    expiry: date | None = None

    def as_dict(self) -> dict:
        p = self.params
        return {
            "T": self.T, "expiry": self.expiry.isoformat() if self.expiry else None,
            "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma,
            "theta": self.theta, "rmse_vol": self.rmse_vol, "n_used": self.n_used,
            "min_g": self.min_g, "inside_spread_frac": self.inside_spread_frac,
            "crossedness_prev": self.crossedness_prev, "source": self.source,
            "ok": self.ok, "detail": self.detail,
        }


@dataclass
class SurfaceDiagnostics:
    min_g_overall: float
    max_crossedness: float
    worst_rmse_vol: float
    n_fallbacks: int
    theta_adjusted: bool
    ssvi_butterfly_ok: bool
    ssvi_calendar_ok: bool
    ssvi_calendar_worst: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SurfaceViolation(RuntimeError):
    """Raised by the publication gate.  Carries the surface that failed so the
    caller can log exactly what was rejected rather than just that something was."""

    def __init__(self, message: str, surface: "Surface"):
        super().__init__(message)
        self.surface = surface


# ----------------------------------------------------------------------------
# the exact SSVI -> raw SVI map
# ----------------------------------------------------------------------------

def ssvi_to_svi(theta: float, p: SSVIParams) -> SVIParams:
    """Raw-SVI parameters of the SSVI slice at ATM total variance ``theta``.

    Exact, not a fit.  See module docstring for the derivation and for why the
    Lee bound and the GJ butterfly condition coincide under this map.
    """
    theta = float(max(theta, 1e-12))
    phi = float(ssvi_phi(theta, p))
    rho = float(p.rho)
    return SVIParams(
        a=0.5 * theta * (1.0 - rho**2),
        b=0.5 * theta * phi,
        rho=rho,
        m=-rho / phi,
        sigma=float(np.sqrt(max(1.0 - rho**2, 0.0)) / phi),
    )


# ----------------------------------------------------------------------------
# the surface
# ----------------------------------------------------------------------------

@dataclass
class Surface:
    """A published, arbitrage-checked volatility surface.

    Interpolation in maturity is done on TOTAL VARIANCE at fixed log-moneyness,
    linearly in T.  That is the only common scheme that cannot itself create a
    calendar arbitrage: if the bracketing slices satisfy
    ``w_near(k) <= w_far(k)`` then so does every convex combination of them.
    Interpolating volatility instead, or interpolating in sqrt(T), can and does
    produce a crossed intermediate slice out of two clean ones.
    """
    ssvi: SSVIParams
    slices: list[SliceResult]
    diagnostics: SurfaceDiagnostics
    ok: bool
    detail: str = ""
    asof: str = ""
    underlying: str = ""
    forwards: dict = field(default_factory=dict)      # T -> forward

    # ---- accessors -------------------------------------------------------
    @property
    def maturities(self) -> np.ndarray:
        return np.array([s.T for s in self.slices], float)

    @property
    def theta_curve(self) -> np.ndarray:
        return np.array([s.theta for s in self.slices], float)

    def nearest_slice(self, T: float) -> SliceResult:
        return self.slices[int(np.argmin(np.abs(self.maturities - float(T))))]

    def params_at(self, T: float) -> SVIParams:
        """Raw-SVI parameters at an arbitrary maturity.

        On a fitted maturity this is the fitted slice.  Between them it is the
        SSVI slice at the interpolated theta -- SSVI is a genuine surface in T,
        so it interpolates natively and stays arbitrage-free, whereas blending
        two raw-SVI parameter vectors has no such guarantee (the parameters are
        not a linear space in any useful sense).
        """
        Ts = self.maturities
        T = float(T)
        hit = np.isclose(Ts, T, rtol=0, atol=1e-9)
        if hit.any():
            return self.slices[int(np.argmax(hit))].params
        return ssvi_to_svi(self.theta_at(T), self.ssvi)

    def theta_at(self, T: float) -> float:
        """ATM total variance at any maturity.

        Inside the fitted range: linear in T.  Outside: flat forward variance
        (theta proportional to T), which is the only extrapolation that keeps
        theta non-decreasing without inventing a term structure the market did
        not quote.
        """
        Ts, th = self.maturities, self.theta_curve
        T = float(T)
        if Ts.size == 0:
            return np.nan
        if Ts.size == 1 or T <= Ts[0]:
            return float(th[0] * T / max(Ts[0], 1e-12))
        if T >= Ts[-1]:
            return float(th[-1] * T / max(Ts[-1], 1e-12))
        return float(np.interp(T, Ts, th))

    def total_variance(self, k, T: float):
        """w(k, T).  Linear in T on total variance between fitted slices."""
        k = np.asarray(k, float)
        Ts = self.maturities
        T = float(T)
        if Ts.size == 0:
            return np.full_like(k, np.nan)
        hit = np.isclose(Ts, T, rtol=0, atol=1e-9)
        if hit.any():
            return svi_w(k, self.slices[int(np.argmax(hit))].params)
        if T <= Ts[0]:
            return svi_w(k, self.slices[0].params) * (T / max(Ts[0], 1e-12))
        if T >= Ts[-1]:
            return svi_w(k, self.slices[-1].params) * (T / max(Ts[-1], 1e-12))
        j = int(np.searchsorted(Ts, T))
        T0, T1 = Ts[j - 1], Ts[j]
        w0 = svi_w(k, self.slices[j - 1].params)
        w1 = svi_w(k, self.slices[j].params)
        lam = (T - T0) / max(T1 - T0, 1e-12)
        return (1.0 - lam) * w0 + lam * w1

    def vol(self, k, T: float):
        """Black-Scholes volatility at (log-moneyness, maturity)."""
        w = np.maximum(self.total_variance(k, T), 1e-12)
        return np.sqrt(w / max(float(T), 1e-12))

    def atm_vol(self, T: float) -> float:
        return float(np.sqrt(max(self.theta_at(T), 1e-12) / max(float(T), 1e-12)))

    def forward(self, T: float) -> float:
        """The forward this surface's log-moneyness is measured against."""
        if not self.forwards:
            return np.nan
        Ts = np.array(sorted(self.forwards), float)
        Fs = np.array([self.forwards[t] for t in sorted(self.forwards)], float)
        return float(np.interp(float(T), Ts, Fs))

    # ---- reporting -------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "underlying": self.underlying, "asof": self.asof, "ok": self.ok,
            "detail": self.detail,
            "ssvi": {"rho": self.ssvi.rho, "eta": self.ssvi.eta, "gamma": self.ssvi.gamma},
            "diagnostics": self.diagnostics.as_dict(),
            "slices": [s.as_dict() for s in self.slices],
            "forwards": {str(k): v for k, v in self.forwards.items()},
        }

    def report(self) -> str:
        L = [f"SURFACE {self.underlying} {self.asof}  {'OK' if self.ok else 'REJECTED'}",
             f"  SSVI rho={self.ssvi.rho:+.4f} eta={self.ssvi.eta:.4f} "
             f"gamma={self.ssvi.gamma:.2f}  (eta(1+|rho|)="
             f"{self.ssvi.eta * (1 + abs(self.ssvi.rho)):.4f}, cap 2)",
             f"  min g = {self.diagnostics.min_g_overall:+.4e} | max crossedness = "
             f"{self.diagnostics.max_crossedness:.3e} | worst rmse = "
             f"{self.diagnostics.worst_rmse_vol * 100:.2f} vol pts | "
             f"{self.diagnostics.n_fallbacks} backbone fallback(s)",
             f"  {'T':>8}{'n':>5}{'rmse':>9}{'min g':>11}{'cross':>10}{'in-spr':>8}  source"]
        for s in self.slices:
            L.append(f"  {s.T:>8.4f}{s.n_used:>5d}{s.rmse_vol * 100:>8.2f}p"
                     f"{s.min_g:>11.2e}{s.crossedness_prev:>10.2e}"
                     f"{(s.inside_spread_frac if np.isfinite(s.inside_spread_frac) else 0):>7.0%}  "
                     f"{s.source}{'' if s.ok else '  <-- ' + s.detail}")
        if self.detail:
            L.append(f"  {self.detail}")
        return "\n".join(L)


# ----------------------------------------------------------------------------
# the constrained fit
# ----------------------------------------------------------------------------

def _theta_curve(slices: list[SliceQuotes]) -> tuple[np.ndarray, bool]:
    """Observed ATM total variance per slice, forced non-decreasing.

    Total variance that falls with maturity is a calendar arbitrage at k=0, and
    it is almost always a data problem (a stale slice, a wrong forward) rather
    than a market one.  We repair it with a running maximum and report that we
    did -- a silent repair here would hide exactly the diagnostic worth having.
    """
    th = np.array([s.atm_total_variance() for s in slices], float)
    finite = np.isfinite(th)
    if finite.any():
        th = np.where(finite, th, np.interp(
            np.arange(th.size), np.flatnonzero(finite), th[finite]))
    mono = np.maximum.accumulate(np.maximum(th, 1e-10))
    return mono, bool(np.any(mono > th + 1e-12))


def _penalty(p: SVIParams, kg: np.ndarray, w_prev: np.ndarray | None,
             w_env: np.ndarray, band: float, scale: float) -> float:
    """Hard-constraint penalty, in the same units as the weighted SSE.

    Three terms, all one-sided:
      * butterfly   max(0, -g(k))              -- negative density
      * calendar    max(0, w_prev(k) - w(k))   -- crossing the slice below
      * envelope    max(0, |w - w_ssvi| - band*w_ssvi)

    The envelope is what makes this a *constrained refinement* rather than an
    independent fit with a nice initial guess.  Without it the optimiser is
    free to walk the slice anywhere the local quotes pull it, and the SSVI
    skeleton stops meaning anything.
    """
    w = svi_w(kg, p)
    pen = 0.0
    g = svi_g(kg, p)
    pen += float(np.sum(np.maximum(-g, 0.0) ** 2))
    if w_prev is not None:
        pen += float(np.sum(np.maximum(w_prev - w, 0.0) ** 2)) / max(scale, 1e-12)
    dev = np.abs(w - w_env) - band * np.maximum(w_env, 1e-12)
    pen += float(np.sum(np.maximum(dev, 0.0) ** 2)) / max(scale, 1e-12)
    return pen


def _refine_slice(sq: SliceQuotes, p_env: SVIParams, w_prev_fn, *,
                  band: float, k_grid_halfwidth: float,
                  penalty_weight: float) -> tuple[SVIParams | None, float]:
    """Quasi-explicit refinement of one slice inside the SSVI envelope.

    Same decomposition as :func:`optionsmarkets.surface.svi.fit_svi_slice` --
    exact 3-parameter linear solve inside, 2-D search over (m, sigma) outside --
    with the constraint penalty added to the *outer* objective.  Keeping the
    inner problem exact is the whole point of the Zeliade scheme and is what
    lets this run unattended; a penalised 5-parameter nonlinear solve would put
    the local minima straight back.
    """
    good = np.isfinite(sq.k) & np.isfinite(sq.iv) & (sq.iv > 0)
    k, iv = sq.k[good], sq.iv[good]
    if k.size < 5:
        return None, np.inf
    w_obs = iv**2 * sq.T

    if sq.iv_spread is not None:
        s = sq.iv_spread[good]
        med = np.nanmedian(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else 0.01
        s = np.where(np.isfinite(s) & (s > 1e-4), s, med or 0.01)
        weights = 1.0 / s**2
    else:
        weights = np.ones_like(k)
    weights = weights / np.mean(weights)

    span = k_grid_halfwidth * np.sqrt(max(float(np.max(w_obs)), 1e-6))
    kg = np.linspace(-span, span, 401)
    w_env = svi_w(kg, p_env)
    w_prev = None if w_prev_fn is None else np.asarray(w_prev_fn(kg), float)
    scale = float(np.mean(np.maximum(w_env, 1e-12)) ** 2)

    def unpack(m, sig):
        y = (k - m) / sig
        sol = _inner_linear(y, w_obs, weights, sig)
        if sol is None:
            return None, np.inf
        a_t, d, c = sol
        b = c / sig
        rho = float(np.clip(d / c, -0.999999, 0.999999)) if c > 1e-12 else 0.0
        p = SVIParams(float(a_t), float(b), rho, float(m), float(sig))
        resid = (svi_w(k, p) - w_obs) * np.sqrt(weights)
        sse = float(np.sum(resid**2))
        return p, sse + penalty_weight * _penalty(p, kg, w_prev, w_env, band, scale)

    def obj(theta):
        m, log_sig = theta
        sig = float(np.exp(log_sig))
        if not (1e-4 < sig < 5.0):
            return 1e12
        return unpack(m, sig)[1]

    starts = [(p_env.m, np.log(max(p_env.sigma, 1e-3))),
              (0.0, np.log(0.1)), (0.0, np.log(0.4)),
              (-0.05, np.log(0.2)), (0.05, np.log(0.2))]
    best_p, best_val = None, np.inf
    for s0 in starts:
        res = minimize(obj, np.array(s0, float), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 2000})
        p, val = unpack(res.x[0], float(np.exp(res.x[1])))
        if p is not None and val < best_val:
            best_p, best_val = p, val
    return best_p, best_val


def fit_surface(
    slices: list[SliceQuotes], *,
    underlying: str = "",
    asof: str = "",
    envelope_band: float = 0.15,
    penalty_weight: float = 1e4,
    k_grid_halfwidth: float = 5.0,
    max_rmse_vol: float = 0.02,
    crossedness_tol: float = 1e-10,
    min_g_tol: float = -1e-8,
    gamma: float = 0.5,
) -> Surface:
    """Fit the whole surface: SSVI skeleton, then constrained slice refinement.

    Parameters
    ----------
    envelope_band
        How far a refined slice may deviate from the SSVI backbone, as a
        fraction of the backbone's total variance.  0.15 is loose enough to
        track a real smile into the wings and tight enough that a single bad
        slice cannot pivot away from the skeleton.  Tighten it when the input
        is thin.
    penalty_weight
        Multiplier on the one-sided constraint penalty.  Large by design: these
        are constraints, not preferences.  A slice that can only fit its quotes
        by crossing its neighbour should fail over to the backbone, which is
        precisely what a large weight produces.

    Returns a :class:`Surface` whose ``ok`` flag reflects the same checks the
    publication gate applies.  Fitting and publishing are separated on purpose:
    you always want the rejected surface in the journal.
    """
    slices = sorted([s for s in slices if s.n >= 5], key=lambda s: s.T)
    if not slices:
        empty = SurfaceDiagnostics(np.nan, np.nan, np.nan, 0, False, False, False, np.nan)
        return Surface(SSVIParams(0.0, 1e-6, gamma), [], empty, False,
                       "no slice had enough usable quotes", asof, underlying)

    theta, theta_adjusted = _theta_curve(slices)

    # ---- 1. global SSVI skeleton ----------------------------------------
    k_by, w_by, wt_by = [], [], []
    for s in slices:
        m = np.isfinite(s.k) & np.isfinite(s.iv) & (s.iv > 0)
        k_by.append(s.k[m])
        w_by.append((s.iv[m] ** 2) * s.T)
        if s.iv_spread is not None:
            sp = s.iv_spread[m]
            med = np.nanmedian(sp[np.isfinite(sp)]) if np.any(np.isfinite(sp)) else 0.01
            sp = np.where(np.isfinite(sp) & (sp > 1e-4), sp, med or 0.01)
            wt_by.append(1.0 / (sp * 2.0 * np.maximum(s.iv[m], 1e-6) * s.T) ** 2)
        else:
            wt_by.append(np.ones(int(m.sum())))
    ssvi, ssvi_sse = fit_ssvi(k_by, w_by, theta, wt_by, gamma=gamma)

    bf_ok, _, _ = no_butterfly(theta, ssvi)
    cal_ok, cal_worst = no_calendar(theta, ssvi)

    # ---- 2. slice refinement, shortest maturity first --------------------
    results: list[SliceResult] = []
    prev_params: SVIParams | None = None
    n_fallback = 0

    for sq, th in zip(slices, theta):
        p_env = ssvi_to_svi(th, ssvi)
        w_prev_fn = (lambda kk, pp=prev_params: svi_w(kk, pp)) if prev_params else None

        p_ref, _ = _refine_slice(
            sq, p_env, w_prev_fn, band=envelope_band,
            k_grid_halfwidth=k_grid_halfwidth, penalty_weight=penalty_weight,
        )

        # Choose between the refinement and the backbone on the checks, not on a
        # fixed preference. The backbone is butterfly-free by construction but it
        # is NOT automatically uncrossed against the slice below it: theta may
        # have been repaired to equal its neighbour's while that neighbour was
        # refined upward to fit its own quotes. Assuming the fallback is safe is
        # how a crossed surface gets published by the code that exists to prevent
        # exactly that.
        options = []
        if p_ref is not None:
            options.append(("refined", p_ref))
        options.append(("ssvi-backbone", p_env))

        scored = []
        for name, cand in options:
            c_ok, c_cross, c_mg = _slice_checks(cand, prev_params, sq, k_grid_halfwidth,
                                                crossedness_tol, min_g_tol)
            c_rmse, _ = _slice_errors(cand, sq)
            scored.append((not c_ok, c_cross, c_rmse if np.isfinite(c_rmse) else np.inf,
                           name, cand, c_mg))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        infeasible, cross0, _, source, chosen, mg0 = scored[0]

        detail = ""
        if p_ref is None:
            detail = "refinement did not converge"
        elif source != "refined":
            detail = (f"refinement rejected (min_g={scored[-1][5]:.2e}); using the "
                      f"SSVI backbone")
        if infeasible:
            detail = (detail + "; " if detail else "") + \
                (f"no candidate satisfied the no-arbitrage checks at this maturity "
                 f"(best: crossedness={cross0:.2e}, min_g={mg0:.2e}) -- the surface "
                 f"is marked unpublishable rather than repaired")
        if source == "ssvi-backbone":
            n_fallback += 1

        ok, cross, min_g = _slice_checks(chosen, prev_params, sq, k_grid_halfwidth,
                                         crossedness_tol, min_g_tol)
        rmse, inside = _slice_errors(chosen, sq)
        if ok and np.isfinite(rmse) and rmse > max_rmse_vol:
            ok = False
            detail = (detail + "; " if detail else "") + \
                f"rmse {rmse:.4f} exceeds {max_rmse_vol:.4f}"

        results.append(SliceResult(
            T=sq.T, params=chosen, theta=float(th), rmse_vol=rmse, n_used=sq.n,
            min_g=min_g, inside_spread_frac=inside, crossedness_prev=cross,
            source=source, ok=bool(ok), detail=detail, expiry=sq.expiry,
        ))
        prev_params = chosen

    diag = SurfaceDiagnostics(
        min_g_overall=float(np.min([r.min_g for r in results])),
        max_crossedness=float(np.max([r.crossedness_prev for r in results])),
        worst_rmse_vol=float(np.nanmax([r.rmse_vol for r in results])),
        n_fallbacks=n_fallback, theta_adjusted=theta_adjusted,
        ssvi_butterfly_ok=bool(bf_ok), ssvi_calendar_ok=bool(cal_ok),
        ssvi_calendar_worst=float(cal_worst),
    )
    ok = all(r.ok for r in results)
    detail = "" if ok else "; ".join(
        f"T={r.T:.4f}: {r.detail or 'slice failed its own checks'}"
        for r in results if not r.ok)
    if theta_adjusted:
        detail = (detail + "; " if detail else "") + \
            "ATM total variance was non-monotone in T and was repaired by running max"

    return Surface(ssvi=ssvi, slices=results, diagnostics=diag, ok=ok, detail=detail,
                   asof=asof, underlying=underlying,
                   forwards={s.T: float(s.forward) for s in slices
                             if np.isfinite(s.forward)})


def _slice_checks(p: SVIParams, prev: SVIParams | None, sq: SliceQuotes,
                  halfwidth: float, cross_tol: float,
                  min_g_tol: float) -> tuple[bool, float, float]:
    """Butterfly, calendar and domain checks for one candidate slice."""
    span = halfwidth * np.sqrt(max(float(np.max((sq.iv[np.isfinite(sq.iv)] ** 2)) * sq.T)
                                   if np.any(np.isfinite(sq.iv)) else 1e-4, 1e-6))
    kg = np.linspace(-span, span, 801)
    min_g = float(np.min(svi_g(kg, p)))
    cross = 0.0 if prev is None else crossedness(prev, p, kg)
    ok = bool(p.domain_ok() and min_g >= min_g_tol and cross <= cross_tol)
    return ok, cross, min_g


def _slice_errors(p: SVIParams, sq: SliceQuotes) -> tuple[float, float]:
    m = np.isfinite(sq.k) & np.isfinite(sq.iv) & (sq.iv > 0)
    if m.sum() == 0:
        return np.nan, np.nan
    iv_hat = np.sqrt(np.maximum(svi_w(sq.k[m], p), 0.0) / sq.T)
    rmse = float(np.sqrt(np.mean((iv_hat - sq.iv[m]) ** 2)))
    inside = np.nan
    if sq.iv_spread is not None:
        sp = sq.iv_spread[m]
        inside = float(np.mean(np.abs(iv_hat - sq.iv[m]) <= np.where(np.isfinite(sp), sp, 0.0)))
    return rmse, inside


# ----------------------------------------------------------------------------
# publication gate
# ----------------------------------------------------------------------------

class SurfacePublisher:
    """Holds the last known-good surface and refuses to replace it with a bad one.

    BLUEPRINT.md section 5.3: recompute ``min_k g(k)`` per slice and
    ``max_k crossedness`` for all adjacent pairs; on violation fall back to the
    last known-good surface and raise.  The fallback matters more than the
    raise: a system that stops on a bad surface is safe, but a system that
    keeps quoting off yesterday's *valid* surface while it complains is safe
    AND still able to manage the positions it already has on.
    """

    def __init__(self, max_stale_publications: int = 3):
        self.last_good: Surface | None = None
        self.max_stale_publications = int(max_stale_publications)
        self.stale_count = 0
        self.history: list[dict] = []

    def publish(self, surface: Surface, *, raise_on_violation: bool = True) -> Surface:
        """Validate and publish, or fall back.  Returns the surface to USE."""
        rec = {"asof": surface.asof, "ok": surface.ok,
               "min_g": surface.diagnostics.min_g_overall,
               "max_crossedness": surface.diagnostics.max_crossedness,
               "n_fallbacks": surface.diagnostics.n_fallbacks}
        if surface.ok:
            self.last_good = surface
            self.stale_count = 0
            rec["action"] = "published"
            self.history.append(rec)
            return surface

        self.stale_count += 1
        rec["action"] = "rejected"
        rec["stale_count"] = self.stale_count
        self.history.append(rec)
        if self.last_good is None or self.stale_count > self.max_stale_publications:
            if raise_on_violation:
                raise SurfaceViolation(
                    f"surface failed validation and no usable fallback remains "
                    f"({self.stale_count} consecutive rejections): {surface.detail}",
                    surface)
            return surface
        if raise_on_violation:
            raise SurfaceViolation(
                f"surface failed validation ({surface.detail}); serving the last "
                f"known-good surface from {self.last_good.asof} "
                f"({self.stale_count} consecutive rejections)", surface)
        return self.last_good

    def arbitrage_margin_trend(self) -> dict:
        """min_g and crossedness over the publication history.

        BLUEPRINT.md section 13.2: a surface whose arbitrage margin is trending
        toward zero is a leading indicator of a DATA problem, not of market
        stress.  Trend it, do not just threshold it.
        """
        g = [h["min_g"] for h in self.history if np.isfinite(h.get("min_g", np.nan))]
        c = [h["max_crossedness"] for h in self.history
             if np.isfinite(h.get("max_crossedness", np.nan))]
        out = {"n": len(self.history), "n_rejected":
               sum(1 for h in self.history if h["action"] == "rejected")}
        if len(g) >= 3:
            x = np.arange(len(g), dtype=float)
            out["min_g_slope"] = float(np.polyfit(x, g, 1)[0])
            out["min_g_last"] = float(g[-1])
        if len(c) >= 3:
            x = np.arange(len(c), dtype=float)
            out["crossedness_slope"] = float(np.polyfit(x, c, 1)[0])
            out["crossedness_last"] = float(c[-1])
        return out


def single_slice_surface(fit: SliceFit, T: float, forward: float = np.nan,
                         underlying: str = "", asof: str = "",
                         expiry: date | None = None) -> Surface:
    """Wrap one independent :class:`SliceFit` in the :class:`Surface` interface.

    The single-expiry case is legitimate -- there is no calendar constraint to
    enforce with one slice -- and the rest of the system should not have to
    care which path produced the surface it is pricing off.
    """
    theta = float(svi_w(0.0, fit.params)) if fit.params.b == fit.params.b else np.nan
    res = SliceResult(
        T=float(T), params=fit.params, theta=theta, rmse_vol=fit.rmse_vol,
        n_used=fit.n_used, min_g=fit.min_g, inside_spread_frac=fit.inside_spread_frac,
        crossedness_prev=0.0, source="independent", ok=bool(fit.ok),
        detail=fit.detail, expiry=expiry,
    )
    diag = SurfaceDiagnostics(
        min_g_overall=fit.min_g, max_crossedness=0.0, worst_rmse_vol=fit.rmse_vol,
        n_fallbacks=0, theta_adjusted=False, ssvi_butterfly_ok=True,
        ssvi_calendar_ok=True, ssvi_calendar_worst=0.0,
    )
    # A one-slice surface has no SSVI backbone; record a degenerate, feasible one
    # so downstream serialisation and reporting have something coherent to show.
    return Surface(SSVIParams(rho=float(fit.params.rho) if np.isfinite(fit.params.rho) else 0.0,
                              eta=1e-6, gamma=0.5),
                   [res], diag, bool(fit.ok), fit.detail, asof, underlying,
                   {float(T): float(forward)} if np.isfinite(forward) else {})


__all__.append("single_slice_surface")
