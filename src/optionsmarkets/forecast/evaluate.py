"""Walk-forward forecast evaluation.  BLUEPRINT.md section 13.3.

Compare HAR-log / HARQ / SHAR / GJR-GARCH / naive-RV / implied-as-forecast on:
QLIKE and MSE against realised, a Mincer-Zarnowitz regression (slope should be
1), Diebold-Mariano against the naive benchmark, and -- most importantly -- the
empirical coverage of the stated intervals.

**QLIKE is the primary loss and MSE is the secondary one.** This ordering is not
a preference.  MSE on variance is dominated by the few highest-variance days, so
it ranks models almost entirely on how they behave during vol spikes, and it is
not invariant to whether you forecast variance or volatility -- the same two
models can swap places depending on which you happen to report.  QLIKE,

    QLIKE = RV/F - ln(RV/F) - 1

is a proper loss for variance forecasting, is robust to noise in the *proxy* used
for the true variance (Patton), and is minimised at F = RV with a penalty that is
asymmetric in the right direction: under-forecasting variance is punished harder
than over-forecasting it, which is exactly the asymmetry a short-premium book
faces.

**Walk-forward only.** Every function here takes forecasts that were produced
out of sample.  Nothing in this module can detect look-ahead; it is the caller's
job not to commit it, and the recorded journal is what makes that checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

__all__ = ["qlike", "ForecastScore", "score_forecast", "diebold_mariano",
           "mincer_zarnowitz", "compare_forecasts", "interval_coverage"]


def qlike(realised, forecast) -> float:
    """Mean QLIKE loss.  Both arguments in VARIANCE units, both positive.

    Passing volatility instead of variance does not error -- it silently
    computes a different, still-finite number -- so the units are the caller's
    responsibility and are worth asserting at the call site.
    """
    r = np.asarray(realised, float).ravel()
    f = np.asarray(forecast, float).ravel()
    n = min(r.size, f.size)
    r, f = r[:n], f[:n]
    m = np.isfinite(r) & np.isfinite(f) & (r > 0) & (f > 0)
    if not m.any():
        return np.nan
    ratio = r[m] / f[m]
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mincer_zarnowitz(realised, forecast) -> dict:
    """Regress realised on forecast: ``RV = a + b F + e``.

    An unbiased forecast has a = 0 and b = 1.  The informative failure is
    **b < 1**, which means the forecast *overreacts*: it is too high when it is
    high and too low when it is low, and shrinking it toward its mean would
    improve it.  That is a fixable defect and it is what the Kalman
    recalibration in :mod:`optionsmarkets.learning.feedback` corrects online.
    """
    r = np.asarray(realised, float).ravel()
    f = np.asarray(forecast, float).ravel()
    n = min(r.size, f.size)
    r, f = r[:n], f[:n]
    m = np.isfinite(r) & np.isfinite(f)
    r, f = r[m], f[m]
    if r.size < 5 or np.std(f) <= 0:
        return {"intercept": np.nan, "slope": np.nan, "r2": np.nan, "n": int(r.size),
                "slope_se": np.nan, "t_slope_eq_1": np.nan}
    A = np.column_stack([np.ones_like(f), f])
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    resid = r - A @ coef
    dof = max(r.size - 2, 1)
    cov = float(resid @ resid) / dof * np.linalg.pinv(A.T @ A)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    return {
        "intercept": float(coef[0]), "slope": float(coef[1]),
        "slope_se": se,
        "t_slope_eq_1": float((coef[1] - 1.0) / se) if se > 0 else np.nan,
        "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan,
        "n": int(r.size),
        "verdict": ("unbiased" if se > 0 and abs((coef[1] - 1.0) / se) < 2.0 else
                    ("overreacts (slope < 1): shrink it toward its mean"
                     if coef[1] < 1 else "underreacts (slope > 1)")),
    }


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Diebold-Mariano test that two loss series differ.  Negative stat favours A.

    The variance of the loss differential uses a Newey-West correction with
    ``h-1`` lags, which is required rather than optional: h-step-ahead forecast
    errors from overlapping windows are serially correlated by construction, and
    the uncorrected statistic is anticonservative by a factor that grows with h.
    At h=21 it will hand you significance that is not there.
    """
    a = np.asarray(loss_a, float).ravel()
    b = np.asarray(loss_b, float).ravel()
    n = min(a.size, b.size)
    d = a[:n] - b[:n]
    d = d[np.isfinite(d)]
    if d.size < 10:
        return {"stat": np.nan, "p_value": np.nan, "n": int(d.size),
                "note": "too few paired observations"}
    dbar = float(np.mean(d))
    dev = d - dbar
    gamma0 = float(np.mean(dev**2))
    var = gamma0
    for L in range(1, max(int(h) - 1, 0) + 1):
        if L >= d.size:
            break
        w = 1.0 - L / float(h)
        var += 2.0 * w * float(np.mean(dev[L:] * dev[:-L]))
    var = max(var, 1e-300)
    stat = dbar / np.sqrt(var / d.size)
    return {
        "stat": float(stat), "p_value": float(2.0 * (1.0 - norm.cdf(abs(stat)))),
        "mean_loss_diff": dbar, "n": int(d.size),
        "favours": "A" if dbar < 0 else "B",
        "note": f"Newey-West with {max(int(h) - 1, 0)} lags for the {h}-step overlap",
    }


def interval_coverage(realised, lo, hi) -> dict:
    """Empirical coverage of stated intervals.

    BLUEPRINT.md section 13.3 calls this the most important number in the
    forecast evaluation, and section 8.3 says why: a model whose stated 90%
    interval covers 60% will overbet by a factor no Kelly fraction repairs.  The
    adaptive-conformal layer exists to make this converge; this is how you check
    that it did.
    """
    r = np.asarray(realised, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    n = min(r.size, lo.size, hi.size)
    r, lo, hi = r[:n], lo[:n], hi[:n]
    m = np.isfinite(r) & np.isfinite(lo) & np.isfinite(hi)
    if not m.any():
        return {"coverage": np.nan, "n": 0}
    inside = (r[m] >= lo[m]) & (r[m] <= hi[m])
    width = hi[m] - lo[m]
    return {"coverage": float(np.mean(inside)), "n": int(m.sum()),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width))}


@dataclass
class ForecastScore:
    name: str
    n: int
    qlike: float
    mse: float
    rmse_vol: float
    bias_vol: float
    mz: dict = field(default_factory=dict)
    dm_vs_benchmark: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def score_forecast(name: str, realised_var, forecast_var, *,
                   benchmark_loss=None, horizon: int = 1,
                   lo=None, hi=None) -> ForecastScore:
    """Score one forecast series.  Inputs are VARIANCE, outputs mix both units.

    ``rmse_vol`` and ``bias_vol`` are reported in volatility points because that
    is the unit every threshold in the policy is written in -- a variance RMSE is
    not a number anyone can sanity-check against a gate that says "2 vol points".
    """
    r = np.asarray(realised_var, float).ravel()
    f = np.asarray(forecast_var, float).ravel()
    n = min(r.size, f.size)
    r, f = r[:n], f[:n]
    m = np.isfinite(r) & np.isfinite(f) & (r > 0) & (f > 0)
    r, f = r[m], f[m]
    sc = ForecastScore(
        name=name, n=int(r.size), qlike=qlike(r, f),
        mse=float(np.mean((r - f) ** 2)) if r.size else np.nan,
        rmse_vol=float(np.sqrt(np.mean((np.sqrt(r) - np.sqrt(f)) ** 2))) if r.size else np.nan,
        bias_vol=float(np.mean(np.sqrt(f) - np.sqrt(r))) if r.size else np.nan,
        mz=mincer_zarnowitz(r, f),
    )
    if benchmark_loss is not None and r.size:
        own = _qlike_series(r, f)
        sc.dm_vs_benchmark = diebold_mariano(own, np.asarray(benchmark_loss, float)[:own.size],
                                             h=horizon)
    if lo is not None and hi is not None:
        sc.coverage = interval_coverage(np.sqrt(r), lo, hi)
    return sc


def _qlike_series(r, f):
    ratio = np.asarray(r, float) / np.asarray(f, float)
    return ratio - np.log(ratio) - 1.0


def compare_forecasts(realised_var, forecasts: dict, *, horizon: int = 1,
                      benchmark: str = "naive") -> dict:
    """Score every model against the same realisations, ranked by QLIKE.

    ``forecasts`` maps name -> variance forecast array.  The benchmark's own
    QLIKE series is used for every Diebold-Mariano test, so all models are
    compared against one reference rather than against each other pairwise --
    which would multiply the testing problem by the number of pairs and is the
    same multiple-comparison trap section 16 warns about for thresholds.
    """
    r = np.asarray(realised_var, float).ravel()
    bench = forecasts.get(benchmark)
    bench_loss = None
    if bench is not None:
        b = np.asarray(bench, float).ravel()
        n = min(r.size, b.size)
        m = np.isfinite(r[:n]) & np.isfinite(b[:n]) & (r[:n] > 0) & (b[:n] > 0)
        bench_loss = _qlike_series(r[:n][m], b[:n][m])

    scores = {}
    for name, f in forecasts.items():
        scores[name] = score_forecast(
            name, r, f, horizon=horizon,
            benchmark_loss=None if name == benchmark else bench_loss)
    ranked = sorted((s for s in scores.values() if np.isfinite(s.qlike)),
                    key=lambda s: s.qlike)
    return {
        "scores": {k: v.as_dict() for k, v in scores.items()},
        "ranking_by_qlike": [s.name for s in ranked],
        "best": ranked[0].name if ranked else None,
        "note": ("Ranked by QLIKE, which is robust to noise in the realised-variance "
                 "proxy and asymmetric in the direction that matters for short "
                 "premium. MSE is reported but not ranked on: it is dominated by the "
                 "few highest-variance days and is not invariant to forecasting "
                 "variance versus volatility."),
    }


def report(comparison: dict) -> str:
    """Render :func:`compare_forecasts` as a table."""
    L = [f"{'model':<16}{'n':>6}{'QLIKE':>10}{'RMSE vol':>10}{'bias vol':>10}"
         f"{'MZ slope':>10}{'DM p':>8}  verdict"]
    L.append("-" * len(L[0]))
    for name in comparison["ranking_by_qlike"]:
        s = comparison["scores"][name]
        mz, dm = s.get("mz", {}), s.get("dm_vs_benchmark", {})
        L.append(f"{name:<16}{s['n']:>6}{s['qlike']:>10.4f}"
                 f"{s['rmse_vol'] * 100:>9.2f}p{s['bias_vol'] * 100:>9.2f}p"
                 f"{mz.get('slope', float('nan')):>10.3f}"
                 f"{dm.get('p_value', float('nan')):>8.3f}  {mz.get('verdict', '')}")
    L.append(comparison["note"])
    return "\n".join(L)


__all__.append("report")
