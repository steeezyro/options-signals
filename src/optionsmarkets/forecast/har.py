"""HAR-family realised-volatility forecasting.

Corsi's heterogeneous autoregressive model is a constrained AR(22) that mimics
long memory with three parameters -- daily, weekly and monthly components:

    ln RV_{t+h} = c + b_d ln RV^(d)_t + b_w ln RV^(w)_t + b_m ln RV^(m)_t + e

We fit the **log** form by default.  Its residuals are the closest to Gaussian
and homoskedastic of any variant, it needs no positivity constraint, and it is
the form whose forecast errors behave well enough for the downstream
uncertainty machinery (conformal intervals, Kelly shrinkage) to mean anything.
Converting back to levels requires the Jensen correction
``E[RV] = exp(mu + s^2/2)``; forgetting it biases every forecast low by about
half the residual variance, which for daily equity vol is not small.

Standard errors are Newey-West: the overlapping weekly/monthly aggregates
induce strong serial correlation in the residuals, and OLS standard errors are
roughly 2-3x too tight.  Since those standard errors feed the Kelly shrinkage
factor, understating them means overbetting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["HARModel", "har_features", "HARFit"]


def har_features(rv: pd.Series, include_today_in_week: bool = True) -> pd.DataFrame:
    """Build the (daily, weekly, monthly) design matrix.

    ``include_today_in_week`` picks between the two conventions in circulation:
    the published JFE version includes RV_t in the weekly average, the JAE
    working paper starts at t-1.  Both are used; the choice must be explicit
    and stable, because switching it mid-backtest silently changes every
    coefficient.
    """
    rv = pd.Series(rv).astype(float)
    off = 0 if include_today_in_week else 1
    d = rv.shift(off)
    w = rv.shift(off).rolling(5).mean()
    m = rv.shift(off).rolling(22).mean()
    return pd.DataFrame({"rv_d": d, "rv_w": w, "rv_m": m})


def _newey_west(X: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xl, Xr = (X * resid[:, None])[L:], (X * resid[:, None])[:-L]
        G = Xl.T @ Xr
        S += w * (G + G.T)
    return XtX_inv @ S @ XtX_inv * n / max(n - k, 1)


@dataclass
class HARFit:
    beta: np.ndarray
    names: list[str]
    se: np.ndarray
    sigma2: float
    n: int
    r2: float
    log_space: bool
    use_q: bool

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "coef": self.beta, "se": self.se,
            "t": self.beta / np.where(self.se > 0, self.se, np.nan),
        }, index=self.names)


class HARModel:
    """HAR / HARQ realised-volatility forecaster.

    Parameters
    ----------
    horizon    forecast horizon h in days; the LHS is the average RV over the
               next h days (the direct-forecast approach).  Match this to the
               option's tenor -- forecasting 1-day RV to trade a 30-day option
               is a horizon mismatch that no amount of model quality fixes.
    log_space  fit ln RV (default, recommended)
    use_q      HARQ: interact the daily term with sqrt(RQ), demeaned.  The
               coefficient comes out negative -- on days when yesterday's RV
               was measured badly the model automatically shifts weight to the
               weekly and monthly components.
    """

    def __init__(self, horizon: int = 21, log_space: bool = True, use_q: bool = False,
                 include_today_in_week: bool = True):
        self.horizon = int(horizon)
        self.log_space = bool(log_space)
        self.use_q = bool(use_q)
        self.include_today_in_week = bool(include_today_in_week)
        self.fit_: HARFit | None = None
        self._sqrt_rq_mean: float = 0.0

    def _design(self, rv: pd.Series, rq: pd.Series | None):
        X = har_features(rv, self.include_today_in_week)
        if self.log_space:
            X = np.log(X.clip(lower=1e-12))
        X.insert(0, "const", 1.0)
        if self.use_q:
            if rq is None:
                raise ValueError("use_q=True requires realized quarticity")
            sq = np.sqrt(pd.Series(rq).astype(float).clip(lower=0)).shift(
                0 if self.include_today_in_week else 1)
            X["rv_d_x_sqrtRQ"] = (sq - self._sqrt_rq_mean) * X["rv_d"]
        return X

    def fit(self, rv: pd.Series, rq: pd.Series | None = None) -> HARFit:
        rv = pd.Series(rv).astype(float)
        if self.use_q and rq is not None:
            self._sqrt_rq_mean = float(np.sqrt(pd.Series(rq).clip(lower=0)).mean())
        X = self._design(rv, rq)
        y = rv.rolling(self.horizon).mean().shift(-self.horizon)
        y = np.log(y.clip(lower=1e-12)) if self.log_space else y

        d = pd.concat([X, y.rename("y")], axis=1).dropna()
        Xm, ym = d[X.columns].to_numpy(float), d["y"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        resid = ym - Xm @ beta
        n = len(ym)
        s2 = float(resid @ resid / max(n - Xm.shape[1], 1))
        lags = max(int(1.5 * self.horizon), 5)
        V = _newey_west(Xm, resid, lags)
        r2 = 1.0 - float(resid @ resid) / float(((ym - ym.mean()) ** 2).sum())
        self.fit_ = HARFit(beta, list(X.columns), np.sqrt(np.maximum(np.diag(V), 0.0)),
                           s2, n, r2, self.log_space, self.use_q)
        return self.fit_

    def predict(self, rv: pd.Series, rq: pd.Series | None = None) -> pd.Series:
        """Forecast in LEVEL units (variance or vol, whatever rv was), Jensen-corrected."""
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        X = self._design(pd.Series(rv).astype(float), rq)
        yhat = pd.Series(X.to_numpy(float) @ self.fit_.beta, index=X.index)
        if self.log_space:
            yhat = np.exp(yhat + 0.5 * self.fit_.sigma2)     # Jensen correction
        return yhat

    def predict_interval(self, rv, rq=None, z: float = 1.645):
        """Parametric band.  Superseded at runtime by the conformal layer
        (:mod:`optionsmarkets.learning.conformal`), which keeps its coverage
        under distribution shift; this is the cold-start fallback only."""
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        X = self._design(pd.Series(rv).astype(float), rq)
        lin = pd.Series(X.to_numpy(float) @ self.fit_.beta, index=X.index)
        s = np.sqrt(self.fit_.sigma2)
        if self.log_space:
            return np.exp(lin - z * s), np.exp(lin + 0.5 * self.fit_.sigma2), np.exp(lin + z * s)
        return lin - z * s, lin, lin + z * s


# ----------------------------------------------------------------------------
# SHAR: signed semivariances
# ----------------------------------------------------------------------------

def shar_features(rv_neg: pd.Series, rv_pos: pd.Series,
                  include_today_in_week: bool = True) -> pd.DataFrame:
    """SHAR design matrix: signed daily semivariances + pooled weekly/monthly.

    Patton-Sheppard: the NEGATIVE semivariance carries most of the predictive
    content for future volatility, and pooling them throws that away.  The
    mechanism is the leverage effect -- the same asymmetry GJR-GARCH captures
    with its indicator term -- so :class:`SHARModel` and
    :class:`~optionsmarkets.forecast.garch.GJRGarch` are two views of one
    phenomenon and should broadly agree.  Where they do not, one is misspecified
    for the current regime, and that disagreement carries more information than
    either model's own standard error, which only knows its own specification.

    Only the DAILY term is split.  By weekly and monthly aggregation the
    asymmetry has largely averaged out, and splitting those too spends degrees of
    freedom on coefficients that come out indistinguishable -- which then makes
    the daily asymmetry look weaker than it is.

    Both inputs must be STRICTLY POSITIVE for the log form to work; build them
    with :func:`optionsmarkets.forecast.realized.semivariance_proxy`, which
    guarantees that and whose components sum to the same variance estimator the
    pooled HAR uses (so SHAR properly nests HAR).
    """
    off = 0 if include_today_in_week else 1
    neg = pd.Series(rv_neg).astype(float)
    pos = pd.Series(rv_pos).astype(float)
    total = neg + pos
    return pd.DataFrame({
        "rv_d_neg": neg.shift(off),
        "rv_d_pos": pos.shift(off),
        "rv_w": total.shift(off).rolling(5).mean(),
        "rv_m": total.shift(off).rolling(22).mean(),
    })


class SHARModel(HARModel):
    """HAR with the daily term split into signed semivariances.

    Subclasses :class:`HARModel` so the log form, the Jensen correction, the
    Newey-West standard errors and the horizon convention are shared rather than
    reimplemented -- those are exactly the details that go quietly wrong in a
    parallel implementation.

    Usage differs from :class:`HARModel` in one way: ``fit`` and ``predict`` take
    the two semivariance series instead of a single pooled one.
    """

    def __init__(self, horizon: int = 21, log_space: bool = True,
                 include_today_in_week: bool = True):
        super().__init__(horizon=horizon, log_space=log_space, use_q=False,
                         include_today_in_week=include_today_in_week)
        self._neg: pd.Series | None = None
        self._pos: pd.Series | None = None

    def _design(self, rv: pd.Series, rq: pd.Series | None = None):
        if self._neg is None or self._pos is None:
            raise ValueError("SHARModel needs both semivariance series; call "
                             "fit(rv_neg, rv_pos) / predict(rv_neg, rv_pos)")
        X = shar_features(self._neg, self._pos, self.include_today_in_week)
        if self.log_space:
            X = np.log(X.clip(lower=1e-12))
        X.insert(0, "const", 1.0)
        return X

    def fit(self, rv_neg: pd.Series, rv_pos: pd.Series) -> HARFit:
        self._neg = pd.Series(rv_neg).astype(float)
        self._pos = pd.Series(rv_pos).astype(float)
        return super().fit(self._neg + self._pos, None)

    def predict(self, rv_neg: pd.Series, rv_pos: pd.Series) -> pd.Series:
        self._neg = pd.Series(rv_neg).astype(float)
        self._pos = pd.Series(rv_pos).astype(float)
        return super().predict(self._neg + self._pos, None)

    def asymmetry(self) -> dict:
        """Is the negative-semivariance coefficient larger, and significantly so?"""
        if self.fit_ is None:
            raise RuntimeError("call fit() first")
        idx = {n: i for i, n in enumerate(self.fit_.names)}
        bn, bp = self.fit_.beta[idx["rv_d_neg"]], self.fit_.beta[idx["rv_d_pos"]]
        sn, sp = self.fit_.se[idx["rv_d_neg"]], self.fit_.se[idx["rv_d_pos"]]
        se_diff = float(np.sqrt(sn**2 + sp**2))       # ignores their covariance
        return {
            "beta_negative": float(bn), "beta_positive": float(bp),
            "difference": float(bn - bp),
            "t_difference": float((bn - bp) / se_diff) if se_diff > 0 else np.nan,
            "negative_dominates": bool(bn > bp),
            "note": ("The t-statistic uses independent standard errors and so ignores "
                     "the covariance between the two coefficients; treat it as "
                     "indicative. The OHLC semivariance split is a proxy for the "
                     "intraday quantity and attenuates the difference, so a "
                     "significant result here is real while an insignificant one is "
                     "not evidence of symmetry."),
        }


__all__ += ["SHARModel", "shar_features"]
