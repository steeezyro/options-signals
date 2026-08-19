"""Forward and discount factor implied by the option market itself.

Never trust an externally supplied rate/dividend pair to define the forward.
If F is wrong the entire slice tilts, and the tilt is indistinguishable from
skew -- you will "discover" a risk reversal that is really a bad borrow rate.
Solve F from put-call parity across the strikes you actually have.

Put-call parity for European options:      C - P = DF * (F - K)

There are two ways to run that, and which one you pick decides whether the
live path works at all.

**Free fit** (``df_known=None``) regresses (C - P) on K, so slope = -DF and
intercept = DF * F.  It is the honest thing to do when nothing is known about
the discount curve -- and it does not survive contact with American equity
options.  Measured on live chains 2026-08-18, every underlying returned DF > 1
(SPY 1.003-1.014, MA 1.002-1.257, INSM 1.008), i.e. a negative implied rate,
so ``ok`` was False and no slice was ever published.  Two effects combine:

  * the early-exercise premium sits in whichever leg is in-the-money and grows
    with |K - F|, so it enters the regression as SLOPE, not as noise;
  * DF is estimated off a strike lever arm of +/-10%, which multiplies any
    per-strike error by ~1/0.10 before it lands in DF.

The second is the larger one.  Per-strike parity error on live quotes measures
~10 bp; the free fit turned that into 100-2500 bp of DF error.

**Constrained fit** (``df_known=DF``) is what the pipeline uses.  DF over a
10-100 day tenor is known from the Treasury curve to a few bp, so spending a
regression degree of freedom to re-estimate it badly is a bad trade.  Fix DF,
rearrange parity to give a per-strike forward estimate

    f_k = K + (C - P) / DF

and take a robust weighted average.  Note f_k -> K as K -> F: at the money the
estimate is almost independent of DF, which is exactly the right division of
labour.  The market prices F precisely and DF terribly; the curve does the
reverse.

Validated against known truth on live quotes the same day.  MU, NVDA and TSLA
pay no meaningful dividend, so F_true = S*exp(rT) on an entirely independent
path; the constrained fit landed within 3-17 bp on all 24 expiries, and moving
``atm_window`` from 0.02 to 0.15 shifted it by under 20 bp.  The free fit was
unusable on the same data.

For a slice that must be exact rather than merely unbiased, run
:mod:`.american.de_americanise` on the quotes first -- parity does not hold for
American options and no amount of fitting repairs that.  The residual bias the
constrained fit leaves is second-order (it is why the ATM core is preferred),
and it is reported in ``residual_bp`` rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ForwardFit", "implied_forward"]

# Per-strike parity estimates disagree for two reasons with very different
# magnitudes, and the screen exists to separate them rather than to tidy up.
#
# Legitimate disagreement -- early exercise plus quote noise -- measured 4-243
# bp of full spread across 56 live expiries on 8 underlyings.  Corrupt vendor
# rows measured 989-1522 bp: on MA 2026-09-18 the K=610 put printed 110.55 x
# 117.00 while the K=605 put printed 31.00 x 35.70 and the K=615 put 39.40 x
# 44.40, which is not a price, and one such row rotated the whole fit.
#
# The floor sits in the ~4x gap between those populations. It is a floor and
# not a fixed tolerance because a very tight chain gives a tiny MAD, and 6 MADs
# of a tiny MAD would start trimming good strikes.
OUTLIER_MAD_K = 6.0
OUTLIER_REL_FLOOR = 0.03


@dataclass(frozen=True)
class ForwardFit:
    F: float
    DF: float
    r_implied: float
    n_pairs: int
    r2: float
    residual_bp: float
    n_dropped: int = 0
    df_source: str = "fitted"

    @property
    def ok(self) -> bool:
        # The DF band is retained for BOTH paths on purpose. On the free fit it
        # is what catches the American bias described in the module docstring.
        # On the constrained fit DF is an input, so the band is checking the
        # CURVE -- a rate that implies DF > 1 or DF < 0.5 at these tenors means
        # the curve fetch returned garbage, and inheriting it silently would put
        # the error straight into the surface.
        #
        # Do not widen it to make live runs pass. A tilted forward looks exactly
        # like skew, which is the failure this module exists to prevent.
        return (
            self.n_pairs >= 3
            and np.isfinite(self.F)
            and self.F > 0.0
            and 0.5 < self.DF <= 1.0001
            and np.isfinite(self.residual_bp)
            # 250 bp is deliberately far above anything observed post-screen on
            # live data (0.8-19 bp on liquid names, ~100 bp on the thinnest
            # KTOS expiry). It fires only when the strikes genuinely disagree
            # about where the forward is, which is a slice with no information
            # in it rather than a slice with a wide error bar.
            and self.residual_bp <= 250.0
        )


def implied_forward(
    strikes, call_mid, put_mid, T: float,
    weights=None,
    atm_window: float = 0.10,
    S_ref: float | None = None,
    df_known: float | None = None,
) -> ForwardFit:
    """Forward from put-call parity, with DF either fitted or supplied.

    Parameters
    ----------
    df_known
        Discount factor for this tenor, from the risk-free curve.  When given,
        DF is held fixed and F is the only free parameter -- see the module
        docstring for why that is not a shortcut but the more accurate
        estimator.  When None, DF and F are both regressed out, which is only
        appropriate for European quotes.
    atm_window
        Only strikes within this fractional distance of ``S_ref`` are used.
        Parity is exact at every strike in theory; in practice the wings have
        wide spreads and stale quotes, and including them lets a single bad
        far-OTM pair rotate the whole regression.  Restricting to the liquid
        core is what makes this robust.  It also bounds the early-exercise
        premium, which is smallest at the money and grows into the wings.
    weights
        Optional per-strike weights, typically ``1/(spread_c + spread_p)``.
    """
    K = np.asarray(strikes, float)
    C = np.asarray(call_mid, float)
    P = np.asarray(put_mid, float)
    m = np.isfinite(K) & np.isfinite(C) & np.isfinite(P) & (K > 0)
    if S_ref is not None and atm_window is not None:
        m &= np.abs(K / S_ref - 1.0) <= atm_window
    if m.sum() < 3:
        return ForwardFit(np.nan, np.nan, np.nan, int(m.sum()), np.nan, np.nan)

    K, y = K[m], (C - P)[m]
    w = np.ones_like(K) if weights is None else np.asarray(weights, float)[m]
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(K)

    # Screen corrupt rows BEFORE either fit. f_k is the per-strike forward; a
    # DF of 1 is close enough to locate outliers three orders of magnitude out,
    # and using it here keeps the screen identical on both paths.
    df_screen = float(df_known) if (df_known and np.isfinite(df_known) and df_known > 0) else 1.0
    keep = _inlier_mask(K + y / df_screen)
    n_dropped = int((~keep).sum())
    if keep.sum() < 3:
        return ForwardFit(np.nan, np.nan, np.nan, int(keep.sum()), np.nan, np.nan,
                          n_dropped, "known" if df_known else "fitted")
    K, y, w = K[keep], y[keep], w[keep]

    if df_known is None:
        F, DF = _free_fit(K, y, w)
        source = "fitted"
    else:
        F, DF = _constrained_fit(K, y, w, float(df_known))
        source = "known"

    # Both paths are scored the same way, against the SAME model C - P =
    # DF*(F - K), so the two fits are directly comparable on a real chain.
    pred = DF * (F - K) if np.isfinite(F) else np.full_like(K, np.nan)
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resid_bp = (float(np.sqrt(ss_res / w.sum()) / max(F, 1e-9) * 1e4)
                if np.isfinite(F) else np.nan)
    r_imp = float(-np.log(max(DF, 1e-9)) / T) if T > 0 else np.nan
    return ForwardFit(float(F), float(DF), r_imp, int(K.size), r2, resid_bp,
                      n_dropped, source)


def _inlier_mask(f_k: np.ndarray) -> np.ndarray:
    """Median-absolute-deviation screen on the per-strike forward estimates."""
    med = float(np.median(f_k))
    mad = float(np.median(np.abs(f_k - med))) * 1.4826
    tol = max(OUTLIER_MAD_K * mad, OUTLIER_REL_FLOOR * abs(med))
    return np.abs(f_k - med) <= tol


def _free_fit(K, y, w):
    """Regress C - P on K: slope = -DF, intercept = DF*F.  European quotes only."""
    A = np.column_stack([np.ones_like(K), -K])          # [DF*F, DF]
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    dfF, DF = float(coef[0]), float(coef[1])
    return (dfF / DF if DF != 0 else np.nan), DF


def _constrained_fit(K, y, w, DF: float):
    """DF fixed: every strike votes on F directly, and the votes are averaged.

    This is a weighted mean rather than a regression because with DF known the
    model has one parameter and each strike is an independent estimate of it.
    The spread of those estimates is what ``residual_bp`` reports.
    """
    if not (np.isfinite(DF) and DF > 0):
        return np.nan, DF
    return float(np.average(K + y / DF, weights=w)), DF
