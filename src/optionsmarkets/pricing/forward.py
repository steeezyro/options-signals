"""Forward and discount factor implied by the option market itself.

Never trust an externally supplied rate/dividend pair to define the forward.
If F is wrong the entire slice tilts, and the tilt is indistinguishable from
skew -- you will "discover" a risk reversal that is really a bad borrow rate.
Solve F and DF from put-call parity across the strikes you actually have.

Put-call parity for European options:      C - P = DF * (F - K)
so regressing (C - P) on K gives slope = -DF and intercept = DF * F.

For American quotes, run :mod:`.american.de_americanise` first: the parity
relation does not hold for American options, and the regression will absorb
the early-exercise premium into the forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ForwardFit", "implied_forward"]


@dataclass(frozen=True)
class ForwardFit:
    F: float
    DF: float
    r_implied: float
    n_pairs: int
    r2: float
    residual_bp: float

    @property
    def ok(self) -> bool:
        return self.n_pairs >= 3 and np.isfinite(self.F) and 0.5 < self.DF <= 1.0001


def implied_forward(
    strikes, call_mid, put_mid, T: float,
    weights=None,
    atm_window: float = 0.10,
    S_ref: float | None = None,
) -> ForwardFit:
    """Weighted least-squares fit of C - P = DF*F - DF*K.

    Parameters
    ----------
    atm_window
        Only strikes within this fractional distance of ``S_ref`` are used.
        Parity is exact at every strike in theory; in practice the wings have
        wide spreads and stale quotes, and including them lets a single bad
        far-OTM pair rotate the whole regression.  Restricting to the liquid
        core is what makes this robust.
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

    A = np.column_stack([np.ones_like(K), -K])          # [DF*F, DF]
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    dfF, DF = float(coef[0]), float(coef[1])
    F = dfF / DF if DF != 0 else np.nan

    pred = A @ coef
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resid_bp = float(np.sqrt(ss_res / w.sum()) / max(F, 1e-9) * 1e4) if np.isfinite(F) else np.nan
    r_imp = float(-np.log(max(DF, 1e-9)) / T) if T > 0 else np.nan
    return ForwardFit(float(F), DF, r_imp, int(m.sum()), r2, resid_bp)
