"""Performance statistics, and the one that stops you fooling yourself.

BLUEPRINT.md section 13.4 asks for: annualised growth of log wealth (the actual
objective), max drawdown and time to recover, Sharpe *and* Sortino, hit rate vs
average win/loss, the distribution of the gate that blocked, and a deflated
Sharpe / reality check for the multiple testing you did while tuning.

**Growth of log wealth is reported first and deliberately.** It is what Kelly
maximises and what the sizing chain was derived from.  Total return and Sharpe
are diagnostics; the objective is the growth rate, and a strategy can improve
its Sharpe while lowering its growth rate by underbetting.

**The deflated Sharpe is not optional.** Section 16 counts fifteen-plus free
thresholds in this system.  Every one is a knob that was, implicitly or
explicitly, selected against a record -- and the maximum of N sample Sharpes
under a *zero-edge* null is not zero, it grows like sqrt(2 ln N).  A nominal
Sharpe of 1.0 chosen from twenty configurations is roughly what a coin-flipping
strategy produces.  ``deflated_sharpe`` reports the probability that the
observed Sharpe survives that correction, along with the skew and kurtosis
adjustments, because option-selling returns are exactly the negatively-skewed,
fat-tailed shape that inflates a naive Sharpe most.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["performance", "drawdown_profile", "deflated_sharpe", "probabilistic_sharpe"]

_EULER = 0.5772156649015329


def drawdown_profile(equity) -> dict:
    """Max drawdown, when it happened, and how long recovery took.

    Time-to-recover is reported because the depth alone understates the damage:
    a 20% drawdown that recovers in a month and a 20% drawdown that takes three
    years are not the same experience, and only one of them lets you keep
    trading the strategy.
    """
    eq = np.asarray(equity, float)
    eq = eq[np.isfinite(eq)]
    if eq.size < 2:
        return {"max_drawdown": np.nan, "peak_index": -1, "trough_index": -1,
                "recovery_index": -1, "periods_to_recover": np.nan,
                "periods_underwater": np.nan}
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    trough = int(np.argmin(dd))
    max_dd = float(dd[trough])
    pk = int(np.argmax(eq[: trough + 1])) if trough > 0 else 0
    rec = -1
    after = np.flatnonzero(eq[trough:] >= eq[pk])
    if after.size:
        rec = int(trough + after[0])
    return {
        "max_drawdown": max_dd,
        "peak_index": pk, "trough_index": trough, "recovery_index": rec,
        "periods_to_recover": float(rec - trough) if rec >= 0 else np.inf,
        "periods_underwater": float((rec if rec >= 0 else eq.size - 1) - pk),
        "final_drawdown": float(dd[-1]),
    }


def probabilistic_sharpe(sharpe: float, n: int, skew: float, kurt: float,
                         benchmark: float = 0.0) -> float:
    """Bailey-Lopez de Prado PSR: P(true Sharpe > benchmark).

    The denominator is the standard error of the Sharpe estimator under
    non-normal returns.  Negative skew and excess kurtosis -- the signature of
    short-premium option returns -- both INFLATE that standard error, so the
    same nominal Sharpe is weaker evidence for this strategy family than for a
    Gaussian one.  Using the Gaussian standard error here would flatter exactly
    the strategy this repo implements.
    """
    if n < 3 or not np.isfinite(sharpe):
        return np.nan
    denom = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe**2
    if denom <= 0:
        return np.nan
    z = (sharpe - benchmark) * np.sqrt(n - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def deflated_sharpe(returns, n_trials: int, sharpe_variance: float | None = None) -> dict:
    """Deflated Sharpe ratio: PSR against the Sharpe a null strategy would win.

    The benchmark is the EXPECTED MAXIMUM Sharpe across ``n_trials``
    zero-edge trials::

        SR0 = sqrt(V) * [ (1-g) z(1 - 1/N) + g z(1 - 1/(N e)) ],   g = Euler-Mascheroni

    where ``V`` is the variance of the Sharpe estimates across trials.  When it
    is not supplied it is approximated by the estimator variance of the single
    observed track record, which is the conservative choice available with one
    series in hand -- and the approximation is flagged in the output rather than
    hidden, because it is the weakest link in the calculation.

    ``n_trials`` must be the honest count of configurations you *considered*,
    not the number you reported.  Every threshold tweak is a trial.
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {"n": int(n), "n_trials": int(n_trials)}
    if n < 3:
        return out | {"sharpe": np.nan, "sr0": np.nan, "dsr": np.nan, "psr": np.nan}

    sd = float(np.std(r, ddof=1))
    sr = float(np.mean(r) / sd) if sd > 0 else 0.0
    m = r - r.mean()
    skew = float(np.mean(m**3) / sd**3) if sd > 0 else 0.0
    kurt = float(np.mean(m**4) / sd**4) if sd > 0 else 3.0

    N = max(int(n_trials), 1)
    if sharpe_variance is None:
        var_sr = max((1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / max(n - 1, 1), 1e-12)
        approx = True
    else:
        var_sr = max(float(sharpe_variance), 1e-12)
        approx = False
    if N <= 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(var_sr) * ((1.0 - _EULER) * norm.ppf(1.0 - 1.0 / N)
                                 + _EULER * norm.ppf(1.0 - 1.0 / (N * np.e)))

    return out | {
        "sharpe": sr, "skew": skew, "kurtosis": kurt,
        "sr0": float(sr0),
        "psr": probabilistic_sharpe(sr, n, skew, kurt, 0.0),
        "dsr": probabilistic_sharpe(sr, n, skew, kurt, float(sr0)),
        "sharpe_variance_approximated": approx,
        "note": ("DSR is P(true Sharpe > the best a zero-edge strategy would show "
                 "across n_trials). Below ~0.95 the record does not survive the "
                 "multiple testing that produced it."
                 + ("  Sharpe variance was approximated from this single track "
                    "record; supply the cross-trial variance for a tighter number."
                    if approx else "")),
    }


def performance(trade_returns, equity, *, periods_per_year: float = 252.0,
                n_trials: int = 1, rf: float = 0.0) -> dict:
    """The full section-13.4 statistics block.

    ``trade_returns`` are per-trade returns on capital at risk; ``equity`` is
    the account equity curve sampled once per decision step.  They are kept
    separate on purpose: the Sharpe of a per-trade series and the Sharpe of a
    per-day equity series are different numbers, and reporting one while calling
    it the other is a common way to inflate a result by the square root of the
    trade frequency.
    """
    r = np.asarray(trade_returns, float)
    r = r[np.isfinite(r)]
    eq = np.asarray(equity, float)
    eq = eq[np.isfinite(eq) & (eq > 0)]

    out: dict = {"n_trades": int(r.size), "n_periods": int(eq.size)}

    if eq.size >= 2:
        log_r = np.diff(np.log(eq))
        # n equity points span n-1 periods, not n. The off-by-one is small on a
        # long record and embarrassing on a short one, and it biases the reported
        # growth rate DOWNWARD -- which is the direction that hides a problem.
        span_years = log_r.size / max(periods_per_year, 1e-9)
        out["total_return"] = float(eq[-1] / eq[0] - 1.0)
        # THE objective: annualised growth rate of log wealth.
        out["log_growth_annualised"] = float(np.sum(log_r) / max(span_years, 1e-9))
        out["vol_of_log_wealth_annualised"] = float(
            np.std(log_r, ddof=1) * np.sqrt(periods_per_year)) if log_r.size > 1 else np.nan
        out.update({f"equity_{k}": v for k, v in drawdown_profile(eq).items()})

    if r.size >= 2:
        sd = float(np.std(r, ddof=1))
        down = r[r < rf]
        dsd = float(np.sqrt(np.mean((down - rf) ** 2))) if down.size else 0.0
        out["mean_trade_return"] = float(np.mean(r))
        out["sharpe_per_trade"] = float((np.mean(r) - rf) / sd) if sd > 0 else np.nan
        # Sortino uses the downside SEMI-deviation. For short premium the two
        # differ a lot -- the distribution is a wall of small wins and a thin
        # tail of large losses -- and only the Sortino sees the shape that
        # actually ends the strategy.
        out["sortino_per_trade"] = float((np.mean(r) - rf) / dsd) if dsd > 0 else np.inf
        wins, losses = r[r > 0], r[r <= 0]
        out["hit_rate"] = float(wins.size / r.size)
        out["avg_win"] = float(np.mean(wins)) if wins.size else 0.0
        out["avg_loss"] = float(np.mean(losses)) if losses.size else 0.0
        out["win_loss_ratio"] = float(abs(out["avg_win"] / out["avg_loss"])) \
            if out["avg_loss"] != 0 else np.inf
        out["worst_trade"] = float(np.min(r))
        out["best_trade"] = float(np.max(r))
        out["deflated_sharpe"] = deflated_sharpe(r, n_trials)
    return out
