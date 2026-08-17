"""Growth-optimal position sizing for option structures.

Options payoffs are violently non-normal, so the Gaussian Kelly formula
``f* = (mu - r)/sigma^2`` is not usable.  We solve the *discrete multi-outcome*
problem directly on a scenario grid::

    g(f) = sum_s  p_s * ln( 1 + f * x_s )
    FOC:  sum_s  p_s * x_s / ( 1 + f * x_s ) = 0

``g`` is strictly concave (g'' = -E[x^2/(1+fx)^2] < 0), so the root is unique
and Newton converges from any feasible start.  The binding constraint for a
short-premium structure is not the FOC but ``1 + f*x > 0`` almost surely --
i.e. ``f < 1/|min x|``.  Violating it makes g = -inf: ruin.

The raw Kelly fraction is then knocked down through four independent layers:

  1. **uncertainty shrinkage**  c_unc = 1/(1 + 1/(S^2 T))
     This is not a folk haircut.  Taking expectations of g over the estimation
     error in mu gives exactly c* = mu^2/(mu^2 + s_mu^2), and with T years of
     data and Sharpe S that is 1/(1 + 1/(S^2 T)).  A Sharpe-0.5 strategy with
     4 years of evidence *deserves* half Kelly; "half Kelly" is a special case
     of the shrinkage formula, not a rule of thumb.

  2. **drawdown cap**  c_dd = 2 / (1 + ln p / ln x)
     For fractional Kelly, P(wealth ever falls to fraction x) = x^(2/c - 1).
     Full Kelly therefore implies a 50% chance of at some point halving the
     account.  Inverting that first-passage result turns a drawdown *mandate*
     into a sizing constraint.

  3. **hard cap at half Kelly**, always, because the model is misspecified in
     ways the math cannot see.

  4. **absolute limits** -- max risk per trade, per underlying, per day; % of
     open interest; % of ADV.  Kelly assumes continuous costless rebalancing
     with unlimited capacity.  An options book satisfies none of those.

The asymmetry is what justifies all of it.  Excess growth scales as
``2c - c^2``: half Kelly keeps 75% of the growth at half the volatility, while
1.5x Kelly gives up the *same* 25% of growth for 50% more volatility -- strictly
dominated.  At 2x Kelly growth is zero; beyond it, negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["kelly_fraction", "KellySizing", "size_position", "drawdown_cap",
           "uncertainty_shrinkage", "risk_of_ruin"]


def kelly_fraction(x, p=None, f_max: float = 10.0, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Solve  max_f  sum_s p_s ln(1 + f x_s)  for the growth-optimal f.

    ``x`` is the per-unit-stake return of each scenario (so -1.0 is a total
    loss of the amount staked).  Returns 0.0 when the edge is non-positive.
    """
    x = np.asarray(x, float).ravel()
    p = np.ones_like(x) / x.size if p is None else np.asarray(p, float).ravel()
    m = np.isfinite(x) & np.isfinite(p) & (p > 0)
    x, p = x[m], p[m]
    if x.size == 0:
        return 0.0
    p = p / p.sum()

    if float(p @ x) <= 0:                       # no edge -> do not bet
        return 0.0
    worst = float(np.min(x))
    hi = f_max if worst >= 0 else min(f_max, 0.999999 / abs(worst))
    if hi <= 0:
        return 0.0

    def gp(f):                                   # g'(f)
        return float(np.sum(p * x / (1.0 + f * x)))

    if gp(hi) > 0:                               # optimum is past the ruin bound
        return float(hi)

    lo, f = 0.0, min(0.5 * hi, 0.05)
    for _ in range(max_iter):
        d1 = gp(f)
        if d1 > 0:
            lo = f
        else:
            hi = f
        d2 = -float(np.sum(p * x**2 / (1.0 + f * x) ** 2))
        step = -d1 / d2 if abs(d2) > 1e-300 else 0.0
        f_new = f + step
        if not np.isfinite(f_new) or f_new <= lo or f_new >= hi:
            f_new = 0.5 * (lo + hi)
        if abs(f_new - f) < tol:
            f = f_new
            break
        f = f_new
    return float(max(f, 0.0))


def uncertainty_shrinkage(sharpe: float, years: float) -> float:
    """c* = 1/(1 + 1/(S^2 T)).  Derived, not assumed -- see module docstring."""
    if not np.isfinite(sharpe) or not np.isfinite(years) or sharpe <= 0 or years <= 0:
        return 0.0
    return float(1.0 / (1.0 + 1.0 / (sharpe**2 * years)))


def drawdown_cap(max_drawdown: float = 0.30, prob: float = 0.10) -> float:
    """Largest Kelly fraction c with P(ever drawing down to (1-max_drawdown)) <= prob."""
    x = 1.0 - float(max_drawdown)
    if not (0 < x < 1) or not (0 < prob < 1):
        return 0.5
    return float(np.clip(2.0 / (1.0 + np.log(prob) / np.log(x)), 0.0, 1.0))


def risk_of_ruin(c: float, level: float) -> float:
    """P(wealth ever falls to `level` x starting) = level^(2/c - 1)."""
    if c <= 0:
        return 0.0
    return float(np.clip(level, 1e-12, 1.0) ** (2.0 / c - 1.0))


@dataclass
class KellySizing:
    contracts: int
    f_kelly_raw: float
    f_applied: float
    c_total: float
    c_uncertainty: float
    c_drawdown: float
    capital_at_risk: float
    capital_fraction: float
    expected_log_growth: float
    binding_constraint: str
    scenarios: dict = field(default_factory=dict)

    def explain(self) -> str:
        return (
            f"Kelly f*={self.f_kelly_raw:.4f} -> applied {self.f_applied:.4f} "
            f"(c={self.c_total:.3f} = min(unc {self.c_uncertainty:.3f}, "
            f"dd {self.c_drawdown:.3f}, 0.50)); "
            f"{self.contracts} contract(s), ${self.capital_at_risk:,.0f} at risk "
            f"({self.capital_fraction:.2%} of bankroll); binding: {self.binding_constraint}"
        )


def size_position(
    pnl_per_contract,
    probabilities,
    max_loss_per_contract: float,
    bankroll: float,
    *,
    sharpe: float = 0.5,
    years_of_evidence: float = 2.0,
    max_drawdown: float = 0.30,
    drawdown_prob: float = 0.10,
    hard_cap_fraction: float = 0.50,
    max_risk_fraction_per_trade: float = 0.02,
    max_contracts: int | None = None,
) -> KellySizing:
    """Turn a scenario P&L distribution into an integer contract count.

    Parameters
    ----------
    pnl_per_contract     dollar P&L of ONE contract (or one spread) per scenario
    probabilities        physical-measure probabilities of those scenarios
    max_loss_per_contract  the defined maximum loss -- the capital genuinely at
                         risk.  For a defined-risk spread this is
                         (width - credit) * 100, plus fees.
    bankroll             total account equity
    """
    pnl = np.asarray(pnl_per_contract, float).ravel()
    prob = np.asarray(probabilities, float).ravel()
    risk = float(max_loss_per_contract)
    if risk <= 0 or bankroll <= 0 or pnl.size == 0:
        return KellySizing(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "invalid inputs")

    x = pnl / risk                                  # per-unit-of-capital-at-risk return
    f_raw = kelly_fraction(x, prob)

    c_unc = uncertainty_shrinkage(sharpe, years_of_evidence)
    c_dd = drawdown_cap(max_drawdown, drawdown_prob)
    c = min(c_unc, c_dd, float(hard_cap_fraction))
    binding = min(
        [("uncertainty shrinkage", c_unc), ("drawdown cap", c_dd), ("hard half-Kelly cap", hard_cap_fraction)],
        key=lambda kv: kv[1],
    )[0]

    f = f_raw * c
    capital = f * bankroll
    if capital / bankroll > max_risk_fraction_per_trade:
        capital = max_risk_fraction_per_trade * bankroll
        binding = f"per-trade risk cap ({max_risk_fraction_per_trade:.1%})"

    n = int(np.floor(capital / risk))               # always round toward zero
    if max_contracts is not None and n > max_contracts:
        n, binding = int(max_contracts), "max_contracts"
    n = max(n, 0)

    actual_risk = n * risk
    f_applied = actual_risk / bankroll
    growth = float(np.sum(prob / prob.sum() * np.log(np.maximum(1.0 + f_applied * x, 1e-12)))) if n else 0.0
    if n == 0:
        binding = ("no Kelly edge (f*=0): the scenario distribution has "
                   "non-positive expected growth" if f_raw <= 0 else
                   f"rounds to zero contracts -- ${capital:,.0f} of budgeted risk "
                   f"vs ${risk:,.0f} needed for one spread")

    return KellySizing(
        contracts=n, f_kelly_raw=float(f_raw), f_applied=float(f_applied), c_total=float(c),
        c_uncertainty=float(c_unc), c_drawdown=float(c_dd),
        capital_at_risk=float(actual_risk), capital_fraction=float(f_applied),
        expected_log_growth=growth, binding_constraint=binding,
        scenarios={"n_scenarios": int(pnl.size),
                   "ev_per_contract": float(np.sum(prob / prob.sum() * pnl)),
                   "worst": float(pnl.min()), "best": float(pnl.max())},
    )
