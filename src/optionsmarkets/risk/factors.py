"""Factor-level correlation, and Kelly on the bet you are actually making.

BLUEPRINT.md section 9 and phase 13.  The point, restated because it is the
single most expensive mistake available in this strategy:

    Kelly assumes independence across bets.  If you are short volatility on
    forty names you have ONE bet, not forty.

Counting positions, or netting raw vega across underlyings, understates the true
exposure -- sometimes by an order of magnitude -- because every short-premium
position loads on the same market-volatility factor and they all gap together.
The per-underlying cap and the portfolio vega limit are blunt proxies for this;
what follows is the actual decomposition.

Three pieces:

**Betas from returns** (:func:`estimate_betas`), with the standard error and
R-squared reported, because a beta estimated from 30 noisy days is not a number
you should be netting millions of dollars of exposure against.  Names whose beta
is not identified fall back to 1.0 -- the conservative direction, since it
assumes maximum co-movement.

**A factor decomposition** (:class:`FactorModel`) that splits gross vega into a
common market-vol component and a residual dispersion component, and reports the
**effective number of independent bets**.  For a book of one-directional
short-vol positions that number collapses toward 1 however many tickers are in
it, and seeing it collapse is the warning the position count cannot give you.

**Portfolio Kelly on the factor** (:func:`portfolio_kelly`), which couples the
per-position scenario P&Ls through their common factor before solving for the
growth-optimal fraction, instead of sizing each position as if the others were
not there.  The difference is not marginal: with five perfectly correlated
short-vol spreads, independent sizing bets five times the Kelly fraction of the
single bet they collectively are, and 5x Kelly has *negative* growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sizing.kelly import kelly_fraction
from .portfolio import PortfolioRisk

__all__ = ["BetaEstimate", "estimate_betas", "FactorExposure", "FactorModel",
           "portfolio_kelly", "PortfolioKellyResult"]


# ----------------------------------------------------------------------------
# betas
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class BetaEstimate:
    symbol: str
    beta: float
    stderr: float
    r2: float
    n: int
    identified: bool
    detail: str = ""

    @property
    def t_stat(self) -> float:
        return float(self.beta / self.stderr) if self.stderr > 0 else np.nan


def estimate_betas(returns_by_symbol: dict, index_returns, *,
                   min_obs: int = 60, min_t: float = 2.0) -> dict[str, BetaEstimate]:
    """OLS beta of each name on the index, with honest identification flags.

    ``min_obs`` and ``min_t`` decide whether a beta is *identified*.  An
    unidentified beta is reported with ``beta=1.0``, which is the conservative
    choice in this application: beta enters as a co-movement multiplier, and
    assuming full co-movement overstates concentration.  Overstating
    concentration makes the book smaller; understating it is what blows it up.
    """
    y_idx = np.asarray(index_returns, float).ravel()
    out: dict[str, BetaEstimate] = {}
    for sym, r in returns_by_symbol.items():
        x = np.asarray(r, float).ravel()
        n = min(x.size, y_idx.size)
        xi, yi = x[-n:], y_idx[-n:]
        m = np.isfinite(xi) & np.isfinite(yi)
        xi, yi = xi[m], yi[m]
        if xi.size < min_obs or np.std(yi) <= 0:
            out[sym] = BetaEstimate(sym, 1.0, np.nan, np.nan, int(xi.size), False,
                                    f"only {xi.size} usable observations "
                                    f"(need {min_obs}); beta defaulted to 1.0, the "
                                    f"conservative direction")
            continue
        A = np.column_stack([np.ones_like(yi), yi])
        coef, *_ = np.linalg.lstsq(A, xi, rcond=None)
        resid = xi - A @ coef
        dof = max(xi.size - 2, 1)
        s2 = float(resid @ resid) / dof
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        ss_tot = float(np.sum((xi - xi.mean()) ** 2))
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
        beta = float(coef[1])
        ident = se > 0 and abs(beta / se) >= min_t
        out[sym] = BetaEstimate(
            sym, beta if ident else 1.0, se, r2, int(xi.size), bool(ident),
            "" if ident else (f"beta {beta:.3f} not distinguishable from noise "
                              f"(t={beta / se if se > 0 else float('nan'):.2f}); "
                              f"defaulted to 1.0"))
    return out


# ----------------------------------------------------------------------------
# factor decomposition
# ----------------------------------------------------------------------------

@dataclass
class FactorExposure:
    market_vol_dollars: float        # beta-weighted net vega: the real bet
    gross_vega_dollars: float        # sum of |vega|: what a naive count sees
    residual_vega_dollars: float     # dispersion: what genuinely diversifies
    effective_bets: float
    concentration_ratio: float
    beta_weighted_delta_dollars: float
    worst_case_simultaneous_loss: float
    per_underlying: dict = field(default_factory=dict)
    detail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def report(self) -> str:
        return "\n".join([
            "FACTOR EXPOSURE",
            f"  market-vol factor    ${self.market_vol_dollars:>12,.0f} per vol point "
            f"(beta-weighted net vega -- THE bet)",
            f"  gross vega           ${self.gross_vega_dollars:>12,.0f} "
            f"(what a position count sees)",
            f"  residual/dispersion  ${self.residual_vega_dollars:>12,.0f} "
            f"(the part that actually diversifies)",
            f"  effective bets       {self.effective_bets:>13.2f}  "
            f"(vs {len(self.per_underlying)} underlying(s))",
            f"  concentration ratio  {self.concentration_ratio:>13.2f}  "
            f"(1.0 = one trade wearing several hats)",
            f"  beta-weighted delta  ${self.beta_weighted_delta_dollars:>12,.0f} per 1% index move",
            f"  contractual worst    ${self.worst_case_simultaneous_loss:>12,.0f} "
            f"(every defined-risk leg at max loss at once)",
        ] + [f"  note: {d}" for d in self.detail])


@dataclass
class FactorModel:
    """Collapse a book onto a market-volatility factor plus a residual.

    The model is deliberately simple -- one common factor, plus whatever does not
    load on it -- because with a handful of positions on correlated index
    products there is not enough data to identify more, and a richer factor
    model estimated from too little data understates concentration exactly when
    it matters.  One factor you can defend beats five you cannot.
    """
    betas: dict[str, BetaEstimate] = field(default_factory=dict)
    # Correlation assumed between residual (dispersion) components when no
    # estimate is available. NOT zero: idiosyncratic option vols are positively
    # correlated in a selloff, and assuming independence is the assumption that
    # makes a short-vol book look diversified right up until it is not.
    residual_correlation: float = 0.30

    def beta(self, symbol: str) -> float:
        est = self.betas.get(symbol)
        return float(est.beta) if est is not None else 1.0

    def exposure(self, pf: PortfolioRisk) -> FactorExposure:
        if not pf.positions:
            return FactorExposure(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, {},
                                  ["no open positions"])
        detail: list[str] = []
        vegas, betas, syms = [], [], []
        for p in pf.positions:
            b = self.beta(p.underlying)
            vegas.append(float(p.vega_dollars))
            betas.append(b)
            syms.append(p.underlying)
            est = self.betas.get(p.underlying)
            if est is not None and not est.identified:
                detail.append(f"{p.underlying}: {est.detail}")

        v = np.asarray(vegas, float)
        b = np.asarray(betas, float)
        market = float(np.sum(v * b))
        gross = float(np.sum(np.abs(v)))
        # Residual is what is left after the common component -- the genuinely
        # name-specific part, which is the only part that diversifies.
        resid = float(np.sum(np.abs(v - b * (market / max(np.sum(b**2), 1e-12)) * b)))

        eff = _effective_bets(v, b, self.residual_correlation)
        return FactorExposure(
            market_vol_dollars=market, gross_vega_dollars=gross,
            residual_vega_dollars=resid,
            effective_bets=eff,
            concentration_ratio=abs(market) / gross if gross > 1e-9 else 0.0,
            beta_weighted_delta_dollars=float(pf.delta_dollars),
            worst_case_simultaneous_loss=float(pf.capital_at_risk),
            per_underlying={s: float(x) for s, x in zip(syms, vegas)},
            detail=detail or ["all betas identified from returns"],
        )


def _effective_bets(v: np.ndarray, b: np.ndarray, rho_resid: float) -> float:
    """Effective number of independent bets = (sum w)^2 / (w' C w), w = |exposure|.

    With a single common factor and correlated residuals, the correlation
    between any two positions is bounded below by ``rho_resid`` and rises toward
    1 as their betas align.  The ratio above is the standard diversification
    measure: it equals N for N uncorrelated equal bets and 1 for N perfectly
    correlated ones.  For a book of same-signed short-vol spreads on index
    products it lands near 1, which is the number the Kelly constraint should be
    applied to -- not the position count.
    """
    w = np.abs(v)
    if w.sum() <= 0:
        return 0.0
    n = w.size
    if n == 1:
        return 1.0
    bn = b / max(float(np.sqrt(np.mean(b**2))), 1e-12)
    C = np.full((n, n), float(rho_resid))
    np.fill_diagonal(C, 1.0)
    common = np.outer(bn, bn) / max(float(np.max(np.abs(np.outer(bn, bn)))), 1e-12)
    C = np.clip(np.maximum(C, np.abs(common)), -1.0, 1.0)
    np.fill_diagonal(C, 1.0)
    denom = float(w @ C @ w)
    return float(w.sum() ** 2 / denom) if denom > 0 else 1.0


def _effective_bets_empirical(P: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Effective bets from the probability-weighted correlation of scenario P&Ls.

    Same ratio as :func:`_effective_bets`, but with the correlation MEASURED
    from the joint scenario grid rather than assumed from betas.  Returns N for
    N uncorrelated equal-risk positions and 1 for N identical ones, which is the
    behaviour the Kelly constraint needs: it must be applied to the number of
    bets you actually have, not the number of tickers you have.
    """
    n = P.shape[0]
    if n == 1:
        return 1.0
    mu = P @ p
    dev = P - mu[:, None]
    cov = (dev * p[None, :]) @ dev.T
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    C = cov / np.outer(sd, sd)
    C = np.clip(np.nan_to_num(C, nan=1.0), -1.0, 1.0)
    ww = np.abs(np.asarray(w, float))
    denom = float(ww @ C @ ww)
    return float(ww.sum() ** 2 / denom) if denom > 0 else 1.0


# ----------------------------------------------------------------------------
# portfolio Kelly
# ----------------------------------------------------------------------------

@dataclass
class PortfolioKellyResult:
    f_joint: float
    f_if_independent: float
    overbet_multiple: float
    effective_bets: float
    n_positions: int
    expected_log_growth: float
    detail: str = ""

    def explain(self) -> str:
        return (f"joint Kelly f*={self.f_joint:.4f} across {self.n_positions} position(s) "
                f"({self.effective_bets:.2f} effective bets); sizing them independently "
                f"would stake {self.f_if_independent:.4f}, i.e. "
                f"{self.overbet_multiple:.2f}x the joint optimum. {self.detail}")


def portfolio_kelly(scenario_pnls, probabilities, capital_at_risk,
                    *, betas=None, factor_loading: float = 1.0) -> PortfolioKellyResult:
    """Growth-optimal total stake for a book whose positions share a factor.

    Parameters
    ----------
    scenario_pnls
        Sequence of per-position P&L arrays, all on the SAME scenario grid.  The
        shared grid is what couples them: scenario ``s`` is one state of the
        world, so summing across positions within a scenario reproduces the
        joint distribution without needing a copula.  This is why the scenario
        representation was chosen in the first place.
    probabilities
        Physical-measure probability of each scenario.
    capital_at_risk
        Per-position defined maximum loss.

    Returns the joint fraction alongside what independent sizing would have
    staked, and the ratio between them.  The ratio is the number to look at: it
    is the factor by which per-position sizing overbets, and since excess growth
    scales as ``2c - c^2``, an overbet multiple of 2 earns ZERO growth and
    anything beyond it is negative despite a genuine positive edge.
    """
    pnls = [np.asarray(x, float).ravel() for x in scenario_pnls]
    if not pnls:
        return PortfolioKellyResult(0.0, 0.0, 0.0, 0.0, 0, 0.0, "no positions")
    n_s = min(len(x) for x in pnls)
    pnls = [x[:n_s] for x in pnls]
    p = np.asarray(probabilities, float).ravel()[:n_s]
    p = p / p.sum()
    risk = np.asarray(capital_at_risk, float).ravel()
    if risk.size != len(pnls) or np.any(risk <= 0):
        return PortfolioKellyResult(0.0, 0.0, 0.0, 0.0, len(pnls), 0.0,
                                    "invalid capital-at-risk vector")

    b = np.ones(len(pnls)) if betas is None else np.asarray(betas, float).ravel()
    total_risk = float(risk.sum())
    P = np.stack(pnls, axis=0)

    # JOINT. Positions are combined WITHIN each scenario, so their co-movement is
    # exact rather than assumed -- this is the payoff of carrying a shared
    # scenario grid instead of a correlation matrix. The stake is allocated
    # pro-rata to capital at risk, so the solved fraction is the TOTAL fraction
    # of wealth staked across the whole book.
    agg = np.sum(P * (b[:, None] * factor_loading), axis=0)
    x_joint = agg / total_risk
    f_joint = kelly_fraction(x_joint, p)

    # INDEPENDENT. Each position solves its own Kelly problem as though it were
    # the only bet, and each answer is a fraction of WEALTH -- so the book ends
    # up staking the SUM of them. That sum is the whole point: it is not
    # normalised by total risk, because normalising is exactly the step that
    # hides the overbetting this function exists to measure.
    f_indep = float(sum(kelly_fraction(x / r, p) for x, r in zip(pnls, risk)))

    growth = float(np.sum(p * np.log(np.maximum(1.0 + f_joint * x_joint, 1e-12))))
    # Effective bets from the EMPIRICAL correlation of the scenario P&Ls, not
    # from an assumed structure. The arrays are right here; assuming a
    # correlation when it can be measured would be inventing information.
    eff = _effective_bets_empirical(P, p, risk)
    return PortfolioKellyResult(
        f_joint=float(f_joint), f_if_independent=f_indep,
        overbet_multiple=float(f_indep / f_joint) if f_joint > 1e-12 else np.inf,
        effective_bets=float(eff), n_positions=len(pnls),
        expected_log_growth=growth,
        detail=("Excess growth scales as 2c - c^2, so an overbet multiple of 2.0 earns "
                "zero growth and beyond it growth is negative despite positive edge."),
    )
