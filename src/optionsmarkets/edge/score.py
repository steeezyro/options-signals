"""Edge measurement and the scenario distribution everything else consumes.

The chain is:

  fitted SVI slice
     -> risk-neutral density  (Breeden-Litzenberger, in SVI closed form)
     -> PHYSICAL density      (reweight so the variance matches the RV forecast
                               rather than the implied variance)
     -> scenario P&L of the structure
     -> PoP, EV, RoC, and the Kelly input

The Q -> P step is where the edge actually lives.  The risk-neutral density
prices options; it does not describe what the underlying will do.  The gap
between them is the variance risk premium, and for SPX it is large and
persistently positive -- implied variance runs roughly 2x realised, about 5-7
volatility points.  A system that computes "probability of profit" from the
risk-neutral density is measuring the market's price, not its own forecast,
and will conclude it has no edge anywhere.

Reweighting method: we keep the *shape* of the risk-neutral density (its skew
and kurtosis are real information about the market's fear) and rescale its
width so the second moment matches the horizon-matched RV forecast, then
re-centre on the forward drift.  This is deliberately conservative -- it
assumes the market is right about shape and only potentially wrong about
level, which is the part we have an actual forecasting model for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..surface.svi import SVIParams, svi_density, svi_w

__all__ = ["ScenarioSet", "build_scenarios", "EdgeReport", "score_structure",
           "variance_risk_premium"]


@dataclass
class ScenarioSet:
    S_T: np.ndarray
    prob_P: np.ndarray            # physical measure
    prob_Q: np.ndarray            # risk-neutral
    forward: float
    T: float
    sigma_implied_atm: float
    sigma_forecast: float

    @property
    def vrp(self) -> float:
        """IV^2 - E_P[RV^2], in variance units."""
        return float(self.sigma_implied_atm**2 - self.sigma_forecast**2)


def _normalise(p, dk):
    p = np.maximum(np.asarray(p, float), 0.0)
    m = float(np.sum(p * dk))
    return p / m if m > 0 else p


def build_scenarios(
    params: SVIParams, F: float, T: float, sigma_forecast: float,
    n: int = 1201, halfwidth_sd: float = 6.0, tail_inflation: float = 1.25,
) -> ScenarioSet:
    """Risk-neutral and physical scenario grids over terminal underlying price.

    ``tail_inflation`` widens the physical grid beyond anything the forecast
    implies.  Kelly assumes continuous rebalancing; a short-gamma position
    cannot be trimmed through a gap, so the effective worst case is worse than
    a fitted density suggests.  Extending the grid into the tail is cheap
    insurance against sizing off a distribution that has never seen a limit-down
    open.
    """
    w_atm = float(svi_w(0.0, params))
    sig_atm = float(np.sqrt(max(w_atm, 1e-12) / T))
    span = halfwidth_sd * max(sig_atm, sigma_forecast) * np.sqrt(T) * tail_inflation
    k = np.linspace(-span, span, int(n))
    dk = float(k[1] - k[0])

    qk = _normalise(svi_density(k, params), dk)

    # Q -> P: keep shape, rescale width to the forecast, recentre on the forward.
    ratio = float(np.clip(sigma_forecast / max(sig_atm, 1e-8), 0.25, 4.0))
    mu_q = float(np.sum(k * qk) * dk)
    k_scaled = mu_q + (k - mu_q) / ratio            # sample points that map to k under scaling
    pk = _normalise(np.interp(k_scaled, k, qk) / ratio, dk)

    S_T = F * np.exp(k)
    return ScenarioSet(S_T, pk * dk, qk * dk, float(F), float(T), sig_atm, float(sigma_forecast))


def variance_risk_premium(sigma_implied: float, sigma_forecast: float) -> dict:
    """VRP in variance and volatility-point terms, plus the ratio.

    Sign convention: POSITIVE means options are rich relative to the forecast
    (sell premium); negative means cheap (buy premium).
    """
    iv2, rv2 = float(sigma_implied) ** 2, float(sigma_forecast) ** 2
    return {
        "vrp_variance": iv2 - rv2,
        "vrp_vol_points": (float(sigma_implied) - float(sigma_forecast)) * 100.0,
        "iv_rv_ratio": float(sigma_implied / sigma_forecast) if sigma_forecast > 0 else np.inf,
        "direction": "SELL premium (options rich)" if iv2 > rv2 else "BUY premium (options cheap)",
    }


@dataclass
class EdgeReport:
    pop: float                      # P(P&L > 0) under the PHYSICAL measure
    pop_riskneutral: float          # under Q -- the market's own number
    ev_per_spread: float
    ev_pct_of_risk: float
    max_loss: float
    max_gain: float
    return_on_capital: float        # max_gain / max_loss
    expected_roc: float             # ev / max_loss
    breakevens: list[float]
    cvar_5: float
    edge_z: float
    scenarios: ScenarioSet
    payoff: np.ndarray = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "PoP (physical)": f"{self.pop:.1%}",
            "PoP (risk-neutral)": f"{self.pop_riskneutral:.1%}",
            "PoP edge vs market": f"{(self.pop - self.pop_riskneutral) * 100:+.1f} pts",
            "EV / spread": f"${self.ev_per_spread:,.2f}",
            "EV as % of capital at risk": f"{self.ev_pct_of_risk:.2%}",
            "Max gain / Max loss": f"${self.max_gain:,.0f} / ${self.max_loss:,.0f}",
            "Return on capital (max)": f"{self.return_on_capital:.1%}",
            "Expected RoC": f"{self.expected_roc:.2%}",
            "CVaR(5%)": f"${self.cvar_5:,.2f}",
            "Edge z-score": f"{self.edge_z:.2f}",
            "Breakevens": [round(b, 2) for b in self.breakevens],
        }


def score_structure(structure, scen: ScenarioSet, net_price: float,
                    fees: float = 0.0) -> EdgeReport:
    """Evaluate one structure against the scenario set.

    ``net_price`` must be the MARKETABLE net (crossing the spread) plus fees --
    scoring at mid manufactures edge that the fill will take straight back.
    """
    pl = structure.payoff(scen.S_T, net_price) - fees
    pP, pQ = scen.prob_P / scen.prob_P.sum(), scen.prob_Q / scen.prob_Q.sum()

    win = pl > 0
    ev = float(np.sum(pP * pl))
    max_loss = float(max(structure.max_loss(net_price) + fees, 1e-9))
    max_gain = float(structure.max_gain(net_price) - fees)

    order = np.argsort(pl)
    csum = np.cumsum(pP[order])
    tail = order[csum <= 0.05]
    cvar = float(np.sum(pP[tail] * pl[tail]) / max(pP[tail].sum(), 1e-12)) if tail.size else float(pl.min())

    sd = float(np.sqrt(max(np.sum(pP * (pl - ev) ** 2), 1e-18)))
    return EdgeReport(
        pop=float(np.sum(pP[win])), pop_riskneutral=float(np.sum(pQ[win])),
        ev_per_spread=ev, ev_pct_of_risk=ev / max_loss,
        max_loss=max_loss, max_gain=max_gain,
        return_on_capital=max_gain / max_loss, expected_roc=ev / max_loss,
        breakevens=structure.breakevens(net_price), cvar_5=cvar,
        edge_z=ev / sd, scenarios=scen, payoff=pl,
    )
