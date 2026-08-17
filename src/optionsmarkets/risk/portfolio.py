"""Portfolio-level risk aggregation and limits.

Feeds the ``risk.*`` gates in :mod:`optionsmarkets.policy.decide`.  Two ideas
drive the design:

**Greeks aggregate in dollars, not in units.** A delta of 0.30 on a $780
underlying and a delta of 0.30 on a $40 underlying are not the same risk.
Everything here is converted to dollar sensitivity before it is summed or
compared to a limit.

**Correlation is the thing that actually kills a short-vol book.** If you are
short volatility on forty names you have one bet, not forty -- they load on the
same market-volatility factor and they gap together.  Counting positions or
netting raw vega across underlyings understates the true exposure, sometimes by
an order of magnitude.  ``factor_exposure`` collapses the book onto a small set
of common factors so the Kelly constraint is applied to the bet you are
actually making.  Until per-name betas are estimated from data, it uses
beta-to-index proxies and reports the assumption explicitly rather than
pretending the netting is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PositionRisk", "PortfolioRisk", "aggregate", "LimitBreach", "check_limits"]


@dataclass
class PositionRisk:
    underlying: str
    spot: float
    contracts: int
    greeks: dict            # trader units, per spread, from Structure.greeks
    max_loss: float         # defined risk, dollars, per spread
    dte: int
    beta: float = 1.0       # to the risk index (SPX/SPY); 1.0 until estimated

    @property
    def delta_dollars(self) -> float:
        """Delta x spot: the dollar move in the position per 1% move in the name."""
        return self.greeks.get("delta", 0.0) * self.contracts * self.spot * 0.01

    @property
    def beta_delta_dollars(self) -> float:
        """Index-equivalent delta.  This is the number that nets across names."""
        return self.delta_dollars * self.beta

    @property
    def vega_dollars(self) -> float:
        """Vega is already per vol point in trader units -- dollars per 1 vol point."""
        return self.greeks.get("vega", 0.0) * self.contracts

    @property
    def theta_dollars(self) -> float:
        return self.greeks.get("theta", 0.0) * self.contracts

    @property
    def gamma_dollars_per_pct(self) -> float:
        """Dollar change in delta_dollars per 1% move -- the convexity that
        matters operationally, because it is what you cannot hedge through a gap."""
        return self.greeks.get("gamma", 0.0) * self.contracts * (self.spot * 0.01) ** 2

    @property
    def capital_at_risk(self) -> float:
        return self.max_loss * self.contracts


@dataclass
class PortfolioRisk:
    bankroll: float
    positions: list[PositionRisk] = field(default_factory=list)

    def _sum(self, attr: str) -> float:
        return float(sum(getattr(p, attr) for p in self.positions))

    @property
    def delta_dollars(self) -> float:
        return self._sum("beta_delta_dollars")

    @property
    def vega_dollars(self) -> float:
        return self._sum("vega_dollars")

    @property
    def theta_dollars(self) -> float:
        return self._sum("theta_dollars")

    @property
    def gamma_dollars(self) -> float:
        return self._sum("gamma_dollars_per_pct")

    @property
    def capital_at_risk(self) -> float:
        return self._sum("capital_at_risk")

    def by_underlying(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.positions:
            out[p.underlying] = out.get(p.underlying, 0) + 1
        return out

    def concentration(self) -> dict[str, float]:
        """Fraction of total capital-at-risk per underlying."""
        tot = max(self.capital_at_risk, 1e-9)
        out: dict[str, float] = {}
        for p in self.positions:
            out[p.underlying] = out.get(p.underlying, 0.0) + p.capital_at_risk / tot
        return out

    def factor_exposure(self) -> dict:
        """Collapse the book onto common factors.

        ``naive_vega`` sums vega as if every name were independent;
        ``factor_vega`` weights by beta, which is the exposure that actually
        moves together.  The ratio is the diversification you are NOT getting.
        A ratio near 1.0 means the book is one trade wearing several hats.
        """
        naive = float(sum(abs(p.vega_dollars) for p in self.positions))
        signed = float(sum(p.vega_dollars * p.beta for p in self.positions))
        return {
            "naive_gross_vega": naive,
            "factor_vega": signed,
            "concentration_ratio": abs(signed) / naive if naive > 1e-9 else 0.0,
            "note": ("beta=1.0 proxies until per-name betas are estimated from "
                     "returns; treat factor_vega as an upper bound on netting"),
        }

    def worst_case_simultaneous_loss(self) -> float:
        """Every defined-risk position at its maximum loss at once.

        Not a scenario forecast -- a contractual bound.  It is the number to
        compare against the account, because a correlated gap is precisely when
        several short-premium spreads go to max loss together, and that is the
        one day the diversification assumption fails completely.
        """
        return self.capital_at_risk


@dataclass
class LimitBreach:
    name: str
    value: float
    limit: float
    detail: str


def aggregate(positions: list[PositionRisk], bankroll: float) -> PortfolioRisk:
    return PortfolioRisk(bankroll=bankroll, positions=list(positions))


def check_limits(
    pf: PortfolioRisk, *,
    max_delta_pct: float = 0.15,
    max_vega_pct: float = 0.02,
    max_gamma_pct: float = 0.05,
    max_capital_pct: float = 0.25,
    max_single_underlying_pct: float = 0.40,
    max_positions_per_underlying: int = 2,
) -> list[LimitBreach]:
    """Return every breached limit.  Empty list means the book is inside its
    mandate.  Called before sizing a new position, with that position included."""
    b = max(pf.bankroll, 1e-9)
    out: list[LimitBreach] = []

    checks = [
        ("portfolio_delta", abs(pf.delta_dollars) / b, max_delta_pct,
         "beta-weighted dollar delta per 1% index move"),
        ("portfolio_vega", abs(pf.vega_dollars) / b, max_vega_pct,
         "dollar P&L per 1 volatility point"),
        ("portfolio_gamma", abs(pf.gamma_dollars) / b, max_gamma_pct,
         "dollar delta change per 1% move -- the part you cannot hedge through a gap"),
        ("capital_at_risk", pf.capital_at_risk / b, max_capital_pct,
         "sum of contractual max losses; assume they can all hit at once"),
    ]
    for name, val, lim, why in checks:
        if val > lim:
            out.append(LimitBreach(name, val, lim, f"{val:.2%} vs limit {lim:.2%} -- {why}"))

    for sym, frac in pf.concentration().items():
        if frac > max_single_underlying_pct:
            out.append(LimitBreach(f"concentration.{sym}", frac, max_single_underlying_pct,
                                   f"{sym} is {frac:.0%} of capital at risk"))
    for sym, n in pf.by_underlying().items():
        if n > max_positions_per_underlying:
            out.append(LimitBreach(f"positions.{sym}", float(n), float(max_positions_per_underlying),
                                   f"{n} open positions on {sym}"))
    return out
