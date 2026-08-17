"""The BUY / SELL / HOLD decision, and why HOLD is the default.

A decision is emitted only when EVERY gate passes.  Gates are hard predicates,
not weighted scores: a score lets a strong signal on one axis paper over a
disqualifying failure on another, and in options the disqualifying failures
(stale data, unfittable surface, illiquid strike, event inside the window) are
precisely the ones that produce the most confident-looking edge.

Gate order matters -- cheapest and most disqualifying first:

  1. DATA        quotes fresh, two-sided, surface fit converged and arb-free
  2. LIQUIDITY   spread, open interest, volume on every leg
  3. MODEL       conformal layer not in kill state, calibration reliability ok
  4. EDGE        signed edge exceeds the *cost-adjusted* threshold
  5. RISK        portfolio Greek limits, correlation/concentration, event window
  6. SIZE        Kelly chain returns at least one contract

Direction then follows from the sign of the variance risk premium, not from a
view on the underlying.  This is the "sensitivities, not directional bias"
requirement made concrete: the system is long or short *volatility*, and its
delta is a residual to be managed, not a bet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

__all__ = ["Action", "Gate", "Decision", "Thresholds", "decide", "rank_and_select"]


class Action(str, Enum):
    BUY = "BUY"      # pay premium / open a debit structure
    SELL = "SELL"    # collect premium / open a credit structure
    HOLD = "HOLD"    # no action
    CLOSE = "CLOSE"  # exit an existing position


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str = ""
    value: float | None = None

    def __str__(self):
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class Decision:
    action: Action
    structure_name: str
    contracts: int
    gates: list[Gate]
    rationale: str
    metrics: dict = field(default_factory=dict)

    @property
    def blocked_by(self) -> list[str]:
        return [g.name for g in self.gates if not g.passed]

    def report(self) -> str:
        lines = [f"DECISION: {self.action.value}  x{self.contracts}  {self.structure_name}",
                 f"  {self.rationale}", "  Gates:"]
        lines += [f"    {g}" for g in self.gates]
        return "\n".join(lines)


@dataclass
class Thresholds:
    """Every number here is a policy choice and should be re-derived from your
    own out-of-sample record, not inherited.  The defaults are deliberately
    conservative for a $5k-$50k Reg-T account where commission and spread drag
    dominate."""
    max_quote_age_s: float = 900.0          # 15 min: the yfinance chain latency
    max_rel_spread: float = 0.15
    min_open_interest: float = 25.0
    min_leg_volume: float = 0.0
    max_surface_rmse_vol: float = 0.015
    min_g: float = -1e-8                    # butterfly-arbitrage tolerance
    min_edge_z: float = 0.05
    min_vrp_vol_points: float = 2.0         # below this, VRP is inside model error
    # EV is compared on an ANNUALISED basis. A flat "EV must beat 5% of risk"
    # threshold silently favours long-dated trades, which tie up capital for
    # months to earn the same nominal number a 3-week trade earns once.
    # Growth rate is what Kelly maximises, so growth rate is what we gate on.
    min_annualised_ev_on_risk: float = 0.12
    min_pop: float = 0.55
    min_pop_edge_pts: float = 3.0           # our PoP must beat the market's by this
    # Credit/width is a property of the STRIKES, so it is measured at mid --
    # loading it with slippage and fees double-counts costs that the EV gate
    # already charges. 12% is a defensible floor for 20-30 delta shorts; below
    # it the payoff asymmetry needs a win rate the model cannot honestly claim.
    min_credit_to_width: float = 0.12
    # The gate that actually bites on a small account: expected value must
    # clear a multiple of round-trip transaction cost. At $0.65/contract a
    # 1-lot vertical costs $2.60 to open and close; an edge of $8 is not an
    # edge, it is a rounding error with extra steps.
    min_ev_to_cost_multiple: float = 3.0
    max_portfolio_delta_pct: float = 0.15   # |delta $| / bankroll
    max_portfolio_vega_pct: float = 0.02
    max_positions_per_underlying: int = 2
    block_days_around_earnings: int = 2
    min_dte: int = 7
    max_dte: int = 60


def decide(
    edge, sizing, structure, *,
    quote_age_s: float,
    slice_fit,
    legs_tradeable: list[bool],
    conformal_killed: bool = False,
    calibration_rel: float = 0.0,
    portfolio_delta_dollars: float = 0.0,
    portfolio_vega_dollars: float = 0.0,
    bankroll: float = 0.0,
    dte: int = 30,
    days_to_earnings: int | None = None,
    open_positions_same_underlying: int = 0,
    credit_to_width: float | None = None,
    round_trip_cost: float = 0.0,
    thresholds: Thresholds | None = None,
) -> Decision:
    """Run every gate and return the decision plus the full audit trail."""
    th = thresholds or Thresholds()
    g: list[Gate] = []

    # ---- 1. data ---------------------------------------------------------
    g.append(Gate("data.freshness", quote_age_s <= th.max_quote_age_s,
                  f"{quote_age_s:.0f}s old (limit {th.max_quote_age_s:.0f}s)", quote_age_s))
    g.append(Gate("data.surface_fit", bool(slice_fit.ok),
                  f"rmse={slice_fit.rmse_vol:.4f} vol, min_g={slice_fit.min_g:.3e} "
                  f"{slice_fit.detail}", slice_fit.rmse_vol))
    g.append(Gate("data.surface_precision", np.isfinite(slice_fit.rmse_vol)
                  and slice_fit.rmse_vol <= th.max_surface_rmse_vol,
                  f"fit rmse {slice_fit.rmse_vol:.4f} vs limit {th.max_surface_rmse_vol:.4f}"))

    # ---- 2. liquidity ----------------------------------------------------
    g.append(Gate("liquidity.all_legs", all(legs_tradeable),
                  f"{sum(legs_tradeable)}/{len(legs_tradeable)} legs pass spread/OI screen"))

    # ---- 3. model health -------------------------------------------------
    g.append(Gate("model.conformal", not conformal_killed,
                  "coverage collapsed -- model is in kill state" if conformal_killed
                  else "conformal coverage nominal"))
    g.append(Gate("model.calibration", calibration_rel <= 0.02,
                  f"Brier reliability {calibration_rel:.4f} (limit 0.0200)", calibration_rel))

    # ---- 4. edge ---------------------------------------------------------
    vrp_pts = (edge.scenarios.sigma_implied_atm - edge.scenarios.sigma_forecast) * 100.0
    g.append(Gate("edge.z", edge.edge_z >= th.min_edge_z,
                  f"edge z={edge.edge_z:.2f} vs {th.min_edge_z:.2f}", edge.edge_z))
    g.append(Gate("edge.vrp", abs(vrp_pts) >= th.min_vrp_vol_points,
                  f"VRP={vrp_pts:+.2f} vol pts (need |VRP|>={th.min_vrp_vol_points})", vrp_pts))
    ann = edge.ev_pct_of_risk * (365.0 / max(dte, 1))
    g.append(Gate("edge.ev_annualised", ann >= th.min_annualised_ev_on_risk,
                  f"EV={edge.ev_pct_of_risk:.2%} of risk over {dte}d = {ann:.1%} "
                  f"annualised (need {th.min_annualised_ev_on_risk:.0%})", ann))
    ctw = credit_to_width if credit_to_width is not None else \
        edge.max_gain / max(edge.max_loss + edge.max_gain, 1e-9)
    g.append(Gate("edge.credit_to_width", ctw >= th.min_credit_to_width,
                  f"credit/width = {ctw:.1%} at mid (need {th.min_credit_to_width:.0%}) -- "
                  f"below this the payoff asymmetry needs a win rate we cannot claim", ctw))
    if round_trip_cost and round_trip_cost > 0:
        mult = edge.ev_per_spread / round_trip_cost
        g.append(Gate("edge.ev_vs_cost", mult >= th.min_ev_to_cost_multiple,
                      f"EV ${edge.ev_per_spread:,.2f} = {mult:.1f}x round-trip cost "
                      f"${round_trip_cost:.2f} (need {th.min_ev_to_cost_multiple:.0f}x)", mult))
    g.append(Gate("edge.pop", edge.pop >= th.min_pop,
                  f"PoP={edge.pop:.1%} (need {th.min_pop:.0%})", edge.pop))
    pop_edge = (edge.pop - edge.pop_riskneutral) * 100.0
    g.append(Gate("edge.pop_vs_market", pop_edge >= th.min_pop_edge_pts,
                  f"our PoP beats risk-neutral by {pop_edge:+.1f} pts "
                  f"(need {th.min_pop_edge_pts:.1f})", pop_edge))

    # ---- 5. risk ---------------------------------------------------------
    if bankroll > 0:
        g.append(Gate("risk.portfolio_delta",
                      abs(portfolio_delta_dollars) / bankroll <= th.max_portfolio_delta_pct,
                      f"|net delta| = {abs(portfolio_delta_dollars) / bankroll:.1%} of bankroll"))
        g.append(Gate("risk.portfolio_vega",
                      abs(portfolio_vega_dollars) / bankroll <= th.max_portfolio_vega_pct,
                      f"|net vega| = {abs(portfolio_vega_dollars) / bankroll:.2%} of bankroll"))
    g.append(Gate("risk.concentration",
                  open_positions_same_underlying < th.max_positions_per_underlying,
                  f"{open_positions_same_underlying} open on this underlying "
                  f"(max {th.max_positions_per_underlying})"))
    g.append(Gate("risk.dte_window", th.min_dte <= dte <= th.max_dte,
                  f"{dte} DTE (window {th.min_dte}-{th.max_dte})", dte))
    if days_to_earnings is not None:
        g.append(Gate("risk.event_window", abs(days_to_earnings) > th.block_days_around_earnings,
                      f"earnings in {days_to_earnings}d "
                      f"(blackout +/-{th.block_days_around_earnings}d)"))
    g.append(Gate("risk.defined", structure.is_defined_risk(),
                  "every leg is hedged -- max loss is contractual"))

    # ---- 6. size ---------------------------------------------------------
    g.append(Gate("size.contracts", sizing.contracts >= 1,
                  sizing.binding_constraint, float(sizing.contracts)))

    passed = all(x.passed for x in g)
    if not passed:
        return Decision(
            Action.HOLD, structure.name, 0, g,
            "HOLD -- blocked by: " + ", ".join(x.name for x in g if not x.passed),
            {"vrp_vol_points": vrp_pts},
        )

    action = Action.SELL if vrp_pts > 0 else Action.BUY
    rationale = (
        f"{action.value} {sizing.contracts}x {structure.name}: implied "
        f"{edge.scenarios.sigma_implied_atm:.1%} vs forecast "
        f"{edge.scenarios.sigma_forecast:.1%} = {vrp_pts:+.2f} vol points of "
        f"{'premium to harvest' if vrp_pts > 0 else 'cheap convexity to own'}. "
        f"PoP {edge.pop:.1%} (market {edge.pop_riskneutral:.1%}), EV "
        f"${edge.ev_per_spread:,.2f}/spread = {edge.ev_pct_of_risk:.2%} of risk, "
        f"edge z={edge.edge_z:.2f}. {sizing.explain()}"
    )
    return Decision(action, structure.name, sizing.contracts, g, rationale, {
        "vrp_vol_points": vrp_pts, "pop": edge.pop, "ev": edge.ev_per_spread,
        "kelly_raw": sizing.f_kelly_raw, "kelly_applied": sizing.f_applied,
    })


def rank_and_select(candidates, decide_kwargs_for, objective=lambda ed, sz: sz.expected_log_growth):
    """Rank candidate structures by the SAME objective the sizer maximises, and
    select only from those that actually clear the gate stack.

    This is the fix for a failure mode that is easy to build and hard to see:
    a selector that ranks on raw edge will keep nominating the structure with
    the best edge, the policy will keep rejecting it on some other axis, and
    the system will report "no trade" every day while sitting on candidates it
    would have accepted.  Screen first, then rank -- never the reverse.

    Parameters
    ----------
    candidates       iterable of (structure, edge_report, sizing) triples
    decide_kwargs_for  callable (structure, edge, sizing) -> dict of kwargs
                       forwarded to :func:`decide`
    objective        what to maximise among the survivors; defaults to expected
                     log growth, which is the Kelly objective

    Returns
    -------
    (best_triple, best_decision, all_evaluated)   best_* are None if nothing passes.
    """
    evaluated = []
    for st, ed, sz in candidates:
        d = decide(ed, sz, st, **decide_kwargs_for(st, ed, sz))
        evaluated.append((st, ed, sz, d))
    viable = [x for x in evaluated if x[3].action is not Action.HOLD and x[2].contracts >= 1]
    if not viable:
        return None, None, evaluated
    best = max(viable, key=lambda x: objective(x[1], x[2]))
    return (best[0], best[1], best[2]), best[3], evaluated
