"""Schwab execution: order ticket, limit-price ladder, and the click path.

Facts below were verified against Schwab's April 2026 Pricing Guide, Schwab's
options application form, thinkorswim documentation, FINRA/SEC/OCC primary
sources, and the Trader API enum set.  Items Schwab does not publish are
marked UNVERIFIED in :data:`OPEN_QUESTIONS` -- they are operational hazards,
not footnotes, and the runbook tells you to resolve them by phone.

Key economics for a small account, which the sizing layer must see:

  * commission is $0.65 PER CONTRACT, counted across ALL legs.  A 1-lot
    vertical = 2 contracts = $1.30; a 1-lot iron condor = 4 = $2.60.  On a
    $50-credit vertical that is 2.6% of the credit before you have any P&L.
  * per-contract fees are WAIVED on buy-to-close executed online at $0.05 or
    less.  This changes the endgame: closing a near-worthless short leg is
    free, so there is no fee argument for carrying assignment risk into
    expiration.
  * exercise and assignment are $0 commission.  The fee argument favours
    letting it run; the risk argument does not, and the risk argument wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..domain.structures import MULTIPLIER, Right, Side, Structure, occ_symbol

__all__ = ["SchwabCosts", "OrderTicket", "build_ticket", "OPEN_QUESTIONS"]


@dataclass(frozen=True)
class SchwabCosts:
    """Schwab retail options cost model (Pricing Guide, effective April 2026)."""
    per_contract: float = 0.65
    base_ticket: float = 0.0
    close_fee_waiver_price: float = 0.05      # BTC online at <= $0.05: fee waived
    # Regulatory pass-throughs.  Schwab does NOT itemise these -- it bundles
    # them into a single discretionary "Industry Fee" and reserves the right
    # for it to exceed what it actually paid.  Treat as an estimate.
    sec_31_per_million: float = 20.60         # effective 2026-04-04; was $0 for 11 months
    finra_taf_per_contract: float = 0.00279   # sales only
    orf_per_contract: float = 0.0             # exchange-set, varies monthly: UNVERIFIED

    def estimate(self, structure: Structure, contracts: int, credit_received: float = 0.0) -> dict:
        n_legs_contracts = sum(lg.quantity for lg in structure.legs) * contracts
        commission = self.base_ticket + self.per_contract * n_legs_contracts
        sold = sum(lg.quantity for lg in structure.legs if lg.side is Side.SELL) * contracts
        taf = self.finra_taf_per_contract * sold
        orf = self.orf_per_contract * n_legs_contracts
        sec31 = max(credit_received, 0.0) / 1e6 * self.sec_31_per_million
        total = commission + taf + orf + sec31
        return {
            "commission": round(commission, 2), "finra_taf": round(taf, 4),
            "orf": round(orf, 4), "sec_31": round(sec31, 4),
            "total_open": round(total, 2),
            "total_round_trip_estimate": round(total + commission, 2),
            "note": ("Round-trip assumes you pay commission again to close. "
                     "If every short leg is bought back online at <= $0.05 the "
                     "closing per-contract fee is waived."),
        }


@dataclass
class OrderTicket:
    underlying: str
    structure_name: str
    contracts: int
    order_type: str                 # NET_DEBIT | NET_CREDIT
    limit_price: float              # per spread, positive number
    duration: str
    legs: list[dict]
    complex_strategy: str
    cost_estimate: dict
    price_ladder: list[float]
    natural: float
    mid: float
    api_json: dict
    web_steps: list[str]
    tos_steps: list[str]
    exit_plan: list[str]
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        L = [
            "=" * 78,
            f"ORDER TICKET -- {self.underlying}  {self.structure_name}  x{self.contracts}",
            "=" * 78,
            f"  Order type : {self.order_type}",
            f"  Limit      : ${self.limit_price:.2f} per spread "
            f"({'debit paid' if self.order_type == 'NET_DEBIT' else 'credit received'})",
            f"  Mid ${self.mid:.2f} | Natural (crossing the spread) ${self.natural:.2f}",
            f"  Duration   : {self.duration}",
            "", "  LEGS:",
        ]
        for i, lg in enumerate(self.legs, 1):
            L.append(f"    {i}. {lg['instruction']:14s} {lg['quantity']:>3d}  "
                     f"{lg['description']}   [{lg['symbol']}]")
        L += ["", f"  Est. cost to open: ${self.cost_estimate['total_open']:.2f} "
                  f"(commission ${self.cost_estimate['commission']:.2f} + fees)"]
        L += ["", "  PRICE LADDER (start passive, walk toward natural):",
              "    " + "  ->  ".join(f"${p:.2f}" for p in self.price_ladder)]
        L += ["", "  SCHWAB.COM WEB:"] + [f"    {s}" for s in self.web_steps]
        L += ["", "  THINKORSWIM DESKTOP:"] + [f"    {s}" for s in self.tos_steps]
        L += ["", "  EXIT PLAN (set this before you send the open):"] + [f"    {s}" for s in self.exit_plan]
        if self.warnings:
            L += ["", "  WARNINGS:"] + [f"    !! {w}" for w in self.warnings]
        L.append("=" * 78)
        return "\n".join(L)


def _limit_ladder(mid: float, natural: float, steps: int = 5) -> list[float]:
    """Walk from mid toward the natural price in $0.01 increments.

    Schwab's Walk Limit order type automates exactly this: place, wait, cancel,
    replace one increment closer, repeat.  Schwab documents it as
    'particularly useful in multi-leg options strategies', because each leg
    carries its own bid/ask and the composite natural is punitively wide.
    Starting at the natural price on a 4-leg structure typically gives up more
    than the whole modelled edge.
    """
    if not (np.isfinite(mid) and np.isfinite(natural)):
        return []
    lo, hi = min(mid, natural), max(mid, natural)
    vals = np.round(np.linspace(mid, natural, max(steps, 2)), 2)
    seen, out = set(), []
    for v in vals:
        v = float(np.clip(v, lo, hi))
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def build_ticket(
    structure: Structure, contracts: int, spot: float,
    *, duration: str = "DAY", costs: SchwabCosts | None = None,
    profit_target_pct: float = 0.50, stop_multiple: float = 2.0,
    manage_dte: int = 21,
) -> OrderTicket:
    """Produce a fully-specified, human-executable Schwab order ticket."""
    costs = costs or SchwabCosts()
    mid = structure.net_price("mid")
    natural = structure.net_price("marketable")
    is_debit = natural > 0
    order_type = "NET_DEBIT" if is_debit else "NET_CREDIT"

    limit = abs(mid) if np.isfinite(mid) else abs(natural)
    limit_per_spread = round(limit / MULTIPLIER, 2)
    natural_per_spread = round(abs(natural) / MULTIPLIER, 2)
    mid_per_spread = round(abs(mid) / MULTIPLIER, 2)

    legs, api_legs = [], []
    for lg in structure.legs:
        opening = True
        instr = ("BUY_TO_OPEN" if lg.side is Side.BUY else "SELL_TO_OPEN") if opening else \
                ("BUY_TO_CLOSE" if lg.side is Side.BUY else "SELL_TO_CLOSE")
        sym = occ_symbol(structure.underlying, lg.expiry, lg.right, lg.strike)
        desc = (f"{structure.underlying} {lg.expiry:%d %b %y} "
                f"{lg.strike:g} {'Call' if lg.right is Right.CALL else 'Put'}")
        legs.append({"instruction": instr, "quantity": lg.quantity * contracts,
                     "description": desc, "symbol": sym})
        api_legs.append({"instruction": instr, "quantity": lg.quantity * contracts,
                         "instrument": {"symbol": sym, "assetType": "OPTION"}})

    strat = {2: "VERTICAL", 3: "BUTTERFLY", 4: "IRON_CONDOR"}.get(len(structure.legs), "CUSTOM")
    if "calendar" in structure.name:
        strat = "CALENDAR"

    api_json = {
        "orderType": order_type,
        "session": "NORMAL",
        "price": f"{limit_per_spread:.2f}",
        "duration": "DAY" if duration.upper() == "DAY" else "GOOD_TILL_CANCEL",
        # A multi-leg spread is still ONE order strategy.  orderStrategyType is
        # SINGLE; the multi-leg nature lives in orderLegCollection plus
        # complexOrderStrategyType.  "MULTI_LEG" is not a valid enum value and
        # is the most commonly copied wrong example on the internet.
        "orderStrategyType": "SINGLE",
        "complexOrderStrategyType": strat,
        "orderLegCollection": api_legs,
    }

    credit = abs(natural) * contracts if not is_debit else 0.0
    cost = costs.estimate(structure, contracts, credit_received=credit)

    exp = structure.legs[0].expiry
    web_steps = [
        "1. Trade -> Options.",
        f"2. Enter the symbol {structure.underlying}.",
        f"3. Strategy dropdown -> select the multi-leg strategy "
        f"({structure.name}).",
        "4. Click the chains (link) icon to open the option chain popup; "
        f"select expiration {exp:%d %b %Y}.",
        "5. Select the strikes by clicking in the BID/ASK area. Calls sit to "
        "the LEFT of the strike column, puts to the RIGHT.",
        "   Legs: " + "; ".join(f"{l['instruction'].replace('_',' ').title()} "
                                 f"{l['quantity']}x {l['description']}" for l in legs),
        f"6. Set Quantity = {contracts}, Order type = "
        f"{'Net debit' if is_debit else 'Net credit'}, "
        f"Limit = {limit_per_spread:.2f}, Timing = {duration}.",
        "   (Consider order type 'Walk limit - "
        f"{'debit' if is_debit else 'credit'}' instead: set start "
        f"{mid_per_spread:.2f}, end {natural_per_spread:.2f}, increment 0.01, "
        "interval 5s. Schwab walks the price for you.)",
        "7. Click REVIEW ORDER. Verify the Estimated Amount and every leg.",
        "8. Click PLACE ORDER.",
    ]
    tos_steps = [
        "1. Trade tab -> All Products sub-tab.",
        f"2. Type {structure.underlying}, press Enter.",
        f"3. Expand the {exp:%d %b %y} expiration in the option chain.",
        f"4. RIGHT-CLICK the price of the leg you are BUYING -> choose "
        f"'{'Vertical' if strat == 'VERTICAL' else strat.title().replace('_', ' ')}'. "
        "The offsetting leg auto-populates.",
        "   (For non-standard legs: click the ASK of the buy leg, then "
        "CTRL+CLICK the BID of the sell leg.)",
        "5. Click the second strike to widen/narrow to the intended strikes.",
        f"6. Double-click the price field and type the net "
        f"{'debit' if is_debit else 'credit'}: {limit_per_spread:.2f}.",
        f"7. Set quantity {contracts}. Click 'DAY' to toggle DAY/GTC -> {duration}.",
        "8. Click CONFIRM AND SEND, review, then SEND.",
    ]

    # "Close at X% of MAX PROFIT" is not the same price for a debit as for a
    # credit, and the two must not share a formula.
    #
    #   credit: max profit IS the credit, so the target is buying it back at
    #           (1 - X) of the credit.
    #   debit:  max profit is (width - debit), NOT the debit. Using (1 - X) of
    #           the debit would mean SELLING at half what you paid -- a 50%
    #           loss dressed up as a 50% profit target.
    max_gain_per_spread = structure.max_gain(natural) / MULTIPLIER
    max_loss_per_spread = structure.max_loss(natural) / MULTIPLIER
    if is_debit:
        tgt = round(limit_per_spread + profit_target_pct * max_gain_per_spread, 2)
    else:
        tgt = round(limit_per_spread * (1.0 - profit_target_pct), 2)
    # The stop cannot exceed the contractual max loss, or it never fires and the
    # "stop" is decoration.
    stop_adverse = round(min(limit_per_spread * stop_multiple, max_loss_per_spread), 2)
    exit_plan = [
        f"Profit target: close at {profit_target_pct:.0%} of max profit "
        f"(${max_gain_per_spread:.2f}/spread) -> "
        f"{'sell to close' if is_debit else 'buy to close'} at ~${tgt:.2f} net.",
        f"Stop: close if the structure's loss reaches {stop_multiple:.1f}x the "
        f"{'debit' if is_debit else 'credit'} (${stop_adverse:.2f} adverse, capped at "
        f"the ${max_loss_per_spread:.2f} contractual max loss).",
        f"Time stop: flatten at {manage_dte} DTE regardless of P&L. Gamma and "
        "assignment risk both accelerate inside three weeks and neither is in "
        "the edge that justified the trade.",
        "Attach the exit via Trade -> All-In-One Trade Ticket -> 'Add "
        "Conditional' -> One Cancels Other (OCO). NOTE: whether a multi-leg "
        "spread is an eligible OCO leg in the WEB/DESKTOP GUI is not "
        "documented by Schwab -- it IS supported via the Trader API. Verify "
        "before relying on it.",
    ]

    warnings = [
        "Both legs auto-exercise/assign at $0.01 ITM (OCC exercise-by-exception). "
        "A vertical whose short leg finishes $0.01 ITM and long leg OTM becomes "
        "a stock position overnight, settled before you can react.",
        "Schwab does NOT publish its own internal cutoff for contrary-exercise "
        "(DNE) instructions -- only the 5:30pm ET industry deadline, with a note "
        "that firms set earlier ones. Do not plan around DNE. Plan around "
        "CLOSING the position during regular hours on expiration day.",
        "Level 2 (Spread Trading) approval and a MARGIN account are required: "
        "'Securities regulations require that options spreads occur in a margin "
        "account.' In an IRA this is limited margin and must be applied for "
        "separately.",
    ]
    if contracts * sum(l.quantity for l in structure.legs) >= 20:
        warnings.append(f"{contracts * sum(l.quantity for l in structure.legs)} total "
                        f"contracts -> ${cost['commission']:.2f} commission on the open alone.")

    return OrderTicket(
        underlying=structure.underlying, structure_name=structure.name, contracts=contracts,
        order_type=order_type, limit_price=limit_per_spread, duration=duration,
        legs=legs, complex_strategy=strat, cost_estimate=cost,
        price_ladder=_limit_ladder(mid_per_spread, natural_per_spread),
        natural=natural_per_spread, mid=mid_per_spread, api_json=api_json,
        web_steps=web_steps, tos_steps=tos_steps, exit_plan=exit_plan, warnings=warnings,
    )


OPEN_QUESTIONS = [
    ("Schwab's own internal cutoff time for contrary-exercise / DNE instructions",
     "HIGHEST PRIORITY. Schwab publishes only the 5:30pm ET industry deadline and "
     "notes firms set earlier ones. Call the options desk (888-245-6864), get the "
     "actual time, and hard-code it with margin."),
    ("Iron condor's placement at approval Level 2",
     "Schwab's application form lists 'condors, butterflies' at Level 2 but never "
     "the words 'iron condor'. Structurally it is two credit verticals and the API "
     "exposes IRON_CONDOR as a complexOrderStrategyType. Confirm before relying on it."),
    ("Schwab's per-contract ORF and proprietary index-option fee pass-through",
     "Not published. Schwab bundles them into a discretionary 'Industry Fee' that "
     "'may differ from or exceed the actual fees properly paid by Schwab'. The cost "
     "model treats these as an estimate; reconcile against real confirms after 10 fills."),
    ("Whether a multi-leg spread can be a leg of an OCO/conditional order in the GUI",
     "Confirmed supported via the Trader API (orderStrategyType OCO / TRIGGER with "
     "childOrderStrategies). Undocumented for the web and thinkorswim GUIs."),
    ("Schwab's numeric trigger for liquidating short options in an undercapitalised "
     "account near expiry",
     "Not published. The account agreement grants discretionary liquidation 'without "
     "prior demand or notice'. Assume no warning."),
]
