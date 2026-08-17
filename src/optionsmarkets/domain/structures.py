"""Option contracts, defined-risk structures, and their aggregate risk.

The system is restricted to structures with a *provable* maximum loss.  That
is a modelling decision as much as a risk one: Kelly sizing requires a bounded
``min x`` in the scenario set, and a naked short has none.  With a defined-risk
structure the capital at risk is a contractual fact, not an estimate, and the
whole sizing chain rests on solid ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np

from ..pricing import black

__all__ = ["Right", "Side", "Leg", "Structure", "OptionQuote", "occ_symbol",
           "vertical", "iron_condor", "butterfly", "calendar"]

MULTIPLIER = 100


class Right(str, Enum):
    CALL = "C"
    PUT = "P"

    @property
    def cp(self) -> int:
        return 1 if self is Right.CALL else -1


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1


@dataclass(frozen=True)
class OptionQuote:
    """One contract's market state.  ``mid`` is spread-midpoint; the system
    never assumes it can trade there -- see :mod:`optionsmarkets.execution`."""
    strike: float
    right: Right
    expiry: date
    bid: float
    ask: float
    last: float = np.nan
    volume: float = 0.0
    open_interest: float = 0.0
    iv_vendor: float = np.nan          # vendor IV: diagnostic only, never consumed
    asof: str = ""

    @property
    def mid(self) -> float:
        if not (np.isfinite(self.bid) and np.isfinite(self.ask)) or self.ask <= 0:
            return np.nan
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return float(self.ask - self.bid) if np.isfinite(self.ask) and np.isfinite(self.bid) else np.nan

    @property
    def rel_spread(self) -> float:
        m = self.mid
        return float(self.spread / m) if np.isfinite(m) and m > 0 else np.inf

    def tradeable(self, max_rel_spread: float = 0.15, min_oi: float = 25) -> bool:
        return (
            np.isfinite(self.bid) and self.bid > 0.0
            and np.isfinite(self.ask) and self.ask > self.bid
            and self.rel_spread <= max_rel_spread
            and self.open_interest >= min_oi
        )


@dataclass(frozen=True)
class Leg:
    side: Side
    right: Right
    strike: float
    expiry: date
    quantity: int = 1
    quote: OptionQuote | None = None

    @property
    def signed_qty(self) -> int:
        return self.side.sign * self.quantity

    def payoff(self, S_T):
        """Intrinsic payoff per SHARE at expiry (multiply by 100 for dollars)."""
        S_T = np.asarray(S_T, float)
        intr = np.maximum(self.right.cp * (S_T - self.strike), 0.0)
        return self.signed_qty * intr


def occ_symbol(underlying: str, expiry: date, right: Right, strike: float) -> str:
    """OCC 21-character option symbol as Schwab's Trader API expects it:
    6-char underlying RIGHT-PADDED WITH SPACES + YYMMDD + C/P + 8-digit strike
    (5 whole + 3 decimal).  e.g. ``'SPY   260918P00780000'``.

    The dotted thinkorswim display form (``.SPY260918P780``) is NOT valid on
    the API and is a common source of rejected orders.
    """
    return (f"{underlying.upper():<6}{expiry:%y%m%d}{right.value}"
            f"{int(round(strike * 1000)):08d}")


@dataclass
class Structure:
    """A multi-leg defined-risk position."""
    name: str
    underlying: str
    legs: list[Leg]

    # ---- economics ------------------------------------------------------
    def net_price(self, use: str = "mid") -> float:
        """Net debit (+) or credit (-) PER SPREAD in dollars.

        ``use='marketable'`` prices each leg at the side of the book you would
        actually have to cross -- buys at the ask, sells at the bid.  That is
        the honest worst case, and the difference between it and mid is the
        single largest cost in a retail multi-leg options strategy.  Any
        expected-value calculation that quietly uses mid is overstating edge.
        """
        tot = 0.0
        for lg in self.legs:
            if lg.quote is None:
                return np.nan
            if use == "mid":
                px = lg.quote.mid
            elif use == "marketable":
                px = lg.quote.ask if lg.side is Side.BUY else lg.quote.bid
            else:  # 'passive' -- the price you'd post and hope to get
                px = lg.quote.bid if lg.side is Side.BUY else lg.quote.ask
            if not np.isfinite(px):
                return np.nan
            tot += lg.signed_qty * px * MULTIPLIER
        return float(tot)

    def payoff(self, S_T, net_price: float | None = None):
        """Dollar P&L per spread at expiry across underlying prices."""
        S_T = np.asarray(S_T, float)
        gross = sum(lg.payoff(S_T) for lg in self.legs) * MULTIPLIER
        cost = self.net_price("marketable") if net_price is None else net_price
        return gross - cost

    def breakevens(self, net_price: float | None = None, lo=None, hi=None, n=200001):
        ks = [lg.strike for lg in self.legs]
        lo = 0.01 if lo is None else lo
        hi = (max(ks) * 3.0) if hi is None else hi
        grid = np.linspace(lo, hi, n)
        pl = self.payoff(grid, net_price)
        sign = np.sign(pl)
        cross = np.where(np.diff(sign) != 0)[0]
        return [float(grid[i] + (grid[i + 1] - grid[i]) * abs(pl[i]) / (abs(pl[i]) + abs(pl[i + 1])))
                for i in cross]

    def max_loss(self, net_price: float | None = None) -> float:
        """Worst dollar loss per spread.  Evaluated on the exact kink set --
        the payoff is piecewise linear, so its extrema are at the strikes, at
        zero, and at infinity.  Sampling a grid can miss the true worst point."""
        ks = sorted({lg.strike for lg in self.legs})
        pts = np.array([0.0] + ks + [max(ks) * 5.0])
        pl = self.payoff(pts, net_price)
        return float(-np.min(pl))

    def max_gain(self, net_price: float | None = None) -> float:
        ks = sorted({lg.strike for lg in self.legs})
        pts = np.array([0.0] + ks + [max(ks) * 5.0])
        return float(np.max(self.payoff(pts, net_price)))

    def is_defined_risk(self, net_price: float | None = None) -> bool:
        ks = sorted({lg.strike for lg in self.legs})
        far = self.payoff(np.array([max(ks) * 50.0]), net_price)[0]
        near = self.payoff(np.array([1e-8]), net_price)[0]
        return bool(np.isfinite(far) and np.isfinite(near)
                    and far > -1e12 and near > -1e12
                    and abs(sum(lg.signed_qty for lg in self.legs if lg.right is Right.CALL)) == 0
                    and abs(sum(lg.signed_qty for lg in self.legs if lg.right is Right.PUT)) == 0)

    # ---- risk -----------------------------------------------------------
    def greeks(self, S, r, q, sigma_by_leg, t_by_leg) -> dict:
        """Position Greeks, in trader units, summed across legs.

        ``sigma_by_leg`` should come from the fitted surface at each leg's own
        (k, T) -- NOT a single flat vol.  A vertical priced with one vol has
        zero vega by construction, which is exactly wrong: the whole point of
        a spread is the differential exposure across the smile.
        """
        agg = {k: 0.0 for k in ("delta", "gamma", "vega", "theta", "rho",
                                "vanna", "volga", "charm")}
        for lg, sg, T in zip(self.legs, sigma_by_leg, t_by_leg, strict=True):
            g = black.scale_for_trader(black.greeks(S, lg.strike, T, r, q, sg, lg.right.cp))
            for k in agg:
                agg[k] += lg.signed_qty * float(np.ravel(g[k])[0]) * MULTIPLIER
        return agg


# ----------------------------------------------------------------------------
# constructors
# ----------------------------------------------------------------------------

def vertical(underlying, expiry, right: Right, long_strike: float, short_strike: float,
             q_long=None, q_short=None, qty: int = 1) -> Structure:
    kind = "debit" if ((right is Right.CALL and long_strike < short_strike) or
                       (right is Right.PUT and long_strike > short_strike)) else "credit"
    return Structure(
        f"{right.name.lower()} vertical ({kind})", underlying,
        [Leg(Side.BUY, right, long_strike, expiry, qty, q_long),
         Leg(Side.SELL, right, short_strike, expiry, qty, q_short)],
    )


def iron_condor(underlying, expiry, put_long, put_short, call_short, call_long,
                quotes: dict | None = None, qty: int = 1) -> Structure:
    quotes = quotes or {}
    return Structure(
        "iron condor", underlying,
        [Leg(Side.BUY, Right.PUT, put_long, expiry, qty, quotes.get(("P", put_long))),
         Leg(Side.SELL, Right.PUT, put_short, expiry, qty, quotes.get(("P", put_short))),
         Leg(Side.SELL, Right.CALL, call_short, expiry, qty, quotes.get(("C", call_short))),
         Leg(Side.BUY, Right.CALL, call_long, expiry, qty, quotes.get(("C", call_long)))],
    )


def butterfly(underlying, expiry, right: Right, lower, body, upper,
              quotes: dict | None = None, qty: int = 1) -> Structure:
    quotes = quotes or {}
    r = right.value
    return Structure(
        f"{right.name.lower()} butterfly", underlying,
        [Leg(Side.BUY, right, lower, expiry, qty, quotes.get((r, lower))),
         Leg(Side.SELL, right, body, expiry, 2 * qty, quotes.get((r, body))),
         Leg(Side.BUY, right, upper, expiry, qty, quotes.get((r, upper)))],
    )


def calendar(underlying, near_expiry, far_expiry, right: Right, strike,
             q_near=None, q_far=None, qty: int = 1) -> Structure:
    return Structure(
        f"{right.name.lower()} calendar", underlying,
        [Leg(Side.SELL, right, strike, near_expiry, qty, q_near),
         Leg(Side.BUY, right, strike, far_expiry, qty, q_far)],
    )
