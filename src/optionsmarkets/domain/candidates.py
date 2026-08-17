"""Candidate structure enumeration.

The policy can only accept what the enumerator proposes, so this module is a
silent constraint on the entire system: a vertical-only enumerator produces a
vertical-only strategy no matter what the surface says.  It generates, from a
real chain and a fitted surface:

    verticals        credit and debit, both rights, by SHORT-LEG DELTA and width
    iron condors     two credit verticals, symmetric in delta
    butterflies      long the wings, short the body
    calendars        same strike, two expiries
    diagonals        different strike AND expiry

Three rules are enforced here rather than downstream, because a candidate that
violates them is not a trade the policy should have to reason about:

**Strikes are selected by delta, not by percentage moneyness.** A 5% OTM put is
a different trade at 12 vol than at 40 vol; a 20-delta put is the same trade.
Delta is computed off the *fitted surface* volatility at each strike's own
log-moneyness, not off a flat ATM vol -- using one vol across strikes puts the
"20-delta" strike in the wrong place by several strikes on a skewed tape, and
does it worst in the wings where the skew is steepest.

**Every leg must have a real, two-sided quote.** No leg is ever priced off the
model.  If the strike is not quotable the structure is not enumerated -- an
un-executable candidate that scores well is worse than no candidate, because it
occupies the top of the ranking and the system reports "no trade" while looking
at something it was never going to be able to fill.

**Direction follows the variance risk premium.** VRP > 0 enumerates credit
(short-premium) structures; VRP < 0 enumerates debit (long-premium) ones.  The
enumerator does not propose both sides and let the scorer pick, because the
scorer would then be choosing direction on an estimate whose sign the policy has
already decided from a more reliable signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from ..pricing import black
from .structures import (
    Leg, OptionQuote, Right, Side, Structure, butterfly, calendar, iron_condor, vertical,
)

__all__ = [
    "ChainIndex", "CandidateConfig", "strike_at_delta", "enumerate_candidates",
    "diagonal",
]


# ----------------------------------------------------------------------------
# chain indexing
# ----------------------------------------------------------------------------

@dataclass
class ChainIndex:
    """Quotes indexed by (expiry, right, strike), with the liquidity screen baked in.

    ``get`` returns None for anything not quotable, so a structure builder that
    forgets to check simply produces a leg with no quote, and the net-price
    computation returns NaN rather than a plausible number.  Failing loudly at
    the cheapest point is the whole design.
    """
    quotes: dict[tuple[date, Right, float], OptionQuote] = field(default_factory=dict)
    max_rel_spread: float = 0.25
    min_open_interest: float = 25.0

    @classmethod
    def from_quotes(cls, quotes, **kw) -> "ChainIndex":
        idx = cls(**kw)
        for q in quotes:
            idx.quotes[(q.expiry, q.right, float(q.strike))] = q
        return idx

    def get(self, expiry: date, right: Right, strike: float) -> OptionQuote | None:
        q = self.quotes.get((expiry, right, float(strike)))
        if q is None:
            return None
        if not q.tradeable(self.max_rel_spread, self.min_open_interest):
            return None
        return q

    def strikes(self, expiry: date, right: Right, tradeable_only: bool = True) -> list[float]:
        out = [K for (e, r, K) in self.quotes if e == expiry and r == right
               and (not tradeable_only or self.get(e, r, K) is not None)]
        return sorted(out)

    def expiries(self) -> list[date]:
        return sorted({e for (e, _, _) in self.quotes})

    def strike_step(self, expiry: date, right: Right) -> float:
        """Modal gap between listed strikes -- the grid the market actually quotes."""
        ks = self.strikes(expiry, right, tradeable_only=False)
        if len(ks) < 3:
            return 5.0
        d = np.diff(ks)
        d = d[d > 0]
        if d.size == 0:
            return 5.0
        vals, counts = np.unique(np.round(d, 4), return_counts=True)
        return float(vals[int(np.argmax(counts))])

    def nearest_strike(self, expiry: date, right: Right, target: float,
                       tradeable_only: bool = True) -> float | None:
        ks = self.strikes(expiry, right, tradeable_only)
        if not ks:
            return None
        return float(min(ks, key=lambda K: abs(K - target)))


# ----------------------------------------------------------------------------
# strike selection
# ----------------------------------------------------------------------------

def strike_at_delta(surface, T: float, F: float, S: float, r: float, q: float,
                    right: Right, target_delta: float,
                    k_lo: float = -0.7, k_hi: float = 0.5, n: int = 4001) -> float:
    """Strike whose |delta| is closest to ``target_delta``, on the fitted surface.

    Each candidate strike is priced at its OWN surface volatility.  Solving with
    a single ATM vol is the standard shortcut and it is wrong in a way that
    matters: on a -0.8-rho equity slice the true 20-delta put sits materially
    further out than the flat-vol calculation says, so a "20-delta" spread built
    that way is systematically closer to the money -- more premium, more risk,
    and a delta the risk gates were never told about.
    """
    kk = np.linspace(k_lo, k_hi, int(n))
    K = F * np.exp(kk)
    sig = np.asarray(surface.vol(kk, T), float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.nan)
    d = np.abs(black.greeks(S, K, T, r, q, sig, right.cp).delta)
    d = np.where(np.isfinite(d), d, np.inf)
    return float(K[int(np.argmin(np.abs(d - float(target_delta))))])


def diagonal(underlying: str, near_expiry: date, far_expiry: date, right: Right,
             short_strike: float, long_strike: float,
             q_short: OptionQuote | None = None, q_long: OptionQuote | None = None,
             qty: int = 1) -> Structure:
    """Sell the near-dated strike, buy the far-dated one at a different strike.

    A diagonal is a calendar with a directional tilt.  Its defined-risk status
    is NOT contractual in general: if the short strike is closer to the money
    than the long one, the position can lose more than the debit paid once the
    near leg expires.  ``Structure.is_defined_risk`` evaluates the payoff at
    expiry of the *first* leg listed and will not catch that, so the policy's
    ``risk.defined`` gate is checked against the near-expiry payoff and the
    enumerator only emits diagonals whose long strike is at least as far OTM as
    the short -- the configuration whose worst case is bounded.
    """
    return Structure(
        f"{right.name.lower()} diagonal", underlying,
        [Leg(Side.SELL, right, short_strike, near_expiry, qty, q_short),
         Leg(Side.BUY, right, long_strike, far_expiry, qty, q_long)],
    )


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------

@dataclass
class CandidateConfig:
    """What the enumerator is allowed to propose.

    Deliberately small.  Every extra family multiplies the number of structures
    scored per decision, and each one is another opportunity for the selector to
    find a candidate that looks good because of a data artefact rather than
    because of the surface.  Add a family when you can say what edge it captures
    that the existing ones do not.
    """
    short_deltas: tuple[float, ...] = (0.16, 0.22, 0.30, 0.38)
    widths_in_strikes: tuple[int, ...] = (1, 2, 3)
    enable_verticals: bool = True
    enable_iron_condors: bool = True
    enable_butterflies: bool = False
    enable_calendars: bool = True
    enable_diagonals: bool = False
    # Iron condors are enumerated at matched put/call short deltas.  Skewed
    # condors (a further call short than put short) are the same trade with a
    # directional overlay bolted on, and this system does not take direction.
    condor_deltas: tuple[float, ...] = (0.16, 0.22, 0.30)
    condor_widths_in_strikes: tuple[int, ...] = (1, 2)
    calendar_deltas: tuple[float, ...] = (0.50, 0.30)
    max_candidates: int = 60


# ----------------------------------------------------------------------------
# enumeration
# ----------------------------------------------------------------------------

def enumerate_candidates(
    underlying: str,
    index: ChainIndex,
    surface,
    *,
    expiry: date,
    T: float,
    F: float,
    S: float,
    r: float,
    q: float,
    vrp_vol_points: float,
    config: CandidateConfig | None = None,
    far_expiry: date | None = None,
    T_far: float | None = None,
    F_far: float | None = None,
) -> list[Structure]:
    """Every executable defined-risk structure worth scoring on this slice.

    ``vrp_vol_points`` sets the direction: positive means implied is rich and
    the enumerator proposes credit structures; negative means it is cheap and
    the enumerator proposes debit structures.  Calendars are enumerated in both
    regimes because their edge is the *term structure* of the premium rather
    than its level, and that sign is separate from the front-slice VRP.
    """
    cfg = config or CandidateConfig()
    sell_premium = vrp_vol_points >= 0.0
    out: list[Structure] = []
    step = index.strike_step(expiry, Right.PUT)

    def _q(exp, right, K):
        return index.get(exp, right, K)

    # ---- verticals -------------------------------------------------------
    if cfg.enable_verticals:
        for right in (Right.PUT, Right.CALL):
            for td in cfg.short_deltas:
                raw = strike_at_delta(surface, T, F, S, r, q, right, td)
                K_short = index.nearest_strike(expiry, right, raw)
                if K_short is None:
                    continue
                for nw in cfg.widths_in_strikes:
                    width = nw * step
                    # The long leg is always further OTM than the short one, so
                    # the structure is a credit spread whose max loss is
                    # (width - credit): contractual, which is what Kelly needs.
                    K_long = K_short - width if right is Right.PUT else K_short + width
                    qs, ql = _q(expiry, right, K_short), _q(expiry, right, K_long)
                    if qs is None or ql is None:
                        continue
                    if sell_premium:
                        st = vertical(underlying, expiry, right, K_long, K_short,
                                      q_long=ql, q_short=qs)
                        kind = "credit"
                        long_k, short_k = K_long, K_short
                    else:
                        # Debit: buy the nearer-the-money strike, sell the further.
                        st = vertical(underlying, expiry, right, K_short, K_long,
                                      q_long=qs, q_short=ql)
                        kind = "debit"
                        long_k, short_k = K_short, K_long
                    st.name = (f"{min(long_k, short_k):.0f}/{max(long_k, short_k):.0f} "
                               f"{right.name.lower()} {kind} vert ({td:.0%}d)")
                    out.append(st)

    # ---- iron condors ----------------------------------------------------
    if cfg.enable_iron_condors and sell_premium:
        for td in cfg.condor_deltas:
            Kp_raw = strike_at_delta(surface, T, F, S, r, q, Right.PUT, td)
            Kc_raw = strike_at_delta(surface, T, F, S, r, q, Right.CALL, td)
            Kp_s = index.nearest_strike(expiry, Right.PUT, Kp_raw)
            Kc_s = index.nearest_strike(expiry, Right.CALL, Kc_raw)
            if Kp_s is None or Kc_s is None or Kc_s <= Kp_s:
                continue
            for nw in cfg.condor_widths_in_strikes:
                width = nw * step
                Kp_l, Kc_l = Kp_s - width, Kc_s + width
                legs = {("P", Kp_l): _q(expiry, Right.PUT, Kp_l),
                        ("P", Kp_s): _q(expiry, Right.PUT, Kp_s),
                        ("C", Kc_s): _q(expiry, Right.CALL, Kc_s),
                        ("C", Kc_l): _q(expiry, Right.CALL, Kc_l)}
                if any(v is None for v in legs.values()):
                    continue
                st = iron_condor(underlying, expiry, Kp_l, Kp_s, Kc_s, Kc_l, legs)
                st.name = (f"{Kp_l:.0f}/{Kp_s:.0f}/{Kc_s:.0f}/{Kc_l:.0f} "
                           f"iron condor ({td:.0%}d)")
                out.append(st)

    # ---- butterflies -----------------------------------------------------
    if cfg.enable_butterflies:
        right = Right.PUT if sell_premium else Right.CALL
        body_raw = strike_at_delta(surface, T, F, S, r, q, right, 0.50)
        body = index.nearest_strike(expiry, right, body_raw)
        if body is not None:
            for nw in cfg.widths_in_strikes:
                width = nw * step
                lo, hi = body - width, body + width
                qq = {(right.value, lo): _q(expiry, right, lo),
                      (right.value, body): _q(expiry, right, body),
                      (right.value, hi): _q(expiry, right, hi)}
                if any(v is None for v in qq.values()):
                    continue
                st = butterfly(underlying, expiry, right, lo, body, hi, qq)
                st.name = f"{lo:.0f}/{body:.0f}/{hi:.0f} {right.name.lower()} fly"
                out.append(st)

    # ---- calendars and diagonals ----------------------------------------
    if far_expiry is not None and far_expiry != expiry and T_far:
        F_far = float(F_far) if F_far else F
        if cfg.enable_calendars:
            for td in cfg.calendar_deltas:
                raw = strike_at_delta(surface, T, F, S, r, q, Right.PUT, td)
                K = index.nearest_strike(expiry, Right.PUT, raw)
                if K is None:
                    continue
                qn, qf = _q(expiry, Right.PUT, K), _q(far_expiry, Right.PUT, K)
                if qn is None or qf is None:
                    continue
                st = calendar(underlying, expiry, far_expiry, Right.PUT, K,
                              q_near=qn, q_far=qf)
                st.name = (f"{K:.0f} put calendar "
                           f"{expiry:%d%b}/{far_expiry:%d%b} ({td:.0%}d)")
                out.append(st)

        if cfg.enable_diagonals:
            for td in cfg.calendar_deltas:
                raw_s = strike_at_delta(surface, T, F, S, r, q, Right.PUT, td)
                K_s = index.nearest_strike(expiry, Right.PUT, raw_s)
                if K_s is None:
                    continue
                # Long strike at least as far OTM as the short: bounds the loss
                # once the near leg expires.  See :func:`diagonal`.
                K_l = index.nearest_strike(far_expiry, Right.PUT, K_s - step)
                if K_l is None or K_l > K_s:
                    continue
                qs, ql = _q(expiry, Right.PUT, K_s), _q(far_expiry, Right.PUT, K_l)
                if qs is None or ql is None:
                    continue
                st = diagonal(underlying, expiry, far_expiry, Right.PUT, K_s, K_l,
                              q_short=qs, q_long=ql)
                st.name = (f"{K_s:.0f}/{K_l:.0f} put diagonal "
                           f"{expiry:%d%b}/{far_expiry:%d%b}")
                out.append(st)

    # Drop anything whose net price is not computable from real quotes, and
    # anything whose max loss is not bounded -- Kelly has no answer for an
    # unbounded left tail and the sizing chain must never be handed one.
    clean: list[Structure] = []
    for st in out:
        net = st.net_price("marketable")
        if not np.isfinite(net):
            continue
        if not np.isfinite(st.max_loss(net)) or st.max_loss(net) <= 0:
            continue
        clean.append(st)
    return clean[: cfg.max_candidates]
