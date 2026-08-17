"""Market-data abstraction and the honest description of the feed we have.

WHAT IS ACTUALLY AVAILABLE ON THIS ACCOUNT (probed 2026-08-17)
-------------------------------------------------------------
Massive Market Data MCP -- entitled:
    /v2/aggs/ticker/{t}/range/...   daily & intraday OHLCV bars    [VERIFIED]
    /v2/aggs/ticker/{t}/prev        previous close                 [VERIFIED]
    /fed/v1/treasury-yields         risk-free curve, 1M-30Y
    /stocks/v1/dividends            declaration/ex/record/pay + amounts
    /tmx/v1/corporate-events        earnings dates (Wall Street Horizon)
    /v2/reference/news              news + sentiment
    /stocks/v1/short-interest       FINRA short interest / short volume
    /v1/marketstatus/now|upcoming   session state, holidays
    server-side functions           bs_price, bs_delta/gamma/theta/vega/rho,
                                    vanna, volga, charm, color, veta, ema,
                                    sharpe_ratio, sortino_ratio
    query_data                      SQL (incl. FTS5) over stored responses

Massive Market Data MCP -- NOT entitled:
    /v3/snapshot/options/*          403 NOT_AUTHORIZED
    /v3/quotes/{optionsTicker}      403
    /v3/trades/{optionsTicker}      403
    -> the entire options plane requires a plan upgrade.

yfinance MCP -- entitled:
    option chains (strike, bid, ask, last, volume, OI, IV, ITM flag)
    price history, ticker info, news, screeners

THE BINDING CONSTRAINT
----------------------
The options feed is yfinance: roughly 15 minutes delayed, single snapshot (no
NBBO timestamp, no exchange, no trade tape), and its ``impliedVolatility``
field emits a **0.500005 sentinel** on zero-bid strikes -- observed directly on
SPY 2026-09-18 puts at the 300/305/310 strikes, all reporting exactly
0.500005 with bid=ask=0.

Three consequences, all designed for rather than papered over:

  1. **Vendor IV is never consumed.** It is stored as ``iv_vendor`` for
     diagnostics only. Every volatility in this system is inverted in-house
     from the mid price by :mod:`optionsmarkets.pricing.implied`, which
     rejects unquotable strikes explicitly instead of imputing them.
  2. **Latency is a gate, not a nuisance.** The freshness gate in the decision
     policy is set to the feed's real latency. With a 15-minute-old chain the
     system must not trade structures whose edge decays faster than that --
     which rules out 0DTE and most gamma-scalping outright. It does not rule
     out 21-45 DTE premium selling, where the edge is a multi-day variance
     premium and 15 minutes is immaterial.
  3. **The forward comes from the option market, not from the feed.** Put-call
     parity regression on the liquid core gives F and DF; a stale spot would
     otherwise tilt the whole slice and be misread as skew.

The Protocol below is the seam. Swapping in a real options feed -- a Massive
plan upgrade, Schwab's own market-data API, Polygon, Databento -- means writing
one adapter and changing one line of config. Nothing above the data layer knows
where the quotes came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ..domain.structures import OptionQuote

__all__ = ["ChainSnapshot", "MarketDataProvider", "FeedQuality", "assess_quality"]


@dataclass
class ChainSnapshot:
    underlying: str
    spot: float
    asof: datetime
    expiries: list[date]
    quotes: dict[date, list[OptionQuote]]
    source: str
    latency_s: float

    def for_expiry(self, exp: date) -> list[OptionQuote]:
        return self.quotes.get(exp, [])

    def age_s(self, now: datetime | None = None) -> float:
        now = now or datetime.now(self.asof.tzinfo)
        return max((now - self.asof).total_seconds(), self.latency_s)


@runtime_checkable
class MarketDataProvider(Protocol):
    """Every implementation must be able to answer these, or raise clearly."""

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot: ...
    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    def risk_free_curve(self, asof: date | None = None) -> dict[float, float]: ...
    def dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame: ...
    def next_earnings(self, symbol: str) -> date | None: ...
    def market_open(self) -> bool: ...


# ----------------------------------------------------------------------------
# quality assessment -- runs on EVERY snapshot, before anything else touches it
# ----------------------------------------------------------------------------

VENDOR_IV_SENTINELS = (0.500005, 0.5, 1e-5, 0.0)


@dataclass
class FeedQuality:
    n_quotes: int
    n_two_sided: int
    n_zero_bid: int
    n_sentinel_iv: int
    median_rel_spread: float
    max_age_s: float
    crossed_or_locked: int
    verdict: str
    detail: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verdict == "OK"


def assess_quality(quotes: list[OptionQuote], max_age_s: float,
                   min_two_sided: int = 12) -> FeedQuality:
    """Screen a chain slice before it is allowed near the surface fitter.

    Bad data does not produce an obviously bad surface -- it produces a
    plausible one with a phantom smile, which is far more dangerous. These
    checks are cheap and they are the difference between a model that admits
    it cannot see and one that confidently prices noise.
    """
    n = len(quotes)
    bids = np.array([q.bid for q in quotes], float)
    asks = np.array([q.ask for q in quotes], float)
    ivv = np.array([q.iv_vendor for q in quotes], float)

    two_sided = int(np.sum((bids > 0) & (asks > bids)))
    zero_bid = int(np.sum(~(bids > 0)))
    crossed = int(np.sum(asks < bids))
    sentinel = int(np.sum([np.any(np.isclose(v, VENDOR_IV_SENTINELS, atol=1e-9))
                           for v in ivv if np.isfinite(v)]))
    rel = np.array([q.rel_spread for q in quotes], float)
    rel = rel[np.isfinite(rel)]
    med_rel = float(np.median(rel)) if rel.size else np.inf

    detail, verdict = [], "OK"
    if two_sided < min_two_sided:
        verdict = "TOO_FEW_QUOTES"
        detail.append(f"only {two_sided} two-sided quotes (need {min_two_sided}) -- "
                      "cannot identify a forward or fit a slice")
    if crossed > 0:
        verdict = "CROSSED_BOOK"
        detail.append(f"{crossed} crossed/locked quotes -- snapshot is inconsistent")
    if med_rel > 0.35:
        verdict = "ILLIQUID"
        detail.append(f"median relative spread {med_rel:.1%} -- mid prices are fiction here")
    if max_age_s > 3600:
        verdict = "STALE"
        detail.append(f"snapshot {max_age_s / 60:.0f} min old")
    if sentinel:
        detail.append(f"{sentinel} vendor IV values are sentinels (0.500005 etc) -- "
                      "ignored by design; IV is inverted in-house")
    return FeedQuality(n, two_sided, zero_bid, sentinel, med_rel, max_age_s,
                       crossed, verdict, detail)
