"""Adapters that map the live MCP servers onto :class:`MarketDataProvider`.

The MCP tools are invoked by the *agent* (Claude), not by this process -- there
is no HTTP client here on purpose.  Each adapter therefore takes a ``call``
callable that the runner injects:

    provider = YFinanceChainProvider(call=lambda tool, **kw: mcp_invoke(tool, kw))

That keeps the library pure and unit-testable, and it means the same code path
works whether the tool call goes through the agent, a local MCP client, or a
recorded fixture during backtesting.  ``RecordingProvider`` wraps any provider
and journals every response to disk -- that journal IS the backtest dataset,
which is the only way to guarantee the backtest sees exactly what the live
system saw, bugs and gaps included.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..domain.structures import OptionQuote, Right
from .provider import ChainSnapshot

__all__ = ["YFinanceChainProvider", "MassiveUnderlyingProvider", "CompositeProvider",
           "RecordingProvider"]

Call = Callable[..., Any]


# ----------------------------------------------------------------------------
# options chains: yfinance
# ----------------------------------------------------------------------------

@dataclass
class YFinanceChainProvider:
    call: Call
    assumed_latency_s: float = 900.0     # ~15 min; the real, honest number

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        raw = self.call(
            "mcp__remote-devices__yfmcp__yfinance_get_option_chain",
            symbol=symbol,
            expiration_date=expiry.isoformat() if expiry else None,
            option_type="all",
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        quotes: dict[date, list[OptionQuote]] = {}
        for exp_str, side_map in data.items():
            exp = date.fromisoformat(exp_str)
            rows: list[OptionQuote] = []
            for side, recs in side_map.items():
                right = Right.CALL if side.lower().startswith("call") else Right.PUT
                for rec in recs:
                    rows.append(OptionQuote(
                        strike=float(rec["strike"]), right=right, expiry=exp,
                        bid=float(rec.get("bid") or 0.0), ask=float(rec.get("ask") or 0.0),
                        last=float(rec.get("lastPrice") or np.nan),
                        volume=float(rec.get("volume") or 0.0),
                        open_interest=float(rec.get("openInterest") or 0.0),
                        # stored, NEVER consumed -- see data.provider docstring
                        iv_vendor=float(rec.get("impliedVolatility") or np.nan),
                        asof=str(rec.get("lastTradeDate", "")),
                    ))
            quotes[exp] = rows

        spot = float(self.call("mcp__remote-devices__yfmcp__yfinance_get_ticker_info",
                               symbol=symbol).get("regularMarketPrice", np.nan)) \
            if callable(self.call) else np.nan
        return ChainSnapshot(
            underlying=symbol, spot=spot, asof=datetime.now(timezone.utc),
            expiries=sorted(quotes), quotes=quotes,
            source="yfinance-mcp", latency_s=self.assumed_latency_s,
        )


# ----------------------------------------------------------------------------
# underlying, rates, dividends, events: Massive
# ----------------------------------------------------------------------------

@dataclass
class MassiveUnderlyingProvider:
    call: Call

    def _api(self, path: str, **params):
        return self.call("mcp__remote-devices__Massive_Market_Data__call_api",
                         path=path, params=params or None)

    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Returns OHLCV indexed by date.  Massive returns CSV text with the
        Polygon-style short column names: T,v,vw,o,c,h,l,t,n."""
        raw = self._api(f"/v2/aggs/ticker/{symbol}/range/1/day/{start:%Y-%m-%d}/{end:%Y-%m-%d}",
                        adjusted=True, sort="asc", limit=50000)
        from io import StringIO
        df = pd.read_csv(StringIO(raw if isinstance(raw, str) else raw["result"]))
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close",
                                "v": "volume", "vw": "vwap", "t": "ts", "n": "trades"})
        df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.date
        return df.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)

    def risk_free_curve(self, asof: date | None = None) -> dict[float, float]:
        """Treasury yields -> {years_to_maturity: continuously-compounded rate}.

        Interpolate this at each expiry's own tenor.  Using a single flat rate
        across a term structure is a small error at 30 DTE and a real one at
        LEAPS tenors, and it shows up as a spurious calendar-arbitrage signal.
        """
        raw = self._api("/fed/v1/treasury-yields", limit=1,
                        **({"date": asof.isoformat()} if asof else {}))
        rec = raw if isinstance(raw, dict) else json.loads(raw)
        tenors = {"yield_1_month": 1 / 12, "yield_3_month": 0.25, "yield_6_month": 0.5,
                  "yield_1_year": 1.0, "yield_2_year": 2.0, "yield_5_year": 5.0,
                  "yield_10_year": 10.0, "yield_30_year": 30.0}
        out = {}
        for k, yrs in tenors.items():
            v = rec.get(k)
            if v is None:
                continue
            simple = float(v) / 100.0
            out[yrs] = float(np.log1p(simple))       # -> continuously compounded
        return out

    def dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raw = self._api("/stocks/v1/dividends", ticker=symbol,
                        ex_dividend_date_gte=start.isoformat(),
                        ex_dividend_date_lte=end.isoformat(), limit=100)
        return pd.DataFrame(raw if isinstance(raw, list) else raw.get("results", []))

    def next_earnings(self, symbol: str) -> date | None:
        raw = self._api("/tmx/v1/corporate-events", ticker=symbol,
                        event_type="earnings", limit=5, sort="date", order="asc")
        recs = raw if isinstance(raw, list) else raw.get("results", [])
        for r in recs:
            d = r.get("date") or r.get("event_date")
            if d:
                try:
                    dd = date.fromisoformat(str(d)[:10])
                except ValueError:
                    continue
                if dd >= date.today():
                    return dd
        return None

    def market_open(self) -> bool:
        raw = self._api("/v1/marketstatus/now")
        rec = raw if isinstance(raw, dict) else json.loads(raw)
        return str(rec.get("market", "")).lower() == "open"


@dataclass
class CompositeProvider:
    """Options from one feed, everything else from another.

    This is the shape imposed by the entitlements, and it is also the shape you
    want anyway: the options feed is the expensive, swappable component; rates,
    dividends and events are commodity data.
    """
    chains: YFinanceChainProvider
    underlying: MassiveUnderlyingProvider

    def option_chain(self, symbol, expiry=None):
        return self.chains.option_chain(symbol, expiry)

    def daily_bars(self, symbol, start, end):
        return self.underlying.daily_bars(symbol, start, end)

    def risk_free_curve(self, asof=None):
        return self.underlying.risk_free_curve(asof)

    def dividends(self, symbol, start, end):
        return self.underlying.dividends(symbol, start, end)

    def next_earnings(self, symbol):
        return self.underlying.next_earnings(symbol)

    def market_open(self):
        return self.underlying.market_open()


@dataclass
class RecordingProvider:
    """Journal every response to disk.  This journal is the backtest dataset.

    Backtesting against a *reconstructed* history is the standard way to get a
    strategy that works beautifully offline and not at all live: the
    reconstruction silently repairs the gaps, stale prints and vendor sentinels
    that the live system actually has to survive.  Replaying your own journal
    removes that entire class of self-deception.
    """
    inner: Any
    path: Path
    # Injectable clock. Live recording stamps with wall time; a synthetic or
    # fast-forward collector stamps with SIMULATED time, so the replay cursor
    # orders the files by market time rather than by how fast the loop ran.
    # Without this every simulated day lands in the same second and the
    # backtest's ordering -- its entire no-look-ahead guarantee -- is undefined.
    clock: Callable[[], datetime] | None = None

    def __post_init__(self):
        self.path = Path(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _now(self) -> datetime:
        return self.clock() if self.clock is not None else datetime.now(timezone.utc)

    def __getattr__(self, name):
        fn = getattr(self.inner, name)
        if not callable(fn):
            return fn

        def wrapped(*a, **kw):
            out = fn(*a, **kw)
            stamp = self._now().strftime("%Y%m%dT%H%M%S%f")
            rec = {"method": name, "args": [str(x) for x in a],
                   "kwargs": {k: str(v) for k, v in kw.items()}, "ts": stamp}
            (self.path / f"{stamp}_{name}.json").write_text(
                json.dumps({"meta": rec, "payload": _jsonable(out)}, indent=1))
            return out
        return wrapped


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return json.loads(o.reset_index().to_json(orient="records", date_format="iso"))
    if isinstance(o, ChainSnapshot):
        return {"underlying": o.underlying, "spot": o.spot, "asof": o.asof.isoformat(),
                "source": o.source, "latency_s": o.latency_s,
                "quotes": {str(k): [vars(q) | {"right": q.right.value, "expiry": str(q.expiry)}
                                    for q in v] for k, v in o.quotes.items()}}
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)
