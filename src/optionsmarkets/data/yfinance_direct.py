"""In-process yfinance provider -- the same data the yfinance MCP server serves.

The MCP adapters in :mod:`.mcp_adapters` take an injected ``call`` because the
tool invocation belongs to the agent, not to this process.  That is the right
seam for an agent-driven run and the wrong one for a cron job, a backtest
collector, or a test: all three want to import a provider and use it.

This adapter talks to the ``yfinance`` package directly, which is what the
bundled MCP server wraps anyway, so it returns the same fields with the same
pathologies -- including the ``0.500005`` implied-volatility sentinel on
zero-bid strikes, which is preserved deliberately.  Sanitising it here would
hide the exact failure mode the quality screen exists to catch, and the first
place you would notice is a phantom 50-vol wing on a live surface.

Install with the ``live`` extra::

    pip install -e ".[live]"

Everything in this module is a *diagnosis-preserving* translation.  It does not
clean, impute, or repair.  The screens downstream are entitled to see the feed
as it really is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..domain.structures import OptionQuote, Right
from .provider import ChainSnapshot

__all__ = ["YFinanceDirectProvider"]


@dataclass
class YFinanceDirectProvider:
    """:class:`~optionsmarkets.data.provider.MarketDataProvider` over ``yfinance``.

    ``assumed_latency_s`` is the honest number, not an aspirational one.  Yahoo
    option chains are delayed roughly 15 minutes and carry no NBBO timestamp, so
    the freshness gate is set to what the feed actually is.  Claiming 60 seconds
    would not make the data fresher; it would make every decision fail the gate
    and teach nothing.
    """
    assumed_latency_s: float = 900.0
    # Each expiry is a separate network round-trip, so the count is capped --
    # but the cap must SPAN the window rather than truncate it. SPY lists daily
    # expiries, so "the nearest 8" is eight consecutive days and never reaches
    # the 21-45 DTE band the strategy actually trades. Selection below samples
    # across ``dte_window`` instead, always keeping the nearest and the furthest.
    max_expiries: int = 8
    dte_window: tuple[int, int] = (5, 120)
    _cache: dict = field(default_factory=dict, repr=False)

    # ---- internals -------------------------------------------------------
    def _yf(self):
        try:
            import yfinance
        except ImportError as exc:                                  # pragma: no cover
            raise RuntimeError(
                "yfinance is not installed. Install the live extra: "
                'pip install -e ".[live]"') from exc
        return yfinance

    def _ticker(self, symbol: str):
        key = symbol.upper()
        if key not in self._cache:
            self._cache[key] = self._yf().Ticker(key)
        return self._cache[key]

    # ---- chains ----------------------------------------------------------
    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        tk = self._ticker(symbol)
        listed = [date.fromisoformat(e) for e in tk.options]
        if not listed:
            raise RuntimeError(f"{symbol}: yfinance returned no listed expiries")
        wanted = [expiry] if expiry is not None else self._span_expiries(listed)

        quotes: dict[date, list[OptionQuote]] = {}
        for exp in wanted:
            if exp not in listed:
                continue
            ch = tk.option_chain(exp.isoformat())
            rows: list[OptionQuote] = []
            for frame, right in ((ch.calls, Right.CALL), (ch.puts, Right.PUT)):
                rows.extend(_frame_to_quotes(frame, right, exp))
            if rows:
                quotes[exp] = rows
        if not quotes:
            raise RuntimeError(f"{symbol}: no quotes returned for {wanted}")

        return ChainSnapshot(
            underlying=symbol.upper(), spot=self._spot(symbol),
            asof=datetime.now(timezone.utc), expiries=sorted(quotes),
            quotes=quotes, source="yfinance-direct",
            latency_s=float(self.assumed_latency_s),
        )

    def _span_expiries(self, listed: list[date]) -> list[date]:
        """Up to ``max_expiries`` listed expiries SPANNING ``dte_window``.

        Evenly sampled across the window rather than taken from the front.  The
        surface needs a term structure to enforce calendar-freedom against, and
        four consecutive daily expiries give you four nearly-identical thetas
        and no leverage on the calendar constraint at all.
        """
        lo, hi = self.dte_window
        today = date.today()
        band = [e for e in sorted(listed) if lo <= (e - today).days <= hi]
        if not band:
            band = sorted(listed)[: self.max_expiries]
        if len(band) <= self.max_expiries:
            return band
        idx = np.unique(np.linspace(0, len(band) - 1, self.max_expiries).round().astype(int))
        return [band[i] for i in idx]

    def _spot(self, symbol: str) -> float:
        """Last price.  Only used to place delta-targeted strikes and to bound
        the parity regression -- the FORWARD always comes from the option market
        itself, so a slightly stale spot cannot tilt the slice."""
        tk = self._ticker(symbol)
        for getter in (lambda: tk.fast_info["last_price"],
                       lambda: tk.info["regularMarketPrice"],
                       lambda: float(tk.history(period="1d")["Close"].iloc[-1])):
            try:
                v = float(getter())
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                continue
        return float("nan")

    # ---- underlying ------------------------------------------------------
    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        tk = self._ticker(symbol)
        df = tk.history(start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                        interval="1d", auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(f"{symbol}: no daily bars for {start}..{end}")
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                "Close": "close", "Volume": "volume"})
        out = df[["open", "high", "low", "close", "volume"]].astype(float)
        out.index = pd.to_datetime(out.index).date
        # Range estimators divide by the low and take logs of the range; a zero
        # or non-positive bar poisons every window it touches rather than just
        # its own day, so it is dropped here where the damage is contained.
        return out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]

    def risk_free_curve(self, asof: date | None = None) -> dict[float, float]:
        """Treasury curve from the CBOE/CME yield proxies Yahoo publishes.

        These are index quotes, not the Fed's H.15 constant-maturity series, and
        they are close enough for per-expiry discounting at the tenors this
        system trades.  For anything long-dated, use the Massive
        ``/fed/v1/treasury-yields`` adapter instead -- the difference is small in
        absolute terms and it lands entirely in the forward, which is the one
        number the whole slice is measured against.
        """
        tenors = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}
        out: dict[float, float] = {}
        for sym, yrs in tenors.items():
            try:
                h = self._ticker(sym).history(period="5d")["Close"].dropna()
                if h.empty:
                    continue
                out[yrs] = float(np.log1p(float(h.iloc[-1]) / 100.0))
            except Exception:
                continue
        if not out:
            raise RuntimeError("no treasury proxy quotes available")
        return out

    def dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        tk = self._ticker(symbol)
        s = tk.dividends
        if s is None or len(s) == 0:
            return pd.DataFrame(columns=["ex_dividend_date", "cash_amount"])
        idx = pd.to_datetime(s.index).date
        df = pd.DataFrame({"ex_dividend_date": idx, "cash_amount": s.to_numpy(float)})
        return df[(df["ex_dividend_date"] >= start) & (df["ex_dividend_date"] <= end)]

    def next_earnings(self, symbol: str) -> date | None:
        """Next earnings date, or None if the name does not report.

        None is ambiguous on purpose and the caller must decide what to do with
        it: an index ETF genuinely has no earnings date, while a single name
        with a missing one is a data gap.  ``RunConfig.earnings_unknown_action``
        is where that decision is made, and its default is to fail closed.
        """
        tk = self._ticker(symbol)
        for getter in (lambda: tk.calendar, lambda: tk.get_earnings_dates(limit=8)):
            try:
                obj = getter()
            except Exception:
                continue
            for d in _earnings_dates(obj):
                if d >= date.today():
                    return d
        return None

    def market_open(self) -> bool:
        """Session state, best effort.

        yfinance exposes no session endpoint, so this is inferred from the clock
        and is NOT authoritative -- it does not know about holidays or early
        closes.  The freshness gate is the real protection: a stale snapshot
        fails it whatever this returns.
        """
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return 13 * 60 + 30 <= minutes < 20 * 60          # 09:30-16:00 ET in UTC (EDT)


# ----------------------------------------------------------------------------
# translation
# ----------------------------------------------------------------------------

def _frame_to_quotes(frame: pd.DataFrame, right: Right, exp: date) -> list[OptionQuote]:
    if frame is None or frame.empty:
        return []
    out: list[OptionQuote] = []
    for rec in frame.to_dict("records"):
        try:
            strike = float(rec["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(OptionQuote(
            strike=strike, right=right, expiry=exp,
            bid=_f(rec.get("bid"), 0.0), ask=_f(rec.get("ask"), 0.0),
            last=_f(rec.get("lastPrice"), np.nan),
            volume=_f(rec.get("volume"), 0.0),
            open_interest=_f(rec.get("openInterest"), 0.0),
            # Stored, NEVER consumed. yfinance emits 0.500005 on zero-bid
            # strikes; that is a sentinel, not a measurement, and feeding it to
            # a surface fitter produces a flat 50-vol wing that looks real.
            iv_vendor=_f(rec.get("impliedVolatility"), np.nan),
            asof=str(rec.get("lastTradeDate", "")),
        ))
    return out


def _f(v, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def _earnings_dates(obj) -> list[date]:
    """Pull dates out of yfinance's several earnings-calendar shapes."""
    out: list[date] = []
    if obj is None:
        return out
    if isinstance(obj, dict):
        for key in ("Earnings Date", "earningsDate", "Earnings High", "earnings_date"):
            v = obj.get(key)
            for item in (v if isinstance(v, (list, tuple)) else [v]):
                d = _to_date(item)
                if d:
                    out.append(d)
    elif isinstance(obj, pd.DataFrame):
        for item in list(obj.index) + (
                list(obj["Earnings Date"]) if "Earnings Date" in obj.columns else []):
            d = _to_date(item)
            if d:
                out.append(d)
    return sorted(set(out))


def _to_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    try:
        return pd.to_datetime(v).date()
    except (ValueError, TypeError):
        return None
