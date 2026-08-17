"""A synthetic provider with a known ground truth, and with the feed's real defects.

Two jobs, and the second is the one that matters:

**Deterministic end-to-end exercise.** The pipeline must be runnable with no
network, on a known surface, so that a test can assert the recovered forward,
the recovered SVI parameters and the arbitrage diagnostics against numbers that
are *true* rather than merely stable.

**Reproducing the pathologies, not sanitising them.** A synthetic feed that
emits clean two-sided quotes at every strike tests a system that does not exist.
This one reproduces, from direct observation of the live SPY chain:

  * zero-bid far-OTM strikes with a ``0.500005`` vendor-IV sentinel and zero
    open interest -- the exact yfinance artefact described in BLUEPRINT.md
    section 1;
  * spreads that widen in the wings, so the ``NO_VEGA`` filter has something to
    reject and the bid-ask-normalised loss has something to be insensitive to;
  * a term structure of ATM variance, so the calendar-arbitrage gate is
    exercised rather than merely present.

The ground truth is an SSVI surface, which means it is arbitrage-free by
construction: any butterfly or calendar violation the system reports against
this feed is a bug in the system, not in the data.  That is what makes it a
useful test oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.structures import OptionQuote, Right
from ..pricing import black
from ..surface.ssvi import SSVIParams, ssvi_w
from .provider import ChainSnapshot

__all__ = ["SyntheticProvider", "SyntheticSpec", "record_synthetic_history"]


@dataclass
class SyntheticSpec:
    """Ground truth.  Everything the provider emits is derived from this."""
    spot: float = 778.54
    rate: float = 0.0421
    dividend_yield: float = 0.0118
    ssvi: SSVIParams = field(default_factory=lambda: SSVIParams(rho=-0.72, eta=0.85, gamma=0.5))
    # ATM total variance as theta(T) = atm_vol^2 * T with a mild term structure.
    atm_vol_30d: float = 0.180
    term_slope: float = 0.35            # vol points of extra ATM vol per year
    # Expiries are FIXED CALENDAR DATES -- Fridays, like the real listing -- not
    # rolling day-counts. This matters far more than it looks: with rolling
    # DTEs the expiry dates shift every session, so a position opened yesterday
    # cannot be found in today's chain, nothing can ever be marked or closed,
    # and every backtest position silently runs to expiry. The management path
    # (profit target, stop, 21-DTE time stop) is only reachable when an expiry
    # persists across sessions.
    dte_window: tuple[int, int] = (5, 100)
    weekly_expiries: bool = True
    strike_step: float = 5.0
    strikes_halfwidth_pct: float = 0.22
    # The realised-vol path the daily bars are generated from. Set BELOW the
    # implied surface to create a positive variance risk premium -- the regime
    # in which the system is supposed to sell premium. 0.145 against an ~18 vol
    # ATM is roughly the 3-4 point SPX premium that is actually observed;
    # a bigger gap makes every gate pass and tests nothing.
    realised_vol: float = 0.145
    n_bars: int = 760
    # Intraday steps used to build each OHLC bar. This is NOT cosmetic: range
    # estimators assume the continuous high and low are observed, and with 8
    # steps the observed range understates the true one by ~29%, versus the
    # ~5% documented for real daily bars. Too few steps and the synthetic feed
    # exaggerates a bias the real feed has only mildly, which would make any
    # test of the forecast layer measure the fixture instead of the model.
    intraday_steps: int = 390
    seed: int = 20260817


@dataclass
class SyntheticProvider:
    """:class:`~optionsmarkets.data.provider.MarketDataProvider` over a known surface."""
    spec: SyntheticSpec = field(default_factory=SyntheticSpec)
    asof: datetime | None = None
    latency_s: float = 900.0
    earnings_in_days: int | None = None
    # Optional precomputed OHLC history. When a multi-session recording is
    # being built, every session must serve a PREFIX OF THE SAME PATH -- that is
    # what history is. Regenerating the bars per session (even from a coherent
    # generator) gives each day an unrelated past, so the realised volatility a
    # trade is scored against belongs to a different world from the one the
    # forecast was fitted on, and the learning layer is handed pure noise while
    # looking perfectly well-formed.
    bars: pd.DataFrame | None = None

    def __post_init__(self):
        self.asof = self.asof or datetime.now(timezone.utc)
        self._rng = np.random.default_rng(self.spec.seed)

    # ---- ground truth ----------------------------------------------------
    def theta(self, T: float) -> float:
        """ATM total variance.  Non-decreasing in T by construction."""
        s = self.spec
        vol = s.atm_vol_30d + s.term_slope * (T - 30.0 / 365.0)
        return float(max(vol, 0.03) ** 2 * T)

    def true_vol(self, k, T: float):
        return np.sqrt(np.maximum(ssvi_w(k, self.theta(T), self.spec.ssvi), 1e-12) / T)

    # ---- chains ----------------------------------------------------------
    def listed_expiries(self) -> list[date]:
        """Fixed-calendar expiries: Fridays inside the DTE window.

        Weeklies out to ~5 weeks then monthlies (third Fridays), which is the
        real SPY listing shape.  Crucially these are absolute dates, so the same
        expiry appears in every session's chain until it expires.
        """
        s = self.spec
        lo, hi = s.dte_window
        today = self.asof.date()
        out: list[date] = []
        d = today
        while (d - today).days <= hi:
            if d.weekday() == 4 and (d - today).days >= lo:      # Friday
                dte = (d - today).days
                third_friday = 15 <= d.day <= 21
                if s.weekly_expiries and dte <= 35:
                    out.append(d)
                elif third_friday:
                    out.append(d)
            d += timedelta(days=1)
        return out

    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        s = self.spec
        today = self.asof.date()
        expiries = self.listed_expiries()
        if expiry is not None:
            expiries = [e for e in expiries if e == expiry] or expiries[:1]
        if not expiries:                                          # pragma: no cover
            expiries = [today + timedelta(days=30)]

        quotes: dict[date, list[OptionQuote]] = {}
        for exp in expiries:
            T = max((exp - today).days, 1) / 365.0
            F = float(black.forward(s.spot, s.rate, s.dividend_yield, T))
            lo = np.floor(s.spot * (1 - s.strikes_halfwidth_pct) / s.strike_step) * s.strike_step
            hi = np.ceil(s.spot * (1 + s.strikes_halfwidth_pct) / s.strike_step) * s.strike_step
            rows: list[OptionQuote] = []
            for K in np.arange(lo, hi + s.strike_step, s.strike_step):
                iv = float(self.true_vol(np.log(K / F), T))
                for right in (Right.CALL, Right.PUT):
                    rows.append(self._quote(float(K), right, exp, T, iv))
            quotes[exp] = rows

        return ChainSnapshot(underlying=symbol.upper(), spot=s.spot, asof=self.asof,
                             expiries=sorted(quotes), quotes=quotes,
                             source="synthetic", latency_s=self.latency_s)

    def _quote(self, K: float, right: Right, exp: date, T: float, iv: float) -> OptionQuote:
        """One quote, dressed with the market microstructure the screens exist for."""
        s = self.spec
        mid = float(black.price(s.spot, K, T, s.rate, s.dividend_yield, iv, right.cp))
        vega = float(black.greeks(s.spot, K, T, s.rate, s.dividend_yield, iv, right.cp).vega)
        # Half-spread: a floor at the penny tick, a proportional part, and a
        # vega-proportional part so the wings widen the way a real book does.
        half = max(0.01, min(0.40, 0.010 * mid + 0.0011 * vega / 100.0 * 100))
        bid, ask = round(mid - half, 2), round(mid + half, 2)
        if bid <= 0.0:
            # The yfinance artefact, reproduced exactly: no bid, no open
            # interest, and a 0.500005 sentinel where an implied vol should be.
            return OptionQuote(K, right, exp, 0.0, max(round(ask, 2), 0.01), mid,
                               float(self._rng.integers(0, 40)), 0.0, 0.500005)
        return OptionQuote(K, right, exp, bid, ask, mid,
                           float(self._rng.integers(1, 2500)),
                           float(self._rng.integers(50, 9000)),
                           iv * (1 + self._rng.normal(0, 0.02)))

    # ---- underlying ------------------------------------------------------
    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """A GBM OHLC path at ``spec.realised_vol``.

        Open/high/low/close are generated from an intraday sub-path rather than
        drawn independently, so the range estimators see a *consistent* bar:
        high >= max(open, close) and low <= min(open, close) hold by
        construction, and Rogers-Satchell gets a range that actually
        corresponds to the close-to-close move.  Drawing them independently
        produces bars whose implied volatility depends on which estimator you
        use, which would make any test of the forecast layer meaningless.

        When ``bars`` was supplied, this serves a slice of that single shared
        path instead of generating a new one.
        """
        if self.bars is not None:
            df = self.bars
            return df[(df.index >= start) & (df.index <= end)]
        s = self.spec
        rng = np.random.default_rng(s.seed + 1)
        n = int(s.n_bars)
        steps = int(s.intraday_steps)
        dt = 1.0 / 252.0 / steps
        sig = s.realised_vol
        drift = (s.rate - s.dividend_yield - 0.5 * sig**2) * dt
        shocks = rng.normal(drift, sig * np.sqrt(dt), size=(n, steps))
        logs = np.log(s.spot) - float(np.sum(shocks)) + np.cumsum(shocks.ravel())
        path = np.exp(logs).reshape(n, steps)

        o = path[:, 0]
        c = path[:, -1]
        h = path.max(axis=1)
        low = path.min(axis=1)
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=n).date
        df = pd.DataFrame({"open": o, "high": h, "low": low, "close": c,
                           "volume": rng.integers(4e7, 1.2e8, n).astype(float)}, index=idx)
        return df[(df.index >= start) & (df.index <= end)]

    def risk_free_curve(self, asof: date | None = None) -> dict[float, float]:
        r = self.spec.rate
        return {1 / 12: r - 0.0004, 0.25: r - 0.0002, 0.5: r, 1.0: r + 0.0003,
                2.0: r + 0.0009, 5.0: r + 0.0021, 10.0: r + 0.0035, 30.0: r + 0.0048}

    def dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Quarterly ex-dates consistent with ``spec.dividend_yield``."""
        s = self.spec
        per_quarter = s.dividend_yield * s.spot / 4.0
        rows = []
        d = start
        while d <= end:
            rows.append({"ex_dividend_date": d, "cash_amount": round(per_quarter, 3)})
            d += timedelta(days=91)
        return pd.DataFrame(rows)

    def next_earnings(self, symbol: str) -> date | None:
        if self.earnings_in_days is None:
            return None
        return self.asof.date() + timedelta(days=int(self.earnings_in_days))

    def market_open(self) -> bool:
        return True


# ----------------------------------------------------------------------------
# recording a replayable history
# ----------------------------------------------------------------------------

def record_synthetic_history(
    path, *, symbol: str = "SPY", days: int = 60, step_days: int = 1,
    start: datetime | None = None, spec: SyntheticSpec | None = None,
    vol_of_vol: float = 0.55, vol_mean_reversion: float = 0.06,
    seed: int = 4242,
):
    """Write a replayable snapshot journal from an evolving synthetic market.

    Produces exactly what a live
    :class:`~optionsmarkets.data.mcp_adapters.RecordingProvider` would have
    written over ``days`` sessions, so
    :class:`~optionsmarkets.data.replay.ReplayProvider` and the backtester can
    be exercised without waiting for real recordings to accumulate.

    The market evolves rather than repeating.  Three properties are deliberate:

      * **One coherent price path.**  A single OHLC history is generated up
        front and every session serves a PREFIX of it, exactly as real history
        behaves.  Regenerating bars per session would give each day an unrelated
        past, so the realised volatility a trade is scored against would belong
        to a different world from the one its forecast was fitted on -- the
        learning layer would be fed noise while looking perfectly well-formed,
        and the backtest would report the variance of a random number generator.
      * **Implied and realised are driven by different shocks.**  ATM implied
        mean-reverts with its own innovations while spot follows the realised
        path.  A fixture where implied is a deterministic function of realised
        would let the forecast layer look brilliant for reasons that do not
        exist in any market.
      * **The variance risk premium changes sign** on some sessions, so both
        branches of the direction rule get exercised rather than only the
        premium-selling one.

    This is a TEST FIXTURE and a smoke-test harness.  It is not history, and a
    backtest over it establishes that the machinery is correct -- never that the
    strategy is profitable.
    """
    from .mcp_adapters import RecordingProvider

    path = Path(path)
    base = spec or SyntheticSpec()
    start = start or (datetime.now(timezone.utc) - timedelta(days=days * step_days))
    rng = np.random.default_rng(seed)

    # ---- one shared price path, covering the lookback AND the replay window --
    sessions = [start + timedelta(days=i * step_days) for i in range(int(days))]
    sessions = [d for d in sessions if d.weekday() < 5]
    if not sessions:
        return []
    master = SyntheticProvider(
        spec=SyntheticSpec(**{**vars(base),
                              "n_bars": base.n_bars + len(sessions) + 5}),
        asof=sessions[-1],
    ).daily_bars(symbol, date(1970, 1, 1), sessions[-1].date())
    closes = {d: float(v) for d, v in master["close"].items()}

    clock = {"t": start}
    atm = base.atm_vol_30d
    stamps: list[datetime] = []

    for i, when in enumerate(sessions):
        clock["t"] = when

        # Implied vol: mean-reverting with its OWN shock, floored well above 0.
        atm += vol_mean_reversion * (base.atm_vol_30d - atm) + \
            atm * vol_of_vol * np.sqrt(step_days / 252.0) * rng.normal()
        atm = float(np.clip(atm, 0.07, 0.85))

        # Spot comes from the shared path, so the chain and the bar history
        # agree about where the underlying is.
        upto = [d for d in master.index if d <= when.date()]
        if not upto:
            continue
        spot = closes[upto[-1]]

        day_spec = SyntheticSpec(**{**vars(base), "spot": float(spot),
                                    "atm_vol_30d": atm, "seed": base.seed + i})
        inner = SyntheticProvider(spec=day_spec, asof=when,
                                  bars=master[master.index <= when.date()])
        rec = RecordingProvider(inner, path, clock=lambda: clock["t"])

        rec.option_chain(symbol)
        rec.daily_bars(symbol, (when - timedelta(days=base.n_bars * 2)).date(), when.date())
        rec.risk_free_curve()
        rec.dividends(symbol, (when - timedelta(days=365)).date(), when.date())
        rec.next_earnings(symbol)
        rec.market_open()
        stamps.append(when)

    return stamps
