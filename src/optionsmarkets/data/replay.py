"""Replay a recorded snapshot journal as a :class:`MarketDataProvider`.

BLUEPRINT.md section 13.4 is emphatic that the backtest must replay the
**recorded journal**, not a reconstructed history, and the reason is worth
restating because it is the difference between a useful backtest and a
comfortable one: a reconstruction silently *repairs* the gaps, stale prints and
vendor sentinels that the live system actually has to survive.  You end up
testing a strategy against a market that never existed, and the first thing you
learn live is which of those repairs you were depending on.

The no-look-ahead guarantee here is structural rather than disciplinary.  The
provider holds a cursor, and every method serves the most recent recording at or
before it:

  * ``option_chain`` returns the snapshot AT the cursor;
  * ``daily_bars`` returns recorded bars **truncated at the cursor date**;
  * rates, dividends and earnings return the last recording at or before it.

There is no code path that returns a later record.  A backtest cannot peek by
accident, because peeking is not expressible -- which is a much stronger
statement than "we were careful".
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.structures import OptionQuote, Right
from .provider import ChainSnapshot

__all__ = ["ReplayProvider", "LookAheadError"]


class LookAheadError(RuntimeError):
    """Raised when a replay is asked for data the cursor cannot legitimately see."""


def _parse_stamp(name: str) -> datetime | None:
    """``20260817T062504123456_option_chain.json`` -> aware datetime."""
    head = name.split("_", 1)[0]
    for fmt in ("%Y%m%dT%H%M%S%f", "%Y%m%dT%H%M%S"):
        try:
            return datetime.strptime(head, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _recorded_symbol(path: Path) -> str | None:
    """The underlying a recording was made FOR, from its ``meta.args``.

    The filename carries only ``{stamp}_{method}``, so the symbol lives inside
    the file.  ``RecordingProvider`` writes ``{"meta": ..., "payload": ...}`` in
    that order, so the meta block sits in the first few hundred bytes and a head
    read avoids parsing 600 KB of chain JSON per file at index time -- measured
    132 files, 5.2 MB, where full parsing costs ~1.4 s and this costs ~4 ms.

    Falls back to a full parse if the head read cannot find a complete meta
    object, and returns None for genuinely symbol-less calls such as
    ``risk_free_curve`` and ``market_open``.
    """
    def _sym(meta) -> str | None:
        args = (meta or {}).get("args") or []
        return str(args[0]) if args else None

    try:
        with path.open("rb") as fh:
            head = fh.read(2048).decode("utf-8", "replace")
        m = re.search(r'"meta"\s*:\s*\{', head)
        if m:
            depth, i = 1, m.end()
            while i < len(head) and depth:
                depth += (head[i] == "{") - (head[i] == "}")
                i += 1
            if not depth:
                return _sym(json.loads(head[m.end() - 1:i]))
        return _sym(json.loads(path.read_text()).get("meta"))
    except (OSError, ValueError):
        return None


@dataclass
class ReplayProvider:
    """Serve a :class:`~optionsmarkets.data.mcp_adapters.RecordingProvider` journal.

    Parameters
    ----------
    path
        Directory the ``RecordingProvider`` wrote to.
    strict
        When True (the default) a request for a method with no recording at or
        before the cursor raises rather than returning a fallback.  Strict is
        the right default for a backtest: a silent fallback is an assumption you
        did not make deliberately, and it will differ from what the live system
        did on exactly the days that matter.
    """
    path: Path
    strict: bool = True
    _index: dict[str, list[tuple[datetime, Path, str | None]]] = field(
        default_factory=dict, repr=False)
    _cursor: datetime | None = field(default=None, repr=False)

    def __post_init__(self):
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"no recording directory at {self.path}")
        self._index = {}
        for f in sorted(self.path.glob("*.json")):
            ts = _parse_stamp(f.name)
            if ts is None:
                continue
            method = f.stem.split("_", 1)[1] if "_" in f.stem else f.stem
            self._index.setdefault(method, []).append((ts, f, _recorded_symbol(f)))
        for v in self._index.values():
            v.sort(key=lambda kv: kv[0])
        if not self._index.get("option_chain"):
            raise FileNotFoundError(
                f"{self.path} contains no recorded option_chain snapshots; "
                f"there is nothing to replay")
        self._cursor = self.timeline()[0]

    def symbols(self) -> set[str]:
        """Every underlying with a recorded chain in this directory."""
        return {s for _, _, s in self._index.get("option_chain", []) if s}

    # ---- cursor ----------------------------------------------------------
    def timeline(self) -> list[datetime]:
        """Every recorded chain-snapshot time, in order.  The backtest's clock."""
        return [ts for ts, _, _ in self._index["option_chain"]]

    def seek(self, when: datetime) -> "ReplayProvider":
        self._cursor = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
        return self

    @property
    def cursor(self) -> datetime:
        return self._cursor

    def _latest(self, method: str, symbol: str | None = None):
        """Most recent recording of ``method`` at or before the cursor.

        ``symbol`` is not decoration.  ``RecordingProvider`` names files
        ``{stamp}_{method}.json`` with no underlying in them, so one directory
        collecting several tickers interleaves them, and serving the newest
        recording regardless of who asked hands SPY's caller whatever happened
        to be written last.  That is a silent wrong answer -- it does not raise,
        it just backtests a different instrument -- so an unfiltered read is
        refused rather than approximated.  Observed 2026-08-18: a collector run
        over SPY/QQQ/IWM wrote 24 files into one flat directory, after which
        ``daily_bars("SPY", ...)`` returned IWM's bars.
        """
        entries = self._index.get(method, [])
        if symbol is not None:
            known = {s for _, _, s in entries if s}
            if known:
                filtered = [e for e in entries if e[2] == symbol]
                if not filtered and self.strict:
                    raise LookAheadError(
                        f"no '{method}' recordings for {symbol!r} in {self.path}; "
                        f"it holds {sorted(known)}. Serving another underlying's "
                        f"data would not raise -- it would silently backtest the "
                        f"wrong instrument. Record {symbol} or point --snapshots "
                        f"at its own directory.")
                entries = filtered
        if not entries:
            if self.strict:
                raise LookAheadError(
                    f"no recordings of '{method}' in {self.path}. The live system "
                    f"had this data; a backtest that substitutes a default for it "
                    f"is not replaying the live system.")
            return None
        stamps = [ts for ts, _, _ in entries]
        i = bisect_right(stamps, self._cursor) - 1
        if i < 0:
            if self.strict:
                raise LookAheadError(
                    f"the earliest '{method}' recording is {stamps[0].isoformat()}, "
                    f"after the cursor {self._cursor.isoformat()}; serving it would "
                    f"be look-ahead")
            return None
        return json.loads(entries[i][1].read_text())

    # ---- provider surface -----------------------------------------------
    def option_chain(self, symbol: str, expiry: date | None = None) -> ChainSnapshot:
        rec = self._latest("option_chain", symbol)
        snap = _chain_from_payload(rec["payload"])
        if expiry is not None:
            snap = ChainSnapshot(snap.underlying, snap.spot, snap.asof, [expiry],
                                 {expiry: snap.quotes.get(expiry, [])}, snap.source,
                                 snap.latency_s)
        return snap

    def daily_bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Recorded bars, TRUNCATED AT THE CURSOR.

        The truncation is the whole point.  A recording made on day 100 contains
        bars up to day 100; replaying day 40 must not see them, and the cheapest
        way to guarantee that is to cut here rather than to trust every caller
        to pass the right ``end``.
        """
        rec = self._latest("daily_bars", symbol)
        if rec is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rec["payload"])
        if df.empty:
            return df
        idx_col = next((c for c in ("date", "index", "Date") if c in df.columns), None)
        if idx_col is not None:
            df.index = pd.to_datetime(df[idx_col]).dt.date
            df = df.drop(columns=[idx_col])
        cutoff = min(end, self._cursor.date())
        df = df[(df.index >= start) & (df.index <= cutoff)]
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].astype(float)

    def risk_free_curve(self, asof: date | None = None) -> dict[float, float]:
        rec = self._latest("risk_free_curve")
        if rec is None:
            return {}
        return {float(k): float(v) for k, v in (rec["payload"] or {}).items()}

    def dividends(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rec = self._latest("dividends", symbol)
        if rec is None:
            return pd.DataFrame()
        return pd.DataFrame(rec["payload"] or [])

    def next_earnings(self, symbol: str) -> date | None:
        rec = self._latest("next_earnings", symbol)
        if rec is None or rec["payload"] in (None, "None", ""):
            return None
        try:
            return date.fromisoformat(str(rec["payload"])[:10])
        except ValueError:
            return None

    def market_open(self) -> bool:
        rec = self._latest("market_open")
        return bool(rec["payload"]) if rec else True


def _chain_from_payload(p: dict) -> ChainSnapshot:
    quotes: dict[date, list[OptionQuote]] = {}
    for exp_str, rows in (p.get("quotes") or {}).items():
        exp = date.fromisoformat(str(exp_str)[:10])
        out = []
        for r in rows:
            out.append(OptionQuote(
                strike=float(r["strike"]), right=Right(r["right"]), expiry=exp,
                bid=float(r.get("bid") or 0.0), ask=float(r.get("ask") or 0.0),
                last=_f(r.get("last")), volume=float(r.get("volume") or 0.0),
                open_interest=float(r.get("open_interest") or 0.0),
                iv_vendor=_f(r.get("iv_vendor")), asof=str(r.get("asof", "")),
            ))
        quotes[exp] = out
    return ChainSnapshot(
        underlying=str(p.get("underlying", "")), spot=_f(p.get("spot")),
        asof=datetime.fromisoformat(p["asof"]), expiries=sorted(quotes),
        quotes=quotes, source=str(p.get("source", "replay")),
        latency_s=float(p.get("latency_s") or 0.0),
    )


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return np.nan
    return x
