"""The outcome journal -- the record that makes the system able to learn.

BLUEPRINT.md section 8.6.  Every decision writes a complete record: the quote
snapshot it saw, the surface it fitted and that surface's diagnostics, the
forecast and its interval, every gate value, the structure chosen, the whole
sizing chain, and the ticket.  Later, the FILL and the REALISED OUTCOME are
appended against the same id.

Two properties are non-negotiable and drive the format:

**Append-only, one JSON object per line.** A trading journal that can be edited
in place is not evidence.  JSONL survives a crash mid-write losing at most the
last line, is greppable, streams without loading, and never silently rewrites
history the way a pickled object graph or an ORM does.

**HOLD decisions are journaled too.** This is the part that gets skipped and it
is the more valuable half of the dataset.  The distribution of *which gate
blocked* is the only way to learn what the system is actually constrained by,
and without it you cannot tell a strategy with no opportunities from a threshold
set too tight to ever fire.  BLUEPRINT.md section 13.4 requires that
distribution in the backtest report; it can only come from here.

What each row feeds (section 8.6):

    realised RV vs forecast    -> Kalman update on HAR coefficients; ACI update
    realised P&L vs EV         -> Sharpe estimate -> uncertainty shrinkage c_unc
    profitable? (binary)       -> probability calibrator -> model.calibration gate
    fill price vs ticket limit -> slippage model -> the marketable assumption
    all of the above           -> ADWIN -> refit trigger and lookback

The second of those is self-correcting in a way worth stating: ``c_unc`` is
computed from the *realised out-of-sample* Sharpe and the length of the record.
A strategy that stops working shrinks its own position size automatically, with
nobody changing a threshold.  That only happens if this file is written
faithfully, including the trades that lost.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

__all__ = [
    "DecisionRecord", "FillRecord", "OutcomeRecord", "OutcomeJournal",
    "ClosedTrade", "new_decision_id", "jsonable",
]


# ----------------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------------

def jsonable(o: Any) -> Any:
    """Convert numpy / dataclass / date soup into something json.dumps accepts.

    Written defensively rather than strictly: a journal write must never be the
    thing that kills a decision run.  Anything genuinely unserialisable is
    stringified and recorded as such, which loses fidelity but keeps the row.
    """
    if o is None or isinstance(o, (bool, str)):
        return o
    # Checked before the generic numeric branch: NaN and +/-inf are legal
    # Python floats and illegal JSON, and they arrive constantly from the
    # quality screen (an infinite relative spread on a zero-bid strike is a
    # real, meaningful value). They become null rather than crashing the write.
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, int):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return [jsonable(x) for x in o.tolist()]
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(x) for x in o]
    if hasattr(o, "__dataclass_fields__"):
        return {k: jsonable(v) for k, v in asdict(o).items()}
    if hasattr(o, "as_dict"):
        return jsonable(o.as_dict())
    if hasattr(o, "value") and hasattr(o, "name"):        # Enum
        return o.value
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)


_COUNTER = {"n": 0}


def new_decision_id(symbol: str, asof: datetime | None = None) -> str:
    """Sortable, human-readable, collision-resistant within a process."""
    asof = asof or datetime.now(timezone.utc)
    _COUNTER["n"] += 1
    return f"{asof.strftime('%Y%m%dT%H%M%S')}-{symbol.upper()}-{_COUNTER['n']:04d}"


# ----------------------------------------------------------------------------
# record types
# ----------------------------------------------------------------------------

@dataclass
class DecisionRecord:
    """Everything that was true at decision time, and nothing that was not.

    The no-look-ahead guarantee in BLUEPRINT.md section 13.4 is enforceable only
    because this record is self-contained: replaying it reproduces the decision
    from the same inputs, and any input that is not here could not have been
    used.  If you find yourself needing a field during a backtest that is not in
    this record, that is the discovery that the live path was using something it
    should not have.
    """
    id: str
    ts: str                          # when the decision was made (UTC ISO)
    symbol: str
    asof: str                        # quote-snapshot timestamp
    quote_age_s: float
    spot: float
    action: str                      # BUY | SELL | HOLD
    contracts: int
    structure_name: str = ""
    legs: list[dict] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    forward: dict = field(default_factory=dict)
    surface: dict = field(default_factory=dict)
    iv_rejections: dict = field(default_factory=dict)
    forecast: dict = field(default_factory=dict)
    vrp: dict = field(default_factory=dict)
    edge: dict = field(default_factory=dict)
    sizing: dict = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    ticket: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    learned_state: dict = field(default_factory=dict)
    portfolio: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    rationale: str = ""
    kind: str = "decision"

    @property
    def blocked_by(self) -> list[str]:
        return [g["name"] for g in self.gates if not g.get("passed", False)]

    @property
    def traded(self) -> bool:
        return self.action in ("BUY", "SELL") and self.contracts >= 1


@dataclass
class FillRecord:
    """What the market actually gave you, against two different references.

    All prices are per spread in dollars, signed the same way as
    ``Structure.net_price`` (debit positive, credit negative).

    Both references are recorded because they answer different questions and
    confusing them breaks the feedback loop in a way that is hard to see:

    ``limit_price`` is the TICKET's limit, which starts at mid.  ``fill_price``
    against it answers *"did I get my price?"* -- the execution-quality question
    a trader asks.  It is typically tens of dollars on a wide spread, because mid
    was always optimistic.

    ``marketable_reference`` is ``net_price("marketable")`` at decision time --
    the price the SCORING layer already assumed it would pay.  ``fill_price``
    against it answers *"was my modelling assumption right?"*, and that is the
    only one that may be fed back into future expected values.

    Using the limit-based number for that purpose double-counts the entire
    mid-to-marketable spread.  Observed directly during a replay: the loop
    measured $27.50/spread of "slippage", added it to every subsequent
    candidate's net price, drove all expected values negative, and the system
    stopped trading permanently -- from two fills that were in fact within
    $2 of what the model assumed.
    """
    id: str
    decision_id: str
    ts: str
    contracts: int
    limit_price: float
    fill_price: float
    marketable_reference: float = np.nan
    commission: float = 0.0
    fees: float = 0.0
    venue: str = "schwab"
    note: str = ""
    kind: str = "fill"

    @property
    def slippage_per_spread(self) -> float:
        """Fill vs the ticket LIMIT.  Execution quality; do not feed to the model."""
        return float(self.fill_price - self.limit_price)

    @property
    def slippage_vs_marketable(self) -> float:
        """Fill vs the price the scoring layer ASSUMED.  This is the model input.

        NaN when the reference was not recorded, and the feedback loop treats
        that as "unknown" rather than substituting the limit-based number -- an
        unknown that defaults to a wrong value is worse than a gap.
        """
        if not np.isfinite(self.marketable_reference):
            return np.nan
        return float(self.fill_price - self.marketable_reference)


@dataclass
class OutcomeRecord:
    """How it actually ended.

    ``realised_rv`` is the realised volatility over the *holding period*, on the
    same annualised scale as the forecast that justified the trade.  That
    pairing is what the Kalman and conformal updates consume; recording a
    forecast without the matching realisation makes the learning layer
    unfalsifiable.
    """
    id: str
    decision_id: str
    ts: str
    exit_price: float                # net per spread, same sign convention
    realised_pnl: float              # dollars, ALL contracts, net of every fee
    contracts: int
    days_held: int
    exit_reason: str                 # profit_target | stop | time_stop | expiry | manual
    realised_rv: float = np.nan      # annualised, over the holding period
    forecast_rv: float = np.nan      # what was predicted, for the same window
    underlying_close: float = np.nan
    commission: float = 0.0
    fees: float = 0.0
    assigned: bool = False
    note: str = ""
    kind: str = "outcome"

    @property
    def profitable(self) -> bool:
        return float(self.realised_pnl) > 0.0


@dataclass
class ClosedTrade:
    """A decision joined to its fill and its outcome.  The learning unit."""
    decision: DecisionRecord
    fill: FillRecord | None
    outcome: OutcomeRecord

    @property
    def pnl(self) -> float:
        return float(self.outcome.realised_pnl)

    @property
    def capital_at_risk(self) -> float:
        return float(self.decision.sizing.get("capital_at_risk", np.nan))

    @property
    def return_on_risk(self) -> float:
        r = self.capital_at_risk
        return self.pnl / r if np.isfinite(r) and r > 0 else np.nan

    @property
    def ev_predicted(self) -> float:
        """Predicted dollar EV for the whole position, not per spread."""
        ev = self.decision.edge.get("ev_per_spread", np.nan)
        return float(ev) * self.decision.contracts if np.isfinite(ev) else np.nan

    @property
    def pop_predicted(self) -> float:
        return float(self.decision.edge.get("pop", np.nan))


# ----------------------------------------------------------------------------
# the journal
# ----------------------------------------------------------------------------

class OutcomeJournal:
    """Append-only JSONL journal of decisions, fills and outcomes.

    One file, three record kinds, joined on ``decision_id``.  Kept in one file
    rather than three so that the ordering between a decision and the fill that
    followed it is a property of the file, not of a join you have to trust.
    """

    def __init__(self, path: str | Path = "journal/outcomes.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- writing ---------------------------------------------------------
    def _append(self, obj: Any) -> None:
        line = json.dumps(jsonable(obj), separators=(",", ":"), allow_nan=False)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def append_decision(self, rec: DecisionRecord) -> DecisionRecord:
        self._append(rec)
        return rec

    def append_fill(self, rec: FillRecord) -> FillRecord:
        self._append(rec)
        return rec

    def append_outcome(self, rec: OutcomeRecord) -> OutcomeRecord:
        self._append(rec)
        return rec

    # ---- reading ---------------------------------------------------------
    def rows(self) -> Iterator[dict]:
        """Stream raw rows.  A corrupt trailing line -- the one case a crash can
        produce -- is skipped rather than aborting the read of everything before it."""
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def decisions(self) -> list[DecisionRecord]:
        return [_as(DecisionRecord, r) for r in self.rows() if r.get("kind") == "decision"]

    def fills(self) -> list[FillRecord]:
        return [_as(FillRecord, r) for r in self.rows() if r.get("kind") == "fill"]

    def outcomes(self) -> list[OutcomeRecord]:
        return [_as(OutcomeRecord, r) for r in self.rows() if r.get("kind") == "outcome"]

    def closed_trades(self) -> list[ClosedTrade]:
        """Decisions that have a matching outcome, in chronological order.

        Chronological by OUTCOME timestamp, not by decision timestamp: the
        learning layer must see results in the order they became knowable, or
        the online updates are quietly using information from the future.
        """
        dec = {d.id: d for d in self.decisions()}
        fil = {f.decision_id: f for f in self.fills()}
        out: list[ClosedTrade] = []
        for o in self.outcomes():
            d = dec.get(o.decision_id)
            if d is None:
                continue
            out.append(ClosedTrade(d, fil.get(o.decision_id), o))
        out.sort(key=lambda t: t.outcome.ts)
        return out

    def open_positions(self) -> list[DecisionRecord]:
        """Traded decisions with no outcome recorded yet."""
        closed = {o.decision_id for o in self.outcomes()}
        return [d for d in self.decisions() if d.traded and d.id not in closed]

    # ---- summary ---------------------------------------------------------
    def gate_block_distribution(self) -> dict[str, int]:
        """How often each gate was the binding constraint.

        BLUEPRINT.md section 13.4 asks for this in the backtest report, and it
        is the single most informative summary of a system that mostly HOLDs:
        it separates "no opportunities existed" from "one threshold is set so
        tight that nothing can ever clear it".
        """
        counts: dict[str, int] = {}
        for d in self.decisions():
            for name in d.blocked_by:
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def summary(self) -> dict:
        decs = self.decisions()
        closed = self.closed_trades()
        traded = [d for d in decs if d.traded]
        pnl = np.array([t.pnl for t in closed], float)
        return {
            "path": str(self.path),
            "n_decisions": len(decs),
            "n_traded": len(traded),
            "n_hold": len(decs) - len(traded),
            "hold_rate": (len(decs) - len(traded)) / len(decs) if decs else np.nan,
            "n_closed": len(closed),
            "total_pnl": float(pnl.sum()) if pnl.size else 0.0,
            "hit_rate": float(np.mean(pnl > 0)) if pnl.size else np.nan,
            "gate_blocks": self.gate_block_distribution(),
        }

    def compact(self) -> None:
        """Rewrite the file with duplicate ids removed, last write winning.

        Only correct thing to do about a genuine duplicate (a re-run that
        re-journaled a decision).  Written to a temp file and atomically
        renamed, so an interrupted compaction cannot destroy the journal.
        """
        seen: dict[tuple[str, str], dict] = {}
        order: list[tuple[str, str]] = []
        for r in self.rows():
            key = (r.get("kind", ""), r.get("id", ""))
            if key not in seen:
                order.append(key)
            seen[key] = r
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".jsonl")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                for key in order:
                    fh.write(json.dumps(seen[key], separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise


def _as(cls, row: dict):
    """Build a record from a row, tolerating fields added by later versions.

    Forward compatibility matters here specifically: the journal outlives the
    code that wrote it, and a schema change must not make the historical record
    unreadable -- that would destroy the only dataset the backtest can use.

    Also undoes the NaN -> null round trip.  JSON has no NaN, so
    :func:`jsonable` writes null, and a float field read back as ``None`` breaks
    every downstream ``np.isfinite`` -- which is exactly what the learning layer
    does to decide whether an observation is usable.  Restoring NaN here, at the
    single read boundary, keeps "missing" spelled the same way everywhere in the
    system instead of leaving each consumer to guess.
    """
    spec = cls.__dataclass_fields__
    out = {}
    for k, v in row.items():
        f = spec.get(k)
        if f is None:
            continue
        if v is None and "float" in str(f.type):
            v = np.nan
        out[k] = v
    return cls(**out)
