"""Journal replay.  BLUEPRINT.md section 13.4 -- "the part everyone gets wrong".

The rules are non-negotiable and each is implemented here rather than trusted:

  * **Fill at the marketable price OR WORSE, never mid.**  :class:`FillModel`
    crosses the spread and then adds a leg-count-dependent penalty, and it can
    decline to fill at all.  Scoring at mid manufactures edge the fill takes
    straight back; on a 4-leg condor at retail spreads that is frequently the
    entire modelled edge.
  * **Model partial fills on multi-leg.**  A 4-leg structure is four separate
    books that must line up simultaneously.  The fill probability declines with
    leg count, and a decline is journaled as a missed trade rather than silently
    dropped -- otherwise the backtest reports the P&L of trades it never got.
  * **Charge the full Schwab cost model on entry AND exit.**
  * **No look-ahead.**  Enforced structurally by
    :class:`~optionsmarkets.data.replay.ReplayProvider`'s cursor and by
    re-deriving the learned state from the journal prefix at every step.  The
    surface, forecast and thresholds at decision time are the ones that existed
    at decision time, and that is checkable rather than aspirational.
  * **Report the distribution of the gate that blocked**, so you can see what
    the system is actually constrained by.
  * **Deflate the Sharpe** for the multiple testing you did while tuning.

The learning loop runs *inside* the replay.  Each closed trade is journaled and
the next decision's ``c_unc``, calibration gate and conformal state are rebuilt
from the journal prefix -- so a backtest measures the adaptive system, not a
frozen snapshot of it.  A backtest that freezes the learned state is testing a
different strategy from the one you would run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np

from ..app.config import RunConfig
from ..app.pipeline import Pipeline, PipelineResult
from ..data.journal import FillRecord, OutcomeJournal, OutcomeRecord
from ..data.provider import ChainSnapshot
from ..domain.structures import MULTIPLIER, Side, Structure
from ..execution.schwab import SchwabCosts
from ..forecast.realized import daily_variance_proxy
from ..learning.feedback import FeedbackLoop
from ..policy.decide import Action
from ..risk.portfolio import PortfolioRisk, PositionRisk
from ..surface.joint import SurfacePublisher
from .metrics import performance

__all__ = ["FillModel", "BacktestConfig", "BacktestReport", "Backtester", "OpenPosition"]


# ----------------------------------------------------------------------------
# fills
# ----------------------------------------------------------------------------

@dataclass
class FillModel:
    """Marketable-or-worse fills, with a leg-count penalty and a refusal rate.

    ``extra_ticks_per_leg`` is the honest admission that a composite limit does
    not execute at the sum of the individual naturals: each leg is a separate
    book, they are not all at their touch at the same instant, and the market
    maker prices the package for that.  One extra cent per leg is mild.

    ``fill_prob_*`` exists because a backtest that always fills is measuring a
    strategy with an execution desk attached.  Two legs at retail size on a
    liquid ETF fill essentially always; four legs on a wide condor do not.
    """
    extra_ticks_per_leg: float = 1.0        # $0.01 per leg, per spread
    tick: float = 0.01
    fill_prob_2_leg: float = 0.98
    fill_prob_4_leg: float = 0.88
    seed: int = 20260817

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def fill_probability(self, n_legs: int) -> float:
        if n_legs <= 2:
            return self.fill_prob_2_leg
        # Linear in leg count between the two anchors, floored: a 6-leg
        # structure is not twice as hard as a 4-leg one, it is worse.
        span = max(self.fill_prob_2_leg - self.fill_prob_4_leg, 0.0)
        return float(max(self.fill_prob_2_leg - span * (n_legs - 2) / 2.0, 0.30))

    def fill(self, structure: Structure, *, opening: bool) -> tuple[bool, float]:
        """Return (filled, net price per spread) with the sign of ``net_price``.

        Marketable is the starting point, never mid, and the penalty always
        moves the price AGAINST the trader: a debit is paid up, a credit is
        received down.  Getting that sign wrong is the single easiest way to
        build a backtest that improves when you add costs.
        """
        net = structure.net_price("marketable")
        if not np.isfinite(net):
            return False, np.nan
        n_legs = sum(lg.quantity for lg in structure.legs)
        penalty = self.extra_ticks_per_leg * self.tick * n_legs * MULTIPLIER
        # ADDING the penalty is correct for both signs and needs no branch: a
        # debit (net > 0) gets larger, so you pay more; a credit (net < 0) moves
        # toward zero, so you receive less. Both are worse for the trader.
        worse = net + penalty
        if self._rng.random() > self.fill_probability(len(structure.legs)):
            return False, float(worse)
        return True, float(worse)


# ----------------------------------------------------------------------------
# positions
# ----------------------------------------------------------------------------

@dataclass
class OpenPosition:
    """An open spread, with the entry economics needed to close it honestly.

    The sign convention throughout is ``Structure.net_price``'s: a NET DEBIT is
    positive (cash paid), a NET CREDIT is negative (cash received).  Closing
    crosses the spread the other way, so the exit mark uses BUY legs at the bid
    and SELL legs at the ask, and then

        P&L per spread = exit_net - entry_net

    holds uniformly for debits and credits, with no branch.  Verified against a
    730/745 put credit vertical: entered at -284 and expiring worthless gives
    ``0 - (-284) = +284``; pinned at max loss gives ``-1500 - (-284) = -1216``,
    which is exactly ``Structure.max_loss``.  A conditional that flips the
    subtraction for credits is the classic version of this bug, and it makes
    every short-premium backtest print its P&L backwards.
    """
    decision_id: str
    structure: Structure
    contracts: int
    entry_net: float                 # per spread, signed like net_price
    entry_cost: float                # dollars, all contracts
    opened: datetime
    max_loss: float                  # per spread
    max_gain: float                  # per spread
    expiry: date
    forecast_rv: float = np.nan
    entry_spot: float = np.nan

    @property
    def credit(self) -> float:
        """Credit received per spread; negative for a debit structure."""
        return -self.entry_net

    def pnl_per_spread(self, exit_net: float) -> float:
        return float(exit_net - self.entry_net)

    def unrealised(self, exit_net: float) -> float:
        """Dollar P&L across all contracts at a mark, before exit costs."""
        return self.pnl_per_spread(exit_net) * self.contracts


# ----------------------------------------------------------------------------
# config / report
# ----------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    profit_target_pct: float = 0.50
    stop_multiple: float = 2.0
    manage_dte: int = 21
    # Honest count of configurations considered while tuning. The deflated
    # Sharpe is only meaningful if this is the real number, and the real number
    # is almost always larger than the one people report.
    n_trials_for_deflation: int = 1
    periods_per_year: float = 252.0
    max_concurrent_positions: int = 4


@dataclass
class BacktestReport:
    config: RunConfig
    steps: int
    decisions: int
    trades: int
    missed_fills: int
    equity: list[float] = field(default_factory=list)
    equity_dates: list[str] = field(default_factory=list)
    trade_returns: list[float] = field(default_factory=list)
    gate_blocks: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    exit_reasons: dict = field(default_factory=dict)
    prediction_vs_realised: dict = field(default_factory=dict)
    journal_path: str = ""
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        s = self.stats
        L = ["=" * 78, "BACKTEST (journal replay)", "=" * 78,
             f"  steps {self.steps} | decisions {self.decisions} | trades "
             f"{self.trades} | missed fills {self.missed_fills}"]
        if "log_growth_annualised" in s:
            L += [f"  ANNUALISED LOG GROWTH  {s['log_growth_annualised']:+.2%}   "
                  f"(the objective Kelly maximises)",
                  f"  total return           {s.get('total_return', float('nan')):+.2%}",
                  f"  vol of log wealth      "
                  f"{s.get('vol_of_log_wealth_annualised', float('nan')):.2%}",
                  f"  max drawdown           {s.get('equity_max_drawdown', float('nan')):.2%}"
                  f"  (recovered after "
                  f"{s.get('equity_periods_to_recover', float('nan'))} periods)"]
        if self.trades and len(self.trade_returns) < 10:
            L += ["", f"  !! {len(self.trade_returns)} closed trade(s). Every per-trade "
                      f"statistic below is dominated by its own sampling error and",
                  "     none of it is interpretable. BLUEPRINT.md section 13.5 asks for "
                  "60 trading days of decisions",
                  "     before comparing predicted to realised, and this is not that."]
        if "sharpe_per_trade" in s:
            L += [f"  Sharpe / trade         {s['sharpe_per_trade']:.3f}",
                  f"  Sortino / trade        {s['sortino_per_trade']:.3f}",
                  f"  hit rate               {s['hit_rate']:.1%}  "
                  f"(avg win {s['avg_win']:+.2%} vs avg loss {s['avg_loss']:+.2%}, "
                  f"ratio {s['win_loss_ratio']:.2f})",
                  f"  worst trade            {s['worst_trade']:+.2%} of risk"]
            d = s.get("deflated_sharpe", {})
            if d:
                L += [f"  DEFLATED SHARPE        DSR={d.get('dsr', float('nan')):.3f} "
                      f"(PSR={d.get('psr', float('nan')):.3f}, SR0="
                      f"{d.get('sr0', float('nan')):.3f} over "
                      f"{d.get('n_trials', 1)} trial(s))",
                      f"                         skew {d.get('skew', float('nan')):+.2f}, "
                      f"kurtosis {d.get('kurtosis', float('nan')):.2f}"]
        if self.exit_reasons:
            L.append("  exits: " + ", ".join(f"{k}={v}" for k, v in self.exit_reasons.items()))
        if self.prediction_vs_realised:
            p = self.prediction_vs_realised
            L += ["", "  PREDICTED vs REALISED:",
                  f"    PoP  stated {p.get('mean_pop', float('nan')):.1%} vs realised "
                  f"{p.get('realised_win_rate', float('nan')):.1%}",
                  f"    EV   stated ${p.get('mean_ev', float('nan')):,.2f} vs realised "
                  f"${p.get('mean_realised_pnl', float('nan')):,.2f} per position",
                  f"    RV   forecast {p.get('mean_forecast_rv', float('nan')):.2%} vs "
                  f"realised {p.get('mean_realised_rv', float('nan')):.2%}"]
        if self.gate_blocks:
            L += ["", "  WHAT ACTUALLY BLOCKED (the constraint you are really trading against):"]
            tot = sum(self.gate_blocks.values())
            for name, n in list(self.gate_blocks.items())[:12]:
                L.append(f"    {name:<28} {n:>5}  {n / max(tot, 1):>6.1%}")
        L += [f"  journal: {self.journal_path}"]
        L += [f"  note: {n}" for n in self.notes]
        L.append("=" * 78)
        return "\n".join(L)


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------

class Backtester:
    """Replay a recorded snapshot journal through the real pipeline.

    The pipeline object is the SAME class the live runner uses.  That is the
    whole design: a backtest that reimplements the decision logic is testing the
    reimplementation, and every difference between the two is a bug you will
    only find with money on it.
    """

    def __init__(self, provider, config: RunConfig | None = None, *,
                 backtest: BacktestConfig | None = None,
                 fills: FillModel | None = None,
                 costs: SchwabCosts | None = None):
        self.provider = provider
        self.cfg = config or RunConfig()
        self.bt = backtest or BacktestConfig()
        self.fills = fills or FillModel()
        self.costs = costs or SchwabCosts()

    def run(self, *, timeline: list[datetime] | None = None) -> BacktestReport:
        cfg = self.cfg
        journal = OutcomeJournal(cfg.journal_path)
        feedback = FeedbackLoop(cfg.feedback)
        # A fresh publisher per run: the last-known-good surface must not leak
        # across backtests, or run 2 starts with run 1's fallback in hand.
        pipeline = Pipeline(self.provider, cfg, journal=journal, feedback=feedback,
                            publisher=SurfacePublisher(), costs=self.costs)

        steps = timeline or self.provider.timeline()
        equity = cfg.bankroll
        rep = BacktestReport(config=cfg, steps=len(steps), decisions=0, trades=0,
                             missed_fills=0, journal_path=str(cfg.journal_path))
        open_pos: list[OpenPosition] = []
        preds: list[dict] = []

        for when in steps:
            self.provider.seek(when)
            try:
                chain = self.provider.option_chain(cfg.symbol)
            except Exception as exc:
                rep.notes.append(f"{when.isoformat()}: chain unavailable ({exc})")
                continue

            # 1. manage what is already on, BEFORE deciding anything new.
            equity, closed = self._manage(open_pos, chain, when, journal, equity, rep)
            preds.extend(closed)

            # 2. the learned state is rebuilt from the journal PREFIX -- exactly
            #    what the live system would have known at this instant.
            learned = feedback.ingest(journal, up_to=when.isoformat())

            # 3. decide, with the live book as the portfolio.
            portfolio = self._portfolio(open_pos, chain, equity)
            res = pipeline.run(portfolio=portfolio, now=when, learned=learned)
            rep.decisions += 1

            # 4. act.
            if res.action is not Action.HOLD and res.structure is not None:
                if len(open_pos) >= self.bt.max_concurrent_positions:
                    rep.notes.append(f"{when.isoformat()}: at the concurrent-position "
                                     f"cap; trade skipped")
                else:
                    equity = self._open(res, when, journal, equity, rep, open_pos)

            rep.equity.append(float(equity))
            rep.equity_dates.append(when.isoformat())

        # 5. anything still open at the end is marked, not counted as a win.
        if open_pos:
            rep.notes.append(f"{len(open_pos)} position(s) still open at the end of the "
                             f"replay; their P&L is excluded from the trade statistics "
                             f"rather than marked to a price nobody traded at")

        rep.trade_returns = [p["return_on_risk"] for p in preds
                             if np.isfinite(p.get("return_on_risk", np.nan))]
        rep.gate_blocks = journal.gate_block_distribution()
        rep.stats = performance(rep.trade_returns, rep.equity,
                                periods_per_year=self.bt.periods_per_year,
                                n_trials=self.bt.n_trials_for_deflation)
        rep.prediction_vs_realised = _prediction_vs_realised(journal)
        return rep

    # ---- opening ------------------------------------------------------
    def _open(self, res: PipelineResult, when: datetime, journal: OutcomeJournal,
              equity: float, rep: BacktestReport, open_pos: list[OpenPosition]) -> float:
        st, sz, ed = res.structure, res.sizing, res.edge
        filled, net = self.fills.fill(st, opening=True)
        limit = res.ticket.limit_price * MULTIPLIER * (1 if res.ticket.order_type
                                                       == "NET_DEBIT" else -1) \
            if res.ticket else st.net_price("mid")
        if not filled:
            rep.missed_fills += 1
            journal.append_fill(FillRecord(
                id=f"{res.record.id}-miss", decision_id=res.record.id,
                ts=when.isoformat(), contracts=0, limit_price=float(limit),
                fill_price=float(net),
                marketable_reference=float(st.net_price("marketable")),
                note="not filled: multi-leg composite did not cross"))
            return equity

        cost = self.costs.estimate(st, sz.contracts,
                                   credit_received=max(-net, 0.0) * sz.contracts)
        cash = -net * sz.contracts - cost["total_open"]
        journal.append_fill(FillRecord(
            id=f"{res.record.id}-fill", decision_id=res.record.id, ts=when.isoformat(),
            contracts=int(sz.contracts), limit_price=float(limit), fill_price=float(net),
            # The price score_structure was given. Slippage that may be fed back
            # into future EVs is measured against THIS, not against the limit.
            marketable_reference=float(st.net_price("marketable")),
            commission=float(cost["commission"]),
            fees=float(cost["total_open"] - cost["commission"])))
        open_pos.append(OpenPosition(
            decision_id=res.record.id, structure=st, contracts=int(sz.contracts),
            entry_net=float(net), entry_cost=float(cost["total_open"]), opened=when,
            max_loss=float(ed.max_loss), max_gain=float(ed.max_gain),
            expiry=min(lg.expiry for lg in st.legs),
            forecast_rv=float(res.forecast.get("sigma_forecast", np.nan)),
            entry_spot=float(res.spot)))
        rep.trades += 1
        return equity + cash

    # ---- managing -----------------------------------------------------
    def _manage(self, open_pos: list[OpenPosition], chain: ChainSnapshot,
                when: datetime, journal: OutcomeJournal, equity: float,
                rep: BacktestReport):
        """Apply the exit plan the ticket promised: target, stop, 21-DTE time stop.

        The time stop is unconditional, matching the ticket: gamma and
        assignment risk both accelerate inside three weeks and neither is part
        of the edge that justified the trade.  A backtest that quietly holds to
        expiry to capture the last of the theta is testing a different strategy
        from the one the ticket describes, and the difference shows up as
        exactly the tail the defined-risk constraint was meant to bound.
        """
        closed_out: list[dict] = []
        still: list[OpenPosition] = []
        for pos in open_pos:
            dte = (pos.expiry - when.date()).days
            mark, priced = _mark(pos.structure, chain)
            reason = None

            if dte <= 0:
                mark = _intrinsic_net(pos.structure, chain.spot)
                reason = "expiry"
            elif not priced:
                # No two-sided market on some leg: HOLD rather than invent a
                # price. Marking an unquotable leg to a model is how a backtest
                # books profits on positions it could not have exited.
                still.append(pos)
                continue
            else:
                pnl = pos.pnl_per_spread(mark)
                # The target is a fraction of MAX GAIN, not of the entry price.
                # For a credit spread those coincide; for a debit spread they do
                # not, and using the debit would turn a "50% profit target" into
                # a 50% loss.
                target = self.bt.profit_target_pct * pos.max_gain
                # The stop cannot exceed the contractual max loss, or it never
                # fires and the "stop" is decorative.
                stop = -min(self.bt.stop_multiple * abs(pos.entry_net), pos.max_loss)
                if pos.max_gain > 0 and pnl >= target:
                    reason = "profit_target"
                elif pnl <= stop:
                    reason = "stop"
                elif dte <= self.bt.manage_dte:
                    reason = "time_stop"

            if reason is None:
                still.append(pos)
                continue

            exit_cost = self.costs.estimate(pos.structure, pos.contracts)["total_open"]
            pnl_per_spread = pos.pnl_per_spread(mark)
            realised = pnl_per_spread * pos.contracts - pos.entry_cost - exit_cost
            # Equity already absorbed (-entry_net*n - entry_cost) when the
            # position opened, so closing adds the exit proceeds and the exit
            # cost only. The two together reduce to realised P&L exactly.
            equity += mark * pos.contracts - exit_cost

            days_held = max((when - pos.opened).days, 0)
            realised_rv = self._realised_rv(pos, when)
            journal.append_outcome(OutcomeRecord(
                id=f"{pos.decision_id}-out", decision_id=pos.decision_id,
                ts=when.isoformat(), exit_price=float(mark),
                realised_pnl=float(realised), contracts=pos.contracts,
                days_held=days_held, exit_reason=reason,
                realised_rv=float(realised_rv), forecast_rv=float(pos.forecast_rv),
                underlying_close=float(chain.spot), commission=float(exit_cost)))
            rep.exit_reasons[reason] = rep.exit_reasons.get(reason, 0) + 1
            risk = pos.max_loss * pos.contracts
            closed_out.append({
                "return_on_risk": float(realised / risk) if risk > 0 else np.nan,
                "realised_pnl": float(realised), "decision_id": pos.decision_id,
                "forecast_rv": float(pos.forecast_rv), "realised_rv": float(realised_rv),
            })
        open_pos[:] = still
        return equity, closed_out

    def _realised_rv(self, pos: OpenPosition, when: datetime) -> float:
        """Annualised realised vol over the holding period, from recorded bars.

        This is what the learning layer scores the forecast against, so it must
        come from the SAME recorded history the forecast was built on -- not
        from a fresh download, which would be look-ahead through the back door.
        """
        try:
            bars = self.provider.daily_bars(self.cfg.symbol, pos.opened.date(), when.date())
        except Exception:
            return np.nan
        if bars is None or len(bars) < 3:
            return np.nan
        v = daily_variance_proxy(bars).dropna()
        return float(np.sqrt(max(v.mean(), 0.0))) if v.size else np.nan

    def _portfolio(self, open_pos: list[OpenPosition], chain: ChainSnapshot,
                   equity: float) -> PortfolioRisk:
        """The live book, for the risk gates.

        Greeks are approximated as zero here rather than recomputed off a
        surface that has not been fitted yet at this step -- the pipeline fits
        the surface after it asks for the portfolio.  The capital-at-risk and
        per-underlying counts, which are the gates that actually bind for this
        strategy, ARE exact.  The approximation is recorded here so it is not
        mistaken for a measurement.
        """
        positions = [PositionRisk(chain.underlying, chain.spot, p.contracts,
                                  {"delta": 0.0, "vega": 0.0, "theta": 0.0, "gamma": 0.0},
                                  p.max_loss, max((p.expiry - chain.asof.date()).days, 0))
                     for p in open_pos]
        return PortfolioRisk(bankroll=equity, positions=positions)


# ----------------------------------------------------------------------------
# marking
# ----------------------------------------------------------------------------

def _mark(structure: Structure, chain: ChainSnapshot) -> tuple[float, bool]:
    """Mark a structure at the MARKETABLE exit price in the recorded chain.

    Exiting means crossing the spread the other way, so a long leg is sold at
    the bid and a short leg bought at the ask -- the mirror of the entry.
    Marking at mid on the way out is the second half of the same self-deception
    as marking at mid on the way in, and it is worth exactly as much.

    Returns (net, priced).  ``priced=False`` when any leg has no two-sided
    quote, in which case the caller must hold rather than invent a price.
    """
    tot = 0.0
    for lg in structure.legs:
        q = _find(chain, lg)
        if q is None or not (q.bid > 0 and q.ask > q.bid):
            return np.nan, False
        px = q.bid if lg.side is Side.BUY else q.ask
        tot += lg.signed_qty * px * MULTIPLIER
    return float(tot), True


def _find(chain: ChainSnapshot, leg):
    for q in chain.for_expiry(leg.expiry):
        if q.right is leg.right and abs(q.strike - leg.strike) < 1e-9:
            return q
    return None


def _intrinsic_net(structure: Structure, S_T: float) -> float:
    """Settlement value of the structure at expiry, as a net price.

    Cash-settled at intrinsic.  BLUEPRINT.md section 12 is clear that carrying
    to expiry is not the plan -- the 21-DTE time stop exists precisely so this
    path is rare -- but a position that reaches it must still be settled
    somehow, and intrinsic is the only defensible answer.
    """
    return float(sum(lg.payoff(np.array([float(S_T)]))[0]
                     for lg in structure.legs)) * MULTIPLIER


def _prediction_vs_realised(journal: OutcomeJournal) -> dict:
    """Every stated number against what actually happened.

    Read back out of the JOURNAL rather than accumulated in memory during the
    run.  That is deliberate: it proves the journal contains everything needed
    to score the system after the fact, which is precisely the claim BLUEPRINT.md
    section 8.6 makes and the claim paper trading (section 13.5) depends on.  If
    this function needed a field the journal does not carry, the journal would
    be incomplete and the live feedback loop would be too.
    """
    closed = journal.closed_trades()
    if not closed:
        return {}
    pnl = np.array([t.pnl for t in closed], float)
    pop = np.array([t.pop_predicted for t in closed], float)
    ev = np.array([t.ev_predicted for t in closed], float)
    frv = np.array([t.outcome.forecast_rv for t in closed], float)
    rrv = np.array([t.outcome.realised_rv for t in closed], float)
    m = np.isfinite(frv) & np.isfinite(rrv)
    out = {
        "n": len(closed),
        "realised_win_rate": float(np.mean(pnl > 0)),
        "mean_realised_pnl": float(np.mean(pnl)),
        "mean_pop": float(np.nanmean(pop)) if np.any(np.isfinite(pop)) else np.nan,
        "mean_ev": float(np.nanmean(ev)) if np.any(np.isfinite(ev)) else np.nan,
        # The gap that matters for sizing: a PoP that is stated 15 points above
        # what happens is an overbet no Kelly fraction repairs.
        "pop_error_pts": (float(np.nanmean(pop) - np.mean(pnl > 0)) * 100.0
                          if np.any(np.isfinite(pop)) else np.nan),
    }
    if m.any():
        out["mean_forecast_rv"] = float(np.mean(frv[m]))
        out["mean_realised_rv"] = float(np.mean(rrv[m]))
        # Mincer-Zarnowitz on the log forecast: slope 1 and intercept 0 is an
        # unbiased forecast. A slope well below 1 means the forecast overreacts.
        if m.sum() >= 5:
            x = np.log(np.maximum(frv[m], 1e-8))
            y = np.log(np.maximum(rrv[m], 1e-8))
            A = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            out["mincer_zarnowitz_intercept"] = float(coef[0])
            out["mincer_zarnowitz_slope"] = float(coef[1])
    return out
