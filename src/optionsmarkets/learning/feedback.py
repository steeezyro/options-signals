"""Wiring the outcome journal back into the model.  BLUEPRINT.md section 8.6.

The learning primitives in :mod:`optionsmarkets.learning.online` are the parts
that converge.  This module is what actually feeds them, and it is the
difference between a system that has a learning layer and a system that learns.

    journal row                  ->  consumer                       ->  effect
    ---------------------------------------------------------------------------
    realised RV vs forecast      ->  Kalman TVP on HAR coefficients ->  forecast
                                     + adaptive conformal               + interval
    realised P&L vs EV           ->  out-of-sample Sharpe            ->  c_unc
    profitable? (binary)         ->  probability calibrator          ->  PoP, and
                                     + Brier reliability                the
                                                                        model.
                                                                        calibration
                                                                        gate
    fill price vs ticket limit   ->  slippage model                  ->  the
                                                                        marketable
                                                                        price
                                                                        assumption
    all of the above             ->  ADWIN / Page-Hinkley            ->  refit
                                                                        trigger +
                                                                        lookback

The self-correcting property is worth being explicit about.  ``c_unc`` is
``1/(1 + 1/(S^2 T))`` computed from the **realised out-of-sample** Sharpe and
the length of the live record.  A strategy that stops working shrinks its own
position size, and one that keeps working grows it, with nobody touching a
threshold.  That is the whole payoff for keeping an honest journal.

Two guards on that, both of which matter more than the formula:

**The realised Sharpe is shrunk toward the prior.** With 8 closed trades the
sample Sharpe is mostly noise, and feeding it raw to ``c_unc`` would swing size
by 3x on evidence that cannot support it.  The estimate is blended
``n/(n + n0)`` toward the configured prior, so early on the system sizes off its
prior and late on it sizes off its record, with a smooth handover.

**Nothing here can ever raise the size.**  Every learned quantity enters as a
*cap* alongside the existing caps, never as a multiplier that could exceed them.
A feedback loop with a path to increasing risk is a feedback loop that can run
away, and this one is allowed to run in the background against a live account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..data.journal import ClosedTrade, OutcomeJournal
from .online import (
    ADWIN, AdaptiveConformal, PageHinkley, ProbabilityCalibrator,
    TimeVaryingRegression, brier_decomposition,
)

__all__ = ["LearnedState", "FeedbackLoop", "FeedbackConfig"]


@dataclass
class FeedbackConfig:
    # Priors used until the live record is long enough to overrule them.
    prior_sharpe: float = 0.50
    prior_years: float = 2.0
    # Shrinkage half-weight: at n0 closed trades the realised estimate and the
    # prior carry equal weight.  30 is roughly where a Sharpe estimate stops
    # being dominated by its own standard error for this trade frequency.
    shrink_n0: float = 30.0
    # The record is not allowed to move sizing until it is long enough to mean
    # something. Both conditions are required, and the SPAN one is the load-
    # bearing half: annualising a per-trade Sharpe multiplies it by
    # sqrt(trades_per_year), so two trades eleven days apart annualise to
    # sqrt(~700) ~ 26x. Observed in a replay: two losing trades produced a
    # "realised Sharpe" of -2367, which drove c_unc to exactly 0, which sized
    # every position to 0 contracts -- permanently, because a system that
    # cannot trade can never gather the evidence that would let it size again.
    # A feedback loop with an absorbing state at zero is not a feedback loop.
    min_trades_for_sizing: int = 20
    min_span_years_for_sizing: float = 0.25
    min_trades_for_calibration: int = 20
    conformal_alpha: float = 0.10
    # Gibbs-Candes gamma. The top of the 0.005-0.05 range: this model has to
    # survive regime breaks without going stale for weeks (BLUEPRINT 8.3).
    conformal_gamma: float = 0.05
    conformal_window: int = 500
    kalman_q: float = 1e-4
    kalman_phi: float = 0.995
    adwin_delta: float = 0.002
    slippage_ph_delta: float = 0.02
    slippage_ph_threshold: float = 5.0
    trading_days_per_year: float = 252.0


@dataclass
class LearnedState:
    """What the pipeline reads back out.  Everything here is a CAP or a MEASURE,
    never a free multiplier -- see the module docstring."""
    n_closed: int = 0
    n_decisions: int = 0

    # -> sizing
    sharpe: float = 0.50
    years_of_evidence: float = 2.0
    sharpe_realised: float = np.nan
    sharpe_shrink_weight: float = 0.0

    # -> gates
    conformal_killed: bool = False
    conformal_alpha_t: float = 0.10
    conformal_coverage: float = np.nan
    conformal_halfwidth: float = np.nan
    calibration_rel: float = 0.0
    calibration_brier: float = np.nan
    calibration_mode: str | None = None

    # -> forecast
    forecast_bias_beta: list[float] = field(default_factory=list)
    forecast_multiplier: float = 1.0
    innovation_health: dict = field(default_factory=dict)

    # -> execution assumptions
    slippage_per_spread: float = 0.0
    slippage_drift: bool = False
    n_fills: int = 0

    # -> refit control
    drift_detected: bool = False
    refit_lookback: int = 0
    n_drift_events: int = 0

    detail: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def report(self) -> str:
        L = [f"LEARNED STATE  ({self.n_closed} closed trades / {self.n_decisions} decisions)",
             f"  sizing      Sharpe {self.sharpe:.3f} over {self.years_of_evidence:.2f}y "
             f"(realised {self.sharpe_realised:.3f}, weight "
             f"{self.sharpe_shrink_weight:.0%} on the record)",
             f"  conformal   alpha_t={self.conformal_alpha_t:.4f} coverage="
             f"{self.conformal_coverage if np.isfinite(self.conformal_coverage) else float('nan'):.1%}"
             f" halfwidth={self.conformal_halfwidth:.4f}"
             f"{'  KILLED' if self.conformal_killed else ''}",
             f"  calibration REL={self.calibration_rel:.4f} (gate 0.0200) "
             f"mode={self.calibration_mode}",
             f"  forecast    multiplier {self.forecast_multiplier:.3f}, "
             f"innovations {self.innovation_health.get('verdict', 'n/a')}",
             f"  execution   slippage ${self.slippage_per_spread:.2f}/spread over "
             f"{self.n_fills} fills{'  DRIFT' if self.slippage_drift else ''}",
             f"  drift       {self.n_drift_events} event(s); refit lookback "
             f"{self.refit_lookback}"]
        L += [f"  note        {d}" for d in self.detail]
        return "\n".join(L)


class FeedbackLoop:
    """Replays the journal through the online learners and reports the state.

    Stateless with respect to the caller: :meth:`ingest` rebuilds every learner
    from the journal in outcome order.  That costs a full pass per call and buys
    something worth far more -- the learned state is a pure function of the
    journal, so it is reproducible, and a backtest that replays the same journal
    prefix gets *exactly* the state the live system had at that point.  An
    incrementally-mutated learner cannot promise that, and the promise is the
    entire no-look-ahead argument.
    """

    def __init__(self, config: FeedbackConfig | None = None):
        self.cfg = config or FeedbackConfig()
        self.state = LearnedState(sharpe=self.cfg.prior_sharpe,
                                  years_of_evidence=self.cfg.prior_years)
        self.kalman: TimeVaryingRegression | None = None
        self.conformal = AdaptiveConformal(self.cfg.conformal_alpha,
                                           self.cfg.conformal_gamma,
                                           self.cfg.conformal_window)
        self.calibrator = ProbabilityCalibrator()
        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        self.slippage_ph = PageHinkley(delta=self.cfg.slippage_ph_delta,
                                       threshold=self.cfg.slippage_ph_threshold)

    # ------------------------------------------------------------------
    def ingest(self, journal: OutcomeJournal, *, up_to: str | None = None) -> LearnedState:
        """Rebuild the learned state from the journal.

        ``up_to`` truncates to outcomes recorded at or before that ISO
        timestamp.  This is the backtester's no-look-ahead lever: at each replay
        step it asks for the state as of that moment and gets exactly what the
        live system would have had.
        """
        trades = journal.closed_trades()
        if up_to is not None:
            trades = [t for t in trades if t.outcome.ts <= up_to]
        decisions = journal.decisions()
        if up_to is not None:
            decisions = [d for d in decisions if d.ts <= up_to]

        st = LearnedState(n_closed=len(trades), n_decisions=len(decisions))
        self._reset_learners()

        self._forecast_feedback(trades, st)
        self._sizing_feedback(trades, st)
        self._calibration_feedback(trades, st)
        self._execution_feedback(trades, st)
        self._drift_feedback(trades, st)

        self.state = st
        return st

    def _reset_learners(self) -> None:
        self.kalman = None
        self.conformal = AdaptiveConformal(self.cfg.conformal_alpha,
                                           self.cfg.conformal_gamma,
                                           self.cfg.conformal_window)
        self.calibrator = ProbabilityCalibrator()
        self.adwin = ADWIN(delta=self.cfg.adwin_delta)
        self.slippage_ph = PageHinkley(delta=self.cfg.slippage_ph_delta,
                                       threshold=self.cfg.slippage_ph_threshold)

    # ---- 1. realised RV vs forecast ----------------------------------
    def _forecast_feedback(self, trades: list[ClosedTrade], st: LearnedState) -> None:
        """Kalman update on the HAR coefficients, plus the conformal interval.

        Two paths, and which one runs depends on what the journal recorded:

          * If the decision stored the HAR design row (``forecast.har_x``), the
            Kalman updates the actual HAR coefficients -- the model itself gets
            time-varying parameters, which is what section 8.2 describes.
          * Otherwise it falls back to a Mincer-Zarnowitz recalibration,
            regressing ln(realised) on [1, ln(forecast)] with time-varying
            coefficients.  That cannot fix the model, but it can and does fix a
            drifting *bias* in it, and it needs only the two numbers every
            outcome record carries.

        Both are fitted in logs, matching the log-form HAR: the errors are much
        closer to homoskedastic there, which is the condition under which the
        conformal interval built on them means anything.
        """
        pairs = [(t.decision.forecast, t.outcome) for t in trades
                 if np.isfinite(t.outcome.realised_rv) and t.outcome.realised_rv > 0]
        if not pairs:
            st.detail.append("no realised-RV observations yet; forecast feedback idle")
            return

        use_har = all(isinstance(f.get("har_x"), list) and f["har_x"] for f, _ in pairs)
        k = len(pairs[0][0]["har_x"]) if use_har else 2
        if self.kalman is None or self.kalman.k != k:
            beta0 = None
            if use_har and isinstance(pairs[0][0].get("har_beta"), list):
                beta0 = np.asarray(pairs[0][0]["har_beta"], float)
            elif not use_har:
                beta0 = np.array([0.0, 1.0])       # unbiased forecast as the prior
            self.kalman = TimeVaryingRegression(
                k, q=self.cfg.kalman_q, R=1.0, phi=self.cfg.kalman_phi,
                prior_var=0.25, mu=beta0)
            if beta0 is not None:
                self.kalman.beta = np.asarray(beta0, float).copy()

        for f, o in pairs:
            y = float(np.log(max(o.realised_rv, 1e-8)))
            x = (np.asarray(f["har_x"], float) if use_har
                 else np.array([1.0, float(np.log(max(o.forecast_rv, 1e-8)))]))
            if not np.all(np.isfinite(x)):
                continue
            self.kalman.update(x, y)
            # Conformal on the LOG scale for the same homoskedasticity reason.
            self.conformal.update(float(np.log(max(o.forecast_rv, 1e-8))), y)

        st.forecast_bias_beta = [float(b) for b in self.kalman.beta]
        st.innovation_health = self.kalman.innovation_health()
        st.conformal_alpha_t = float(self.conformal.alpha_t)
        st.conformal_killed = bool(self.conformal.killed)
        st.conformal_coverage = float(
            1.0 - np.mean(self.conformal.errs)) if self.conformal.errs else np.nan
        qh = self.conformal.quantile()
        st.conformal_halfwidth = float(qh) if np.isfinite(qh) else np.nan

        # Turn the recalibration into a multiplicative adjustment on the level.
        # Applied at the CURRENT forecast, not as a global constant, because a
        # slope != 1 means the correction depends on where the forecast sits.
        if not use_har and len(st.forecast_bias_beta) == 2:
            b0, b1 = st.forecast_bias_beta
            last = float(np.log(max(pairs[-1][1].forecast_rv, 1e-8)))
            adj = float(np.exp(b0 + b1 * last - last))
            # Clamped hard: a recalibration multiplier outside [0.5, 2.0] is
            # a broken model, not a bias, and the conformal kill switch is the
            # correct response to that -- not a 4x scaling of the forecast.
            st.forecast_multiplier = float(np.clip(adj, 0.5, 2.0))
            if not np.isclose(st.forecast_multiplier, adj):
                st.detail.append(
                    f"forecast recalibration multiplier {adj:.2f} clamped to "
                    f"{st.forecast_multiplier:.2f} -- treat as a model-health warning")
        else:
            st.forecast_multiplier = 1.0

    # ---- 2. realised P&L vs EV -> c_unc -------------------------------
    def _sizing_feedback(self, trades: list[ClosedTrade], st: LearnedState) -> None:
        """Out-of-sample Sharpe and years of evidence, shrunk toward the prior."""
        st.sharpe = self.cfg.prior_sharpe
        st.years_of_evidence = self.cfg.prior_years
        rets = np.array([t.return_on_risk for t in trades], float)
        rets = rets[np.isfinite(rets)]
        span_years = _span_years(trades)

        if rets.size < self.cfg.min_trades_for_sizing or \
                span_years < self.cfg.min_span_years_for_sizing:
            st.detail.append(
                f"{rets.size} closed trade(s) over {span_years:.2f}y: below the "
                f"{self.cfg.min_trades_for_sizing}-trade / "
                f"{self.cfg.min_span_years_for_sizing:.2f}y minimum, so sizing stays on "
                f"the prior Sharpe {self.cfg.prior_sharpe:.2f}. Annualising a Sharpe "
                f"from a short span amplifies noise by sqrt(trades per year) and would "
                f"drive c_unc to zero on a handful of samples.")
            return

        per_year = rets.size / max(span_years, 1e-6)
        sd = float(np.std(rets, ddof=1))
        s_per_trade = float(np.mean(rets) / sd) if sd > 0 else 0.0
        s_ann = s_per_trade * np.sqrt(max(per_year, 1e-9))
        st.sharpe_realised = float(s_ann)

        wgt = rets.size / (rets.size + self.cfg.shrink_n0)
        st.sharpe_shrink_weight = float(wgt)
        blended = wgt * s_ann + (1.0 - wgt) * self.cfg.prior_sharpe
        # Never let the record RAISE the assumed Sharpe above the prior. The
        # loop is allowed to shrink size on bad evidence and not to grow it on
        # good evidence, because a run of wins is exactly when the Sharpe
        # estimate is most upward-biased and the temptation to size up is worst.
        st.sharpe = float(max(min(blended, self.cfg.prior_sharpe), 0.0))
        st.years_of_evidence = float(max(span_years, 1e-3))
        if blended > self.cfg.prior_sharpe:
            st.detail.append(
                f"realised Sharpe {s_ann:.2f} exceeds the prior "
                f"{self.cfg.prior_sharpe:.2f}; sizing stays on the prior by design")

    # ---- 3. profitable? -> calibration --------------------------------
    def _calibration_feedback(self, trades: list[ClosedTrade], st: LearnedState) -> None:
        """Fit the calibrator on (predicted PoP, actually profitable) and report REL."""
        p = np.array([t.pop_predicted for t in trades], float)
        y = np.array([1.0 if t.outcome.profitable else 0.0 for t in trades], float)
        m = np.isfinite(p) & np.isfinite(y)
        p, y = p[m], y[m]
        if p.size < self.cfg.min_trades_for_calibration:
            st.calibration_rel = 0.0
            st.detail.append(
                f"{p.size} outcomes: below {self.cfg.min_trades_for_calibration}, "
                f"the calibration gate is held open (REL on a handful of trades "
                f"is noise, and a noisy gate is worse than no gate)")
            return
        self.calibrator.fit(p, y)
        st.calibration_mode = self.calibrator.mode_
        bd = brier_decomposition(self.calibrator.transform(p), y)
        st.calibration_rel = float(bd["rel"])
        st.calibration_brier = float(bd["brier"])

    def calibrate(self, pop: float) -> float:
        """Map a raw model PoP through the fitted calibrator.

        Kelly is a function of the probability LEVEL, so this is applied before
        sizing, not merely reported.  Untouched until the calibrator has been
        fitted, which is the honest behaviour on a cold start.
        """
        if self.calibrator.mode_ is None:
            return float(pop)
        return float(np.ravel(self.calibrator.transform(np.array([float(pop)])))[0])

    # ---- 4. fills -> slippage -----------------------------------------
    def _execution_feedback(self, trades: list[ClosedTrade], st: LearnedState) -> None:
        """Realised slippage against the ticket limit, plus a drift alarm on it.

        The scoring layer assumes it crosses the spread and gets the marketable
        price.  If real fills are consistently worse than that, every EV in the
        system is overstated by the difference, and the EV-vs-cost gate is the
        first thing that stops protecting anything.  Page-Hinkley here rather
        than a rolling mean because the question is not "what is slippage now"
        but "has it stepped", and a step is what a liquidity regime change looks
        like from the inside.
        """
        sl = []
        for t in trades:
            if t.fill is None:
                continue
            # Against the MARKETABLE reference, not the ticket limit. The
            # limit starts at mid, so measuring against it re-charges the whole
            # mid-to-marketable spread that score_structure already paid -- see
            # FillRecord's docstring for the replay where that silently shut the
            # system down.
            s = t.fill.slippage_vs_marketable
            if np.isfinite(s):
                sl.append(float(s))
                if self.slippage_ph.update(abs(s)):
                    st.slippage_drift = True
        st.n_fills = len(sl)
        if sl:
            # Median, not mean: one bad fill in a fast market is not the
            # assumption you want baked into every future EV.
            st.slippage_per_spread = float(np.median(sl))
            if st.slippage_drift:
                st.detail.append(
                    "Page-Hinkley fired on slippage: the marketable-price "
                    "assumption has stepped; re-derive it before sizing up")

    # ---- 5. drift -----------------------------------------------------
    def _drift_feedback(self, trades: list[ClosedTrade], st: LearnedState) -> None:
        """ADWIN on the loss stream -> refit trigger, and the correct lookback.

        The loss stream is the squared log forecast error where an RV
        realisation exists, and the negative return-on-risk otherwise.  ADWIN's
        surviving window length is not a by-product: it IS the adaptive lookback
        to refit over, which is why the refit trigger and the refit window come
        from the same object.
        """
        for t in trades:
            o = t.outcome
            if np.isfinite(o.realised_rv) and o.realised_rv > 0 and \
                    np.isfinite(o.forecast_rv) and o.forecast_rv > 0:
                loss = float(np.log(o.realised_rv / o.forecast_rv) ** 2)
            else:
                r = t.return_on_risk
                loss = float(-r) if np.isfinite(r) else 0.0
            if self.adwin.update(loss):
                st.drift_detected = True
        st.n_drift_events = int(self.adwin.n_detections)
        st.refit_lookback = int(self.adwin.width)
        if st.drift_detected:
            st.detail.append(
                f"ADWIN detected drift; refit over the surviving window of "
                f"{st.refit_lookback} observations, not the full history")


def _span_years(trades: list[ClosedTrade]) -> float:
    """Calendar span of the closed record, in years.

    Calendar span rather than trading days on purpose: ``c_unc``'s ``T`` is the
    length of the evidence period, and a strategy that traded twice in a year
    has one year of evidence, not two days of it.
    """
    ts = []
    for t in trades:
        try:
            ts.append(datetime.fromisoformat(t.outcome.ts.replace("Z", "+00:00")))
        except (ValueError, AttributeError):
            continue
    if len(ts) < 2:
        return 0.0
    return max((max(ts) - min(ts)).total_seconds() / (365.25 * 86400.0), 0.0)
