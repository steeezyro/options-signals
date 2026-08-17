"""The runner: a live :class:`MarketDataProvider` in, a decision and a ticket out.

BLUEPRINT.md phase 9.  Everything the pipeline does was already implemented and
tested in isolation; what was missing was the thing that puts them in order,
against a real feed, and writes the result to the journal.  This is that.

The order is not arbitrary and each step refuses to run on output the previous
step flagged bad:

    1  chain snapshot                 -- and its REAL age, not a hoped-for one
    2  quality screen                 -- crossed books, sentinels, staleness
    3  rates / carry / events         -- per-expiry discounting, earnings window
    4  forward per expiry from parity -- never from the feed's spot
    5  in-house IV inversion          -- vendor IV discarded, rejects recorded
    6  surface: SSVI + slice refit    -- published only if arbitrage-free
    7  RV forecast, horizon-matched   -- with the learned recalibration applied
    8  variance risk premium          -- sets DIRECTION, not a view on the tape
    9  scenarios (Q -> P)             -- where the edge actually lives
    10 candidate enumeration          -- only structures with real quotes
    11 score / size / gate / rank     -- screen first, then rank on log growth
    12 ticket                         -- fully specified, or HOLD with reasons
    13 journal                        -- including the HOLDs, especially those

A failure in any of 1-6 produces a HOLD whose gate list says exactly which
stage refused, rather than a best-effort answer.  A system that degrades
silently is worse than one that stops, because from the P&L alone you cannot
tell a real signal from a data artefact until it is expensive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from ..data.journal import DecisionRecord, OutcomeJournal, new_decision_id
from ..data.provider import ChainSnapshot, assess_quality
from ..domain.candidates import ChainIndex, enumerate_candidates
from ..domain.structures import MULTIPLIER, Right, Structure
from ..edge.score import build_scenarios, score_structure, variance_risk_premium
from ..execution.schwab import SchwabCosts, build_ticket
from ..forecast.har import HARModel
from ..forecast.realized import daily_variance_proxy, yang_zhang
from ..learning.feedback import FeedbackLoop, LearnedState
from ..policy.decide import Action, Decision, Gate, rank_and_select
from ..pricing import black
from ..pricing.american import Dividend, de_americanise
from ..pricing.forward import ForwardFit, implied_forward
from ..pricing.implied import implied_vol_scalar
from ..risk.portfolio import PortfolioRisk, PositionRisk
from ..sizing.kelly import size_position
from ..surface.joint import (
    SliceQuotes, Surface, SurfacePublisher, SurfaceViolation, fit_surface,
    single_slice_surface,
)
from ..surface.svi import SliceFit, fit_svi_slice
from .config import RunConfig

__all__ = ["Pipeline", "PipelineResult", "PipelineAbort"]


class PipelineAbort(Exception):
    """A stage refused to hand its output downstream.  Carries the gate that said so.

    The gate travels with the exception so the HOLD that comes out the other end
    names the stage that refused, rather than reporting a generic failure. The
    ``notes`` list stays with the caller -- an abort is not the place to start
    accumulating a second copy of it.
    """

    def __init__(self, gate: Gate):
        super().__init__(gate.detail)
        self.gate = gate


@dataclass
class PipelineResult:
    symbol: str
    asof: datetime
    spot: float
    decision: Decision
    notes: list[str] = field(default_factory=list)
    quality: object | None = None
    forwards: dict = field(default_factory=dict)
    surface: Surface | None = None
    forecast: dict = field(default_factory=dict)
    vrp: dict = field(default_factory=dict)
    iv_rejections: dict = field(default_factory=dict)
    evaluated: list = field(default_factory=list)
    structure: Structure | None = None
    edge: object | None = None
    sizing: object | None = None
    ticket: object | None = None
    record: DecisionRecord | None = None
    learned: LearnedState | None = None
    expiry: date | None = None
    dte: int = 0

    @property
    def action(self) -> Action:
        return self.decision.action

    def report(self) -> str:
        L = [f"{self.symbol}  spot {self.spot:.2f}  asof {self.asof:%Y-%m-%d %H:%M:%S%z}"]
        if self.expiry:
            L.append(f"  decision expiry {self.expiry} ({self.dte} DTE)")
        if self.quality is not None:
            L.append(f"  feed: {self.quality.verdict} -- {self.quality.n_two_sided}"
                     f"/{self.quality.n_quotes} two-sided, "
                     f"{self.quality.n_sentinel_iv} sentinel IVs")
        if self.surface is not None:
            L.append("  " + self.surface.report().replace("\n", "\n  "))
        if self.forecast:
            L.append(f"  forecast: {self.forecast.get('sigma_forecast', float('nan')):.2%} "
                     f"({self.forecast.get('method', '?')}, horizon "
                     f"{self.forecast.get('horizon_days', 0)}d)")
        if self.vrp:
            L.append(f"  VRP: {self.vrp.get('vrp_vol_points', float('nan')):+.2f} vol pts "
                     f"-> {self.vrp.get('direction', '')}")
        L.append("  " + self.decision.report().replace("\n", "\n  "))
        L += [f"  note: {n}" for n in self.notes]
        return "\n".join(L)


class Pipeline:
    """Compose the library into one decision against a live provider.

    The provider is injected, so the same code path runs against the MCP
    adapters, an in-process yfinance client, or a recorded journal during a
    backtest.  Nothing in this class knows or cares which -- that is what makes
    the backtest in :mod:`optionsmarkets.backtest` a replay of the real system
    rather than a reimplementation of it.
    """

    def __init__(self, provider, config: RunConfig | None = None, *,
                 journal: OutcomeJournal | None = None,
                 feedback: FeedbackLoop | None = None,
                 publisher: SurfacePublisher | None = None,
                 costs: SchwabCosts | None = None):
        self.provider = provider
        self.cfg = config or RunConfig()
        self.journal = journal or OutcomeJournal(self.cfg.journal_path)
        self.feedback = feedback or FeedbackLoop(self.cfg.feedback)
        self.publisher = publisher or SurfacePublisher()
        self.costs = costs or SchwabCosts()

    # ==================================================================
    # entry point
    # ==================================================================
    def run(self, *, portfolio: PortfolioRisk | None = None,
            now: datetime | None = None,
            learned: LearnedState | None = None) -> PipelineResult:
        """One decision.  Never raises for a data problem -- returns HOLD instead.

        Exceptions are reserved for programming errors.  A feed outage, an
        unfittable surface or a chain with no quotable strikes are *expected
        states of the world* and must produce a journaled HOLD with a named
        gate, because those are the days the system most needs a record of.
        """
        cfg = self.cfg
        notes: list[str] = []
        now = now or datetime.now(timezone.utc)
        learned = learned if learned is not None else self._learned_state()
        portfolio = portfolio or PortfolioRisk(bankroll=cfg.bankroll)

        res = PipelineResult(symbol=cfg.symbol, asof=now, spot=float("nan"),
                             decision=_hold("not run", []), learned=learned)
        try:
            chain = self._chain(notes)
            res.asof, res.spot = chain.asof, chain.spot
            quote_age = chain.age_s(now)

            expiries, T_by_exp = self._select_expiries(chain, now, notes)
            decision_expiry = expiries[0]
            res.expiry = decision_expiry
            res.dte = (decision_expiry - now.date()).days

            quotes = [q for e in expiries for q in chain.for_expiry(e)]
            # Assigned BEFORE it is judged: a snapshot that fails the screen is
            # exactly the one whose diagnostics you want in the journal.
            res.quality = assess_quality(quotes, max_age_s=quote_age,
                                         min_two_sided=cfg.min_two_sided_quotes)
            if not res.quality.usable:
                raise PipelineAbort(Gate(
                    "data.quality", False,
                    f"{res.quality.verdict}: " + "; ".join(res.quality.detail)))

            rates, q_yield, divs, days_to_earnings = self._carry_and_events(
                chain, expiries, T_by_exp, notes)

            # The policy's risk.event_window gate measures days from TODAY, which
            # blocks entering ON an earnings date but not entering a 32-DTE trade
            # with earnings 30 days out -- the more dangerous case, and the one
            # that actually happens. Checked here because this is the first point
            # at which the holding window is known.
            #
            # The test is "earnings before this expiry", not "before the planned
            # exit", because the contamination is in the MEASUREMENT: an expiry
            # spanning an earnings date carries an event premium in its implied
            # vol, so comparing it against a diffusive RV forecast overstates the
            # variance risk premium no matter when you intend to close.
            if days_to_earnings is not None and 0 <= days_to_earnings <= res.dte:
                raise PipelineAbort(Gate(
                    "risk.event_in_window", False,
                    f"earnings in {days_to_earnings}d falls inside the "
                    f"{res.dte}-DTE expiry. That expiry's implied vol contains an "
                    f"event premium, so the VRP against a diffusive forecast is "
                    f"overstated -- the IV crush is a different trade with "
                    f"different math.", float(days_to_earnings)))

            slices, fwds, rejects = self._slices(
                chain, expiries, T_by_exp, rates, q_yield, divs, notes)
            res.forwards = {str(e): _fwd_dict(f) for e, f in fwds.items()}
            res.iv_rejections = rejects

            surface = self._surface(slices, chain, notes)
            res.surface = surface

            T = T_by_exp[decision_expiry]
            F = fwds[decision_expiry].F
            r = rates[decision_expiry]

            forecast = self._forecast(res.dte, learned, notes)
            res.forecast = forecast
            sigma_f = forecast["sigma_forecast"]

            iv_atm = surface.atm_vol(T)
            res.vrp = variance_risk_premium(iv_atm, sigma_f)
            res.vrp["sigma_implied_atm"] = float(iv_atm)
            res.vrp["sigma_forecast"] = float(sigma_f)

            scen = build_scenarios(surface.params_at(T), F, T, sigma_f)

            index = ChainIndex.from_quotes(
                quotes, max_rel_spread=cfg.chain_max_rel_spread,
                min_open_interest=cfg.chain_min_open_interest)
            far = expiries[1] if len(expiries) > 1 else None
            cands = enumerate_candidates(
                cfg.symbol, index, surface, expiry=decision_expiry, T=T, F=F,
                S=chain.spot, r=r, q=q_yield,
                vrp_vol_points=res.vrp["vrp_vol_points"], config=cfg.candidates,
                far_expiry=far, T_far=T_by_exp.get(far) if far else None,
                F_far=fwds[far].F if far else None)
            if not cands:
                raise PipelineAbort(Gate(
                    "data.candidates", False,
                    "no structure could be built from quotable strikes at this "
                    "expiry -- every candidate had at least one leg failing the "
                    "spread/open-interest screen"))

            evaluated, best = self._evaluate(
                cands, scen, surface, chain, portfolio, learned, T_by_exp,
                rates, q_yield, quote_age, res.dte, days_to_earnings, notes)
            res.evaluated = evaluated

            if best is None:
                st, ed, sz, dec = evaluated[0]
            else:
                st, ed, sz, dec = best
            res.structure, res.edge, res.sizing, res.decision = st, ed, sz, dec

            if dec.action is not Action.HOLD:
                res.ticket = build_ticket(
                    st, sz.contracts, chain.spot, duration=cfg.duration,
                    costs=self.costs, profit_target_pct=cfg.profit_target_pct,
                    stop_multiple=cfg.stop_multiple, manage_dte=cfg.manage_dte)

        except PipelineAbort as abort:
            res.decision = _hold(abort.gate.detail, [abort.gate])
        except SurfaceViolation as viol:
            notes.append(str(viol))
            res.surface = viol.surface
            res.decision = _hold(
                "surface failed the publication gate",
                [Gate("data.surface_arbfree", False, str(viol))])

        res.notes = notes
        res.record = self._journal(res, learned, portfolio)
        return res

    # ==================================================================
    # stages
    # ==================================================================
    def _learned_state(self) -> LearnedState:
        try:
            return self.feedback.ingest(self.journal)
        except Exception as exc:                                  # pragma: no cover
            st = LearnedState(sharpe=self.cfg.feedback.prior_sharpe,
                              years_of_evidence=self.cfg.feedback.prior_years)
            st.detail.append(f"feedback ingest failed ({exc}); running on priors")
            return st

    # ---- 1. chain ----------------------------------------------------
    def _chain(self, notes: list[str]) -> ChainSnapshot:
        try:
            chain = self.provider.option_chain(self.cfg.symbol)
        except Exception as exc:
            raise PipelineAbort(Gate("data.feed", False,
                                     f"option chain unavailable: {exc}")) from exc
        if not chain.quotes:
            raise PipelineAbort(Gate("data.feed", False, "chain snapshot is empty"))
        if not np.isfinite(chain.spot) or chain.spot <= 0:
            raise PipelineAbort(Gate(
                "data.feed", False,
                f"spot is {chain.spot!r} -- unusable. The forward is recovered from "
                f"parity, but the delta-targeted strike selection needs a spot"))
        return chain

    def _select_expiries(self, chain: ChainSnapshot, now: datetime,
                         notes: list[str]) -> tuple[list[date], dict[date, float]]:
        """Pick the decision expiry, plus neighbours for the surface.

        The FIRST returned expiry is the decision expiry; the rest exist so the
        surface has a term structure to enforce calendar-freedom against.  A
        single-slice surface cannot detect a calendar arbitrage at all, so
        fitting the neighbours is not optional polish -- it is what makes the
        arbitrage gate able to fire.

        Selection inside the band follows ``RunConfig.expiry_preference``, and
        the default of 'furthest' is load-bearing rather than cosmetic: with a
        21-DTE time stop, the nearest expiry in a (21, 45) band is entered and
        stopped out in the same week.  See the field's comment in
        :mod:`optionsmarkets.app.config` for the replay that established it.
        """
        lo_t, hi_t = self.cfg.target_dte
        lo_g, hi_g = self.cfg.thresholds.min_dte, self.cfg.thresholds.max_dte
        today = now.date()
        dte = {e: (e - today).days for e in chain.expiries}

        band = sorted([e for e, d in dte.items() if lo_t <= d <= hi_t], key=lambda e: dte[e])
        if not band:
            band = sorted([e for e, d in dte.items() if lo_g <= d <= hi_g],
                          key=lambda e: dte[e])
            if not band:
                raise PipelineAbort(Gate(
                    "risk.dte_window", False,
                    f"no listed expiry inside the {lo_g}-{hi_g} DTE window "
                    f"(available: {sorted(dte.values())[:8]}...)"))
            notes.append(f"no expiry in the {lo_t}-{hi_t} DTE target band; falling back "
                         f"to the wider {lo_g}-{hi_g} policy window")

        pick = self._prefer(band, dte)

        # The trade must have room to live. A structure entered at or just above
        # the time stop is closed before it has earned any of the premium that
        # justified it, so it is refused here with a named gate rather than
        # opened and immediately stopped out.
        room = dte[pick] - self.cfg.manage_dte
        if room < self.cfg.min_holding_days:
            raise PipelineAbort(Gate(
                "risk.holding_window", False,
                f"best available expiry is {dte[pick]} DTE, leaving {room} day(s) "
                f"above the {self.cfg.manage_dte}-DTE time stop (need "
                f"{self.cfg.min_holding_days}). Opening here buys full gamma and "
                f"delta for a few days and almost none of the variance premium.",
                float(room)))

        # Neighbours for the surface: nearest in maturity to the decision expiry.
        extra = sorted([e for e, d in dte.items() if d > 0 and e != pick],
                       key=lambda e: abs(dte[e] - dte[pick]))
        rest = sorted(extra[: max(self.cfg.max_expiries_for_surface - 1, 0)],
                      key=lambda e: dte[e])
        chosen = [pick] + rest
        T_by = {e: max(dte[e], 1) / 365.0 for e in chosen}
        return chosen, T_by

    def _prefer(self, band: list[date], dte: dict[date, int]) -> date:
        pref = self.cfg.expiry_preference
        if pref == "nearest":
            return band[0]
        if pref == "furthest":
            return band[-1]
        return band[len(band) // 2]                     # 'mid'

    # ---- 3. rates, carry, events -------------------------------------
    def _carry_and_events(self, chain: ChainSnapshot, expiries: list[date],
                          T_by: dict[date, float], notes: list[str]):
        cfg = self.cfg
        try:
            curve = self.provider.risk_free_curve()
        except Exception as exc:
            curve = {}
            notes.append(f"risk-free curve unavailable ({exc}); using the "
                         f"configured fallback {cfg.fallback_rate:.4%} flat. A flat "
                         f"rate is a small error at 30 DTE and a real one further out.")
        rates = {e: _interp_rate(curve, T_by[e], cfg.fallback_rate) for e in expiries}

        divs: list[Dividend] = []
        q_yield = cfg.fallback_dividend_yield
        try:
            end = max(expiries)
            df = self.provider.dividends(cfg.symbol, chain.asof.date() - timedelta(days=365), end)
            q_yield, divs = _dividends_to_carry(df, chain.spot, chain.asof.date(),
                                                cfg.fallback_dividend_yield, notes)
        except Exception as exc:
            notes.append(f"dividend feed unavailable ({exc}); using the configured "
                         f"yield {cfg.fallback_dividend_yield:.4%}")

        days_to_earnings: int | None = None
        try:
            nxt = self.provider.next_earnings(cfg.symbol)
            if nxt is not None:
                days_to_earnings = (nxt - chain.asof.date()).days
        except Exception as exc:
            if cfg.earnings_unknown_action == "block":
                raise PipelineAbort(Gate(
                    "risk.event_window", False,
                    f"earnings calendar unavailable ({exc}) and the policy is set "
                    f"to fail closed. An unknown event date is not the same as no "
                    f"event: the IV crush around earnings is a different trade with "
                    f"different math.")) from exc
            notes.append(f"earnings calendar unavailable ({exc}); event gate skipped "
                         f"by configuration")
        return rates, q_yield, divs, days_to_earnings

    # ---- 4/5. forward + IV inversion ---------------------------------
    def _slices(self, chain: ChainSnapshot, expiries: list[date],
                T_by: dict[date, float], rates: dict[date, float], q_yield: float,
                divs: list[Dividend], notes: list[str]):
        """Per expiry: forward from parity, then in-house IV on the OTM wing."""
        cfg = self.cfg
        slices: list[SliceQuotes] = []
        fwds: dict[date, ForwardFit] = {}
        rejects: dict[str, int] = {}

        for e in expiries:
            T = T_by[e]
            qs = chain.for_expiry(e)
            by_k: dict[float, dict] = {}
            for q in qs:
                by_k.setdefault(float(q.strike), {})[q.right] = q
            paired = sorted(k for k, v in by_k.items()
                            if len(v) == 2 and np.isfinite(v[Right.CALL].mid)
                            and np.isfinite(v[Right.PUT].mid))
            if len(paired) < 3:
                notes.append(f"{e}: only {len(paired)} paired strikes; slice skipped")
                continue

            ff = implied_forward(
                paired, [by_k[k][Right.CALL].mid for k in paired],
                [by_k[k][Right.PUT].mid for k in paired], T,
                weights=[1.0 / max(by_k[k][Right.CALL].spread
                                   + by_k[k][Right.PUT].spread, 1e-3) for k in paired],
                S_ref=chain.spot, atm_window=0.10)
            if not ff.ok:
                notes.append(f"{e}: put-call parity regression failed "
                             f"(F={ff.F}, DF={ff.DF}, n={ff.n_pairs}); slice skipped")
                continue
            fwds[e] = ff

            ks, ivs, sps = [], [], []
            for q in qs:
                # OTM wing only: the ITM quote carries the same information with
                # a worse signal-to-noise ratio, and inverting it invites the
                # deep-ITM cancellation failure the IV module rejects anyway.
                otm = (q.right is Right.CALL and q.strike >= ff.F) or \
                      (q.right is Right.PUT and q.strike < ff.F)
                if not otm or not q.tradeable(cfg.chain_max_rel_spread,
                                              cfg.chain_min_open_interest):
                    continue
                iv, spread_vols = self._invert(q, ff, T, rates[e], q_yield, divs,
                                               chain.spot, rejects)
                if iv is None:
                    continue
                ks.append(math.log(q.strike / ff.F))
                ivs.append(iv)
                sps.append(spread_vols)

            if len(ks) < 5:
                notes.append(f"{e}: only {len(ks)} strikes survived IV inversion; "
                             f"slice skipped")
                continue
            slices.append(SliceQuotes(T=T, k=np.array(ks), iv=np.array(ivs),
                                      iv_spread=np.array(sps), forward=ff.F,
                                      df=ff.DF, expiry=e))

        if not slices or expiries[0] not in fwds:
            raise PipelineAbort(Gate(
                "data.iv_inversion", False,
                f"no usable slice at the decision expiry. IV rejections: "
                f"{rejects or 'none'} -- vendor IV is never substituted."))
        return slices, fwds, rejects

    def _invert(self, q, ff: ForwardFit, T: float, r: float, q_yield: float,
                divs: list[Dividend], spot: float, rejects: dict[str, int]):
        """One quote -> one European IV, or a recorded rejection.  Never an imputation."""
        cfg = self.cfg
        if cfg.de_americanise:
            da = de_americanise(q.mid, spot, q.strike, T, r, q_yield, q.right.cp,
                                dividends=divs or None, n=cfg.de_americanise_steps)
            if not da.ok or not np.isfinite(da.sigma_european):
                rejects["DE_AM_FAILED"] = rejects.get("DE_AM_FAILED", 0) + 1
                return None, None
            sigma = float(da.sigma_european)
            # Spread in vol terms via vega at the recovered vol -- cheaper and
            # more stable than de-Americanising the bid and the ask separately.
            vega = float(black.greeks(spot, q.strike, T, r, q_yield, sigma, q.right.cp).vega)
            if vega <= 1e-10:
                rejects["NO_VEGA"] = rejects.get("NO_VEGA", 0) + 1
                return None, None
            half_vols = 0.5 * q.spread / vega
            if half_vols > cfg.max_spread_vols:
                rejects["NO_VEGA"] = rejects.get("NO_VEGA", 0) + 1
                return None, None
            return sigma, float(half_vols)

        res = implied_vol_scalar(q.mid, ff.F, q.strike, T, q.right.cp, ff.DF,
                                 spread=q.spread, max_spread_vols=cfg.max_spread_vols)
        if not res.ok:
            rejects[res.status] = rejects.get(res.status, 0) + 1
            return None, None
        lo = implied_vol_scalar(q.bid, ff.F, q.strike, T, q.right.cp, ff.DF)
        hi = implied_vol_scalar(q.ask, ff.F, q.strike, T, q.right.cp, ff.DF)
        half = 0.5 * abs(hi.sigma - lo.sigma) if (lo.ok and hi.ok) else 0.01
        return float(res.sigma), float(max(half, 1e-4))

    # ---- 6. surface ---------------------------------------------------
    def _surface(self, slices: list[SliceQuotes], chain: ChainSnapshot,
                 notes: list[str]) -> Surface:
        cfg = self.cfg
        asof = chain.asof.isoformat()
        if len(slices) == 1:
            sq = slices[0]
            fit: SliceFit = fit_svi_slice(sq.k, sq.iv, sq.T, iv_spread=sq.iv_spread)
            surf = single_slice_surface(fit, sq.T, sq.forward, chain.underlying,
                                        asof, sq.expiry)
            notes.append("only one usable expiry: the calendar-arbitrage gate "
                         "cannot fire on a single slice")
        else:
            surf = fit_surface(slices, underlying=chain.underlying, asof=asof,
                               envelope_band=cfg.envelope_band,
                               max_rmse_vol=cfg.surface_max_rmse_vol)
        published = self.publisher.publish(surf, raise_on_violation=True)
        if published.diagnostics.n_fallbacks:
            notes.append(f"{published.diagnostics.n_fallbacks} slice(s) fell back to "
                         f"the SSVI backbone: the local quotes could not be fitted "
                         f"without violating a no-arbitrage constraint")
        return published

    # ---- 7. forecast ---------------------------------------------------
    def _forecast(self, dte: int, learned: LearnedState, notes: list[str]) -> dict:
        """Horizon-matched realised-volatility forecast, in ANNUALISED vol.

        The horizon is the option's own DTE.  Forecasting 1-day RV to trade a
        30-day option is a mismatch no amount of model quality repairs, and it
        is the single easiest way to manufacture a variance risk premium that is
        not there.
        """
        cfg = self.cfg
        end = date.today()
        start = end - timedelta(days=cfg.history_days)
        try:
            bars = self.provider.daily_bars(cfg.symbol, start, end)
        except Exception as exc:
            raise PipelineAbort(Gate(
                "data.history", False,
                f"daily bars unavailable ({exc}); no realised-volatility forecast "
                f"is possible, so there is no variance risk premium to measure")) from exc
        if bars is None or len(bars) < 120:
            raise PipelineAbort(Gate(
                "data.history", False,
                f"only {0 if bars is None else len(bars)} daily bars; a 22-day HAR "
                f"needs a few hundred to estimate anything"))

        var_d = daily_variance_proxy(bars).dropna()
        if var_d.size < 120:
            raise PipelineAbort(Gate("data.history", False,
                                     f"only {var_d.size} usable daily RV observations"))

        horizon = max(int(dte), 1)
        model = HARModel(horizon=horizon, log_space=True, use_q=cfg.har_use_q,
                         include_today_in_week=cfg.har_include_today_in_week)
        fit = model.fit(var_d)
        pred_var = float(model.predict(var_d).dropna().iloc[-1])
        sigma = float(np.sqrt(max(pred_var, 1e-12)))

        # Discreteness correction on the LEVEL. Range estimators read 5-15% low
        # because the observed high/low understates the continuous range; the
        # default of 1.0 means "uncorrected, and the config says so".
        sigma *= float(cfg.rv_discreteness_correction)
        # The learned Mincer-Zarnowitz recalibration, clamped upstream.
        sigma_adj = sigma * float(learned.forecast_multiplier)

        X = model._design(var_d, None).dropna()
        har_x = [float(v) for v in X.iloc[-1].to_numpy(float)]
        yz = yang_zhang(bars).dropna()

        out = {
            "method": f"HAR-log h={horizon}d on a 1-day OHLC variance proxy",
            "horizon_days": horizon,
            "sigma_forecast": float(sigma_adj),
            "sigma_forecast_raw": float(sigma),
            "forecast_multiplier": float(learned.forecast_multiplier),
            "har_beta": [float(b) for b in fit.beta],
            "har_names": list(fit.names),
            "har_se": [float(s) for s in fit.se],
            "har_r2": float(fit.r2),
            "har_sigma2": float(fit.sigma2),
            "har_n": int(fit.n),
            "har_x": har_x,
            "n_bars": int(len(bars)),
            "rv_yang_zhang_21d": float(yz.iloc[-1]) if yz.size else float("nan"),
            "rv_last_daily_annualised": float(np.sqrt(max(var_d.iloc[-1], 0.0))),
        }
        if np.isfinite(learned.conformal_halfwidth):
            # The conformal band is maintained on the LOG scale, so it maps to a
            # multiplicative band on the level. Reported, not consumed by the
            # sizer directly -- the sizer's uncertainty enters through c_unc.
            hw = float(learned.conformal_halfwidth)
            out["interval_lo"] = float(sigma_adj * np.exp(-hw))
            out["interval_hi"] = float(sigma_adj * np.exp(hw))
            out["interval_coverage_target"] = 1.0 - self.cfg.feedback.conformal_alpha
        return out

    # ---- 10/11. score, size, gate, rank -------------------------------
    def _evaluate(self, cands: list[Structure], scen, surface: Surface,
                  chain: ChainSnapshot, portfolio: PortfolioRisk,
                  learned: LearnedState, T_by: dict[date, float],
                  rates: dict[date, float], q_yield: float, quote_age: float,
                  dte: int, days_to_earnings: int | None, notes: list[str]):
        cfg = self.cfg
        triples, meta = [], {}

        for st in cands:
            net = st.net_price("marketable")
            # Learned slippage makes the fill assumption worse, never better.
            # A measured improvement in fills is not evidence you will keep
            # getting them, and the asymmetry of the error says pay for it.
            net = float(net + abs(learned.slippage_per_spread))
            fee = self.costs.estimate(st, 1)["total_open"]
            ed = score_structure(st, scen, net, fees=fee)
            if not np.isfinite(ed.ev_per_spread) or ed.max_loss <= 0:
                continue

            sz = size_position(
                ed.payoff, scen.prob_P, ed.max_loss, cfg.bankroll,
                sharpe=learned.sharpe, years_of_evidence=learned.years_of_evidence,
                max_drawdown=cfg.max_drawdown, drawdown_prob=cfg.drawdown_prob,
                hard_cap_fraction=cfg.hard_cap_fraction,
                max_risk_fraction_per_trade=cfg.max_risk_fraction_per_trade,
                max_contracts=cfg.max_contracts)

            greeks = self._structure_greeks(st, surface, chain.spot, T_by, rates, q_yield)
            meta[id(st)] = {
                "net": net, "fee": fee, "greeks": greeks,
                "credit_to_width": _credit_to_width(st),
                "round_trip_cost": float(self.costs.estimate(st, 1)["total_round_trip_estimate"]),
            }
            triples.append((st, ed, sz))

        if not triples:
            raise PipelineAbort(Gate(
                "edge.scoring", False,
                "every candidate failed to score: net price or max loss was not "
                "computable from the quoted book"))

        def kwargs_for(st, ed, sz):
            m = meta[id(st)]
            pos = PositionRisk(chain.underlying, chain.spot, max(sz.contracts, 0),
                               m["greeks"], ed.max_loss, dte)
            pf = PortfolioRisk(cfg.bankroll, list(portfolio.positions) + [pos])
            return dict(
                quote_age_s=quote_age,
                slice_fit=_slice_fit_view(surface, T_by[st.legs[0].expiry]),
                legs_tradeable=[lg.quote is not None and lg.quote.tradeable(
                    cfg.thresholds.max_rel_spread, cfg.thresholds.min_open_interest)
                    for lg in st.legs],
                conformal_killed=learned.conformal_killed,
                calibration_rel=learned.calibration_rel,
                portfolio_delta_dollars=pf.delta_dollars,
                portfolio_vega_dollars=pf.vega_dollars,
                bankroll=cfg.bankroll, dte=dte,
                days_to_earnings=days_to_earnings,
                open_positions_same_underlying=portfolio.by_underlying().get(
                    chain.underlying, 0),
                credit_to_width=m["credit_to_width"],
                round_trip_cost=m["round_trip_cost"],
                thresholds=cfg.thresholds,
            )

        best_triple, best_dec, evaluated = rank_and_select(triples, kwargs_for)
        best = None
        if best_triple is not None:
            st, ed, sz = best_triple
            best = (st, ed, sz, best_dec)
        evaluated.sort(key=lambda x: (x[3].action is not Action.HOLD,
                                      x[2].expected_log_growth), reverse=True)
        return evaluated, best

    def _structure_greeks(self, st: Structure, surface: Surface, spot: float,
                          T_by: dict[date, float], rates: dict[date, float],
                          q_yield: float) -> dict:
        """Position Greeks with EACH LEG at its own surface volatility.

        Not a single flat vol: a vertical priced with one vol has zero vega by
        construction, which is exactly backwards -- the whole point of a spread
        is the differential exposure across the smile.
        """
        F_by = {e: surface.forward(T_by[e]) for e in T_by}
        sigmas, ts = [], []
        for lg in st.legs:
            T = T_by.get(lg.expiry, max((lg.expiry - date.today()).days, 1) / 365.0)
            F = F_by.get(lg.expiry, np.nan)
            if not np.isfinite(F) or F <= 0:
                F = float(black.forward(spot, rates.get(lg.expiry, 0.0), q_yield, T))
            sigmas.append(float(surface.vol(math.log(lg.strike / F), T)))
            ts.append(T)
        r0 = rates.get(st.legs[0].expiry, 0.0)
        return st.greeks(spot, r0, q_yield, sigmas, ts)

    # ---- 13. journal ---------------------------------------------------
    def _journal(self, res: PipelineResult, learned: LearnedState,
                 portfolio: PortfolioRisk) -> DecisionRecord | None:
        """Write the decision -- HOLD included.  Section 8.6.

        The HOLD rows are the more valuable half of the dataset: the
        distribution of which gate blocked is the only way to tell "no
        opportunity existed" from "one threshold is set so tight nothing can
        clear it", and section 13.4 requires that distribution in the report.
        """
        if not self.cfg.write_journal:
            return None
        d = res.decision
        rec = DecisionRecord(
            id=new_decision_id(self.cfg.symbol, res.asof),
            ts=datetime.now(timezone.utc).isoformat(),
            symbol=self.cfg.symbol, asof=res.asof.isoformat(),
            quote_age_s=float(res.quality.max_age_s) if res.quality else float("nan"),
            spot=float(res.spot), action=d.action.value, contracts=int(d.contracts),
            structure_name=d.structure_name,
            legs=[{"side": lg.side.value, "right": lg.right.value,
                   "strike": float(lg.strike), "expiry": lg.expiry.isoformat(),
                   "quantity": int(lg.quantity),
                   "bid": float(lg.quote.bid) if lg.quote else None,
                   "ask": float(lg.quote.ask) if lg.quote else None,
                   "open_interest": float(lg.quote.open_interest) if lg.quote else None}
                  for lg in (res.structure.legs if res.structure else [])],
            quality=(vars(res.quality) if res.quality else {}),
            forward=res.forwards, surface=(res.surface.as_dict() if res.surface else {}),
            iv_rejections=res.iv_rejections, forecast=res.forecast, vrp=res.vrp,
            edge=(res.edge.summary() | {
                "pop": float(res.edge.pop), "pop_riskneutral": float(res.edge.pop_riskneutral),
                "ev_per_spread": float(res.edge.ev_per_spread),
                "ev_pct_of_risk": float(res.edge.ev_pct_of_risk),
                "max_loss": float(res.edge.max_loss), "max_gain": float(res.edge.max_gain),
                "cvar_5": float(res.edge.cvar_5), "edge_z": float(res.edge.edge_z),
            } if res.edge else {}),
            sizing=(vars(res.sizing) if res.sizing else {}),
            gates=[{"name": g.name, "passed": bool(g.passed), "detail": g.detail,
                    "value": g.value} for g in d.gates],
            ticket=({"order_type": res.ticket.order_type,
                     "limit_price": res.ticket.limit_price,
                     "mid": res.ticket.mid, "natural": res.ticket.natural,
                     "price_ladder": res.ticket.price_ladder,
                     "cost_estimate": res.ticket.cost_estimate,
                     "api_json": res.ticket.api_json} if res.ticket else {}),
            thresholds=vars(self.cfg.thresholds),
            learned_state=learned.as_dict(),
            portfolio={"bankroll": portfolio.bankroll,
                       "delta_dollars": portfolio.delta_dollars,
                       "vega_dollars": portfolio.vega_dollars,
                       "capital_at_risk": portfolio.capital_at_risk,
                       "by_underlying": portfolio.by_underlying()},
            candidates=[{"name": st.name, "action": dd.action.value,
                         "contracts": int(sz.contracts),
                         "growth": float(sz.expected_log_growth),
                         "pop": float(ed.pop), "ev": float(ed.ev_per_spread),
                         "max_loss": float(ed.max_loss),
                         "blocked_by": dd.blocked_by}
                        for st, ed, sz, dd in res.evaluated],
            rationale=d.rationale,
        )
        try:
            self.journal.append_decision(rec)
        except Exception as exc:                                   # pragma: no cover
            res.notes.append(f"journal write failed: {exc}")
        return rec


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _hold(reason: str, gates: list[Gate]) -> Decision:
    return Decision(Action.HOLD, "", 0, gates, f"HOLD -- {reason}", {})


def _fwd_dict(ff: ForwardFit) -> dict:
    return {"F": ff.F, "DF": ff.DF, "r_implied": ff.r_implied, "n_pairs": ff.n_pairs,
            "r2": ff.r2, "residual_bp": ff.residual_bp}


def _credit_to_width(st: Structure) -> float:
    """Credit as a fraction of the widest vertical spread inside the structure.

    Measured at MID, deliberately: credit/width is a property of the STRIKES
    chosen, and loading it with slippage double-counts costs that the EV gate
    already charges in full.
    """
    ks = sorted({lg.strike for lg in st.legs})
    if len(ks) < 2:
        return float("nan")
    width = max(b - a for a, b in zip(ks, ks[1:]))
    mid = st.net_price("mid")
    if not np.isfinite(mid) or width <= 0:
        return float("nan")
    return float(abs(mid) / (width * MULTIPLIER))


class _SliceFitView:
    """Adapts a :class:`Surface` slice to what the policy's data gates read."""

    __slots__ = ("ok", "rmse_vol", "min_g", "detail")

    def __init__(self, ok, rmse_vol, min_g, detail):
        self.ok, self.rmse_vol, self.min_g, self.detail = ok, rmse_vol, min_g, detail


def _slice_fit_view(surface: Surface, T: float) -> _SliceFitView:
    s = surface.nearest_slice(T)
    return _SliceFitView(bool(surface.ok and s.ok), float(s.rmse_vol), float(s.min_g),
                         (s.detail or surface.detail or f"source={s.source}"))


def _interp_rate(curve: dict, T: float, fallback: float) -> float:
    """Continuously-compounded rate at tenor T, log-linear in maturity.

    A single flat rate across a term structure is a small error at 30 DTE and a
    real one at LEAPS tenors, and it shows up downstream as a spurious
    calendar-arbitrage signal rather than as an obviously wrong rate.
    """
    if not curve:
        return float(fallback)
    ts = np.array(sorted(curve), float)
    rs = np.array([curve[t] for t in sorted(curve)], float)
    m = np.isfinite(ts) & np.isfinite(rs)
    if m.sum() == 0:
        return float(fallback)
    return float(np.interp(float(T), ts[m], rs[m]))


def _dividends_to_carry(df, spot: float, asof: date, fallback: float,
                        notes: list[str]) -> tuple[float, list[Dividend]]:
    """Trailing yield for the continuous-carry path, plus discrete ex-dates.

    Both are produced because both are needed and they are not interchangeable:
    the European surface fit wants a smooth carry, while the American lattice
    needs the actual discrete cash amounts on their actual ex-dates.  Using the
    continuous yield inside the lattice is exactly the approximation that makes
    the early-exercise premium wrong.
    """
    if df is None or len(df) == 0:
        return float(fallback), []
    frame = pd.DataFrame(df)
    cols = {c.lower(): c for c in frame.columns}
    amt_col = next((cols[c] for c in ("cash_amount", "amount", "dividend", "value")
                    if c in cols), None)
    date_col = next((cols[c] for c in ("ex_dividend_date", "ex_date", "date")
                     if c in cols), None)
    if amt_col is None:
        notes.append("dividend rows carry no recognisable amount column; using fallback yield")
        return float(fallback), []

    amounts = pd.to_numeric(frame[amt_col], errors="coerce").dropna()
    trailing = float(amounts.sum())
    q = float(np.log1p(trailing / spot)) if spot > 0 and trailing > 0 else float(fallback)

    divs: list[Dividend] = []
    if date_col is not None:
        for _, row in frame.iterrows():
            try:
                ex = pd.to_datetime(row[date_col]).date()
                amt = float(row[amt_col])
            except (ValueError, TypeError):
                continue
            if ex > asof and np.isfinite(amt) and amt > 0:
                divs.append(Dividend(t=(ex - asof).days / 365.0, amount=amt))
    return q, sorted(divs, key=lambda d: d.t)
