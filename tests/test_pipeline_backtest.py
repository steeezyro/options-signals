"""Phases 9, 12 and 14: the runner, the enumerator, replay and the backtester."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest

from optionsmarkets.app.config import RunConfig
from optionsmarkets.app.pipeline import Pipeline
from optionsmarkets.backtest import Backtester, BacktestConfig, FillModel
from optionsmarkets.backtest.metrics import (
    deflated_sharpe, drawdown_profile, performance,
)
from optionsmarkets.data.journal import OutcomeJournal
from optionsmarkets.data.replay import LookAheadError, ReplayProvider
from optionsmarkets.data.synthetic import (
    SyntheticProvider, SyntheticSpec, record_synthetic_history,
)
from optionsmarkets.domain.candidates import ChainIndex, CandidateConfig, enumerate_candidates
from optionsmarkets.domain.structures import MULTIPLIER, Right, Side, vertical
from optionsmarkets.policy.decide import Action

ASOF = datetime(2026, 8, 17, 18, 0, tzinfo=timezone.utc)


def _cfg(tmp_path, **kw):
    kw.setdefault("earnings_unknown_action", "ignore")
    kw.setdefault("max_risk_fraction_per_trade", 0.035)
    return RunConfig(symbol="SPY", bankroll=40_000.0,
                     journal_path=tmp_path / "o.jsonl", **kw)


# ============================================================ phase 9: pipeline

class TestPipelineHappyPath:
    def test_end_to_end_reaches_a_sized_decision_with_a_ticket(self, tmp_path):
        res = Pipeline(SyntheticProvider(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.action is Action.SELL, res.decision.report()
        assert res.decision.contracts >= 1
        assert all(g.passed for g in res.decision.gates)
        assert res.ticket is not None
        assert res.ticket.api_json["orderStrategyType"] == "SINGLE"

    def test_the_surface_it_priced_off_is_arbitrage_free(self, tmp_path):
        res = Pipeline(SyntheticProvider(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.surface.ok
        assert res.surface.diagnostics.min_g_overall > 0
        assert res.surface.diagnostics.max_crossedness <= 1e-10
        assert len(res.surface.slices) >= 2, "need a term structure for the calendar gate"

    def test_vendor_iv_sentinels_are_detected_and_never_consumed(self, tmp_path):
        res = Pipeline(SyntheticProvider(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.quality.n_sentinel_iv > 0, "fixture should contain 0.500005 sentinels"
        # An ATM implied recovered near the 50-vol sentinel would mean they leaked in.
        assert res.vrp["sigma_implied_atm"] < 0.30

    def test_direction_follows_the_sign_of_the_variance_risk_premium(self, tmp_path):
        """Rich implied -> SELL. Cheap implied -> BUY. Never a view on the tape."""
        rich = SyntheticProvider(spec=SyntheticSpec(atm_vol_30d=0.20, realised_vol=0.12),
                                 asof=ASOF)
        r1 = Pipeline(rich, _cfg(tmp_path)).run(now=ASOF)
        assert r1.vrp["vrp_vol_points"] > 0
        assert r1.action in (Action.SELL, Action.HOLD)
        if r1.action is Action.SELL:
            assert any(lg.side is Side.SELL for lg in r1.structure.legs)

        cheap = SyntheticProvider(spec=SyntheticSpec(atm_vol_30d=0.09, realised_vol=0.30),
                                  asof=ASOF)
        r2 = Pipeline(cheap, _cfg(tmp_path / "b")).run(now=ASOF)
        assert r2.vrp["vrp_vol_points"] < 0
        assert r2.action in (Action.BUY, Action.HOLD)

    def test_every_decision_is_journaled_including_the_metadata_to_replay_it(self, tmp_path):
        cfg = _cfg(tmp_path)
        res = Pipeline(SyntheticProvider(asof=ASOF), cfg).run(now=ASOF)
        rec = OutcomeJournal(cfg.journal_path).decisions()[-1]
        assert rec.id == res.record.id
        for field in ("quality", "forward", "surface", "forecast", "vrp", "edge",
                      "sizing", "gates", "thresholds", "learned_state"):
            assert getattr(rec, field), f"{field} missing from the journal record"
        assert rec.candidates, "the whole frontier must be recorded, not just the winner"


class TestPipelineRefusals:
    """Each stage must refuse with a NAMED gate rather than degrade silently."""

    def test_a_dead_feed_holds_on_data_feed(self, tmp_path):
        class Dead(SyntheticProvider):
            def option_chain(self, symbol, expiry=None):
                raise RuntimeError("feed down")
        res = Pipeline(Dead(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "data.feed" in res.decision.blocked_by

    def test_an_unquotable_chain_holds_on_data_quality_and_still_journals_it(self, tmp_path):
        """The snapshot that fails the screen is the one you most want recorded."""
        class Closed(SyntheticProvider):
            def option_chain(self, symbol, expiry=None):
                snap = super().option_chain(symbol, expiry)
                for exp, rows in snap.quotes.items():
                    snap.quotes[exp] = [type(q)(q.strike, q.right, q.expiry, 0.0, 0.0,
                                                q.last, q.volume, q.open_interest,
                                                0.500005) for q in rows]
                return snap
        cfg = _cfg(tmp_path)
        res = Pipeline(Closed(asof=ASOF), cfg).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "data.quality" in res.decision.blocked_by
        assert res.quality is not None and not res.quality.usable
        assert OutcomeJournal(cfg.journal_path).decisions()[-1].quality

    def test_missing_earnings_fails_closed_by_default(self, tmp_path):
        class NoCal(SyntheticProvider):
            def next_earnings(self, symbol):
                raise RuntimeError("calendar unavailable")
        cfg = _cfg(tmp_path, earnings_unknown_action="block")
        res = Pipeline(NoCal(asof=ASOF), cfg).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "risk.event_window" in res.decision.blocked_by

    def test_earnings_inside_the_holding_window_blocks(self, tmp_path):
        """The dangerous case is not earnings TODAY, it is earnings INSIDE the trade.

        The policy's own risk.event_window gate measures days from today, so a
        32-DTE trade with earnings in 30 days passes it. That expiry's implied
        vol carries an event premium, so its VRP against a diffusive forecast is
        overstated whatever the exit plan says.
        """
        prov = SyntheticProvider(asof=ASOF, earnings_in_days=30)
        cfg = _cfg(tmp_path, target_dte=(28, 45))
        res = Pipeline(prov, cfg).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "risk.event_in_window" in res.decision.blocked_by

    def test_earnings_beyond_the_expiry_does_not_block(self, tmp_path):
        prov = SyntheticProvider(asof=ASOF, earnings_in_days=200)
        res = Pipeline(prov, _cfg(tmp_path)).run(now=ASOF)
        assert "risk.event_in_window" not in res.decision.blocked_by

    def test_too_little_history_holds_rather_than_guessing_a_forecast(self, tmp_path):
        class Short(SyntheticProvider):
            def daily_bars(self, symbol, start, end):
                return super().daily_bars(symbol, start, end).tail(30)
        res = Pipeline(Short(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "data.history" in res.decision.blocked_by


class TestExpirySelection:
    def test_default_prefers_the_far_end_of_the_band(self, tmp_path):
        """Entering at the 21-DTE time stop earns no premium and pays full costs."""
        res = Pipeline(SyntheticProvider(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.dte >= 28, f"entered at {res.dte} DTE, too close to the time stop"

    def test_nearest_preference_is_still_available(self, tmp_path):
        cfg = _cfg(tmp_path, expiry_preference="nearest", min_holding_days=1)
        near = Pipeline(SyntheticProvider(asof=ASOF), cfg).run(now=ASOF)
        far = Pipeline(SyntheticProvider(asof=ASOF), _cfg(tmp_path / "f")).run(now=ASOF)
        assert near.dte < far.dte

    def test_an_expiry_clear_of_earnings_is_preferred_over_one_that_straddles_it(
            self, tmp_path):
        """Selection avoids the event; it does not excuse it.

        Observed live 2026-08-18: MU had earnings in 36 days, 'furthest' picked
        the 38-DTE expiry, and the run HELD on risk.event_in_window -- while the
        24- and 31-DTE expiries in the same band expired cleanly before the
        event. Every name that reports would have been unreachable for the weeks
        around its own earnings, which is most names most of the time.
        """
        prov = SyntheticProvider(asof=ASOF, earnings_in_days=40)
        cfg = _cfg(tmp_path, target_dte=(21, 60), earnings_unknown_action="block")
        res = Pipeline(prov, cfg).run(now=ASOF)
        assert res.dte == 32, (
            f"picked {res.dte} DTE; the 60-DTE expiry straddles earnings at day 40 "
            f"and the 32-DTE one does not")
        assert "risk.event_in_window" not in res.decision.blocked_by

    def test_when_every_in_band_expiry_straddles_earnings_the_gate_still_fires(
            self, tmp_path):
        """The selection rule must not become a way around the gate.  With no
        clean expiry to fall back to, the refusal is the whole point."""
        prov = SyntheticProvider(asof=ASOF, earnings_in_days=24)
        cfg = _cfg(tmp_path, target_dte=(28, 45), earnings_unknown_action="block")
        res = Pipeline(prov, cfg).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "risk.event_in_window" in res.decision.blocked_by

    def test_a_listed_but_unquoted_expiry_is_not_selected_over_live_ones(self, tmp_path):
        """Being listed is not being quoted.

        MA on 2026-08-18 listed a 2026-10-02 expiry carrying ZERO strikes with a
        two-sided market on both legs. 'furthest' selected it over three live
        expiries in the same band and the run HELD at data.iv_inversion for want
        of a slice -- a refusal caused by the selection, not by the market.
        """
        import dataclasses

        class DeadFarExpiry(SyntheticProvider):
            def option_chain(self, symbol, expiry=None):
                snap = super().option_chain(symbol, expiry)
                dead = max(e for e in snap.expiries
                           if 21 <= (e - self.asof.date()).days <= 45)
                quotes = dict(snap.quotes)
                quotes[dead] = [dataclasses.replace(q, bid=0.0, ask=0.0)
                                for q in quotes[dead]]
                return dataclasses.replace(snap, quotes=quotes)

        # min_holding_days is relaxed only to isolate SELECTION: at 25 DTE the
        # default leaves 4 days above the 21-DTE time stop, so risk.holding_window
        # would refuse first and mask what this test is about.
        cfg = _cfg(tmp_path, min_holding_days=1)
        res = Pipeline(DeadFarExpiry(asof=ASOF), cfg).run(now=ASOF)
        assert res.dte == 25, f"selected {res.dte} DTE; the 32-DTE expiry has no market"
        assert "data.iv_inversion" not in res.decision.blocked_by

    def test_no_room_above_the_time_stop_is_refused_by_name(self, tmp_path):
        """A valid band, but the LISTING offers nothing far enough out."""
        class NearOnly(SyntheticProvider):
            def listed_expiries(self):
                # Only expiries at 22-24 DTE exist: inside the (21, 45) band but
                # with under 7 days of room above the 21-DTE time stop.
                return [self.asof.date() + timedelta(days=d) for d in (22, 23, 24)]
        res = Pipeline(NearOnly(asof=ASOF), _cfg(tmp_path)).run(now=ASOF)
        assert res.action is Action.HOLD
        assert "risk.holding_window" in res.decision.blocked_by

    def test_a_config_that_can_never_trade_is_rejected_at_construction(self, tmp_path):
        with pytest.raises(ValueError, match="no room above"):
            RunConfig(target_dte=(10, 20), manage_dte=21, min_holding_days=7)


# ========================================================== phase 14: candidates

class TestEnumerator:
    def _setup(self, tmp_path):
        prov = SyntheticProvider(asof=ASOF)
        res = Pipeline(prov, _cfg(tmp_path)).run(now=ASOF)
        chain = prov.option_chain("SPY")
        idx = ChainIndex.from_quotes(
            [q for e in chain.expiries for q in chain.for_expiry(e)])
        return prov, res, chain, idx

    def test_emits_condors_and_calendars_not_only_verticals(self, tmp_path):
        prov, res, chain, idx = self._setup(tmp_path)
        exps = sorted(e for e in chain.expiries if e > res.expiry)
        cands = enumerate_candidates(
            "SPY", idx, res.surface, expiry=res.expiry, T=res.dte / 365.0,
            F=res.surface.forward(res.dte / 365.0), S=chain.spot,
            r=0.0421, q=0.0118, vrp_vol_points=3.0,
            config=CandidateConfig(enable_butterflies=True, enable_diagonals=True),
            far_expiry=exps[0] if exps else None,
            T_far=(exps[0] - ASOF.date()).days / 365.0 if exps else None)
        names = " ".join(c.name for c in cands)
        assert "vert" in names and "iron condor" in names and "calendar" in names
        assert len(cands) > 10

    def test_every_leg_of_every_candidate_has_a_real_two_sided_quote(self, tmp_path):
        prov, res, chain, idx = self._setup(tmp_path)
        cands = enumerate_candidates(
            "SPY", idx, res.surface, expiry=res.expiry, T=res.dte / 365.0,
            F=res.surface.forward(res.dte / 365.0), S=chain.spot,
            r=0.0421, q=0.0118, vrp_vol_points=3.0)
        assert cands
        for st in cands:
            for lg in st.legs:
                assert lg.quote is not None
                assert lg.quote.bid > 0 and lg.quote.ask > lg.quote.bid
            assert np.isfinite(st.net_price("marketable"))
            assert st.max_loss(st.net_price("marketable")) > 0

    def test_credit_structures_when_rich_and_debit_when_cheap(self, tmp_path):
        prov, res, chain, idx = self._setup(tmp_path)
        args = dict(expiry=res.expiry, T=res.dte / 365.0,
                    F=res.surface.forward(res.dte / 365.0), S=chain.spot,
                    r=0.0421, q=0.0118,
                    config=CandidateConfig(enable_iron_condors=False,
                                           enable_calendars=False))
        rich = enumerate_candidates("SPY", idx, res.surface, vrp_vol_points=+4.0, **args)
        cheap = enumerate_candidates("SPY", idx, res.surface, vrp_vol_points=-4.0, **args)
        assert all("credit" in c.name for c in rich)
        assert all("debit" in c.name for c in cheap)

    def test_delta_targeting_uses_the_surface_not_a_flat_vol(self, tmp_path):
        """On a skewed slice the true 16-delta put sits further out than a
        flat-ATM-vol calculation says. Getting this wrong builds a spread closer
        to the money than the risk gates were told about."""
        from optionsmarkets.domain.candidates import strike_at_delta
        prov, res, chain, idx = self._setup(tmp_path)
        T, F = res.dte / 365.0, res.surface.forward(res.dte / 365.0)
        k_surface = strike_at_delta(res.surface, T, F, chain.spot, 0.0421, 0.0118,
                                   Right.PUT, 0.16)

        class Flat:
            def vol(self, k, T_):
                return np.full_like(np.asarray(k, float), res.surface.atm_vol(T_))
        k_flat = strike_at_delta(Flat(), T, F, chain.spot, 0.0421, 0.0118, Right.PUT, 0.16)
        assert k_surface < k_flat, "a -rho slice puts the 16d put further OTM"


# ============================================================= phase 12: replay

class TestReplayNoLookAhead:
    @pytest.fixture(scope="class")
    @staticmethod
    def recorded(tmp_path_factory):
        d = tmp_path_factory.mktemp("snaps")
        record_synthetic_history(d, days=40,
                                 start=datetime(2026, 3, 2, 20, 0, tzinfo=timezone.utc))
        return d

    def test_bars_are_truncated_at_the_cursor(self, recorded):
        rp = ReplayProvider(recorded)
        tl = rp.timeline()
        for i in (2, 10, len(tl) - 1):
            rp.seek(tl[i])
            bars = rp.daily_bars("SPY", date(2000, 1, 1), date(2030, 1, 1))
            assert bars.index.max() <= tl[i].date()

    def test_the_recorded_history_is_one_coherent_path(self, recorded):
        """Day k's bars must be a PREFIX of day k+n's, as real history is.

        A fixture that regenerates its past each session gives every day an
        unrelated history, so the realised volatility a trade is scored against
        belongs to a different world from the forecast that justified it.
        """
        rp = ReplayProvider(recorded)
        tl = rp.timeline()
        a = rp.seek(tl[3]).daily_bars("SPY", date(2000, 1, 1), date(2030, 1, 1))
        b = rp.seek(tl[12]).daily_bars("SPY", date(2000, 1, 1), date(2030, 1, 1))
        common = a.index.intersection(b.index)
        assert len(common) > 100
        assert np.allclose(a.loc[common, "close"], b.loc[common, "close"])

    def test_a_mixed_directory_never_serves_another_underlyings_data(self, tmp_path):
        """One directory holding two tickers must not silently cross them.

        ``RecordingProvider`` names files ``{stamp}_{method}.json`` with no
        underlying in the name, so ``--snapshots journal/snapshots`` collecting
        several tickers interleaves them in one directory.  Serving the newest
        recording regardless of who asked is a silent wrong answer: it does not
        raise, it backtests a different instrument.  Observed 2026-08-18 on real
        recordings -- ``option_chain("SPY")`` returned IWM at spot 300.67
        instead of SPY at 768.09, a 2.5x error, with no diagnostic.
        """
        d = tmp_path / "mixed"
        record_synthetic_history(d, days=6, symbol="SPY",
                                 start=datetime(2026, 3, 2, 20, 0, tzinfo=timezone.utc))
        # IWM written strictly LATER, so an unfiltered read serves IWM for SPY.
        record_synthetic_history(d, days=6, symbol="IWM",
                                 start=datetime(2026, 3, 10, 20, 0, tzinfo=timezone.utc))
        rp = ReplayProvider(d)
        assert rp.symbols() == {"SPY", "IWM"}
        rp.seek(rp.timeline()[-1])
        assert rp.option_chain("SPY").underlying == "SPY"
        assert rp.option_chain("IWM").underlying == "IWM"
        # A ticker with no recordings must refuse rather than substitute.
        with pytest.raises(LookAheadError, match="would silently backtest"):
            rp.option_chain("QQQ")

    def test_the_chain_spot_agrees_with_the_bar_history(self, recorded):
        rp = ReplayProvider(recorded)
        tl = rp.timeline()
        rp.seek(tl[15])
        bars = rp.daily_bars("SPY", date(2000, 1, 1), date(2030, 1, 1))
        assert rp.option_chain("SPY").spot == pytest.approx(float(bars["close"].iloc[-1]))

    def test_expiries_persist_across_sessions(self, recorded):
        """Rolling expiries make every open position unmarkable and unclosable."""
        rp = ReplayProvider(recorded)
        tl = rp.timeline()
        a = set(rp.seek(tl[3]).option_chain("SPY").expiries)
        b = set(rp.seek(tl[8]).option_chain("SPY").expiries)
        assert len(a & b) >= 3

    def test_a_cursor_before_the_first_record_raises_rather_than_inventing_data(self, recorded):
        rp = ReplayProvider(recorded)
        rp.seek(datetime(2020, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(LookAheadError):
            rp.daily_bars("SPY", date(2019, 1, 1), date(2020, 1, 1))

    def test_pipeline_runs_unchanged_against_the_replay(self, recorded, tmp_path):
        rp = ReplayProvider(recorded)
        tl = rp.timeline()
        rp.seek(tl[20])
        res = Pipeline(rp, _cfg(tmp_path)).run(now=tl[20])
        assert res.action in (Action.SELL, Action.BUY, Action.HOLD)
        assert res.record is not None


# =========================================================== phase 12: P&L rules

class TestFillAndPnlConventions:
    def _vert(self):
        E = date(2026, 9, 18)
        from optionsmarkets.domain.structures import OptionQuote
        ql = OptionQuote(730, Right.PUT, E, 0.95, 1.00, open_interest=100)
        qs = OptionQuote(745, Right.PUT, E, 3.84, 3.90, open_interest=100)
        return vertical("SPY", E, Right.PUT, 730, 745, q_long=ql, q_short=qs)

    def test_pnl_is_exit_minus_entry_for_a_credit_spread(self):
        """The classic sign bug prints every short-premium backtest backwards."""
        from optionsmarkets.backtest.engine import OpenPosition
        st = self._vert()
        entry = st.net_price("marketable")
        assert entry == pytest.approx(-284.0)
        pos = OpenPosition("d", st, 1, entry, 1.31, ASOF, 1216.0, 284.0, date(2026, 9, 18))
        assert pos.pnl_per_spread(0.0) == pytest.approx(+284.0)      # expires worthless
        assert pos.pnl_per_spread(-1500.0) == pytest.approx(-1216.0)  # pinned at max loss
        assert pos.pnl_per_spread(-1500.0) == pytest.approx(-st.max_loss(entry))

    def test_expiry_settlement_matches_the_marketable_mark(self):
        from optionsmarkets.backtest.engine import _intrinsic_net
        st = self._vert()
        # Spot at 730: the 745 put is worth 15, the 730 put worthless.
        assert _intrinsic_net(st, 730.0) == pytest.approx(-1500.0)
        assert _intrinsic_net(st, 800.0) == pytest.approx(0.0)

    def test_the_fill_penalty_is_always_worse_for_the_trader(self):
        fm = FillModel(extra_ticks_per_leg=1.0)
        st = self._vert()
        marketable = st.net_price("marketable")
        _, got = fm.fill(st, opening=True)
        assert got > marketable, "a credit must shrink, never grow"
        assert got == pytest.approx(marketable + 2 * 0.01 * MULTIPLIER)

    def test_fill_probability_declines_with_leg_count(self):
        fm = FillModel()
        assert fm.fill_probability(2) > fm.fill_probability(4) > fm.fill_probability(8)
        assert fm.fill_probability(8) >= 0.30


# =========================================================== phase 12: metrics

class TestMetrics:
    def test_drawdown_profile_finds_depth_and_recovery(self):
        eq = np.array([100, 110, 88, 95, 111, 120], float)
        p = drawdown_profile(eq)
        assert p["max_drawdown"] == pytest.approx(88 / 110 - 1.0)
        assert p["trough_index"] == 2 and p["recovery_index"] == 4

    def test_unrecovered_drawdown_reports_infinite_recovery(self):
        p = drawdown_profile(np.array([100, 120, 80, 85], float))
        assert p["periods_to_recover"] == np.inf

    def test_deflating_for_multiple_testing_lowers_the_verdict(self):
        """A Sharpe chosen from 50 configurations is not the same evidence as one."""
        rng = np.random.default_rng(4)
        r = rng.normal(0.05, 0.30, 200)
        one = deflated_sharpe(r, 1)
        many = deflated_sharpe(r, 50)
        assert many["sr0"] > one["sr0"]
        assert many["dsr"] < one["dsr"] <= one["psr"] + 1e-12

    def test_negative_skew_and_fat_tails_weaken_the_same_nominal_sharpe(self):
        """Exactly the shape short-premium returns have."""
        rng = np.random.default_rng(5)
        sym = rng.normal(0.05, 0.2, 400)
        skewed = np.where(rng.random(400) < 0.9, 0.09, -0.31)   # wall of wins, thin tail
        a = deflated_sharpe(sym, 10)
        b = deflated_sharpe(skewed, 10)
        assert b["skew"] < a["skew"]
        assert np.isfinite(b["dsr"])

    def test_performance_reports_log_growth_as_the_objective(self):
        # 252 equity points = 251 daily returns spanning 251/252 of a year, so a
        # per-day log growth of g annualises to 252*g. Dividing by 252 periods
        # instead of 251 returns is an off-by-one that understates growth.
        eq = list(100.0 * np.cumprod(1 + np.full(252, 0.0005)))
        s = performance([0.01] * 20, eq, periods_per_year=252.0, n_trials=5)
        assert s["log_growth_annualised"] == pytest.approx(252 * np.log(1.0005), rel=1e-6)
        assert "sortino_per_trade" in s and "deflated_sharpe" in s


# ========================================================== phase 12: full replay

class TestBacktestEndToEnd:
    def test_replay_closes_the_loop_and_reports_the_binding_constraint(self, tmp_path):
        snaps = tmp_path / "snaps"
        record_synthetic_history(snaps, days=150,
                                 start=datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc))
        cfg = _cfg(tmp_path)
        rep = Backtester(ReplayProvider(snaps), cfg,
                         backtest=BacktestConfig(n_trials_for_deflation=20)).run()

        assert rep.decisions > 50
        assert rep.equity and len(rep.equity) == rep.decisions
        # The gate-block distribution is the point of the report.
        assert rep.gate_blocks
        assert "log_growth_annualised" in rep.stats
        assert rep.render()

        # The loop must actually have run: outcomes journaled and fed back.
        J = OutcomeJournal(cfg.journal_path)
        assert J.decisions()
        if rep.trades:
            assert rep.exit_reasons, "trades must close through the exit plan"
            for t in J.closed_trades():
                # Costs are charged on entry AND exit, so a position closed at
                # its entry price must lose money.
                assert t.fill is not None
                assert np.isfinite(t.fill.marketable_reference)

    def test_no_position_is_left_marked_at_a_price_nobody_traded(self, tmp_path):
        snaps = tmp_path / "snaps"
        record_synthetic_history(snaps, days=60,
                                 start=datetime(2026, 4, 1, 20, 0, tzinfo=timezone.utc))
        rep = Backtester(ReplayProvider(snaps), _cfg(tmp_path)).run()
        # Trade returns come only from CLOSED positions.
        assert len(rep.trade_returns) <= rep.trades
        assert all(np.isfinite(x) for x in rep.trade_returns)
