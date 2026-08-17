"""Phase 10: the outcome journal and the feedback wiring that closes the loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from optionsmarkets.data.journal import (
    DecisionRecord, FillRecord, OutcomeJournal, OutcomeRecord, jsonable, new_decision_id,
)
from optionsmarkets.learning.feedback import FeedbackConfig, FeedbackLoop

T0 = datetime(2026, 1, 5, 20, 0, tzinfo=timezone.utc)


def _decision(i: int, *, action="SELL", pop=0.80, ev=100.0, risk=1000.0,
              forecast=0.14, gates=None, ts=None) -> DecisionRecord:
    ts = ts or (T0 + timedelta(days=i))
    return DecisionRecord(
        id=f"d{i:03d}", ts=ts.isoformat(), symbol="SPY", asof=ts.isoformat(),
        quote_age_s=900.0, spot=780.0, action=action, contracts=1 if action != "HOLD" else 0,
        structure_name="test vert",
        edge={"pop": pop, "ev_per_spread": ev, "max_loss": risk},
        sizing={"capital_at_risk": risk},
        forecast={"sigma_forecast": forecast,
                  "har_x": [1.0, np.log(0.02), np.log(0.02), np.log(0.02)],
                  "har_beta": [0.1, 0.4, 0.3, 0.2]},
        gates=gates or [{"name": "edge.z", "passed": True, "detail": "", "value": 1.0}],
    )


def _close(j: OutcomeJournal, i: int, *, pnl: float, realised=0.14,
           forecast=0.14, days=20, marketable=-300.0, fill=-298.0):
    j.append_fill(FillRecord(
        id=f"f{i:03d}", decision_id=f"d{i:03d}", ts=(T0 + timedelta(days=i)).isoformat(),
        contracts=1, limit_price=-350.0, fill_price=fill,
        marketable_reference=marketable))
    j.append_outcome(OutcomeRecord(
        id=f"o{i:03d}", decision_id=f"d{i:03d}",
        ts=(T0 + timedelta(days=i + days)).isoformat(),
        exit_price=-100.0, realised_pnl=pnl, contracts=1, days_held=days,
        exit_reason="time_stop", realised_rv=realised, forecast_rv=forecast))


class TestSerialisation:
    def test_nan_and_inf_survive_the_round_trip_as_nan(self, tmp_path):
        """JSON has no NaN. It must come back as NaN, not as None.

        A float field read back as None breaks every downstream np.isfinite,
        which is exactly how the learning layer decides an observation is
        usable -- and it crashed the backtester before this was fixed.
        """
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_outcome(OutcomeRecord(
            id="o1", decision_id="d1", ts=T0.isoformat(), exit_price=0.0,
            realised_pnl=10.0, contracts=1, days_held=1, exit_reason="x",
            realised_rv=np.nan, forecast_rv=np.inf))
        back = j.outcomes()[0]
        assert np.isnan(back.realised_rv)
        assert np.isnan(back.forecast_rv)
        assert np.isfinite(back.realised_pnl)

    def test_jsonable_handles_numpy_and_non_finite(self):
        out = jsonable({"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1.0, np.nan]),
                        "d": float("inf"), "e": np.bool_(True)})
        assert out == {"a": 1.5, "b": 3, "c": [1.0, None], "d": None, "e": True}

    def test_unknown_future_fields_do_not_break_the_read(self, tmp_path):
        """The journal outlives the code that wrote it."""
        p = tmp_path / "o.jsonl"
        p.write_text('{"kind":"outcome","id":"o1","decision_id":"d1","ts":"x",'
                     '"exit_price":0,"realised_pnl":5,"contracts":1,"days_held":1,'
                     '"exit_reason":"y","a_field_from_2027":42}\n')
        assert OutcomeJournal(p).outcomes()[0].realised_pnl == 5.0

    def test_corrupt_trailing_line_is_skipped_not_fatal(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(1))
        with open(j.path, "a") as fh:
            fh.write('{"kind":"decision","id":"trunc"')      # crash mid-write
        assert len(j.decisions()) == 1

    def test_decision_ids_are_unique_and_sortable(self):
        ids = [new_decision_id("SPY", T0) for _ in range(3)]
        assert len(set(ids)) == 3 and ids == sorted(ids)


class TestJournalQueries:
    def test_hold_decisions_are_journaled_and_their_blockers_counted(self, tmp_path):
        """The HOLD rows are the more valuable half of the dataset."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(3):
            j.append_decision(_decision(i, action="HOLD", gates=[
                {"name": "edge.z", "passed": False, "detail": "", "value": 0.0},
                {"name": "edge.vrp", "passed": i == 0, "detail": "", "value": 1.0}]))
        dist = j.gate_block_distribution()
        assert dist["edge.z"] == 3 and dist["edge.vrp"] == 2
        assert j.summary()["hold_rate"] == 1.0

    def test_closed_trades_are_ordered_by_when_the_outcome_became_knowable(self, tmp_path):
        """Not by decision time: online updates must not see the future."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(0))
        j.append_decision(_decision(1))
        _close(j, 1, pnl=10.0, days=1)      # decided later, closed FIRST
        _close(j, 0, pnl=20.0, days=40)
        got = [t.decision.id for t in j.closed_trades()]
        assert got == ["d001", "d000"]

    def test_open_positions_are_traded_decisions_without_outcomes(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(0))
        j.append_decision(_decision(1))
        j.append_decision(_decision(2, action="HOLD"))
        _close(j, 0, pnl=5.0)
        assert [d.id for d in j.open_positions()] == ["d001"]

    def test_compact_dedupes_keeping_the_last_write(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(0, ev=1.0))
        j.append_decision(_decision(0, ev=2.0))
        assert len(j.decisions()) == 2
        j.compact()
        assert len(j.decisions()) == 1
        assert j.decisions()[0].edge["ev_per_spread"] == 2.0


class TestSlippageFeedback:
    def test_slippage_is_measured_against_marketable_not_the_ticket_limit(self, tmp_path):
        """The bug that silently shut the system down.

        The ticket limit starts at MID. Measuring fill-vs-limit and feeding it
        back re-charges the whole mid-to-marketable spread that score_structure
        already paid, which drives every future EV negative.
        """
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(0))
        _close(j, 0, pnl=5.0, marketable=-300.0, fill=-298.0)
        f = j.fills()[0]
        assert f.slippage_per_spread == pytest.approx(52.0)      # vs the mid limit
        assert f.slippage_vs_marketable == pytest.approx(2.0)    # vs the assumption

        st = FeedbackLoop().ingest(j)
        assert st.slippage_per_spread == pytest.approx(2.0), \
            "feedback must consume the marketable-referenced number"

    def test_missing_reference_is_unknown_not_the_limit_based_number(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        j.append_decision(_decision(0))
        j.append_fill(FillRecord(id="f0", decision_id="d000", ts=T0.isoformat(),
                                 contracts=1, limit_price=-350.0, fill_price=-298.0))
        j.append_outcome(OutcomeRecord(
            id="o0", decision_id="d000", ts=T0.isoformat(), exit_price=0.0,
            realised_pnl=1.0, contracts=1, days_held=1, exit_reason="x"))
        st = FeedbackLoop().ingest(j)
        assert st.slippage_per_spread == 0.0 and st.n_fills == 0


class TestSizingFeedback:
    def test_a_short_record_cannot_move_sizing(self, tmp_path):
        """Two losing trades days apart must not zero out the Kelly multiplier.

        Annualising a per-trade Sharpe multiplies it by sqrt(trades per year), so
        a handful of trades over a few days annualises to a nonsense magnitude.
        Observed pre-fix: 2 trades -> 'realised Sharpe' -2367 -> c_unc exactly 0
        -> 0 contracts forever, and a system that cannot trade can never gather
        the evidence to size again.
        """
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(2):
            j.append_decision(_decision(i))
            _close(j, i, pnl=-500.0, days=1)
        st = FeedbackLoop().ingest(j)
        assert st.sharpe == pytest.approx(0.50)
        assert st.years_of_evidence == pytest.approx(2.0)
        from optionsmarkets.sizing.kelly import uncertainty_shrinkage
        assert uncertainty_shrinkage(st.sharpe, st.years_of_evidence) > 0.3

    def test_a_long_losing_record_does_shrink_sizing(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        cfg = FeedbackConfig(min_trades_for_sizing=20, min_span_years_for_sizing=0.25)
        for i in range(30):
            j.append_decision(_decision(i * 8))
            _close(j, i * 8, pnl=-200.0 if i % 3 else 100.0, days=5)
        st = FeedbackLoop(cfg).ingest(j)
        assert st.n_closed == 30
        assert st.sharpe < 0.50, "a losing record must reduce the assumed Sharpe"
        assert st.sharpe >= 0.0

    def test_a_winning_record_never_raises_sizing_above_the_prior(self, tmp_path):
        """A run of wins is when the Sharpe estimate is most upward-biased."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(30):
            j.append_decision(_decision(i * 8))
            _close(j, i * 8, pnl=400.0, days=5)
        st = FeedbackLoop().ingest(j)
        assert st.sharpe <= 0.50 + 1e-12
        assert any("stays on the prior" in d for d in st.detail)


class TestCalibrationAndForecastFeedback:
    def test_calibration_gate_is_held_open_below_the_minimum(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(5):
            j.append_decision(_decision(i))
            _close(j, i, pnl=-100.0)
        st = FeedbackLoop().ingest(j)
        assert st.calibration_rel == 0.0
        assert any("calibration gate is held open" in d for d in st.detail)

    def test_a_miscalibrated_pop_raises_reliability(self, tmp_path):
        """Stated 90% that happens 40% of the time must show up as REL."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        rng = np.random.default_rng(3)
        for i in range(60):
            j.append_decision(_decision(i, pop=0.90))
            win = rng.random() < 0.40
            _close(j, i, pnl=100.0 if win else -300.0)
        st = FeedbackLoop().ingest(j)
        assert st.n_closed == 60
        assert st.calibration_rel > 0.0

    def test_forecast_recalibration_is_clamped_and_flagged(self, tmp_path):
        """A recalibration multiplier outside [0.5, 2] is a broken model."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(30):
            d = _decision(i)
            d.forecast.pop("har_x")             # force the Mincer-Zarnowitz path
            j.append_decision(d)
            _close(j, i, pnl=10.0, forecast=0.10, realised=0.60)
        st = FeedbackLoop().ingest(j)
        assert 0.5 <= st.forecast_multiplier <= 2.0

    def test_conformal_and_drift_streams_are_populated(self, tmp_path):
        j = OutcomeJournal(tmp_path / "o.jsonl")
        rng = np.random.default_rng(5)
        for i in range(80):
            j.append_decision(_decision(i))
            _close(j, i, pnl=50.0, forecast=0.14,
                   realised=float(0.14 * np.exp(rng.normal(0, 0.25))))
        st = FeedbackLoop().ingest(j)
        assert 0.0 < st.conformal_coverage <= 1.0
        assert np.isfinite(st.conformal_halfwidth)
        assert st.refit_lookback > 0

    def test_state_is_a_pure_function_of_the_journal_prefix(self, tmp_path):
        """Reproducibility is the whole no-look-ahead argument."""
        j = OutcomeJournal(tmp_path / "o.jsonl")
        for i in range(40):
            j.append_decision(_decision(i))
            _close(j, i, pnl=100.0 if i % 2 else -80.0)
        cut = (T0 + timedelta(days=25)).isoformat()
        a = FeedbackLoop().ingest(j, up_to=cut)
        b = FeedbackLoop().ingest(j, up_to=cut)
        assert a.as_dict() == b.as_dict()
        full = FeedbackLoop().ingest(j)
        assert full.n_closed > a.n_closed, "truncation must actually truncate"
