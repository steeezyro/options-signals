"""Phase 13 (factor Kelly) and the section 6.2 / 13.3 forecast additions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optionsmarkets.forecast.evaluate import (
    compare_forecasts, diebold_mariano, interval_coverage, mincer_zarnowitz, qlike,
    score_forecast,
)
from optionsmarkets.forecast.garch import GJRGarch
from optionsmarkets.forecast.har import HARModel, SHARModel
from optionsmarkets.forecast.realized import (
    daily_variance_proxy, semivariance_proxy,
)
from optionsmarkets.risk.factors import (
    FactorModel, estimate_betas, portfolio_kelly,
)
from optionsmarkets.risk.portfolio import PortfolioRisk, PositionRisk


# ------------------------------------------------------------------ fixtures

def _gjr_ohlc(n=2200, om=3e-6, al=0.02, xi=0.14, be=0.88, steps=78, seed=9):
    """A GJR-driven path with OHLC built from intraday sub-paths."""
    rng = np.random.default_rng(seed)
    s2 = np.empty(n); rd = np.empty(n)
    O = np.empty(n); H = np.empty(n); L = np.empty(n); C = np.empty(n)
    s2[0] = om / (1 - al - xi / 2 - be); px = 600.0
    for t in range(n):
        if t:
            s2[t] = om + (al + xi * (rd[t - 1] < 0)) * rd[t - 1] ** 2 + be * s2[t - 1]
        sub = rng.normal(0, np.sqrt(s2[t] / steps), steps)
        path = px * np.exp(np.cumsum(sub))
        O[t], H[t], L[t], C[t] = path[0], path.max(), path.min(), path[-1]
        rd[t] = np.log(C[t] / px); px = C[t]
    bars = pd.DataFrame({"open": O, "high": H, "low": L, "close": C, "volume": 1.0},
                        index=pd.bdate_range("2016-01-04", periods=n).date)
    return bars, rd, s2


# ================================================================ phase 13

class TestBetaEstimation:
    def test_recovers_a_known_beta(self):
        rng = np.random.default_rng(1)
        idx = rng.normal(0, 0.01, 500)
        rets = {"AAA": 1.30 * idx + rng.normal(0, 0.002, 500)}
        e = estimate_betas(rets, idx)["AAA"]
        assert e.beta == pytest.approx(1.30, abs=0.05)
        assert e.identified and e.r2 > 0.8

    def test_an_unidentified_beta_defaults_to_one_the_conservative_direction(self):
        """Beta multiplies co-movement, so assuming 1.0 OVERSTATES concentration.

        Overstating concentration makes the book smaller; understating it is what
        blows the book up. The default must err toward the former.
        """
        rng = np.random.default_rng(2)
        idx = rng.normal(0, 0.01, 500)
        e = estimate_betas({"NOISE": rng.normal(0, 0.01, 20)}, idx)["NOISE"]
        assert not e.identified
        assert e.beta == 1.0
        assert "conservative" in e.detail or "defaulted" in e.detail

    def test_a_zero_beta_name_is_flagged_not_silently_netted(self):
        rng = np.random.default_rng(3)
        idx = rng.normal(0, 0.01, 400)
        e = estimate_betas({"INDEP": rng.normal(0, 0.01, 400)}, idx)["INDEP"]
        assert not e.identified and e.beta == 1.0


class TestFactorExposure:
    def test_a_same_signed_short_vol_book_collapses_to_about_one_bet(self):
        pf = PortfolioRisk(40_000.0, [
            PositionRisk(f"N{i}", 100.0, 1, {"delta": -0.1, "vega": -8.0}, 1000.0, 30)
            for i in range(5)])
        ex = FactorModel().exposure(pf)
        assert ex.concentration_ratio == pytest.approx(1.0, abs=1e-9)
        assert ex.effective_bets < 1.5, "five short-vol names are not five bets"
        assert ex.gross_vega_dollars == pytest.approx(40.0)
        assert ex.worst_case_simultaneous_loss == pytest.approx(5000.0)

    def test_offsetting_vega_lowers_the_factor_exposure_below_the_gross(self):
        pf = PortfolioRisk(40_000.0, [
            PositionRisk("A", 100.0, 1, {"delta": 0.0, "vega": -8.0}, 1000.0, 30),
            PositionRisk("B", 100.0, 1, {"delta": 0.0, "vega": +8.0}, 1000.0, 30)])
        ex = FactorModel().exposure(pf)
        assert abs(ex.market_vol_dollars) < 1e-9 < ex.gross_vega_dollars
        assert ex.concentration_ratio < 0.01

    def test_empty_book_is_reported_not_divided_by_zero(self):
        ex = FactorModel().exposure(PortfolioRisk(40_000.0, []))
        assert ex.effective_bets == 0.0 and "no open positions" in ex.detail[0]


class TestPortfolioKelly:
    def _credit_spread(self, rng, n):
        """+284 on 87% of scenarios, -1216 on 13%: a real 15-wide credit vertical."""
        return np.where(rng.random(n) > 0.13, 284.0, -1216.0)

    def test_identical_positions_reveal_an_n_fold_overbet(self):
        rng = np.random.default_rng(7)
        n = 4000
        one = self._credit_spread(rng, n)
        r = portfolio_kelly([one] * 5, np.ones(n) / n, [1216.0] * 5)
        assert r.effective_bets == pytest.approx(1.0, abs=0.05)
        assert r.overbet_multiple == pytest.approx(5.0, abs=0.3)
        # 2x Kelly earns ZERO growth; 5x is deep into negative territory.
        assert r.overbet_multiple > 2.0

    def test_independent_positions_show_no_overbet(self):
        rng = np.random.default_rng(8)
        n = 4000
        many = [self._credit_spread(rng, n) for _ in range(5)]
        r = portfolio_kelly(many, np.ones(n) / n, [1216.0] * 5)
        assert r.effective_bets > 4.0
        assert r.overbet_multiple < 1.35

    def test_joint_kelly_is_the_total_wealth_fraction(self):
        """Sanity: one position's joint answer is its own Kelly fraction."""
        from optionsmarkets.sizing.kelly import kelly_fraction
        rng = np.random.default_rng(9)
        n = 4000
        one = self._credit_spread(rng, n)
        p = np.ones(n) / n
        r = portfolio_kelly([one], p, [1216.0])
        assert r.f_joint == pytest.approx(kelly_fraction(one / 1216.0, p), rel=1e-9)

    def test_no_edge_gives_no_stake(self):
        n = 1000
        losing = np.full(n, -50.0)
        r = portfolio_kelly([losing], np.ones(n) / n, [1000.0])
        assert r.f_joint == 0.0

    def test_invalid_capital_is_refused(self):
        r = portfolio_kelly([np.ones(10)], np.ones(10) / 10, [0.0])
        assert r.f_joint == 0.0 and "invalid" in r.detail


# ================================================== section 6.2: GJR-GARCH

class TestGJRGarch:
    @pytest.fixture(scope="class")
    @staticmethod
    def fitted():
        rng = np.random.default_rng(11)
        n, om, al, xi, be = 4000, 2.0e-6, 0.02, 0.12, 0.90
        s2 = np.empty(n); r = np.empty(n)
        s2[0] = om / (1 - al - xi / 2 - be)
        for t in range(n):
            if t:
                s2[t] = om + (al + xi * (r[t - 1] < 0)) * r[t - 1] ** 2 + be * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.normal()
        m = GJRGarch("normal")
        m.fit(r)
        return m, (om, al, xi, be)

    def test_recovers_known_parameters(self, fitted):
        m, (om, al, xi, be) = fitted
        f = m.fit_
        assert f.omega == pytest.approx(om, rel=0.35)
        assert f.alpha == pytest.approx(al, abs=0.015)
        assert f.xi == pytest.approx(xi, abs=0.03)
        assert f.beta == pytest.approx(be, abs=0.02)

    def test_leverage_dominates_the_symmetric_arch_term(self, fitted):
        """BLUEPRINT 6.2: for an equity index a model without asymmetry is wrong."""
        m, _ = fitted
        assert m.fit_.leverage_dominates
        assert m.fit_.xi > 3 * m.fit_.alpha

    def test_forecast_persistence_uses_half_the_asymmetry_term(self, fitted):
        """alpha + xi/2 + beta, not alpha + xi + beta.

        The indicator fires half the time under a symmetric error distribution.
        Using the full xi overstates persistence and makes the model look
        near-unit-root when it is not.
        """
        m, _ = fitted
        f = m.fit_
        assert f.persistence == pytest.approx(f.alpha + 0.5 * f.xi + f.beta)
        assert f.persistence < f.alpha + f.xi + f.beta
        assert 0 < f.persistence < 1
        assert f.half_life_days > 1

    def test_standardised_residuals_have_unit_variance(self, fitted):
        m, _ = fitted
        assert float(np.var(m.standardised_residuals())) == pytest.approx(1.0, abs=0.1)

    def test_forecast_is_horizon_matched_and_annualised(self, fitted):
        m, _ = fitted
        f21 = m.forecast(21)
        assert 0.02 < f21 < 1.5, "must be an annualised vol, not a daily variance"
        # Mean reversion: a long horizon sits closer to the long-run level.
        lr = float(np.sqrt(m.fit_.long_run_variance * 252.0))
        assert abs(m.forecast(250) - lr) <= abs(m.forecast(2) - lr) + 1e-9

    def test_too_short_a_sample_is_refused(self):
        with pytest.raises(ValueError, match="few hundred"):
            GJRGarch().fit(np.random.default_rng(0).normal(0, 0.01, 100))


# ======================================================= section 6.2: SHAR

class TestSemivarianceAndSHAR:
    def test_semivariances_sum_exactly_to_the_pooled_proxy(self):
        """SHAR must NEST HAR, which requires the split to be exact."""
        bars, _, _ = _gjr_ohlc(n=600)
        semi = semivariance_proxy(bars)
        tot = daily_variance_proxy(bars)
        assert float(np.nanmax(np.abs(semi.sum(axis=1) - tot))) < 1e-15

    def test_both_components_are_strictly_positive(self):
        """A zero semivariance makes log(0) appear in the design matrix.

        On a trending day the observed low equals the open, so the raw split
        assigns zero negative semivariance. Floored to a constant in log space the
        two columns go near-collinear and the estimated asymmetry collapses --
        measured at t = 0.05 on data simulated WITH a strong leverage effect.
        """
        bars, _, _ = _gjr_ohlc(n=800)
        semi = semivariance_proxy(bars).dropna()
        assert len(semi) > 700
        assert (semi > 0).all().all()

    def test_a_flat_untraded_bar_is_nan_not_a_fabricated_split(self):
        bars = pd.DataFrame(
            {"open": [100.0, 100.0], "high": [100.0, 101.0], "low": [100.0, 99.0],
             "close": [100.0, 100.5], "volume": [0.0, 1.0]},
            index=pd.bdate_range("2026-01-05", periods=2).date)
        semi = semivariance_proxy(bars)
        assert semi.iloc[0].isna().all()

    def test_shar_detects_asymmetry_when_it_is_present(self):
        bars, _, _ = _gjr_ohlc(n=2200)
        semi = semivariance_proxy(bars)
        m = SHARModel(horizon=21)
        m.fit(semi["rv_neg"], semi["rv_pos"])
        a = m.asymmetry()
        assert a["negative_dominates"]
        assert a["t_difference"] > 2.0

    def test_shar_nests_har_and_does_not_fit_worse(self):
        bars, _, _ = _gjr_ohlc(n=2200)
        semi = semivariance_proxy(bars)
        tot = daily_variance_proxy(bars)
        s = SHARModel(horizon=21); s.fit(semi["rv_neg"], semi["rv_pos"])
        h = HARModel(horizon=21); h.fit(tot)
        assert s.fit_.r2 >= h.fit_.r2 - 1e-6

    def test_shar_refuses_without_both_series(self):
        with pytest.raises(ValueError, match="semivariance"):
            SHARModel()._design(pd.Series([1.0, 2.0]))


# ================================================== section 13.3: evaluation

class TestForecastEvaluation:
    def test_qlike_is_minimised_at_a_perfect_forecast(self):
        rv = np.full(200, 0.04)
        assert qlike(rv, rv) == pytest.approx(0.0, abs=1e-12)
        assert qlike(rv, rv * 1.3) > 0
        assert qlike(rv, rv * 0.7) > 0

    def test_qlike_punishes_under_forecasting_harder(self):
        """The asymmetry a short-premium book actually faces."""
        rv = np.full(200, 0.04)
        under = qlike(rv, rv * 0.5)      # forecast half the true variance
        over = qlike(rv, rv * 2.0)       # forecast double it
        assert under > over

    def test_mincer_zarnowitz_slope_is_one_for_an_unbiased_forecast(self):
        rng = np.random.default_rng(4)
        f = np.exp(rng.normal(np.log(0.04), 0.3, 600))
        rv = f * np.exp(rng.normal(0, 0.10, 600))
        mz = mincer_zarnowitz(rv, f)
        assert mz["slope"] == pytest.approx(1.0, abs=0.15)
        assert abs(mz["t_slope_eq_1"]) < 3

    def test_mincer_zarnowitz_flags_an_overreacting_forecast(self):
        rng = np.random.default_rng(5)
        rv = np.exp(rng.normal(np.log(0.04), 0.25, 800))
        # Overreaction means the forecast's DEVIATION FROM ITS MEAN is too large:
        # too high when high and too low when low. Regressing rv on such a
        # forecast gives a slope below 1, and shrinking it toward the mean would
        # improve it -- which is exactly what the online recalibration does.
        f = np.maximum(rv.mean() + 1.8 * (rv - rv.mean()), 1e-6)
        mz = mincer_zarnowitz(rv, f)
        assert mz["slope"] < 1.0
        assert "overreacts" in mz["verdict"]

    def test_diebold_mariano_favours_the_lower_loss_series(self):
        rng = np.random.default_rng(6)
        good = rng.normal(1.0, 0.2, 400)
        bad = rng.normal(1.5, 0.2, 400)
        dm = diebold_mariano(good, bad, h=1)
        assert dm["favours"] == "A" and dm["p_value"] < 0.01

    def test_diebold_mariano_corrects_for_overlapping_horizons(self):
        """An h-step test without the Newey-West correction is anticonservative."""
        rng = np.random.default_rng(7)
        a = rng.normal(1.0, 0.2, 400)
        b = rng.normal(1.05, 0.2, 400)
        h1 = diebold_mariano(a, b, h=1)
        h21 = diebold_mariano(a, b, h=21)
        assert abs(h21["stat"]) <= abs(h1["stat"]) + 1e-9
        assert "Newey-West" in h21["note"]

    def test_interval_coverage_measures_what_was_actually_covered(self):
        rng = np.random.default_rng(8)
        truth = rng.normal(0.2, 0.05, 1000)
        # A deliberately too-narrow band: a stated 90% that covers far less.
        cov = interval_coverage(truth, truth * 0 + 0.19, truth * 0 + 0.21)
        assert 0.0 < cov["coverage"] < 0.9
        assert cov["mean_width"] == pytest.approx(0.02)

    def test_compare_ranks_by_qlike_and_names_a_winner(self):
        rng = np.random.default_rng(9)
        rv = np.exp(rng.normal(np.log(0.04), 0.3, 500))
        out = compare_forecasts(rv, {
            "naive": np.roll(rv, 1),
            "good": rv * np.exp(rng.normal(0, 0.05, 500)),
            "bad": np.full(500, 0.20),
        }, horizon=1, benchmark="naive")
        assert out["best"] == "good"
        assert out["ranking_by_qlike"].index("good") < out["ranking_by_qlike"].index("bad")
        assert np.isfinite(out["scores"]["good"]["dm_vs_benchmark"]["p_value"])

    def test_score_reports_volatility_units_not_only_variance(self):
        rv = np.full(300, 0.04)                     # 20 vol
        s = score_forecast("f", rv, np.full(300, 0.0441))   # 21 vol
        assert s.bias_vol == pytest.approx(0.01, abs=1e-9)
        assert s.rmse_vol == pytest.approx(0.01, abs=1e-9)
