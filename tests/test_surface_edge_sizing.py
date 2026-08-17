import numpy as np
import pytest

from optionsmarkets.edge.score import build_scenarios, variance_risk_premium
from optionsmarkets.learning.online import (ADWIN, AdaptiveConformal, PageHinkley,
                                            ProbabilityCalibrator, TimeVaryingRegression,
                                            brier_decomposition)
from optionsmarkets.sizing.kelly import (drawdown_cap, kelly_fraction, risk_of_ruin,
                                         size_position, uncertainty_shrinkage)
from optionsmarkets.surface.ssvi import SSVIParams, fit_ssvi, no_butterfly, ssvi_w
from optionsmarkets.surface.svi import (SVIParams, crossedness, fit_svi_slice, svi_density,
                                        svi_g, svi_jw, svi_w)

TRUE = SVIParams(a=0.012, b=0.16, rho=-0.62, m=0.018, sigma=0.115)
T = 30 / 365


# ------------------------------------------------------------------ surface
def test_svi_recovers_known_parameters_from_noisy_quotes():
    rng = np.random.default_rng(7)
    k = np.linspace(-0.45, 0.32, 41)
    iv = np.sqrt(svi_w(k, TRUE) / T) + rng.normal(0, 0.0015, 41)
    fit = fit_svi_slice(k, iv, T, iv_spread=np.full(41, 0.004))
    assert fit.ok
    assert fit.rmse_vol < 0.002
    for got, want in zip(fit.params.as_array(), TRUE.as_array()):
        assert got == pytest.approx(want, abs=0.01)


def test_svi_density_is_a_density():
    k = np.linspace(-3, 3, 20001)
    d = svi_density(k, TRUE)
    assert d.min() >= -1e-12
    assert np.trapezoid(d, k) == pytest.approx(1.0, abs=5e-3)


def test_butterfly_detector_fires_where_domain_checks_pass():
    """Gatheral-Jacquier's central point: satisfying the SVI parameter domain
    does NOT imply the slice is butterfly-arbitrage-free."""
    bad = SVIParams(a=0.001, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
    assert bad.domain_ok()
    assert float(np.min(svi_g(np.linspace(-1, 1, 4001), bad))) < 0


def test_svi_jw_roundtrip_is_interpretable():
    jw = svi_jw(TRUE, T)
    assert jw["v"] == pytest.approx(svi_w(0.0, TRUE) / T)
    assert jw["p_wing"] > jw["c_wing"]        # negative rho -> steeper put wing


def test_crossedness_detects_calendar_arbitrage():
    near = SVIParams(0.020, 0.16, -0.6, 0.0, 0.1)
    far = SVIParams(0.010, 0.16, -0.6, 0.0, 0.1)      # LESS total variance later: illegal
    assert crossedness(near, far, np.linspace(-0.5, 0.5, 501)) > 0
    assert crossedness(far, near, np.linspace(-0.5, 0.5, 501)) == 0.0


def test_ssvi_no_arbitrage_conditions_and_recovery():
    p = SSVIParams(rho=-0.7, eta=0.9, gamma=0.5)
    th = np.array([0.0012, 0.004, 0.010, 0.021, 0.045])
    assert p.ok() and no_butterfly(th, p)[0]
    assert not SSVIParams(rho=-0.7, eta=2.5).ok()

    ks = [np.linspace(-0.3, 0.25, 25)] * 5
    ws = [ssvi_w(k, t, p) for k, t in zip(ks, th)]
    rec, sse = fit_ssvi(ks, ws, th, n_rho=61)
    assert rec.rho == pytest.approx(-0.7, abs=0.01)
    assert rec.eta == pytest.approx(0.9, abs=0.01)


# -------------------------------------------------------------------- Kelly
def test_kelly_binary_matches_closed_form():
    p_win, b = 0.6, 1.0
    f = kelly_fraction(np.array([b, -1.0]), np.array([p_win, 1 - p_win]))
    assert f == pytest.approx(p_win - (1 - p_win) / b, abs=1e-9)


def test_kelly_zero_without_edge():
    assert kelly_fraction(np.array([1.0, -1.0]), np.array([0.5, 0.5])) == 0.0
    assert kelly_fraction(np.array([1.0, -1.0]), np.array([0.4, 0.6])) == 0.0


def test_kelly_respects_ruin_bound():
    f = kelly_fraction(np.array([0.2, -1.0]), np.array([0.95, 0.05]))
    assert 0 < f < 1.0                       # never stakes past total loss


def test_growth_curve_is_symmetric_about_full_kelly():
    """2c - c^2: half Kelly keeps 75% of growth; 1.5x Kelly keeps the same
    75% with 50% more volatility; 2x Kelly gives zero excess growth."""
    g = lambda c: 2 * c - c * c
    assert g(0.5) == pytest.approx(0.75)
    assert g(1.5) == pytest.approx(0.75)
    assert g(2.0) == pytest.approx(0.0)
    assert g(2.5) < 0


def test_uncertainty_shrinkage_reproduces_half_kelly():
    """Sharpe 0.5 with 4 years of evidence deserves exactly half Kelly."""
    assert uncertainty_shrinkage(0.5, 4.0) == pytest.approx(0.5)
    assert uncertainty_shrinkage(1.0, 10.0) == pytest.approx(0.909, abs=1e-3)


def test_drawdown_cap_and_risk_of_ruin_are_inverses():
    c = drawdown_cap(max_drawdown=0.50, prob=0.05)
    assert c == pytest.approx(0.376, abs=1e-3)
    assert risk_of_ruin(c, 0.5) == pytest.approx(0.05, abs=1e-9)
    assert risk_of_ruin(1.0, 0.5) == pytest.approx(0.5)      # full Kelly: 50% chance


def test_size_position_rounds_toward_zero_and_reports_binding():
    pnl = np.array([200.0, -800.0])
    prob = np.array([0.85, 0.15])
    s = size_position(pnl, prob, 800.0, 40_000.0, sharpe=0.6, years_of_evidence=3.0,
                      max_risk_fraction_per_trade=0.035)
    assert s.contracts == int(s.capital_at_risk // 800.0)
    assert s.capital_at_risk <= 0.035 * 40_000.0 + 1e-9
    assert s.c_total <= 0.5
    assert s.binding_constraint


# ------------------------------------------------------------------- online
def test_kalman_tracks_a_step_change_in_the_coefficient():
    rng = np.random.default_rng(1)
    m = TimeVaryingRegression(1, q=5e-3, R=0.01, prior_var=1.0)
    for t in range(600):
        beta = 1.0 if t < 300 else -2.0
        x = np.array([rng.normal()])
        m.update(x, float(x[0] * beta + rng.normal(0, 0.1)))
    assert m.beta[0] == pytest.approx(-2.0, abs=0.2)
    assert m.innovation_health()["n"] > 100


def test_kalman_covariance_stays_positive_definite():
    rng = np.random.default_rng(2)
    m = TimeVaryingRegression(3, q=1e-4, R=1.0)
    for _ in range(3000):
        x = rng.normal(size=3)
        m.update(x, float(x @ np.array([1.0, -0.5, 0.2]) + rng.normal()))
    assert np.all(np.linalg.eigvalsh(m.P) > 0)
    assert np.allclose(m.P, m.P.T, atol=1e-12)


def test_conformal_restores_coverage_after_a_shift():
    rng = np.random.default_rng(3)
    aci = AdaptiveConformal(alpha=0.10, gamma=0.02, window=400)
    for t in range(3000):
        scale = 1.0 if t < 1500 else 6.0       # regime break
        aci.update(0.0, float(rng.normal(0, scale)))
    cov = 1.0 - float(np.mean(list(aci.errs)))
    assert cov == pytest.approx(0.90, abs=0.06)


def test_conformal_kill_switch_triggers_on_hopeless_model():
    aci = AdaptiveConformal(alpha=0.10, gamma=0.10, window=50, kill_after=15)
    for t in range(400):
        aci.update(0.0, 1e6 * (t + 1))         # never inside any interval
    assert aci.killed


def test_probability_calibrator_improves_reliability():
    rng = np.random.default_rng(4)
    n = 800
    truth = rng.random(n)
    y = (rng.random(n) < truth).astype(float)
    raw = np.clip(truth * 0.5 + 0.25, 1e-6, 1 - 1e-6)     # squashed => miscalibrated
    before = brier_decomposition(raw, y)["rel"]
    cal = ProbabilityCalibrator().fit(raw, y)
    after = brier_decomposition(cal.transform(raw), y)["rel"]
    assert after < before


def test_brier_decomposition_identity():
    rng = np.random.default_rng(5)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)
    d = brier_decomposition(p, y, n_bins=20)
    assert d["brier"] == pytest.approx(d["rel"] - d["res"] + d["unc"], abs=0.01)


def test_drift_detectors_fire_on_a_mean_shift_and_not_on_noise():
    rng = np.random.default_rng(6)
    ph, ad = PageHinkley(delta=0.005, threshold=5.0), ADWIN(delta=0.002)
    quiet = [ph.update(float(rng.normal(0, 0.1))) for _ in range(300)]
    assert not any(quiet)
    for _ in range(300):
        ad.update(float(rng.normal(0, 0.1)))
    fired = any(ad.update(float(rng.normal(5, 0.1))) for _ in range(300))
    assert fired
    assert any(ph.update(float(rng.normal(5, 0.1))) for _ in range(300))


# --------------------------------------------------------------------- edge
def test_vrp_sign_convention():
    assert "SELL" in variance_risk_premium(0.28, 0.14)["direction"]
    assert "BUY" in variance_risk_premium(0.12, 0.20)["direction"]


def test_physical_measure_differs_from_risk_neutral_when_vrp_is_positive():
    scen = build_scenarios(TRUE, 780.0, T, sigma_forecast=0.12)
    sd_q = np.sqrt(np.sum(scen.prob_Q * (scen.S_T - np.sum(scen.prob_Q * scen.S_T)) ** 2))
    sd_p = np.sqrt(np.sum(scen.prob_P * (scen.S_T - np.sum(scen.prob_P * scen.S_T)) ** 2))
    assert sd_p < sd_q                       # forecast vol below implied -> tighter P
    assert scen.prob_P.sum() == pytest.approx(1.0, abs=1e-6)
    assert scen.prob_Q.sum() == pytest.approx(1.0, abs=1e-6)
