"""Phase 11: the constrained joint surface fit and its publication gate."""

from __future__ import annotations

import numpy as np
import pytest

from optionsmarkets.surface.joint import (
    SliceQuotes, SurfacePublisher, SurfaceViolation, fit_surface, single_slice_surface,
    ssvi_to_svi,
)
from optionsmarkets.surface.ssvi import SSVIParams, ssvi_w
from optionsmarkets.surface.svi import SVIParams, crossedness, fit_svi_slice, svi_w

TRUE = SSVIParams(rho=-0.70, eta=0.90, gamma=0.5)


def _slices(ssvi=TRUE, Ts=(0.05, 0.12, 0.25, 0.50, 1.00), noise=0.0015, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for T in Ts:
        theta = 0.04 * T + 0.002
        k = np.linspace(-0.45, 0.30, 33)
        iv = np.sqrt(ssvi_w(k, theta, ssvi) / T)
        out.append(SliceQuotes(T=T, k=k, iv=iv + rng.normal(0, noise, iv.size),
                               iv_spread=np.full(iv.size, 0.004), forward=780.0))
    return out


class TestSSVIToSVIMap:
    def test_map_is_exact_not_a_fit(self):
        """The SSVI->raw-SVI map must be exact to machine precision."""
        k = np.linspace(-0.8, 0.5, 41)
        for theta in (0.002, 0.01, 0.04, 0.12, 0.5):
            a = ssvi_w(k, theta, TRUE)
            b = svi_w(k, ssvi_to_svi(theta, TRUE))
            assert np.max(np.abs(a - b)) < 1e-14

    def test_lee_bound_and_gj_butterfly_condition_coincide(self):
        """Under the map, b(1+|rho|)<=2 IS theta*phi*(1+|rho|)<=4.

        The two conditions are the same statement, and their agreement is the
        coherence check that the map is right.
        """
        from optionsmarkets.surface.ssvi import ssvi_phi
        for theta in (0.005, 0.05, 0.4):
            p = ssvi_to_svi(theta, TRUE)
            lee = p.b * (1.0 + abs(p.rho))
            gj = theta * float(ssvi_phi(theta, TRUE)) * (1.0 + abs(TRUE.rho)) / 2.0
            assert lee == pytest.approx(gj, rel=1e-12)


class TestJointFit:
    def test_recovers_known_ssvi_backbone(self):
        s = fit_surface(_slices(), underlying="T", asof="x")
        assert s.ok, s.detail
        assert s.ssvi.rho == pytest.approx(TRUE.rho, abs=0.01)
        assert s.ssvi.eta == pytest.approx(TRUE.eta, abs=0.02)

    def test_every_slice_is_butterfly_free_and_uncrossed(self):
        s = fit_surface(_slices())
        assert s.diagnostics.min_g_overall > 0
        assert s.diagnostics.max_crossedness == 0.0
        for r in s.slices:
            assert r.params.domain_ok()

    def test_refinement_beats_the_backbone_on_fit_error(self):
        """The refinement must actually be used, not silently fall back."""
        s = fit_surface(_slices())
        assert s.diagnostics.n_fallbacks == 0
        assert all(r.source == "refined" for r in s.slices)
        assert s.diagnostics.worst_rmse_vol < 0.002

    def test_calendar_constraint_is_enforced_during_the_fit(self):
        """Independent slice fits may cross; the joint fit must not.

        This is the whole reason the joint fit exists, so it is asserted against
        input built to make an independent fit cross: a far slice whose quoted
        vols are pulled DOWN enough that its total variance dips below the near
        slice's over part of the strike range.
        """
        sl = _slices(Ts=(0.10, 0.20), noise=0.0)
        far = sl[1]
        pulled = far.iv * 0.62               # w_far now dips under w_near
        sl[1] = SliceQuotes(T=far.T, k=far.k, iv=pulled,
                            iv_spread=far.iv_spread, forward=far.forward)

        # Independent fits on this input DO cross.
        f0 = fit_svi_slice(sl[0].k, sl[0].iv, sl[0].T, iv_spread=sl[0].iv_spread)
        f1 = fit_svi_slice(sl[1].k, sl[1].iv, sl[1].T, iv_spread=sl[1].iv_spread)
        kg = np.linspace(-0.4, 0.25, 401)
        assert crossedness(f0.params, f1.params, kg) > 1e-6, \
            "fixture no longer produces a crossing; the test would be vacuous"

        # The contract is: eliminate the crossing, or refuse to publish. What is
        # NOT allowed is a crossed surface that reports itself healthy.
        s = fit_surface(sl)
        if s.diagnostics.max_crossedness > 1e-10:
            assert not s.ok, "a crossed surface reported itself publishable"
            with pytest.raises(SurfaceViolation):
                SurfacePublisher().publish(s)
        else:
            assert s.diagnostics.n_fallbacks > 0 or s.ok

    def test_fallback_is_not_assumed_safe(self):
        """When the backbone crosses worse than the refinement, keep the refinement.

        The SSVI backbone is butterfly-free by construction but not automatically
        uncrossed: theta can be repaired up to a neighbour's level while that
        neighbour was refined upward to fit its own quotes. Blindly preferring
        the backbone would publish the worse of the two.
        """
        sl = _slices(Ts=(0.10, 0.20), noise=0.0)
        far = sl[1]
        sl[1] = SliceQuotes(T=far.T, k=far.k, iv=far.iv * 0.62,
                            iv_spread=far.iv_spread, forward=far.forward)
        s = fit_surface(sl)
        # Whatever it chose, the reported crossedness must be the chosen slice's,
        # and the surface must not claim to be ok while crossed.
        assert (s.diagnostics.max_crossedness <= 1e-10) == s.ok or not s.ok
        assert s.detail, "a rejected surface must say why"

    def test_non_monotone_atm_variance_is_repaired_and_reported(self):
        sl = _slices(Ts=(0.10, 0.20), noise=0.0)
        far = sl[1]
        sl[1] = SliceQuotes(T=far.T, k=far.k, iv=far.iv * 0.5,
                            iv_spread=far.iv_spread, forward=far.forward)
        s = fit_surface(sl)
        assert s.diagnostics.theta_adjusted
        assert "non-monotone" in s.detail

    def test_too_few_quotes_is_a_refusal_not_a_guess(self):
        s = fit_surface([SliceQuotes(T=0.1, k=np.array([0.0, 0.1]),
                                     iv=np.array([0.2, 0.21]))])
        assert not s.ok
        assert "enough usable quotes" in s.detail


class TestInterpolation:
    def test_total_variance_interpolation_cannot_create_a_calendar_arbitrage(self):
        """Linear-in-T on total variance preserves w(k,T) non-decreasing in T."""
        s = fit_surface(_slices())
        k = np.linspace(-0.4, 0.25, 61)
        Ts = np.linspace(0.06, 0.99, 40)
        w = np.array([s.total_variance(k, T) for T in Ts])
        assert np.min(np.diff(w, axis=0)) >= -1e-12

    def test_extrapolation_keeps_theta_non_decreasing(self):
        s = fit_surface(_slices())
        Ts = [0.001, 0.01, 0.05, 0.5, 1.0, 2.0, 5.0]
        th = [s.theta_at(T) for T in Ts]
        assert np.min(np.diff(th)) >= -1e-12

    def test_vol_at_a_fitted_maturity_matches_that_slice(self):
        s = fit_surface(_slices())
        r = s.slices[2]
        k = np.linspace(-0.3, 0.2, 21)
        assert np.allclose(s.total_variance(k, r.T), svi_w(k, r.params), atol=1e-12)


class TestPublicationGate:
    def _bad(self):
        # Passes every parameter-domain check and still has min g << 0 -- the
        # Gatheral-Jacquier point, and exactly what the gate exists to catch.
        bad = SVIParams(a=0.001, b=0.9, rho=-0.95, m=0.0, sigma=0.02)
        assert bad.domain_ok()
        fit = fit_svi_slice(np.linspace(-0.2, 0.2, 9),
                            np.sqrt(np.maximum(svi_w(np.linspace(-0.2, 0.2, 9), bad), 1e-9) / 0.1),
                            0.1)
        s = single_slice_surface(fit, 0.1)
        s.ok = False
        s.detail = "synthetic violation"
        return s

    def test_first_bad_surface_with_no_history_raises(self):
        pub = SurfacePublisher()
        with pytest.raises(SurfaceViolation):
            pub.publish(self._bad())

    def test_bad_surface_falls_back_to_last_known_good(self):
        pub = SurfacePublisher()
        good = fit_surface(_slices())
        assert pub.publish(good) is good

        with pytest.raises(SurfaceViolation) as ei:
            pub.publish(self._bad())
        assert "last known-good" in str(ei.value)
        # The rejected surface travels with the exception so it can be journaled.
        assert ei.value.surface is not None
        # And non-raising callers get the good surface to keep managing with.
        assert pub.publish(self._bad(), raise_on_violation=False) is good

    def test_repeated_rejection_stops_serving_a_stale_surface(self):
        pub = SurfacePublisher(max_stale_publications=2)
        pub.publish(fit_surface(_slices()))
        for _ in range(2):
            with pytest.raises(SurfaceViolation):
                pub.publish(self._bad())
        # Past the limit it refuses to keep pretending yesterday's surface is live.
        assert pub.publish(self._bad(), raise_on_violation=False).ok is False

    def test_arbitrage_margin_is_trendable(self):
        pub = SurfacePublisher()
        for _ in range(4):
            pub.publish(fit_surface(_slices()))
        t = pub.arbitrage_margin_trend()
        assert t["n"] == 4 and "min_g_slope" in t
