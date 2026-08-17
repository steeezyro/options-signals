from datetime import date

import numpy as np
import pytest

from optionsmarkets.data.provider import assess_quality
from optionsmarkets.domain.structures import (MULTIPLIER, OptionQuote, Right, butterfly, iron_condor, occ_symbol, vertical)
from optionsmarkets.edge.score import build_scenarios, score_structure
from optionsmarkets.execution.schwab import SchwabCosts, build_ticket
from optionsmarkets.policy.decide import Action, Thresholds, decide, rank_and_select
from optionsmarkets.sizing.kelly import size_position
from optionsmarkets.surface.svi import SVIParams, fit_svi_slice

EXP = date(2026, 9, 18)


def q(strike, right, bid, ask, oi=500.0, iv=0.2):
    return OptionQuote(strike, right, EXP, bid, ask, 0.5 * (bid + ask), 100.0, oi, iv)


# ----------------------------------------------------------------- symbols
def test_occ_symbol_format():
    """21 chars: 6-char underlying space-padded, YYMMDD, C/P, 8-digit strike."""
    s = occ_symbol("SPY", EXP, Right.PUT, 780.0)
    assert s == "SPY   260918P00780000"
    assert len(s) == 21
    assert occ_symbol("AAPL", date(2026, 1, 16), Right.CALL, 62.5) == "AAPL  260116C00062500"


# -------------------------------------------------------------- structures
def test_credit_vertical_payoff_and_bounds():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    net = st.net_price("marketable")          # buy the ask, sell the bid
    assert net == pytest.approx((2.20 - 4.60) * MULTIPLIER)
    assert net < 0                            # credit
    assert st.max_loss(net) == pytest.approx(15.0 * MULTIPLIER + net)
    assert st.max_gain(net) == pytest.approx(-net)
    assert st.is_defined_risk()
    be = st.breakevens(net)
    assert len(be) == 1 and 730 < be[0] < 745


def test_marketable_pricing_is_worse_than_mid():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    assert abs(st.net_price("marketable")) < abs(st.net_price("mid"))


def test_iron_condor_and_butterfly_are_defined_risk():
    quotes = {("P", 700.0): q(700, Right.PUT, 1.0, 1.1), ("P", 720.0): q(720, Right.PUT, 2.0, 2.2),
              ("C", 820.0): q(820, Right.CALL, 2.0, 2.2), ("C", 840.0): q(840, Right.CALL, 1.0, 1.1)}
    ic = iron_condor("SPY", EXP, 700, 720, 820, 840, quotes)
    assert ic.is_defined_risk()
    assert ic.max_loss(ic.net_price("marketable")) < 20 * MULTIPLIER

    bq = {("C", k): q(k, Right.CALL, 5.0 - i, 5.2 - i) for i, k in enumerate((780.0, 790.0, 800.0))}
    bf = butterfly("SPY", EXP, Right.CALL, 780, 790, 800, bq)
    assert bf.is_defined_risk()


def test_position_greeks_sum_with_sign():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    g = st.greeks(778.54, 0.042, 0.012, [0.22, 0.20], [0.088, 0.088])
    assert g["delta"] > 0          # short put spread is long delta
    assert g["theta"] > 0          # and collects time decay
    assert g["vega"] < 0           # and is short vol


def test_quote_tradeability_screen():
    assert q(730, Right.PUT, 2.00, 2.20).tradeable()
    assert not q(730, Right.PUT, 0.0, 0.05).tradeable()       # zero bid
    assert not q(730, Right.PUT, 1.00, 2.00).tradeable()      # 67% spread
    assert not q(730, Right.PUT, 2.00, 2.10, oi=1).tradeable()  # no open interest


# ----------------------------------------------------------- feed quality
def test_quality_flags_yfinance_sentinels_and_zero_bids():
    quotes = [q(700 + 5 * i, Right.PUT, 1.0 + i, 1.1 + i) for i in range(20)]
    quotes += [OptionQuote(300.0 + 5 * j, Right.PUT, EXP, 0.0, 0.0, 0.01, 210.0, 0.0, 0.500005)
               for j in range(3)]
    fq = assess_quality(quotes, max_age_s=900)
    assert fq.n_sentinel_iv == 3
    assert fq.n_zero_bid == 3
    assert any("sentinel" in d for d in fq.detail)
    assert fq.usable                       # sentinels are noted, not fatal


def test_quality_rejects_thin_and_crossed_books():
    assert assess_quality([q(700, Right.PUT, 1.0, 1.1)], 900).verdict == "TOO_FEW_QUOTES"
    crossed = [q(700 + i, Right.PUT, 2.0, 1.0) for i in range(20)]
    assert assess_quality(crossed, 900).verdict in ("CROSSED_BOOK", "ILLIQUID")


# ---------------------------------------------------------------- policy
def _setup():
    params = SVIParams(a=0.0025, b=0.048, rho=-0.72, m=0.03, sigma=0.07)
    k = np.linspace(-0.25, 0.15, 31)
    from optionsmarkets.surface.svi import svi_w
    T = 32 / 365
    iv = np.sqrt(svi_w(k, params) / T)
    fit = fit_svi_slice(k, iv, T, iv_spread=np.full(31, 0.004))
    scen = build_scenarios(fit.params, 780.6, T, sigma_forecast=0.135)
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.85, 5.15))
    net = st.net_price("marketable")
    ed = score_structure(st, scen, net, fees=1.31)
    sz = size_position(ed.payoff, scen.prob_P, ed.max_loss, 40_000.0,
                       sharpe=0.55, years_of_evidence=2.0, max_risk_fraction_per_trade=0.035)
    return st, ed, sz, fit


def _kw(**over):
    base = dict(quote_age_s=900.0, legs_tradeable=[True, True], conformal_killed=False,
                calibration_rel=0.004, bankroll=40_000.0, dte=32,
                credit_to_width=0.19, round_trip_cost=2.60, thresholds=Thresholds())
    base.update(over)
    return base


def test_policy_emits_sell_when_every_gate_passes():
    st, ed, sz, fit = _setup()
    d = decide(ed, sz, st, slice_fit=fit, **_kw())
    assert d.action is Action.SELL
    assert d.contracts >= 1
    assert not d.blocked_by
    assert "vol points" in d.rationale


@pytest.mark.parametrize("override,gate", [
    (dict(quote_age_s=7200.0), "data.freshness"),
    (dict(legs_tradeable=[True, False]), "liquidity.all_legs"),
    (dict(conformal_killed=True), "model.conformal"),
    (dict(calibration_rel=0.09), "model.calibration"),
    (dict(dte=3), "risk.dte_window"),
    (dict(days_to_earnings=1), "risk.event_window"),
    (dict(open_positions_same_underlying=5), "risk.concentration"),
    (dict(credit_to_width=0.03), "edge.credit_to_width"),
    (dict(round_trip_cost=500.0), "edge.ev_vs_cost"),
    (dict(portfolio_vega_dollars=50_000.0), "risk.portfolio_vega"),
])
def test_every_gate_can_block_independently(override, gate):
    st, ed, sz, fit = _setup()
    d = decide(ed, sz, st, slice_fit=fit, **_kw(**override))
    assert d.action is Action.HOLD
    assert gate in d.blocked_by


def test_hold_is_the_default_when_nothing_is_known():
    st, ed, sz, fit = _setup()
    d = decide(ed, sz, st, slice_fit=fit,
               **_kw(quote_age_s=1e9, legs_tradeable=[False, False], conformal_killed=True))
    assert d.action is Action.HOLD
    assert len(d.blocked_by) >= 3


def test_rank_and_select_only_returns_gate_passing_candidates():
    st, ed, sz, fit = _setup()
    best, dec, evaluated = rank_and_select(
        [(st, ed, sz)], lambda s, e, z: dict(slice_fit=fit, **_kw(credit_to_width=0.01)))
    assert best is None and dec is None
    assert len(evaluated) == 1

    best, dec, _ = rank_and_select(
        [(st, ed, sz)], lambda s, e, z: dict(slice_fit=fit, **_kw()))
    assert best is not None and dec.action is Action.SELL


# ------------------------------------------------------------- execution
def test_schwab_cost_model_counts_every_leg():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    c = SchwabCosts().estimate(st, contracts=1)
    assert c["commission"] == pytest.approx(1.30)          # 2 contracts x $0.65
    c10 = SchwabCosts().estimate(st, contracts=10)
    assert c10["commission"] == pytest.approx(13.00)

    quotes = {("P", 700.0): q(700, Right.PUT, 1.0, 1.1), ("P", 720.0): q(720, Right.PUT, 2.0, 2.2),
              ("C", 820.0): q(820, Right.CALL, 2.0, 2.2), ("C", 840.0): q(840, Right.CALL, 1.0, 1.1)}
    ic = iron_condor("SPY", EXP, 700, 720, 820, 840, quotes)
    assert SchwabCosts().estimate(ic, 1)["commission"] == pytest.approx(2.60)   # 4 legs


def test_ticket_api_payload_uses_valid_schwab_enums():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    t = build_ticket(st, 2, 778.54)
    j = t.api_json
    assert j["orderType"] in ("NET_DEBIT", "NET_CREDIT", "NET_ZERO")
    assert j["orderStrategyType"] == "SINGLE"        # NOT "MULTI_LEG" -- invalid
    assert j["complexOrderStrategyType"] == "VERTICAL"
    assert j["duration"] in ("DAY", "GOOD_TILL_CANCEL")
    for leg in j["orderLegCollection"]:
        assert leg["instruction"] in ("BUY_TO_OPEN", "SELL_TO_OPEN",
                                      "BUY_TO_CLOSE", "SELL_TO_CLOSE")
        assert leg["instrument"]["assetType"] == "OPTION"
        assert len(leg["instrument"]["symbol"]) == 21
        assert leg["quantity"] == 2


def test_ticket_price_ladder_walks_from_mid_to_natural():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    t = build_ticket(st, 1, 778.54)
    assert t.mid >= t.natural                       # credit: mid better than natural
    assert t.price_ladder[0] == pytest.approx(t.mid, abs=0.011)
    assert t.price_ladder[-1] == pytest.approx(t.natural, abs=0.011)


def test_ticket_carries_the_assignment_and_approval_warnings():
    st = vertical("SPY", EXP, Right.PUT, 730.0, 745.0,
                  q_long=q(730, Right.PUT, 2.00, 2.20), q_short=q(745, Right.PUT, 4.60, 4.90))
    t = build_ticket(st, 1, 778.54)
    joined = " ".join(t.warnings)
    assert "$0.01 ITM" in joined
    assert "margin account" in joined
    assert any("Time stop" in s for s in t.exit_plan)
    assert "Trade -> Options" in t.web_steps[0] or "Trade" in t.web_steps[0]
    assert len(t.tos_steps) >= 8


# ------------------------------------------------------ portfolio risk
def test_portfolio_risk_aggregates_in_dollars_and_flags_correlation():
    from optionsmarkets.risk.portfolio import PositionRisk, aggregate, check_limits

    g = dict(delta=25.0, gamma=1.2, vega=-40.0, theta=8.0)
    pos = [PositionRisk("SPY", 780.0, 1, g, 1266.0, 32, beta=1.0),
           PositionRisk("QQQ", 620.0, 1, g, 1100.0, 30, beta=1.15),
           PositionRisk("IWM", 240.0, 1, g, 900.0, 28, beta=1.05)]
    pf = aggregate(pos, bankroll=40_000.0)

    # dollar delta must scale with the underlying's price, not just the greek
    assert pos[0].delta_dollars > pos[2].delta_dollars
    assert pf.capital_at_risk == pytest.approx(3266.0)
    assert pf.vega_dollars < 0                      # short vol across the book

    fx = pf.factor_exposure()
    assert fx["concentration_ratio"] > 0.9          # three names, one bet
    breaches = {b.name for b in check_limits(pf, max_capital_pct=0.05)}
    assert "capital_at_risk" in breaches
    assert not check_limits(pf, max_delta_pct=1.0, max_vega_pct=1.0,
                            max_gamma_pct=1.0, max_capital_pct=1.0)


def test_portfolio_concentration_limit():
    from optionsmarkets.risk.portfolio import PositionRisk, aggregate, check_limits
    g = dict(delta=10.0, gamma=0.5, vega=-20.0, theta=4.0)
    pf = aggregate([PositionRisk("SPY", 780.0, 1, g, 5000.0, 30),
                    PositionRisk("XYZ", 50.0, 1, g, 200.0, 30)], bankroll=100_000.0)
    names = {b.name for b in check_limits(pf)}
    assert "concentration.SPY" in names
