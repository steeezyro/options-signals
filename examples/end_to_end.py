"""End-to-end demonstration on a synthetic-but-realistic SPY chain.

Runs the entire pipeline with no network access so the mechanics are testable:

    synthetic chain (SVI ground truth + bid/ask + zero-bid junk + vendor
    sentinels)  ->  quality screen  ->  forward from put-call parity
    ->  in-house IV inversion  ->  SVI fit + arbitrage gates
    ->  RV forecast  ->  VRP  ->  Q->P scenarios
    ->  structure enumeration  ->  edge scoring  ->  Kelly sizing
    ->  gate stack  ->  decision  ->  Schwab order ticket

Run:  PYTHONPATH=src python examples/end_to_end.py
"""

from __future__ import annotations

from datetime import date

import numpy as np

from optionsmarkets.data.provider import assess_quality
from optionsmarkets.domain.structures import OptionQuote, Right, vertical
from optionsmarkets.policy.decide import Action
from optionsmarkets.edge.score import build_scenarios, score_structure, variance_risk_premium
from optionsmarkets.execution.schwab import SchwabCosts, build_ticket
from optionsmarkets.policy.decide import Thresholds, rank_and_select
from optionsmarkets.pricing import black
from optionsmarkets.pricing.forward import implied_forward
from optionsmarkets.pricing.implied import implied_vol_scalar
from optionsmarkets.sizing.kelly import size_position
from optionsmarkets.surface.svi import SVIParams, fit_svi_slice, svi_w

rng = np.random.default_rng(20260817)

# ---------------------------------------------------------------- 0. setup
SPOT = 778.54                     # SPY, previous close from the live Massive feed
EXPIRY = date(2026, 9, 18)
TODAY = date(2026, 8, 17)
T = (EXPIRY - TODAY).days / 365.0
R, Q = 0.0421, 0.0118             # from the treasury curve + SPY yield
BANKROLL = 40_000.0
# Per-trade risk cap is a POLICY DIAL, not a model output. At 2% of a $25k
# account ($500) no SPY vertical fits at all -- $780 notional makes even a
# 5-wide risk ~$430 and a 15-wide ~$1,400. That is a capital constraint, and
# the right responses are: a lower-notional underlying, narrower spreads, or a
# deliberately larger cap. Pretending it away by sizing to fractional contracts
# is how a backtest diverges from an account.
MAX_RISK_PER_TRADE = 0.035

# Two regimes so both branches of the policy are exercised.
REGIMES = {
    # ATM ~13.6 vol, VRP ~+2 pts: a normal SPY tape. Should HOLD.
    "calm":     (SVIParams(a=0.00040, b=0.0165, rho=-0.80, m=0.020, sigma=0.055), 0.1180),
    # ATM ~24 vol after a shock, RV forecast lags: fat VRP. Should SELL.
    "stressed": (SVIParams(a=0.00250, b=0.0480, rho=-0.72, m=0.030, sigma=0.070), 0.1350),
}


def synth_chain(TRUE):
    """Ground-truth SVI slice, dressed with realistic market pathologies."""
    F = float(black.forward(SPOT, R, Q, T))
    df = float(np.exp(-R * T))
    strikes = np.arange(640, 900, 5.0)
    quotes = []
    for K in strikes:
        k = np.log(K / F)
        iv = float(np.sqrt(max(svi_w(k, TRUE), 1e-9) / T))
        for right in (Right.CALL, Right.PUT):
            mid = float(black.price(SPOT, K, T, R, Q, iv, right.cp))
            vega = float(black.greeks(SPOT, K, T, R, Q, iv, right.cp).vega) / 100.0
            half = max(0.01, min(0.35, 0.012 * mid + 0.0009 * vega * 100))
            bid, ask = round(mid - half, 2), round(mid + half, 2)
            if bid <= 0.0:                     # far OTM: zero bid, junk sentinel IV
                bid, ask = 0.0, max(round(ask, 2), 0.01)
                iv_vendor = 0.500005
                oi, vol = 0.0, rng.integers(0, 40)
            else:
                iv_vendor = iv * (1 + rng.normal(0, 0.02))
                oi = float(rng.integers(50, 9000))
                vol = float(rng.integers(1, 2500))
            quotes.append(OptionQuote(K, right, EXPIRY, bid, ask, mid,
                                      float(vol), oi, iv_vendor))
    return quotes, F, df


def run(regime: str):
    TRUE, rv_forecast = REGIMES[regime]
    quotes, F_true, df_true = synth_chain(TRUE)
    print("=" * 78)
    print(f"SPY  spot {SPOT}  expiry {EXPIRY}  ({(EXPIRY-TODAY).days} DTE)  "
          f"bankroll ${BANKROLL:,.0f}   [regime: {regime.upper()}]")
    print("=" * 78)

    # ------------------------------------------------------- 1. quality gate
    qual = assess_quality(quotes, max_age_s=900.0)
    print(f"\n[1] FEED QUALITY  {qual.verdict}")
    print(f"    {qual.n_quotes} quotes | {qual.n_two_sided} two-sided | "
          f"{qual.n_zero_bid} zero-bid | {qual.n_sentinel_iv} sentinel IVs | "
          f"median spread {qual.median_rel_spread:.2%}")
    for d in qual.detail:
        print(f"    - {d}")
    if not qual.usable:
        print("    ABORT: unusable snapshot")
        return

    # ---------------------------------------------- 2. forward from parity
    by_k: dict[float, dict] = {}
    for q in quotes:
        by_k.setdefault(q.strike, {})[q.right] = q
    Ks = sorted(k for k, v in by_k.items() if len(v) == 2)
    cm = [by_k[k][Right.CALL].mid for k in Ks]
    pm = [by_k[k][Right.PUT].mid for k in Ks]
    wts = [1.0 / max(by_k[k][Right.CALL].spread + by_k[k][Right.PUT].spread, 1e-3) for k in Ks]
    ff = implied_forward(Ks, cm, pm, T, weights=wts, S_ref=SPOT, atm_window=0.08)
    print("\n[2] FORWARD FROM PUT-CALL PARITY")
    print(f"    F = {ff.F:.4f}  (true {F_true:.4f}, err {ff.F - F_true:+.4f})")
    print(f"    DF = {ff.DF:.6f}  implied r = {ff.r_implied:.4%}  R2 = {ff.r2:.8f}  "
          f"n = {ff.n_pairs}  residual = {ff.residual_bp:.2f} bp")
    F, DF = ff.F, ff.DF

    # ------------------------------------------- 3. in-house IV inversion
    ks, ivs, sps, rejected = [], [], [], {}
    for q in quotes:
        otm = (q.right is Right.CALL and q.strike >= F) or (q.right is Right.PUT and q.strike < F)
        if not otm or not q.tradeable(max_rel_spread=0.25, min_oi=25):
            continue
        res = implied_vol_scalar(q.mid, F, q.strike, T, q.right.cp, DF,
                                 spread=q.spread, max_spread_vols=0.05)
        if not res.ok:
            rejected[res.status] = rejected.get(res.status, 0) + 1
            continue
        lo = implied_vol_scalar(q.bid, F, q.strike, T, q.right.cp, DF)
        hi = implied_vol_scalar(q.ask, F, q.strike, T, q.right.cp, DF)
        ks.append(np.log(q.strike / F)); ivs.append(res.sigma)
        sps.append(0.5 * abs(hi.sigma - lo.sigma) if (lo.ok and hi.ok) else 0.01)
    print("\n[3] IV INVERSION (vendor IV discarded)")
    print(f"    {len(ivs)} OTM strikes inverted | rejected: {rejected or 'none'}")
    print(f"    IV range {min(ivs):.2%} - {max(ivs):.2%}, median spread "
          f"{np.median(sps) * 100:.2f} vol pts")

    # --------------------------------------------------- 4. surface fit
    fit = fit_svi_slice(np.array(ks), np.array(ivs), T, iv_spread=np.array(sps))
    print(f"\n[4] SVI SLICE FIT  {'OK' if fit.ok else 'REJECTED: ' + fit.detail}")
    print(f"    rmse = {fit.rmse_vol * 100:.3f} vol pts | {fit.inside_spread_frac:.0%} of "
          f"quotes fit inside their spread")
    print(f"    min g(k) = {fit.min_g:.4f} (butterfly-free iff >= 0) | "
          f"Lee slope b(1+|rho|) = {fit.params.lee_slope:.3f} (cap 2.0)")
    print(f"    params: a={fit.params.a:.5f} b={fit.params.b:.5f} rho={fit.params.rho:.4f} "
          f"m={fit.params.m:.5f} sigma={fit.params.sigma:.5f}")
    if not fit.ok:
        print("    ABORT: unfittable / arbitrageable surface")
        return

    # ------------------------------------------------- 5. RV forecast + VRP
    iv_atm = float(np.sqrt(svi_w(0.0, fit.params) / T))
    # rv_forecast: HAR-RV log form, 22d horizon, Jensen-corrected
    vrp = variance_risk_premium(iv_atm, rv_forecast)
    print("\n[5] VARIANCE RISK PREMIUM")
    print(f"    ATM implied {iv_atm:.2%} | HAR-RV forecast {rv_forecast:.2%} | "
          f"VRP {vrp['vrp_vol_points']:+.2f} vol pts | IV/RV {vrp['iv_rv_ratio']:.2f}x")
    print(f"    -> {vrp['direction']}")

    scen = build_scenarios(fit.params, F, T, rv_forecast)
    print(f"    scenario grid: {len(scen.S_T)} nodes, "
          f"${scen.S_T.min():.0f} - ${scen.S_T.max():.0f}")

    # ------------------------------------ 6. structure enumeration + scoring
    # The selector and the sizer MUST optimise the same objective. Ranking
    # candidates by raw edge and then sizing them separately produces a
    # selector that keeps proposing trades the sizer refuses -- which is what
    # happens when a system "finds great trades" and never places any.
    # The shared objective is expected LOG GROWTH, which is what Kelly
    # maximises and what the mandate asks for.
    costs = SchwabCosts()
    print("\n[6] CANDIDATE FRONTIER  (ranked by expected log growth, the Kelly objective)")
    header = (f"    {'structure':<32}{'PoP':>7}{'mkt':>6}{'EV/spr':>9}{'EVann':>8}"
              f"{'c/w':>6}{'risk$':>8}{'n':>4}{'growth':>10}")
    print(header); print("    " + "-" * (len(header) - 4))

    rows = []
    for short_delta in (0.16, 0.22, 0.30, 0.38, 0.45):
        kk = np.linspace(-0.5, 0.2, 4001)
        sig_k = np.sqrt(np.maximum(svi_w(kk, fit.params), 1e-12) / T)
        dl = np.abs(black.greeks(SPOT, F * np.exp(kk), T, R, Q, sig_k, -1).delta)
        Ksh = round(float(F * np.exp(kk[np.argmin(np.abs(dl - short_delta))])) / 5) * 5
        for width in (5.0, 10.0, 15.0):
            Kl = Ksh - width
            if Ksh not in by_k or Kl not in by_k:
                continue
            st = vertical("SPY", EXPIRY, Right.PUT, Kl, Ksh,
                          q_long=by_k[Kl][Right.PUT], q_short=by_k[Ksh][Right.PUT])
            st.name = f"{Kl:.0f}/{Ksh:.0f} put credit vert ({short_delta:.0%}d)"
            net = st.net_price("marketable")
            fee = costs.estimate(st, 1)["total_open"]
            ed = score_structure(st, scen, net, fees=fee)
            if ed.max_loss <= 0 or not np.isfinite(ed.ev_per_spread):
                continue
            sz = size_position(ed.payoff, scen.prob_P, ed.max_loss, BANKROLL,
                               sharpe=0.55, years_of_evidence=2.0,
                               max_drawdown=0.30, drawdown_prob=0.10,
                               max_risk_fraction_per_trade=MAX_RISK_PER_TRADE)
            ctw = ed.max_gain / max(ed.max_gain + ed.max_loss, 1e-9)
            ann = ed.ev_pct_of_risk * 365.0 / (EXPIRY - TODAY).days
            rows.append((st, ed, net, fee, sz, ctw, ann))

    def _kw(st_, ed_, sz_):
        return dict(quote_age_s=900.0, slice_fit=fit,
                    legs_tradeable=[lg.quote.tradeable() for lg in st_.legs],
                    conformal_killed=False, calibration_rel=0.004,
                    portfolio_delta_dollars=0.0, portfolio_vega_dollars=0.0,
                    bankroll=BANKROLL, dte=(EXPIRY - TODAY).days,
                    days_to_earnings=None, open_positions_same_underlying=0,
                    credit_to_width=abs(st_.net_price("mid")) / (100.0 * abs(
                        st_.legs[0].strike - st_.legs[1].strike)),
                    round_trip_cost=2.0 * costs.estimate(st_, 1)["total_open"],
                    thresholds=Thresholds())

    # Screen with the gate stack FIRST, then rank the survivors by growth.
    best, best_dec, evaluated = rank_and_select(
        [(r[0], r[1], r[4]) for r in rows], _kw)

    by_name = {r[0].name: r for r in rows}
    order = sorted(evaluated, key=lambda x: (x[3].action is not Action.HOLD,
                                             x[2].expected_log_growth), reverse=True)
    for st_, ed_, sz_, d_ in order:
        r = by_name[st_.name]
        flag = "OK " if d_.action is not Action.HOLD else "-- "
        print(f"    {flag}{st_.name:<30}{ed_.pop:>6.1%}{ed_.pop_riskneutral:>6.1%}"
              f"{ed_.ev_per_spread:>9.2f}{r[6]:>8.1%}{r[5]:>6.0%}"
              f"{ed_.max_loss:>8.0f}{sz_.contracts:>4d}{sz_.expected_log_growth:>10.5f}")
    print(f"    {'':4s}(OK = clears every gate; -- = rejected, first blocker shown below)")
    for st_, ed_, sz_, d_ in order:
        if d_.action is Action.HOLD:
            print(f"    {'':4s}{st_.name:<30} rejected: {', '.join(d_.blocked_by[:3])}")

    if best is None:
        print("\n    NO CANDIDATE CLEARS THE GATE STACK -> HOLD")
        st, ed, sizing = rows[0][0], rows[0][1], rows[0][4]
        dec = evaluated[0][3]
    else:
        st, ed, sizing = best
        dec = best_dec
        print(f"\n    SELECTED: {st.name}")
        for k, v in ed.summary().items():
            print(f"      {k:<32} {v}")

    # ------------------------------------------------------- 7. Kelly sizing
    print("\n[7] POSITION SIZING")
    print(f"    {sizing.explain()}")
    c = max(sizing.c_total, 1e-9)
    print(f"    risk-of-ruin at c={sizing.c_total:.2f}: "
          f"P(-30%) = {0.7 ** (2 / c - 1):.2%}, P(-50%) = {0.5 ** (2 / c - 1):.2%}")
    if all(r[4].contracts == 0 for r in rows):
        cheapest = min(rows, key=lambda r: r[1].max_loss)
        print(f"    DIAGNOSIS: every candidate rounds to zero contracts. The 2% "
              f"per-trade cap on ${BANKROLL:,.0f} allows ${0.02 * BANKROLL:,.0f} of risk;")
        print(f"    the cheapest structure here risks ${cheapest[1].max_loss:,.0f}. "
              f"At SPY's ${SPOT:.0f} notional this is a capital constraint, not a")
        print("    model failure -- trade a lower-notional underlying, narrow the "
              "spreads, or raise the per-trade cap deliberately.")

    # -------------------------------------------------------- 8. decision
    print(f"\n[8] DECISION\n{dec.report()}")

    # ---------------------------------------------------------- 9. ticket
    print("\n[9] EXECUTION")
    n_show = dec.contracts if dec.contracts > 0 else 1
    ticket = build_ticket(st, n_show, SPOT, duration="DAY", costs=costs)
    if dec.action.value == "HOLD":
        print("    HOLD -- no order sent. Ticket rendered anyway so the audit "
              "trail records exactly what was declined.\n")
    print(ticket.render())


if __name__ == "__main__":
    for rg in ("calm", "stressed"):
        run(rg)
        print("\n")
