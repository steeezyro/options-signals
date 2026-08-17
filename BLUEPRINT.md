# Options Relative-Value Engine — Build Blueprint

**Target:** a defined-risk, volatility-relative-value options system for a $5k–$50k Reg-T
margin account, executing manually at Charles Schwab, fed by live MCP market data,
recalibrating online against its own realised errors.

**Status of this document:** engineering specification + the reasoning behind each choice.
Phases 0–14 are implemented, with **173 passing tests**. The remaining item is phase 15
(Schwab Trader API automation), which is deliberately not built — see §12.

Phases 9–14 were built against this specification and the build surfaced six defects that
only appear once the loop is closed. They are documented at their point of occurrence rather
than quietly fixed, because each one is a general trap:

| # | Defect | Where |
|---|---|---|
| 1 | `target_dte=(21,45)` selecting the NEAREST expiry against a 21-DTE time stop entered and stopped out in the same week — 3 trades, all losers, PoP stated 75% | §14 phase 9, `RunConfig.expiry_preference` |
| 2 | Slippage measured against the ticket's **mid-based** limit and fed back into every future EV, re-charging the whole mid-to-marketable spread until the system stopped trading permanently | §8.6, `FillRecord` |
| 3 | Realised Sharpe annualised from a 0.001-year span read −2367, driving `c_unc` to exactly 0 → 0 contracts forever, an absorbing state | §10, `FeedbackConfig.min_span_years_for_sizing` |
| 4 | The earnings blackout measured days from *today*, so a 32-DTE trade with earnings in 30 days passed the gate | §9, `risk.event_in_window` |
| 5 | `build_ticket`'s profit target used the same formula for debits and credits, turning a "50% profit target" into a 50% loss on debit structures | §12 |
| 6 | The SSVI backbone fallback was assumed uncrossed; it is not, when θ was repaired up to a refined neighbour's level | §5.3 |

A seventh is a **modelling** gap rather than a defect, and it is left open deliberately: PoP is
computed to expiry while the exit plan closes at 21 DTE, so stated PoP overstates the realised
win rate (78.1% vs 55.0% measured). The calibration gate detects it unaided; §16.7 explains why
the fix is deferred rather than applied.

---

## 0. The one-paragraph version

The system does not predict direction. It measures the gap between what options *imply*
about future variance and what a realised-volatility forecast *expects*, expresses that gap
as a signed edge, converts it into a scenario P&L distribution for a small set of
defined-risk structures, sizes the best one by fractional Kelly under explicit uncertainty
and drawdown constraints, and either emits a fully-specified Schwab order ticket or — much
more often — emits HOLD with a complete list of which gate stopped it. Every realised
outcome is journaled and fed back into a Kalman filter, an adaptive-conformal interval, and
a probability calibrator, so the *stated* uncertainty stays honest as the market changes.

---

## 1. What the data plane actually is

Before any architecture, the binding constraint. Probed live on 2026-08-17 against your
connected MCP servers:

### Massive Market Data — **entitled**

| Endpoint | Use |
|---|---|
| `/v2/aggs/ticker/{t}/range/...`, `/prev` | daily & intraday OHLCV — **verified working** (SPY prev close 776.34) |
| `/fed/v1/treasury-yields` | risk-free curve 1M–30Y, per-expiry discounting |
| `/stocks/v1/dividends` | ex-dates and amounts → discrete-dividend American pricing |
| `/tmx/v1/corporate-events` | earnings dates → event blackout gate |
| `/v2/reference/news`, `/benzinga/v2/news` | news + sentiment |
| `/stocks/v1/short-interest`, `/short-volume` | positioning |
| `/v1/marketstatus/now`, `/upcoming` | session state, holidays |
| server-side functions | `bs_price`, `bs_delta/gamma/theta/vega/rho`, `vanna`, `volga`, `charm`, `color`, `veta`, `ema`, `sharpe_ratio`, `sortino_ratio` |
| `query_data` | SQL (incl. FTS5) over stored responses |

### Massive Market Data — **NOT entitled**

```
GET /v3/snapshot/options/SPY
  -> 403 {"status":"NOT_AUTHORIZED","message":"You are not entitled to this data."}
```

The **entire options plane** — chain snapshots, options quotes, options trades — requires a
plan upgrade. This is the single most consequential fact about the build.

### yfinance MCP — **entitled**

Option chains with strike / bid / ask / last / volume / open interest / IV / ITM flag, plus
price history, ticker info, news, screeners.

### The three consequences, designed for rather than papered over

**(a) Vendor IV is never consumed.** Direct observation from the live SPY 2026-09-18 put
chain: strikes 300, 305, 310 all report `impliedVolatility: 0.500005` with `bid: 0.0,
ask: 0.0, openInterest: 0`. That is a sentinel, not a measurement. Feeding it to a surface
fitter produces a flat 50-vol wing that looks like a real volatility smile and is entirely
fictional. So: the system stores `iv_vendor` for diagnostics and **inverts every volatility
itself** from mid price (§4.2), rejecting unquotable strikes explicitly rather than
imputing them.

**(b) Latency is a gate, not a nuisance.** ~15-minute delayed chains rule out 0DTE, gamma
scalping, and anything whose edge decays inside the quote age. They do not rule out 21–45
DTE premium selling, where the edge is a multi-day variance premium and 15 minutes is
immaterial. The freshness gate is set to the feed's *real* latency and the strategy universe
is chosen to be latency-tolerant. This is a deliberate scope decision, not a limitation
worked around.

**(c) The forward comes from the option market, not the feed.** Put-call parity regression
on the liquid core recovers F and DF (§4.4). A stale or wrong spot would otherwise tilt the
entire slice and be misread as skew.

### The seam

`data/provider.py` defines a `MarketDataProvider` Protocol; `data/mcp_adapters.py`
implements `YFinanceChainProvider`, `MassiveUnderlyingProvider`, and a `CompositeProvider`
that routes chains to one and everything else to the other. Upgrading to a real options
feed — Massive plan upgrade, Schwab's own market-data API, Polygon, Databento — means one
new adapter and one config line. **Nothing above the data layer knows where quotes came
from.** Build this seam first; it is the difference between an upgrade and a rewrite.

`RecordingProvider` journals every response to disk. **That journal is the backtest
dataset.** Backtesting against reconstructed history is the standard route to a strategy
that works offline and not live, because the reconstruction silently repairs the gaps,
stale prints and sentinels the live system has to survive.

---

## 2. Architecture

```
                        ┌───────────────────────────────────────┐
                        │  MCP: Massive (underlying/rates/divs/  │
                        │  events)  +  yfinance (option chains)  │
                        └────────────────┬──────────────────────┘
                                         │
   ┌─────────────────────────────────────▼─────────────────────────────────────┐
   │ data/    provider.Protocol · mcp_adapters · assess_quality · Recording     │
   │          ↳ crossed books, zero bids, sentinels, staleness → REJECT         │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         │
   ┌─────────────────────────────────────▼─────────────────────────────────────┐
   │ pricing/ forward (put-call parity)  →  american (LR tree, de-Americanise)  │
   │          →  implied (in-house IV inversion)  →  black (price + 14 Greeks)  │
   └─────────────────────────────────────┬─────────────────────────────────────┘
                                         │
   ┌─────────────────────────────────────▼─────────────────────────────────────┐
   │ surface/ SSVI backbone (2 global params, closed-form no-arb)               │
   │          + per-slice raw SVI, quasi-explicit calibration                   │
   │          ↳ g(k) ≥ 0 butterfly gate · crossedness calendar gate → REJECT    │
   └────────────────┬──────────────────────────────────┬───────────────────────┘
                    │                                  │
   ┌────────────────▼─────────────┐   ┌────────────────▼───────────────────────┐
   │ forecast/ realized (Yang-    │   │ edge/  risk-neutral density (B-L)      │
   │           Zhang, bipower,RQ) │──▶│        → physical reweight (VRP)       │
   │           har (HAR/HARQ log) │   │        → scenario P&L → PoP/EV/RoC/CVaR│
   └──────────────────────────────┘   └────────────────┬───────────────────────┘
                                                       │
   ┌───────────────────────────────────────────────────▼───────────────────────┐
   │ sizing/  scenario multi-outcome Kelly                                      │
   │          × uncertainty shrinkage × drawdown cap × ½-Kelly floor × caps     │
   └───────────────────────────────────────────────────┬───────────────────────┘
                                                       │
   ┌───────────────────────────────────────────────────▼───────────────────────┐
   │ policy/  17-gate stack → BUY / SELL / HOLD  (+ rank_and_select)            │
   └───────────────────────────────────────────────────┬───────────────────────┘
                                                       │
   ┌───────────────────────────────────────────────────▼───────────────────────┐
   │ execution/ Schwab order ticket · web + thinkorswim click path · cost model │
   │            · price ladder · exit plan · assignment warnings · API JSON     │
   └───────────────────────────────────────────────────┬───────────────────────┘
                                                       │  realised outcome
   ┌───────────────────────────────────────────────────▼───────────────────────┐
   │ learning/ Kalman TVP · adaptive conformal · Platt/isotonic · ADWIN/PH      │
   │           ↳ feeds thresholds, uncertainty, and the kill switch back up     │
   └────────────────────────────────────────────────────────────────────────────┘
```

**Design rule enforced throughout: each stage refuses to run on output the previous stage
flagged bad.** There is no "best effort" path. A system that degrades silently is worse
than one that stops, because you cannot tell the difference between a real signal and a
data artefact from the P&L alone until it is expensive.

---

## 3. Repository map

```
optionsmarkets/
├── BLUEPRINT.md                    this document
├── README.md                       quickstart
├── pyproject.toml                  numpy/scipy/pandas; extras: live, fast, learn, dev
├── examples/end_to_end.py          single-slice walkthrough, two regimes, no network
├── src/optionsmarkets/
│   ├── cli.py                      decide / collect / backtest / journal / learned
│   ├── app/
│   │   ├── config.py               RunConfig -- every policy dial in one place
│   │   └── pipeline.py             THE RUNNER: provider in, decision + ticket out   [9]
│   ├── pricing/
│   │   ├── black.py                BSM price + 14 Greeks, single unit-conversion boundary
│   │   ├── implied.py              erfcx-normalised Black, Householder(3) + bracket
│   │   ├── american.py             Leisen-Reimer, Bjerksund-Stensland 2002, de-Americanise
│   │   └── forward.py              F and DF from put-call parity regression
│   ├── surface/
│   │   ├── svi.py                  raw SVI, g(k), SVI-JW, quasi-explicit fit, crossedness
│   │   ├── ssvi.py                 SSVI backbone, GJ Thm 4.1/4.2 conditions, global fit
│   │   └── joint.py                SSVI envelope + slice refinement, publication gate  [11]
│   ├── forecast/
│   │   ├── realized.py             Yang-Zhang, GK, RS, bipower, RQ, daily + semivariance
│   │   ├── har.py                  HAR / HARQ / SHAR, log form, Newey-West, Jensen
│   │   ├── garch.py                GJR-GARCH(1,1), asymmetry cross-check            [6.2]
│   │   └── evaluate.py             QLIKE, Mincer-Zarnowitz, Diebold-Mariano, coverage [13.3]
│   ├── edge/score.py               Q→P reweight, scenarios, PoP/EV/RoC/CVaR/edge-z
│   ├── sizing/kelly.py             scenario Kelly + shrinkage + drawdown + hard caps
│   ├── learning/
│   │   ├── online.py               Kalman TVP, ACI, calibration, Brier, ADWIN, Page-Hinkley
│   │   └── feedback.py             journal → learners → LearnedState → sizing/gates   [10]
│   ├── policy/decide.py            gate stack, Thresholds, rank_and_select
│   ├── execution/schwab.py         ticket, cost model, click paths, OPEN_QUESTIONS
│   ├── risk/
│   │   ├── portfolio.py            dollar Greek aggregation, limit checks
│   │   └── factors.py              betas, effective bets, portfolio Kelly            [13]
│   ├── domain/
│   │   ├── structures.py           contracts, legs, verticals/condors/flies/calendars
│   │   └── candidates.py           delta-targeted enumeration of all families        [14]
│   ├── backtest/
│   │   ├── engine.py               journal replay under the §13.4 rules              [12]
│   │   └── metrics.py              log growth, drawdown, Sortino, DEFLATED Sharpe
│   └── data/
│       ├── provider.py             MarketDataProvider Protocol + quality screen
│       ├── mcp_adapters.py         yfinance / Massive MCP adapters, RecordingProvider
│       ├── yfinance_direct.py      in-process live provider (no MCP needed)          [9]
│       ├── synthetic.py            known-truth provider WITH the feed's real defects
│       ├── journal.py              append-only outcome journal                       [10]
│       └── replay.py               replay a recording; no-look-ahead by construction [12]
└── tests/                          173 tests
```

---

## 4. The math core

### 4.1 Black-Scholes-Merton and the Greek set

`pricing/black.py` returns 14 sensitivities: delta, gamma, vega, theta, rho, vanna, volga,
charm, veta, speed, zomma, color, ultima, dual-delta. All verified against central finite
differences to ≤2e-4 relative (`test_greeks_match_finite_difference`).

Two conventions that are enforced rather than assumed, because both are classic sources of
silent error:

**All time Greeks are d/dt** with t = calendar time moving forward (so dT = −dt). Sanity
check baked into a test: for a long ATM option, value falls (θ<0), vega falls (veta<0),
gamma rises (color>0). Mixing dT and dt conventions across modules produces a theta that is
right and a veta that is backwards, and nothing crashes.

**Unit conversion happens exactly once**, in `scale_for_trader()` — vega/vanna per vol
*point*, theta/charm/veta/color per calendar *day*. The single most common cause of a
100× position-sizing error in an options system is two modules disagreeing about whether
vega is per 1.00 or per 0.01 of volatility.

`dual_delta = ∂V/∂K = −cp·DF·N(cp·d₂)` is included deliberately: it *is* the risk-neutral
CDF, and it is what the market's own probability-of-touch/PoP numbers are computed from.
Having it lets the system show the market's PoP next to its own (§7).

### 4.2 Implied volatility inversion

Reference implementation is Jäckel's *Let's Be Rational* — two Householder iterations to
full double precision for every representable input. The repo implements a self-contained
version with the same structural ideas so there is no hard dependency, and so the two can
be cross-validated:

- **Normalised Black through `erfcx`.** With x = ln(F/K), s = σ√T, h = x/s, t = s/2:

  ```
  b(x,s) = ½ · exp(−(h² + t²)/2) · [ erfcx(−(h+t)/√2) − erfcx(−(h−t)/√2) ]
  ```

  One exponential, no cancellation. The naive `N(h+t)e^{ht} − N(h−t)e^{−ht}` form loses
  every significant digit past ~4 standard deviations OTM — which silently poisons exactly
  the wings a short-premium book lives on.

- **Manaster-Koehler seed** at s_c = √(2|x|), the vega-maximising vol, from which Newton
  converges monotonically on the OTM branch.

- **ITM → OTM map** with a **catastrophic-cancellation guard**. For a deep-ITM quote almost
  all the price is parity; if what survives `price − (F−K)` is within a few ulps of the
  operands it is rounding noise, not time value. Inverting it produces a confident,
  completely fictitious volatility — empirically a 1-day deep-ITM call reporting a 200 vol.
  The guard rejects with status `UNDERFLOW`.

- **Objective in ln b, not b**, so one formulation covers both the exponentially-flat
  low-price regime and the well-behaved one above the inflection point.

- **Maintained bracket + Brent backstop.** Every evaluation tightens `[lo, hi]`, so a
  guaranteed fallback is always available.

**Measured result:** across every quote with ≥ $0.01 of time value spanning 1 day to 1 year,
0.6× to 1.6× moneyness, 8 to 150 vol, calls and puts — **1324 cases, 0 rejections, worst
absolute error 3.3e-12**. Mean 5.4 iterations, ~190 µs/inversion in pure Python. Install
the `fast` extra (`py_lets_be_rational` + numba) for ~100× if throughput ever matters.

**Explicit rejection statuses, never imputation:**

| Status | Meaning |
|---|---|
| `NO_ARBITRAGE` | price below discounted intrinsic or above the upper bound |
| `UNDERFLOW` | true price below double precision, or ITM cancellation destroyed all signal |
| `NO_VEGA` | `spread/(2·vega)` exceeds `max_spread_vols` (default 5 vol points) — the market is not quoting a volatility here and any number is a tick-size artefact |
| `NOT_CONVERGED` | bracket exhausted (unreachable in testing) |

The `NO_VEGA` filter is the single most valuable line in the pipeline. It is what stops the
surface fit from chasing the mid of a 0.05/0.60 quote.

### 4.3 American exercise and discrete dividends

US single-name and ETF options are American on underlyings paying discrete cash dividends,
and are not dividend-protected. Inverting a European formula on an American quote gives a
biased vol, and the bias is largest exactly where the early-exercise premium is largest.

**Engines:**

| Engine | Role | Accuracy |
|---|---|---|
| **Leisen-Reimer**, odd n, 101–201 steps | production mark | matches 5000-step CRR to well under a cent |
| **Bjerksund-Stensland 2002** | fast screen, Jacobians | ~1–3 cents; validated against Haug's published 5.2704 reference to 5e-3 |

LR chooses u/d so the terminal binomial distribution is a high-order Peizer-Pratt
approximation to the lognormal with the strike at its centre. European error is O(1/n²) and
— critically — smooth and monotone in n rather than CRR's sawtooth, which is what makes
Richardson extrapolation legitimate at all. **n is forced odd**: the Peizer-Pratt correction
is only centred when the terminal node count n+1 is even; with even n the strike lands on a
node, the correction misaligns, and the convergence order is destroyed. A test asserts
`n=100` and `n=101` give identical results.

**Discrete dividends use Vellekoop-Nieuwenhuis:** the lattice stays recombining and the
post-dividend value at an ex-date is obtained by interpolating the continuation function at
S−D. Subtracting D from node prices directly destroys recombination and makes the tree
O(2^m) in the dividend count.

**Early-exercise screen.** A dividend can never make early exercise of an American call
optimal at that ex-date unless

```
D_i  >  K · (1 − exp(−r·(t_{i+1} − t_i)))
```

— the dividend must exceed the interest earned on the strike over the remaining sub-period.
Note the regime dependence, which the code comments call out: at r ≈ 4–5% this binds often
for ITM calls on high-yield names; at ZIRP it is ≈0 and almost every dividend triggers the
full comparison.

**De-Americanisation loop** (`de_americanise`), the industry-standard path from an American
quote to a surface-legal European IV:

1. Solve σ_A such that `LR_american(σ_A) = market price`
2. Reprice the **European** option at that same σ_A with the same dividends
3. Invert that synthetic European price → the IV handed to the SVI fit

Model-dependent but internally consistent, which is what makes the resulting surface
arbitrage-*checkable*. Feeding raw American IVs into SVI produces butterfly-test failures
that have nothing to do with the market. Tested: de-Americanising an American put quote
recovers σ to 1e-4 and produces a European IV strictly below the naive one.

### 4.4 Forward and discount factor from the option market

```
C − P = DF·(F − K)     →   regress (C−P) on K:  slope = −DF,  intercept = DF·F
```

Weighted least squares, weights `1/(spread_C + spread_P)`, restricted to strikes within 8–10%
of spot. Parity holds at every strike in theory; in practice the wings have wide spreads and
stale quotes and a single bad far-OTM pair rotates the whole regression. Reported
diagnostics: R², residual in basis points of F, implied r, n pairs. **Measured on a
synthetic chain: F recovered to 1e-3 of a point, DF to 1e-8, R² = 0.99999999.**

---

## 5. The volatility surface

### 5.1 Why a parametric surface at all

You need a continuous, arbitrage-free σ(K,T) for three things a discrete chain cannot give
you: the risk-neutral density (§7), interpolation to strikes that are not listed, and a
*rejection criterion*. The last is the important one. An arbitrage check is the only
automatic test that distinguishes "the market moved" from "my data is wrong."

### 5.2 Raw SVI per slice

```
w(k) = a + b·( ρ(k−m) + √((k−m)² + σ²) )     k = ln(K/F), w = σ_BS²·T
```

Domain: b ≥ 0, |ρ| < 1, σ > 0, a + bσ√(1−ρ²) ≥ 0, and the Roger Lee moment bound
**b(1+|ρ|) ≤ 2** in total-variance units.

**Butterfly arbitrage — the Gatheral-Jacquier g(k):**

```
g(k) = (1 − k·w′/(2w))² − (w′²/4)(1/w + ¼) + w″/2
```

The slice is butterfly-free iff `g(k) ≥ 0` everywhere. `g` is not decoration: the
risk-neutral density is `p(k) = g(k)/√(2πw) · exp(−d₋²/2)`, so `g < 0` is literally negative
probability mass — a butterfly you could buy for a negative price.

**The point GJ make and this system enforces:** satisfying the parameter domain does **not**
imply butterfly-freedom. A test constructs `SVIParams(a=0.001, b=0.9, ρ=−0.95, m=0, σ=0.02)`
which passes every domain check and has `min g = −7.05`. This is the entire motivation for
an SSVI backbone.

**Calendar arbitrage** is checked by crossedness: `max_k (w_near(k) − w_far(k))⁺`. Total
variance must be non-decreasing in maturity at every k; plotted as w vs k, slices must never
cross. Any positive value is a calendar spread with a negative price.

**Calibration — Zeliade quasi-explicit.** After y = (k−m)/σ the model is *linear* in three of
its five parameters:

```
w̃(y) = ã + d·y + c·√(y²+1),      c = bσ,  d = ρbσ
```

so for fixed (m,σ) the inner problem is a 3-parameter convex least squares on a polytope —
solved exactly — and the outer problem is only a 2-D search. Naive 5-parameter nonlinear
least squares finds a local minimum often enough that it cannot be run unattended.

**Loss function: bid-ask-normalised with an epsilon-insensitive core.** Residuals are divided
by the quoted half-spread; a model vol landing inside the spread contributes zero loss. This
is what most desks actually run, because it stops the fit from chasing the mid of a wide
quote. Vega weighting is the fallback when spreads are unavailable.

**Measured:** recovers known parameters from 41 noisy quotes to ≤0.01 absolute on every
parameter, RMSE 0.0011 vol, 100% of quotes fit inside their spread, density integrates to
0.999 and is non-negative.

### 5.3 SSVI backbone

```
w(k,θ) = (θ/2)·{ 1 + ρφ(θ)k + √((φ(θ)k + ρ)² + (1−ρ²)) },   θ_T = w(0,T)
```

with the **modified power law** `φ(θ) = η / (θ^γ (1+θ)^{1−γ})`, γ = ½ default. This is the
only common choice that is butterfly-free at *every* maturity, under the single inequality
`η(1+|ρ|) ≤ 2`. The plain power law fails calendar-freedom beyond a finite maturity — a nasty
failure mode where the front months look fine and the back silently goes arbitrageable.

No-arbitrage conditions are enforced by the *search bounds*, not by a penalty: η is
optimised on `(0, 2/(1+|ρ|)]`, so an infeasible surface is unreachable rather than merely
discouraged. **Measured: recovers ρ=−0.7, η=0.9 to 3 decimal places.**

The whole surface is two global numbers plus an observed ATM curve. **Production
architecture is SSVI as the arbitrage-free skeleton + per-slice raw-SVI refinement
constrained to stay inside the SSVI envelope**, fitted shortest-maturity-first with explicit
crossedness constraints. SSVI alone is too rigid to fit a real equity surface to bid/ask
across all slices; independent slice fits develop local pathologies invisible until you try
to price a calendar. **Implemented** in `surface/joint.py`: SSVI is fitted globally, then each
slice is refined shortest-maturity-first by the same quasi-explicit scheme with the previous
published slice entering the objective as a one-sided calendar penalty and the SSVI slice as
an envelope (default ±15% of total variance) the refinement may not leave.

The SSVI↔raw-SVI map is exact rather than a fit:

```
a = θ(1−ρ²)/2,   b = θφ/2,   ρ_raw = ρ,   m = −ρ/φ,   σ = √(1−ρ²)/φ
```

verified to 1e-14. Under it the Lee bound `b(1+|ρ|) ≤ 2` **is** the GJ Thm 4.2 condition
`θφ(1+|ρ|) ≤ 4` — the two are the same statement, which is the coherence check that the map
is right.

⚠️ **The fallback is not automatically safe.** A slice that fails refinement falls back to the
backbone, but the backbone slice can still cross its neighbour: θ may have been repaired
upward by the running-max to equal a neighbour whose own slice was refined further up to fit
its quotes. The fit therefore *scores* both candidates on (feasible, crossedness, rmse) and
takes the least-bad, marking the surface unpublishable rather than repairing it when neither
is feasible. Maturity interpolation is linear in **total variance** at fixed k, which is the
only common scheme that cannot itself create a calendar arbitrage.

**Publication gate.** Recompute `min_k g(k)` per slice and `max_k crossedness` for all
adjacent pairs. On violation, fall back to the last known-good surface and raise. **Never
publish an unvalidated surface.**

---

## 6. Volatility forecasting

### 6.1 Realised-volatility estimators

With daily OHLC and no tick data, the workhorse is **Yang-Zhang**:

```
σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1−k)·σ²_RS,   k = 0.34/(1.34 + (n+1)/(n−1))
```

US equities gap overnight. Parkinson and Garman-Klass ignore the overnight move entirely and
systematically understate total volatility on exactly the population an options book cares
about — names with earnings and news. Yang-Zhang is drift- *and* gap-independent and up to
~14× more efficient than close-to-close.

Known bias, documented in the module: all range estimators assume the continuous high/low is
observed. Discrete sampling biases the range down, so these read 5–15% low for liquid names.
Calibrate a multiplicative correction against 5-minute RV on a liquid subset before trusting
the *level*; the *changes* are far less affected.

Also provided: bipower variation (jump-robust IV estimate) and realised quarticity.

### 6.2 HAR / HARQ

```
ln RV_{t+h} = c + β_d·ln RV^(d)_t + β_w·ln RV^(w)_t + β_m·ln RV^(m)_t + ε
```

**Log form by default.** Residuals closest to Gaussian and homoskedastic, no positivity
constraint, and — the reason it matters here — its forecast errors behave well enough for the
downstream conformal intervals and Kelly shrinkage to mean anything. Converting back to
levels applies the **Jensen correction** `E[RV] = exp(μ̂ + s²/2)`; omitting it biases every
forecast low by half the residual variance, which for daily equity vol is not small.

**Newey-West standard errors, lag = 1.5h.** The overlapping weekly/monthly aggregates induce
strong serial correlation and OLS standard errors run 2–3× too tight. Those standard errors
feed the Kelly shrinkage factor — understating them means overbetting.

**HARQ option** interacts the daily term with demeaned √RQ. The coefficient comes out
negative: on days when yesterday's RV was badly measured the model automatically shifts
weight to the weekly and monthly components. Without it, classical errors-in-variables
attenuates the RV_{t−1} coefficient, and the attenuation is worst exactly on the days you
most need the forecast to be right.

**Horizon matching is mandatory.** `HARModel(horizon=h)` puts the *h-day average* RV on the
left-hand side (direct forecasting). Forecasting 1-day RV to trade a 30-day option is a
horizon mismatch no amount of model quality repairs.

**Convention flag.** `include_today_in_week` selects between the published JFE version
(includes RV_t in the weekly average) and the JAE working paper (starts at t−1). Both are in
circulation; the choice must be explicit and stable, because flipping it mid-backtest
silently changes every coefficient.

**SHAR and GJR-GARCH are implemented** (`forecast/har.py::SHARModel`, `forecast/garch.py`).
For equity index vol a model with no asymmetry term is wrong: representative SPX estimates are
α ≈ 0.02, ξ ≈ 0.12, β ≈ 0.90 — the leverage effect dominates the symmetric ARCH effect
outright. Measured on simulated data with those parameters, GJR recovers α=0.016, ξ=0.126,
β=0.901, with ξ/α = 7.8.

Two things the implementations are careful about:

**GJR's forecasting persistence is α + ξ/2 + β, not α + ξ + β.** The indicator fires half the
time under a symmetric error distribution, so only half of ξ enters the expected recursion.
Using the full ξ overstates persistence by ~0.06 for SPX-like parameters and compounds over a
30-day horizon into materially too-slow mean reversion.

**Semivariances from daily OHLC must be strictly positive.** The obvious construction — label
the whole day's variance by the sign of its close-to-close return — yields a semivariance that
is exactly **zero on half the days**, so the log-form HAR takes `log(0)`. Floored to a
constant, the two columns go near-collinear and the estimated asymmetry collapses: measured at
**t = 0.05 on data simulated with a strong leverage effect** — the model looked fine and had
silently lost the only effect it exists to capture. `realized.semivariance_proxy` instead
assigns the overnight gap by its sign and splits the Rogers-Satchell intraday variance in
proportion to the squared up/down excursions from the open, with a 5% floor on each side's
share (a trending day whose low equals its open still had down-ticks). The two components sum
**exactly** to `daily_variance_proxy`, so SHAR properly nests HAR, and the same fixture now
gives t = 2.96.

---

## 7. Edge: from surface to a tradeable expectation

### 7.1 The Q → P step is where the edge lives

The risk-neutral density prices options. It does not describe what the underlying will do.
A system that computes "probability of profit" from the risk-neutral density is measuring
the market's price, not its own forecast, and will correctly conclude it has no edge
anywhere.

The gap between the two measures is the **variance risk premium**, and for SPX it is large
and persistently positive. Bollerslev-Tauchen-Zhou (1990–2005 monthly, %² units): mean IV
40.87, mean RV 19.47, mean VRP 21.40 — implied variance roughly **2× realised**, about **5–7
volatility points**. Robustly positive; it goes sharply and briefly negative during vol
spikes (Aug 2007, Oct 2008, Feb 2018, Mar 2020) when realised blows through implied.

**Reweighting method** (`edge/build_scenarios`): keep the *shape* of the risk-neutral
density — its skew and kurtosis are real information about the market's fear — and rescale
its width so the second moment matches the horizon-matched RV forecast, then recentre on the
forward. Deliberately conservative: it assumes the market is right about shape and possibly
wrong about level, which is the part we actually have a forecasting model for.

**Tail inflation (default 1.25×).** Kelly assumes continuous rebalancing; a short-gamma
position cannot be trimmed through a gap, so the effective worst case is worse than any
fitted density suggests. Extending the scenario grid beyond anything the forecast implies is
cheap insurance against sizing off a distribution that has never seen a limit-down open.

### 7.2 What gets reported per structure

| Metric | Definition | Why |
|---|---|---|
| **PoP (physical)** | P(P&L > 0) under the reweighted density | the number the mandate asks to maximise |
| **PoP (risk-neutral)** | same, under Q | the market's own number — the benchmark |
| **PoP edge** | difference, in points | if this is ~0 you have no view, only a fee |
| **EV / spread** | Σ p·PnL, at *marketable* prices, net of fees | |
| **EV as % of capital at risk** | expected return on capital | |
| **Annualised EV on risk** | ×365/DTE | growth rate is what Kelly maximises |
| **Return on capital (max)** | max gain / max loss | the leverage the mandate asks about |
| **CVaR(5%)** | mean P&L in the worst 5% | what the max-loss number hides |
| **Edge z** | EV / σ(P&L) | signal-to-noise of the whole structure |

**Every price is marketable, never mid.** `net_price("marketable")` buys at the ask and sells
at the bid. Scoring at mid manufactures edge that the fill takes straight back — on a 4-leg
condor at retail spreads that is frequently the entire modelled edge. This one choice moves
several candidates in the worked example from apparently-profitable to actually-negative.

---

## 8. The learning loop

The requirement was: *learn from mistakes in real time, driving error toward zero as t → ∞*.
Here is the honest decomposition of what that can and cannot mean, followed by what the
system actually does.

### 8.1 What converges, what does not, and what actively decays

| Quantity | Converges to 0? | Why |
|---|---|---|
| **Calibration error** (stated 70% happens 70%) | **Yes** — provably | ACI's guarantee is deterministic: \|(1/T)Σerr_t − α\| ≤ (max(α₁,1−α₁)+γ)/(γT) → 0. No assumptions at all. |
| **Coverage error** of prediction intervals | **Yes** | same result, under *arbitrary* distribution shift |
| **Model-specification error** vs. the best model in your class | Asymptotically, in stationary regimes | online regret bounds: O(√T) for OGD, O(log T) strongly convex |
| **Prediction error** (forecast RV vs realised RV) | **No** | bounded below by irreducible conditional variance. RV is *stochastic*, not a deterministic function of observables. |
| **Edge** (VRP capture per unit risk) | **No — and it decays** | competition. Any exploitable premium attracts capital. |

Anyone who tells you a live trading model's error rate converges to zero is describing
either a calibration metric or a backtest. **The system therefore optimises the things that
genuinely converge, and monitors the things that do not so it can stand down when they
deteriorate.** That is the strongest honest form of "learns from its mistakes."

### 8.2 Time-varying coefficients — Kalman, not RLS

```
β_t = μ + Φ(β_{t−1} − μ) + η_t,   η ~ N(0,Q)
y_t = x_t′β_t + e_t,              e ~ N(0,R)
```

Joseph-form covariance update, mean-reverting transition (Φ<1) rather than a random walk —
real financial coefficients do not wander to infinity and a random-walk prior lets them.

**Why Kalman and not RLS-with-forgetting**, stated explicitly because they look equivalent:
they are the same algorithm up to how the covariance is inflated — RLS multiplies by 1/λ,
Kalman adds Q. The multiplicative form **winds up unboundedly in directions the data does
not excite**, and an options model routinely goes hours with no informative observation. The
next informative sample then produces a huge destabilising parameter jump. The additive form
converges to a finite steady state set by Q and R. This is the difference between a model
that degrades gracefully over a quiet afternoon and one that detonates on the first trade
after lunch.

The only real free parameter is the signal-to-noise ratio q = Q/R. Set it by maximising the
filter log-likelihood on history.

**Free diagnostic:** standardised innovations v_t/√S_t should be iid N(0,1). Running variance
above 1 ⇒ model too rigid (raise q); below ⇒ too loose. `innovation_health()` reports this
verdict every step, and that stream is itself a concept-drift signal.

**Tested:** tracks a step change from β=+1 to β=−2 within 0.2; covariance stays symmetric
positive-definite over 3000 updates.

### 8.3 Adaptive conformal inference — the honest uncertainty

```
α_{t+1} = α_t + γ·(α − err_t)
```

Maintains long-run marginal coverage under arbitrary distribution shift, no exchangeability
assumption — the right framework for a live trading model, because market data is emphatically
not exchangeable. γ = 0.005–0.05; use the top of that range if the model must survive regime
breaks without going stale for weeks.

**Why this matters for sizing specifically:** a model whose stated 90% interval actually covers
60% will overbet by a factor that no amount of Kelly fractioning repairs. The interval the
sizer consumes has to be honest *by construction over time*.

**Kill switch.** A sustained run of α_t ≤ 0 means only an infinite-width interval achieves
coverage — the model has broken. That is a kill state, not a wide interval, and it is a hard
gate in the policy (`model.conformal`). **Tested:** restores 90% coverage within 6 points
after a 6× volatility regime break; kill switch fires on a hopeless model.

### 8.4 Probability calibration

Platt scaling below ~1000 calibration points (with Lin-Weng-Keerthi target smoothing, which
prevents the degenerate A→−∞ fit on separable data), isotonic above.

**Why this module exists:** Kelly is a function of the probability *level*, not the ranking.
A model can have excellent AUC and be badly miscalibrated, and sizing off an uncalibrated
probability is the fastest route to systematic overbetting. The metric that matters is
**REL**, the reliability term of the Murphy decomposition `BS = REL − RES + UNC` — the squared
vertical distance from the diagonal on a reliability diagram. It is a hard gate
(`model.calibration`, limit 0.02).

Log loss is the model-selection criterion, because it is the proper scoring rule that
*matches* the Kelly objective — both are logarithmic.

### 8.5 Drift detection

- **ADWIN on the loss stream.** Bounded false-positive rate with no threshold tuning, and the
  surviving window length doubles as the correct adaptive lookback for refitting.
- **Page-Hinkley / CUSUM on specific monitored scalars** where you know the shift magnitude
  you care about: realised-vs-implied spread, fill ratio, slippage per contract.

Run both. Drift in the feature distribution (covariate shift) and drift in the conditional
(concept drift) require different responses, and monitoring only the loss conflates them.

### 8.6 The outcome journal — what actually closes the loop

**Implemented** in `data/journal.py` (append-only JSONL) and `learning/feedback.py`. Every
decision writes a record: timestamp, full quote snapshot, surface params and diagnostics,
forecast + interval, all gate values, the chosen structure, the whole candidate frontier, the
sizing chain, the ticket, and later the *fill* and the *realised outcome*. **HOLD decisions are
journaled too** — the distribution of which gate blocked is the only way to tell "no
opportunity existed" from "one threshold is set so tight nothing can clear it", and §13.4
requires it. Then:

| Feed | Consumer |
|---|---|
| realised RV vs forecast | Kalman update on HAR coefficients; ACI update |
| realised P&L vs EV | edge-model coefficients; Sharpe estimate → uncertainty shrinkage c_unc |
| profitable? (binary) | probability calibrator; Brier reliability → `model.calibration` gate |
| fill price vs ticket limit | slippage model → the marketable-price assumption |
| all of the above | ADWIN → refit trigger and lookback |

Note the second row is self-correcting in a way worth stating: `c_unc = 1/(1 + 1/(S²T))` uses
the **realised out-of-sample** Sharpe and sample length. A strategy that stops working
shrinks its own position size automatically, without anyone changing a threshold.

---

## 9. Risk: Greeks as the control surface

Direction is a *residual*, not a bet. The system is long or short volatility; its delta is
something to be managed.

**Position Greeks** (`Structure.greeks`) are summed across legs with sign and quantity, each
leg priced at the surface volatility for its own (k,T) — **not a single flat vol**. A vertical
priced with one vol has zero vega by construction, which is exactly backwards: the whole point
of a spread is the differential exposure across the smile.

**Portfolio limits, enforced as gates:**

| Limit | Default | Rationale |
|---|---|---|
| \|net delta $\| / bankroll | 15% | you are not paid for direction |
| \|net vega $\| / bankroll | 2% | the actual factor exposure |
| positions per underlying | 2 | 40 short-vol positions on 40 names is *one* bet |
| DTE window | 7–60 | below 7, gamma and pin risk dominate the modelled edge |
| earnings blackout | ±2 days | the IV crush is a different trade with different math |
| defined risk | mandatory | max loss must be contractual, not estimated |

**The correlation point deserves emphasis.** Kelly assumes independence across bets. If you
are short vol on 40 names you have one bet, not 40. Compute Kelly on the **factor** exposure
(market vol, dispersion), not the leg count. **Implemented** in `risk/factors.py`:
`estimate_betas` (with explicit identification flags — an unidentified beta defaults to 1.0,
the conservative direction, since overstating co-movement shrinks the book while understating
it blows the book up), `FactorModel.exposure` (reporting the **effective number of independent
bets** from the empirical correlation of the scenario P&Ls), and `portfolio_kelly`.

The measurement that matters is the **overbet multiple**. Five identical short-vol spreads
score 1.00 effective bets, and sizing them independently stakes **5.00×** the joint Kelly
optimum. Five genuinely independent ones score 4.90 and 1.28×. Since excess growth scales as
2c − c², an overbet multiple of 2 earns *zero* growth and beyond it growth is negative despite
a real positive edge — so this is the difference between a book that compounds and one that
does not, on identical per-trade edge.

---

## 10. Position sizing

Options payoffs are violently non-normal, so `f* = (μ−r)/σ²` is not usable. Solve the
discrete multi-outcome problem directly on the scenario grid:

```
g(f) = Σ_s p_s·ln(1 + f·x_s)          FOC:  Σ_s p_s·x_s/(1 + f·x_s) = 0
```

Strictly concave (g″ = −E[x²/(1+fx)²] < 0), unique root, Newton with a maintained bracket.
The binding constraint for short premium is not the FOC but `1 + f·x > 0` a.s. — i.e.
f < 1/|min x|. Violating it gives g = −∞: ruin.

### The four shrinkage layers

**1. Uncertainty shrinkage.** Taking expectations of g over estimation error in μ:

```
c_unc = μ²/(μ² + s_μ²) = 1/(1 + 1/(S²T))
```

| true Sharpe | years of data | c* |
|---|---|---|
| 1.0 | 10 | 0.91 |
| 0.5 | 4 | **0.50** |
| 0.5 | 1 | 0.20 |

**"Half Kelly" is not a folk haircut — it is exactly what a Sharpe-0.5 strategy with four
years of evidence deserves.** The system computes it from its own realised record.

**2. Drawdown cap.** For fractional Kelly, P(wealth ever falls to fraction x) = x^(2/c − 1).

| c | P(ever −50%) | P(ever −80%) |
|---|---|---|
| 1.00 (full) | **50.0%** | 80.0% |
| 0.50 | 12.5% | 51.2% |
| 0.25 | 0.78% | 21.0% |

Full Kelly implies a coin-flip chance of at some point halving the account. Inverting turns
a drawdown mandate into a sizing constraint: `c_dd = 2/(1 + ln p / ln x)`. Cap
P(−30%) ≤ 10% ⇒ c ≤ 0.268.

**3. Hard floor at half Kelly**, always. The model is misspecified in ways the math cannot see.

**4. Absolute limits** — max risk per trade, per underlying, per day; % of open interest; % of
ADV. Kelly assumes continuous costless rebalancing with unlimited capacity; an options book
satisfies none of those. **Contracts always round toward zero.**

### The asymmetry that justifies all of it

Excess growth scales as **2c − c²**:

| c | fraction of max growth | vol of log-wealth |
|---|---|---|
| 0.50 | 0.750 | 0.50× |
| 1.00 | 1.000 | 1.00× |
| 1.50 | **0.750** | 1.50× |
| 2.00 | 0.000 | 2.00× |

Half Kelly costs 25% of growth and halves volatility. 1.5× Kelly costs the *same* 25% and
*increases* volatility 50% — strictly dominated. 2× Kelly earns zero. Beyond that, growth is
negative: you go broke almost surely despite having positive edge. Since f* is estimated with
error and the loss is asymmetric in the direction of error, **deliberately bet below your
point estimate.**

---

## 11. The decision policy

17 hard gates, evaluated cheapest-and-most-disqualifying first. **Gates are predicates, not a
weighted score.** A score lets a strong signal on one axis paper over a disqualifying failure
on another, and in options the disqualifying failures — stale data, unfittable surface,
illiquid strike, event inside the window — are precisely the ones that produce the most
confident-looking edge.

```
DATA       freshness · surface fit converged · surface arb-free · fit precision
LIQUIDITY  every leg: two-sided, spread, open interest
MODEL      conformal not in kill state · Brier reliability ≤ 0.02
EDGE       edge z · |VRP| ≥ 2 vol pts · annualised EV on risk · credit/width
           · EV ≥ 3× round-trip cost · PoP · PoP beats risk-neutral by ≥3 pts
RISK       portfolio delta · portfolio vega · concentration · DTE window
           · earnings blackout · defined-risk
SIZE       Kelly chain returns ≥ 1 contract
```

**Direction follows the sign of the VRP**, not a view on the underlying. VRP > 0 ⇒ SELL
premium; VRP < 0 ⇒ BUY. **HOLD is the default and by far the most common output.**

Three thresholds worth explaining because their form matters more than their value:

- **EV is gated annualised.** A flat "EV ≥ 5% of risk" silently favours long-dated trades that
  tie up capital for months to earn what a three-week trade earns once. Growth rate is what
  Kelly maximises, so growth rate is what the gate measures.
- **Credit/width is measured at MID**, not marketable. It is a property of the strikes chosen;
  loading it with slippage double-counts costs the EV gate already charges.
- **EV ≥ 3× round-trip cost** is the gate that actually bites on a small account. At
  $0.65/contract a 1-lot vertical costs $2.60 round trip. An $8 edge is not an edge.

### `rank_and_select` — screen first, then rank

A selector that ranks on raw edge will keep nominating the best-edge structure, the policy
will keep rejecting it on some other axis, and the system will report "no trade" every day
while sitting on candidates it would have accepted. **The selector must apply the same gates
the policy applies, then rank the survivors by the same objective the sizer maximises
(expected log growth).** This is implemented and it is not a subtlety you discover cheaply.

---

## 12. Execution at Schwab

### Approval and account requirements

Schwab uses Levels 0–3. Everything this system produces is **Level 2 (Spread Trading)**:
verticals, calendars, diagonals, condors, butterflies. Level 2 **requires a margin account** —
Schwab's application form: *"Securities regulations require that options spreads occur in a
margin account."* In an IRA this is *limited margin*, applied for separately, and naked short
calls are prohibited outright. Approval takes 3 business days by email.

⚠️ Schwab's form lists "condors, butterflies" at Level 2 but never the words "iron condor."
Structurally it is two credit verticals and the API exposes `IRON_CONDOR` as a
`complexOrderStrategyType`, but **confirm with the options desk before relying on it.**

### Cost model (Pricing Guide, effective April 2026)

| Item | Amount |
|---|---|
| Online commission | **$0 base + $0.65 per contract**, counted across ALL legs |
| 1-lot vertical | 2 contracts = **$1.30** |
| 1-lot iron condor | 4 contracts = **$2.60** |
| Buy-to-close online at ≤ $0.05 | **per-contract fee waived** |
| Exercise / assignment | **$0 commission** |
| SEC Section 31 (sales) | $20.60 per $1M — effective 2026-04-04, was **$0.00** for the preceding 11 months |
| FINRA TAF (sales) | $0.00279 per contract |
| ORF, proprietary index fee | **not published** — bundled into a discretionary "Industry Fee" |

Two consequences the system encodes:

The closing-fee waiver means **there is no fee argument for carrying assignment risk into
expiration.** Buying back a near-worthless short leg online is free.

Schwab does not itemise regulatory fees; it bundles them into an "Industry Fee" it sets *"in
its sole and reasonable discretion"* and which *"may differ from or exceed the actual fees
properly paid by Schwab."* The cost model treats these as an estimate — reconcile against
real confirms after ~10 fills and recalibrate.

### The ticket

`build_ticket()` emits: order type (NET_DEBIT / NET_CREDIT), limit per spread, every leg with
its **21-character OCC symbol** (`SPY   260918P00780000` — 6-char underlying *space-padded*,
YYMMDD, C/P, 8-digit strike), the cost estimate, a price ladder, numbered click paths for both
schwab.com and thinkorswim, an exit plan, warnings, and a valid Trader API payload.

**Price ladder, not a single limit.** Start at mid, walk toward natural in $0.01 steps.
Schwab's **Walk Limit** order type automates exactly this (place → wait → cancel → replace one
increment closer), and Schwab documents it as *"particularly useful in multi-leg options
strategies"* — each leg carries its own bid/ask and the composite natural is punitively wide.
Sending at natural on a 4-leg structure typically gives up more than the entire modelled edge.

**API payload correctness.** A multi-leg spread is **one** order strategy:
`orderStrategyType: "SINGLE"` with the multi-leg nature carried by `orderLegCollection` +
`complexOrderStrategyType`. A widely-circulated AI-generated example uses
`orderStrategyType: "MULTI_LEG"` with instructions `"BUY"`/`"SELL"` and dotted thinkorswim
symbols — **all three are invalid** against Schwab's documented enums. A test asserts the
correct enums.

### Assignment, pin risk, and the thing that will actually hurt you

**Both legs auto-exercise/assign at $0.01 ITM** (OCC exercise-by-exception; Schwab states the
3pm CT snapshot as reference). A vertical whose short leg finishes $0.01 ITM and long leg OTM
becomes a **stock position overnight**, settled before you can react.

The industry deadline for contrary-exercise instructions is **5:30pm ET**, and firms may set
earlier internal cutoffs. **Schwab does not publish its own.** Schwab's account agreement also
grants it discretion to liquidate *"without prior demand or notice."*

Therefore the exit plan is built around **closing during regular hours on expiration day**,
never around submitting a DNE. And the time stop is **21 DTE regardless of P&L** — gamma and
assignment risk both accelerate inside three weeks, and neither is part of the edge that
justified the trade.

### Open questions — call the desk (888-245-6864)

`execution/schwab.py::OPEN_QUESTIONS` carries these in code:

1. **[HIGHEST]** Schwab's own internal DNE/contrary-exercise cutoff. Unpublished. Hard-code it with margin once you have it.
2. Iron condor's placement at Level 2 (implied, never stated verbatim).
3. Per-contract ORF and proprietary index-option fee pass-through.
4. Whether a multi-leg spread can be an OCO leg in the **GUI** (confirmed supported via API).
5. Schwab's numeric trigger for liquidating short options in an undercapitalised account.

### If you later automate: Schwab Trader API

Individual developer registration is available. Three-legged OAuth. **Access token: 30
minutes. Refresh token: 7 days, hard stop, not extendable.** There is no headless renewal and
no service-account model. **Any unattended automation requires a human at a browser at least
once every 7 days** — build a calendar-driven re-auth step and alert on `invalid_client`, or
the strategy silently goes dark mid-week. Order throttle is configurable 0–120 requests/minute
per account; GET order requests are unthrottled.

---

## 13. Validation — how you find out whether any of this works

**Nothing in this repo is evidence the strategy is profitable.** The tests prove the math is
correct, not that the edge exists. Here is the order in which to find out.

### 13.1 Unit level (done — 173 tests)

Put-call parity exact to 1e-12 · every Greek vs finite differences · IV round-trip 3.3e-12
over 1324 tradeable quotes · IV *rejects* rather than hallucinating on degenerate input · LR
monotone convergence and odd-n enforcement · BS2002 vs Haug's published table · American ≥
European · de-Americanisation lowers IV · forward recovery · SVI parameter recovery · density
non-negative and integrating to 1 · butterfly detector fires where domain checks pass ·
crossedness detects calendar arbitrage · SSVI parameter recovery · Kelly vs closed form ·
growth-curve symmetry (2c−c²) · shrinkage reproduces half-Kelly · drawdown cap and
risk-of-ruin are inverses · Kalman tracks a step change and stays PSD · conformal restores
coverage after a 6× regime break · kill switch fires · calibration improves reliability ·
Brier decomposition identity · drift detectors fire on shift and not on noise · every policy
gate blocks independently · OCC symbol format · Schwab enum validity · cost model per-leg
counting.

Added with phases 9–14:

**Surface (16)** SSVI↔raw-SVI map exact to 1e-14 · Lee bound and GJ Thm 4.2 coincide under it ·
joint fit recovers ρ and η · every slice butterfly-free and uncrossed · refinement is actually
used, not silently falling back · a crossing that an *independent* fit produces is either
eliminated or the surface is marked unpublishable and the gate refuses it · non-monotone θ
repaired and reported · total-variance interpolation cannot create a calendar arbitrage ·
publication gate falls back to last-known-good, raises, and stops serving a stale surface after
a limit · arbitrage margin is trendable.

**Pipeline (36)** end-to-end to a sized decision with a valid ticket · sentinel IVs detected and
never consumed · direction follows the VRP sign in both regimes · every decision journaled with
enough to replay it · each stage refuses by NAME (dead feed, unquotable chain, missing earnings
fails *closed*, earnings inside the holding window, too little history) · expiry preference and
the holding-window refusal · a config that could never trade is rejected at construction ·
enumerator emits condors and calendars with every leg quotable and defined-risk · delta
targeting uses the surface, not a flat vol · replay truncates bars at the cursor, serves one
coherent path, keeps expiries fixed across sessions, and raises rather than inventing data ·
P&L is `exit_net − entry_net` for debits *and* credits, checked against `max_loss` · fill
penalty is always worse for the trader · drawdown recovery · deflating for multiple testing
lowers the verdict.

**Journal / feedback (19)** NaN survives the JSON round trip as NaN · future fields and a
corrupt trailing line do not break the read · HOLDs counted · closed trades ordered by when the
outcome became *knowable* · slippage measured against marketable and not the mid-based limit ·
a two-trade record cannot move sizing · a long losing record does · a winning record never
raises sizing above the prior · calibration gate held open below its minimum · learned state is
a pure function of the journal prefix.

**Factors / forecast (32)** identical positions reveal a 5× overbet and 1.00 effective bets,
independent ones 1.28× and 4.90 · unidentified beta defaults to 1.0 · GJR recovers known
parameters with leverage dominating ARCH · persistence uses ξ/2 · semivariances sum *exactly*
to the pooled proxy and are strictly positive · SHAR finds the asymmetry (t = 2.96) and nests
HAR · QLIKE minimised at a perfect forecast and punishes under-forecasting harder ·
Mincer-Zarnowitz flags overreaction · Diebold-Mariano corrects for overlapping horizons.

### 13.2 Surface level

Refit the last 250 trading days of chains from the recorded journal. Track daily: fit RMSE in
vol points, fraction of quotes inside their spread, `min g(k)` per slice, max crossedness,
count of IV rejections by status. **A surface whose arbitrage margin is trending toward zero
is a leading indicator of a data problem, not of market stress.** Trend it.

### 13.3 Forecast level

Walk-forward only, expanding window, refit monthly. Compare HAR-log / HARQ / GJR-GARCH /
naive-RV / implied-as-forecast on: QLIKE and MSE against realised, Mincer-Zarnowitz regression
(slope should be 1), Diebold-Mariano vs the naive benchmark, and — most importantly — **ACI
empirical coverage** of the stated intervals.

### 13.4 Strategy level — the part everyone gets wrong

Replay the **recorded journal**, not reconstructed history. Non-negotiable rules:

- Fill at the **marketable** price or worse, never mid. Model partial fills on multi-leg.
- Charge the full Schwab cost model on entry *and* exit.
- No look-ahead: the surface, forecast and thresholds at decision time must be the ones
  available at decision time. The journal makes this checkable rather than aspirational.
- Report: annualised growth of log wealth (the actual objective), max drawdown and time to
  recover, Sharpe *and* Sortino, hit rate vs. average win/loss, and **the distribution of the
  gate that blocked**, so you can see what the system is actually constrained by.
- Deflated Sharpe / White's reality check for the multiple-testing you did while tuning.

### 13.5 Paper then live

Minimum **60 trading days** of decisions logged with no orders placed, comparing predicted to
realised on every axis. Then 1-lot live for another 60. Then let the shrinkage layer size up
on its own — `c_unc` uses the realised out-of-sample Sharpe, so **the system scales itself as
evidence accumulates and de-scales when it stops working**, with no threshold changes.

---

## 14. Build order

| Phase | Deliverable | State |
|---|---|---|
| **0** | Data seam, quality screen, recording journal | **done** |
| **1** | Pricing core: BSM+Greeks, IV inversion, LR/BS2002, forward, de-Am | **done** |
| **2** | Surface: raw SVI + arbitrage diagnostics + quasi-explicit fit; SSVI backbone | **done** |
| **3** | Forecast: YZ/Parkinson/GK/RS/bipower/RQ, HAR/HARQ log form | **done** |
| **4** | Edge: Q→P scenarios, VRP, PoP/EV/RoC/CVaR | **done** |
| **5** | Sizing: scenario Kelly + shrinkage chain | **done** |
| **6** | Policy: 17-gate stack, `rank_and_select` | **done** |
| **7** | Execution: ticket, click paths, cost model, API payload | **done** |
| **8** | Learning primitives: Kalman, ACI, calibration, drift | **done** |
| **9** | Live wiring: `app/pipeline.py` runner, `data/yfinance_direct.py`, `app/config.py` | **done** |
| **10** | Outcome journal + feedback wiring (§8.6): `data/journal.py`, `learning/feedback.py` | **done** |
| **11** | Constrained joint surface fit (SSVI envelope + slice refinement): `surface/joint.py` | **done** |
| **12** | Backtester over the journal with the §13.4 rules: `backtest/`, `data/replay.py` | **done** |
| **13** | Factor-level correlation model for portfolio Kelly: `risk/factors.py` | **done** |
| **14** | Iron condors, butterflies, calendars, diagonals: `domain/candidates.py` | **done** |
| — | SHAR + GJR-GARCH cross-check (§6.2), forecast evaluation suite (§13.3) | **done** |
| — | CLI: `decide` / `collect` / `backtest` / `journal` / `learned` | **done** |
| **15** | Schwab Trader API automation | **deliberately not built** — see §12 |

**Phases 9, 10 and 12 were built before any modelling sophistication, and that ordering paid
for itself immediately.** The journal replay found six defects in three runs — a degenerate
DTE/time-stop interaction, a slippage feedback path that shut the system down permanently, a
Sharpe annualisation with an absorbing state at zero, an earnings gate that missed the case it
existed for, a debit profit target that was a 50% loss, and an unsafe surface fallback. None of
them is visible from unit tests, because each requires the loop to be closed before it can
appear. A system with a mediocre forecast and an honest feedback loop beats a sophisticated one that cannot tell you whether it
is working.

---

## 15. Worked example

`examples/end_to_end.py` runs the full pipeline offline in two regimes.

**CALM** (ATM implied 13.6%, HAR forecast 11.8%, VRP +1.8 vol points):

```
[1] FEED QUALITY OK — 104 quotes, 95 two-sided, 9 zero-bid, 9 sentinel IVs
    - 9 vendor IV values are sentinels (0.500005) -- ignored by design
[2] F = 780.6119 (true 780.6109, err +0.0010), DF = 0.996326, R2 = 0.99999999
[4] SVI fit OK — rmse 0.018 vol pts, 100% inside spread, min g = 0.316
[5] VRP +1.82 vol pts
[6] all 15 candidates rejected
[8] DECISION: HOLD
```

**STRESSED** (ATM implied 28.6%, forecast 13.5%, VRP +15.1 vol points):

```
[6] CANDIDATE FRONTIER (screened by gates, ranked by expected log growth)
       structure                          PoP    mkt   EV/spr   EVann   c/w  risk$  n   growth
    OK 730/745 put credit vert (30%d)   87.3%  79.9%   62.64   56.4%   16%   1266  1  0.00150
    OK 705/720 put credit vert (22%d)   92.5%  85.9%   59.12   50.3%   11%   1340  1  0.00143
    OK 745/760 put credit vert (38%d)   81.0%  74.2%   49.37   46.9%   20%   1200  1  0.00114
    -- 680/695 put credit vert (16%d)   95.3%  89.9%   49.67   40.9%    8%   1386  1  0.00121
       └ rejected: edge.credit_to_width
    -- 690/695 put credit vert (16%d)   95.2%  89.8%   -3.46   -8.2%    4%    480  0  0.00000
       └ rejected: edge.z, edge.ev_annualised, edge.credit_to_width

[7] Kelly f*=0.2842 -> applied 0.0317 (c=0.268 = min(unc 0.377, dd 0.268, 0.50));
    1 contract, $1,266 at risk (3.17% of bankroll); binding: per-trade risk cap
[8] DECISION: SELL x1 — all 17 gates PASS
[9] ORDER TICKET: NET_CREDIT $2.84/spread (mid $2.84, natural $2.35)
    ladder $2.84 -> $2.72 -> $2.60 -> $2.47 -> $2.35
    BUY_TO_OPEN  1  SPY 18 Sep 26 730 Put  [SPY   260918P00730000]
    SELL_TO_OPEN 1  SPY 18 Sep 26 745 Put  [SPY   260918P00745000]
    open cost $1.31 · exit at 50% of max profit · time stop 21 DTE
```

Note the 16-delta structures: highest PoP (95.3%), highest apparent growth — and rejected on
credit/width. That is the gate stack doing its job. Selling a 15-wide for 8% of its width
needs a win rate the model cannot honestly claim.

Note also what the CALM run demonstrates: 9 zero-bid strikes with `0.500005` sentinel IVs
detected and discarded, the exact yfinance pathology, reproduced end-to-end.

---

## 16. Honest limitations

1. **Latency.** ~15-minute chains. Fine for 21–45 DTE variance premium; disqualifying for
   anything faster. Do not let scope creep past that without upgrading the feed.
2. **No options tape.** No trade prints, no NBBO timestamps, no exchange codes. Order-flow
   signals, true microstructure filters and realistic fill modelling are all out of reach
   until the options plane is entitled.
3. **The edge is one factor.** Short index vol is one bet however many names you spread it
   across. Diversification here is mostly illusory.
4. **VRP goes negative exactly when it hurts.** Aug 2007, Oct 2008, Feb 2018, Mar 2020. The
   drawdown cap and defined-risk constraint are what stand between the model and those days —
   not the forecast.
5. **Every threshold is a free parameter.** Fifteen-plus of them. That is a lot of implicit
   multiple testing. Derive them from your own out-of-sample record and report the deflated
   statistic, not the nominal one.
6. **Backtests over a journal you have not yet collected do not exist.** Phase 10 first.
7. **PoP is computed to expiry; the exit plan closes at 21 DTE.** These are not the same
   probability, and the journal replay measures the gap: stated PoP 78.1% against a realised
   win rate of 55.0% over 20 closed trades. Nothing is mis-implemented — `edge/score.py`
   correctly reports P(P&L > 0 **at expiry**) under the reweighted density, while §12's time
   stop deliberately flattens at 21 DTE and forgoes the last of the theta. The consequence is
   that the number feeding `min_pop` and the probability calibrator describes a trade the
   system does not actually hold to the end.

   The §8.4 calibration gate catches this without being told about it — REL rose and
   `model.calibration` blocked 142 of 1057 candidate evaluations in the replay — which is the
   layer working as designed. But catching a systematic bias is not the same as fixing it. The
   correct fix is to score the structure at the **planned exit horizon** rather than at expiry:
   evaluate the payoff at 21 DTE using the surface, not the terminal intrinsic. That is a real
   change to `score_structure` and it is not made here, because it would silently move every
   threshold in §11 at once and those thresholds must be re-derived, not inherited. Until then,
   treat stated PoP as an upper bound and let the calibrator do its job.

8. **Small-account capital constraint is real and the model will tell you about it.** At $25k
   with a 2% per-trade cap, *no* SPY vertical fits — $780 notional makes even a 5-wide risk
   ~$430. The honest responses are a lower-notional underlying, narrower spreads, or a
   deliberately larger cap. The system reports this as a diagnosis rather than sizing to
   fractional contracts.

---

*Not investment advice. Options carry substantial risk of loss, including total loss of the
premium or the full defined risk of a spread. Nothing here has been validated against live
markets; the tests establish mathematical correctness only.*
