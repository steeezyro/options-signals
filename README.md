# optionsmarkets

A defined-risk options relative-value engine: arbitrage-free volatility surface, realised-vol
forecasting, online recalibration, fractional-Kelly sizing, and Schwab execution tickets.

**Read [`BLUEPRINT.md`](BLUEPRINT.md) first** — it is the design document and the reasoning
behind every choice here.

## Quickstart

```bash
pip install -e ".[dev]"
pytest -q                                   # 173 tests
python examples/end_to_end.py               # single-slice walkthrough, no network needed

# One decision against a known-truth synthetic market, offline:
python -m optionsmarkets --provider synthetic --etf decide

# One decision against live yfinance chains (needs the 'live' extra):
python -m optionsmarkets --symbol SPY --bankroll 40000 --etf decide
```

Optional extras:

```bash
pip install -e ".[live]"    # yfinance: in-process live provider, no MCP needed
pip install -e ".[fast]"    # py_lets_be_rational + numba: ~100x faster IV inversion
pip install -e ".[learn]"   # scikit-learn: isotonic probability calibration
```

## Operating it

```bash
# Record a snapshot. Run on a schedule -- THIS JOURNAL IS THE BACKTEST DATASET.
python -m optionsmarkets --symbol SPY --snapshots journal/snapshots collect

# Replay it under the section 13.4 rules: marketable-or-worse fills, partial
# fills on multi-leg, full costs both ways, no look-ahead, deflated Sharpe.
python -m optionsmarkets --snapshots journal/snapshots --trials 20 --etf backtest

# What actually blocked, and what the record is doing to position sizing.
python -m optionsmarkets journal
python -m optionsmarkets learned
```

No command places an order, and there is no flag that makes one. Schwab's Trader API refresh
token expires after 7 days with no headless renewal (§12), so an unattended execution path goes
dark mid-week without telling you. Execution is manual, from a printed ticket.

## What it does

Given a ticker, a bankroll and current positions, it produces one of:

- **SELL** *n* × a specific defined-risk credit structure, with a complete Schwab order ticket
- **BUY** *n* × a specific defined-risk debit structure, same
- **HOLD**, with the exact list of gates that blocked and their values

It does not predict direction. It measures the gap between implied and forecast variance,
converts it into a scenario P&L distribution, and sizes by fractional Kelly under explicit
uncertainty and drawdown constraints.

## Pipeline

```
feed → quality screen → forward from put-call parity → de-Americanise
     → in-house IV inversion → SVI/SSVI surface + arbitrage gates
     → RV forecast (HAR/HARQ) → variance risk premium
     → Q→P scenario density → structure enumeration → edge scoring
     → fractional Kelly (uncertainty × drawdown × ½-Kelly × hard caps)
     → 17-gate stack → BUY/SELL/HOLD → Schwab order ticket
     → outcome journal → Kalman / conformal / calibration update
```

Each stage refuses to run on output the previous stage flagged bad. There is no best-effort
path.

## Module map

| Module | Contents |
|---|---|
| `pricing/black` | BSM price + 14 Greeks; one unit-conversion boundary |
| `pricing/implied` | erfcx-normalised Black, Householder(3), bracket + Brent backstop |
| `pricing/american` | Leisen-Reimer lattice, Bjerksund-Stensland 2002, de-Americanisation |
| `pricing/forward` | F and DF from put-call parity regression |
| `surface/svi` | raw SVI, `g(k)` butterfly test, SVI-JW, quasi-explicit calibration |
| `surface/ssvi` | SSVI backbone, GJ Thm 4.1/4.2 conditions enforced by search bounds |
| `forecast/realized` | Yang-Zhang, Parkinson, Garman-Klass, Rogers-Satchell, bipower, RQ |
| `forecast/har` | HAR / HARQ, log form, Newey-West, Jensen correction |
| `edge/score` | Q→P reweighting, scenarios, PoP / EV / RoC / CVaR / edge-z |
| `sizing/kelly` | scenario multi-outcome Kelly + the four shrinkage layers |
| `learning/online` | Kalman TVP, adaptive conformal, Platt/isotonic, ADWIN, Page-Hinkley |
| `learning/feedback` | journal → learners → `LearnedState` → sizing and gates |
| `policy/decide` | gate stack, `Thresholds`, `rank_and_select` |
| `execution/schwab` | ticket, cost model, click paths, `OPEN_QUESTIONS` |
| `domain/structures` | contracts, legs, verticals / condors / butterflies / calendars |
| `domain/candidates` | delta-targeted enumeration of every structure family |
| `surface/joint` | SSVI envelope + constrained slice refinement, publication gate |
| `forecast/garch` | GJR-GARCH(1,1) asymmetry cross-check |
| `forecast/evaluate` | QLIKE, Mincer-Zarnowitz, Diebold-Mariano, interval coverage |
| `risk/factors` | betas, effective number of bets, portfolio Kelly on the factor |
| `app/pipeline` | **the runner** — a provider in, a decision and a ticket out |
| `backtest/` | journal replay under the §13.4 rules, deflated Sharpe |
| `data/` | provider Protocol, MCP + direct + synthetic + replay adapters, journal |

## Live data

Wire the MCP adapters by injecting a `call` function (see `data/mcp_adapters.py`):

```python
from optionsmarkets.data.mcp_adapters import (
    CompositeProvider, MassiveUnderlyingProvider, RecordingProvider, YFinanceChainProvider
)

provider = RecordingProvider(
    CompositeProvider(
        chains=YFinanceChainProvider(call=my_mcp_invoke),
        underlying=MassiveUnderlyingProvider(call=my_mcp_invoke),
    ),
    path="journal/",
)
```

`RecordingProvider` journals every response — **that journal is the backtest dataset.**

## Three things to know before trusting output

1. **Vendor IV is discarded.** yfinance emits `0.500005` on zero-bid strikes. Every volatility
   is inverted in-house and unquotable strikes are rejected, not imputed.
2. **Every price is marketable, never mid.** Buys at the ask, sells at the bid. Scoring at mid
   manufactures edge the fill takes straight back.
3. **HOLD is the default.** If the system trades most days, a threshold is wrong.

## Three findings from actually closing the loop

Building phases 9–14 surfaced defects that unit tests cannot reach, because each one needs the
feedback loop running before it can appear. All are fixed, documented at the point of
occurrence, and covered by regression tests:

1. **A degenerate DTE/time-stop interaction.** `target_dte=(21,45)` selecting the *nearest*
   expiry against a 21-DTE time stop entered and stopped out within days — earning none of the
   variance premium and carrying full gamma. Three trades, all losers, average −25% of capital
   at risk, while the stated PoP was 75% (because PoP is computed to expiry and the trade never
   got there). `expiry_preference` now defaults to the far end of the band.
2. **A feedback path that shut the system down permanently.** Slippage was measured against the
   ticket's *mid-based* limit and fed into every future expected value, re-charging the entire
   mid-to-marketable spread the scorer had already paid — $27.50/spread from fills that were
   within $2 of the assumption. Now measured against the marketable reference the scorer used.
3. **An absorbing state at zero size.** A realised Sharpe annualised from a 0.001-year span read
   −2367, driving `c_unc` to exactly 0 and every position to 0 contracts — permanently, since a
   system that cannot trade can never gather the evidence to size again. The record now cannot
   move sizing below a 20-trade / 0.25-year minimum.

Three more are listed in BLUEPRINT.md's status table.

## Status

Phases 0–14 implemented, 173 tests. Phase 15 (Schwab Trader API automation) is deliberately not
built — the 7-day refresh-token limit means unattended execution goes dark mid-week.

**None of this is evidence the strategy is profitable.** The tests establish that the math is
correct and that the machinery is honest about what it does not know. A backtest over the
synthetic fixture establishes that the replay works, and nothing more.

---

*Not investment advice. The tests establish mathematical correctness only — nothing here has
been validated against live markets.*
