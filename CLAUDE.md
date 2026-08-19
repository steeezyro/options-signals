# optionsmarkets — working notes

Defined-risk options relative-value engine. **[`BLUEPRINT.md`](BLUEPRINT.md) is the
specification and the reasoning behind every choice — read it before changing anything
non-trivial.** This file is only the operational layer on top of it.

## State

Phases 0–14 built, **181 tests passing**, lint clean. Phase 15 (Schwab Trader API automation)
is deliberately not built — the 7-day refresh-token limit means unattended execution goes dark
mid-week without telling you.

First run against a live open market on 2026-08-18. It failed immediately — every underlying
HELD at `data.iv_inversion` on a put-call parity defect — and that is now fixed and verified
(see the forward convention below). Live runs reach the surface, the forecast and the edge
gates. **The forward is the only stage confirmed correct on live quotes** (to 3–17 bp against
a known-truth check); everything downstream of it is still only confirmed against fixtures.

## Environment

Already set up. Do not recreate it.

```bash
cd ~/Desktop/projects/optionsmarkets
.venv/bin/python -m pytest -q          # 173 tests, ~4 min
.venv/bin/optionsmarkets --help
```

- venv: `.venv`, installed with `pip install -e ".[dev,live]"`.
- `timeout(1)` does **not** exist on this macOS. Use `--timeout`/`--retries` flags on the tool
  itself, or a background run.
- Long operations: full test suite ~4 min; a 330-session backtest ~5 min; one pipeline
  decision ~1 s.

## Running it

```bash
# Offline, known-truth market — works any time, exercises the whole path
.venv/bin/optionsmarkets --provider synthetic --etf decide

# Live — meaningful only during 09:30–16:00 ET
.venv/bin/optionsmarkets --symbol SPY --bankroll 40000 --etf decide

# Collect / replay / inspect
.venv/bin/optionsmarkets --symbol SPY --snapshots journal/snapshots/SPY collect
.venv/bin/optionsmarkets --symbol SPY --snapshots journal/snapshots/SPY --trials 20 --etf backtest
.venv/bin/optionsmarkets journal
.venv/bin/optionsmarkets learned
```

`--etf` is required for SPY: without it a missing earnings date fails **closed** (by design)
and blocks every decision.

**Outside market hours the live provider returns zero two-sided quotes and the run HOLDs on
`data.quality`. That is correct behaviour, not a bug.** Use `--provider synthetic` to exercise
the full path off-hours.

## Where things are

`src/optionsmarkets/`

| Path | Role |
|---|---|
| `app/pipeline.py` | **the runner** — provider in, decision + ticket out. Start here. |
| `app/config.py` | `RunConfig`: every policy dial, each with its rationale in a comment |
| `data/provider.py` | `MarketDataProvider` Protocol + quality screen — the seam |
| `data/synthetic.py` | known-truth provider **with the feed's real defects**; test oracle |
| `data/replay.py` | replays a recording; no-look-ahead is structural, not disciplinary |
| `data/journal.py` | append-only JSONL outcome journal |
| `learning/feedback.py` | journal → learners → `LearnedState` → sizing and gates |
| `surface/joint.py` | SSVI envelope + constrained slice refinement + publication gate |
| `backtest/engine.py` | journal replay under the §13.4 rules |
| `domain/candidates.py` | delta-targeted enumeration of every structure family |

Tests mirror this: `test_pipeline_backtest.py`, `test_surface_joint.py`,
`test_journal_feedback.py`, `test_factors_forecast.py`, plus the three original files.

## Conventions that will bite you

These are load-bearing. Each was a real defect at some point; see BLUEPRINT.md's status table.

- **P&L is `exit_net − entry_net`, uniformly for debits and credits.** No branch. A conditional
  that flips the subtraction for credits prints every short-premium result backwards.
- **Prices are marketable, never mid.** `net_price("marketable")` buys at ask, sells at bid.
  Scoring at mid manufactures edge the fill takes straight back.
- **Slippage fed back into the model is measured against the marketable reference, not the
  ticket limit.** The limit starts at mid; using it re-charges the whole mid-to-marketable
  spread and drives every future EV negative.
- **Vendor IV is never consumed.** yfinance emits `0.500005` on zero-bid strikes. Every vol is
  inverted in-house; unquotable strikes are rejected, never imputed.
- **The forward's DF comes from the risk-free curve; only F is fitted.** These are American
  quotes. Regressing DF out of parity estimates it off a ±10% strike lever arm, which
  amplifies ~10 bp of per-strike quote error into 100–2500 bp of DF error and additionally
  absorbs the early-exercise premium — measured live, that returned `DF > 1` (a negative
  implied rate) on *every* underlying and blocked every decision. Do not restore the free fit,
  and do not widen the `DF <= 1.0001` bound to make runs pass: a tilted forward is
  indistinguishable from skew. `pricing/forward.py`'s docstring carries the measurements.
- **Expiry selection filters before it prefers.** `furthest` applies to the expiries that are
  *tradeable* — quoted, and clear of earnings — not to the raw band. Both filters are selection
  rules and neither weakens a gate: `risk.event_in_window` and `data.iv_inversion` still fire
  when no expiry qualifies. Skipping this made MU unreachable for the weeks around its own
  earnings and made MA refuse on a listed-but-dead expiry while three live ones sat beside it.
- **A replay directory holds exactly one underlying, and `ReplayProvider` enforces it.**
  `RecordingProvider` names files `{stamp}_{method}.json` with no symbol in them, so one
  directory collecting several tickers interleaves them. Every symbol-taking replay method used
  to ignore its `symbol` argument and serve whatever was newest — measured 2026-08-18 on real
  recordings, `option_chain("SPY")` returned IWM at spot 300.67 instead of SPY at 768.09, a 2.5x
  error that never raised. The symbol now comes from each file's `meta.args` and a request for
  an unrecorded ticker refuses. Still always collect with `--snapshots journal/snapshots/SYMBOL`.
- **Each stage refuses with a NAMED gate rather than degrading.** A data problem returns a
  journaled HOLD, not an exception and not a best-effort answer.
- **HOLD decisions are journaled too.** The distribution of which gate blocked is the most
  valuable half of the dataset.
- **The learned state is a pure function of the journal prefix.** `FeedbackLoop.ingest(...,
  up_to=...)` must stay reproducible — that is the entire no-look-ahead argument.
- **Nothing in the feedback loop may raise position size.** Every learned quantity enters as a
  cap alongside the existing caps.

## Known open items

**The surface fitter is the current live blocker.** SSVI fits live SPY at rmse 1.16–1.45 vol
points with 0–6% of strikes inside the bid–ask, against 0.01–0.04 and 100% on the synthetic
fixture; the run HOLDs on `data.surface_arbfree`. Live spreads are ~0.1–0.2 vol points wide, so
the in-spread test is far harsher on real quotes than on the fixture. This is a fitter
limitation the parity defect was masking, not a forward error. Worth checking first: MU fits
across its own earnings date (decision expiry 09-18, neighbour 09-25, earnings 09-23), so the
term structure SSVI must fit smoothly contains an event jump. Unverified.

**PoP is computed to expiry; the exit plan closes at 21 DTE.** Measured in replay: stated PoP
78.1% vs realised win rate 55.0%. The §8.4 calibration gate detects it unaided, but the real
fix is scoring the payoff at the planned exit horizon off the surface rather than at terminal
intrinsic. Deferred because it moves every §11 threshold at once and those must be re-derived,
not inherited. Written up in BLUEPRINT.md §16.7.

## House style

Match the existing code: dense explanatory docstrings that say *why*, not *what*; the reasoning
for a non-obvious choice goes in the comment next to it. When a fix comes from an observed
failure, record the observation and its measured numbers alongside the fix — several comments
in this repo do exactly that, and they are the reason the traps stay fixed.
