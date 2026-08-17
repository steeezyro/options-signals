# optionsmarkets — working notes

Defined-risk options relative-value engine. **[`BLUEPRINT.md`](BLUEPRINT.md) is the
specification and the reasoning behind every choice — read it before changing anything
non-trivial.** This file is only the operational layer on top of it.

## State

Phases 0–14 built, **173 tests passing**, lint clean. Phase 15 (Schwab Trader API automation)
is deliberately not built — the 7-day refresh-token limit means unattended execution goes dark
mid-week without telling you.

Nothing here has been validated against live markets. The tests establish that the math is
correct and that the machinery is honest about what it does not know.

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
.venv/bin/optionsmarkets --symbol SPY --snapshots journal/snapshots collect
.venv/bin/optionsmarkets --snapshots journal/snapshots --trials 20 --etf backtest
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
- **Each stage refuses with a NAMED gate rather than degrading.** A data problem returns a
  journaled HOLD, not an exception and not a best-effort answer.
- **HOLD decisions are journaled too.** The distribution of which gate blocked is the most
  valuable half of the dataset.
- **The learned state is a pure function of the journal prefix.** `FeedbackLoop.ingest(...,
  up_to=...)` must stay reproducible — that is the entire no-look-ahead argument.
- **Nothing in the feedback loop may raise position size.** Every learned quantity enters as a
  cap alongside the existing caps.

## Known open item

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
