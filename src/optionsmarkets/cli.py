"""Command line: ``python -m optionsmarkets <command>``.

Five commands, matching the operational loop the blueprint describes:

    decide     one decision now, against a live or synthetic provider
    collect    record a snapshot to the journal (run this on a schedule)
    backtest   replay the recorded journal under the section 13.4 rules
    journal    summarise the outcome journal and what actually blocked
    learned    show the current learned state and what it is doing to sizing

``decide`` never places an order.  There is no command that does, and that is
deliberate: BLUEPRINT.md section 12 documents that Schwab's Trader API refresh
token expires after 7 days with no headless renewal, so any unattended execution
path silently goes dark mid-week.  Execution stays manual, from a printed ticket,
until that is solved with a human-in-the-loop re-auth step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _provider(name: str, journal_dir: Path | None = None):
    if name == "yfinance":
        from .data.yfinance_direct import YFinanceDirectProvider
        p = YFinanceDirectProvider()
    elif name == "synthetic":
        from .data.synthetic import SyntheticProvider
        p = SyntheticProvider()
    elif name == "replay":
        from .data.replay import ReplayProvider
        if journal_dir is None:
            raise SystemExit("--snapshots is required with --provider replay")
        p = ReplayProvider(journal_dir)
    else:
        raise SystemExit(f"unknown provider {name!r}")
    return p


def _config(args):
    from .app.config import RunConfig
    cfg = RunConfig(symbol=args.symbol.upper(), bankroll=args.bankroll,
                    journal_path=Path(args.journal))
    if args.max_risk is not None:
        cfg.max_risk_fraction_per_trade = float(args.max_risk)
    if args.etf:
        # An index ETF genuinely has no earnings date, so failing closed on a
        # missing one would block every decision forever.
        cfg.earnings_unknown_action = "ignore"
    return cfg


def cmd_decide(args) -> int:
    from .app.pipeline import Pipeline
    cfg = _config(args)
    prov = _provider(args.provider, Path(args.snapshots) if args.snapshots else None)
    res = Pipeline(prov, cfg).run()
    print(res.report())
    if res.ticket is not None:
        print()
        print(res.ticket.render())
    else:
        print("\nNo ticket: the decision was HOLD. The gates above say which stage "
              "refused and with what value.")
    return 0


def cmd_collect(args) -> int:
    """Record one snapshot.  Run on a schedule; the journal IS the backtest dataset."""
    from .data.mcp_adapters import RecordingProvider
    cfg = _config(args)
    prov = _provider(args.provider, None)
    out = Path(args.snapshots or cfg.recording_path)
    rec = RecordingProvider(prov, out)
    sym = cfg.symbol
    n_ok, n_fail = 0, 0
    for call, label in ((lambda: rec.option_chain(sym), "option_chain"),
                        (lambda: rec.risk_free_curve(), "risk_free_curve"),
                        (lambda: rec.market_open(), "market_open")):
        try:
            call()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"  {label}: FAILED ({exc})", file=sys.stderr)
    from datetime import date, timedelta
    today = date.today()
    for call, label in (
        (lambda: rec.daily_bars(sym, today - timedelta(days=cfg.history_days), today), "daily_bars"),
        (lambda: rec.dividends(sym, today - timedelta(days=400), today + timedelta(days=120)),
         "dividends"),
        (lambda: rec.next_earnings(sym), "next_earnings"),
    ):
        try:
            call()
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"  {label}: FAILED ({exc})", file=sys.stderr)
    print(f"recorded {n_ok} response(s) to {out} ({n_fail} failed)")
    return 0 if n_fail == 0 else 1


def cmd_backtest(args) -> int:
    from .backtest import Backtester, BacktestConfig
    cfg = _config(args)
    prov = _provider("replay", Path(args.snapshots))
    bt = Backtester(prov, cfg, backtest=BacktestConfig(
        n_trials_for_deflation=int(args.trials)))
    print(bt.run().render())
    return 0


def cmd_journal(args) -> int:
    from .data.journal import OutcomeJournal
    J = OutcomeJournal(Path(args.journal))
    s = J.summary()
    print(f"JOURNAL {s['path']}")
    print(f"  decisions {s['n_decisions']} | traded {s['n_traded']} | held "
          f"{s['n_hold']} ({(s['hold_rate'] or 0):.0%}) | closed {s['n_closed']}")
    print(f"  total P&L ${s['total_pnl']:,.2f} | hit rate "
          f"{s['hit_rate'] if s['hit_rate'] == s['hit_rate'] else float('nan'):.1%}")
    if s["gate_blocks"]:
        print("  what blocked (the constraint you are really trading against):")
        tot = sum(s["gate_blocks"].values())
        for name, n in list(s["gate_blocks"].items())[:15]:
            print(f"    {name:<28}{n:>6}  {n / max(tot, 1):>6.1%}")
    for d in J.open_positions():
        print(f"  OPEN  {d.id}  {d.structure_name}  x{d.contracts}")
    return 0


def cmd_learned(args) -> int:
    from .data.journal import OutcomeJournal
    from .learning.feedback import FeedbackLoop
    from .sizing.kelly import drawdown_cap, uncertainty_shrinkage
    cfg = _config(args)
    st = FeedbackLoop(cfg.feedback).ingest(OutcomeJournal(Path(args.journal)))
    print(st.report())
    c_unc = uncertainty_shrinkage(st.sharpe, st.years_of_evidence)
    c_dd = drawdown_cap(cfg.max_drawdown, cfg.drawdown_prob)
    c = min(c_unc, c_dd, cfg.hard_cap_fraction)
    print(f"\n  => Kelly multiplier c = {c:.4f} = min(uncertainty {c_unc:.4f}, "
          f"drawdown {c_dd:.4f}, hard cap {cfg.hard_cap_fraction:.2f})")
    print(f"     Excess growth scales as 2c - c^2, so this keeps "
          f"{2 * c - c * c:.1%} of maximum growth at {c:.0%} of full-Kelly volatility.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="optionsmarkets",
        description="Defined-risk options relative-value engine. Emits decisions and "
                    "order tickets; never places an order.")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--bankroll", type=float, default=40_000.0)
    ap.add_argument("--journal", default="journal/outcomes.jsonl")
    ap.add_argument("--snapshots", default=None,
                    help="snapshot-recording directory (required for replay/backtest)")
    ap.add_argument("--provider", default="yfinance",
                    choices=("yfinance", "synthetic", "replay"))
    ap.add_argument("--max-risk", type=float, default=None,
                    help="max fraction of bankroll at risk per trade")
    ap.add_argument("--etf", action="store_true",
                    help="treat the symbol as an index ETF with no earnings date")
    ap.add_argument("--trials", type=int, default=1,
                    help="honest count of configurations considered, for the "
                         "deflated Sharpe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, helptext in (
        ("decide", cmd_decide, "one decision now"),
        ("collect", cmd_collect, "record a snapshot to the journal"),
        ("backtest", cmd_backtest, "replay the recorded journal"),
        ("journal", cmd_journal, "summarise the outcome journal"),
        ("learned", cmd_learned, "show the learned state and its effect on sizing"),
    ):
        sub.add_parser(name, help=helptext).set_defaults(func=fn)

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
