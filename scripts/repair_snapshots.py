#!/usr/bin/env python
"""Re-home misfiled snapshot recordings and reconcile the market-wide ones.

A ``collect`` run pointed at the wrong ``--snapshots`` directory files six
recordings under someone else's ticker.  That is not cosmetic: ReplayProvider
ignores the ``symbol`` argument and serves whatever chain sits at the cursor,
so one stray option_chain makes the backtest replay one underlying as another
-- silently, and with a plausible-looking result.

Two passes, both idempotent:

  1. Symbol-scoped recordings (option_chain, daily_bars, dividends,
     next_earnings) carry their ticker in ``meta.args[0]``.  Any file whose
     ticker disagrees with its parent directory is MOVED to the right one.

  2. risk_free_curve and market_open carry no ticker -- they describe the
     market, not the name.  Every ticker directory therefore wants every one
     of them, so the union is COPIED into each.  This is what keeps a strict
     replay from raising LookAheadError on a directory whose curve recordings
     happened to be written during some other ticker's collect cycle.

Nothing is ever deleted.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "journal/snapshots")


def meta_of(f: Path) -> dict:
    try:
        return json.loads(f.read_text())["meta"]
    except (json.JSONDecodeError, KeyError, OSError):
        return {}


def main() -> int:
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    if not dirs:
        print(f"no ticker directories under {ROOT}")
        return 1

    # ---- pass 1: re-home symbol-scoped recordings ------------------------
    moved = 0
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            args = meta_of(f).get("args") or []
            if not args:
                continue
            owner = str(args[0]).upper()
            if owner == d.name.upper():
                continue
            dest = ROOT / owner
            dest.mkdir(exist_ok=True)
            shutil.move(str(f), str(dest / f.name))
            print(f"  moved  {f.name}  {d.name} -> {owner}")
            moved += 1

    # ---- pass 2: every ticker gets every market-wide recording -----------
    dirs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    shared: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.glob("*.json")):
            if not (meta_of(f).get("args") or []):
                shared.setdefault(f.name, f)

    copied = 0
    for d in dirs:
        for name, src in shared.items():
            dst = d / name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))
                copied += 1

    print(f"\n{moved} file(s) re-homed, {copied} market-wide file(s) reconciled")
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        chains = [f for f in d.glob("*_option_chain.json")]
        owners = {str(meta_of(f).get("args", ["?"])[0]).upper() for f in chains}
        flag = "" if owners <= {d.name.upper()} else f"  <-- STILL MIXED: {sorted(owners)}"
        print(f"  {d.name:6s} {len(chains)} chain snapshot(s){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
