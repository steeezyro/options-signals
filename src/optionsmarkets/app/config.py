"""Run configuration.

Every field here is a policy dial, and BLUEPRINT.md section 16 is blunt about
what that means: there are more than fifteen of them, that is a lot of implicit
multiple testing, and they must be re-derived from your own out-of-sample record
and reported with a deflated statistic rather than a nominal one.  The defaults
are a starting point for a $5k-$50k Reg-T account, not a recommendation.

The fields are grouped by which stage consumes them so that changing one is a
local decision with a visible blast radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..domain.candidates import CandidateConfig
from ..learning.feedback import FeedbackConfig
from ..policy.decide import Thresholds


@dataclass
class RunConfig:
    # ---- what and how much ---------------------------------------------
    symbol: str = "SPY"
    bankroll: float = 40_000.0

    # ---- data -----------------------------------------------------------
    # The freshness gate is set to the FEED'S REAL LATENCY, not to a number
    # that sounds tight. yfinance chains are ~15 min delayed; claiming 60s here
    # would simply mean every decision fails the gate, which teaches nothing.
    assumed_latency_s: float = 900.0
    history_days: int = 750           # ~3 years: enough for a 22-day HAR
    min_two_sided_quotes: int = 12
    # spread/(2*vega) rejection limit, in vol points. The single most valuable
    # filter in the pipeline -- see BLUEPRINT.md section 4.2.
    max_spread_vols: float = 0.05
    chain_max_rel_spread: float = 0.25   # for IV inversion (looser than the trade gate)
    chain_min_open_interest: float = 25.0

    # ---- maturity selection --------------------------------------------
    # 21-45 DTE is the latency-tolerant window the whole design is scoped to:
    # the edge is a multi-day variance premium, so a 15-minute-old chain is
    # immaterial. Below ~7 DTE gamma and pin risk dominate the modelled edge.
    target_dte: tuple[int, int] = (21, 45)
    max_expiries_for_surface: int = 5
    # Which expiry inside the target band to actually trade.
    #
    # 'furthest' is the default and it is not a preference -- it is a
    # correctness requirement given the 21-DTE time stop. Selecting the NEAREST
    # expiry in a (21, 45) band means entering at 21-25 DTE and time-stopping
    # within days: the position earns almost none of the variance premium that
    # justified it, carries full gamma and delta for the few days it is on, and
    # pays entry and exit costs plus two spread crossings. A replay over 180
    # sessions with 'nearest' produced three trades, all losers, average -25% of
    # capital at risk, every one closed by the time stop -- while the stated PoP
    # was 75%, because PoP is computed to EXPIRY and the trade never got there.
    # Entering at the far end instead leaves ~24 days between entry and the time
    # stop, which is the window the edge was measured over.
    expiry_preference: str = "furthest"        # 'furthest' | 'nearest' | 'mid'
    # Minimum room between entry DTE and the time stop. Below this the trade is
    # refused outright rather than opened and immediately closed: a structure
    # whose exit is already triggered at entry has no thesis, only costs.
    min_holding_days: int = 7

    # ---- rates, carry, events ------------------------------------------
    fallback_rate: float = 0.0421
    fallback_dividend_yield: float = 0.0118
    # Fail CLOSED when the earnings feed is unavailable. An unknown event date
    # is not the same as no event, and the IV crush around earnings is a
    # different trade with different math (BLUEPRINT.md section 9).
    earnings_unknown_action: str = "block"     # 'block' | 'ignore'
    # De-Americanisation is a Leisen-Reimer root-solve PER QUOTE. It is correct
    # and it is expensive; on a broad index ETF with a small, well-behaved
    # early-exercise premium the bias it removes is under a tenth of a vol
    # point. Turn it on for single names with real dividends.
    de_americanise: bool = False
    de_americanise_steps: int = 101

    # ---- forecast -------------------------------------------------------
    rv_estimator: str = "yang_zhang"           # 'yang_zhang' | 'daily_proxy'
    har_include_today_in_week: bool = True
    har_use_q: bool = False                    # needs intraday RQ; off on daily bars
    # Range estimators read 5-15% low because the discrete high/low understates
    # the continuous range. Calibrate this against 5-minute RV before trusting
    # the LEVEL; 1.0 means "uncorrected, and I know it".
    rv_discreteness_correction: float = 1.0

    # ---- sizing ---------------------------------------------------------
    max_risk_fraction_per_trade: float = 0.02
    max_drawdown: float = 0.30
    drawdown_prob: float = 0.10
    hard_cap_fraction: float = 0.50            # the half-Kelly floor, always
    max_contracts: int | None = None

    # ---- policy ---------------------------------------------------------
    thresholds: Thresholds = field(default_factory=Thresholds)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)

    # ---- surface --------------------------------------------------------
    envelope_band: float = 0.15
    surface_max_rmse_vol: float = 0.02

    # ---- execution ------------------------------------------------------
    duration: str = "DAY"
    profit_target_pct: float = 0.50
    stop_multiple: float = 2.0
    manage_dte: int = 21

    # ---- io -------------------------------------------------------------
    journal_path: Path = Path("journal/outcomes.jsonl")
    recording_path: Path = Path("journal/snapshots")
    write_journal: bool = True

    def __post_init__(self):
        self.journal_path = Path(self.journal_path)
        self.recording_path = Path(self.recording_path)
        lo, hi = self.target_dte
        if not (0 < lo <= hi):
            raise ValueError(f"target_dte must be an increasing positive pair, got {self.target_dte}")
        if self.earnings_unknown_action not in ("block", "ignore"):
            raise ValueError("earnings_unknown_action must be 'block' or 'ignore'")
        if self.expiry_preference not in ("furthest", "nearest", "mid"):
            raise ValueError("expiry_preference must be 'furthest', 'nearest' or 'mid'")
        # The band must be able to contain a tradeable holding period at all,
        # or every run HOLDs on risk.holding_window and the config is a trap.
        if hi < self.manage_dte + self.min_holding_days:
            raise ValueError(
                f"target_dte upper bound {hi} leaves no room above the "
                f"{self.manage_dte}-DTE time stop for {self.min_holding_days} days of "
                f"holding. Raise target_dte, lower manage_dte, or lower "
                f"min_holding_days -- as configured every trade would be time-stopped "
                f"at or near entry.")
        # The freshness gate must not be tighter than the feed can ever satisfy.
        if self.thresholds.max_quote_age_s < self.assumed_latency_s:
            self.thresholds.max_quote_age_s = float(self.assumed_latency_s)


__all__ = ["RunConfig"]
