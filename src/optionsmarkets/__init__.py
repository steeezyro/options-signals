"""optionsmarkets -- a defined-risk options relative-value engine.

Pipeline (each stage refuses to run on output the previous stage flagged bad):

    feed -> quality screen -> forward from parity -> de-Americanise
         -> in-house IV inversion -> SVI/SSVI surface + arbitrage gates
         -> RV forecast (HAR/HARQ) -> variance risk premium
         -> Q->P scenario density -> structure enumeration -> edge scoring
         -> fractional Kelly with uncertainty + drawdown shrinkage
         -> gate stack -> BUY/SELL/HOLD -> Schwab order ticket
         -> outcome journal -> Kalman / conformal / calibration update

Nothing here is investment advice, and no backtest in this repo constitutes
evidence that the strategy is profitable. See BLUEPRINT.md section 12.
"""

__version__ = "0.1.0"

from . import data, domain, edge, execution, forecast, learning, policy, pricing, sizing, surface  # noqa: F401

__all__ = ["data", "domain", "edge", "execution", "forecast", "learning",
           "policy", "pricing", "sizing", "surface", "__version__"]
