"""Application layer: the runner that composes the library into a decision.

Everything below this package is a library of independently-testable pieces.
This package is the only place that knows the *order* they go in, which keeps
the ordering itself reviewable in one file rather than implicit across ten.
"""

from .config import RunConfig
from .pipeline import Pipeline, PipelineResult

__all__ = ["RunConfig", "Pipeline", "PipelineResult"]
