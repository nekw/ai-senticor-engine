"""Core analysis modules.

Heavy competitive agent dependencies are loaded lazily to avoid startup overhead
for flows that only need sentiment/data processing.
"""

__all__ = ["CompetitiveAnalysisOrchestrator", "CompetitiveQuery"]


def __getattr__(name: str):
    """Lazily expose competitive agent symbols at package level."""
    if name in __all__:
        from .agents import CompetitiveAnalysisOrchestrator, CompetitiveQuery

        exports = {
            "CompetitiveAnalysisOrchestrator": CompetitiveAnalysisOrchestrator,
            "CompetitiveQuery": CompetitiveQuery,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
