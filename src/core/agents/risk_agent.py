"""Risk agent for confidence scoring and autonomous guardrails."""

from .state import CompetitiveAnalysisState, RiskAssessment


class RiskAgent:
    """Score confidence using data completeness and error burden."""

    def run(self, state: CompetitiveAnalysisState) -> CompetitiveAnalysisState:
        """Attach confidence score and flags for downstream consumers."""
        flags: list[str] = []

        if not state.peer_universe:
            flags.append("No peer universe was resolved")

        if len(state.signals) < max(1, len(state.peer_universe) // 2):
            flags.append("Low signal coverage across peer universe")

        if state.errors:
            flags.append("One or more agents reported recoverable errors")

        if not state.narrative:
            flags.append("Narrative output missing or empty")

        coverage = len(state.signals) / max(1, len(state.peer_universe))
        error_penalty = min(len(state.errors) * 0.1, 0.5)
        confidence = max(0.0, min(1.0, coverage - error_penalty))

        state.risk = RiskAssessment(confidence=confidence, flags=flags)
        return state
