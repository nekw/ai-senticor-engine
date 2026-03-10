"""Report agent for producing compact competitive-analysis payloads."""

from .state import CompetitiveAnalysisState


class ReportAgent:
    """Compile final serializable payload from shared state."""

    def run(self, state: CompetitiveAnalysisState) -> dict:
        """Return normalized output for UI/API consumers."""
        return {
            "generated_at": state.generated_at,
            "focus_ticker": state.query.focus_ticker.upper(),
            "sector": state.sector,
            "peer_universe": state.peer_universe,
            "signals": [
                {
                    "ticker": s.ticker,
                    "sentiment": s.sentiment,
                    "sentiment_trend": s.sentiment_trend,
                    "volatility": s.volatility,
                    "news_count": s.news_count,
                }
                for s in state.signals
            ],
            "narrative": state.narrative,
            "citations": state.citations,
            "confidence": state.risk.confidence if state.risk else 0.0,
            "risk_flags": state.risk.flags if state.risk else [],
            "errors": state.errors,
        }
