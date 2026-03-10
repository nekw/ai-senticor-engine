"""Narrative agent that adds RAG-backed sector commentary and citations."""

from src.core.rag_engine import RAGEngine

from .state import CompetitiveAnalysisState


class NarrativeAgent:
    """Generate competitive narrative from sector RAG context."""

    def __init__(self, rag_engine: RAGEngine):
        """Initialize the narrative agent.

        Args:
            rag_engine: RAG engine used to generate sector-aware commentary.
        """
        self._rag_engine = rag_engine

    def run(self, state: CompetitiveAnalysisState) -> CompetitiveAnalysisState:
        """Generate commentary for focus ticker and capture source headlines."""
        if not state.signals:
            state.errors.append("Narrative skipped: no computed signals available")
            return state

        focus = state.query.focus_ticker.upper()
        focus_signal = next(
            (s for s in state.signals if s.ticker == focus), state.signals[0]
        )

        result = self._rag_engine.get_sector_commentary(
            ticker=focus_signal.ticker,
            company_sentiment=focus_signal.sentiment,
            k=min(10, state.query.max_articles_per_ticker),
            return_sources=True,
        )

        if isinstance(result, tuple):
            commentary, sources = result
        else:  # pragma: no cover - compatibility path
            commentary, sources = result, []

        state.narrative = commentary
        state.citations[focus_signal.ticker] = [
            article.get("headline", "")
            for article in sources
            if article.get("headline")
        ]
        return state
