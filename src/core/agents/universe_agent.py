"""Universe agent for selecting a competitor set from sector mappings."""

from src.core.rag_engine import RAGEngine

from .state import CompetitiveAnalysisState


class UniverseAgent:
    """Resolve sector and build a bounded peer universe."""

    def __init__(self, rag_engine: RAGEngine):
        """Initialize the universe agent.

        Args:
            rag_engine: RAG engine used for sector mapping and peer lookup.
        """
        self._rag_engine = rag_engine

    def run(self, state: CompetitiveAnalysisState) -> CompetitiveAnalysisState:
        """Populate sector and peer list for the focus ticker."""
        focus = state.query.focus_ticker.upper()
        sector = state.query.sector or self._rag_engine.get_sector(focus)

        if not sector or sector == "Unknown":
            state.errors.append(
                f"{focus}: sector not found in mapping; provide query.sector explicitly"
            )
            state.sector = "Unknown"
            state.peer_universe = [focus]
            return state

        peers = self._rag_engine.get_sector_tickers(sector)
        peers = [p for p in peers if p != focus]
        peers = [focus] + sorted(peers)[: max(state.query.max_competitors - 1, 0)]

        state.sector = sector
        state.peer_universe = peers
        return state
