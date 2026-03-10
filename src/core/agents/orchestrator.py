"""Lightweight asyncio orchestrator for competitive analysis agents."""

from src.core.analyzer import SentimentEngine
from src.core.data_fetcher import MarketDataClient
from src.core.rag_engine import RAGEngine

from .narrative_agent import NarrativeAgent
from .news_intel_agent import NewsIntelAgent
from .report_agent import ReportAgent
from .risk_agent import RiskAgent
from .signal_agent import SignalAgent
from .state import CompetitiveAnalysisState, CompetitiveQuery
from .universe_agent import UniverseAgent


class CompetitiveAnalysisOrchestrator:
    """Coordinator that runs specialized agents over shared state."""

    def __init__(
        self,
        client: MarketDataClient,
        sentiment_engine: SentimentEngine,
        rag_engine: RAGEngine,
    ):
        """Initialize orchestrator dependencies and worker agents.

        Args:
            client: Market data client for news and price retrieval.
            sentiment_engine: Sentiment engine used by signal computation.
            rag_engine: RAG engine for sector mapping and narrative generation.
        """
        self.universe_agent = UniverseAgent(rag_engine)
        self.news_intel_agent = NewsIntelAgent(client)
        self.signal_agent = SignalAgent(client, sentiment_engine)
        self.narrative_agent = NarrativeAgent(rag_engine)
        self.risk_agent = RiskAgent()
        self.report_agent = ReportAgent()

    async def run(self, query: CompetitiveQuery) -> dict:
        """Execute the end-to-end competitive analysis workflow."""
        state = CompetitiveAnalysisState(query=query)

        state = self.universe_agent.run(state)
        state = await self.news_intel_agent.run(state)
        state = await self.signal_agent.run(state)
        state = self.narrative_agent.run(state)
        state = self.risk_agent.run(state)

        return self.report_agent.run(state)
