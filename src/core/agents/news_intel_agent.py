"""News intelligence agent for gathering competitor headlines."""

import asyncio

from src.core.data_fetcher import MarketDataClient

from .state import CompetitiveAnalysisState


class NewsIntelAgent:
    """Fetch and normalize recent headlines for each competitor."""

    def __init__(self, client: MarketDataClient):
        """Initialize the news intelligence agent.

        Args:
            client: Market data client used to fetch company news.
        """
        self._client = client

    async def run(self, state: CompetitiveAnalysisState) -> CompetitiveAnalysisState:
        """Collect bounded headline lists for all peers in parallel."""

        async def fetch_for_ticker(ticker: str) -> tuple[str, list[str], str | None]:
            try:
                df = await self._client.fetch_company_news(ticker)
                if df.empty or "title" not in df.columns:
                    return ticker, [], f"{ticker}: no usable news returned"

                headlines = (
                    df["title"]
                    .dropna()
                    .astype(str)
                    .head(state.query.max_articles_per_ticker)
                    .tolist()
                )
                return ticker, headlines, None
            except Exception as exc:  # pragma: no cover - defensive
                return ticker, [], f"{ticker}: news fetch failed ({exc})"

        tasks = [fetch_for_ticker(t) for t in state.peer_universe]
        results = await asyncio.gather(*tasks)

        for ticker, headlines, err in results:
            state.news_by_ticker[ticker] = headlines
            if err:
                state.errors.append(err)

        return state
