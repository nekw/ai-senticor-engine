"""Signal agent for sentiment and volatility computation per competitor."""

import asyncio

from src.config import CURRENT_SENTIMENT_HEADLINES, HISTORICAL_SENTIMENT_HEADLINES
from src.core.analyzer import SentimentEngine
from src.core.data_fetcher import MarketDataClient
from src.core.processor import calculate_volatility

from .state import CompetitiveAnalysisState, CompetitorSignal


class SignalAgent:
    """Compute comparable per-ticker quantitative signals."""

    def __init__(self, client: MarketDataClient, sentiment_engine: SentimentEngine):
        """Initialize the signal agent.

        Args:
            client: Market data client used for historical price retrieval.
            sentiment_engine: Engine used to score financial headline sentiment.
        """
        self._client = client
        self._sentiment_engine = sentiment_engine

    async def run(self, state: CompetitiveAnalysisState) -> CompetitiveAnalysisState:
        """Calculate sentiment/trend/volatility for each peer in parallel."""

        async def compute_for_ticker(
            ticker: str,
        ) -> tuple[CompetitorSignal | None, str | None]:
            try:
                headlines = state.news_by_ticker.get(ticker, [])
                current = self._sentiment_engine.analyze_headlines(
                    headlines[:CURRENT_SENTIMENT_HEADLINES]
                )
                historical = self._sentiment_engine.analyze_headlines(
                    headlines[
                        CURRENT_SENTIMENT_HEADLINES:HISTORICAL_SENTIMENT_HEADLINES
                    ]
                )

                price_df = await self._client.fetch_historical_prices(ticker)
                if price_df.empty or "close" not in price_df.columns:
                    return (
                        None,
                        f"{ticker}: insufficient price data for signal computation",
                    )

                signal = CompetitorSignal(
                    ticker=ticker,
                    sentiment=current,
                    sentiment_trend=current - historical,
                    volatility=calculate_volatility(price_df["close"]),
                    news_count=len(headlines),
                )
                return signal, None
            except Exception as exc:  # pragma: no cover - defensive
                return None, f"{ticker}: signal computation failed ({exc})"

        results = await asyncio.gather(
            *(compute_for_ticker(t) for t in state.peer_universe)
        )
        state.signals = [signal for signal, err in results if signal is not None]

        for signal, err in results:
            if err:
                state.errors.append(err)

        # Keep the table stable for downstream rendering.
        state.signals.sort(key=lambda item: item.ticker)
        return state
