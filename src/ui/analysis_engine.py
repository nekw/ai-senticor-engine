"""Data analysis and processing logic for ticker analysis."""

import asyncio

import pandas as pd

from src.config import CURRENT_SENTIMENT_HEADLINES, HISTORICAL_SENTIMENT_HEADLINES
from src.core.analyzer import SentimentEngine
from src.core.data_fetcher import MarketDataClient
from src.core.processor import calculate_volatility
from src.utils.logger import AppLogger


def validate_ticker_data(
    ticker: str, price_data: pd.DataFrame, news_data: pd.DataFrame
) -> tuple[bool, str]:
    """Validate that ticker has sufficient data for analysis.

    Args:
        ticker: Stock ticker symbol.
        price_data: Historical price DataFrame.
        news_data: News articles DataFrame.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if price_data.empty or len(price_data) < 5:
        return False, f"{ticker}: Insufficient price data"

    if news_data.empty or "title" not in news_data.columns or len(news_data) < 5:
        return (
            False,
            f"{ticker}: Insufficient news data (try a different news provider)",
        )

    return True, ""


async def analyze_ticker(
    ticker: str, client: MarketDataClient, engine: SentimentEngine
) -> dict:
    """Analyze a single ticker and return metrics.

    Args:
        ticker: Stock ticker symbol.
        client: Market data client instance.
        engine: Sentiment analysis engine instance.

    Returns:
        Dictionary with analysis results and cache data.

    Raises:
        Exception: If data fetching or analysis fails.
    """
    AppLogger.info(
        "Fetching data", "Retrieving price and news data in parallel", ticker=ticker
    )

    # Fetch price and news data in parallel using asyncio.gather
    price_task = asyncio.create_task(client.fetch_historical_prices(ticker))
    news_task = asyncio.create_task(client.fetch_company_news(ticker))

    price_data, news_data = await asyncio.gather(price_task, news_task)

    AppLogger.info(
        "Data fetched",
        f"Price: {len(price_data)} rows, News: {len(news_data)} articles",
        ticker=ticker,
    )

    # Validate data
    is_valid, error_msg = validate_ticker_data(ticker, price_data, news_data)
    if not is_valid:
        AppLogger.error("Data validation failed", error_msg, ticker=ticker)
        raise ValueError(error_msg)

    AppLogger.info(
        "Analyzing sentiment",
        "Processing {} current + {} historical headlines".format(
            CURRENT_SENTIMENT_HEADLINES,
            HISTORICAL_SENTIMENT_HEADLINES - CURRENT_SENTIMENT_HEADLINES,
        ),
        ticker=ticker,
    )

    # Analyze sentiment
    current_sentiment = engine.analyze_headlines(
        news_data["title"].head(CURRENT_SENTIMENT_HEADLINES).tolist()
    )
    historical_sentiment = engine.analyze_headlines(
        news_data["title"]
        .iloc[CURRENT_SENTIMENT_HEADLINES:HISTORICAL_SENTIMENT_HEADLINES]
        .tolist()
    )
    sentiment_trend = current_sentiment - historical_sentiment

    AppLogger.success(
        "Sentiment analysis complete",
        f"Current: {current_sentiment:.2f}, Trend: {sentiment_trend:+.2f}",
        ticker=ticker,
    )

    # Calculate volatility
    volatility = calculate_volatility(price_data["close"])
    AppLogger.info(
        "Volatility calculated", f"Volatility: {volatility:.4f}", ticker=ticker
    )

    return {
        "result": {
            "ticker": ticker,
            "sentiment": current_sentiment,
            "volatility": volatility,
            "trend": sentiment_trend,
        },
        "cache": {
            "price": price_data,
            "news": news_data,
            "sent": current_sentiment,
            "trend": sentiment_trend,
        },
    }


async def run_analysis(
    ticker_list: list[str],
    client: MarketDataClient,
    engine: SentimentEngine,
    progress_callback=None,
) -> tuple[list[dict], list[str], dict]:
    """Run analysis on a list of tickers in parallel.

    Args:
        ticker_list: List of ticker symbols to analyze.
        client: Market data client instance.
        engine: Sentiment analysis engine instance.
        progress_callback: Optional callback function(current, total, ticker) for progress updates.

    Returns:
        Tuple of (results, errors, cache).
    """
    results = []
    errors = []
    cache = {}
    total = len(ticker_list)
    completed = {"count": 0}  # Mutable counter for tracking completed tasks

    async def analyze_with_error_handling(ticker: str, idx: int):
        """Analyze a ticker and handle errors."""
        try:
            AppLogger.info(
                "Processing ticker",
                f"Starting analysis pipeline [{idx}/{total}]",
                ticker=ticker,
            )

            analysis = await analyze_ticker(ticker, client, engine)

            # Update progress after completion
            completed["count"] += 1
            if progress_callback:
                progress_callback(completed["count"], total, ticker)

            AppLogger.success(
                "Ticker analysis complete",
                f"Successfully analyzed {ticker} [{completed['count']}/{total}]",
                ticker=ticker,
            )
            return ("success", ticker, analysis)
        except Exception as e:
            error_msg = f"{ticker}: {str(e)}"

            # Update progress even on error
            completed["count"] += 1
            if progress_callback:
                progress_callback(completed["count"], total, ticker)

            AppLogger.error("Ticker analysis failed", str(e), ticker=ticker)
            return ("error", ticker, error_msg)

    # Create tasks for all tickers to run in parallel
    tasks = [
        asyncio.create_task(analyze_with_error_handling(ticker, idx))
        for idx, ticker in enumerate(ticker_list, 1)
    ]

    # Wait for all tasks to complete
    task_results = await asyncio.gather(*tasks)

    # Process results
    for status, ticker, data in task_results:
        if status == "success":
            results.append(data["result"])
            cache[ticker] = data["cache"]
        else:
            errors.append(data)

    return results, errors, cache
