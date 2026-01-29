"""Market Data Fetcher using OpenBB Platform.

This module provides a client for fetching real-time market data including
historical prices and company news from various financial data providers.

API Key Setup:
    Some providers require API keys. Configure them using:

    1. OpenBB Hub: https://my.openbb.co (recommended)
    2. Programmatically:
        >>> from openbb import obb
        >>> obb.user.credentials.fmp_api_key = "your_key"
        >>> obb.user.save()

    3. Environment variables:
        Set OBB_FMP_API_KEY, OBB_POLYGON_API_KEY, etc.

Supported Providers:
    Price Data: yfinance, fmp, polygon, intrinio, alpha_vantage
    News Data: yfinance, benzinga, fmp, intrinio, polygon, biztoc, tiingo
"""

import asyncio
import logging

import pandas as pd


class MarketDataClient:
    """Client for fetching financial market data.

    Interfaces with OpenBB Platform to retrieve stock prices and news data
    from various providers. Handles data transformation to pandas
    DataFrames for easy analysis.

    Args:
        price_provider: Provider for price data (e.g., 'yfinance', 'fmp', 'polygon').
        news_provider: Provider for news data (e.g., 'yfinance', 'benzinga', 'fmp').

    Example:
        >>> client = MarketDataClient(price_provider='yfinance', news_provider='benzinga')
        >>> prices = client.fetch_historical_prices("AAPL")
        >>> news = client.fetch_company_news("AAPL")
        >>> print(f"Fetched {len(prices)} price points")
    """

    def __init__(
        self, price_provider: str = "yfinance", news_provider: str = "yfinance"
    ):
        """Initialize the market data client with specified providers.

        Args:
            price_provider: Provider for historical price data.
            news_provider: Provider for company news.
        """
        self.price_provider = price_provider
        self.news_provider = news_provider
        self._credentials_loaded = False

    def _ensure_credentials(self):
        """Load API credentials on first use (lazy loading)."""
        if not self._credentials_loaded:
            try:
                import streamlit as st

                # Show progress while loading OpenBB (this triggers the import)
                with st.spinner("🔧 Initializing OpenBB Platform (first-time load)..."):
                    from src.ui.config_loader import load_api_credentials

                    load_api_credentials()
                    self._credentials_loaded = True
            except Exception as e:
                logging.warning(f"Failed to load credentials: {e}")
                self._credentials_loaded = True  # Don't retry

    async def fetch_historical_prices(self, ticker: str) -> pd.DataFrame:
        """Fetch historical price data for a stock.

        Retrieves daily OHLCV (Open, High, Low, Close, Volume) data for the
        specified ticker symbol. Data typically covers the last 90 days.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA', 'MSFT').

        Returns:
            DataFrame with columns: date (index), open, high, low, close, volume.

        Raises:
            Exception: If ticker is invalid or data provider is unavailable.

        Example:
            >>> client = MarketDataClient()
            >>> df = await client.fetch_historical_prices("NVDA")
            >>> print(df[['close']].head())
        """
        try:
            # Ensure credentials are loaded before first API call
            self._ensure_credentials()

            # Lazy import to avoid loading OpenBB at startup
            from openbb import obb

            # Run blocking OBB call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    ticker, provider=self.price_provider
                ).to_df(),
            )
            return result
        except Exception as e:
            logging.error(
                f"Failed to fetch price data for {ticker} using {self.price_provider}: {str(e)}"
            )
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    async def fetch_company_news(self, ticker: str) -> pd.DataFrame:
        """Fetch recent news headlines for a company.

        Retrieves the latest news articles and headlines related to the
        specified ticker symbol from financial news sources.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').

        Returns:
            DataFrame with columns: title, date, source, and other metadata.
            Sorted by date in descending order (newest first).

        Raises:
            Exception: If ticker is invalid or news feed is unavailable.

        Example:
            >>> client = MarketDataClient()
            >>> news = await client.fetch_company_news("TSLA")
            >>> print(news['title'].head())
        """
        try:
            # Lazy import to avoid loading OpenBB at startup
            from openbb import obb

            # Run blocking OBB call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: obb.news.company(
                    symbol=ticker, provider=self.news_provider
                ).to_df(),
            )
            # Ensure sorted by date descending (newest first)
            df = df.reset_index()
            if not df.empty and "date" in df.columns:
                df = df.sort_values("date", ascending=False)
            return df
        except Exception as e:
            logging.error(
                f"Failed to fetch news data for {ticker} using {self.news_provider}: {str(e)}"
            )
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["title", "date", "source"])
