"""Unit tests for market data fetcher."""

import asyncio
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.core.data_fetcher import MarketDataClient


class TestMarketDataClient:
    """Test suite for MarketDataClient class."""

    @pytest.fixture
    def mock_obb(self):
        """Create a mock obb object for all tests."""
        with patch("openbb.obb") as mock:
            yield mock

    @pytest.fixture
    def client(self):
        """Create a MarketDataClient instance."""
        return MarketDataClient()

    @pytest.fixture
    def mock_price_data(self):
        """Create mock historical price data."""
        return pd.DataFrame(
            {
                "open": [100, 102, 101, 103],
                "high": [103, 104, 104, 106],
                "low": [99, 101, 100, 102],
                "close": [102, 101, 103, 105],
                "volume": [1000000, 1200000, 900000, 1100000],
            }
        )

    @pytest.fixture
    def mock_news_data(self):
        """Create mock news data."""
        return pd.DataFrame(
            {
                "title": [
                    "Stock reaches new highs",
                    "Company announces earnings",
                    "Market volatility increases",
                ],
                "date": ["2026-01-24", "2026-01-23", "2026-01-22"],
                "source": ["Reuters", "Bloomberg", "CNBC"],
            }
        )

    @pytest.mark.asyncio
    async def test_fetch_historical_prices_success(
        self, mock_obb, client, mock_price_data
    ):
        """Test successful historical price fetch."""
        mock_result = Mock()
        mock_result.to_df.return_value = mock_price_data
        mock_obb.equity.price.historical.return_value = mock_result

        result = await client.fetch_historical_prices("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]
        mock_obb.equity.price.historical.assert_called_once_with(
            "AAPL", provider="yfinance"
        )

    @pytest.mark.asyncio
    async def test_fetch_historical_prices_different_ticker(
        self, mock_obb, client, mock_price_data
    ):
        """Test fetching prices for different tickers."""
        mock_result = Mock()
        mock_result.to_df.return_value = mock_price_data
        mock_obb.equity.price.historical.return_value = mock_result

        tickers = ["AAPL", "TSLA", "NVDA"]
        for ticker in tickers:
            await client.fetch_historical_prices(ticker)
            # Verify it was called with the correct ticker
            assert mock_obb.equity.price.historical.call_args[0][0] == ticker

    @pytest.mark.asyncio
    async def test_fetch_company_news_success(self, mock_obb, client, mock_news_data):
        """Test successful company news fetch."""
        mock_result = Mock()
        mock_result.to_df.return_value = mock_news_data
        mock_obb.news.company.return_value = mock_result

        result = await client.fetch_company_news("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "title" in result.columns
        assert "date" in result.columns
        mock_obb.news.company.assert_called_once_with(
            symbol="AAPL", provider="yfinance"
        )

    @pytest.mark.asyncio
    async def test_fetch_company_news_different_ticker(
        self, mock_obb, client, mock_news_data
    ):
        """Test fetching news for different tickers."""
        mock_result = Mock()
        mock_result.to_df.return_value = mock_news_data
        mock_obb.news.company.return_value = mock_result

        tickers = ["GOOGL", "MSFT", "AMZN"]
        for ticker in tickers:
            await client.fetch_company_news(ticker)
            # Verify it was called with the correct ticker
            assert mock_obb.news.company.call_args[1]["symbol"] == ticker

    @pytest.mark.asyncio
    async def test_fetch_prices_provider_parameter(self, mock_obb, client):
        """Test that provider parameter is correctly passed."""
        mock_result = Mock()
        mock_result.to_df.return_value = pd.DataFrame()
        mock_obb.equity.price.historical.return_value = mock_result

        await client.fetch_historical_prices("AAPL")

        # Check provider is yfinance
        call_kwargs = mock_obb.equity.price.historical.call_args[1]
        assert call_kwargs["provider"] == "yfinance"

    @pytest.mark.asyncio
    async def test_fetch_news_provider_parameter(self, mock_obb, client):
        """Test that provider parameter is correctly passed for news."""
        mock_result = Mock()
        mock_result.to_df.return_value = pd.DataFrame()
        mock_obb.news.company.return_value = mock_result

        await client.fetch_company_news("AAPL")

        # Check provider is yfinance
        call_kwargs = mock_obb.news.company.call_args[1]
        assert call_kwargs["provider"] == "yfinance"

    @pytest.mark.asyncio
    async def test_fetch_prices_empty_dataframe(self, mock_obb, client):
        """Test handling of empty price data."""
        mock_result = Mock()
        mock_result.to_df.return_value = pd.DataFrame()
        mock_obb.equity.price.historical.return_value = mock_result

        result = await client.fetch_historical_prices("INVALID")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_news_empty_dataframe(self, mock_obb, client):
        """Test handling of empty news data."""
        mock_result = Mock()
        mock_result.to_df.return_value = pd.DataFrame()
        mock_obb.news.company.return_value = mock_result

        result = await client.fetch_company_news("INVALID")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_parallel_data_fetching(
        self, mock_obb, client, mock_price_data, mock_news_data
    ):
        """Test that price and news can be fetched in parallel."""
        mock_price_result = Mock()
        mock_price_result.to_df.return_value = mock_price_data
        mock_news_result = Mock()
        mock_news_result.to_df.return_value = mock_news_data

        mock_obb.equity.price.historical.return_value = mock_price_result
        mock_obb.news.company.return_value = mock_news_result

        # Fetch in parallel using asyncio.gather
        price_task = client.fetch_historical_prices("AAPL")
        news_task = client.fetch_company_news("AAPL")
        price_data, news_data = await asyncio.gather(price_task, news_task)

        assert isinstance(price_data, pd.DataFrame)
        assert isinstance(news_data, pd.DataFrame)
        assert len(price_data) == 4
        assert len(news_data) == 3

    @pytest.mark.asyncio
    async def test_fetch_prices_error_handling(self, mock_obb, client):
        """Test error handling in fetch_historical_prices."""
        mock_obb.equity.price.historical.side_effect = Exception("API Error")

        result = await client.fetch_historical_prices("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    @pytest.mark.asyncio
    async def test_fetch_news_error_handling(self, mock_obb, client):
        """Test error handling in fetch_company_news."""
        mock_obb.news.company.side_effect = Exception("API Error")

        result = await client.fetch_company_news("INVALID")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["title", "date", "source"]

    @pytest.mark.asyncio
    async def test_fetch_prices_connection_timeout(self, mock_obb, client):
        """Test handling of connection timeout errors."""
        mock_obb.equity.price.historical.side_effect = TimeoutError(
            "Connection timeout"
        )

        result = await client.fetch_historical_prices("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    @pytest.mark.asyncio
    async def test_fetch_news_connection_timeout(self, mock_obb, client):
        """Test handling of connection timeout in news fetch."""
        mock_obb.news.company.side_effect = TimeoutError("Connection timeout")

        result = await client.fetch_company_news("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["title", "date", "source"]

    @pytest.mark.asyncio
    async def test_fetch_prices_invalid_ticker_format(self, mock_obb, client):
        """Test handling of invalid ticker format."""
        mock_obb.equity.price.historical.side_effect = ValueError(
            "Invalid ticker format"
        )

        result = await client.fetch_historical_prices("!!!INVALID!!!")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_news_rate_limit_error(self, mock_obb, client):
        """Test handling of API rate limit errors."""
        mock_obb.news.company.side_effect = Exception("Rate limit exceeded")

        result = await client.fetch_company_news("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_prices_malformed_response(self, mock_obb, client):
        """Test handling of malformed API response."""
        mock_result = Mock()
        mock_result.to_df.side_effect = AttributeError("Missing attribute")
        mock_obb.equity.price.historical.return_value = mock_result

        result = await client.fetch_historical_prices("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_news_malformed_response(self, mock_obb, client):
        """Test handling of malformed news response."""
        mock_result = Mock()
        mock_result.to_df.side_effect = KeyError("Missing key")
        mock_obb.news.company.return_value = mock_result

        result = await client.fetch_company_news("AAPL")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
