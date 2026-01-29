"""Unit tests for analysis engine."""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from src.ui.analysis_engine import analyze_ticker, run_analysis, validate_ticker_data


class TestAnalysisEngine:
    """Test suite for analysis engine functions."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock MarketDataClient."""
        client = Mock()
        client.fetch_historical_prices = AsyncMock()
        client.fetch_company_news = AsyncMock()
        return client

    @pytest.fixture
    def mock_engine(self):
        """Create a mock SentimentEngine."""
        engine = Mock()
        engine.analyze_headlines.return_value = 0.5
        return engine

    @pytest.fixture
    def mock_price_data(self):
        """Create mock price data."""
        return pd.DataFrame(
            {
                "open": [100, 102, 101, 103, 104, 105],
                "high": [103, 104, 104, 106, 107, 108],
                "low": [99, 101, 100, 102, 103, 104],
                "close": [102, 101, 103, 105, 106, 107],
                "volume": [1000000, 1200000, 900000, 1100000, 1050000, 980000],
            }
        )

    @pytest.fixture
    def mock_news_data(self):
        """Create mock news data."""
        return pd.DataFrame(
            {
                "title": [f"News headline {i}" for i in range(20)],
                "date": [f"2026-01-{i:02d}" for i in range(1, 21)],
                "source": ["Reuters"] * 20,
            }
        )

    def test_validate_ticker_data_success(self, mock_price_data, mock_news_data):
        """Test successful ticker data validation."""
        is_valid, error_msg = validate_ticker_data(
            "AAPL", mock_price_data, mock_news_data
        )
        assert is_valid is True
        assert error_msg == ""

    def test_validate_ticker_data_insufficient_price(self, mock_news_data):
        """Test validation fails with insufficient price data."""
        price_data = pd.DataFrame({"close": [100, 101]})
        is_valid, error_msg = validate_ticker_data("AAPL", price_data, mock_news_data)
        assert is_valid is False
        assert "Insufficient price data" in error_msg

    def test_validate_ticker_data_insufficient_news(self, mock_price_data):
        """Test validation fails with insufficient news data."""
        news_data = pd.DataFrame({"title": ["News 1", "News 2"]})
        is_valid, error_msg = validate_ticker_data("AAPL", mock_price_data, news_data)
        assert is_valid is False
        assert "Insufficient news data" in error_msg

    def test_validate_ticker_data_empty_price(self, mock_news_data):
        """Test validation fails with empty price data."""
        price_data = pd.DataFrame()
        is_valid, error_msg = validate_ticker_data("AAPL", price_data, mock_news_data)
        assert is_valid is False
        assert "AAPL" in error_msg

    def test_validate_ticker_data_empty_news(self, mock_price_data):
        """Test validation fails with empty news data."""
        news_data = pd.DataFrame()
        is_valid, error_msg = validate_ticker_data("AAPL", mock_price_data, news_data)
        assert is_valid is False
        assert "AAPL" in error_msg

    @pytest.mark.asyncio
    async def test_analyze_ticker_success(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test successful ticker analysis."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data

        result = await analyze_ticker("AAPL", mock_client, mock_engine)

        assert "result" in result
        assert "cache" in result
        assert result["result"]["ticker"] == "AAPL"
        assert "sentiment" in result["result"]
        assert "volatility" in result["result"]
        assert "trend" in result["result"]

    @pytest.mark.asyncio
    async def test_analyze_ticker_invalid_data(self, mock_client, mock_engine):
        """Test analyze_ticker with invalid data raises exception."""
        mock_client.fetch_historical_prices.return_value = pd.DataFrame()
        mock_client.fetch_company_news.return_value = pd.DataFrame()

        with pytest.raises(ValueError):
            await analyze_ticker("INVALID", mock_client, mock_engine)

    @pytest.mark.asyncio
    async def test_analyze_ticker_sentiment_calculation(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test sentiment calculation in analyze_ticker."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data
        mock_engine.analyze_headlines.side_effect = [0.7, 0.3]  # current, historical

        result = await analyze_ticker("AAPL", mock_client, mock_engine)

        assert result["result"]["sentiment"] == 0.7
        assert (
            abs(result["result"]["trend"] - 0.4) < 0.0001
        )  # Use approximate comparison for float

    @pytest.mark.asyncio
    async def test_run_analysis_single_ticker(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test run_analysis with a single ticker."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data

        results, errors, cache = await run_analysis(["AAPL"], mock_client, mock_engine)

        assert len(results) == 1
        assert len(errors) == 0
        assert "AAPL" in cache
        assert results[0]["ticker"] == "AAPL"

    @pytest.mark.asyncio
    async def test_run_analysis_multiple_tickers_parallel(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test run_analysis processes multiple tickers in parallel."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data

        tickers = ["AAPL", "TSLA", "NVDA", "MSFT"]
        results, errors, cache = await run_analysis(tickers, mock_client, mock_engine)

        assert len(results) == 4
        assert len(errors) == 0
        assert len(cache) == 4

        # All tickers should be in results
        result_tickers = [r["ticker"] for r in results]
        assert set(result_tickers) == set(tickers)

    @pytest.mark.asyncio
    async def test_run_analysis_with_errors(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test run_analysis handles errors gracefully."""

        # First ticker succeeds, second fails
        async def mock_fetch_prices(ticker):
            if ticker == "INVALID":
                return pd.DataFrame()
            return mock_price_data

        async def mock_fetch_news(ticker):
            if ticker == "INVALID":
                return pd.DataFrame()
            return mock_news_data

        mock_client.fetch_historical_prices.side_effect = mock_fetch_prices
        mock_client.fetch_company_news.side_effect = mock_fetch_news

        results, errors, cache = await run_analysis(
            ["AAPL", "INVALID"], mock_client, mock_engine
        )

        assert len(results) == 1
        assert len(errors) == 1
        assert "AAPL" in cache
        assert "INVALID" not in cache
        assert "INVALID" in errors[0]

    @pytest.mark.asyncio
    async def test_run_analysis_progress_callback(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test run_analysis calls progress callback."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data

        progress_calls = []

        def progress_callback(current, total, ticker):
            progress_calls.append((current, total, ticker))

        tickers = ["AAPL", "TSLA"]
        await run_analysis(
            tickers, mock_client, mock_engine, progress_callback=progress_callback
        )

        assert len(progress_calls) == 2
        assert progress_calls[0] == (1, 2, "AAPL")
        assert progress_calls[1] == (2, 2, "TSLA")

    @pytest.mark.asyncio
    async def test_run_analysis_empty_ticker_list(self, mock_client, mock_engine):
        """Test run_analysis with empty ticker list."""
        results, errors, cache = await run_analysis([], mock_client, mock_engine)

        assert len(results) == 0
        assert len(errors) == 0
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_run_analysis_all_errors(self, mock_client, mock_engine):
        """Test run_analysis when all tickers fail."""
        mock_client.fetch_historical_prices.return_value = pd.DataFrame()
        mock_client.fetch_company_news.return_value = pd.DataFrame()

        results, errors, cache = await run_analysis(
            ["INVALID1", "INVALID2"], mock_client, mock_engine
        )

        assert len(results) == 0
        assert len(errors) == 2
        assert len(cache) == 0

    @pytest.mark.asyncio
    async def test_analyze_ticker_cache_structure(
        self, mock_client, mock_engine, mock_price_data, mock_news_data
    ):
        """Test that analyze_ticker returns correct cache structure."""
        mock_client.fetch_historical_prices.return_value = mock_price_data
        mock_client.fetch_company_news.return_value = mock_news_data
        mock_engine.analyze_headlines.side_effect = [0.5, 0.3]

        result = await analyze_ticker("AAPL", mock_client, mock_engine)

        assert "price" in result["cache"]
        assert "news" in result["cache"]
        assert "sent" in result["cache"]
        assert "trend" in result["cache"]
        assert result["cache"]["sent"] == 0.5
        assert result["cache"]["trend"] == 0.2
