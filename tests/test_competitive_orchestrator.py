"""Unit tests for the lightweight competitive-analysis orchestrator."""

from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from src.core.agents import CompetitiveAnalysisOrchestrator, CompetitiveQuery


@pytest.fixture
def mock_client():
    """Create mocked market data client."""
    client = Mock()
    client.fetch_company_news = AsyncMock()
    client.fetch_historical_prices = AsyncMock()
    return client


@pytest.fixture
def mock_sentiment_engine():
    """Create mocked sentiment engine."""
    engine = Mock()
    engine.analyze_headlines.return_value = 0.4
    return engine


@pytest.fixture
def mock_rag_engine():
    """Create mocked RAG engine."""
    rag = Mock()
    rag.get_sector.return_value = "Technology"
    rag.get_sector_tickers.return_value = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    rag.get_sector_commentary.return_value = (
        "Competitive narrative",
        [
            {"headline": "Cloud capex accelerates", "ticker": "MSFT"},
            {"headline": "AI inference demand rises", "ticker": "NVDA"},
        ],
    )
    return rag


@pytest.fixture
def mock_news_df():
    """Create consistent mock news frame."""
    return pd.DataFrame(
        {
            "title": [f"Headline {i}" for i in range(20)],
            "date": [f"2026-03-{i:02d}" for i in range(1, 21)],
        }
    )


@pytest.fixture
def mock_price_df():
    """Create consistent mock price frame."""
    return pd.DataFrame(
        {
            "close": [100, 101, 99, 103, 105, 102, 106],
            "open": [99, 100, 100, 102, 104, 103, 105],
        }
    )


@pytest.mark.asyncio
async def test_orchestrator_happy_path(
    mock_client,
    mock_sentiment_engine,
    mock_rag_engine,
    mock_news_df,
    mock_price_df,
):
    """Ensure orchestrator returns complete report payload on success."""
    mock_client.fetch_company_news.return_value = mock_news_df
    mock_client.fetch_historical_prices.return_value = mock_price_df

    orchestrator = CompetitiveAnalysisOrchestrator(
        client=mock_client,
        sentiment_engine=mock_sentiment_engine,
        rag_engine=mock_rag_engine,
    )

    report = await orchestrator.run(
        CompetitiveQuery(focus_ticker="AAPL", max_competitors=4)
    )

    assert report["focus_ticker"] == "AAPL"
    assert report["sector"] == "Technology"
    assert len(report["peer_universe"]) == 4
    assert len(report["signals"]) == 4
    assert report["narrative"] == "Competitive narrative"
    assert report["confidence"] > 0
    assert report["errors"] == []
    assert "AAPL" in report["citations"]


@pytest.mark.asyncio
async def test_orchestrator_handles_missing_price_data(
    mock_client,
    mock_sentiment_engine,
    mock_rag_engine,
    mock_news_df,
):
    """Ensure orchestrator degrades gracefully when price data is missing."""
    mock_client.fetch_company_news.return_value = mock_news_df
    mock_client.fetch_historical_prices.return_value = pd.DataFrame()

    orchestrator = CompetitiveAnalysisOrchestrator(
        client=mock_client,
        sentiment_engine=mock_sentiment_engine,
        rag_engine=mock_rag_engine,
    )

    report = await orchestrator.run(
        CompetitiveQuery(focus_ticker="AAPL", max_competitors=3)
    )

    assert report["signals"] == []
    assert report["confidence"] == 0.0
    assert report["risk_flags"]
    assert any("insufficient price data" in err for err in report["errors"])


@pytest.mark.asyncio
async def test_orchestrator_with_unknown_sector(
    mock_client, mock_sentiment_engine, mock_rag_engine
):
    """Ensure unknown sector does not crash pipeline and records errors."""
    mock_rag_engine.get_sector.return_value = "Unknown"
    mock_client.fetch_company_news.return_value = pd.DataFrame()
    mock_client.fetch_historical_prices.return_value = pd.DataFrame()

    orchestrator = CompetitiveAnalysisOrchestrator(
        client=mock_client,
        sentiment_engine=mock_sentiment_engine,
        rag_engine=mock_rag_engine,
    )

    report = await orchestrator.run(CompetitiveQuery(focus_ticker="ZZZZ"))

    assert report["sector"] == "Unknown"
    assert report["peer_universe"] == ["ZZZZ"]
    assert report["errors"]
    assert report["confidence"] == 0.0
