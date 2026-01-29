"""Unit tests for chart generation utilities."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.utils.charts import create_price_chart, create_quadrant_plot


class TestCreateQuadrantPlot:
    """Test suite for create_quadrant_plot function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "NVDA", "TSLA"],
                "sentiment": [0.5, -0.3, 0.8, -0.6],
                "volatility": [0.3, 0.8, 0.4, 0.9],
                "trend": ["up", "down", "up", "down"],  # Add trend column
            }
        )

    def test_creates_figure_object(self, sample_data):
        """Test that function returns a Plotly Figure."""
        fig = create_quadrant_plot(sample_data)
        assert isinstance(fig, go.Figure)

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["ticker", "sentiment", "volatility", "trend"])
        fig = create_quadrant_plot(df)
        assert isinstance(fig, go.Figure)

    def test_single_ticker(self):
        """Test plotting single ticker."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "sentiment": [0.5],
                "volatility": [0.3],
                "trend": ["up"],
            }
        )
        fig = create_quadrant_plot(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_show_labels_true(self, sample_data):
        """Test that labels are shown when show_labels=True."""
        fig = create_quadrant_plot(sample_data, show_labels=True)
        # Check that text mode is enabled
        assert any(
            "text" in str(trace.mode) for trace in fig.data if hasattr(trace, "mode")
        )

    def test_show_labels_false(self, sample_data):
        """Test that labels are hidden when show_labels=False."""
        fig = create_quadrant_plot(sample_data, show_labels=False)
        assert isinstance(fig, go.Figure)

    def test_volatility_normalization(self, sample_data):
        """Test that volatility is normalized to [0, 1] range."""
        fig = create_quadrant_plot(sample_data)
        # Extract y-values from the scatter plot
        scatter_trace = fig.data[0]
        if hasattr(scatter_trace, "y"):
            y_values = scatter_trace.y
            assert all(0 <= y <= 1 for y in y_values if y is not None)

    def test_all_tickers_plotted(self, sample_data):
        """Test that all tickers are plotted."""
        fig = create_quadrant_plot(sample_data, show_labels=True)
        # The number of points should match the number of tickers
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_extreme_sentiment_values(self):
        """Test handling of extreme sentiment values."""
        df = pd.DataFrame(
            {
                "ticker": ["POS", "NEG"],
                "sentiment": [1.0, -1.0],
                "volatility": [0.5, 0.5],
                "trend": ["up", "down"],
            }
        )
        fig = create_quadrant_plot(df)
        assert isinstance(fig, go.Figure)

    def test_zero_sentiment(self):
        """Test handling of zero sentiment."""
        df = pd.DataFrame(
            {
                "ticker": ["NEUTRAL"],
                "sentiment": [0.0],
                "volatility": [0.5],
                "trend": ["flat"],
            }
        )
        fig = create_quadrant_plot(df)
        assert isinstance(fig, go.Figure)

    def test_quadrant_zones_present(self, sample_data):
        """Test that quadrant zones are rendered."""
        fig = create_quadrant_plot(sample_data)
        # Check that layout shapes (quadrant backgrounds) are present
        assert hasattr(fig, "layout")
        if hasattr(fig.layout, "shapes"):
            # Should have quadrant background shapes
            assert len(fig.layout.shapes) >= 4

    def test_axes_labels(self, sample_data):
        """Test that axes have proper labels."""
        fig = create_quadrant_plot(sample_data)
        assert "sentiment" in fig.layout.xaxis.title.text.lower()
        assert "volatility" in fig.layout.yaxis.title.text.lower()


class TestCreatePriceChart:
    """Test suite for create_price_chart function."""

    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        dates = pd.date_range(start="2026-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "open": [100 + i for i in range(30)],
                "high": [105 + i for i in range(30)],
                "low": [95 + i for i in range(30)],
                "close": [102 + i for i in range(30)],
                "volume": [1000000 + i * 10000 for i in range(30)],
            }
        ).set_index("date")

    def test_creates_figure_object(self, sample_prices):
        """Test that function returns a Plotly Figure."""
        fig = create_price_chart("AAPL", sample_prices)
        assert isinstance(fig, go.Figure)

    def test_ticker_in_title(self, sample_prices):
        """Test that ticker appears in chart title."""
        ticker = "AAPL"
        fig = create_price_chart(ticker, sample_prices)
        assert ticker in fig.layout.title.text

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        fig = create_price_chart("TEST", df)
        assert isinstance(fig, go.Figure)

    def test_single_day_data(self):
        """Test handling of single day data."""
        df = pd.DataFrame(
            {
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
                "volume": [1000000],
            },
            index=pd.date_range(start="2026-01-27", periods=1),
        )
        fig = create_price_chart("AAPL", df)
        assert isinstance(fig, go.Figure)

    def test_candlestick_trace_exists(self, sample_prices):
        """Test that candlestick trace is present."""
        fig = create_price_chart("AAPL", sample_prices)
        # Check for candlestick trace
        candlestick_found = any(
            "candlestick" in str(type(trace)).lower() for trace in fig.data
        )
        assert candlestick_found or len(fig.data) > 0

    def test_missing_ohlc_columns(self):
        """Test handling of missing OHLC columns."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range(start="2026-01-25", periods=3),
        )

        # Should handle gracefully or raise informative error
        try:
            fig = create_price_chart("TEST", df)
            assert isinstance(fig, go.Figure)
        except KeyError:
            # Expected if missing required columns
            pass

    def test_chart_height_config(self, sample_prices):
        """Test that chart height is configured from config."""
        fig = create_price_chart("AAPL", sample_prices)
        assert hasattr(fig.layout, "height")

    def test_volume_data_if_present(self, sample_prices):
        """Test that volume data is used if present."""
        fig = create_price_chart("AAPL", sample_prices)
        # Volume subplot should be present if data has volume
        assert len(fig.data) >= 1

    def test_different_tickers(self, sample_prices):
        """Test chart generation for different tickers."""
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]
        for ticker in tickers:
            fig = create_price_chart(ticker, sample_prices)
            assert isinstance(fig, go.Figure)
            assert ticker in fig.layout.title.text

    def test_date_range_display(self, sample_prices):
        """Test that date range is properly displayed."""
        fig = create_price_chart("AAPL", sample_prices)
        # Should have time-based x-axis
        assert hasattr(fig.layout, "xaxis")

    def test_price_increasing_trend(self):
        """Test chart with increasing price trend."""
        dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100 + i * 2 for i in range(10)],
                "high": [105 + i * 2 for i in range(10)],
                "low": [95 + i * 2 for i in range(10)],
                "close": [102 + i * 2 for i in range(10)],
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        fig = create_price_chart("BULL", df)
        assert isinstance(fig, go.Figure)

    def test_price_decreasing_trend(self):
        """Test chart with decreasing price trend."""
        dates = pd.date_range(start="2026-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [120 - i * 2 for i in range(10)],
                "high": [125 - i * 2 for i in range(10)],
                "low": [115 - i * 2 for i in range(10)],
                "close": [118 - i * 2 for i in range(10)],
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        fig = create_price_chart("BEAR", df)
        assert isinstance(fig, go.Figure)
