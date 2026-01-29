"""Unit tests for data processing utilities."""

import numpy as np
import pandas as pd

from src.core.processor import calculate_volatility, normalize_series


class TestNormalizeSeries:
    """Test suite for normalize_series function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        s = pd.Series([10, 20, 30])
        result = normalize_series(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.5
        assert result.iloc[2] == 1.0

    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        s = pd.Series([-10, 0, 10])
        result = normalize_series(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 0.5
        assert result.iloc[2] == 1.0

    def test_normalize_single_value(self):
        """Test normalization with single value."""
        s = pd.Series([5])
        result = normalize_series(s)
        assert result.iloc[0] == 0.0  # Single value normalizes to 0

    def test_normalize_identical_values(self):
        """Test normalization when all values are identical."""
        s = pd.Series([5, 5, 5, 5])
        result = normalize_series(s)
        assert all(result == 0.0)  # All zeros to avoid division by zero

    def test_normalize_empty_series(self):
        """Test normalization with empty series."""
        s = pd.Series([], dtype=float)
        result = normalize_series(s)
        assert result.empty
        assert len(result) == 0

    def test_normalize_decimal_values(self):
        """Test normalization with decimal values."""
        s = pd.Series([0.1, 0.5, 0.9])
        result = normalize_series(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[2] == 1.0
        assert 0.4 < result.iloc[1] < 0.6  # Approximate middle

    def test_normalize_preserves_order(self):
        """Test that normalization preserves relative order."""
        s = pd.Series([50, 10, 30, 20, 40])
        result = normalize_series(s)
        assert result.iloc[0] == 1.0  # Max
        assert result.iloc[1] == 0.0  # Min
        assert result.iloc[2] == 0.5  # Middle
        assert result.iloc[3] < result.iloc[2]  # Relative order preserved
        assert result.iloc[4] > result.iloc[2]


class TestCalculateVolatility:
    """Test suite for calculate_volatility function."""

    def test_volatility_constant_prices(self):
        """Test volatility with constant prices is zero."""
        prices = pd.Series([100, 100, 100, 100, 100])
        result = calculate_volatility(prices)
        assert result == 0.0

    def test_volatility_positive_prices(self):
        """Test volatility calculation with normal price series."""
        prices = pd.Series([100, 102, 101, 103, 105])
        result = calculate_volatility(prices)
        assert result > 0  # Volatility should be positive
        assert isinstance(result, float)

    def test_volatility_high_variance(self):
        """Test higher variance produces higher volatility."""
        low_variance = pd.Series([100, 101, 100, 101, 100])
        high_variance = pd.Series([100, 110, 90, 115, 85])

        vol_low = calculate_volatility(low_variance)
        vol_high = calculate_volatility(high_variance)

        assert vol_high > vol_low

    def test_volatility_trending_up(self):
        """Test volatility with upward trending prices."""
        prices = pd.Series([100, 105, 110, 115, 120])
        result = calculate_volatility(prices)
        assert result > 0

    def test_volatility_trending_down(self):
        """Test volatility with downward trending prices."""
        prices = pd.Series([120, 115, 110, 105, 100])
        result = calculate_volatility(prices)
        assert result > 0

    def test_volatility_annualization_factor(self):
        """Test that volatility uses correct annualization factor."""
        # Create series with known daily volatility
        prices = pd.Series([100, 102, 101, 103, 102])
        result = calculate_volatility(prices)

        # Calculate expected value
        returns = prices.pct_change().dropna()
        expected = returns.std() * np.sqrt(252)

        assert abs(result - expected) < 1e-10

    def test_volatility_with_two_prices(self):
        """Test volatility calculation with minimum data points."""
        prices = pd.Series([100, 110])
        result = calculate_volatility(prices)
        # With only 2 prices, we get 1 return, which is not enough for meaningful std
        assert result == 0.0  # Should return 0 for insufficient data

    def test_volatility_large_dataset(self):
        """Test volatility with larger dataset."""
        np.random.seed(42)
        # Generate 100 price points with some noise
        prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.5))
        result = calculate_volatility(prices)
        assert result > 0
        assert result < 10  # Should be reasonable

    def test_volatility_realistic_values(self):
        """Test volatility produces realistic values for stocks."""
        # Typical stock with 1% daily moves
        prices = pd.Series([100, 101, 100.5, 102, 101.5])
        result = calculate_volatility(prices)
        # Annual volatility for 1% daily moves should be around 0.15-0.20
        assert 0.05 < result < 0.50
