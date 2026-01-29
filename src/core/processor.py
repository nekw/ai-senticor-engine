"""Data processing utilities for financial metrics.

This module provides helper functions for processing and transforming
financial data, including normalization and volatility calculations.
"""

import numpy as np
import pandas as pd


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series to range [0, 1].

    Applies min-max normalization to scale values between 0 and 1,
    useful for visualization and comparison across different metrics.

    Args:
        series: Pandas Series with numerical values to normalize.

    Returns:
        Normalized Series with values between 0 and 1.
        Returns zeros if all values are equal (to avoid division by zero).
        Returns original series if empty.

    Example:
        >>> data = pd.Series([10, 20, 30, 40, 50])
        >>> normalized = normalize_series(data)
        >>> print(normalized.tolist())
        [0.0, 0.25, 0.5, 0.75, 1.0]
    """
    if series.empty:
        return series

    if series.max() == series.min():
        return series * 0

    return (series - series.min()) / (series.max() - series.min())


def calculate_volatility(prices: pd.Series) -> float:
    """Calculate annualized volatility from price series.

    Computes the standard deviation of daily returns and annualizes it
    using the square root of time rule (252 trading days per year).

    Args:
        prices: Pandas Series of closing prices (or any price series).

    Returns:
        Annualized volatility as a decimal (e.g., 0.25 = 25% annual volatility).

    Note:
        - Uses daily returns calculated as percentage change
        - Assumes 252 trading days per year for annualization
        - Higher values indicate more price variability/risk

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105])
        >>> vol = calculate_volatility(prices)
        >>> print(f"Annualized volatility: {vol*100:.2f}%")
        Annualized volatility: 32.45%
    """
    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return 0.0  # Not enough data for meaningful volatility
    volatility = returns.std() * np.sqrt(252)
    return 0.0 if np.isnan(volatility) else volatility
