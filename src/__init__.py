"""
AI-Senticor-Engine Package.

A professional financial analysis platform combining AI sentiment analysis
with market data for intelligent stock evaluation.
"""

__version__ = "1.0.0"
__author__ = "nekw"
__email__ = "nekw1122@gmail.com"

from src.core.analyzer import SentimentEngine
from src.core.data_fetcher import MarketDataClient
from src.core.processor import calculate_volatility, normalize_series

__all__ = [
    "SentimentEngine",
    "MarketDataClient",
    "calculate_volatility",
    "normalize_series",
]
