"""Configuration settings for AI-Senticor-Engine.

This module centralizes all configuration constants and settings for the application,
making it easy to modify behavior without changing core code.
"""

from typing import Final

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_TITLE: Final[str] = "Senticor Engine"
APP_LAYOUT: Final[str] = "wide"
APP_ICON: Final[str] = "🛠️"


# ============================================================================
# AI MODEL SETTINGS
# ============================================================================

SENTIMENT_MODEL: Final[str] = "ProsusAI/finbert"
"""FinBERT model for financial sentiment analysis."""

SENTIMENT_SCORE_MAP: Final[dict[str, int]] = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}
"""Mapping from model labels to numerical scores."""


# ============================================================================
# DATA PROVIDER SETTINGS
# ============================================================================

DEFAULT_PRICE_PROVIDER: Final[str] = "yfinance"
"""Default provider for price data queries."""

DEFAULT_NEWS_PROVIDER: Final[str] = "yfinance"
"""Default provider for news data queries."""

AVAILABLE_PRICE_PROVIDERS: Final[list[str]] = [
    "yfinance",
    "polygon",
    "fmp",
    "intrinio",
    "alpha_vantage",
]
"""Available providers for historical price data."""

AVAILABLE_NEWS_PROVIDERS: Final[list[str]] = [
    "yfinance",
    "polygon",
    "benzinga",
    "fmp",
    "intrinio",
    "biztoc",
    "tiingo",
]
"""Available providers for company news data."""

DEFAULT_TICKERS: Final[
    str
] = "AAPL,GOOGL,MSFT,NVDA,MU,SNDK,LITE,V,MA,DIS,SLV,IBIT,ETH,TQQQ,SQQQ"
"""Default ticker symbols for analysis."""


# ============================================================================
# RAG / LLM SETTINGS
# ============================================================================

AVAILABLE_LLM_PROVIDERS: Final[list[str]] = [
    "HuggingFace (Free)",
    "OpenAI",
    "Anthropic",
]
"""Available LLM providers for RAG generation."""

DEFAULT_LLM_PROVIDER: Final[str] = "HuggingFace (Free)"
"""Default LLM provider for strategy generation."""

LLM_MODELS: Final[dict[str, list[str]]] = {
    "HuggingFace (Free)": [
        "google/flan-t5-base",
        "google/flan-t5-large",
        "microsoft/phi-2",
    ],
    "OpenAI": ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
}
"""Available models for each LLM provider."""

DEFAULT_LLM_TEMPERATURE: Final[float] = 0.3
"""Default temperature for LLM generation (0-1)."""


# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

CURRENT_SENTIMENT_HEADLINES: Final[int] = 5
"""Number of recent headlines to use for current sentiment."""

HISTORICAL_SENTIMENT_HEADLINES: Final[int] = 15
"""Number of historical headlines for baseline sentiment."""

TRADING_DAYS_PER_YEAR: Final[int] = 252
"""Number of trading days used for annualized volatility calculation."""


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

CHART_HEIGHT: Final[int] = 600
"""Default height for Plotly charts in pixels."""

MARKER_SIZE: Final[int] = 15
"""Size of markers in scatter plots."""

MARKER_SYMBOL: Final[str] = "diamond"
"""Symbol shape for markers in plots."""


# Quadrant zone definitions
QUADRANT_ZONES: Final[list[tuple]] = [
    (
        0,
        1,
        0,
        0.5,
        "rgba(0,255,0,0.1)",
        "ALPHA ZONE",
        "Strong Buy Opportunity",
    ),  # Green - Positive + Low Vol
    (
        0,
        1,
        0.5,
        1,
        "rgba(0,0,255,0.1)",
        "HYPE ZONE",
        "Overbought Risk",
    ),  # Blue - Positive + High Vol
    (
        -1,
        0,
        0.5,
        1,
        "rgba(255,0,0,0.1)",
        "DANGER ZONE",
        "High Risk Sell",
    ),  # Red - Negative + High Vol
    (
        -1,
        0,
        0,
        0.5,
        "rgba(128,128,128,0.1)",
        "OVERSIGHT",
        "Undervalued Watch",
    ),  # Gray - Negative + Low Vol
]
"""Zone definitions for sentiment-volatility quadrant plot.
Format: (x0, x1, y0, y1, color, label, remark)
"""


# ============================================================================
# ALPHA DETECTION SETTINGS
# ============================================================================

ALPHA_SENTIMENT_THRESHOLD: Final[float] = 0.0
"""Minimum sentiment score to be considered for alpha flags."""

ALPHA_VOLATILITY_THRESHOLD: Final[float] = 0.5
"""Maximum volatility to be considered for alpha flags."""


# ============================================================================
# PDF REPORT SETTINGS
# ============================================================================

PDF_FONT_FAMILY: Final[str] = "Arial"
"""Font family for PDF reports."""

PDF_HEADER_COLOR: Final[tuple[int, int, int]] = (40, 70, 120)
"""RGB color for PDF report header (Deep Blue)."""

PDF_DISCLAIMER: Final[str] = (
    "Disclaimer: This report is generated by an AI-driven agentic system. "
    "Information is for educational purposes only and does not constitute financial advice."
)
"""Legal disclaimer for PDF reports."""


# ============================================================================
# STREAMLIT UI SETTINGS
# ============================================================================

SIDEBAR_TITLE: Final[str] = "🛠️ Senticor Engine"
"""Title displayed in Streamlit sidebar."""

TAB_NAMES: Final[tuple[str, str]] = ("🌐 Map", "🔍 Company Intelligence")
"""Names for main application tabs."""

SHOW_LABELS_DEFAULT: Final[bool] = True
"""Default state for showing labels in quadrant plot."""


# ============================================================================
# CACHING SETTINGS
# ============================================================================

ENABLE_MODEL_CACHE: Final[bool] = True
"""Whether to cache the AI model in memory for faster loading."""

SESSION_STATE_KEYS: Final[list[str]] = ["client", "engine", "data", "cache"]
"""Keys used in Streamlit session state."""
