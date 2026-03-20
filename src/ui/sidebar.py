"""Sidebar UI component for data source configuration and controls."""

import time

import pandas as pd
import streamlit as st

from src.config import (
    ALPHA_SENTIMENT_THRESHOLD,
    ALPHA_VOLATILITY_THRESHOLD,
    AVAILABLE_LLM_PROVIDERS,
    AVAILABLE_NEWS_PROVIDERS,
    AVAILABLE_PRICE_PROVIDERS,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_NEWS_PROVIDER,
    DEFAULT_PRICE_PROVIDER,
    LLM_MODELS,
    SIDEBAR_TITLE,
)
from src.core.data_fetcher import MarketDataClient
from src.ui.rag_mapping import get_effective_tickers
from src.utils.logger import AppLogger


def render_provider_selection() -> tuple[str, str]:
    """Render data provider selection dropdowns.

    Returns:
        Tuple of (price_provider, news_provider).
    """
    with st.sidebar.expander("⚙️ Data Sources", expanded=False):
        price_provider = st.selectbox(
            "Price Provider",
            options=AVAILABLE_PRICE_PROVIDERS,
            index=AVAILABLE_PRICE_PROVIDERS.index(DEFAULT_PRICE_PROVIDER),
        )

        news_provider = st.selectbox(
            "News Provider",
            options=AVAILABLE_NEWS_PROVIDERS,
            index=AVAILABLE_NEWS_PROVIDERS.index(DEFAULT_NEWS_PROVIDER),
        )

    return price_provider, news_provider


def render_llm_selection() -> tuple[str, str, float]:
    """Render LLM provider and model selection.

    Returns:
        Tuple of (llm_provider, model, temperature).
    """
    # Toggle to enable/disable RAG Sector News
    enable_rag = st.sidebar.checkbox(
        "📰 Sector News Analysis",
        value=True,
        help="Enable sector-wide news commentary using RAG and LLM generation",
    )

    if not enable_rag:
        return None, None, DEFAULT_LLM_TEMPERATURE

    with st.sidebar.expander("⚙️ Sector Analysis Settings", expanded=False):
        llm_provider = st.selectbox(
            "LLM Provider",
            options=AVAILABLE_LLM_PROVIDERS,
            index=AVAILABLE_LLM_PROVIDERS.index(DEFAULT_LLM_PROVIDER),
            help=(
                "Select AI model provider for sector analysis. "
                "HuggingFace is free and runs locally."
            ),
        )

        # Map display name to internal provider name
        provider_map = {
            "None (Retrieval Only)": None,
            "OpenAI": "openai",
            "Anthropic": "anthropic",
            "HuggingFace (Free)": "huggingface",
        }

        provider_internal = provider_map[llm_provider]

        # Model selection based on provider
        if provider_internal and provider_internal != "huggingface":
            # Show model dropdown for API providers
            models = LLM_MODELS[llm_provider]
            model = st.selectbox(
                "Model", options=models, help="Select specific model version"
            )
        elif provider_internal == "huggingface":
            # Default to FLAN-T5-base for HuggingFace
            models = LLM_MODELS[llm_provider]
            model = st.selectbox(
                "Model",
                options=models,
                help="Free models from HuggingFace. Larger models = better quality but slower.",
            )
        else:
            model = None

        # Temperature slider (only if LLM is selected)
        if provider_internal:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_LLM_TEMPERATURE,
                step=0.1,
                help="Lower = more focused/deterministic, Higher = more creative/varied",
            )
        else:
            temperature = DEFAULT_LLM_TEMPERATURE

    return provider_internal, model, temperature


def update_market_client(price_provider: str, news_provider: str):
    """Update market data client if providers have changed.

    Args:
        price_provider: Selected price data provider.
        news_provider: Selected news data provider.
    """
    if (
        "client" not in st.session_state
        or st.session_state.get("price_provider") != price_provider
        or st.session_state.get("news_provider") != news_provider
    ):
        st.session_state.client = MarketDataClient(
            price_provider=price_provider, news_provider=news_provider
        )
        st.session_state.price_provider = price_provider
        st.session_state.news_provider = news_provider


def ensure_market_client() -> MarketDataClient:
    """Ensure a market client exists using current provider selections."""
    price_provider = st.session_state.get("price_provider", DEFAULT_PRICE_PROVIDER)
    news_provider = st.session_state.get("news_provider", DEFAULT_NEWS_PROVIDER)
    update_market_client(price_provider, news_provider)
    return st.session_state.client


def _log_sidebar_step(step_name: str, step_start: float, enabled: bool):
    """Emit timing logs for sidebar startup steps."""
    if not enabled:
        return

    elapsed_ms = (time.perf_counter() - step_start) * 1000
    AppLogger.info("Sidebar step", f"{step_name} took {elapsed_ms:.0f} ms")


def render_alpha_flags(data: pd.DataFrame):
    """Render alpha flag indicators in sidebar.

    Args:
        data: DataFrame with analysis results.
    """
    alpha_stocks = data[
        (data["sentiment"] > ALPHA_SENTIMENT_THRESHOLD)
        & (data["volatility"] < ALPHA_VOLATILITY_THRESHOLD)
    ]["ticker"].tolist()

    st.sidebar.subheader("🎯 Alpha Flags")
    if alpha_stocks:
        for stock in alpha_stocks:
            st.sidebar.success(stock)
    else:
        st.sidebar.info("No alpha opportunities found")


def render_sidebar(emit_startup_logs: bool = False) -> str:
    """Render complete sidebar with all controls.

    Note: Sidebar is automatically responsive via CSS - auto-collapses on mobile.

    Returns:
        Comma-separated string of ticker symbols.
    """
    st.sidebar.title(SIDEBAR_TITLE)

    # Provider selection
    step_start = time.perf_counter()
    price_provider, news_provider = render_provider_selection()
    st.session_state.price_provider = price_provider
    st.session_state.news_provider = news_provider
    _log_sidebar_step("provider_selection", step_start, emit_startup_logs)

    # Tickers from mapping (read-only)
    step_start = time.perf_counter()
    st.sidebar.divider()
    mapping_tickers = get_effective_tickers()
    tickers_value = ",".join(mapping_tickers)
    tickers = st.sidebar.text_area(
        "Tickers (from Ticker-Sector Mapping)",
        tickers_value,
        height=120,
        help="Read-only list sourced from the Ticker-Sector Mapping table in the ⚙️ Config tab.",
        disabled=True,
    )
    st.session_state.current_tickers = tickers
    _log_sidebar_step("effective_ticker_mapping", step_start, emit_startup_logs)

    # Run Engine button
    run_clicked = st.sidebar.button("🚀 Run Engine", use_container_width=True)
    st.session_state.run_clicked = run_clicked

    # LLM configuration
    step_start = time.perf_counter()
    llm_provider, model, temperature = render_llm_selection()
    _log_sidebar_step("llm_selection", step_start, emit_startup_logs)

    # Store LLM config in session state
    st.session_state.llm_provider = llm_provider
    st.session_state.llm_model = model
    st.session_state.llm_temperature = temperature

    if run_clicked:
        step_start = time.perf_counter()
        ensure_market_client()
        _log_sidebar_step("market_client_init", step_start, emit_startup_logs)

    return tickers
