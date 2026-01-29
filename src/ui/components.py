"""Reusable UI components for the Streamlit application."""

import pandas as pd
import streamlit as st

from src.core.analyzer import SentimentEngine
from src.core.processor import normalize_series


def render_trade_advisory(
    sentiment: float,
    volatility: float,
    ticker: str = None,
    data: pd.DataFrame = None,
    show_strategy: bool = True,
    show_sector_commentary: bool = True,
):
    """Render AI trade advisory box with risk-adjusted recommendations.

    Args:
        sentiment: Sentiment score.
        volatility: Raw volatility score.
        ticker: Stock ticker symbol for risk-adjusted analysis.
        data: Optional DataFrame with all tickers to normalize volatility against.
              If not provided, volatility is assumed to already be normalized.
        show_strategy: Whether to display risk-adjusted recommendations.
        show_sector_commentary: Whether to show sector news commentary in the advisory.
    """
    # Lazy import to avoid loading heavy dependencies at startup
    from src.core.rag_engine import RAGEngine

    # Normalize volatility if DataFrame is provided
    if data is not None:
        vol_norm = normalize_series(data["volatility"])
        ticker_idx = data[data["volatility"] == volatility].index[0]
        vol_normalized = vol_norm.iloc[ticker_idx]
    else:
        # Assume volatility is already normalized
        vol_normalized = volatility

    rec = SentimentEngine().get_trade_recommendation(sentiment, vol_normalized)

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; border:2px solid {rec['color']};
                    background-color: rgba(0,0,0,0.05)">
            <h3 style="color:{rec['color']}; margin-top:0;">
                💡 Market Insight: {rec['action']}
            </h3>
            <p style="font-size:1.1rem;">
                <b>Rationale:</b> {rec['rationale']}
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Add sector news commentary (only if RAG is enabled and show_sector_commentary is True)
    if show_strategy and ticker and show_sector_commentary:
        # Check if RAG is enabled
        llm_provider = st.session_state.get("llm_provider", None)

        # Only show if user has enabled RAG (llm_provider is not None means RAG is enabled)
        if llm_provider is not None:
            with st.expander(
                "📰 Sector News Commentary (RAG-Based Sector Analysis)", expanded=False
            ):
                # Check cache first
                cache_key = f"rag_{ticker}_{sentiment:.2f}"

                # Initialize RAG cache if not exists
                if "rag_cache" not in st.session_state:
                    st.session_state.rag_cache = {}

                # Use cached result if available
                if cache_key in st.session_state.rag_cache:
                    commentary = st.session_state.rag_cache[cache_key]
                    st.markdown(commentary, unsafe_allow_html=True)
                    st.caption("📦 *Cached result - refresh page to regenerate*")
                else:
                    with st.spinner("Analyzing sector news and trends..."):
                        try:
                            # Get LLM config from session state
                            llm_model = st.session_state.get("llm_model", None)
                            llm_temperature = st.session_state.get(
                                "llm_temperature", 0.3
                            )

                            # Initialize RAG with configured LLM
                            rag = RAGEngine(
                                llm_provider=llm_provider,
                                model=llm_model,
                                temperature=llm_temperature,
                            )

                            # Get sector commentary
                            commentary = rag.get_sector_commentary(
                                ticker=ticker, company_sentiment=sentiment, k=10
                            )

                            # Cache the result
                            st.session_state.rag_cache[cache_key] = commentary
                            st.markdown(commentary, unsafe_allow_html=True)

                        except Exception as e:
                            st.warning(f"Sector analysis unavailable: {str(e)}")
                            st.info(
                                "Add sector news using `rag.add_sector_news()` "
                                "for sector commentary."
                            )


def render_news_feed(news_data):
    """Render news articles feed.

    Args:
        news_data: DataFrame with news articles.
    """
    for _, article in news_data.iterrows():
        st.write(f"**{article['title']}**")
        st.caption(f"{article.get('date', '')}")
        st.write(f"{article.get('text', '')}")
        st.divider()
