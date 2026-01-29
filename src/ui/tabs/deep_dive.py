"""Company Intelligence tab - Detailed ticker analysis with charts and news."""

import pandas as pd
import streamlit as st

from src.ui.components import render_news_feed, render_trade_advisory
from src.utils.charts import create_price_chart


def render_deep_dive_tab(data: pd.DataFrame, cache: dict):
    """Render the company intelligence analysis tab.

    Args:
        data: DataFrame with analysis results.
        cache: Cached data for each ticker.
    """
    st.header("🔍 Company Intelligence")

    if data is None or data.empty:
        st.info("⚠️ No analysis data available.")
        st.markdown(
            """
        **To view Company Intelligence:**
        1. Enter tickers in the sidebar
        2. Click the **🚀 Run Engine** button
        3. Return to this tab to see detailed analysis
        """
        )
        return

    selected_ticker = st.selectbox("Ticker", data["ticker"])
    ticker_data = cache[selected_ticker]

    # Get ticker metrics
    row = data[data["ticker"] == selected_ticker].iloc[0]

    # Company Position - Trade advisory with risk-adjusted recommendations
    st.subheader("📊 Company Position")

    # Show company sentiment score prominently
    col_sent1, col_sent2, col_sent3 = st.columns(3)
    with col_sent1:
        sentiment_trend = ticker_data.get("trend", 0)
        st.metric(
            "Company Sentiment",
            f"{row['sentiment']:.2f}",
            delta=f"{sentiment_trend:.2f}",
        )
    with col_sent2:
        st.metric("Volatility", f"{row['volatility']:.2f}")
    with col_sent3:
        # Sentiment label based on score
        sentiment_label = (
            "🟢 Positive"
            if row["sentiment"] > 0.3
            else "🔴 Negative"
            if row["sentiment"] < -0.3
            else "🟡 Neutral"
        )
        st.metric("Sentiment Type", sentiment_label)

    render_trade_advisory(
        row["sentiment"],
        row["volatility"],
        ticker=selected_ticker,
        data=data,
        show_sector_commentary=False,
    )

    # Sector News Commentary (if RAG is enabled)
    _render_sector_commentary(selected_ticker, row["sentiment"])

    # Technical Analysis and Recent News side by side
    st.subheader("📈 Technical Analysis & Recent News")

    col1, col2 = st.columns([3, 2])

    with col1:
        _render_price_chart(selected_ticker, ticker_data["price"])

    with col2:
        st.markdown("### 📰 Recent News")
        render_news_feed(ticker_data["news"])


def _render_sentiment_metric(ticker_data: dict):
    """Render sentiment metric with trend.

    Args:
        ticker_data: Cached data for the ticker.
    """
    st.metric(
        "Sentiment", f"{ticker_data['sent']:.2f}", delta=f"{ticker_data['trend']:.2f}"
    )


def _render_price_chart(ticker: str, price_data: pd.DataFrame):
    """Render price chart.

    Args:
        ticker: Stock ticker symbol.
        price_data: Historical price DataFrame.
    """
    st.plotly_chart(create_price_chart(ticker, price_data))


def _render_sector_commentary(ticker: str, sentiment: float):
    """Render sector commentary section if RAG is enabled.

    Args:
        ticker: Stock ticker symbol.
        sentiment: Sentiment score.
    """
    # Check if RAG is enabled
    llm_provider = st.session_state.get("llm_provider", None)

    # Only show if user has enabled RAG
    if llm_provider is not None:
        with st.expander("📊 Sector Intelligence & News Analysis", expanded=False):
            # Check cache first
            cache_key = f"rag_{ticker}_{sentiment:.2f}"
            cache_key_sources = f"rag_sources_{ticker}_{sentiment:.2f}"

            # Initialize RAG cache if not exists
            if "rag_cache" not in st.session_state:
                st.session_state.rag_cache = {}

            # Use cached result if available
            if cache_key in st.session_state.rag_cache:
                commentary = st.session_state.rag_cache[cache_key]
                news_sources = st.session_state.rag_cache.get(cache_key_sources, [])
                st.markdown(commentary, unsafe_allow_html=True)
                st.caption("📦 *Cached result - refresh page to regenerate*")
            else:
                with st.spinner("Analyzing sector news and trends..."):
                    try:
                        # Lazy import to avoid loading at startup
                        from src.core.rag_engine import RAGEngine

                        # Get LLM config from session state
                        llm_model = st.session_state.get("llm_model", None)
                        llm_temperature = st.session_state.get("llm_temperature", 0.3)

                        # Initialize RAG with configured LLM
                        rag = RAGEngine(
                            llm_provider=llm_provider,
                            model=llm_model,
                            temperature=llm_temperature,
                        )

                        # Get sector commentary with sources
                        result = rag.get_sector_commentary(
                            ticker=ticker,
                            company_sentiment=sentiment,
                            k=10,
                            return_sources=True,
                        )

                        # Unpack tuple result
                        if isinstance(result, tuple):
                            commentary, news_sources = result
                        else:
                            commentary = result
                            news_sources = []

                        # Cache both commentary and sources
                        st.session_state.rag_cache[cache_key] = commentary
                        st.session_state.rag_cache[cache_key_sources] = news_sources

                        st.markdown(commentary, unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"Sector analysis unavailable: {str(e)}")
                        st.info(
                            "Add sector news using `rag.add_sector_news()` for sector commentary."
                        )
                        news_sources = []

            # Show news sources in expandable section
            if news_sources:
                with st.expander(
                    f"📋 View Source News Articles ({len(news_sources)} articles)",
                    expanded=False,
                ):
                    for i, article in enumerate(news_sources, 1):
                        st.caption(
                            "📅 {} | 🏢 {}".format(
                                article["date"],
                                article["ticker"]
                                if article["ticker"]
                                else "Sector-wide",
                            )
                        )
                        st.markdown(article["content"])
                        st.divider()
