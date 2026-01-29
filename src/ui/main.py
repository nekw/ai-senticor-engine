"""Main Streamlit application entry point."""

import asyncio

import pandas as pd
import streamlit as st

from src.config import ALPHA_SENTIMENT_THRESHOLD, ALPHA_VOLATILITY_THRESHOLD
from src.ui.analysis_engine import run_analysis
from src.ui.config_loader import configure_page
from src.ui.sidebar import render_alpha_flags, render_sidebar
from src.ui.tabs import render_deep_dive_tab, render_logs_tab, render_market_map_tab
from src.utils.logger import AppLogger


def run_app(reload_db: bool = False):
    """Run the main Streamlit application.

    Args:
        reload_db: If True, reload the RAG vector database with sample news.
    """
    # Initialize application
    configure_page(reload_db=reload_db)

    # Render sidebar and get ticker input
    tickers = render_sidebar()

    # Display alpha flags immediately after sidebar (before heavy tab rendering)
    if st.session_state.data is not None:
        render_alpha_flags(st.session_state.data)

    # Check if run button was clicked
    if st.session_state.get("run_clicked", False):
        _execute_analysis(tickers)
        st.rerun()  # Force rerun to display results and alpha flags

    # Always show tabs (Home tab visible before running engine)
    _render_tabs()


def _execute_analysis(tickers: str):
    """Execute analysis for the given tickers.

    Args:
        tickers: Comma-separated string of ticker symbols.
    """
    AppLogger.info("Analysis started", f"Tickers: {tickers}")

    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    # Create progress bar and status text
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(current, total, ticker):
        """Update progress bar and status text."""
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"✓ {ticker} complete | Progress: {current}/{total} tickers")

    try:
        results, errors, cache = asyncio.run(
            run_analysis(
                ticker_list,
                st.session_state.client,
                st.session_state.engine,
                progress_callback=update_progress,
            )
        )

        # Complete progress
        progress_bar.progress(1.0)
        status_text.text(
            f"✅ All analysis complete! Successfully processed {len(ticker_list)} ticker(s)"
        )

        # Store results and cache
        if results:
            st.session_state.data = pd.DataFrame(results)
            st.session_state.cache = cache
            st.session_state.just_completed = True  # Flag for auto-switching
            AppLogger.success(
                "Analysis completed", f"{len(results)} tickers analyzed successfully"
            )
        else:
            AppLogger.error("Analysis failed", "No results generated")

        # Display errors
        if errors:
            st.sidebar.error("\n".join(["⚠️ Errors:"] + errors))
            for error in errors:
                AppLogger.error("Analysis error", error)
    finally:
        # Clean up progress indicators after a short delay
        import time

        time.sleep(1)
        progress_bar.empty()
        status_text.empty()


def _render_tabs():
    """Render all application tabs."""
    tab_home, tab1, tab2, tab3 = st.tabs(
        ["🏠 Home", "📊 Market Intelligence", "🔍 Company Intelligence", "📋 Logs"]
    )

    with tab_home:
        _render_home_tab()

    with tab1:
        if st.session_state.data is not None:
            render_market_map_tab(st.session_state.data)
        else:
            st.info(
                "👈 Enter tickers in the sidebar and click **Run Engine** to see market analysis."
            )

    with tab2:
        render_deep_dive_tab(st.session_state.data, st.session_state.cache)

    with tab3:
        render_logs_tab()


def _render_home_tab():
    """Render the Home tab with app description."""
    # Show success message if analysis just completed
    if st.session_state.get("just_completed", False):
        st.success(
            "✅ Analysis complete! Check the **📊 Market Intelligence** tab to see your results."
        )
        st.session_state.just_completed = False  # Reset flag
        st.info("👆 Click the **Market Intelligence** tab above to view your analysis")

    # Show quick stats if data exists
    if st.session_state.data is not None:
        st.divider()
        st.markdown("### 📊 Current Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tickers", len(st.session_state.data))
        with col2:
            avg_sentiment = st.session_state.data["sentiment"].mean()
            st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        with col3:
            # Alpha signals use normalized volatility and config thresholds
            alpha_count = len(
                st.session_state.data[
                    (st.session_state.data["sentiment"] > ALPHA_SENTIMENT_THRESHOLD)
                    & (st.session_state.data["volatility"] < ALPHA_VOLATILITY_THRESHOLD)
                ]
            )
            st.metric("Alpha Signals", alpha_count)

    st.markdown(
        """
    # 🚀 AI Senticor Engine

    **AI-Powered Stock Market Sentiment Analysis & Sector Intelligence**

    ---
    """
    )

    st.markdown(
        """

    ## 📋 Features

    - **FinBERT Sentiment Analysis**: AI-powered news sentiment scoring
    - **RAG Sector Insights**: LLM-generated sector commentary (OpenAI/Anthropic/HuggingFace)
    - **Market Intelligence Map**: Visual positioning across sentiment vs volatility
    - **Alpha Detection**: Automated identification of high-sentiment, low-volatility opportunities
    - **PDF Reports**: Downloadable professional reports with market map visualization

    ---

    ## ⚠️ Disclaimer

    **FOR EDUCATIONAL AND INFORMATIONAL PURPOSES ONLY**

    This application provides AI-generated sentiment analysis and market
    insights based on news data and machine learning models.

    - **Not Financial Advice**: This tool does NOT provide investment advice,
      recommendations, or guidance for buying/selling securities.
    - **No Guarantee of Accuracy**: Sentiment scores, volatility metrics, and
      LLM-generated commentary may contain errors or inaccuracies.
    - **Market Risk**: Past performance and sentiment analysis do not guarantee
      future results. All investments carry risk.
    - **Do Your Own Research**: Always conduct thorough research and consult
      with qualified financial advisors before making investment decisions.
    - **No Liability**: The creators and contributors of this software assume
      no liability for financial losses or investment decisions made using
      this tool.

    By using this application, you acknowledge that you understand these risks
    and limitations.
    """
    )


if __name__ == "__main__":
    run_app()
