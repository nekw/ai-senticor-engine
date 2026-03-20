"""Main Streamlit application entry point."""

import asyncio
import time

import pandas as pd
import streamlit as st

from src.config import ALPHA_SENTIMENT_THRESHOLD, ALPHA_VOLATILITY_THRESHOLD
from src.ui.analysis_engine import run_analysis
from src.ui.config_loader import configure_page
from src.ui.sidebar import ensure_market_client, render_alpha_flags, render_sidebar
from src.ui.tabs.competitive import render_competitive_tab
from src.ui.tabs.config import render_config_tab
from src.ui.tabs.deep_dive import render_deep_dive_tab
from src.ui.tabs.logs import render_logs_tab
from src.ui.tabs.market_map import render_market_map_tab
from src.ui.tabs.sector_news import render_sector_news_tab
from src.utils.logger import AppLogger


def run_app():
    """Run the main Streamlit application."""
    is_initial_load = not st.session_state.get("app_bootstrapped", False)
    run_start = time.perf_counter()

    if is_initial_load:
        AppLogger.info("App startup", "Initial load started")

    # Initialize application
    step_start = time.perf_counter()
    configure_page()
    _log_step_duration("Startup step", "configure_page", step_start, is_initial_load)

    # Render sidebar and get ticker input
    step_start = time.perf_counter()
    tickers = render_sidebar(emit_startup_logs=is_initial_load)
    _log_step_duration("Startup step", "render_sidebar", step_start, is_initial_load)

    # Display alpha flags immediately after sidebar (before heavy tab rendering)
    if st.session_state.data is not None:
        step_start = time.perf_counter()
        render_alpha_flags(st.session_state.data)
        _log_step_duration(
            "Startup step", "render_alpha_flags", step_start, is_initial_load
        )

    # Check if run button was clicked
    if st.session_state.get("run_clicked", False):
        _execute_analysis(tickers)
        st.rerun()  # Force rerun to display results and alpha flags

    # Always show tabs (Home tab visible before running engine)
    step_start = time.perf_counter()
    _render_tabs(emit_logs=is_initial_load)
    _log_step_duration("Startup step", "render_tabs", step_start, is_initial_load)

    if is_initial_load:
        AppLogger.info(
            "App startup",
            "Initial load complete in {:.0f} ms".format(
                (time.perf_counter() - run_start) * 1000
            ),
        )
        st.session_state.app_bootstrapped = True


def _log_step_duration(
    action: str, step_name: str, step_start: float, enabled: bool = True
):
    """Log elapsed time for a startup step."""
    if not enabled:
        return

    elapsed_ms = (time.perf_counter() - step_start) * 1000
    AppLogger.info(action, f"{step_name} took {elapsed_ms:.0f} ms")


def _execute_analysis(tickers: str):
    """Execute analysis for the given tickers.

    Args:
        tickers: Comma-separated string of ticker symbols.
    """
    AppLogger.info("Analysis started", f"Tickers: {tickers}")

    # Lazily create market client only when analysis is requested.
    ensure_market_client()

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


def _render_tabs(emit_logs: bool = True):
    """Render all application tabs."""
    tabs_start = time.perf_counter()
    tab_home, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏠 Home",
            "📊 Market Intelligence",
            "🔍 Company Intelligence",
            "🏁 Competitive Analysis",
            "📰 Sector News",
            "⚙️ Config",
            "📋 Logs",
        ]
    )
    _log_step_duration("Tab render", "create_tabs", tabs_start, emit_logs)

    tab_start = time.perf_counter()
    with tab_home:
        _render_home_tab()
    _log_step_duration("Tab render", "home", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab1:
        if st.session_state.data is not None:
            render_market_map_tab(st.session_state.data)
        else:
            st.info(
                "👈 Enter tickers in the sidebar and click **Run Engine** to see market analysis."
            )
    _log_step_duration("Tab render", "market_intelligence", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab2:
        render_deep_dive_tab(st.session_state.data, st.session_state.cache)
    _log_step_duration("Tab render", "company_intelligence", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab3:
        render_competitive_tab(st.session_state.data)
    _log_step_duration("Tab render", "competitive_analysis", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab4:
        render_sector_news_tab()
    _log_step_duration("Tab render", "sector_news", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab5:
        render_config_tab()
    _log_step_duration("Tab render", "config", tab_start, emit_logs)

    tab_start = time.perf_counter()
    with tab6:
        render_logs_tab()
    _log_step_duration("Tab render", "logs", tab_start, emit_logs)


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

        # Columns will stack on mobile via CSS
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
