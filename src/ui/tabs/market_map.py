"""Market Map tab - Quadrant visualization and market summary."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.charts import create_quadrant_plot
from src.utils.pdf_gen import generate_pdf_report


def render_market_map_tab(data: pd.DataFrame):
    """Render the market map visualization tab.

    Args:
        data: DataFrame with analysis results.
    """
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Senticor Market Map")

        # Controls row: Show Labels toggle and PDF button
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 3])
        with ctrl_col1:
            show_labels = st.toggle("Show Labels", value=True)
        with ctrl_col2:
            _render_pdf_download(data, None)  # Will pass fig after creation

        # Create the quadrant plot
        quadrant_fig = create_quadrant_plot(data, show_labels)
        _render_quadrant_plot(quadrant_fig)

    with col2:
        _render_market_summary(data)


def _render_quadrant_plot(quadrant_fig: go.Figure):
    """Render the quadrant plot visualization.

    Args:
        quadrant_fig: Pre-created Plotly figure.
    """
    st.plotly_chart(quadrant_fig, use_container_width=True)


def _render_pdf_download(data: pd.DataFrame, quadrant_fig: go.Figure = None):
    """Render PDF report download button.

    Args:
        data: DataFrame with analysis results.
        quadrant_fig: Plotly figure to include in PDF (optional, will be created if None).
    """
    # Show generate button
    if st.button(
        "📄 PDF Report",
        help="Create and download comprehensive PDF report with market map visualization",
    ):
        # Create figure if not provided
        if quadrant_fig is None:
            quadrant_fig = create_quadrant_plot(data, show_labels=True)

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_pdf_progress(step, message):
            """Update progress during PDF generation."""
            progress = (step + 1) / 6  # 6 total steps (0-5)
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            pdf = generate_pdf_report(
                data, quadrant_fig, progress_callback=update_pdf_progress
            )

            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("✅ PDF generated successfully!")

            # Show download button
            st.download_button(
                "📥 Download PDF Report",
                data=pdf,
                file_name="senticor_market_report_{}.pdf".format(
                    pd.Timestamp.now().strftime("%Y%m%d_%H%M")
                ),
                mime="application/pdf",
                help="Download your generated report",
            )
        finally:
            # Clean up progress indicators after a short delay
            import time

            time.sleep(1.5)
            progress_bar.empty()
            status_text.empty()


def _render_market_summary(data: pd.DataFrame):
    """Render market summary table.

    Args:
        data: DataFrame with analysis results.
    """
    st.subheader("Market Summary")

    # Prepare data with proper column names and sorting
    summary_data = data[["ticker", "sentiment", "volatility"]].copy()
    summary_data.columns = ["Ticker", "Sentiment", "Volatility"]

    # Default sort by sentiment desc, then volatility desc
    summary_data = summary_data.sort_values(
        by=["Sentiment", "Volatility"], ascending=[False, False]
    )

    st.dataframe(summary_data, hide_index=True, height=600, use_container_width=True)
