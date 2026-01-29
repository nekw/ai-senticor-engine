"""PDF report generation utilities.

This module provides functionality to generate professional PDF reports
from market analysis results for client presentations and documentation.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF


def generate_pdf_report(
    df: pd.DataFrame, quadrant_fig: go.Figure = None, progress_callback=None
) -> bytes:
    """Generate a professional PDF report from market analysis data.

    Creates a formatted PDF document containing:
    - Header with branding and title
    - Senticor Market Map visualization (if provided)
    - Summary table with ticker, sentiment, and volatility metrics
    - Investment disclaimer

    Args:
        df: DataFrame with columns 'ticker', 'sentiment', and 'volatility'.
        quadrant_fig: Optional Plotly figure of the quadrant market map.
        progress_callback: Optional callback function(step, message) for progress updates.

    Returns:
        PDF file content as bytes, ready for download or file writing.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'ticker': ['AAPL', 'TSLA'],
        ...     'sentiment': [0.65, -0.25],
        ...     'volatility': [0.22, 0.48]
        ... })
        >>> pdf_bytes = generate_pdf_report(data)
        >>> with open('report.pdf', 'wb') as f:
        ...     f.write(pdf_bytes)

    Note:
        The PDF is optimized for A4 paper size and uses Arial font family.
        Volatility is displayed as a percentage for readability.
    """
    if progress_callback:
        progress_callback(0, "Initializing PDF...")

    pdf = FPDF()
    pdf.add_page()

    # --- Header ---
    if progress_callback:
        progress_callback(1, "Creating header...")

    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(40, 70, 120)  # Deep Blue
    pdf.cell(0, 15, "AI Senticor Engine: Intelligence Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100)
    pdf.cell(0, 5, "Quantitative Sentiment & Volatility Analysis", ln=True, align="C")

    # Date timestamp
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(120)
    pdf.cell(
        0,
        5,
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ln=True,
        align="C",
    )
    pdf.ln(10)

    # --- Market Map Visualization ---
    if progress_callback:
        progress_callback(2, "Adding market map visualization...")

    if quadrant_fig is not None:
        pdf.set_font("Arial", "B", 14)
        pdf.set_text_color(0)
        pdf.cell(0, 8, "Senticor Market Map", ln=True, align="L")
        pdf.ln(2)

        # Convert Plotly figure to image bytes and save to temporary file
        img_bytes = quadrant_fig.to_image(format="png", width=800, height=500, scale=2)

        # Create temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_file.write(img_bytes)
            tmp_path = tmp_file.name

        try:
            # Add image to PDF
            pdf.image(tmp_path, x=10, w=190)  # Full page width (210mm - 20mm margins)
            pdf.ln(5)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    pdf.ln(5)

    # --- Summary Table Header ---
    if progress_callback:
        progress_callback(3, "Generating data table...")

    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0)
    pdf.cell(0, 8, "Analysis Summary", ln=True, align="L")
    pdf.ln(3)

    pdf.set_fill_color(40, 70, 120)
    pdf.set_text_color(255)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(40, 10, "Ticker", 1, 0, "C", True)
    pdf.cell(75, 10, "Sentiment Score", 1, 0, "C", True)
    pdf.cell(75, 10, "Ann. Volatility", 1, 1, "C", True)

    # --- Table Content ---
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0)

    # Sort by sentiment descending
    df_sorted = df.sort_values("sentiment", ascending=False)

    for idx, row in enumerate(df_sorted.itertuples()):
        # Alternate row colors
        if idx % 2 == 0:
            pdf.set_fill_color(250, 250, 250)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.cell(40, 8, str(row.ticker), 1, 0, "C", True)

        # Color-code sentiment
        sentiment = row.sentiment
        if sentiment > 0.3:
            pdf.set_text_color(0, 128, 0)  # Green
        elif sentiment < -0.3:
            pdf.set_text_color(255, 0, 0)  # Red
        else:
            pdf.set_text_color(0, 0, 0)  # Black

        pdf.cell(75, 8, f"{sentiment:.2f}", 1, 0, "C", True)

        pdf.set_text_color(0)
        pdf.cell(75, 8, f"{row.volatility*100:.2f}%", 1, 1, "C", True)

    pdf.ln(8)

    # --- Key Insights ---
    if progress_callback:
        progress_callback(4, "Calculating key insights...")

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0)
    pdf.cell(0, 8, "Key Insights", ln=True, align="L")
    pdf.ln(2)

    pdf.set_font("Arial", "", 9)

    # Calculate insights
    avg_sentiment = df["sentiment"].mean()
    avg_volatility = df["volatility"].mean()
    positive_count = len(df[df["sentiment"] > 0])
    negative_count = len(df[df["sentiment"] < 0])

    insights_text = f"""
* Average Sentiment: {avg_sentiment:.2f} ({_sentiment_label(avg_sentiment)})
* Average Volatility: {avg_volatility*100:.1f}%
* Tickers with Positive Sentiment: {positive_count} of {len(df)}
* Tickers with Negative Sentiment: {negative_count} of {len(df)}
* Market Tone: {_market_tone(avg_sentiment, positive_count, len(df))}
    """

    pdf.multi_cell(0, 5, insights_text.strip())

    pdf.ln(8)

    # --- Investment Disclaimer ---
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100)
    pdf.multi_cell(
        0,
        4,
        "Disclaimer: This report is generated by an AI-powered sentiment analysis system. "
        "Information is for educational and informational purposes only and does not constitute "
        "financial, investment, or trading advice. Always conduct your own research and consult "
        "with a qualified financial advisor before making investment decisions.",
    )

    # Handle both old (string) and new (bytes) FPDF versions
    if progress_callback:
        progress_callback(5, "Finalizing PDF...")

    output = pdf.output(dest="S")
    if isinstance(output, bytes):
        return output
    else:
        return output.encode("latin-1")


def _sentiment_label(sentiment: float) -> str:
    """Return sentiment label based on score.

    Args:
        sentiment: Sentiment score from -1 to 1.

    Returns:
        Human-readable sentiment label.
    """
    if sentiment > 0.5:
        return "Very Positive"
    elif sentiment > 0.2:
        return "Positive"
    elif sentiment > -0.2:
        return "Neutral"
    elif sentiment > -0.5:
        return "Negative"
    else:
        return "Very Negative"


def _market_tone(avg_sentiment: float, positive_count: int, total_count: int) -> str:
    """Determine overall market tone.

    Args:
        avg_sentiment: Average sentiment score.
        positive_count: Number of tickers with positive sentiment.
        total_count: Total number of tickers.

    Returns:
        Market tone description.
    """
    positive_ratio = positive_count / total_count if total_count > 0 else 0

    if avg_sentiment > 0.3 and positive_ratio > 0.7:
        return "Bullish - Strong positive momentum"
    elif avg_sentiment > 0.1 and positive_ratio > 0.5:
        return "Moderately Bullish - Cautiously optimistic"
    elif avg_sentiment < -0.3 and positive_ratio < 0.3:
        return "Bearish - Significant negative pressure"
    elif avg_sentiment < -0.1 and positive_ratio < 0.5:
        return "Moderately Bearish - Defensive positioning recommended"
    else:
        return "Mixed - No clear directional bias"
