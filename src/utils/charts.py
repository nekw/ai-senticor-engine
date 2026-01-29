"""Visualization utilities for market intelligence.

This module provides Plotly-based interactive charts for visualizing
sentiment and price data in an intuitive, professional manner.
"""

import pandas as pd
import plotly.graph_objects as go

from src.config import CHART_HEIGHT, MARKER_SIZE, MARKER_SYMBOL, QUADRANT_ZONES
from src.core.processor import normalize_series


def create_quadrant_plot(df: pd.DataFrame, show_labels: bool = True) -> go.Figure:
    """Create an interactive sentiment vs. volatility quadrant plot.

    Visualizes stocks positioned across four strategic zones based on their
    sentiment and volatility metrics:
    - Alpha Zone: Positive sentiment + Low volatility (Green)
    - Hype Zone: Positive sentiment + High volatility (Blue)
    - Danger Zone: Negative sentiment + High volatility (Red)
    - Oversight: Negative sentiment + Low volatility (Gray)

    Args:
        df: DataFrame with columns 'ticker', 'sentiment', and 'volatility'.
        show_labels: If True, display ticker labels on the plot (default state).

    Returns:
        Plotly Figure object ready for display in Streamlit or web apps.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     'ticker': ['AAPL', 'TSLA'],
        ...     'sentiment': [0.5, -0.3],
        ...     'volatility': [0.3, 0.8]
        ... })
        >>> fig = create_quadrant_plot(data)
        >>> fig.show()  # Display interactive chart
    """

    def get_hover_text(row):
        sent = row["sentiment"]
        vol = row["vol_norm"]
        if sent > 0 and vol < 0.5:
            return "<b>ACCUMULATE</b><br>Institutional buying detected."
        elif sent > 0 and vol >= 0.5:
            return "<b>HYPE ALERT</b><br>High retail FOMO. Tighten stops."
        elif sent <= 0 and vol >= 0.5:
            return "<b>DANGER</b><br>Panic selling. Avoid entry."
        else:
            return "<b>OVERSIGHT</b><br>Stagnant interest. Wait for catalyst."

    df["vol_norm"] = normalize_series(df["volatility"])
    df["recommendation"] = df.apply(get_hover_text, axis=1)

    plot_mode = "markers+text" if show_labels else "markers"
    fig = go.Figure()

    # Zones from config
    for x0, x1, y0, y1, color, label, remark in QUADRANT_ZONES:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 + 0.08,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(color="rgba(100,100,100,0.2)", size=20),
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=(y0 + y1) / 2 - 0.08,
            text=f"<i>{remark}</i>",
            showarrow=False,
            font=dict(color="rgba(120,120,120,0.4)", size=12),
        )

    fig.add_trace(
        go.Scatter(
            x=df["sentiment"],
            y=df["vol_norm"],
            mode=plot_mode,
            text=df["ticker"],
            textposition="top center",
            marker=dict(size=MARKER_SIZE, color="black", symbol=MARKER_SYMBOL),
            customdata=df[["recommendation", "volatility", "trend"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                + "Rec: %{customdata[0]}<br>"
                + "Actual Vol: %{customdata[1]:.2%}<br>"
                + "Sent. Trend: %{customdata[2]:.2f}"
                + "<extra></extra>"  # Removes the secondary box with trace name
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=CHART_HEIGHT,
        yaxis=dict(showgrid=False, title="Volatility (Normalized)"),
        xaxis=dict(showgrid=False, title="Sentiment Score"),
    )
    return fig


def create_price_chart(ticker: str, df_price: pd.DataFrame) -> go.Figure:
    """Create an interactive candlestick chart for price action.

    Generates a professional financial candlestick chart showing OHLC
    (Open, High, Low, Close) price data over time.

    Args:
        ticker: Stock ticker symbol for the chart title.
        df_price: DataFrame with index as dates and columns: open, high, low, close.

    Returns:
        Plotly Figure object with candlestick chart.

    Example:
        >>> import pandas as pd
        >>> from datetime import datetime, timedelta
        >>> dates = [datetime.now() - timedelta(days=i) for i in range(5)]
        >>> prices = pd.DataFrame({
        ...     'open': [100, 102, 101, 103, 105],
        ...     'high': [103, 104, 104, 106, 107],
        ...     'low': [99, 101, 100, 102, 104],
        ...     'close': [102, 101, 103, 105, 106]
        ... }, index=dates)
        >>> fig = create_price_chart("AAPL", prices)
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_price.index,
                open=df_price["open"],
                high=df_price["high"],
                low=df_price["low"],
                close=df_price["close"],
            )
        ]
    )

    fig.update_layout(
        title=f"{ticker} Price Action",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
    )

    return fig
