"""Tab components for the main application interface."""

from .deep_dive import render_deep_dive_tab
from .logs import render_logs_tab
from .market_map import render_market_map_tab

__all__ = ["render_market_map_tab", "render_deep_dive_tab", "render_logs_tab"]
