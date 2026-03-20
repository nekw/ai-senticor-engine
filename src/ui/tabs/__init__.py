"""Tab components for the main application interface."""


def render_market_map_tab(*args, **kwargs):
    """Lazy import and render Market Map tab."""
    from .market_map import render_market_map_tab as _render_market_map_tab

    return _render_market_map_tab(*args, **kwargs)


def render_deep_dive_tab(*args, **kwargs):
    """Lazy import and render Company Intelligence tab."""
    from .deep_dive import render_deep_dive_tab as _render_deep_dive_tab

    return _render_deep_dive_tab(*args, **kwargs)


def render_competitive_tab(*args, **kwargs):
    """Lazy import and render Competitive Analysis tab."""
    from .competitive import render_competitive_tab as _render_competitive_tab

    return _render_competitive_tab(*args, **kwargs)


def render_sector_news_tab(*args, **kwargs):
    """Lazy import and render Sector News tab."""
    from .sector_news import render_sector_news_tab as _render_sector_news_tab

    return _render_sector_news_tab(*args, **kwargs)


def render_config_tab(*args, **kwargs):
    """Lazy import and render Config tab."""
    from .config import render_config_tab as _render_config_tab

    return _render_config_tab(*args, **kwargs)


def render_logs_tab(*args, **kwargs):
    """Lazy import and render Logs tab."""
    from .logs import render_logs_tab as _render_logs_tab

    return _render_logs_tab(*args, **kwargs)


__all__ = [
    "render_market_map_tab",
    "render_deep_dive_tab",
    "render_competitive_tab",
    "render_sector_news_tab",
    "render_config_tab",
    "render_logs_tab",
]
