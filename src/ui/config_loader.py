"""Configuration and initialization for the Streamlit application."""

import os
import time

import streamlit as st
from dotenv import load_dotenv

from src.config import APP_TITLE
from src.core.analyzer import SentimentEngine
from src.ui.mobile_styles import inject_mobile_styles
from src.utils.logger import AppLogger


def load_api_credentials():
    """Load and set OpenBB API credentials from environment variables."""
    load_dotenv()

    # Lazy import to avoid loading OpenBB at startup
    from openbb import obb

    credentials_mapping = {
        "OBB_POLYGON_API_KEY": "polygon_api_key",
        "OBB_FMP_API_KEY": "fmp_api_key",
        "OBB_BENZINGA_API_KEY": "benzinga_api_key",
        "OBB_ALPHA_VANTAGE_API_KEY": "alpha_vantage_api_key",
        "OBB_INTRINIO_API_KEY": "intrinio_api_key",
    }

    for env_var, credential_name in credentials_mapping.items():
        api_key = os.getenv(env_var)
        if api_key:
            setattr(obb.user.credentials, credential_name, api_key)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    start = time.perf_counter()

    if "engine" not in st.session_state:
        engine_start = time.perf_counter()
        st.session_state.engine = SentimentEngine()
        AppLogger.info(
            "Session init",
            "SentimentEngine initialized in {:.0f} ms".format(
                (time.perf_counter() - engine_start) * 1000
            ),
        )

    if "data" not in st.session_state:
        st.session_state.data = None
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    if "rag_cache" not in st.session_state:
        st.session_state.rag_cache = {}
    if "competitive_cache" not in st.session_state:
        st.session_state.competitive_cache = {}

    AppLogger.info(
        "Session init",
        "Session state ready in {:.0f} ms".format((time.perf_counter() - start) * 1000),
    )


def configure_page():
    """Configure Streamlit page settings and initialize app."""
    step_start = time.perf_counter()
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="auto",  # Auto-collapse on mobile
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": f"# {APP_TITLE}\nAI-Powered Market Sentiment & Volatility Analysis",
        },
    )
    AppLogger.info(
        "Page config",
        "set_page_config completed in {:.0f} ms".format(
            (time.perf_counter() - step_start) * 1000
        ),
    )

    # Inject mobile-responsive CSS
    step_start = time.perf_counter()
    inject_mobile_styles()
    AppLogger.info(
        "Page config",
        "inject_mobile_styles completed in {:.0f} ms".format(
            (time.perf_counter() - step_start) * 1000
        ),
    )

    # Don't load credentials at startup - they'll be loaded when actually needed
    # The MarketDataClient will trigger credential loading on first use

    step_start = time.perf_counter()
    initialize_session_state()
    AppLogger.info(
        "Page config",
        "initialize_session_state completed in {:.0f} ms".format(
            (time.perf_counter() - step_start) * 1000
        ),
    )
