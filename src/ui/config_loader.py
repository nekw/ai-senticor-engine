"""Configuration and initialization for the Streamlit application."""

import os

import streamlit as st
from dotenv import load_dotenv

from src.config import APP_TITLE
from src.core.analyzer import SentimentEngine
from src.ui.mobile_styles import inject_mobile_styles


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


def initialize_rag_database(reload: bool = False):
    """Initialize RAG vector database with sample sector news.

    Args:
        reload: If True, force reload the database even if already initialized.
    """
    if reload:
        try:
            from utils.load_sample_news import load_all_sample_news

            print(f"{'🔄 Reloading' if reload else '🔄 Initializing'} RAG database...")
            load_all_sample_news()
        except Exception as e:
            print(f"Warning: Failed to initialize RAG database: {e}")


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "engine" not in st.session_state:
        st.session_state.engine = SentimentEngine()
    if "data" not in st.session_state:
        st.session_state.data = None
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    if "rag_cache" not in st.session_state:
        st.session_state.rag_cache = {}
    if "competitive_cache" not in st.session_state:
        st.session_state.competitive_cache = {}


def configure_page(reload_db: bool = False):
    """Configure Streamlit page settings and initialize app.

    Args:
        reload_db: If True, reload the RAG vector database with sample news.
    """
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

    # Inject mobile-responsive CSS
    inject_mobile_styles()

    # Don't load credentials at startup - they'll be loaded when actually needed
    # The MarketDataClient will trigger credential loading on first use

    initialize_session_state()

    # Only reload DB if explicitly requested (non-blocking)
    if reload_db:
        initialize_rag_database(reload=reload_db)
