"""Senticor Engine - AI-Powered Market Sentiment & Volatility Analysis Platform.

Main entry point for the Streamlit application.
"""

import sys

from ui import run_app

if __name__ == "__main__":
    # Check for --reload-db flag
    reload_db = "--reload-db" in sys.argv
    run_app(reload_db=reload_db)
