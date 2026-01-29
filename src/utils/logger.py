"""Logging utility for monitoring and auditing backend actions."""

from datetime import datetime
from typing import Optional

import streamlit as st


class AppLogger:
    """Logger for tracking application events and backend actions."""

    @staticmethod
    def log(
        action: str,
        details: Optional[str] = None,
        level: str = "INFO",
        ticker: Optional[str] = None,
    ):
        """Log an application event.

        Args:
            action: Action being performed (e.g., "Fetching price data", "RAG analysis")
            details: Additional details about the action
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
            ticker: Related ticker symbol (optional)
        """
        # Initialize logs in session state if not exists
        if "app_logs" not in st.session_state:
            st.session_state.app_logs = []

        # Create log entry
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "action": action,
            "details": details or "",
            "ticker": ticker or "",
        }

        # Add to logs (keep last 500 entries)
        st.session_state.app_logs.append(log_entry)
        if len(st.session_state.app_logs) > 500:
            st.session_state.app_logs = st.session_state.app_logs[-500:]

    @staticmethod
    def info(action: str, details: Optional[str] = None, ticker: Optional[str] = None):
        """Log an info-level event."""
        AppLogger.log(action, details, "INFO", ticker)

    @staticmethod
    def success(
        action: str, details: Optional[str] = None, ticker: Optional[str] = None
    ):
        """Log a success event."""
        AppLogger.log(action, details, "SUCCESS", ticker)

    @staticmethod
    def warning(
        action: str, details: Optional[str] = None, ticker: Optional[str] = None
    ):
        """Log a warning event."""
        AppLogger.log(action, details, "WARNING", ticker)

    @staticmethod
    def error(action: str, details: Optional[str] = None, ticker: Optional[str] = None):
        """Log an error event."""
        AppLogger.log(action, details, "ERROR", ticker)

    @staticmethod
    def get_logs(
        level_filter: Optional[str] = None, ticker_filter: Optional[str] = None
    ):
        """Get all logs with optional filtering.

        Args:
            level_filter: Filter by log level (INFO, WARNING, ERROR, SUCCESS)
            ticker_filter: Filter by ticker symbol

        Returns:
            List of log entries matching filters
        """
        if "app_logs" not in st.session_state:
            return []

        logs = st.session_state.app_logs

        # Apply filters
        if level_filter:
            logs = [log for log in logs if log["level"] == level_filter]

        if ticker_filter:
            logs = [log for log in logs if log["ticker"] == ticker_filter]

        return logs

    @staticmethod
    def clear_logs():
        """Clear all logs."""
        if "app_logs" in st.session_state:
            st.session_state.app_logs = []
