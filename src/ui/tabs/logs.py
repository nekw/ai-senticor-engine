"""Logs tab for monitoring and auditing backend actions."""

import pandas as pd
import streamlit as st

from src.utils.logger import AppLogger


def render_logs_tab():
    """Render the logs monitoring tab."""
    st.header("📋 System Logs")
    st.markdown("Monitor backend actions, API calls, and system events in real-time.")

    # Get all logs
    all_logs = AppLogger.get_logs()

    if not all_logs:
        st.info("No logs available yet. Run the analysis engine to generate logs.")
        return

    # Filter controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        level_filter = st.selectbox(
            "Filter by Level",
            options=["All", "INFO", "SUCCESS", "WARNING", "ERROR"],
            index=0,
        )

    with col2:
        # Get unique tickers from logs
        unique_tickers = sorted(
            set([log["ticker"] for log in all_logs if log["ticker"]])
        )
        ticker_options = ["All"] + unique_tickers
        ticker_filter = st.selectbox(
            "Filter by Ticker", options=ticker_options, index=0
        )

    with col3:
        if st.button("🗑️ Clear Logs", use_container_width=True):
            AppLogger.clear_logs()
            st.rerun()

    # Apply filters
    filtered_logs = all_logs
    if level_filter != "All":
        filtered_logs = [log for log in filtered_logs if log["level"] == level_filter]
    if ticker_filter != "All":
        filtered_logs = [log for log in filtered_logs if log["ticker"] == ticker_filter]

    # Display summary stats
    st.divider()
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        st.metric("Total Logs", len(all_logs))
    with col_stat2:
        success_count = len([log for log in all_logs if log["level"] == "SUCCESS"])
        st.metric("Success", success_count)
    with col_stat3:
        warning_count = len([log for log in all_logs if log["level"] == "WARNING"])
        st.metric("Warnings", warning_count)
    with col_stat4:
        error_count = len([log for log in all_logs if log["level"] == "ERROR"])
        st.metric("Errors", error_count)

    st.divider()

    # Display logs in reverse chronological order (newest first)
    st.subheader(f"📄 Log Entries ({len(filtered_logs)} entries)")

    if not filtered_logs:
        st.warning("No logs match the current filters.")
        return

    # Display as expandable entries
    for log in reversed(filtered_logs):
        # Color code by level
        level_colors = {"INFO": "🔵", "SUCCESS": "🟢", "WARNING": "🟡", "ERROR": "🔴"}

        icon = level_colors.get(log["level"], "⚪")
        ticker_badge = f"[{log['ticker']}] " if log["ticker"] else ""

        # Create expandable log entry
        with st.expander(
            f"{icon} {log['timestamp']} - {ticker_badge}{log['action']}", expanded=False
        ):
            col_log1, col_log2 = st.columns([1, 3])

            with col_log1:
                st.markdown(f"**Level:** {log['level']}")
                st.markdown(f"**Time:** {log['timestamp']}")
                if log["ticker"]:
                    st.markdown(f"**Ticker:** {log['ticker']}")

            with col_log2:
                st.markdown("**Action:** {}".format(log["action"]))
                if log["details"]:
                    st.markdown("**Details:**")
                    st.code(log["details"], language=None)

    # Export logs option
    st.divider()
    if st.button("📥 Export Logs as CSV"):
        df = pd.DataFrame(filtered_logs)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"senticor_logs_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
