"""Config tab for ticker-sector mapping management."""

import pandas as pd
import streamlit as st

from src.ui.rag_mapping import (
    get_default_sector_mapping,
    get_effective_mapping,
    save_mapping_overrides,
)


def render_config_tab():
    """Render configuration controls for sector intelligence."""
    st.header("⚙️ Configuration")
    st.caption("Manage ticker-to-sector mappings.")

    st.subheader("🏷️ Ticker-Sector Mapping")
    _render_sector_mapping_editor()


def _render_sector_mapping_editor():
    """Render editable ticker-to-sector mapping table."""
    base_mapping = _get_base_sector_mapping()
    effective_mapping = get_effective_mapping()

    mapping_df = pd.DataFrame(
        [
            {"ticker": ticker, "sector": sector}
            for ticker, sector in effective_mapping.items()
        ]
    ).sort_values("ticker")

    edited_df = st.data_editor(
        mapping_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="sector_mapping_editor",
        column_config={
            "ticker": st.column_config.TextColumn(
                "Ticker", help="Stock symbol, e.g. AAPL", max_chars=10
            ),
            "sector": st.column_config.TextColumn(
                "Sector", help="Sector name, e.g. Technology"
            ),
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        save_clicked = st.button("💾 Save Mapping Changes", use_container_width=True)
    with col2:
        reset_clicked = st.button(
            "↩️ Reset To Default Mapping", use_container_width=True
        )

    if save_clicked:
        cleaned = edited_df.dropna(subset=["ticker", "sector"])
        cleaned_map: dict[str, str] = {}

        for _, row in cleaned.iterrows():
            ticker = str(row["ticker"]).strip().upper()
            sector = str(row["sector"]).strip()
            if ticker and sector:
                cleaned_map[ticker] = sector

        overrides = {
            ticker: sector
            for ticker, sector in cleaned_map.items()
            if base_mapping.get(ticker) != sector
        }

        st.session_state.custom_sector_mapping = overrides
        save_mapping_overrides(overrides)
        st.session_state.rag_cache = {}
        st.success(
            "Saved {} mapping override(s).".format(
                len(st.session_state.custom_sector_mapping)
            )
        )

    if reset_clicked:
        st.session_state.custom_sector_mapping = {}
        save_mapping_overrides({})
        st.session_state.rag_cache = {}
        st.success("Mapping reset to defaults.")
        st.rerun()


def _get_base_sector_mapping() -> dict[str, str]:
    """Load the default RAG sector mapping and cache it in session state."""
    if "base_sector_mapping" in st.session_state:
        return st.session_state.base_sector_mapping

    st.session_state.base_sector_mapping = get_default_sector_mapping()
    return st.session_state.base_sector_mapping
