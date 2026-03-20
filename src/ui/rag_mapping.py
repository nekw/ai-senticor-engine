"""Helpers for applying user-defined ticker-sector mappings."""

import json
from pathlib import Path

import streamlit as st

MAPPING_FILE_PATH = Path("data") / "custom_sector_mapping.json"

# Keep a lightweight default mapping here so startup does not need to import
# the heavy RAG engine just to populate ticker lists.
DEFAULT_SECTOR_MAPPING = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOG": "Technology",
    "AMZN": "E-commerce",
    "META": "Social Media",
    # Semiconductors
    "NVDA": "Semiconductors",
    "AMD": "Semiconductors",
    "INTC": "Semiconductors",
    "MU": "Semiconductors",
    "LITE": "Semiconductors",
    "SNDK": "Semiconductors",
    # Financial Services
    "V": "Financial Services",
    "MA": "Financial Services",
    "JPM": "Banking",
    "BAC": "Banking",
    "WFC": "Banking",
    "GS": "Banking",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    # Healthcare
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    # Retail
    "WMT": "Retail",
    "HD": "Retail",
    "NKE": "Retail",
    # Entertainment
    "DIS": "Entertainment",
    # Automotive
    "TSLA": "Automotive",
    # Commodities
    "SLV": "Commodities",
    # Cryptocurrency
    "IBIT": "Cryptocurrency",
    "ETH": "Cryptocurrency",
    # Leveraged ETFs
    "TQQQ": "Leveraged ETFs",
    "SQQQ": "Leveraged ETFs",
}


def get_default_sector_mapping() -> dict[str, str]:
    """Return default ticker->sector mapping without importing RAG dependencies."""
    return dict(sorted(DEFAULT_SECTOR_MAPPING.items()))


def load_mapping_overrides() -> dict[str, str]:
    """Load ticker-sector overrides from local JSON file."""
    if not MAPPING_FILE_PATH.exists():
        return {}

    try:
        with MAPPING_FILE_PATH.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
    except Exception:
        return {}

    if not isinstance(loaded, dict):
        return {}

    normalized: dict[str, str] = {}
    for ticker, sector in loaded.items():
        ticker_text = str(ticker).strip().upper()
        sector_text = str(sector).strip()
        if ticker_text and sector_text:
            normalized[ticker_text] = sector_text

    return normalized


def save_mapping_overrides(overrides: dict[str, str]):
    """Persist ticker-sector overrides to local JSON file."""
    MAPPING_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MAPPING_FILE_PATH.open("w", encoding="utf-8") as file:
        json.dump(dict(sorted(overrides.items())), file, indent=2)


def get_mapping_overrides() -> dict[str, str]:
    """Return normalized ticker->sector overrides stored in session state."""
    if "custom_sector_mapping" not in st.session_state:
        st.session_state.custom_sector_mapping = load_mapping_overrides()

    raw_overrides = st.session_state.get("custom_sector_mapping", {})
    normalized: dict[str, str] = {}

    for ticker, sector in raw_overrides.items():
        ticker_text = str(ticker).strip().upper()
        sector_text = str(sector).strip()
        if ticker_text and sector_text:
            normalized[ticker_text] = sector_text

    st.session_state.custom_sector_mapping = normalized
    return normalized


def apply_mapping_overrides(rag_engine):
    """Apply session mapping overrides to a RAGEngine instance."""
    for ticker, sector in get_mapping_overrides().items():
        rag_engine.add_ticker_sector_mapping(ticker, sector)
    return rag_engine


def get_effective_mapping() -> dict[str, str]:
    """Return effective ticker-sector mapping (defaults + overrides)."""
    if "base_sector_mapping" not in st.session_state:
        st.session_state.base_sector_mapping = get_default_sector_mapping()

    effective = dict(st.session_state.base_sector_mapping)
    effective.update(get_mapping_overrides())
    return effective


def get_effective_tickers() -> list[str]:
    """Return sorted ticker list from the effective mapping table."""
    return sorted(get_effective_mapping().keys())
