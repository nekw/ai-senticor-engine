"""Sector News tab with DB controls and persisted sector news explorer."""

import asyncio

import pandas as pd
import streamlit as st

from src.ui.rag_mapping import apply_mapping_overrides


def render_sector_news_tab():
    """Render sector news management and explorer UI."""
    st.header("📰 Sector News")
    st.caption("Manage the sector vector database and browse all stored sector news.")

    st.subheader("🛠️ Sector DB Controls")
    _render_sector_db_controls()

    st.divider()
    st.subheader("📚 All Sector News")
    _render_sector_news_table()


def _render_sector_db_controls():
    """Render controls for clearing/loading the sector vector database."""
    lookback_options = [f"{day}D" for day in range(1, 30)] + ["1M"]
    lookback_label = st.select_slider(
        "Lookback Window",
        options=lookback_options,
        value="7D",
        help="Choose news recency from 1 day to 1 month (default 7D).",
    )

    col1, col2 = st.columns(2)
    with col1:
        clear_db_clicked = st.button("🗑️ Clear Sector DB", use_container_width=True)
    with col2:
        load_news_clicked = st.button(
            "📥 Load All Sector News", use_container_width=True
        )

    if clear_db_clicked:
        _clear_sector_db()

    if load_news_clicked:
        lookback_days = _parse_lookback_days(lookback_label)
        _load_all_sector_news(lookback_days)


def _render_sector_news_table():
    """Render all persisted sector-news records from Chroma."""
    try:
        from src.core.rag_engine import RAGEngine

        rag = apply_mapping_overrides(RAGEngine(llm_provider=None))
        records_df = _load_sector_news_records(rag)
    except Exception as e:
        st.error(f"Failed to read sector news records: {str(e)}")
        return

    if records_df.empty:
        st.info("No sector news found. Load news using the controls above.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        sector_options = ["All"] + sorted(records_df["sector"].dropna().unique())
        selected_sector = st.selectbox("Filter by Sector", options=sector_options)
    with col2:
        st.metric("Articles", len(records_df))

    filtered_df = records_df
    if selected_sector != "All":
        filtered_df = records_df[records_df["sector"] == selected_sector]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "date": st.column_config.TextColumn("Date"),
            "sector": st.column_config.TextColumn("Sector"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
            "content": st.column_config.TextColumn("Content", width="large"),
            "url": st.column_config.LinkColumn("URL", display_text="Open"),
        },
    )


def _load_sector_news_records(rag) -> pd.DataFrame:
    """Fetch all records from Chroma and normalize for table display."""
    raw = rag.vectorstore.get(include=["documents", "metadatas"])
    documents = raw.get("documents", []) or []
    metadatas = raw.get("metadatas", []) or []

    rows: list[dict[str, str]] = []
    for idx, content in enumerate(documents):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        rows.append(
            {
                "date": str(meta.get("date") or ""),
                "sector": str(meta.get("sector") or "Unknown"),
                "ticker": str(meta.get("ticker") or ""),
                "headline": str(meta.get("headline") or "No headline"),
                "content": _truncate_text(str(content or "")),
                "url": str(meta.get("url") or ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["date", "sector", "ticker", "headline", "content", "url"]
        )

    records_df = pd.DataFrame(rows)
    # Collapse duplicate stories (common from provider fan-out across tickers).
    records_df = records_df.drop_duplicates(
        subset=["sector", "date", "headline", "url"], keep="first"
    )
    date_rank = pd.to_datetime(records_df["date"], errors="coerce")
    records_df = records_df.assign(_date_rank=date_rank)
    records_df = records_df.sort_values(
        by=["_date_rank", "sector", "headline"], ascending=[False, True, True]
    )
    return records_df.drop(columns=["_date_rank"])


def _truncate_text(text: str, max_length: int = 220) -> str:
    """Trim content snippets so the table remains readable."""
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3] + "..."


def _parse_lookback_days(lookback_label: str) -> int:
    """Parse lookback label (1D-1M) into integer days."""
    if lookback_label == "1M":
        return 30
    return int(lookback_label[:-1])


def _clear_sector_db():
    """Clear the persisted sector news vector database and related caches."""
    try:
        from src.core.rag_engine import RAGEngine

        rag = apply_mapping_overrides(RAGEngine(llm_provider=None))
        rag.clear_vector_database()
        st.session_state.rag_cache = {}
        st.session_state.competitive_cache = {}
        st.success("Sector DB cleared.")
        st.rerun()
    except Exception as e:
        st.error(f"Failed to clear sector DB: {str(e)}")


def _load_all_sector_news(lookback_days: int):
    """Load news for all mapped sector tickers into vector database."""
    try:
        from src.ui.sidebar import ensure_market_client

        client = ensure_market_client()
    except Exception as e:
        st.error(f"Failed to initialize market data client: {str(e)}")
        return

    try:
        from src.core.rag_engine import RAGEngine

        rag = apply_mapping_overrides(RAGEngine(llm_provider=None))
        tickers = sorted(set(rag.sector_mapping.keys()))

        with st.spinner(
            f"Loading sector news for {len(tickers)} tickers ({lookback_days}D)..."
        ):
            news_by_ticker = asyncio.run(_fetch_news_for_tickers(client, tickers))

            # Build keyset from existing records to avoid re-ingesting duplicates.
            existing_raw = rag.vectorstore.get(include=["metadatas"])
            existing_metas = existing_raw.get("metadatas", []) or []
            seen_news_keys = {
                _build_news_key(
                    sector=str(meta.get("sector") or ""),
                    date_str=str(meta.get("date") or ""),
                    headline=str(meta.get("headline") or ""),
                    url=str(meta.get("url") or ""),
                )
                for meta in existing_metas
            }

            cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(
                days=lookback_days
            )
            loaded_count = 0
            skipped_duplicates = 0

            for ticker, news_df in news_by_ticker.items():
                if news_df is None or news_df.empty:
                    continue

                filtered_df = _filter_news_by_lookback(news_df, cutoff)
                if filtered_df.empty:
                    continue

                sector = rag.get_sector(ticker)
                if sector == "Unknown":
                    continue

                for _, row in filtered_df.iterrows():
                    headline = str(
                        row.get("title") or row.get("headline") or "No headline"
                    )
                    content = str(
                        row.get("text")
                        or row.get("summary")
                        or row.get("description")
                        or headline
                    )

                    date_value = row.get("date")
                    parsed_date = pd.to_datetime(date_value, errors="coerce", utc=True)
                    date_str = (
                        parsed_date.tz_convert(None).strftime("%Y-%m-%d")
                        if not pd.isna(parsed_date)
                        else None
                    )

                    news_key = _build_news_key(
                        sector=sector,
                        date_str=str(date_str or ""),
                        headline=headline,
                        url=str(row.get("url") or ""),
                    )
                    if news_key in seen_news_keys:
                        skipped_duplicates += 1
                        continue

                    rag.add_sector_news(
                        sector=sector,
                        headline=headline,
                        content=content,
                        ticker=ticker,
                        date=date_str,
                        url=row.get("url"),
                    )
                    seen_news_keys.add(news_key)
                    loaded_count += 1

            st.session_state.rag_cache = {}
            st.session_state.competitive_cache = {}

            final_raw = rag.vectorstore.get(include=["metadatas"])
            final_metas = final_raw.get("metadatas", []) or []
            final_unique_keys = {
                _build_news_key(
                    sector=str(meta.get("sector") or ""),
                    date_str=str(meta.get("date") or ""),
                    headline=str(meta.get("headline") or ""),
                    url=str(meta.get("url") or ""),
                )
                for meta in final_metas
            }
            total_unique_articles = len(final_unique_keys)

        st.success(
            "Added {} new sector news articles from the last {} day(s). "
            "Skipped {} duplicates. Total unique articles in DB: {}.".format(
                loaded_count,
                lookback_days,
                skipped_duplicates,
                total_unique_articles,
            )
        )
    except Exception as e:
        st.error(f"Failed to load sector news: {str(e)}")


def _build_news_key(sector: str, date_str: str, headline: str, url: str) -> str:
    """Build normalized key used for dedupe across runs and tickers."""
    normalized_url = url.strip().lower()
    normalized_headline = " ".join(headline.strip().lower().split())
    return "|".join(
        [sector.strip().lower(), date_str.strip(), normalized_headline, normalized_url]
    )


def _filter_news_by_lookback(
    news_df: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Filter news DataFrame to rows within the lookback window."""
    if "date" not in news_df.columns:
        return news_df

    normalized_dates = pd.to_datetime(news_df["date"], errors="coerce", utc=True)
    normalized_dates = normalized_dates.dt.tz_convert(None)
    filtered = news_df.loc[normalized_dates >= cutoff].copy()
    return filtered


async def _fetch_news_for_tickers(
    client, tickers: list[str]
) -> dict[str, pd.DataFrame]:
    """Fetch news for each ticker concurrently using the configured market client."""

    async def _fetch_single(ticker: str) -> tuple[str, pd.DataFrame]:
        try:
            df = await client.fetch_company_news(ticker)
            return ticker, df
        except Exception:
            return ticker, pd.DataFrame()

    tasks = [_fetch_single(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    return {ticker: df for ticker, df in results}
