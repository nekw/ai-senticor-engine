"""Competitive Analysis tab powered by the asyncio multi-agent orchestrator."""

import asyncio

import pandas as pd
import streamlit as st

from src.ui.rag_mapping import apply_mapping_overrides
from src.utils.pdf_gen import generate_competitive_pdf_report


def render_competitive_tab(data: pd.DataFrame):
    """Render autonomous sector competitive analysis UI."""
    st.header("🏁 Competitive Analysis")
    st.caption(
        "Multi-agent sector research for peer positioning, signals, and catalysts."
    )

    focus_options = _resolve_focus_options(data)
    if not focus_options:
        st.info("Add at least one ticker in the sidebar to run competitive analysis.")
        return

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        focus_ticker = st.selectbox("Focus Ticker", options=focus_options)
    with col2:
        max_competitors = st.slider("Peers", min_value=2, max_value=12, value=6)
    with col3:
        max_articles = st.slider("News / Peer", min_value=5, max_value=30, value=15)

    sector_override = st.text_input(
        "Sector Override (optional)",
        value="",
        help="Leave blank to use the built-in ticker -> sector mapping.",
    ).strip()

    run_clicked = st.button("Run Competitive Analysis", use_container_width=True)

    cache_key = _build_cache_key(
        focus_ticker,
        sector_override,
        max_competitors,
        max_articles,
        st.session_state.get("llm_provider"),
        st.session_state.get("llm_model"),
        st.session_state.get("llm_temperature"),
    )

    if run_clicked:
        with st.spinner("Running multi-agent competitive research..."):
            report = _run_orchestrator(
                focus_ticker=focus_ticker,
                max_competitors=max_competitors,
                max_articles=max_articles,
                sector_override=sector_override or None,
            )
            st.session_state.competitive_cache[cache_key] = report

    report = st.session_state.competitive_cache.get(cache_key)
    if report is None:
        st.info("Configure parameters and click Run Competitive Analysis.")
        return

    _render_report(report)


def _resolve_focus_options(data: pd.DataFrame) -> list[str]:
    """Build selectable ticker options from analysis data or sidebar input."""
    if data is not None and not data.empty and "ticker" in data.columns:
        return sorted(data["ticker"].astype(str).str.upper().unique().tolist())

    raw_tickers = st.session_state.get("current_tickers", "")
    options = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    return sorted(set(options))


def _build_cache_key(*parts) -> str:
    """Create deterministic cache key from query and model parameters."""
    normalized = [str(p) if p is not None else "none" for p in parts]
    return "competitive_" + "|".join(normalized)


def _run_orchestrator(
    focus_ticker: str,
    max_competitors: int,
    max_articles: int,
    sector_override: str | None,
) -> dict:
    """Run orchestrator end-to-end and return normalized report payload."""
    # Keep these imports local so app startup does not pay competitive stack cost.
    from src.core.agents import CompetitiveAnalysisOrchestrator, CompetitiveQuery
    from src.core.rag_engine import RAGEngine

    rag_engine = RAGEngine(
        llm_provider=st.session_state.get("llm_provider"),
        model=st.session_state.get("llm_model") or "gpt-4o-mini",
        temperature=st.session_state.get("llm_temperature", 0.3),
    )
    rag_engine = apply_mapping_overrides(rag_engine)

    orchestrator = CompetitiveAnalysisOrchestrator(
        client=st.session_state.client,
        sentiment_engine=st.session_state.engine,
        rag_engine=rag_engine,
    )

    query = CompetitiveQuery(
        focus_ticker=focus_ticker,
        sector=sector_override,
        max_competitors=max_competitors,
        max_articles_per_ticker=max_articles,
    )

    return asyncio.run(orchestrator.run(query))


def _render_report(report: dict):
    """Render report payload returned by the orchestrator."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sector", report.get("sector", "Unknown"))
    with col2:
        st.metric("Peers", len(report.get("peer_universe", [])))
    with col3:
        st.metric("Confidence", f"{report.get('confidence', 0.0):.2f}")
    with col4:
        st.metric("Errors", len(report.get("errors", [])))

    _render_competitive_pdf_download(report)

    st.subheader("Peer Universe")
    st.write(", ".join(report.get("peer_universe", [])) or "No peers resolved")

    signals = report.get("signals", [])
    st.subheader("Signal Comparison")
    if signals:
        signal_df = pd.DataFrame(signals).sort_values("sentiment", ascending=False)
        st.dataframe(signal_df, hide_index=True, use_container_width=True)
    else:
        st.warning("No competitor signals were generated.")

    st.subheader("Competitive Narrative")
    narrative = report.get("narrative", "")
    if narrative:
        st.markdown(narrative, unsafe_allow_html=True)
    else:
        st.info("Narrative not available for this run.")

    citations = report.get("citations", {})
    if citations:
        with st.expander("Sources", expanded=False):
            for ticker, items in citations.items():
                st.markdown(f"**{ticker}**")
                for headline in items:
                    st.write(f"- {headline}")

    risk_flags = report.get("risk_flags", [])
    if risk_flags:
        st.subheader("Risk Flags")
        for flag in risk_flags:
            st.warning(flag)

    errors = report.get("errors", [])
    if errors:
        with st.expander("Agent Errors", expanded=False):
            for err in errors:
                st.code(err)


def _render_competitive_pdf_download(report: dict):
    """Render generate/download controls for competitive PDF report."""
    if st.button(
        "📄 Competitive PDF",
        help="Create and download a competitive analysis report",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_pdf_progress(step, message):
            progress = (step + 1) / 6
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            pdf = generate_competitive_pdf_report(
                report, progress_callback=update_pdf_progress
            )
            progress_bar.progress(1.0)
            status_text.text("✅ Competitive PDF generated successfully!")

            st.download_button(
                "📥 Download Competitive PDF",
                data=pdf,
                file_name="senticor_competitive_report_{}.pdf".format(
                    pd.Timestamp.now().strftime("%Y%m%d_%H%M")
                ),
                mime="application/pdf",
            )
        finally:
            import time

            time.sleep(1.5)
            progress_bar.empty()
            status_text.empty()
