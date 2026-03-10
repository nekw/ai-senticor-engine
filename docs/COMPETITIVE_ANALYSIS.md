# Competitive Analysis Agent

## Overview

The Competitive Analysis Agent adds autonomous sector research to Senticor using a lightweight multi-agent architecture.

The feature is available in the Streamlit `Competitive Analysis` tab and is designed to answer:
- Who are the closest peers for my focus ticker?
- How do peer sentiment and volatility compare right now?
- What sector developments matter most for competitive positioning?
- How confident is the system in its own output?

## Architecture

The implementation uses a supervisor-style orchestrator with specialized workers and a shared typed state.

- `CompetitiveAnalysisOrchestrator`: Coordinates the full run lifecycle.
- `UniverseAgent`: Resolves sector and builds bounded peer universe.
- `NewsIntelAgent`: Fetches peer news headlines in parallel.
- `SignalAgent`: Computes sentiment, trend, and annualized volatility per peer.
- `NarrativeAgent`: Uses `RAGEngine` to generate competitive narrative and source headlines.
- `RiskAgent`: Computes confidence score and risk flags.
- `ReportAgent`: Returns normalized report payload for UI/export.

## Multi-Stage Service Orchestration View

This feature can be read as a deterministic service pipeline where each "agent"
is a bounded stage with a clear input/output contract.

Stage mapping:
- `UniverseAgent` -> Universe Resolution Service
- `NewsIntelAgent` -> News Ingestion Service
- `SignalAgent` -> Quant Signal Service
- `NarrativeAgent` -> Narrative Generation Service
- `RiskAgent` -> Confidence and Guardrails Service
- `ReportAgent` -> Response Assembly Service

Why this matters:
- Easier debugging: each stage can be tested and inspected independently.
- Safer evolution: new stages can be inserted with minimal regression risk.
- Better reliability: partial failures degrade gracefully into `errors` and lower confidence.

Service-style sequence diagram (ASCII):

```text
+-------------------+
|      UI Query     |
+---------+---------+
      |
      v
+-------------------------+
|  Orchestrator Service   |
+-----------+-------------+
      |
      v
+-------------------------------+
| Universe Resolution Service   |
+---------------+---------------+
        |
        v
+-------------------------------+        uses        +------------------+
| News Ingestion Service        +------------------->| MarketDataClient |
+---------------+---------------+                    +------------------+
        |
        v
+-------------------------------+        uses        +-----------------------------+
| Quant Signal Service          +------------------->| SentimentEngine + Processor |
+---------------+---------------+                    +-----------------------------+
        |
        v
+-------------------------------+        uses        +-----------+
| Narrative Generation Service  +------------------->| RAGEngine |
+---------------+---------------+                    +-----------+
        |
        v
+-------------------------------------+
| Confidence and Guardrails Service   |
+-----------------+-------------------+
          |
          v
+-------------------------------+
| Response Assembly Service     |
+---------------+---------------+
        |
        v
+-------------------------------+
| UI Render + Competitive PDF   |
+-------------------------------+
```

Primary code locations:
- `src/core/agents/orchestrator.py`
- `src/core/agents/state.py`
- `src/core/agents/universe_agent.py`
- `src/core/agents/news_intel_agent.py`
- `src/core/agents/signal_agent.py`
- `src/core/agents/narrative_agent.py`
- `src/core/agents/risk_agent.py`
- `src/core/agents/report_agent.py`

## Execution Flow

```text
Query -> Universe -> News -> Signals -> Narrative -> Risk -> Report
```

Detailed sequence:
1. User selects focus ticker and limits in the `Competitive Analysis` tab.
2. Orchestrator creates shared `CompetitiveAnalysisState`.
3. UniverseAgent resolves sector and peer list (`max_competitors`).
4. NewsIntelAgent fetches normalized headlines (`max_articles_per_ticker`).
5. SignalAgent computes per-peer metrics using FinBERT + price data.
6. NarrativeAgent requests sector commentary from `RAGEngine`.
7. RiskAgent computes confidence and warning flags.
8. ReportAgent emits UI-friendly payload.
9. UI renders results and optionally exports Competitive PDF.

## UI Usage

1. Run app:
```bash
streamlit run src/app.py
```

2. Open `Competitive Analysis` tab.
3. Select:
- `Focus Ticker`
- `Peers` count
- `News / Peer` limit
- optional `Sector Override`

4. Click `Run Competitive Analysis`.
5. Review:
- Peer universe
- Signal comparison table
- Narrative and source headlines
- Confidence and risk flags

6. Click `Competitive PDF` to export report.

## Output Schema

The orchestrator returns a normalized `dict` payload:

```python
{
  "generated_at": str,
  "focus_ticker": str,
  "sector": str,
  "peer_universe": list[str],
  "signals": [
    {
      "ticker": str,
      "sentiment": float,
      "sentiment_trend": float,
      "volatility": float,
      "news_count": int,
    }
  ],
  "narrative": str,
  "citations": dict[str, list[str]],
  "confidence": float,
  "risk_flags": list[str],
  "errors": list[str],
}
```

## Caching

Session state entries used by this feature:
- `competitive_cache`: Caches report payload by query/model tuple.
- `current_tickers`: Stores latest sidebar input for focus defaults.
- `llm_provider`, `llm_model`, `llm_temperature`: Controls narrative generation.

## Testing

Coverage for core orchestration is in:
- `tests/test_competitive_orchestrator.py`

Related integration tests:
- `tests/test_analysis_engine.py`
- `tests/test_rag_engine.py`
- `tests/test_pdf_gen.py`

## Current Limitations

- Peer universe currently derives from static ticker-to-sector mapping in `RAGEngine`.
- Confidence is heuristic (coverage and error penalty), not probabilistic calibration.
- Narrative quality depends on available sector-news corpus and selected LLM.

## Future Extensions

- Add dynamic universe expansion (market cap, correlation, thematic similarity).
- Add hypothesis generation/testing loops (v2.0 LangGraph plan).
- Add JSON/Markdown exports and API endpoint for machine-consumable output.
