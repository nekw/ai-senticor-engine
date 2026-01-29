# RAG-Powered Sector News Analysis

## Overview

The AI Senticor Engine uses **Retrieval-Augmented Generation (RAG)** with Large Language Models to provide intelligent sector-wide market commentary. Instead of analyzing companies in isolation, the system retrieves relevant news from the entire sector and generates comprehensive insights using AI.

**Key Features:**
- ✅ Semantic search across sector news using vector embeddings
- ✅ LLM-powered contextual analysis and insights
- ✅ Three provider options: HuggingFace (FREE), OpenAI, Anthropic
- ✅ Graceful fallback to template-based responses
- ✅ Comprehensive sector mapping across 9+ industries

---

## How It Works

### 1. Sector Mapping
Each ticker is mapped to its industry sector:

| Sector | Tickers |
|--------|---------|
| **Technology** | AAPL, MSFT, GOOGL |
| **Semiconductors** | NVDA, AMD, TSM, INTC |
| **Banking** | JPM, BAC, WFC, C, GS, MS |
| **Energy** | XOM, CVX |
| **Healthcare** | JNJ, UNH, PFE |
| **Retail** | WMT, TGT, COST |
| **E-commerce** | AMZN |
| **Social Media** | META |
| **Automotive** | TSLA, F, GM |

### 2. News Storage
News is stored in a ChromaDB vector database with semantic embeddings:
- **Collection**: `sector_news`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Metadata**: sector, headline, date, ticker, url, added_at

### 3. Semantic Retrieval
When analyzing a stock (e.g., AAPL):
1. System identifies the sector (Technology)
2. Retrieves top-k most semantically relevant news from that sector
3. Passes the context + company sentiment to an LLM

### 4. LLM Commentary Generation
The LLM generates sector-wide insights covering:
- Overall sector trends and dynamics
- How the company fits within sector landscape
- Key opportunities and risk factors
- Competitive positioning

## Usage Examples

### Adding Sector News
```python
from src.core.rag_engine import RAGEngine

# No LLM needed for adding news
rag = RAGEngine(llm_provider=None)

# Add tech sector news
rag.add_sector_news(
    sector="Technology",
    headline="Apple Announces AI Features in iOS",
    content="Apple unveiled new AI enhancements...",
    ticker="AAPL",
    date="2024-01-15",
    url="https://example.com/article"
)

# Add semiconductor news
rag.add_sector_news(
    sector="Semiconductors",
    headline="NVIDIA Launches AI Chips for Data Centers",
    content="NVIDIA unveiled next-gen GPUs...",
    ticker="NVDA",
    date="2024-01-18",
    url="https://example.com/nvda-ai-chips"
)
```

### Getting Sector Commentary
```python
# Initialize with LLM provider
rag = RAGEngine(
    llm_provider="openai",
    model="gpt-4o-mini",
    temperature=0.3
)

# Get sector analysis for a ticker
commentary = rag.get_sector_commentary(
    ticker="AAPL",
    company_sentiment=0.75,  # Positive sentiment
    k=10  # Retrieve top 10 relevant news
)

print(commentary)
```

### Using Free HuggingFace Models
```python
# Use FLAN-T5 (free, local)
rag = RAGEngine(
    llm_provider="huggingface",
    model="google/flan-t5-large",
    temperature=0.3
)

commentary = rag.get_sector_commentary(
    ticker="NVDA",
    company_sentiment=0.85,
    k=8
)
```

### Managing Sector Mappings
```python
# Add custom ticker mapping
rag.add_ticker_sector_mapping("SHOP", "E-commerce")

# Get sector for a ticker
sector = rag.get_sector("AAPL")  # Returns: "Technology"

# List all available sectors
sectors = rag.get_all_sectors()
print(sectors)  # ['Technology', 'Semiconductors', 'Banking', ...]

# Get all tickers in a sector
tech_tickers = rag.get_sector_tickers("Technology")
print(tech_tickers)  # ['AAPL', 'MSFT', 'GOOGL']
```

---

## Temperature Settings

Control the creativity vs consistency of LLM responses:

```python
# Conservative (factual, focused)
rag = RAGEngine(llm_provider="openai", temperature=0.1)

# Balanced (recommended for sector analysis)
rag = RAGEngine(llm_provider="openai", temperature=0.3)

# Creative (more varied insights)
rag = RAGEngine(llm_provider="openai", temperature=0.7)
```

**Recommendations:**
- **0.1-0.3**: Factual financial analysis
- **0.4-0.6**: Balanced insights with some creativity
- **0.7-0.9**: Exploratory analysis, hypothesis generation

---

## Streamlit UI Integration

### Enabling Sector News Analysis
1. Open sidebar in Streamlit app
2. Enable "📰 Sector News Analysis" checkbox
3. Select LLM Provider:
   - **HuggingFace** (FREE) - No setup required
   - **OpenAI** - Requires `OBB_OPENAI_API_KEY` in .env
   - **Anthropic** - Requires `OBB_ANTHROPIC_API_KEY` in .env
4. Choose model (or use default)
5. Adjust temperature slider (0.0-1.0)
6. Analyze any ticker to see sector commentary

### Viewing Sector Intelligence
In the **Company Intelligence** tab, sector analysis appears as:
- **📊 Sector Intelligence & News Analysis**: Expandable section with LLM-generated commentary
- **📋 View Source News Articles**: Expandable list showing the actual news articles used by the RAG system
  - Displays headline, date, ticker, and content preview
  - Provides transparency into AI analysis sources
  - Helps users verify the context used for commentary generation

---

## Architecture Deep Dive

### Vector Database (ChromaDB)
**Storage Structure:**
- **Collection**: `sector_news`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Metadata Fields**:
  - `sector`: Industry sector
  - `headline`: News headline
  - `date`: Publication date
  - `ticker`: Company ticker
  - `url`: Source URL
  - `added_at`: Timestamp when added

### Retrieval Process
1. User selects ticker (e.g., "AAPL")
2. System identifies sector ("Technology")
3. ChromaDB searches for semantically similar news in that sector
4. Top-k results ranked by relevance score
5. News context + company sentiment → LLM prompt

### LLM Generation

**Prompt Structure (OpenAI/Anthropic):**
```
System: You are a financial markets analyst specializing in sector-wide analysis.
Analyze recent sector news to provide comprehensive commentary on market trends,
key developments, competitive dynamics, and outlook. Focus on actionable insights.

User: Sector: {sector}
Company: {ticker}
Company Sentiment: {company_sentiment:.2f} ({sentiment_label})

Recent Sector News:
{news_context}

Provide a comprehensive sector commentary including:

1. **Sector Overview**
   - Current market sentiment and momentum
   - Key themes affecting the sector

2. **Major Developments**
   - Most significant recent news
   - Competitive dynamics and market shifts

3. **Impact on {ticker}**
   - How sector trends affect this specific company
   - Relative positioning within the sector

4. **Outlook**
   - Near-term catalysts or headwinds
   - Sector-wide opportunities or risks

Keep response concise but insightful (250-350 words).
```

**HuggingFace (FLAN-T5) Prompt Format:**
```
Context: Recent {sector} sector news shows {ticker} competitors: {competitor_tickers}.

Key developments:
{news_context}

Based on this {sector} sector news, write a 3-paragraph market analysis covering:
1) Overall sector trends
2) Impact on {ticker} (sentiment: {sentiment_label})
3) Outlook and risks.
```

**Generation Parameters:**
- **API Models (GPT/Claude)**: Uses temperature setting from configuration
- **HuggingFace FLAN-T5**:
  - Max length: 350 tokens, Min length: 100 tokens
  - Temperature: 0.7, Top-p: 0.95
  - Repetition penalty: 1.5
  - No repeat n-gram size: 4

**Fallback Mechanism:**
If the LLM fails or is unavailable, the system provides a formatted response with the top 5 most relevant news headlines, dates, and related tickers.

---

## Advantages Over Traditional Analysis

### 1. Sector Context
Unlike individual stock analysis, sector news provides:
- Industry-wide trends and momentum
- Competitive dynamics and market share shifts
- Regulatory changes affecting the sector
- Market sentiment across similar companies

### 2. Semantic Search
Vector database enables:
- Finding conceptually similar news (e.g., "AI investment" matches "machine learning spending")
- Time-aware relevance (recent news weighted higher)
- Cross-company insights (NVDA news may be relevant for AMD analysis)

### 3. LLM Reasoning
LLMs can:
- Synthesize multiple news sources into coherent narrative
- Identify conflicting signals and explain nuances
- Provide context-aware interpretation
- Adapt to different sentiment scenarios

---

## References

- **ChromaDB Documentation**: https://docs.trychroma.com/
- **LangChain RAG Guide**: https://python.langchain.com/docs/use_cases/question_answering/
- **Sentence Transformers**: https://www.sbert.net/
- **OpenAI API**: https://platform.openai.com/docs/
- **Anthropic Claude**: https://docs.anthropic.com/
- **HuggingFace Models**: https://huggingface.co/models

---

## Summary

The RAG-powered sector news system provides:
- ✅ Intelligent, context-aware market commentary
- ✅ Flexible LLM provider options (free to premium)
- ✅ Semantic search across sector news
- ✅ Easy integration with Streamlit UI
- ✅ Graceful fallbacks for reliability

**Recommendation:** Start with **HuggingFace (free)** for testing, then upgrade to **GPT-4o-mini** for production use if you need faster, higher-quality insights.
