# 📚 AI Senticor Engine - Learning Guide

Welcome to the AI Senticor Engine learning guide! This document helps you understand the project from three critical perspectives: Business, Software Architecture, and AI/ML.

---

## 📖 Documentation Overview

This learning guide provides a high-level overview. For detailed technical information:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete system architecture, module breakdowns, design patterns, and data flow
- **[docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)** - RAG system implementation, LLM providers, sector news analysis, and configuration
- **[README.md](README.md)** - Quick start guide, installation, and feature overview
- **[ROADMAP.md](ROADMAP.md)** - Future features and development timeline

---

## 🎯 Who Should Read This?

- **Business Analysts**: Understand the value proposition and use cases
- **Software Engineers**: Learn the architecture and design patterns
- **Data Scientists/ML Engineers**: Explore AI/ML implementations
- **Students**: Study a real-world AI application
- **Contributors**: Get up to speed before contributing

---

## 📊 Part 1: Business Knowledge

### What Problem Does This Solve?

**The Challenge:**
- Stock investors need to analyze sentiment from news and market data
- Manual analysis of hundreds of articles is time-consuming
- Sector-wide trends are hard to identify without context
- Opportunity identification requires cross-referencing multiple data sources

**The Solution:**
AI Senticor Engine automates:
1. **News sentiment analysis** using FinBERT (financial NLP model)
2. **Sector-wide intelligence** via RAG (Retrieval-Augmented Generation)
3. **Alpha signal detection** by combining sentiment + volatility metrics
4. **Visual market mapping** for quick decision-making

### Business Value Proposition

| Stakeholder | Value Delivered |
|-------------|-----------------|
| **Retail Investors** | Quick sentiment scoring for portfolio stocks |
| **Portfolio Managers** | Sector-wide trend analysis |
| **Risk Analysts** | Volatility + sentiment correlation |
| **Researchers** | Study financial sentiment patterns |
| **Students** | Learn AI-powered financial analysis |

### Key Business Metrics

**Input Metrics:**
- Number of tickers analyzed per session
- News articles processed per ticker
- Sector coverage (Technology, Semiconductors, Banking, etc.)

**Output Metrics:**
- Sentiment score (-1 to +1 scale)
- Volatility (standard deviation of returns)
- Alpha flags (positive sentiment + low volatility)
- Trend direction (current vs historical sentiment)

**Performance Metrics:**
- Analysis time per ticker
- API call efficiency (caching reduces redundancy)
- Accuracy of sentiment vs actual price movement (backtest)

### Use Cases

**1. Daily Portfolio Check**
```
User Input: AAPL, MSFT, GOOGL
Output: Sentiment scores, volatility, alpha flags
Decision: Rebalance if alpha flags change
```

**2. Sector Analysis**
```
User Input: NVDA (with RAG enabled)
Output: Semiconductor sector commentary
Decision: Understand competitive dynamics
```

**3. Risk Monitoring**
```
User Input: Multiple tech stocks
Output: Quadrant map showing danger zones
Decision: Exit positions in DANGER ZONE
```

### Financial Concepts Used

- **Sentiment**: Market perception based on news (positive/negative/neutral)
- **Volatility**: Price fluctuation risk (high = risky, low = stable)
- **Alpha**: Excess returns above market benchmark
- **Trend**: Direction of sentiment change over time
- **Sector**: Industry grouping (Tech, Energy, Finance, etc.)
- **RAG (Retrieval-Augmented Generation)**: AI system that retrieves relevant sector news and generates contextual commentary

**📖 For RAG system details, LLM providers, and sector news analysis, see [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)**

**Recommended Resources:**
- 📖 [Sentiment Analysis in Finance (Investopedia)](https://www.investopedia.com/terms/s/sentimentindicator.asp)
- 📖 [What is Alpha in Investing?](https://www.investopedia.com/terms/a/alpha.asp)
- 📖 [Understanding Stock Volatility](https://www.investopedia.com/terms/v/volatility.asp)

---

## 🏗️ Part 2: Software Development & Architecture

### System Overview

The AI Senticor Engine follows a modular, layered architecture with clear separation of concerns:
- **Presentation Layer**: Streamlit web interface
- **Application Layer**: Charts, PDF generation, configuration
- **Business Logic Layer**: Sentiment engine, data fetcher, RAG engine
- **External Services**: FinBERT, OpenBB, ChromaDB, LLMs

**📖 For detailed architecture diagrams, module breakdowns, and design patterns, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

### Design Patterns Used

**1. Separation of Concerns**
- **UI Layer** (`ui/`): User interaction, visualization
- **Business Logic** (`core/`): Data processing, analysis
- **Utilities** (`utils/`): Shared helpers, logging

**2. Factory Pattern**
```python
# MarketDataClient creates appropriate provider
client = MarketDataClient(
    price_provider="yfinance",
    news_provider="yfinance"
)
```

**3. Strategy Pattern**
```python
# RAGEngine supports multiple LLM providers
rag = RAGEngine(
    llm_provider="openai",  # or "anthropic", "huggingface"
    model="gpt-4o-mini"
)
```

**4. Caching Pattern**
```python
# Session-based caching prevents redundant API calls
if cache_key in st.session_state.rag_cache:
    return cached_result
```

**5. Observer Pattern**
```python
# Logger tracks events across system
AppLogger.info("Analysis started", ticker=ticker)
```

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **Data Processing** | Pandas | Data manipulation |
| **Visualization** | Plotly | Interactive charts |
| **NLP** | HuggingFace Transformers | FinBERT sentiment model |
| **Vector DB** | ChromaDB | Semantic news search |
| **Embeddings** | sentence-transformers | Text vectorization |
| **RAG** | LangChain | Retrieval orchestration |
| **LLM** | OpenAI/Anthropic/HF | Text generation |
| **Market Data** | yfinance | Stock prices & news |
| **Testing** | pytest | Unit/integration tests |

### Code Quality Practices

**Type Hints:**
```python
def analyze_ticker(
    ticker: str,
    client: MarketDataClient,
    engine: SentimentEngine
) -> dict:
    """Analyze a single ticker and return metrics."""
```

**Error Handling:**
```python
try:
    analysis = analyze_ticker(ticker, client, engine)
except Exception as e:
    errors.append(f"{ticker}: {str(e)}")
    AppLogger.error("Analysis failed", str(e), ticker=ticker)
```

**Configuration Management:**
```python
# Centralized in config.py
ALPHA_SENTIMENT_THRESHOLD: Final[float] = 0.0
ALPHA_VOLATILITY_THRESHOLD: Final[float] = 0.5
```

**Logging:**
```python
# Comprehensive event tracking
AppLogger.info("Fetching data", ticker=ticker)
AppLogger.success("Analysis complete", ticker=ticker)
AppLogger.error("API call failed", details=str(e))
```

**Recommended Resources:**
- 📖 [Streamlit Documentation](https://docs.streamlit.io/)
- 📖 [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- 📖 [ChromaDB Getting Started](https://docs.trychroma.com/getting-started)
- 📖 [Plotly Python](https://plotly.com/python/)

---

## 🤖 Part 3: AI & Machine Learning

### AI Components Overview

**1. Sentiment Analysis (Supervised Learning)**
- **Model**: FinBERT (BERT fine-tuned on financial text)
- **Task**: Multi-class classification (positive/neutral/negative)
- **Input**: News headlines (text)
- **Output**: Sentiment label + confidence score

**2. Embeddings (Self-Supervised Learning)**
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Task**: Text → dense vector (384 dimensions)
- **Input**: Sector news articles
- **Output**: Semantic embeddings for similarity search

**3. RAG (Retrieval-Augmented Generation)**
- **Retrieval**: ChromaDB semantic similarity search
- **Generation**: LLM (GPT-4, Claude, FLAN-T5)
- **Task**: Generate sector commentary based on retrieved news

**📖 For complete RAG pipeline details, prompt engineering, and LLM configuration, see [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)**

### AI Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SENTIMENT ANALYSIS PIPELINE                              │
└─────────────────────────────────────────────────────────────┘
News Headlines (text)
    ↓
Tokenization (BERT tokenizer)
    ↓
FinBERT Model (768-dim hidden states)
    ↓
Classification Head (3 classes)
    ↓
Softmax → [positive: 0.7, neutral: 0.2, negative: 0.1]
    ↓
Aggregate Score: 0.7 * 1 + 0.2 * 0 + 0.1 * (-1) = 0.6

┌─────────────────────────────────────────────────────────────┐
│ 2. RAG PIPELINE                                              │
└─────────────────────────────────────────────────────────────┘
User Query: "NVDA sentiment analysis"
    ↓
Determine Sector: Semiconductors
    ↓
Build Query: "Semiconductors sector market news trends"
    ↓
Embed Query → [0.12, -0.34, 0.56, ..., 0.23] (384-dim vector)
    ↓
ChromaDB Similarity Search (cosine similarity)
    ↓
Retrieve Top-K News Articles (k=10)
    ↓
Construct Prompt with Context
    ↓
LLM Generation (GPT-4/Claude/FLAN-T5)
    ↓
Sector Commentary (natural language)
```

### Model Details

**FinBERT Sentiment Model**
- Architecture: BERT-base (110M parameters)
- Performance: 97% accuracy on financial sentiment
- Inference: ~50ms per headline (CPU)

**Embedding Model**
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimensions: 384
- Use Case: Semantic similarity search

**📖 For LLM provider comparison, pricing, and configuration options, see [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)**

### Vector Database (ChromaDB)

**Key Concepts:**
- Stores news articles as semantic embeddings (384 dimensions)
- Uses cosine similarity for semantic search
- Returns top-K most relevant documents

**📖 For ChromaDB schema, indexing details, and retrieval configuration, see [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)**

### AI Concepts Explained

**1. Transfer Learning**
- FinBERT: Pre-trained on general text → fine-tuned on financial text
- Benefit: Leverages language understanding from massive datasets

**2. Embeddings**
- Convert text to vectors in semantic space
- Similar meanings → similar vectors
- Enables similarity search

**3. Retrieval-Augmented Generation (RAG)**
- Problem: LLMs have outdated knowledge
- Solution: Retrieve relevant docs → augment prompt → generate
- Benefit: Up-to-date, grounded responses

**4. Semantic Search**
- Traditional: Keyword matching ("NVDA" in text)
- Semantic: Meaning matching ("GPU manufacturer" finds NVDA)

**5. Temperature in LLMs**
- Low (0.0-0.3): Deterministic, focused responses
- High (0.7-1.0): Creative, varied responses
- Default: 0.3 for financial analysis (consistency)

### Evaluation Metrics

**Sentiment Analysis:**
- **Accuracy**: % of correct predictions
- **Precision/Recall**: For each class (positive/neutral/negative)
- **F1-Score**: Harmonic mean of precision/recall

**RAG System:**
- **Retrieval Quality**: Are retrieved docs relevant?
- **Generation Quality**: Is commentary accurate and helpful?
- **Latency**: Time to generate response
- **Cost**: API calls per analysis

**Alpha Signals:**
- **Precision**: % of flagged stocks that outperform
- **Recall**: % of outperformers that were flagged
- **Backtest Returns**: Simulated portfolio performance

**Recommended Resources:**
- 📖 [FinBERT Paper](https://arxiv.org/abs/1908.10063)
- 📖 [BERT Explained](http://jalammar.github.io/illustrated-bert/)
- 📖 [Sentence Transformers](https://www.sbert.net/)
- 📖 [RAG Tutorial (LangChain)](https://python.langchain.com/docs/tutorials/rag/)
- 📖 [ChromaDB Documentation](https://docs.trychroma.com/)
- 📖 [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## 🚀 Quick Start Paths

### For Business Users
1. Run the app: `streamlit run src/app.py`
   - Optional: Force reload sample news database: `streamlit run src/app.py -- --reload-db`
2. Analyze 5 stocks in your portfolio
3. Read the **Business Knowledge** section above
4. Experiment with RAG-powered sector analysis (see [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md))
5. Share insights with your team

### For Developers
1. Set up dev environment: `pip install -r requirements.txt`
2. Initialize database (optional): `streamlit run src/app.py -- --reload-db`
3. Read **Software Architecture** section above
4. Study the full architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
5. Run tests: `pytest tests/`
5. Read existing code in `core/` and `ui/`
6. Pick a feature from issues to implement

### For Data Scientists
1. Study **AI & ML** section above
2. Experiment with FinBERT in a notebook
3. Learn about RAG implementation: [docs/RAG_SECTOR_NEWS.md](docs/RAG_SECTOR_NEWS.md)
4. Analyze RAG retrieval quality
5. Read `core/rag_engine.py` implementation
6. Propose model improvements

---

**Happy Learning! 🎓**
