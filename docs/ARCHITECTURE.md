# Project Architecture

## 🏗️ System Design Overview

AI-Senticor-Engine follows a modular, layered architecture designed for maintainability, testability, and scalability.

```
┌─────────────────────────────────────────────────────────┐
│                    Presentation Layer                   │
│              (Streamlit Web Interface)                  │
│                      src/app.py                         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                   Application Layer                     │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Charts     │  │  PDF Gen     │  │   Config     │    │
│  │ (utils/)    │  │  (utils/)    │  │  (config.py) │    │
│  └─────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                   Business Logic Layer                  │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Sentiment  │  │  Data        │  │  Processor   │    │
│  │  Engine     │  │  Fetcher     │  │  (Metrics)   │    │
│  │ (core/)     │  │  (core/)     │  │  (core/)     │    │
│  └─────────────┘  └──────────────┘  └──────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  RAG Engine (Strategy Context)                  │    │
│  │  (core/rag_engine.py)                           │    │
│  └─────────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│                  External Services Layer                │
│                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  FinBERT    │  │   OpenBB     │  │  ChromaDB    │    │
│  │ (HuggingFace)│ │  (Yahoo Fin) │  │ (Vector DB)  │    │
│  └─────────────┘  └──────────────┘  └──────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Sentence Transformers (Embeddings)             │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Module Breakdown

### Core Modules (`src/core/`)

#### `analyzer.py` - Sentiment Analysis Engine
**Purpose**: AI-powered sentiment analysis of financial text

**Key Components**:
- `SentimentEngine`: Main class for sentiment analysis
- Uses FinBERT model from ProsusAI
- Caching mechanism for model loading

**Dependencies**:
- `transformers` (Hugging Face)
- `torch` (PyTorch backend)

**Flow**:
```
Headlines → Model → Labels + Confidence → Weighted Scores → Average
```

**Sentiment Calculation**:
- Uses FinBERT's confidence scores (0-1) to weight each sentiment
- Formula: `weighted_score = sentiment_label * confidence`
- Provides more nuanced scores instead of binary -1/0/+1 values

#### `data_fetcher.py` - Market Data Client
**Purpose**: Fetch real-time market data from external providers using async operations

**Key Components**:
- `MarketDataClient`: Async wrapper for OpenBB Platform
- Methods: `async fetch_historical_prices()`, `async fetch_company_news()`
- News data is automatically sorted by date descending (newest first)
- **Async Architecture**: Uses `asyncio` for non-blocking I/O operations
- **Parallel Fetching**: Price and news data fetched simultaneously for better performance

**Dependencies**:
- `openbb` (OpenBB Platform)
- `asyncio` (Python standard library)

**Data Sources**:
- Yahoo Finance (primary)
- Extensible to other providers

**Performance**:
- ~2x faster data fetching through parallel price + news retrieval
- Non-blocking executor pattern for synchronous OpenBB SDK calls

#### `processor.py` - Data Processing Utilities
**Purpose**: Transform and calculate financial metrics

**Key Components**:
- `normalize_series()`: Min-max normalization
- `calculate_volatility()`: Annualized volatility from returns

**Dependencies**:
- `pandas`
- `numpy`

#### `rag_engine.py` - RAG Sector News Analysis System
**Purpose**: Retrieval-Augmented Generation for sector-wide market commentary

**Key Components**:
- `RAGEngine`: Main class for sector news retrieval and analysis
- Semantic search over sector-specific news knowledge base
- LLM-powered sector commentary with multiple provider support
- ChromaDB vector database for sector news storage
- Sector mapping system (Technology, Semiconductors, Banking, etc.)

**LLM Providers**:
- OpenAI (GPT-4, GPT-4o-mini)
- Anthropic (Claude-3-sonnet, Claude-3-opus)
- HuggingFace (FLAN-T5-base, FLAN-T5-large, microsoft/phi-2) - **FREE**

**Dependencies**:
- `langchain` (RAG framework)
- `langchain-community` (integrations)
- `chromadb` (vector database)
- `sentence-transformers` (embeddings)
- `openai`, `anthropic`, `transformers` (LLM providers)

**Flow**:
```
Ticker → Sector Mapping → Semantic Query → Vector Search → Sector News → LLM → Market Commentary
```

**Sector News Features**:
- Sector-specific news document retrieval
- Context-aware analysis combining sentiment and sector trends
- Multiple LLM options (paid and free)
- Configurable retrieval parameters (top-k news articles)
- Optional UI toggle (can be disabled)
- Sample news auto-loaded on first run
- Use `--reload-db` flag to force reload: `streamlit run src/app.py -- --reload-db`

---

### Utility Modules (`src/utils/`)

#### `charts.py` - Visualization Components
**Purpose**: Generate interactive Plotly charts

**Key Components**:
- `create_quadrant_plot()`: Sentiment vs. volatility map
- `create_price_chart()`: Candlestick price chart

**Dependencies**:
- `plotly`

**Chart Types**:
- Scatter plots with annotations
- Candlestick (OHLC) charts
- Custom quadrant overlays

#### `pdf_gen.py` - Report Generation
**Purpose**: Create professional PDF reports

**Key Components**:
- `generate_pdf_report()`: Main PDF builder
- Custom formatting and styling

**Dependencies**:
- `fpdf`

**Output**:
- A4 format
- Tables, headers, disclaimers

#### `load_sample_news.py` - Sector News Database Initialization
**Purpose**: Populate RAG vector database with sample sector news

**Key Components**:
- `load_all_sample_news()`: Main initialization function
- Pre-populated news for Technology, Semiconductors, Banking, Energy sectors
- Automatically called on app startup

**Dependencies**:
- `core.rag_engine`

**Features**:
- Sample news for 15+ tickers across 4+ sectors
- Database clearing and initialization
- Sector-specific news organization

---

### Configuration (`src/config.py`)

**Purpose**: Centralized, type-safe configuration hub for all application settings

**Configuration Categories**:
- **LLM Settings**: Model names, temperature, max tokens, retry logic
- **Data Provider Settings**: Default provider selection, API timeouts
- **Analysis Parameters**: Headline counts (current: 5, historical: 15), trading days (30)
- **UI Settings**: Sidebar title, theme colors (blue, not red)
- **Alpha Detection Thresholds**: Sentiment (0.0), Volatility (0.5)
- **Visualization**: Chart height (600px), marker sizes, symbols, colors
- **Quadrant Zones**: 4 named zones with remarks (Alpha, Hype, Danger, Oversight)

**Benefits**:
- Single source of truth for all magic numbers
- Type-safe constants (no hardcoded values scattered in code)
- Easy environment-specific overrides
- Improved maintainability and testability

---

## 🔄 Data Flow

### Complete Analysis Pipeline

```
1. User Input (Tickers, Data Provider, Optional: Enable RAG)
   ↓
2. PARALLEL TICKER ANALYSIS (via asyncio.gather)
   ├─ Ticker 1 ──┐
   ├─ Ticker 2 ──┤  All tickers analyzed simultaneously
   ├─ Ticker 3 ──┤  using asyncio.create_task()
   └─ Ticker N ──┘
   │
   For each ticker (in parallel):
   ├─ 2a. MarketDataClient.fetch_historical_prices() ──┐
   │                                                     ├─ asyncio.gather (parallel)
   └─ 2b. MarketDataClient.fetch_company_news() ────────┘
   ↓
3. SentimentEngine.analyze_headlines(current: 5 headlines)
   ↓
4. SentimentEngine.analyze_headlines(historical: 15 headlines)
   ↓
5. calculate_volatility(prices, 30 trading days)
   ↓
6. Aggregate to DataFrame with alpha flag detection
   ↓
7. [Optional] RAGEngine.get_sector_commentary(ticker, sentiment, k=10, return_sources=True)
   ↓
8. Visualization (Charts + Tables + Alpha Flags + Optional: Sector Commentary + Source Articles)
   ↓
9. Optional: PDF Export with progress tracking
```

**Performance Optimizations**:
- **Parallel Ticker Analysis**: All tickers analyzed simultaneously (~N× speedup for N tickers)
- **Parallel Data Fetching**: Price and news fetched concurrently (~2× speedup per ticker)
- **Progress Tracking**: Real-time progress bars for user feedback
- **Total Speedup**: ~2N× faster for multi-ticker analysis

### Session State Management

Streamlit session state stores:
- `client`: MarketDataClient instance
- `engine`: SentimentEngine instance (with cached model)
- `rag_engine`: RAGEngine instance (if RAG enabled)
- `data`: Results DataFrame with alpha flags
- `cache`: Per-ticker detailed data (prices, news, sentiment, trends)
- `enable_rag`: User preference for RAG sector news analysis
- `llm_provider`: Selected LLM provider (OpenAI/Anthropic/HuggingFace/None)
- `rag_cache`: Cached sector commentary to avoid redundant LLM calls

**Caching Strategy**:
- AI models: `@st.cache_resource` (singleton, shared across sessions)
- User data: Session state (per-session, isolated)
- Config constants: Module-level (loaded once)

---

## 🧪 Testing Architecture

### Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_analyzer.py         # SentimentEngine tests
├── test_processor.py        # Processing utilities tests
├── test_data_fetcher.py     # Async MarketDataClient tests (pytest-asyncio)
├── test_analysis_engine.py  # Async analysis pipeline tests
├── test_rag_engine.py       # RAG engine tests
├── test_charts.py           # Chart generation tests
└── test_pdf_gen.py          # PDF generation tests
```

### Testing Approach

- **Unit Tests**: Individual function testing
- **Integration Tests**: Module interaction testing
- **Async Tests**: Using `pytest-asyncio` for async function testing
- **Mock External APIs**: Avoid live API calls in tests
- **Parallel Execution Tests**: Verify concurrent ticker analysis behavior

### Test Coverage

**Async Components**:
- `test_data_fetcher.py`: Async price/news fetching, parallel data retrieval
- `test_analysis_engine.py`: Async ticker analysis, parallel multi-ticker processing, progress callbacks

**Total Test Cases**: 40+ including async parallel execution scenarios

### CI/CD Pipeline

GitHub Actions workflow:
1. **Lint**: Code style checks (Black, Flake8)
2. **Type Check**: MyPy static analysis
3. **Test**: Pytest with coverage
4. **Build**: Package building
5. **Security**: Bandit, Safety scans

---

## 🗄️ Data Models

### Analysis Result DataFrame

| Column       | Type    | Description                      |
|--------------|---------|----------------------------------|
| `ticker`     | str     | Stock symbol (e.g., "AAPL")      |
| `sentiment`  | float   | Score from -1 to 1               |
| `volatility` | float   | Annualized volatility (0 to 1+)  |
| `trend`      | float   | Sentiment change (current - hist)|
| `alpha_flag` | bool    | Alpha opportunity detected (sentiment > 0.0, volatility > 0.5) |

### Cache Data Structure

```python
{
    'AAPL': {
        'price': DataFrame,          # OHLCV data (30 days)
        'news': DataFrame,           # News articles (current + historical)
        'sent': float,               # Current sentiment (-1 to 1)
        'trend': float,              # Sentiment trend (current - historical)
        'sector_commentary': str     # [Optional] RAG-generated sector analysis
    },
    ...
}
```

---

## 🚀 Performance Considerations

### Optimization Strategies

1. **Model Caching**
   - FinBERT loaded once per session with `@st.cache_resource`
   - RAG embeddings model cached similarly
   - Reduces startup time from 30s to <1s

2. **Async Parallel Processing**
   - **Parallel Ticker Analysis**: All tickers analyzed concurrently using `asyncio.create_task()`
   - **Parallel Data Fetching**: Price + news data fetched simultaneously per ticker
   - **Non-blocking I/O**: Uses `loop.run_in_executor()` for synchronous OpenBB SDK calls
   - **Performance Gain**: ~2N× speedup for N tickers (e.g., 5 tickers: ~10× faster)

3. **Progress Tracking**
   - Real-time progress bars during analysis
   - Accurate completion tracking with shared counters
   - PDF generation progress indicators
   - User-friendly status messages

4. **Batch Processing**
   - Headlines processed in batches (5 current + 15 historical)
   - Leverages GPU if available (CUDA/MPS)

5. **Session State**
   - Avoids re-fetching data on UI interaction
   - Preserves analysis results across reruns
   - Caches per-ticker detailed data for deep dives
   - Sector commentary caching to avoid redundant LLM calls

6. **Lazy Loading**
   - RAG analysis only runs if enabled by user
   - Deep dive data loaded on-demand (tab switching)
   - Charts rendered only for active tabs

7. **Config-Driven Design**
   - All constants in `config.py` reduce runtime lookups
   - Hardcoded values eliminated for performance and maintainability

### Scalability

**Current Capabilities**:
- ✅ Parallel ticker processing (asyncio-based)
- ✅ In-memory data storage
- ✅ Single-user sessions
- ✅ Concurrent data fetching

**Future Enhancements**:
- Database for historical results
- Multi-user deployment with load balancing
- WebSocket streaming for real-time updates

---

## 🔐 Security Architecture

### Data Privacy
- No data persistence by default
- All data in-memory only
- No API keys stored in code

### External Dependencies
- Pinned versions in `requirements.txt`
- Security scans in CI/CD
- Regular dependency updates

### Safe Defaults
- Read-only operations
- No file system writes (except reports)
- Network timeouts configured

---

## 🛠️ Extension Points

### Adding New Data Providers

```python
# In data_fetcher.py
def fetch_historical_prices(self, ticker, provider="new_provider"):
    return obb.equity.price.historical(
        ticker,
        provider=provider
    ).to_df()
```

### Custom Sentiment Models

```python
# In analyzer.py
class CustomSentimentEngine(SentimentEngine):
    def __init__(self, model_name="custom/model"):
        self.model = pipeline("sentiment-analysis", model=model_name)
```

## 📊 Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Frontend** | Streamlit | Rapid prototyping, Python-native, professional UI |
| **AI Model** | FinBERT | Finance-specific, proven accuracy, pre-trained |
| **RAG Framework** | LangChain | Mature ecosystem, multi-LLM support, modular |
| **Vector Database** | ChromaDB | Efficient similarity search, local deployment |
| **Embeddings** | Sentence Transformers | Free, local, high-quality semantic embeddings |
| **LLM Providers** | OpenAI, Anthropic, HuggingFace | Flexibility: paid (quality) + free (cost-effective) options |
| **Data** | OpenBB Platform | Free, comprehensive, multi-provider aggregation |
| **Charts** | Plotly | Interactive, professional, Streamlit-compatible |
| **Testing** | Pytest | Industry standard, extensive ecosystem |
| **Config** | Python module | Type-safe, centralized, zero overhead |
