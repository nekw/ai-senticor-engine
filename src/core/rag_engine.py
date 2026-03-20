"""RAG Engine for sector news analysis and commentary.

This module provides a Retrieval-Augmented Generation system that retrieves
relevant news from companies in the same sector and generates LLM-powered
sector-wide market commentary and insights.
"""

import os
import random
import shutil
from datetime import datetime
from hashlib import sha256
from typing import List, Optional, Tuple, Union

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils.logger import AppLogger


class _HashEmbeddings:
    """Lightweight deterministic embedding fallback (no torch dependencies)."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        seed = hash(text)
        rng = random.Random(seed)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(self.dimension)]

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


class _ResilientEmbeddings:
    """Wrap primary embeddings and fallback on runtime backend/device errors."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._fallback_enabled = False

    def _should_fallback(self, error: Exception) -> bool:
        error_text = str(error)
        return "meta tensor" in error_text.lower() or "to_empty()" in error_text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._fallback_enabled:
            return self.fallback.embed_documents(texts)

        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            if self._should_fallback(e):
                self._fallback_enabled = True
                AppLogger.warning(
                    "RAG embeddings fallback",
                    "Primary embeddings failed with device/meta error; "
                    "switching to hash embeddings.",
                )
                return self.fallback.embed_documents(texts)
            raise

    def embed_query(self, text: str) -> List[float]:
        if self._fallback_enabled:
            return self.fallback.embed_query(text)

        try:
            return self.primary.embed_query(text)
        except Exception as e:
            if self._should_fallback(e):
                self._fallback_enabled = True
                AppLogger.warning(
                    "RAG embeddings fallback",
                    "Primary query embeddings failed with device/meta error; "
                    "switching to hash embeddings.",
                )
                return self.fallback.embed_query(text)
            raise


class RAGEngine:
    """Sector News Analysis System using RAG.

    Stores and retrieves sector-specific news articles to provide comprehensive
    sector-wide market commentary and insights using LLM generation.

    Attributes:
        embeddings: HuggingFace embeddings model for text vectorization.
        vectorstore: Chroma vector database for sector news articles.
        llm: Optional LLM for generation (OpenAI GPT-4, Anthropic Claude, etc.).
        use_generation: Whether to use LLM generation or retrieval-only.
        persist_directory: Directory path for vector database persistence.
        sector_mapping: Dictionary mapping tickers to sectors.

    Example:
        >>> # Initialize sector news analysis system
        >>> rag = RAGEngine(llm_provider="openai", model="gpt-4o-mini")
        >>>
        >>> # Add sector news
        >>> rag.add_sector_news(
        ...     sector="Technology",
        ...     headline="Apple unveils new AI chips",
        ...     content="Apple announced...",
        ...     ticker="AAPL",
        ...     date="2026-01-26"
        ... )
        >>>
        >>> # Get sector commentary
        >>> commentary = rag.get_sector_commentary(
        ...     ticker="AAPL",
        ...     company_sentiment=0.75
        ... )
    """

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        llm_provider: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        """Initialize RAG engine with vector stores and optional LLM.

        Args:
            persist_directory: Path to persist the vector database.
            llm_provider: LLM provider ("openai", "anthropic",
                "huggingface", or None for retrieval-only).
            model: Model name (e.g., "gpt-4", "claude-3-sonnet-20240229",
                "google/flan-t5-base").
            temperature: LLM temperature for generation (0-1,
                lower = more focused).
        """
        self.persist_directory = persist_directory
        self.use_generation = False
        self.llm = None

        # Sector mapping for major companies
        self.sector_mapping = {
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

        # Initialize LLM if provider specified
        if llm_provider:
            try:
                if llm_provider.lower() == "openai":
                    self.llm = ChatOpenAI(model=model, temperature=temperature)
                elif llm_provider.lower() == "anthropic":
                    self.llm = ChatAnthropic(model=model, temperature=temperature)
                elif llm_provider.lower() == "huggingface":
                    # Load HuggingFace model locally (free, no API key needed)
                    # Security: Pin to 'main' revision and disable remote code execution
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(
                        model, revision="main", trust_remote_code=False
                    )
                    self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                        model,
                        revision="main",
                        trust_remote_code=False,
                        low_cpu_mem_usage=False,
                        device_map=None,
                    )
                    self.hf_temperature = temperature
                    self.llm = "huggingface"  # Flag to use custom generation
                    print(f"✓ RAG with HuggingFace generation enabled: {model}")
                else:
                    print(
                        "Warning: Unknown LLM provider '{}'. Using "
                        "retrieval-only mode.".format(llm_provider)
                    )

                if self.llm:
                    self.use_generation = True
                    if llm_provider.lower() in ["openai", "anthropic"]:
                        print(f"✓ RAG with generation enabled: {llm_provider} {model}")
            except Exception as e:
                print(
                    f"Warning: Failed to initialize LLM: {e}. Using retrieval-only mode."
                )

        # Use HuggingFace embeddings (free, no API key needed).
        # If torch/transformers device state is broken (meta tensor errors),
        # gracefully degrade to deterministic hash embeddings so core flows keep working.
        self.embeddings: Union[_ResilientEmbeddings, _HashEmbeddings]
        fallback_embeddings = _HashEmbeddings(dimension=384)
        try:
            primary_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self.embeddings = _ResilientEmbeddings(
                primary=primary_embeddings,
                fallback=fallback_embeddings,
            )
        except Exception as e:
            AppLogger.warning(
                "RAG embeddings init fallback",
                f"HuggingFace embeddings init failed, using hash fallback: {e}",
            )
            self.embeddings = fallback_embeddings

        # Initialize or load Chroma vector store for sector news.
        # If tenant metadata is corrupted, auto-heal by recreating local DB files.
        self.vectorstore = self._create_vectorstore_with_recovery()

    def get_sector(self, ticker: str) -> str:
        """Get sector for a given ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Sector name or "Unknown" if not found.
        """
        return self.sector_mapping.get(ticker.upper(), "Unknown")

    def add_sector_news(
        self,
        sector: str,
        headline: str,
        content: str,
        ticker: Optional[str] = None,
        date: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """Add news article to sector knowledge base.

        Args:
            sector: Sector/industry (e.g., "Technology", "Banking").
            headline: News headline.
            content: Full news content or summary.
            ticker: Related ticker symbol (optional).
            date: Publication date (optional).
            url: Source URL (optional).

        Example:
            >>> rag.add_sector_news(
            ...     sector="Technology",
            ...     headline="Apple unveils new AI chips",
            ...     content="Apple announced breakthrough AI processors...",
            ...     ticker="AAPL",
            ...     date="2026-01-26"
            ... )
        """
        # Build document content
        doc_content = f"**{headline}**\n\n{content}"

        resolved_date = date or datetime.now().strftime("%Y-%m-%d")

        # Build metadata
        metadata = {
            "sector": sector,
            "headline": headline,
            "date": resolved_date,
            "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if ticker:
            metadata["ticker"] = ticker.upper()
        if url:
            metadata["url"] = url

        doc_id = self._build_sector_news_id(
            sector=sector,
            headline=headline,
            ticker=ticker,
            date=resolved_date,
        )

        # Upsert by deterministic ID so repeated loads do not create duplicates.
        doc = Document(page_content=doc_content, metadata=metadata)
        try:
            self.vectorstore.add_documents([doc], ids=[doc_id])
        except Exception as e:
            if not self._is_tenant_connection_error(e):
                raise

            AppLogger.warning(
                "RAG insert recovery",
                "Tenant connection failed during insert. Recreating local vector DB and retrying.",
            )
            self.vectorstore = self._create_vectorstore_with_recovery()
            self.vectorstore.add_documents([doc], ids=[doc_id])

    def _build_sector_news_id(
        self,
        sector: str,
        headline: str,
        ticker: Optional[str],
        date: Optional[str],
    ) -> str:
        """Build deterministic document ID for sector-news deduplication."""
        raw = "|".join(
            [
                str(sector or "").strip().lower(),
                str(ticker or "").strip().upper(),
                str(date or "").strip(),
                str(headline or "").strip().lower(),
            ]
        )
        return sha256(raw.encode("utf-8")).hexdigest()

    def _build_vectorstore(self) -> Chroma:
        """Create Chroma vector store instance for sector news collection."""
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="sector_news",
        )

    def _is_tenant_connection_error(self, error: Exception) -> bool:
        """Return True when exception indicates corrupted/missing Chroma tenant metadata."""
        text = str(error)
        return "default_tenant" in text or "Could not connect to tenant" in text

    def _create_vectorstore_with_recovery(self) -> Chroma:
        """Create vector store and auto-recover known tenant metadata corruption."""
        try:
            return self._build_vectorstore()
        except Exception as e:
            if not self._is_tenant_connection_error(e):
                raise

            AppLogger.warning(
                "RAG vectorstore recovery",
                "Detected Chroma tenant metadata issue. Recreating local vector DB.",
            )
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory, ignore_errors=True)
            os.makedirs(self.persist_directory, exist_ok=True)
            return self._build_vectorstore()

    def get_sector_commentary(
        self,
        ticker: str,
        company_sentiment: float,
        k: int = 10,
        return_sources: bool = False,
    ) -> Union[str, Tuple[str, List[dict]]]:
        """Generate sector-wide commentary based on recent news.

        Retrieves relevant sector news and uses LLM to generate comprehensive
        sector market commentary and insights.

        Args:
            ticker: Stock ticker to determine sector.
            company_sentiment: Company's current sentiment score (-1 to 1).
            k: Number of sector news articles to retrieve.
            return_sources: If True, return tuple of (commentary, news_articles).

        Returns:
            LLM-generated sector commentary or formatted news summary.
            If return_sources=True, returns (commentary, list of news dicts).

        Example:
            >>> rag = RAGEngine(llm_provider="openai")
            >>> commentary = rag.get_sector_commentary("AAPL", 0.75)
            >>> print(commentary)
        """
        sector = self.get_sector(ticker)

        if sector == "Unknown":
            AppLogger.warning(
                "RAG sector unknown", f"No sector mapping for {ticker}", ticker=ticker
            )
            result = (
                "Sector information not available for {}. Unable to provide "
                "sector commentary.".format(ticker)
            )
            return (result, []) if return_sources else result

        AppLogger.info(
            "RAG retrieval",
            "Retrieving {} news articles for {} sector".format(k, sector),
            ticker=ticker,
        )

        # Query for sector-specific news
        query = f"{sector} sector market news trends analysis outlook"

        # Retrieve relevant sector news
        try:
            results = self.vectorstore.similarity_search(
                query, k=k, filter={"sector": sector} if sector != "Unknown" else None
            )
            AppLogger.success(
                "RAG retrieval complete",
                f"Retrieved {len(results)} articles for {sector}",
                ticker=ticker,
            )
        except Exception as e:
            AppLogger.warning(
                "RAG filter failed",
                f"Falling back to unfiltered search: {str(e)}",
                ticker=ticker,
            )
            results = self.vectorstore.similarity_search(query, k=k)

        if not results:
            AppLogger.warning(
                "RAG no results", f"No news found for {sector} sector", ticker=ticker
            )
            result = (
                "No recent sector news available for {}. Add news articles to "
                "enable sector analysis.".format(sector)
            )
            return (result, []) if return_sources else result

        # Extract news articles for return
        news_articles = []
        if return_sources:
            for doc in results:
                meta = doc.metadata
                news_articles.append(
                    {
                        "headline": meta.get("headline", "No headline"),
                        "date": meta.get("date", "Unknown date"),
                        "ticker": meta.get("ticker", ""),
                        "content": doc.page_content[:300] + "..."
                        if len(doc.page_content) > 300
                        else doc.page_content,
                    }
                )

        # Use LLM generation if available
        if self.use_generation and self.llm:
            AppLogger.info(
                "RAG generation",
                "Generating commentary using {}".format(
                    self.llm if isinstance(self.llm, str) else type(self.llm).__name__
                ),
                ticker=ticker,
            )
            commentary = self._generate_sector_commentary(
                ticker, sector, company_sentiment, results
            )
            AppLogger.success(
                "RAG commentary generated",
                f"Generated sector analysis for {sector}",
                ticker=ticker,
            )
            return (commentary, news_articles) if return_sources else commentary

        # Fallback: retrieval-only formatting
        AppLogger.info(
            "RAG fallback mode",
            "Using retrieval-only formatting (no LLM)",
            ticker=ticker,
        )
        news_items = []
        for i, doc in enumerate(results[:5], 1):
            meta = doc.metadata
            headline = meta.get("headline", "No headline")
            date = meta.get("date", "Unknown date")
            related_ticker = meta.get("ticker", "")

            news_items.append(f"{i}. **{headline}** ({date})")
            if related_ticker:
                news_items.append(f"   Related: {related_ticker}")

        summary_parts = [
            f"**📰 {sector} Sector News Summary**\n",
            f"Recent developments affecting {ticker} and the {sector} sector:\n",
            "\n".join(news_items),
            f"\n*{len(results)} sector news articles analyzed*",
        ]

        commentary = "\n".join(summary_parts)
        return (commentary, news_articles) if return_sources else commentary

    def _generate_sector_commentary(
        self, ticker: str, sector: str, company_sentiment: float, retrieved_docs: List
    ) -> str:
        """Generate sector commentary using LLM and retrieved news.

        Args:
            ticker: Stock ticker.
            sector: Sector name.
            company_sentiment: Company sentiment score.
            retrieved_docs: Retrieved news documents.

        Returns:
            LLM-generated sector commentary.
        """
        # Extract news summaries
        news_summaries = []
        for i, doc in enumerate(retrieved_docs[:10], 1):
            meta = doc.metadata
            headline = meta.get("headline", "")
            date = meta.get("date", "")
            related_ticker = meta.get("ticker", "")
            content_preview = doc.page_content[:200]

            news_summaries.append(
                f"{i}. {headline} ({date}) [{related_ticker}]: {content_preview}..."
            )

        news_context = "\n".join(news_summaries)
        sentiment_label = "Positive" if company_sentiment > 0 else "Negative"

        # HuggingFace model (local generation)
        if self.llm == "huggingface":
            # FLAN-T5 works better with simple Q&A format
            prompt_text = (
                "Context: Recent {} sector news shows {} competitors: {}."
                "\n\nKey developments:\n{}\n\nBased on this {} sector news, "
                "write a 3-paragraph market analysis covering: "
                "1) Overall sector trends, 2) Impact on {} (sentiment: {}), "
                "3) Outlook and risks."
            ).format(
                sector,
                ticker,
                ", ".join(
                    [doc.metadata.get("ticker", "") for doc in retrieved_docs[:5]]
                ),
                news_context[:600],
                sector,
                ticker,
                sentiment_label,
            )

            inputs = self.hf_tokenizer(
                prompt_text, return_tensors="pt", max_length=512, truncation=True
            )

            # Generate with better parameters
            outputs = self.hf_model.generate(
                **inputs,
                max_length=350,
                min_length=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
            )

            response = self.hf_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Format with header
            return f"**📰 {sector} Sector Commentary**\n\n{response}"

        # API-based models (OpenAI, Anthropic)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a financial markets analyst specializing in "
                        "sector-wide analysis. Analyze recent sector news to "
                        "provide comprehensive commentary on market trends, key "
                        "developments, competitive dynamics, and outlook. Focus "
                        "on actionable insights.",
                    ),
                    (
                        "user",
                        """Sector: {sector}
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

                Keep response concise but insightful (250-350 words).""",
                    ),
                ]
            )

            chain = prompt | self.llm
            response = chain.invoke(
                {
                    "sector": sector,
                    "ticker": ticker,
                    "company_sentiment": company_sentiment,
                    "sentiment_label": sentiment_label,
                    "news_context": news_context,
                }
            )

            return f"**📰 {sector} Sector Commentary**\n\n{response.content}"

    def _format_fallback_commentary(
        self,
        ticker: str,
        sector: str,
        company_sentiment: float,
        news_summaries: List[str],
    ) -> str:
        """Format fallback commentary when LLM generation fails.

        Args:
            ticker: Stock ticker.
            sector: Sector name.
            company_sentiment: Company sentiment score.
            news_summaries: List of news summary strings.

        Returns:
            Formatted sector commentary.
        """
        sentiment_icon = (
            "🟢" if company_sentiment > 0.3 else "🔴" if company_sentiment < -0.3 else "🟡"
        )
        sentiment_label = (
            "Positive"
            if company_sentiment > 0.3
            else "Negative"
            if company_sentiment < -0.3
            else "Neutral"
        )

        # Format news summaries in expandable section
        news_details = "\n".join(news_summaries[:5])

        commentary_parts = [
            "**📰 {} Sector News Analysis**\n".format(sector),
            ("**Company:** {} | **Sentiment:** {} {} " "({:+.2f})\n").format(
                ticker, sentiment_icon, sentiment_label, company_sentiment
            ),
            "**Sector Context:**",
            (
                "The {} sector is experiencing multiple developments that may "
                "impact {}."
            ).format(sector, ticker),
            (
                "Recent news from sector peers ({}) provides context for "
                "market dynamics and competitive positioning."
            ).format(
                ", ".join(
                    set(
                        [
                            s.split("[")[1].split("]")[0]
                            for s in news_summaries[:3]
                            if "[" in s
                        ]
                    )
                )
            ),
            "\n**{} Position:**".format(ticker),
            (
                "Given the {} sentiment ({:+.2f}), {} is positioned {} within "
                "the current {} market environment."
            ).format(
                sentiment_label.lower(),
                company_sentiment,
                ticker,
                "favorably" if company_sentiment > 0 else "cautiously",
                sector,
            ),
            "<details>",
            "<summary><b>📰 Recent {} Sector News (click to expand)</b></summary>\n".format(
                sector
            ),
            news_details,
            "</details>\n",
        ]

        return "\n".join(commentary_parts)

    def database_exists(self) -> bool:
        """Check if the vector database directory exists.

        Returns:
            True if database directory exists, False otherwise.
        """
        return os.path.exists(self.persist_directory)

    def clear_vector_database(self):
        """Clear all documents from the vector database.

        Warning: This will permanently delete all stored sector news articles.
        """
        try:
            self.vectorstore.delete_collection()

            self.vectorstore = self._create_vectorstore_with_recovery()

            # Verify the recreated collection is empty.
            remaining = self.vectorstore.get()
            remaining_ids = (
                remaining.get("ids", []) if isinstance(remaining, dict) else []
            )
            if remaining_ids:
                raise RuntimeError(
                    "Vector database clear verification failed: collection is not empty"
                )

            AppLogger.success("RAG clear", "Vector database cleared successfully")
        except Exception as e:
            AppLogger.error("RAG clear failed", str(e))
            raise RuntimeError(f"Failed to clear vector database: {e}") from e

    def add_ticker_sector_mapping(self, ticker: str, sector: str):
        """Add or update ticker-to-sector mapping.

        Args:
            ticker: Stock ticker symbol.
            sector: Sector/industry name.
        """
        self.sector_mapping[ticker.upper()] = sector

    def get_all_sectors(self) -> List[str]:
        """Get list of all unique sectors.

        Returns:
            List of sector names.
        """
        return sorted(set(self.sector_mapping.values()))

    def get_sector_tickers(self, sector: str) -> List[str]:
        """Get all tickers in a given sector.

        Args:
            sector: Sector name.

        Returns:
            List of ticker symbols.
        """
        return [ticker for ticker, sec in self.sector_mapping.items() if sec == sector]
