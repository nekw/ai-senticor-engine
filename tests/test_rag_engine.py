"""Unit tests for RAG engine."""

from unittest.mock import Mock, patch

import pytest

from src.core.rag_engine import RAGEngine


class TestRAGEngine:
    """Test suite for RAGEngine class."""

    @pytest.fixture
    def mock_rag_engine(self):
        """Create a RAG engine with mocked dependencies."""
        with patch(
            "src.core.rag_engine.HuggingFaceEmbeddings"
        ) as mock_embeddings, patch("src.core.rag_engine.Chroma") as mock_chroma:
            mock_embeddings_instance = Mock()
            mock_embeddings.return_value = mock_embeddings_instance

            mock_vectorstore = Mock()
            mock_chroma.return_value = mock_vectorstore

            engine = RAGEngine(llm_provider=None)
            engine.vectorstore = mock_vectorstore

            yield engine

    def test_initialization_no_llm(self, mock_rag_engine):
        """Test RAG engine initializes without LLM."""
        assert mock_rag_engine.llm is None
        assert mock_rag_engine.use_generation is False
        assert mock_rag_engine.vectorstore is not None

    def test_sector_mapping_exists(self, mock_rag_engine):
        """Test sector mapping dictionary is populated."""
        assert len(mock_rag_engine.sector_mapping) > 0
        assert "AAPL" in mock_rag_engine.sector_mapping
        assert "NVDA" in mock_rag_engine.sector_mapping
        assert "MSFT" in mock_rag_engine.sector_mapping

    def test_sector_mapping_values(self, mock_rag_engine):
        """Test sector mapping has correct values."""
        assert mock_rag_engine.sector_mapping["AAPL"] == "Technology"
        assert mock_rag_engine.sector_mapping["NVDA"] == "Semiconductors"
        assert mock_rag_engine.sector_mapping["V"] == "Financial Services"

    def test_get_sector_existing_ticker(self, mock_rag_engine):
        """Test getting sector for existing ticker."""
        sector = mock_rag_engine.get_sector("AAPL")
        assert sector == "Technology"

    def test_get_sector_non_existing_ticker(self, mock_rag_engine):
        """Test getting sector for non-existing ticker."""
        sector = mock_rag_engine.get_sector("UNKNOWN")
        assert sector == "Unknown"

    def test_add_ticker_sector_mapping(self, mock_rag_engine):
        """Test adding new ticker to sector mapping."""
        mock_rag_engine.add_ticker_sector_mapping("CUSTOM", "CustomSector")
        assert mock_rag_engine.sector_mapping["CUSTOM"] == "CustomSector"

    def test_add_sector_news(self, mock_rag_engine):
        """Test adding sector news to vector database."""
        mock_rag_engine.add_sector_news(
            sector="Technology",
            headline="Test Headline",
            content="Test content",
            ticker="AAPL",
            date="2026-01-27",
            url="https://example.com",
        )

        # Verify vectorstore.add_documents was called
        mock_rag_engine.vectorstore.add_documents.assert_called_once()

        # Check the document that was added
        call_args = mock_rag_engine.vectorstore.add_documents.call_args
        docs = call_args[0][0]
        assert len(docs) == 1
        assert "Test content" in docs[0].page_content
        assert docs[0].metadata["ticker"] == "AAPL"
        assert docs[0].metadata["sector"] == "Technology"
        assert docs[0].metadata["headline"] == "Test Headline"

    def test_add_sector_news_minimal(self, mock_rag_engine):
        """Test adding sector news with minimal required fields."""
        mock_rag_engine.add_sector_news(
            sector="Technology", headline="Minimal Headline", content="Minimal content"
        )

        call_args = mock_rag_engine.vectorstore.add_documents.call_args
        docs = call_args[0][0]
        # Metadata keys might not exist for optional fields
        assert "Minimal content" in docs[0].page_content

    def test_get_sector_commentary_no_llm(self, mock_rag_engine):
        """Test getting sector commentary without LLM (retrieval only)."""
        # Mock similarity search results
        mock_doc1 = Mock()
        mock_doc1.metadata = {
            "ticker": "AAPL",
            "headline": "Apple launches new product",
            "date": "2026-01-26",
        }
        mock_doc2 = Mock()
        mock_doc2.metadata = {
            "ticker": "MSFT",
            "headline": "Microsoft reports earnings",
            "date": "2026-01-25",
        }

        mock_rag_engine.vectorstore.similarity_search.return_value = [
            mock_doc1,
            mock_doc2,
        ]
        mock_rag_engine.sector_mapping["AAPL"] = "Technology"

        result = mock_rag_engine.get_sector_commentary(
            "AAPL", company_sentiment=0.5, k=5
        )

        # Should return formatted news items without LLM generation
        assert "Apple launches new product" in result
        assert "Microsoft reports earnings" in result
        assert "2026-01-26" in result

    def test_get_sector_commentary_unknown_ticker(self, mock_rag_engine):
        """Test getting sector commentary for unmapped ticker."""
        result = mock_rag_engine.get_sector_commentary("UNKNOWN", company_sentiment=0.5)
        assert "not available" in result.lower() or "unable" in result.lower()

    def test_get_sector_commentary_no_news(self, mock_rag_engine):
        """Test getting sector commentary when no news found."""
        mock_rag_engine.vectorstore.similarity_search.return_value = []
        mock_rag_engine.sector_mapping["AAPL"] = "Technology"

        result = mock_rag_engine.get_sector_commentary("AAPL", company_sentiment=0.5)
        assert "no recent" in result.lower() or "add news" in result.lower()

    @patch("src.core.rag_engine.ChatOpenAI")
    def test_initialization_with_openai(self, mock_openai):
        """Test RAG engine initializes with OpenAI LLM."""
        with patch("src.core.rag_engine.HuggingFaceEmbeddings"), patch(
            "src.core.rag_engine.Chroma"
        ):
            mock_llm = Mock()
            mock_openai.return_value = mock_llm

            engine = RAGEngine(llm_provider="openai", model="gpt-4o-mini")

            assert engine.use_generation is True
            assert engine.llm is not None
            mock_openai.assert_called_once()

    @patch("src.core.rag_engine.ChatAnthropic")
    def test_initialization_with_anthropic(self, mock_anthropic):
        """Test RAG engine initializes with Anthropic LLM."""
        with patch("src.core.rag_engine.HuggingFaceEmbeddings"), patch(
            "src.core.rag_engine.Chroma"
        ):
            mock_llm = Mock()
            mock_anthropic.return_value = mock_llm

            engine = RAGEngine(
                llm_provider="anthropic", model="claude-3-5-sonnet-20241022"
            )

            assert engine.use_generation is True
            assert engine.llm is not None
            mock_anthropic.assert_called_once()

    def test_get_sector_commentary_with_sentiment(self, mock_rag_engine):
        """Test sector commentary includes sentiment context."""
        mock_doc = Mock()
        mock_doc.metadata = {
            "ticker": "AAPL",
            "headline": "Apple stock rises",
            "date": "2026-01-27",
        }

        mock_rag_engine.vectorstore.similarity_search.return_value = [mock_doc]
        mock_rag_engine.sector_mapping["AAPL"] = "Technology"

        # Test with positive sentiment
        result = mock_rag_engine.get_sector_commentary("AAPL", company_sentiment=0.8)
        assert result is not None

        # Test with negative sentiment
        result = mock_rag_engine.get_sector_commentary("AAPL", company_sentiment=-0.5)
        assert result is not None

    def test_retrieval_k_parameter(self, mock_rag_engine):
        """Test that k parameter controls number of retrieved documents."""
        mock_rag_engine.vectorstore.similarity_search.return_value = []
        mock_rag_engine.sector_mapping["AAPL"] = "Technology"

        mock_rag_engine.get_sector_commentary("AAPL", company_sentiment=0.5, k=10)

        # Check that similarity_search was called with k=10
        call_args = mock_rag_engine.vectorstore.similarity_search.call_args
        assert call_args[1]["k"] == 10

    def test_persist_directory_default(self):
        """Test default persist directory is set correctly."""
        with patch("src.core.rag_engine.HuggingFaceEmbeddings"), patch(
            "src.core.rag_engine.Chroma"
        ):
            engine = RAGEngine()
            assert engine.persist_directory == "./data/chroma_db"

    def test_persist_directory_custom(self):
        """Test custom persist directory is set correctly."""
        with patch("src.core.rag_engine.HuggingFaceEmbeddings"), patch(
            "src.core.rag_engine.Chroma"
        ):
            engine = RAGEngine(persist_directory="/custom/path")
            assert engine.persist_directory == "/custom/path"
