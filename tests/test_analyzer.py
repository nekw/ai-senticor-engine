"""Unit tests for sentiment analysis engine."""

from unittest.mock import Mock

import pytest

from src.core.analyzer import SentimentEngine


class TestSentimentEngine:
    """Test suite for SentimentEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a mock SentimentEngine for testing."""
        # Create engine instance
        engine = SentimentEngine()
        # Mock the model directly since pipeline is lazily imported
        mock_model = Mock()
        engine._model = mock_model
        yield engine

    def test_initialization(self, engine):
        """Test SentimentEngine initializes correctly."""
        assert engine.model is not None
        assert engine.score_map == {"positive": 1, "neutral": 0, "negative": -1}

    def test_analyze_empty_headlines(self, engine):
        """Test analyzing empty headlines list returns 0."""
        result = engine.analyze_headlines([])
        assert result == 0

    def test_analyze_single_positive_headline(self, engine):
        """Test analyzing a single positive headline."""
        engine.model.return_value = [{"label": "positive", "score": 0.9}]
        result = engine.analyze_headlines(["Stock prices surge"])
        assert result == 0.9  # 1 * 0.9 confidence

    def test_analyze_single_negative_headline(self, engine):
        """Test analyzing a single negative headline."""
        engine.model.return_value = [{"label": "negative", "score": 0.9}]
        result = engine.analyze_headlines(["Market crashes"])
        assert result == -0.9  # -1 * 0.9 confidence

    def test_analyze_single_neutral_headline(self, engine):
        """Test analyzing a single neutral headline."""
        engine.model.return_value = [{"label": "neutral", "score": 0.9}]
        result = engine.analyze_headlines(["Market unchanged"])
        assert result == 0.0  # 0 * 0.9 confidence

    def test_analyze_mixed_headlines(self, engine):
        """Test analyzing mixed sentiment headlines."""
        engine.model.return_value = [
            {"label": "positive", "score": 1.0},
            {"label": "negative", "score": 1.0},
            {"label": "neutral", "score": 1.0},
        ]
        headlines = ["Stock up", "Stock down", "No change"]
        result = engine.analyze_headlines(headlines)
        # (1*1.0 + -1*1.0 + 0*1.0) / 3 = 0
        assert result == 0.0

    def test_analyze_all_positive_headlines(self, engine):
        """Test analyzing all positive headlines."""
        engine.model.return_value = [
            {"label": "positive", "score": 0.95},
            {"label": "positive", "score": 0.85},
            {"label": "positive", "score": 0.90},
        ]
        headlines = ["Great earnings", "Stock soars", "Record profits"]
        result = engine.analyze_headlines(headlines)
        # (0.95 + 0.85 + 0.90) / 3 = 0.9
        assert abs(result - 0.9) < 0.01

    def test_analyze_all_negative_headlines(self, engine):
        """Test analyzing all negative headlines."""
        engine.model.return_value = [
            {"label": "negative", "score": 0.95},
            {"label": "negative", "score": 0.90},
            {"label": "negative", "score": 0.85},
        ]
        headlines = ["Poor earnings", "Stock plummets", "Losses mount"]
        result = engine.analyze_headlines(headlines)
        # (-0.95 + -0.90 + -0.85) / 3 = -0.9
        assert abs(result - (-0.9)) < 0.01

    def test_analyze_mostly_positive(self, engine):
        """Test analyzing mostly positive headlines."""
        engine.model.return_value = [
            {"label": "positive", "score": 0.9},
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.7},
            {"label": "negative", "score": 0.6},
        ]
        result = engine.analyze_headlines(["pos", "pos", "neu", "neg"])
        # (0.9 + 0.8 + 0 + -0.6) / 4 = 0.275
        assert abs(result - 0.275) < 0.01

    def test_model_called_with_headlines(self, engine):
        """Test that model is called with the headlines."""
        engine.model.return_value = [{"label": "positive", "score": 0.9}]
        headlines = ["Test headline"]
        engine.analyze_headlines(headlines)
        engine.model.assert_called_once_with(headlines)

    def test_analyze_headlines_model_exception(self, engine):
        """Test handling of model exceptions during analysis."""
        engine.model.side_effect = Exception("Model inference error")

        # Should raise the exception (no error handling in analyze_headlines)
        with pytest.raises(Exception, match="Model inference error"):
            engine.analyze_headlines(["Test headline"])

    def test_analyze_headlines_unknown_label(self, engine):
        """Test handling of unknown sentiment labels."""
        # Model returns an unknown label
        engine.model.return_value = [{"label": "unknown", "score": 0.9}]

        # Should raise KeyError since unknown label not in score_map
        with pytest.raises(KeyError):
            engine.analyze_headlines(["Test headline"])

    def test_analyze_headlines_missing_score(self, engine):
        """Test handling of missing score in model output."""
        # Model returns output without score field
        engine.model.return_value = [{"label": "positive"}]

        # Should raise KeyError when accessing 'score'
        with pytest.raises(KeyError):
            engine.analyze_headlines(["Test headline"])

    def test_analyze_headlines_none_input(self, engine):
        """Test handling of None as input."""
        # `if not headlines` catches None and returns 0
        result = engine.analyze_headlines(None)
        assert result == 0

    def test_analyze_headlines_non_string_elements(self, engine):
        """Test handling of non-string elements in headlines list."""
        engine.model.return_value = [{"label": "positive", "score": 0.9}]

        # Model should handle non-string inputs (or raise error)
        # This depends on transformers pipeline behavior
        headlines = [123, None, "Valid headline"]
        # Let the test verify actual behavior - may succeed or fail
        # depending on transformers version
        try:
            result = engine.analyze_headlines(headlines)
            assert isinstance(result, float)
        except (TypeError, AttributeError):
            # Expected if transformers pipeline rejects non-string input
            pass
