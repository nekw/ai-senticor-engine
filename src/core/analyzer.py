"""Sentiment Analysis Engine using FinBERT.

This module provides AI-powered sentiment analysis for financial text using
the ProsusAI FinBERT model, specifically trained on financial domain data.
"""


class SentimentEngine:
    """AI-powered sentiment analyzer for financial headlines.

    Uses the ProsusAI FinBERT model to analyze sentiment in financial text.
    The model is loaded once during initialization for efficiency.

    Attributes:
        model: Hugging Face sentiment analysis pipeline with FinBERT.
        score_map: Mapping from sentiment labels to numerical scores.

    Example:
        >>> engine = SentimentEngine()
        >>> headlines = ["Stock prices surge", "Market crashes"]
        >>> score = engine.analyze_headlines(headlines)
        >>> print(f"Average sentiment: {score:.2f}")
        Average sentiment: 0.00
    """

    def __init__(self):
        """Initialize the sentiment engine with FinBERT model."""
        # Lazy loading: model loaded only when first needed
        self._model = None
        self.score_map = {"positive": 1, "neutral": 0, "negative": -1}

    @property
    def model(self):
        """Lazy load the FinBERT model on first access."""
        if self._model is None:
            try:
                import streamlit as st

                with st.spinner("🤖 Loading FinBERT AI model (first-time load)..."):
                    from transformers import pipeline

                    self._model = pipeline(
                        "sentiment-analysis", model="ProsusAI/finbert"
                    )
            except ImportError:
                # Fallback if not in Streamlit context
                print("📥 Loading FinBERT model...")
                from transformers import pipeline

                self._model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                print("✅ FinBERT model loaded successfully")
        return self._model

    def analyze_headlines(self, headlines: list[str]) -> float:
        """Analyze sentiment of financial headlines.

        Processes a list of headlines and returns weighted average sentiment score.
        Uses FinBERT's confidence scores to weight each sentiment classification.

        Args:
            headlines: List of financial news headline strings to analyze.

        Returns:
            Weighted average sentiment score between -1 (most negative) and 1 (most positive).
            Returns 0 if headlines list is empty.

        Example:
            >>> engine = SentimentEngine()
            >>> score = engine.analyze_headlines(["Stock soars to new highs"])
            >>> assert -1 <= score <= 1
        """
        if not headlines:
            return 0

        outputs = self.model(headlines)

        # Use weighted sentiment based on confidence scores
        weighted_scores = []
        for output in outputs:
            label = output["label"]
            confidence = output["score"]  # Model's confidence in the prediction
            sentiment_value = self.score_map[label]

            # Weight the sentiment by confidence for more nuanced scoring
            weighted_score = sentiment_value * confidence
            weighted_scores.append(weighted_score)

        return sum(weighted_scores) / len(weighted_scores)

    def get_trade_recommendation(self, sentiment, vol_norm):
        """Map quadrant coordinates to specific trading actions."""
        if sentiment > 0 and vol_norm < 0.5:
            return {
                "action": "ACCUMULATE / BUY",
                "color": "green",
                "rationale": (
                    "High sentiment with low risk suggests institutional "
                    "accumulation. High conviction entry."
                ),
            }
        elif sentiment > 0 and vol_norm >= 0.5:
            return {
                "action": "PROFIT TAKING / TIGHTEN STOPS",
                "color": "blue",
                "rationale": (
                    "High hype and high volatility. Risk of a 'blow-off top' "
                    "is elevated. Protect your gains."
                ),
            }
        elif sentiment <= 0 and vol_norm >= 0.5:
            return {
                "action": "AVOID / SHORT",
                "color": "orange",
                "rationale": (
                    "Negative sentiment paired with panic-level volatility. "
                    "High risk of capital loss."
                ),
            }
        else:  # Oversight Zone
            return {
                "action": "NEUTRAL / WATCHLIST",
                "color": "gray",
                "rationale": (
                    "Market interest is low and sentiment is stagnant. "
                    "Wait for a new news catalyst."
                ),
            }
