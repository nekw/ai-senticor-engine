"""Typed state objects for competitive analysis multi-agent workflows."""

from dataclasses import asdict, dataclass, field
from datetime import datetime


@dataclass(slots=True)
class CompetitiveQuery:
    """Input parameters for a competitive analysis run."""

    focus_ticker: str
    sector: str | None = None
    lookback_days: int = 30
    max_competitors: int = 6
    max_articles_per_ticker: int = 15


@dataclass(slots=True)
class CompetitorSignal:
    """Computed metrics for one competitor."""

    ticker: str
    sentiment: float
    sentiment_trend: float
    volatility: float
    news_count: int


@dataclass(slots=True)
class RiskAssessment:
    """Confidence and risk flags for autonomous outputs."""

    confidence: float
    flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CompetitiveAnalysisState:
    """Shared mutable state passed between specialized agents."""

    query: CompetitiveQuery
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    sector: str | None = None
    peer_universe: list[str] = field(default_factory=list)
    news_by_ticker: dict[str, list[str]] = field(default_factory=dict)
    signals: list[CompetitorSignal] = field(default_factory=list)
    narrative: str = ""
    citations: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    risk: RiskAssessment | None = None

    def to_dict(self) -> dict:
        """Convert full analysis state into JSON-serializable primitives."""
        return asdict(self)
