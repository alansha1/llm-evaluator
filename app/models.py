"""SQLAlchemy ORM models."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from app.database import Base


class EvalRun(Base):
    """A single evaluation run — one prompt tested against one or more providers."""
    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, unique=True, index=True)
    prompt = Column(Text, nullable=False)
    category = Column(String, nullable=False)          # factual / safety / consistency / open
    reference_answer = Column(Text, nullable=True)     # ground truth for accuracy scoring
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("EvalResult", back_populates="run", cascade="all, delete-orphan")


class EvalResult(Base):
    """Scores for one provider's response on a given EvalRun."""
    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String, ForeignKey("eval_runs.run_id"), index=True)
    provider = Column(String, nullable=False)          # openai / huggingface / mock

    response_text = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=False)

    # Metric scores (0.0 – 1.0)
    accuracy_score = Column(Float, nullable=True)       # ROUGE-L vs reference answer
    safety_score = Column(Float, nullable=False)        # 1.0 = safe, 0.0 = unsafe
    consistency_score = Column(Float, nullable=True)    # similarity across repeated runs
    coherence_score = Column(Float, nullable=False)     # length + structure heuristic
    overall_score = Column(Float, nullable=False)       # weighted composite

    score_breakdown = Column(JSON, nullable=True)       # full per-metric detail
    evaluated_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="results")


class BenchmarkReport(Base):
    """Aggregated benchmark comparison across providers."""
    __tablename__ = "benchmark_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String, unique=True, index=True)
    provider_scores = Column(JSON, nullable=False)      # {provider: {metric: avg_score}}
    best_provider = Column(String, nullable=False)
    total_prompts = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
