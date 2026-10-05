"""Pydantic v2 request/response schemas."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime


# ── Requests ──────────────────────────────────────────────────────────────────

class EvalRequest(BaseModel):
    prompt: str
    category: str = "open"                # factual | safety | consistency | open
    reference_answer: Optional[str] = None
    providers: List[str] = ["mock"]       # which LLM providers to test


class BenchmarkRequest(BaseModel):
    providers: List[str] = ["mock"]
    prompt_set: str = "default"           # default | safety | factual


# ── Responses ─────────────────────────────────────────────────────────────────

class MetricBreakdown(BaseModel):
    accuracy_score: Optional[float]
    safety_score: float
    consistency_score: Optional[float]
    coherence_score: float
    overall_score: float
    latency_ms: float


class EvalResultOut(BaseModel):
    provider: str
    response_text: str
    scores: MetricBreakdown
    evaluated_at: datetime

    model_config = {"from_attributes": True}


class EvalRunOut(BaseModel):
    run_id: str
    prompt: str
    category: str
    results: List[EvalResultOut]
    created_at: datetime

    model_config = {"from_attributes": True}


class ProviderSummary(BaseModel):
    provider: str
    avg_accuracy: Optional[float]
    avg_safety: float
    avg_coherence: float
    avg_overall: float
    avg_latency_ms: float
    prompts_evaluated: int


class BenchmarkReportOut(BaseModel):
    report_id: str
    best_provider: str
    total_prompts: int
    provider_summaries: List[ProviderSummary]
    created_at: datetime

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    total_runs: int
    total_results: int
    providers_tested: List[str]
    avg_safety_score: float
    avg_overall_score: float
    best_performing_provider: Optional[str]
