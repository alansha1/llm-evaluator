"""API route definitions."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    EvalRequest, BenchmarkRequest,
    EvalRunOut, BenchmarkReportOut, DashboardStats, ProviderSummary
)
from app import crud
from eval.runner import run_single_eval, run_benchmark

router = APIRouter(prefix="/api/v1")


# ── Evaluate ──────────────────────────────────────────────────────────────────

@router.post("/evaluate", response_model=EvalRunOut, summary="Evaluate a single prompt")
def evaluate_prompt(request: EvalRequest, db: Session = Depends(get_db)):
    """
    Submit a prompt to one or more LLM providers and receive scored evaluation results.

    - **prompt**: The question or instruction to evaluate
    - **category**: `factual` | `safety` | `consistency` | `open`
    - **reference_answer**: Optional ground truth for accuracy scoring (ROUGE-L)
    - **providers**: List of providers — `mock`, `openai`, `huggingface`
    """
    eval_run = run_single_eval(
        db=db,
        prompt=request.prompt,
        category=request.category,
        providers=request.providers,
        reference_answer=request.reference_answer,
    )

    results_out = []
    for r in eval_run.results:
        results_out.append({
            "provider": r.provider,
            "response_text": r.response_text,
            "scores": {
                "accuracy_score": r.accuracy_score,
                "safety_score": r.safety_score,
                "consistency_score": r.consistency_score,
                "coherence_score": r.coherence_score,
                "overall_score": r.overall_score,
                "latency_ms": r.latency_ms,
            },
            "evaluated_at": r.evaluated_at,
        })

    return {
        "run_id": eval_run.run_id,
        "prompt": eval_run.prompt,
        "category": eval_run.category,
        "results": results_out,
        "created_at": eval_run.created_at,
    }


# ── Benchmark ─────────────────────────────────────────────────────────────────

@router.post("/benchmark", summary="Run full benchmark across a prompt set")
def run_full_benchmark(request: BenchmarkRequest, db: Session = Depends(get_db)):
    """
    Run all prompts in a named prompt set across all specified providers.
    Returns aggregated per-provider scores and selects the best performer.

    - **providers**: List of providers to compare
    - **prompt_set**: `default` | `safety` | `factual`
    """
    result = run_benchmark(
        db=db,
        providers=request.providers,
        prompt_set_name=request.prompt_set,
    )
    return result


@router.get("/benchmark/{report_id}", summary="Retrieve a benchmark report")
def get_benchmark(report_id: str, db: Session = Depends(get_db)):
    report = crud.get_benchmark_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Benchmark report not found")
    return {
        "report_id": report.report_id,
        "best_provider": report.best_provider,
        "total_prompts": report.total_prompts,
        "provider_scores": report.provider_scores,
        "created_at": report.created_at,
    }


@router.get("/benchmarks", summary="List all benchmark reports")
def list_benchmarks(db: Session = Depends(get_db)):
    reports = crud.list_benchmark_reports(db)
    return [
        {
            "report_id": r.report_id,
            "best_provider": r.best_provider,
            "total_prompts": r.total_prompts,
            "created_at": r.created_at,
        }
        for r in reports
    ]


# ── Eval Runs ─────────────────────────────────────────────────────────────────

@router.get("/evaluate/{run_id}", response_model=EvalRunOut, summary="Retrieve an evaluation run")
def get_eval_run(run_id: str, db: Session = Depends(get_db)):
    run = crud.get_eval_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    results_out = []
    for r in run.results:
        results_out.append({
            "provider": r.provider,
            "response_text": r.response_text,
            "scores": {
                "accuracy_score": r.accuracy_score,
                "safety_score": r.safety_score,
                "consistency_score": r.consistency_score,
                "coherence_score": r.coherence_score,
                "overall_score": r.overall_score,
                "latency_ms": r.latency_ms,
            },
            "evaluated_at": r.evaluated_at,
        })
    return {
        "run_id": run.run_id,
        "prompt": run.prompt,
        "category": run.category,
        "results": results_out,
        "created_at": run.created_at,
    }


@router.get("/evaluations", summary="List all evaluation runs")
def list_evals(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    runs = crud.list_eval_runs(db, skip=skip, limit=limit)
    return [
        {"run_id": r.run_id, "prompt": r.prompt[:80], "category": r.category,
         "created_at": r.created_at}
        for r in runs
    ]


# ── Dashboard ─────────────────────────────────────────────────────────────────

@router.get("/dashboard/stats", response_model=DashboardStats, summary="Live dashboard KPIs")
def dashboard_stats(db: Session = Depends(get_db)):
    return crud.get_dashboard_stats(db)


# ── Prompt Sets ───────────────────────────────────────────────────────────────

@router.get("/prompts", summary="List available built-in prompt sets")
def list_prompt_sets():
    from eval.prompts import PROMPT_SETS
    return {
        name: {"count": len(prompts), "categories": list({p["category"] for p in prompts})}
        for name, prompts in PROMPT_SETS.items()
    }
