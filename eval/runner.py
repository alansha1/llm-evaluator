"""Evaluation runner — orchestrates provider calls and metric scoring."""

import uuid
from typing import List, Optional
from sqlalchemy.orm import Session

from providers import get_provider
from eval.metrics import (
    accuracy_score, safety_score, coherence_score,
    consistency_score, overall_score
)
from app.models import EvalRun, EvalResult


def run_single_eval(
    db: Session,
    prompt: str,
    category: str,
    providers: List[str],
    reference_answer: Optional[str] = None,
) -> EvalRun:
    """
    Evaluate one prompt across one or more providers.
    For consistency scoring, runs each provider twice and computes similarity.
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    eval_run = EvalRun(
        run_id=run_id,
        prompt=prompt,
        category=category,
        reference_answer=reference_answer,
    )
    db.add(eval_run)
    db.flush()

    for provider_name in providers:
        provider = get_provider(provider_name)

        # First response
        resp1 = provider.generate(prompt)

        # Second response for consistency scoring
        resp2 = provider.generate(prompt)

        # Compute metrics
        acc = accuracy_score(resp1.text, reference_answer) if reference_answer else None
        saf = safety_score(resp1.text)
        coh = coherence_score(resp1.text)
        con = consistency_score(resp1.text, resp2.text)
        overall = overall_score(saf, coh, acc, con)

        result = EvalResult(
            run_id=run_id,
            provider=provider_name,
            response_text=resp1.text,
            latency_ms=round(resp1.latency_ms, 2),
            accuracy_score=acc,
            safety_score=saf,
            consistency_score=con,
            coherence_score=coh,
            overall_score=overall,
            score_breakdown={
                "accuracy":    acc,
                "safety":      saf,
                "coherence":   coh,
                "consistency": con,
                "overall":     overall,
                "latency_ms":  round(resp1.latency_ms, 2),
                "second_response_preview": resp2.text[:120],
            },
        )
        db.add(result)

    db.commit()
    db.refresh(eval_run)
    return eval_run


def run_benchmark(
    db: Session,
    providers: List[str],
    prompt_set_name: str = "default",
) -> dict:
    """
    Run all prompts in a prompt set across all providers.
    Returns aggregated per-provider stats.
    """
    from eval.prompts import PROMPT_SETS
    from app.models import BenchmarkReport

    prompts = PROMPT_SETS.get(prompt_set_name, PROMPT_SETS["default"])
    report_id = f"bench_{uuid.uuid4().hex[:10]}"

    # Track per-provider metrics
    stats: dict = {p: {"accuracy": [], "safety": [], "coherence": [],
                       "consistency": [], "overall": [], "latency": []}
                   for p in providers}

    for item in prompts:
        eval_run = run_single_eval(
            db=db,
            prompt=item["prompt"],
            category=item["category"],
            providers=providers,
            reference_answer=item.get("reference_answer"),
        )
        for result in eval_run.results:
            p = result.provider
            if result.accuracy_score is not None:
                stats[p]["accuracy"].append(result.accuracy_score)
            stats[p]["safety"].append(result.safety_score)
            stats[p]["coherence"].append(result.coherence_score)
            if result.consistency_score is not None:
                stats[p]["consistency"].append(result.consistency_score)
            stats[p]["overall"].append(result.overall_score)
            stats[p]["latency"].append(result.latency_ms)

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

    provider_scores = {}
    for p, s in stats.items():
        provider_scores[p] = {
            "avg_accuracy":    avg(s["accuracy"]),
            "avg_safety":      avg(s["safety"]),
            "avg_coherence":   avg(s["coherence"]),
            "avg_consistency": avg(s["consistency"]),
            "avg_overall":     avg(s["overall"]),
            "avg_latency_ms":  avg(s["latency"]),
            "prompts_evaluated": len(prompts),
        }

    best = max(provider_scores, key=lambda p: provider_scores[p]["avg_overall"] or 0)

    report = BenchmarkReport(
        report_id=report_id,
        provider_scores=provider_scores,
        best_provider=best,
        total_prompts=len(prompts),
    )
    db.add(report)
    db.commit()

    return {"report_id": report_id, "best_provider": best,
            "total_prompts": len(prompts), "provider_scores": provider_scores}
