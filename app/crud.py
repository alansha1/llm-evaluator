"""Database query helpers."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import EvalRun, EvalResult, BenchmarkReport


def get_eval_run(db: Session, run_id: str) -> EvalRun | None:
    return db.query(EvalRun).filter(EvalRun.run_id == run_id).first()


def list_eval_runs(db: Session, skip: int = 0, limit: int = 50):
    return (
        db.query(EvalRun)
        .order_by(EvalRun.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_benchmark_report(db: Session, report_id: str) -> BenchmarkReport | None:
    return db.query(BenchmarkReport).filter(BenchmarkReport.report_id == report_id).first()


def list_benchmark_reports(db: Session, limit: int = 20):
    return (
        db.query(BenchmarkReport)
        .order_by(BenchmarkReport.created_at.desc())
        .limit(limit)
        .all()
    )


def get_dashboard_stats(db: Session) -> dict:
    total_runs = db.query(func.count(EvalRun.id)).scalar()
    total_results = db.query(func.count(EvalResult.id)).scalar()
    avg_safety = db.query(func.avg(EvalResult.safety_score)).scalar()
    avg_overall = db.query(func.avg(EvalResult.overall_score)).scalar()

    providers = [r[0] for r in db.query(EvalResult.provider).distinct().all()]

    # Best provider by average overall score
    best = (
        db.query(EvalResult.provider, func.avg(EvalResult.overall_score).label("avg"))
        .group_by(EvalResult.provider)
        .order_by(func.avg(EvalResult.overall_score).desc())
        .first()
    )

    return {
        "total_runs": total_runs or 0,
        "total_results": total_results or 0,
        "providers_tested": providers,
        "avg_safety_score": round(avg_safety or 0, 4),
        "avg_overall_score": round(avg_overall or 0, 4),
        "best_performing_provider": best[0] if best else None,
    }
