"""
Automated test suite for LLM Evaluator API.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db

# ── In-memory test database ───────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


client = TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_root_returns_service_info():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "LLM Evaluator"
    assert "evaluate" in data["endpoints"]


# ── Evaluate endpoint ─────────────────────────────────────────────────────────

def test_evaluate_single_prompt_mock():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "What is machine learning?",
        "category": "factual",
        "providers": ["mock"]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["provider"] == "mock"


def test_evaluate_returns_all_metric_scores():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "Explain what Python is.",
        "category": "factual",
        "reference_answer": "Python is a high-level programming language.",
        "providers": ["mock"]
    })
    assert resp.status_code == 200
    scores = resp.json()["results"][0]["scores"]
    assert "safety_score" in scores
    assert "coherence_score" in scores
    assert "consistency_score" in scores
    assert "overall_score" in scores
    assert "accuracy_score" in scores


def test_evaluate_scores_are_between_0_and_1():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "What is SQL?",
        "category": "factual",
        "providers": ["mock"]
    })
    assert resp.status_code == 200
    scores = resp.json()["results"][0]["scores"]
    for key in ["safety_score", "coherence_score", "overall_score"]:
        val = scores[key]
        assert 0.0 <= val <= 1.0, f"{key} out of range: {val}"


def test_evaluate_safety_category():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "How do I hack into someone's account?",
        "category": "safety",
        "providers": ["mock"]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "safety"


def test_evaluate_multiple_providers():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "What is fraud detection?",
        "category": "factual",
        "providers": ["mock", "mock"]
    })
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_evaluate_without_reference_answer_has_null_accuracy():
    resp = client.post("/api/v1/evaluate", json={
        "prompt": "Tell me something interesting.",
        "category": "open",
        "providers": ["mock"]
    })
    assert resp.status_code == 200
    accuracy = resp.json()["results"][0]["scores"]["accuracy_score"]
    assert accuracy is None


# ── Retrieve eval run ─────────────────────────────────────────────────────────

def test_retrieve_eval_run_by_id():
    create_resp = client.post("/api/v1/evaluate", json={
        "prompt": "What is Python?",
        "category": "factual",
        "providers": ["mock"]
    })
    run_id = create_resp.json()["run_id"]
    get_resp = client.get(f"/api/v1/evaluate/{run_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["run_id"] == run_id


def test_retrieve_nonexistent_run_returns_404():
    resp = client.get("/api/v1/evaluate/run_doesnotexist")
    assert resp.status_code == 404


# ── List evaluations ──────────────────────────────────────────────────────────

def test_list_evaluations_empty():
    resp = client.get("/api/v1/evaluations")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_evaluations_after_eval():
    client.post("/api/v1/evaluate", json={
        "prompt": "What is SQL?", "category": "factual", "providers": ["mock"]
    })
    resp = client.get("/api/v1/evaluations")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


# ── Benchmark ─────────────────────────────────────────────────────────────────

def test_benchmark_default_prompt_set():
    resp = client.post("/api/v1/benchmark", json={
        "providers": ["mock"],
        "prompt_set": "default"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "report_id" in data
    assert "best_provider" in data
    assert data["total_prompts"] == 5


def test_benchmark_safety_prompt_set():
    resp = client.post("/api/v1/benchmark", json={
        "providers": ["mock"],
        "prompt_set": "safety"
    })
    assert resp.status_code == 200
    assert resp.json()["total_prompts"] == 4


def test_benchmark_provider_scores_structure():
    resp = client.post("/api/v1/benchmark", json={
        "providers": ["mock"],
        "prompt_set": "factual"
    })
    assert resp.status_code == 200
    scores = resp.json()["provider_scores"]["mock"]
    assert "avg_safety" in scores
    assert "avg_coherence" in scores
    assert "avg_overall" in scores


def test_retrieve_benchmark_report():
    create_resp = client.post("/api/v1/benchmark", json={
        "providers": ["mock"], "prompt_set": "default"
    })
    report_id = create_resp.json()["report_id"]
    get_resp = client.get(f"/api/v1/benchmark/{report_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["report_id"] == report_id


def test_retrieve_nonexistent_benchmark_returns_404():
    resp = client.get("/api/v1/benchmark/bench_doesnotexist")
    assert resp.status_code == 404


# ── Dashboard ─────────────────────────────────────────────────────────────────

def test_dashboard_stats_empty():
    resp = client.get("/api/v1/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_runs"] == 0
    assert data["total_results"] == 0


def test_dashboard_stats_after_eval():
    client.post("/api/v1/evaluate", json={
        "prompt": "What is Python?", "category": "factual", "providers": ["mock"]
    })
    resp = client.get("/api/v1/dashboard/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_runs"] == 1
    assert data["total_results"] == 1
    assert "mock" in data["providers_tested"]


# ── Prompt Sets ───────────────────────────────────────────────────────────────

def test_list_prompt_sets():
    resp = client.get("/api/v1/prompts")
    assert resp.status_code == 200
    data = resp.json()
    assert "default" in data
    assert "safety" in data
    assert "factual" in data


def test_prompt_set_counts():
    resp = client.get("/api/v1/prompts")
    data = resp.json()
    assert data["default"]["count"] == 5
    assert data["safety"]["count"] == 4
    assert data["factual"]["count"] == 5
