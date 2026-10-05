# LLM Evaluator — Production-Grade Language Model Evaluation Framework

A FastAPI-based benchmarking system that evaluates LLM responses across **accuracy, safety, consistency and coherence** metrics. Supports multiple providers, stores all results to an audit trail, and generates quantified benchmark comparison reports — directly analogous to GenAI quality evaluation workflows used by teams like Google AI Answers.

---

## Evaluation Metrics

Four metrics, each scored 0.0 – 1.0:

| Metric | Method | Description |
|---|---|---|
| **Accuracy** | ROUGE-L F1 (LCS) | Similarity to a reference answer — measures factual correctness |
| **Safety** | Rule-based harm detection | Detects harmful content; rewards correct refusals |
| **Consistency** | Cosine similarity (BoW) | Compares two responses to the same prompt — measures reliability |
| **Coherence** | Structural heuristic | Length appropriateness, sentence structure, non-repetition |
| **Overall** | Weighted composite | Accuracy 35% · Safety 30% · Coherence 20% · Consistency 15% |

---

## Provider Support

| Provider | Requirement | Notes |
|---|---|---|
| `mock` | None — works out of the box | Realistic deterministic responses, simulated latency |
| `openai` | `OPENAI_API_KEY` in `.env` | GPT-3.5-turbo by default |
| `huggingface` | `HF_API_KEY` in `.env` | Free tier — Mistral-7B-Instruct by default |

---

## Quick Start

```bash
git clone https://github.com/alansha1/llm-evaluator.git
cd llm-evaluator
pip install -r requirements.txt

# Train the model (run once)
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### Evaluate a single prompt

```bash
POST /api/v1/evaluate
```

```json
{
  "prompt": "What is the capital of France?",
  "category": "factual",
  "reference_answer": "The capital of France is Paris.",
  "providers": ["mock"]
}
```

**Response:**

```json
{
  "run_id": "run_a3f1b2c4d5e6",
  "prompt": "What is the capital of France?",
  "category": "factual",
  "results": [
    {
      "provider": "mock",
      "response_text": "The capital of France is Paris...",
      "scores": {
        "accuracy_score": 0.8571,
        "safety_score": 1.0,
        "consistency_score": 0.9234,
        "coherence_score": 0.9,
        "overall_score": 0.9142,
        "latency_ms": 214.3
      }
    }
  ]
}
```

### Run a full benchmark

```bash
POST /api/v1/benchmark
```

```json
{
  "providers": ["mock", "openai"],
  "prompt_set": "default"
}
```

Returns aggregated per-provider scores and automatically selects the best performer — identical to the 4-model evaluation framework in FraudGuard AI, applied to LLMs.

### All endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/evaluate` | Evaluate a single prompt |
| `GET` | `/api/v1/evaluate/{run_id}` | Retrieve an evaluation run |
| `GET` | `/api/v1/evaluations` | List all evaluation runs |
| `POST` | `/api/v1/benchmark` | Run full benchmark across a prompt set |
| `GET` | `/api/v1/benchmark/{report_id}` | Retrieve a benchmark report |
| `GET` | `/api/v1/benchmarks` | List all benchmark reports |
| `GET` | `/api/v1/dashboard/stats` | Live KPIs |
| `GET` | `/api/v1/prompts` | List available prompt sets |

---

## Prompt Sets

Three built-in prompt sets for benchmarking:

| Set | Prompts | Focus |
|---|---|---|
| `default` | 5 | Mixed factual + ML/data science questions |
| `safety` | 4 | Harmful request detection + safe handling |
| `factual` | 5 | ML concepts, statistics, SQL, APIs |

---

## Tests

```bash
pytest tests/ -v
# 20 passed
```

---

## Project Structure

```
llm-evaluator/
├── app/
│   ├── main.py        # FastAPI app entry point
│   ├── database.py    # SQLAlchemy engine + session
│   ├── models.py      # ORM models (EvalRun, EvalResult, BenchmarkReport)
│   ├── schemas.py     # Pydantic v2 schemas
│   ├── crud.py        # Database query helpers
│   └── routes.py      # API endpoints
├── eval/
│   ├── metrics.py     # Accuracy (ROUGE-L), Safety, Consistency, Coherence
│   ├── prompts.py     # Built-in prompt datasets (default / safety / factual)
│   └── runner.py      # Evaluation orchestrator
├── providers/
│   ├── base.py        # Abstract provider interface
│   ├── mock.py        # Mock provider (no API key required)
│   ├── openai_provider.py
│   └── huggingface_provider.py
├── tests/
│   └── test_api.py    # 20 test cases
├── .env.example
└── requirements.txt
```

---

*Built by [Alan Sha](https://github.com/alansha1) ·
