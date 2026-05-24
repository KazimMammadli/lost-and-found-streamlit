# Smart Lost & Found

> AI-powered lost-and-found item matching, built with FastAPI, multi-provider
> Vision Language Models, and semantic embedding search.

![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![Pydantic](https://img.shields.io/badge/Pydantic-2.6-e92063)
![Tests](https://img.shields.io/badge/tests-164%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-71%25-brightgreen)
![Docker](https://img.shields.io/badge/docker-multi--stage-2496ED)
![Status](https://img.shields.io/badge/status-v1.0--final-success)

**Project:** AI-ENG-110 Capstone — Topic 1
**Team:** Avaz · Kazim · Gulnar
**Submission:** May 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage](#usage)
   - [REST API](#rest-api)
   - [Command-Line Interface](#command-line-interface)
   - [Web UI](#web-ui)
7. [Docker Deployment](#docker-deployment)
8. [Testing and Quality](#testing-and-quality)
9. [Benchmarks](#benchmarks)
10. [Project Structure](#project-structure)
11. [Bonus Features Delivered](#bonus-features-delivered)
12. [Tech Stack](#tech-stack)
13. [Team and Contribution](#team-and-contribution)
14. [License and Academic Notice](#license-and-academic-notice)

---

## Overview

**Smart Lost & Found** connects people who have lost something with people
who have found something, using AI-driven image understanding and
semantic search.

When a user submits a photo of an item — whether they are reporting it
as lost or as found — the system performs three operations end-to-end:

1. **Describe the image** through a Vision Language Model (OpenAI GPT-4o
   as the primary, Google Gemini 2.0 Flash as the secondary/failover),
   producing a structured description (object class, colours, brand,
   distinguishing marks).
2. **Embed the description** into a unit-normalised vector via a text
   embedding model.
3. **Match against the opposite pool** (lost ↔ found) using cosine
   similarity, returning the top-k most likely matches together with a
   short, human-readable explanation.

Because matching happens in embedding space, a *"navy backpack"*
reported lost still surfaces when someone registers a *"dark-blue
rucksack"* as found.

---

## Key Features

| Capability | Implementation |
| --- | --- |
| **Vision understanding** | OpenAI GPT-4o (primary) · Google Gemini 2.0 Flash (secondary) |
| **Semantic matching** | Cosine similarity on text embeddings |
| **REST API** | FastAPI with auto-generated OpenAPI documentation at `/docs` |
| **CLI** | Full `argparse` CLI for headless workflows and scripting |
| **Streamlit UI** | Browser-based registration, search, browse, and cost report |
| **Multi-provider failover** | Transparent fallback across ranked providers (`+3` bonus) |
| **Cost telemetry** | Append-only JSONL log with per-provider pricing table (`+2` bonus) |
| **Token rate limiting** | Sliding 60-second TPM budget (`+2` bonus) |
| **Distributed tracing** | OpenTelemetry spans exportable to Jaeger (`+2` bonus) |
| **Web UI** | Streamlit-powered single-page app (`+2` bonus) |
| **Containerization** | Multi-stage Dockerfile, non-root user, `HEALTHCHECK` (`+1` bonus) |
| **CI/CD** | GitHub Actions: lint → typecheck → test → docker build (`+2` bonus) |
| **Async + concurrent** | `asyncio`, `to_thread` bridge, semaphore-bounded batch |
| **Resilience** | Tenacity retries with exponential back-off, per-call timeouts |
| **Safe storage** | UUID-based filenames, path-traversal guard, MIME + size validation |
| **Repository abstraction** | SQLite for development, PostgreSQL for production |

---

## Architecture at a Glance

```
                        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
                        │  REST API    │  │     CLI      │  │ Streamlit UI │
                        │ (src/api.py) │  │ (src/cli.py) │  │  (ui/app.py) │
                        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
                               │                 │                 │
                               └────────┬────────┴────────┬────────┘
                                        │                 │
                              ┌─────────▼─────────┐  ┌────▼─────────┐
                              │  AI Service Layer │  │   Matcher    │
                              │ describe / embed  │  │ find_matches │
                              │  retries · cache  │  │  ranking +   │
                              │  rate limit · OTel│  │  reasons     │
                              └─────────┬─────────┘  └────┬─────────┘
                                        │                 │
                            ┌───────────▼───────────┐     │
                            │   ai/ (provided lib)  │     │
                            │  VLM + embedding +    │     │
                            │     similarity        │     │
                            └───────────┬───────────┘     │
                                        │                 │
                                ┌───────▼─────────────────▼───────┐
                                │     AbstractRepository (ABC)    │
                                │ ┌─────────────┐ ┌─────────────┐ │
                                │ │   SQLite    │ │ PostgreSQL  │ │
                                │ │ (aiosqlite) │ │  (asyncpg)  │ │
                                │ └─────────────┘ └─────────────┘ │
                                └─────────────────────────────────┘
```

For the full ASCII diagram and a description of the layers, see
[`docs/architecture.md`](docs/architecture.md).

**Design principles**

- **Async-first** — every I/O path is non-blocking; synchronous provider
  SDKs are bridged through `asyncio.to_thread()`.
- **Layered separation** — presentation (`api`, `cli`, `ui`) ↔ business
  logic (`core`) ↔ services (`services`) ↔ persistence (`storage`).
- **Repository pattern** — swapping SQLite for PostgreSQL is a config
  change, not a code change.
- **Single configuration source** — `src/config.py` (pydantic-settings)
  is the only place environment variables are read.
- **No bare prints for diagnostics** — `logging` is used throughout.
- **Path-safe storage** — UUID-generated filenames, never user-supplied
  paths.

---

## Quick Start

### Prerequisites

- Python **3.12** (the project is tested on 3.12; the Docker image
  pins this version).
- `git`
- (Optional) Docker Desktop for the containerised path.

### 1. Clone and create a virtual environment

```bash
git clone <repository-url>
cd smart-lost-and-found

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and provide your provider keys:
#   OPENAI_API_KEY=sk-...        # primary VLM + embedding
#   GOOGLE_API_KEY=...           # secondary VLM (Gemini failover)
```

### 3. Run a fully-offline demo (no API keys required)

```bash
python scripts/demo.py
```

This walks through the entire registration → embedding → matching
pipeline using deterministic fake providers, so you can verify the
system works before spending any API credits.

### 4. Start the API

```bash
uvicorn src.api:app --reload
# Interactive docs → http://localhost:8000/docs
```

### 5. Start the Web UI

```bash
streamlit run ui/app.py
# → http://localhost:8501
```

---

## Configuration

All settings are loaded from environment variables (or a `.env` file at
the project root). The complete list is documented in `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | Primary VLM provider (`openai`, `gemini`) |
| `LLM_MODEL` | `gpt-4o` | Model identifier passed to the primary provider |
| `OPENAI_API_KEY` | — | OpenAI API key (primary VLM + embedding) |
| `GOOGLE_API_KEY` | — | Google / Gemini API key (secondary VLM) |
| `EMBEDDING_PROVIDER` | `openai` | Embedding provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `SECONDARY_LLM_PROVIDER` | `gemini` | Failover VLM provider |
| `SECONDARY_LLM_MODEL` | `gemini-2.0-flash` | Failover VLM model identifier |
| `DATABASE_URL` | `sqlite+aiosqlite:///./dev.db` | SQLite dev URL or PostgreSQL connection string |
| `IMAGES_DIR` | `data/images` | Filesystem location for uploaded images |
| `MAX_IMAGE_BYTES` | `5242880` | Maximum upload size (default 5 MB) |
| `SEMAPHORE_LIMIT` | `10` | Cap on concurrent AI calls |
| `AI_CALL_TIMEOUT_S` | `30.0` | Per-call timeout in seconds |
| `AI_CALL_TPM_LIMIT` | `40000` | Tokens-per-minute rate limit |
| `LOG_LEVEL` | `INFO` | Standard library logging level |
| `ENABLE_TRACING` | `false` | When `true`, exports OTel spans via OTLP |
| `OTLP_ENDPOINT` | `http://localhost:4317` | OTLP gRPC endpoint (Jaeger or alternative) |
| `COST_LOG_PATH` | `artefacts/cost.jsonl` | JSONL file where cost events are appended |

---

## Usage

### REST API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/items/lost` | Register a lost item (multipart image upload) |
| `POST` | `/items/found` | Register a found item |
| `GET` | `/items/{id}/matches` | Top-k matches from the opposite pool |
| `GET` | `/items` | List items (optional `?status=lost\|found\|all`) |
| `POST` | `/items/batch-lost` | Register multiple lost items concurrently |
| `POST` | `/items/batch-found` | Register multiple found items concurrently |
| `GET` | `/health` | Liveness probe including database connectivity |

Interactive documentation is available at
`http://localhost:8000/docs` once the API is running.

**Register a lost item**

```bash
curl -X POST http://localhost:8000/items/lost \
  -F "file=@bag.jpg" \
  -F "user_text=navy backpack with laptop sticker"
```

**Find matches**

```bash
curl "http://localhost:8000/items/1/matches?k=3"
```

**List all found items**

```bash
curl "http://localhost:8000/items?status=found"
```

### Command-Line Interface

```bash
# Register a lost item
python -m src.cli register-lost path/to/bag.jpg --text "navy backpack"

# Register a found item
python -m src.cli register-found path/to/wallet.png

# Top-5 matches for item ID 1
python -m src.cli search-matches --id 1 --k 5

# List items
python -m src.cli list --status lost

# Aggregated cost report over the last 24 hours
python -m src.cli cost-report --since 24
```

### Web UI

```bash
streamlit run ui/app.py
```

The UI exposes five pages: **Register Lost**, **Register Found**,
**Search Matches**, **Browse Items**, and **Cost Report**.

---

## Docker Deployment

### Full stack with `docker-compose`

```bash
cp .env.example .env  # fill in keys
docker compose up --build
```

| Service | URL |
| --- | --- |
| REST API | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Jaeger tracing UI | http://localhost:16686 (only when `ENABLE_TRACING=true`) |

### Single-image build

```bash
docker build -t smart-lost-and-found .
docker run -p 8000:8000 --env-file .env smart-lost-and-found
```

The image is **multi-stage** (build → runtime), runs as the non-root
`appuser`, and ships a `HEALTHCHECK` that probes `/health` every 30 s.

---

## Testing and Quality

```bash
# Full pytest suite
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff check src/ tests/ scripts/ ui/

# Type check
mypy src/ --ignore-missing-imports
```

**Latest measured results** (2026-05-22):

| Metric | Value |
| --- | --- |
| Total tests | **164 passed, 0 failed** |
| Coverage (`src/`) | **71%** |
| In-container tests | **164/164 pass inside the runtime image** |
| `ruff` | Clean |
| `mypy` | Clean (19 source files) |
| Docker build | Successful |
| Container health | `/health` returns `{"status":"ok","db":"ok"}` |

**High-coverage modules:** `config` 100%, `models` 100%, `matcher` 98%,
`sqlite_repo` 97%, `base` 97%, `cost_meter` 96%, `rate_limiter` 91%,
`tracing` 87%, `api` 87%.

---

## Benchmarks

The benchmark script compares sequential against semaphore-bounded
concurrent execution of the registration pipeline:

```bash
python scripts/bench.py --n 12
```

Offline run with the `FakeVLM` provider:

```text
+---------------------+------+-------------+---------+
| Mode                |  N   |  Wall time  | Speedup |
+---------------------+------+-------------+---------+
| Sequential          |   12 |       0.18s |    1.0x |
| Concurrent (sem=10) |   12 |       0.04s |    4.5x |
+---------------------+------+-------------+---------+
```

The same benchmark with real providers typically yields a 3–5×
speedup, bounded by the provider's tokens-per-minute limit and the
`SEMAPHORE_LIMIT` setting.

---

## Project Structure

```text
smart-lost-and-found/
├── ai/                    Provided AI library (used unmodified)
│   ├── providers/         OpenAI, Google, Anthropic adapters
│   ├── vlm.py             describe_item() entry point
│   ├── embedding.py       embed() entry point
│   ├── similarity.py      cosine() and top_k()
│   └── schemas.py         ItemDescription and MatchResult
│
├── src/                   Application source code
│   ├── api.py             FastAPI HTTP server
│   ├── cli.py             argparse command-line interface
│   ├── config.py          pydantic-settings configuration
│   ├── models.py          Pydantic domain models
│   ├── tracing.py         OpenTelemetry setup and decorators
│   ├── core/              Business logic
│   │   ├── matcher.py     find_matches()
│   │   └── failover.py    FailoverVLM, FailoverEmbedder
│   ├── services/          AI integrations
│   │   ├── ai_service.py  describe_item_async, embed_async
│   │   └── cost_meter.py  Cost telemetry
│   ├── storage/           Persistence layer
│   │   ├── base.py        AbstractRepository ABC
│   │   ├── sqlite_repo.py SQLiteRepository (dev)
│   │   └── repository.py  PostgreSQLRepository (prod)
│   └── concurrency/
│       ├── pipeline.py    Bounded-concurrency batch runner
│       └── rate_limiter.py TokenBudget
│
├── ui/
│   └── app.py             Streamlit Web UI
│
├── tests/                 Full pytest suite (164 tests)
│   ├── fakes.py           FakeVLM and FakeEmbedder
│   ├── test_api.py
│   ├── test_ai_service.py
│   ├── test_ai_smoke.py   Provided grading contract tests
│   ├── test_cost_meter.py
│   ├── test_failover.py
│   ├── test_failures.py
│   ├── test_matcher.py
│   ├── test_pipeline.py
│   ├── test_rate_limiter.py
│   ├── test_repository.py
│   └── test_smoke_e2e.py
│
├── scripts/
│   ├── demo.py            Offline end-to-end demo
│   └── bench.py           Sequential vs concurrent benchmark
│
├── docs/
│   ├── REPORT.pdc         Final project report
│   └── SLIDES.pdf         Presentation deck
│
├── .github/workflows/     GitHub Actions CI pipeline
├── Dockerfile             Multi-stage container build
├── docker-compose.yml     API + UI + Jaeger
├── pyproject.toml         pytest, ruff, mypy configuration
├── requirements.txt       Pinned dependencies
├── conftest.py            Pytest fixtures
├── .env.example           Environment-variable template
└── README.md              You are here
```

---

## Bonus Features Delivered

| Feature | Source File | Bonus |
| --- | --- | --- |
| Multi-provider failover | `src/core/failover.py` | **+3** |
| Cost telemetry | `src/services/cost_meter.py` | **+2** |
| OpenTelemetry tracing | `src/tracing.py` (decorators applied to AI + DB) | **+2** |
| Token-aware rate limiter | `src/concurrency/rate_limiter.py` | **+2** |
| Streamlit Web UI | `ui/app.py` | **+2** |
| GitHub Actions CI | `.github/workflows/ci.yml` | **+2** |
| Multi-stage Dockerfile | `Dockerfile` | **+1** |
| **Total bonus claimed** | | **+14** |

---

## Tech Stack

**Application**

- Python 3.12
- FastAPI 0.109 · Uvicorn 0.27
- Pydantic 2.6 · pydantic-settings 2.2

**AI**

- OpenAI SDK (primary VLM + embedding) · Google Generative AI SDK
  (secondary VLM via failover) — both accessed through the provided
  `ai/` library
- NumPy 1.26 for vector arithmetic
- Tenacity 8.2 for retries

**Persistence**

- aiosqlite 0.20 (development)
- asyncpg 0.29 (production)

**UI and HTTP**

- Streamlit 1.32
- httpx 0.27 (FastAPI `TestClient` and outbound calls)

**Observability**

- OpenTelemetry 1.24 (API, SDK, OTLP exporter)
- Standard-library `logging`

**Quality**

- pytest 8.1 · pytest-asyncio 0.23 · pytest-cov 5.0
- ruff 0.3 · mypy 1.9

**Deployment**

- Docker (multi-stage)
- docker-compose (API + UI + Jaeger)
- GitHub Actions

---

## Team and Contribution

| Member | Role | Areas Owned |
| --- | --- | --- |
| **Avaz** | Platform & Architecture Lead | Storage layer, FastAPI endpoints, concurrency pipeline, project structure, Docker, release tagging |
| **Kazim** | AI Integration Lead | AI service wrappers, multi-provider failover, cost telemetry, rate limiter, OpenTelemetry tracing |
| **Gulnar** | Quality & UX Lead | Pytest suite (164 tests), CLI, Streamlit UI, GitHub Actions CI, benchmarks, end-to-end smoke tests |

Detailed contribution attribution per source file and per team member
is recorded in
[`docs/templates/CONTRIBUTION_STATEMENT.md`](docs/CONTRIBUTION_STATEMENT.md)
and signed by all three members.

**AI tool disclosure.** Claude Code was used as a development assistant
during the project — it helped with implementation support, debugging,
refactoring, documentation drafts, and workflow organisation. Every
change was reviewed and adapted by the team before merge. Full
disclosure is included in
[`docs/templates/CONTRIBUTION_STATEMENT.md`](docs/CONTRIBUTION_STATEMENT.md).

---

## License and Academic Notice

This project was developed as the AI-ENG-110 capstone for Topic 1
(*Smart Lost & Found*). The code in `src/`, `tests/`, `ui/`, `scripts/`,
and the configuration files is original work by the team. The
contents of `ai/` constitute a **provided library** distributed by the
course staff and are used unmodified.

Code submitted for grading is intended for academic evaluation. Reuse
beyond the course context should preserve the academic-attribution
notice above.

---

**Status:** v1.0-final — 164/164 tests passing, 71% coverage, Docker
image builds clean, container smoke verified.
