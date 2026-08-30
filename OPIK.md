# Opik Observability — Setup Guide

This project integrates [Opik by Comet](https://www.comet.com/site/products/opik/) for LLM tracing and the Agent Playground. Every RAG query is traced end-to-end: context retrieval, query rewriting, reranking, and LLM generation.

---

## How it works

| Span | What it traces |
|---|---|
| `tunisia_rag_agent` | Top-level entrypoint — visible in Agent Playground |
| `rag_query` | Full pipeline: retrieve → rerank → generate |
| `contextualize_query` | Query rewrite using chat history |
| LangChain spans | LLM calls, token counts, latency (auto-instrumented) |

The integration lives in two files:
- `src/opik_setup.py` — initialises Opik, exposes `@track` and `get_langchain_tracer()`
- `src/rag_service.py` — decorates pipeline functions and passes the tracer to LangChain chains

---

## Option 1 — Local Docker (recommended for development)

### 1. Start the Opik server

```bash
# From the project root
docker run --rm -d \
  --name opik \
  -p 5173:5173 \
  -p 8000:8000 \
  ghcr.io/comet-ml/opik/opik-local:latest
```

Or using Docker Compose if you have the full Opik source:

```bash
docker compose -f opik-src/deployment/docker-compose.yml up -d
```

Wait ~15 seconds for the stack to be ready, then open **http://localhost:5173**.

### 2. Create your workspace

On first load, sign up at http://localhost:5173 and note your workspace name (e.g. `bouhmid`).

### 3. Configure `.env`

```dotenv
OPIK_ENABLED=true
OPIK_USE_LOCAL=true
OPIK_PROJECT_NAME=tunisia-rag
OPIK_WORKSPACE=your-workspace-name   # the name you chose at sign-up
```

### 4. Run the app with the Opik endpoint connector

```bash
opik endpoint --project "Default Project" -- python run.py ui
```

The `opik endpoint` wrapper registers your agent with the Agent Playground. Once the app is running, open http://localhost:5173 → Agent Playground to see `tunisia_rag_agent` listed.

---

## Option 2 — Comet Cloud

### 1. Get an API key

Sign up at https://www.comet.com and copy your API key from Account Settings.

### 2. Configure `.env`

```dotenv
OPIK_ENABLED=true
OPIK_USE_LOCAL=false
OPIK_API_KEY=your-api-key
OPIK_PROJECT_NAME=tunisia-rag
OPIK_WORKSPACE=your-comet-workspace
```

### 3. Run normally

```bash
python run.py ui
# or
opik endpoint --project "Default Project" -- python run.py ui
```

Traces will appear at https://www.comet.com/opik.

---

## Disabling tracing

Set `OPIK_ENABLED=false` in `.env` to turn off all tracing with zero code changes. The decorators degrade to no-ops so nothing breaks.

---

## Installation

Opik is already in `requirements.txt`:

```
opik==2.2.44
```

To install manually:

```bash
pip install opik==2.2.44
```

> **Important:** Do not have a folder named `opik/` at the project root — Python will treat it as a namespace package and shadow the installed SDK. The cloned Opik source repo (if present) should be renamed to `opik-src/`.

---

## One-time CLI configuration (alternative to `.env`)

You can also configure Opik interactively once via the CLI. It writes to `~/.opik.config` and persists across runs:

```bash
opik configure
```

Select:
- `3` for local deployment
- Enter your workspace name when prompted

After this, you only need `OPIK_ENABLED=true` in `.env` — the rest is read from `~/.opik.config`.

---

## Stopping the Docker container

```bash
docker stop opik
```

Data is not persisted with the single-container run command above. To persist traces across restarts, use Docker Compose with a named volume.
