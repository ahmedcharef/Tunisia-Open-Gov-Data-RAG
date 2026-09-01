# Arize Phoenix Observability — Setup Guide

This project integrates [Arize Phoenix](https://phoenix.arize.com) for LLM tracing via OpenInference. Every LangChain call is automatically instrumented — retrieval, reranking, LLM generation, token counts, and latency — with zero manual span creation.

---

## How it works

| Span | What it traces |
|---|---|
| `rag_query` | Full pipeline root span (from Opik `@track`) |
| `contextualize_query` | Query rewrite using chat history |
| Chroma retrieval | Retrieved docs, latency |
| LLM call | Prompt, completion, token usage, model name |

The integration lives in `src/app.py` — Phoenix registers a `tracer_provider` at startup and `LangChainInstrumentor` auto-patches all LangChain chains from that point on.

---

## Option 1 — Local server (recommended, zero config)

### 1. Start the Phoenix server

```bash
python -m phoenix.server.main serve
```

UI available at **http://localhost:6006**. Keep this terminal open.

### 2. Run the app

```bash
python run.py ui
```

### 3. Send queries from the Streamlit UI

Open http://localhost:8501, ask a few questions, then go back to http://localhost:6006 → select the **tunisia-rag** project → traces appear in real time.

No `.env` changes needed for local mode.

---

## Option 2 — Phoenix Cloud

### 1. Sign up

Create an account at https://app.phoenix.arize.com and grab your API key.

### 2. Configure `.env`

```dotenv
PHOENIX_ENABLED=true
PHOENIX_PROJECT_NAME=tunisia-rag
PHOENIX_API_KEY=your-phoenix-api-key
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/v1/traces
```

### 3. Run normally

```bash
python run.py ui
```

Traces appear at https://app.phoenix.arize.com under your project.

---

## Disabling Phoenix

```dotenv
PHOENIX_ENABLED=false
```

The instrumentation block in `app.py` is fully guarded — setting this to `false` skips import and setup entirely.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PHOENIX_ENABLED` | `true` | Set `false` to disable all Phoenix tracing |
| `PHOENIX_PROJECT_NAME` | `tunisia-rag` | Project label shown in the Phoenix UI |
| `PHOENIX_API_KEY` | — | Only needed for Phoenix cloud |
| `PHOENIX_COLLECTOR_ENDPOINT` | — | Override collector URL (local or cloud) |

---

## Installation

Both packages are in `requirements.txt`:

```
arize-phoenix
openinference-instrumentation-langchain
```

To install manually:

```bash
pip install arize-phoenix openinference-instrumentation-langchain
```

---

## Getting more metrics from the dashboard

Send a variety of queries to exercise all code paths:

```bash
# Different datasets
"What public universities are in Tunis?"
"List TRANSTU bus stops in Ariana"
"How many families receive child allocations in Sfax?"

# Governorate filter (tests metadata filtering path)
"Private schools in Sfax"

# Follow-up queries (tests chat history + contextualization span)
"Universities in Tunis?"  →  "Which one is the oldest?"

# Arabic query (tests multilingual embedding path)
"كم عدد المدارس الثانوية في ولاية بن عروس؟"
```

In the Phoenix UI you can then:

- **Filter by latency** — find which queries are slow and why (retrieval vs LLM)
- **Compare token usage** — across models or query types
- **Inspect retrieval quality** — see exactly which chunks were returned for each query
- **Sort by span type** — isolate LLM cost vs retrieval cost
- **Replay any trace** — re-examine inputs and outputs after the fact
