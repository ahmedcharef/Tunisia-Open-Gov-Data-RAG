# Tunisia Open Government Data RAG Pipeline

**Retrieval-Augmented Generation** over real Tunisian public datasets using LangChain, multilingual embeddings, and frontier LLMs via OpenRouter.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python version">
  <img src="https://img.shields.io/badge/LangChain-0.3+-orange?style=flat-square" alt="LangChain">
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square" alt="Chroma">
  <img src="https://img.shields.io/badge/Embeddings-multilingual--e5--large-purple?style=flat-square" alt="Embeddings">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-important?style=flat-square" alt="OpenRouter">
</p>

## ✨ Features

- Ingestion of Tunisian open government **CSV and XLSX** datasets
- Supports **Arabic, French, and mixed-language** data with a multilingual embedding model
- Auto-detection of CSV delimiter (`,` or `;`) and encoding (`utf-8`, `latin-1`, etc.)
- Column-aware metadata extraction across different schemas — governorate, name, type, delegation, coordinates
- Persistent Chroma vector database with named collections per dataset
- Conversational RAG chain with chat history awareness
- Governorate filtering at query time
- Easy model switching via OpenRouter (Qwen, Llama 3, Mistral, Claude, Gemini, …)
- Local fallback to Ollama
- CLI interface + Streamlit Web UI
- Single launcher: `run.py`

## 🏗️ Project Structure

```text
tunisia-rag/
├── data/                    # Place your CSV / XLSX files here
├── chroma_db/               # Persistent Chroma vector store (git-ignored)
├── src/
│   ├── __init__.py
│   ├── config.py            # Central configuration and dataset registry
│   ├── ingest.py            # Data → vector store pipeline
│   ├── retriever.py         # Vectorstore access, filtering, stats
│   ├── rag_service.py       # Shared RAG chain logic (used by CLI and UI)
│   ├── prompts.py           # LangChain prompt templates
│   ├── query.py             # CLI entry point
│   ├── utils.py             # Governorate extraction, source citation helpers
│   └── app.py               # Streamlit web UI
├── run.py                   # Unified launcher
├── .env                     # API keys (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

### 1. Prerequisites

- Python ≥ 3.10
- An [OpenRouter](https://openrouter.ai/keys) API key (free tier available)
- [Ollama](https://ollama.com) (optional — only needed for local LLM fallback)

### 2. Clone and install

```bash
git clone https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG
cd Tunisia-Open-Gov-Data-RAG
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your keys
```

Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `ollama` |
| `OPENROUTER_API_KEY` | — | Your OpenRouter key |
| `OPENROUTER_MODEL` | `qwen/qwen2.5-72b-instruct` | Any model on OpenRouter |
| `OLLAMA_MODEL` | `mistral` | Local Ollama model name |
| `TEMPERATURE` | `0.25` | LLM sampling temperature |
| `RETRIEVER_K` | `8` | Documents retrieved per query |
| `SEARCH_TYPE` | `mmr` | `mmr` or `similarity` |

## Supported Data Files

The ingestion pipeline loads **all `.csv` and `.xlsx` files** found in `data/`. Each file is automatically categorized and its schema is mapped to a set of canonical metadata fields.

### Current datasets in `data/`

| File | Category | Key fields extracted |
|---|---|---|
| `Etablissements-publics-enseignement-superieur-en-Tunisie.csv` | `etablissement_superieur_public` | gouvernorat, nom, type, lat, lon |
| `Les-Universites-Etatiques-Publiques-en-Tunisie.csv` | `universite_publique` | nom, website, adresse |
| `liste-des-etablissements-scolaires-publics.csv` | `ecole_publique` | gouvernorat, delegation, nom, type |
| `liste-des-etablissements-scolaires-prives.csv` | `ecole_privee` | gouvernorat, delegation, nom, type |
| `liste des établissements scolaires privés.csv` | `ecole_privee` | gouvernorat, delegation, nom, type |
| `etablissements_de_la_formation_professionnelle_agricole.xlsx` | `formation_professionnelle` | gouvernorat, nom (Arabic + French) |
| `Référentiel d'arrêt de la TRANSTU.csv` | `transport_public` | gouvernorat, nom, type, delegation, lat, lon |
| `Positions géographiques des stations du réseau bus de la TRANSTU.xlsx` | `transport_public` | nom, lat, lon |
| `presentes-bac-2024.csv` | `statistiques_scolaires` | gouvernorat (via CRE column) |
| `Effectif des élèves admis au concours d'entrée aux lycées pilotes.csv` | `statistiques_scolaires` | gouvernorat (via CRE column) |
| `Programme des allocations enfants 0-5 ans-2024.csv` | `programme_social` | gouvernorat (Arabic الولاية) |
| `Programme des allocations enfants 6-18 ans- 2023csv` | `programme_social` | gouvernorat (Arabic الولاية) |
| `reseau Tunisair.csv` | `transport_aerien` | nom (destination) |

To add a new file, just drop it in `data/` and re-run ingestion. If its column names match a known alias, metadata is extracted automatically. To support a new schema, add the column names to `_COLUMN_ALIASES` in `src/ingest.py`.

### Metadata fields

All documents are stored with these canonical metadata fields (when available):

| Field | Description |
|---|---|
| `gouvernorat` | Governorate — always stored **uppercase** for consistent filtering |
| `nom` | Establishment / station / destination name |
| `nom_ar` | Arabic name |
| `type` | Type of establishment |
| `delegation` | Delegation (sub-governorate) |
| `adresse` | Address |
| `lat` / `lon` | Geographic coordinates |
| `website` | Website URL |
| `source_file` | Original filename |
| `category` | Auto-assigned category (see table above) |

## Usage

All operations go through `run.py`:

```bash
# Ingest data into the vector store (run once, or after adding new files)
python run.py ingest

# Ingest into a specific dataset collection
python run.py ingest --dataset education
python run.py ingest --dataset agriculture

# Run the CLI (interactive Q&A)
python run.py

# Run the Streamlit web UI
python run.py ui

# Enable debug logging
python run.py --debug
```

You can also run the CLI directly with more options:

```bash
# Single query
python -m src.query --query "List public universities in Tunis"

# Single query with governorate filter and custom k
python -m src.query --query "Private schools in Sfax" --k 12

# Show collection statistics
python -m src.query --stats

# Use a specific dataset
python -m src.query --dataset agriculture
```

## When to re-run ingestion

```text
Did you:
  ├─ add / modify / delete any file in data/?              → YES → python run.py ingest
  ├─ change EMBEDDING_MODEL in config.py?                  → YES → python run.py ingest
  ├─ change chunking parameters (CHUNK_SIZE, CHUNK_OVERLAP)?→ YES → python run.py ingest
  ├─ change a dataset's collection name in Config.DATASETS? → YES → python run.py ingest (old collection stays but won't be used)
  └─ only changed prompts, LLM model, or query settings?   → NO  → no re-ingestion needed
```

## Adding a New Dataset

1. Add an entry to `Config.DATASETS` in `src/config.py`:
   ```python
   DATASETS = {
       "education": "tn_education_etablissements_2025",
       "agriculture": "tn_agriculture_ressources_hydrauliques_peche_maritime",
       "mydata": "tn_mydata_collection_name",   # ← add here
   }
   ```

2. Drop the CSV/XLSX files in `data/`.

3. Run ingestion for that dataset:
   ```bash
   python run.py ingest --dataset mydata
   ```

4. Select the dataset from the sidebar in the UI, or pass `--dataset mydata` to the CLI.

## Example Questions

**Education:**
```
What are the public universities in Tunis?
List private schools in Sfax.
Where is ENIT located?
Show me higher education institutions in Sousse.
كم عدد المدارس الثانوية العمومية في ولاية بن عروس؟
```

**Transport:**
```
What bus stops are in Ariana?
List TRANSTU stations in Manouba.
Which Tunisair destinations are in the Middle East?
```

**Social programs:**
```
How many families benefit from the children's allocation program in Kairouan?
```

**Follow-up (tests chat history):**
```
First: "Universities in Tunis?"
Then: "Which one is the oldest?" or "Tell me more about the first one."
```

## Architecture

```
User query
    │
    ▼
RAGService.query()              ← src/rag_service.py
    │
    ├─ extract_gouvernorat()    ← src/utils.py  (fuzzy match from query text)
    │
    ├─ get_retriever()          ← src/retriever.py
    │       └─ Chroma MMR search with optional {"gouvernorat": {"$eq": "SFAX"}} filter
    │
    ├─ contextualize_q_prompt   ← src/prompts.py  (reformulate question using history)
    │
    ├─ qa_prompt + LLM          ← OpenRouter / Ollama
    │
    └─ format_source_citation() ← src/utils.py
```

## Data Sources

All datasets are from [data.gov.tn](https://data.gov.tn) — Tunisia's official open government data portal.
