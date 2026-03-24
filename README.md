# Tunisia Open Government Data RAG Pipeline

**Retrieval-Augmented Generation** over real Tunisian public datasets using LangChain, multilingual embeddings and frontier LLMs via OpenRouter.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python version">
  <img src="https://img.shields.io/badge/LangChain-0.3+-orange?style=flat-square" alt="LangChain">
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-green?style=flat-square" alt="Chroma">
  <img src="https://img.shields.io/badge/Embeddings-multilingual--e5--large-purple?style=flat-square" alt="Embeddings">
  <img src="https://img.shields.io/badge/LLM-OpenRouter-important?style=flat-square" alt="OpenRouter">
</p>

## ✨ Features

- Ingestion of Tunisian open government CSV datasets about **education establishments**
- Multilingual embedding model that handles **Arabic + French** very well
- Persistent Chroma vector database
- Conversational RAG chain with history awareness
- Easy model switching via OpenRouter (Qwen, Llama 3.3, Mistral, Claude, Gemini, …)
- Local fallback to Ollama possible
- Clean separation: ingestion / querying / configuration
- CLI interface + **Streamlit Web UI**

## 🏗️ Project Structure

```text
tunisia-rag/
├── data/                    # Put downloaded CSVs here
├── chroma_db/               # Persistent Chroma vector store (git ignored)
├── .env                     # API keys (git ignored)
├── requirements.txt
├── README.md
├── ingest.py                # One-time data → vector store pipeline
├── query.py                 # Conversational RAG interface
└── config.py                # Central model & path configuration
```

## Installation Guide

Follow these steps to get the **Tunisia Open Government Data RAG Pipeline** up and running on your machine.

### 1. Prerequisites

- **Python** ≥ 3.10  
- **[Ollama](https://ollama.com)** (optional – only if you want to use a local LLM fallback instead of OpenRouter)  
- An **OpenRouter** account and API key  
  → Create one (free tier available): [https://openrouter.ai/keys](https://openrouter.ai/keys)

### 2. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG
cd Tunisia-Open-Gov-Data-RAG
```

#### (Recommended) Create and activate a virtual environment

```bash
python -m venv .venv

# On macOS / Linux

source .venv/bin/activate

# On Windows

# .venv\Scripts\activate

# Install all required Python packages

pip install -r requirements.txt
```

### 3. Configure Environment Variables

copy and edit .env and add llm that you need it

```bash
cp .env.example .env
```

### 4. Download Tunisian Open Government Data (Education)

You are currently using the following datasets:

- **Établissements publics d'enseignement supérieur en Tunisie**  
- **Les Universités Étatiques Publiques en Tunisie**  
- **Liste des établissements scolaires privés**  
- **Liste des établissements scolaires publics**

These files are already in your `data/` folder.

**Alternative / additional sources** (if you want to expand later):

- <https://catalog.data.gov.tn/dataset/liste-des-etablissements-scolaires-publics>  
- <https://catalog.data.gov.tn/dataset/liste-des-etablissements-scolaires-prives>  
- <https://catalog.data.gov.tn/dataset/etablissements-publics-enseignement-superieur-en-tunisie>  
- <https://catalog.data.gov.tn/dataset/les-universites-etatiques-publiques-en-tunisie>

The ingestion script automatically loads **all .csv files** present in the `data/` directory.

Example structure:

```text
tunisia-rag/
├── data/
│   ├── Etablissements-publics-enseignement-superieur-en-Tunisie.csv
│   ├── Les-Universites-Etatiques-Publiques-en-Tunisie.csv
│   ├── liste-des-etablissements-scolaires-prives.csv
│   └── liste-des-etablissements-scolaires-publics.csv
├── ingest.py
├── query.py
├── config.py

### 5. Ingest the Data into the Vector Database

Run the ingestion script once to process the CSVs, create embeddings, and store them in Chroma:

```Bash
python ingest.py
```

This step creates the ./chroma_db folder (persistent vector store).
It may take a few minutes the first time, depending on the size of your CSV files and your internet connection (for downloading the embedding model).

### 6. Start Querying the Data

Launch the interactive query interface:

```Bash
python query.py
```

You will see a prompt where you can ask questions in French or Arabic.

Example questions you can try right away:

```text
Quelles sont les universités publiques à Tunis ?
Liste des écoles secondaires privées à Sfax
Quelle est l'adresse de l'ENIT (École Nationale d'Ingénieurs de Tunis) ?
Y a-t-il des écoles primaires publiques à Ariana ?
Combien d'établissements d'enseignement supérieur publics à Sousse ?
Quels lycées techniques existe-t-il à Monastir ?
كم عدد المدارس الثانوية العمومية في ولاية بن عروس؟
Quelles sont les écoles privées à Nabeul ?
```

Enjoy exploring real Tunisian open government data with natural language!

## When Should you run ingest.py?

You should run ingest.py in the following situations:

```text
Did you:
  ├─ add / modify / delete any file in data/ ?                → YES → run ingest.py
  ├─ change EMBEDDING_MODEL ?                                 → YES → run ingest.py
  ├─ change chunking parameters (size, overlap, splitter) ?   → YES → run ingest.py
  ├─ change COLLECTION_NAME ?                                 → YES → run ingest.py (old data stays but won't be used)
  └─ only changed query.py / prompt / LLM settings ?          → NO  → don't run ingest.py
```

## Running the Streamlit Web Interface

In addition to the CLI (query.py), this project includes a clean and interactive web interface built with Streamlit.
How to run the Streamlit app

```bash
# 1. Make sure all dependencies are installed
pip install -r requirements.txt

# 2. Run the Streamlit application
streamlit run app.py
```

## New Project Structure

```text
tunisia-rag/
├── data/                          # your CSV files
├── chroma_db/                     # vector database (auto-generated)
├── src/
│   ├── **init**.py
│   ├── config.py
│   ├── ingest.py
│   ├── prompts.py
│   ├── query.py
│   └── app.py                     # Streamlit UI
├── run.py
├── .env
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Use

Now you can launch your project easily with:

```Bash
# Run CLI version
python run.py

# Run Web UI (Streamlit)
python run.py ui

# With debug mode
python run.py --debug
```
