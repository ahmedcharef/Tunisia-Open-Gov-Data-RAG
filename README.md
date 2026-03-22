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

- Ingestion of Tunisian open government CSV datasets (RGPH 2014 census recommended)
- Multilingual embedding model that handles **Arabic + French** very well
- Persistent Chroma vector database
- Conversational RAG chain with history awareness
- Easy model switching via OpenRouter (Qwen, Llama 3.3, Mistral Magistral, Claude, Gemini, …)
- Local fallback to Ollama possible
- Clean separation: ingestion / querying / configuration

## 📊 Recommended Dataset

**Recensement Général de la Population et de l'Habitat 2014 (RGPH 2014)**  
→ <https://catalog.data.gov.tn/fr/dataset/41242bb8-7580-441d-8a93-1e9d190019ff>

Why this dataset?

- Structured CSV files (one per governorate / theme / commune possible)
- Rich demographic, housing, urban/rural statistics
- Real Tunisian government open data → multilingual metadata & content
- Perfect size for local vector database experiments

## 🏗️ Project Structure

```text
tunisia-rag/
├── data/                    # Put downloaded CSVs here
│   └── recensement_population.csv   (example)
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

### 4. Download Tunisian Open Government Data

Visit the RGPH 2014 dataset page:
<https://catalog.data.gov.tn/fr/dataset/41242bb8-7580-441d-8a93-1e9d190019ff>

Download one or more CSV files from the available resources
(population by governorate, age groups, urban/rural distribution, etc.)

Place the downloaded CSV file(s) into the data/ folder in the project root.

Example structure after placing files:

```text
tunisia-rag/
├── data/
│   ├── recensement_population_gouvernorats.csv
│   └── population_par_age_et_sexe.csv
├── ingest.py ...The ingestion script automatically loads all .csv files present in the data/ directory.
```

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
Quelle est la population totale de la Tunisie selon le RGPH 2014 ?
Combien de femmes vivent en milieu rural ?
Quelle est la répartition par tranche d'âge à Tunis ?
Et pour les hommes ?               # ← follow-up questions work thanks to conversation history
كم عدد سكان تونس حسب تعداد 2014؟
```

Enjoy exploring real Tunisian open government data with natural language!
