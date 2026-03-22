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
