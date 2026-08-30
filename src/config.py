"""
Centralized configuration for Tunisia Education RAG.
Single source of truth for all settings and datasets.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tunisia-rag")


class Config:
    """Centralized configuration."""

    # ====================== PATHS ======================
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ====================== DATASETS ======================
    DEFAULT_DATASET: str = "education"
    
    DATASETS: Dict[str, str] = {
        # Education: public/private schools, universities, vocational training
        "education": "tn_education",
        # Agriculture
        "agriculture": "tn_agriculture",
        # Transport: TRANSTU bus stops + geographic positions, Tunisair routes
        "transport": "tn_transport",
        # Social programs: children allocation programs
        "social": "tn_social",
        # Education statistics: BAC, lycées pilotes admissions
        "stats": "tn_stats",
    }

    # Controls what the Statistics Dashboard shows for each dataset.
    # breakdowns: list of {col, label} — one chart per entry.
    # col must match a canonical field name from _COLUMN_ALIASES in ingest.py.
    # primary_metric: label for the total-count metric card.
    DATASET_UI: Dict[str, Dict] = {
        "education": {
            "primary_metric": "Indexed Chunks",
            "breakdowns": [
                {"col": "gouvernorat", "label": "Governorate"},
            ],
        },
        "agriculture": {
            "primary_metric": "Indexed Chunks",
            "breakdowns": [
                {"col": "gouvernorat", "label": "Governorate"},
            ],
        },
        "transport": {
            "primary_metric": "Indexed Chunks",
            "breakdowns": [
                {"col": "gouvernorat", "label": "Governorate (TRANSTU)"},
                {"col": "zone_geo",    "label": "Geographic Zone (Tunisair)"},
            ],
        },
        "social": {
            "primary_metric": "Indexed Chunks",
            "breakdowns": [
                {"col": "gouvernorat", "label": "Governorate (الولاية)"},
            ],
        },
        "stats": {
            "primary_metric": "Indexed Chunks",
            "breakdowns": [
                {"col": "gouvernorat", "label": "Region (CRE)"},
            ],
        },
    }
    # Use None to ingest ALL files in data/ (legacy / catch-all behaviour).
    DATASET_FILES: Dict[str, list] = {
        "education": [
            "Etablissements-publics-enseignement-superieur-en-Tunisie.csv",
            "Les-Universites-Etatiques-Publiques-en-Tunisie.csv",
            "liste-des-etablissements-scolaires-prives.csv",
            "liste des établissements scolaires privés.csv",
            "liste-des-etablissements-scolaires-publics.csv",
            "etablissements_de_la_formation_professionnelle_agricole.xlsx",
        ],
        "agriculture": None,  # no files yet — add filenames when available
        "transport": [
            "Référentiel d'arrêt de la TRANSTU.csv",
            "Positions géographiques des stations du réseau bus de la TRANSTU.xlsx",
            "reseau Tunisair.csv",
        ],
        "social": [
            "Programme des allocations enfants 0-5 ans-2024.csv",
            "Programme des allocations enfants 6-18 ans-2023.csv",
        ],
        "stats": [
            "presentes-bac-2024.csv",
            "Effectif des élèves admis au concours d'entrée aux lycées pilotes.csv",
        ],
    }

    # ====================== TUNISIAN GOVERNORATES ======================
    # Complete list of all 24 governorates, stored UPPERCASE to match Chroma metadata.
    GOVERNORATES: list = [
        "TUNIS", "ARIANA", "MANOUBA", "BEN AROUS",
        "NABEUL", "ZAGHOUAN", "BIZERTE",
        "BÉJA", "JENDOUBA", "LE KEF", "SILIANA",
        "SOUSSE", "MONASTIR", "MAHDIA",
        "SFAX", "KAIROUAN", "KASSERINE", "SIDI BOUZID",
        "GABÈS", "MEDENINE", "TATAOUINE",
        "GAFSA", "TOZEUR", "KÉBILI",
    ]

    # ====================== OPIK / OBSERVABILITY ======================
    OPIK_API_KEY: Optional[str] = os.getenv("OPIK_API_KEY")
    OPIK_PROJECT_NAME: str = os.getenv("OPIK_PROJECT_NAME", "tunisia-rag")
    OPIK_WORKSPACE: Optional[str] = os.getenv("OPIK_WORKSPACE")
    # Set to "false" to disable tracing (e.g. in CI or local dev without an account)
    OPIK_ENABLED: bool = os.getenv("OPIK_ENABLED", "true").strip().lower() != "false"
    # Set to "true" to use a self-hosted local Opik instance instead of Comet cloud
    OPIK_USE_LOCAL: bool = os.getenv("OPIK_USE_LOCAL", "false").strip().lower() == "true"
    # Local Opik server URL (only used when OPIK_USE_LOCAL=true)
    OPIK_URL_OVERRIDE: Optional[str] = os.getenv("OPIK_URL_OVERRIDE")

    # ====================== EMBEDDINGS ======================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # ====================== RERANKER ======================
    # Cross-encoder used to re-rank retrieved chunks before sending to the LLM.
    # Multilingual model — handles Arabic, French, and English queries.
    USE_RERANKER: bool = os.getenv("USE_RERANKER", "true").strip().lower() != "false"
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    ).strip()
    # Retrieve k * RERANK_FACTOR candidates, rerank, then keep k
    RERANK_FACTOR: int = int(os.getenv("RERANK_FACTOR", "3"))

    # ====================== LLM ======================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-72b-instruct").strip()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral").strip()

    # Generation
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.25"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Retrieval
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "8"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "mmr").strip()
    # Weight for hybrid search: 0.0 = pure BM25, 1.0 = pure semantic, 0.5 = balanced
    HYBRID_SEMANTIC_WEIGHT: float = float(os.getenv("HYBRID_SEMANTIC_WEIGHT", "0.7"))
    # Set to false to use pure semantic (vector-only) retrieval
    USE_HYBRID_SEARCH: bool = os.getenv("USE_HYBRID_SEARCH", "true").strip().lower() != "false"

    @classmethod
    def get_collection_name(cls, dataset: str = None) -> str:
        """Return collection name for a given dataset."""
        if not dataset:
            dataset = cls.DEFAULT_DATASET
        return cls.DATASETS.get(dataset, cls.DATASETS[cls.DEFAULT_DATASET])

    @classmethod
    def validate(cls) -> None:
        if cls.LLM_PROVIDER == "openrouter" and not cls.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY is missing → falling back to Ollama")
            cls.LLM_PROVIDER = "ollama"

        if not os.path.exists(cls.DATA_DIR):
            logger.warning(f"Data directory '{cls.DATA_DIR}' does not exist.")

        logger.info(f"Config loaded | Default Dataset: {cls.DEFAULT_DATASET} | LLM: {cls.LLM_PROVIDER}")


Config.validate()