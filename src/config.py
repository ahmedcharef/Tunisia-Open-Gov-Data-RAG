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
            "Programme des allocations enfants 6-18 ans- 2023csv",
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

    # ====================== EMBEDDINGS ======================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

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