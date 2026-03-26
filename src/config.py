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
        "education": "tn_education_etablissements_2025",
        "agriculture": "tn_agriculture_ressources_hydrauliques_peche_maritime",
        # Add more later:
        # "budgets": "tn_public_budgets_2025",
        # "hospitals": "tn_health_facilities",
    }

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