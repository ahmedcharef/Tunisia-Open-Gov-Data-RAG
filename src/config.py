"""
Centralized configuration for Tunisia Education RAG.
Single source of truth for all settings.
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
        # Add more datasets later:
        # "budgets": "tn_public_budgets",
    }

    # ====================== VECTOR STORE ======================
    COLLECTION_NAME: str = "tn_education_etablissements_2025"

    # ====================== EMBEDDINGS ======================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # ====================== LLM ======================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-72b-instruct").strip()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral").strip()

    # Generation parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.25"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Retrieval
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "8"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "mmr").strip()

    @classmethod
    def get_collection_name(cls, dataset: str = None) -> str:
        """Return collection name for a given dataset."""
        if dataset and dataset in cls.DATASETS:
            return cls.DATASETS[dataset]
        return cls.COLLECTION_NAME

    @classmethod
    def validate(cls) -> None:
        """Validate critical settings."""
        if cls.LLM_PROVIDER == "openrouter" and not cls.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY is missing → falling back to Ollama")
            cls.LLM_PROVIDER = "ollama"

        if not os.path.exists(cls.DATA_DIR):
            logger.warning(f"Data directory '{cls.DATA_DIR}' does not exist.")

        logger.info(f"Config loaded | Dataset: {cls.DEFAULT_DATASET} | LLM: {cls.LLM_PROVIDER}")

    def __repr__(self):
        return f"<Config dataset={self.DEFAULT_DATASET} llm={self.LLM_PROVIDER}>"


# Run validation when module is imported
Config.validate()