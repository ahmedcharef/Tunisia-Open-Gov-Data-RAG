"""
Centralized configuration for Tunisia Education RAG.
"""

import os
import logging
import warnings
from dotenv import load_dotenv
from typing import Optional

# Suppress the common multilingual-e5-large warning
warnings.filterwarnings(
    "ignore",
    message=".*XLMRobertaModel LOAD REPORT.*"
)
warnings.filterwarnings(
    "ignore",
    message=".*position_ids.*UNEXPECTED.*"
)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tunisia-rag")


class Config:
    """Centralized configuration"""

    # ====================== PATHS ======================
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ====================== VECTOR STORE ======================
    COLLECTION_NAME: str = "tn_education_etablissements_2025"

    # ====================== EMBEDDINGS ======================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # ====================== LLM SETTINGS ======================
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
    def validate(cls) -> None:
        """Validate config and apply fallback logic."""
        if cls.LLM_PROVIDER == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                logger.warning("OPENROUTER_API_KEY is missing → falling back to Ollama")
                cls.LLM_PROVIDER = "ollama"

        elif cls.LLM_PROVIDER == "ollama":
            if not cls.OLLAMA_MODEL:
                logger.warning("OLLAMA_MODEL not set → using 'mistral'")
                cls.OLLAMA_MODEL = "mistral"

        # Optional warning
        if not os.path.exists(cls.DATA_DIR):
            logger.warning(f"Data directory '{cls.DATA_DIR}' not found.")

        logger.info(f"Config loaded | LLM: {cls.LLM_PROVIDER.upper()} | Model: {cls.OPENROUTER_MODEL if cls.LLM_PROVIDER == 'openrouter' else cls.OLLAMA_MODEL}")


# Auto-validate on import
Config.validate()