"""
Centralized configuration for Tunisia Education RAG project.
All settings are defined here as a single source of truth.
"""

import os
import logging
from dotenv import load_dotenv
from typing import Optional, Dict, Any

load_dotenv()

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tunisia-rag")


class Config:
    """Centralized, immutable-style configuration for the project."""

    # ====================== PATHS ======================
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db"

    # ====================== VECTOR STORE ======================
    COLLECTION_NAME: str = "tn_education_etablissements_2025"

    # ====================== EMBEDDINGS ======================
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # ====================== LLM SETTINGS ======================
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "qwen/qwen2.5-72b-instruct"
    ).strip()

    # Ollama fallback
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral").strip()
    OLLAMA_BASE_URL: Optional[str] = os.getenv("OLLAMA_BASE_URL")

    # Generation parameters
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.25"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # Retrieval
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "8"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "mmr").strip()

    @classmethod
    def validate(cls) -> None:
        """Validate configuration and apply fallback logic."""
        if cls.LLM_PROVIDER == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                logger.warning("OPENROUTER_API_KEY missing in .env → falling back to Ollama")
                cls.LLM_PROVIDER = "ollama"
            if not cls.OPENROUTER_MODEL:
                logger.warning("OPENROUTER_MODEL not set, using default")
                cls.OPENROUTER_MODEL = "qwen/qwen2.5-72b-instruct"

        elif cls.LLM_PROVIDER == "ollama":
            if not cls.OLLAMA_MODEL:
                logger.warning("OLLAMA_MODEL not set → using 'mistral'")
                cls.OLLAMA_MODEL = "mistral"
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: '{cls.LLM_PROVIDER}'. Use 'openrouter' or 'ollama'.")

        # Optional warning
        if not os.path.exists(cls.DATA_DIR):
            logger.warning(f"Data directory '{cls.DATA_DIR}' does not exist.")

        # Final status log
        provider_info = f"{cls.LLM_PROVIDER.upper()} → {cls.OPENROUTER_MODEL if cls.LLM_PROVIDER == 'openrouter' else cls.OLLAMA_MODEL}"
        logger.info(f"Configuration loaded successfully | LLM: {provider_info}")

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return config as dictionary (useful for debugging)."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith("_") and not callable(getattr(cls, key))
        }

    def __repr__(self):
        return f"<Config LLM={self.LLM_PROVIDER} Model={self.OPENROUTER_MODEL if self.LLM_PROVIDER == 'openrouter' else self.OLLAMA_MODEL}>"


# Run validation automatically when module is imported
Config.validate()