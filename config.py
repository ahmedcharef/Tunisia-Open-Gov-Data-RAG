import os
import logging
from dotenv import load_dotenv
from typing import Optional

load_dotenv()  # Load variables from .env file early

# ====================== LOGGING SETUP ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tunisia-rag")

# ====================== CENTRALIZED CONFIG CLASS ======================
class Config:
    """Centralized, type-safe configuration with .env support and validation."""

    # === Paths ===
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "tunisia_education_etablissements"

    # === Embeddings ===
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # === LLM Provider ===
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()

    # === OpenRouter specific ===
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "qwen/qwen2.5-72b-instruct"   # ← sensible default if not set in .env
    ).strip()

    # === Ollama fallback (when OpenRouter is unavailable) ===
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral").strip()
    OLLAMA_BASE_URL: Optional[str] = os.getenv("OLLAMA_BASE_URL")

    # === Generation parameters (also overridable via .env) ===
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.25"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # === Retrieval defaults ===
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "8"))
    SEARCH_TYPE: str = os.getenv("SEARCH_TYPE", "similarity").strip()  # "similarity" or "mmr"

    @classmethod
    def validate(cls) -> None:
        """Validate critical settings and apply fallback logic."""
        if cls.LLM_PROVIDER == "openrouter":
            if not cls.OPENROUTER_API_KEY:
                logger.warning(
                    "OPENROUTER_API_KEY is missing in .env → "
                    "falling back to local Ollama"
                )
                cls.LLM_PROVIDER = "ollama"
            elif not cls.OPENROUTER_MODEL:
                logger.warning("OPENROUTER_MODEL not set → using default model")
                cls.OPENROUTER_MODEL = "qwen/qwen2.5-72b-instruct"

        elif cls.LLM_PROVIDER == "ollama":
            if not cls.OLLAMA_MODEL:
                logger.warning("OLLAMA_MODEL not set → using 'mistral' as fallback")
                cls.OLLAMA_MODEL = "mistral"
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{cls.LLM_PROVIDER}'. "
                "Use 'openrouter' or 'ollama'."
            )

        # Quick existence check (warn only)
        if not os.path.exists(cls.DATA_DIR):
            logger.warning(f"Data directory not found: {cls.DATA_DIR}")

        logger.info(f"Active LLM provider: {cls.LLM_PROVIDER}")
        if cls.LLM_PROVIDER == "openrouter":
            logger.info(f"  → model: {cls.OPENROUTER_MODEL}")
        else:
            logger.info(f"  → model: {cls.OLLAMA_MODEL}")

# Run validation once when module is imported
Config.validate()