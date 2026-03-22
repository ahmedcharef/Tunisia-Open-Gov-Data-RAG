import os
import logging
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tunisia-rag")

# ====================== CONFIG CLASS ======================
class Config:
    """Centralized configuration with validation and defaults."""

    # === Paths ===
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "tunisia_census"

    # === Embeddings (still the best for Arabic + French in 2026) ===
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # === LLM Provider & Model ===
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openrouter").lower()  # "openrouter" or "ollama"

    # Best model for Tunisian Arabic/French government data (March 2026)
    # Qwen3-235B-A22B is currently the Arabic leader + excellent French
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL",
        "qwen/qwen3-235b-a22b"          # ← Top recommendation (MoE, 100+ languages)
    )
    # Strong free/cheap alternatives (uncomment one if you want):
    # "qwen/qwen3-8b-instruct:free"
    # "meta-llama/llama-3.3-70b-instruct:free"
    # "google/gemini-2.5-flash"
    # "anthropic/claude-3-5-sonnet-20241022"

    # === LLM Parameters (tunable via .env) ===
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))

    # === Vector Search Defaults ===
    RETRIEVER_K: int = 8
    SEARCH_TYPE: str = "similarity"          # "similarity" or "mmr"

    @classmethod
    def validate(cls) -> None:
        """Basic validation + fallback logic."""
        if cls.LLM_PROVIDER == "openrouter":
            if not os.getenv("OPENROUTER_API_KEY"):
                cls.logger.warning("OPENROUTER_API_KEY missing → falling back to Ollama")
                cls.LLM_PROVIDER = "ollama"
        elif cls.LLM_PROVIDER == "ollama":
            # Ollama is always "free" if running locally
            pass
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {cls.LLM_PROVIDER}")

        if not os.path.exists(cls.DATA_DIR):
            cls.logger.warning(f"Data directory {cls.DATA_DIR} not found. Create it and add CSVs.")

Config.validate()