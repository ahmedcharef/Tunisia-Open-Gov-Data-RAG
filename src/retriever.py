"""
Centralized retriever logic for the Tunisia Education RAG project.
Handles vectorstore loading and retriever creation with proper error handling.
"""

import logging
from collections import Counter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

logger = logging.getLogger("tunisia-rag")

# Global vectorstore
try:
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,
    )
    logger.info(f"Vector store initialized | Collection: {Config.COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vectorstore = None


def get_retriever(k: int = 8, gouvernorat: str = None):
    """
    Returns a configured retriever with optional governorate filter.
    Uses MMR search for better diversity.
    """
    if vectorstore is None:
        raise RuntimeError("Vector store is not available. Please run ingest.py first.")

    try:
        search_kwargs = {"k": k}
        if gouvernorat:
            search_kwargs["filter"] = {"gouvernorat": {"$eq": gouvernorat.upper()}}

        return vectorstore.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


def get_available_governorates() -> list:
    """Dynamically extract unique governorates from the vectorstore."""
    try:
        if vectorstore is None:
            return []

        # Get all documents' metadata
        results = vectorstore._collection.get(include=["metadatas"])
        governorates = []

        for meta in results.get("metadatas", []):
            gov = meta.get("gouvernorat") or meta.get("governorate")
            if gov:
                governorates.append(str(gov).upper().strip())

        # Return unique sorted list
        unique_govs = sorted(set(g for g in governorates if g))
        return unique_govs

    except Exception as e:
        logger.warning(f"Could not extract governorates dynamically: {e}")
        # Fallback to static list
        return [
            "TUNIS", "SFAX", "SOUSSE", "ARIANA", "BEN AROUS", "MANOUBA", "NABEUL",
            "BIZERTE", "MONASTIR", "MAHDIA", "KAIROUAN", "GAFSA", "MEDENINE"
        ]


def get_vectorstore_stats() -> dict:
    """Return basic statistics about the vector database."""
    try:
        if vectorstore is None:
            return {"total_documents": 0, "status": "unavailable"}
        
        return {
            "total_documents": vectorstore._collection.count(),
            "collection_name": Config.COLLECTION_NAME,
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Failed to get vectorstore stats: {e}")
        return {
            "total_documents": 0,
            "status": "error",
            "error": str(e)
        }