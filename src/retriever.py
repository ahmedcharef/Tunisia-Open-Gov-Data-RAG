"""
Centralized retriever logic for the Tunisia Education RAG project.
Handles vectorstore loading and retriever creation with proper error handling.
"""

import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

logger = logging.getLogger("tunisia-rag")

# Global vectorstore - loaded once when module is imported
try:
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,
    )
    logger.info(f"Vector store initialized successfully | Collection: {Config.COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize vector store: {e}")
    vectorstore = None  # Allow graceful degradation


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