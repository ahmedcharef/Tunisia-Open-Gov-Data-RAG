"""
Centralized retriever logic for the Tunisia Education RAG.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

# Global vectorstore (loaded once)
embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=Config.CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=Config.COLLECTION_NAME,
)


def get_retriever(k: int = 8, gouvernorat: str = None):
    """
    Returns a retriever with optional metadata filtering.
    Uses MMR for better diversity.
    """
    search_kwargs = {"k": k}

    if gouvernorat:
        search_kwargs["filter"] = {"gouvernorat": {"$eq": gouvernorat.upper()}}

    return vectorstore.as_retriever(
        search_type=Config.SEARCH_TYPE,   # "mmr" by default from Config
        search_kwargs=search_kwargs
    )


def get_vectorstore_stats() -> dict:
    """Return basic statistics about the collection."""
    return {
        "total_documents": vectorstore._collection.count(),
        "collection_name": Config.COLLECTION_NAME,
    }