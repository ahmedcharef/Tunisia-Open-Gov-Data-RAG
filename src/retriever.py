"""
Centralized retriever logic for the Tunisia Education RAG project.
Handles vectorstore loading and retriever creation with proper error handling.
"""

import logging
from collections import Counter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config
import pandas as pd

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
            search_kwargs["filter"] = {"gouvernorat": {"$eq": gouvernorat.title()}}

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

        # Sample a maximum of 3000 documents to avoid "too many SQL variables" error
        MAX_SAMPLE = 3000

        # Get a sample of document IDs first
        all_ids = vectorstore._collection.get(include=[])["ids"]
        sample_ids = all_ids[:MAX_SAMPLE] if len(all_ids) > MAX_SAMPLE else all_ids

        # Fetch metadata and documents only for the sample
        results = vectorstore._collection.get(ids=sample_ids, include=["metadatas", "documents"])
        governorates = []

        for meta, doc in zip(results.get("metadatas", []), results.get("documents", [])):
            # Try metadata first
            gov = meta.get("gouvernorat") or meta.get("governorate")
            # Fall back to parsing page_content
            if not gov and doc:
                for line in doc.split("\n"):
                    col, _, val = line.partition(":")
                    if col.strip().lower() == "gouvernorat" and val.strip():
                        gov = val.strip()
                        break
            if gov:
                governorates.append(str(gov).strip().title())

        unique_govs = sorted(set(g for g in governorates if g))
        return unique_govs

    except Exception as e:
        logger.warning(f"Could not extract governorates dynamically: {e}")
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

def get_governorate_breakdown() -> pd.DataFrame:
    """Generate governorate breakdown safely by sampling documents to avoid SQL variable limit."""
    try:
        if vectorstore is None:
            return pd.DataFrame(columns=["Governorate", "Count"])

        # Sample a maximum of 2000 documents to avoid "too many SQL variables" error
        MAX_SAMPLE = 2000

        # Get a sample of document IDs first
        all_ids = vectorstore._collection.get(include=[])["ids"]
        if not all_ids:
            return pd.DataFrame(columns=["Governorate", "Count"])

        sample_ids = all_ids[:MAX_SAMPLE] if len(all_ids) > MAX_SAMPLE else all_ids

        # Fetch metadata only for the sample
        results = vectorstore._collection.get(ids=sample_ids, include=["metadatas"])

        gov_counts = Counter()

        possible_keys = ["gouvernorat", "governorate", "Gouvernorat", "Governorate",
                        "wilaya", "région", "region"]

        for meta in results.get("metadatas", []):
            gov = None
            for key in possible_keys:
                if key in meta and meta[key]:
                    gov = str(meta[key]).strip()
                    break
            if gov:
                gov_upper = gov.upper()
                gov_counts[gov_upper] += 1

        if not gov_counts:
            return pd.DataFrame(columns=["Governorate", "Count"])

        df = pd.DataFrame(list(gov_counts.items()), columns=["Governorate", "Count"])
        df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)

        # Add percentage column
        total_sampled = df["Count"].sum()
        if total_sampled > 0:
            df["Percentage"] = (df["Count"] / total_sampled * 100).round(1)
        else:
            df["Percentage"] = 0.0

        return df

    except Exception as e:
        logger.warning(f"Failed to generate governorate breakdown: {e}")
        return pd.DataFrame(columns=["Governorate", "Count"])