"""
Centralized retriever logic with robust governorate extraction for real Tunisian CSVs.
"""

import logging
from collections import Counter
from typing import List, Dict

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config

logger = logging.getLogger("tunisia-rag")

# ====================== GLOBAL VECTORSTORE ======================
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
    if vectorstore is None:
        raise RuntimeError("Vector store is not available. Please run `ingest.py` first.")

    try:
        search_kwargs: Dict = {"k": k}
        if gouvernorat:
            search_kwargs["filter"] = {"gouvernorat": {"$eq": gouvernorat.upper()}}
        return vectorstore.as_retriever(
            search_type=Config.SEARCH_TYPE,
            search_kwargs=search_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to create retriever: {e}")
        raise


def get_available_governorates() -> List[str]:
    """Robust extraction of governorates with multiple fallback strategies."""
    try:
        if vectorstore is None:
            return []

        MAX_SAMPLE = 2500

        all_ids = vectorstore._collection.get(include=[])["ids"]
        sample_ids = all_ids[:MAX_SAMPLE] if len(all_ids) > MAX_SAMPLE else all_ids

        # Strategy 1: Metadata keys
        results = vectorstore._collection.get(ids=sample_ids, include=["metadatas"])
        governorates = set()
        possible_keys = ["gouvernorat", "governorate", "Gouvernorat", "Governorate", "wilaya", "région", "region"]

        for meta in results.get("metadatas", []):
            if not meta:
                continue
            for key in possible_keys:
                if key in meta and meta[key]:
                    gov = str(meta[key]).strip()
                    if gov and len(gov) > 2:
                        governorates.add(gov.title())
                    break

        # Strategy 2: Parse from page_content (very important for your CSVs)
        if len(governorates) < 3:
            logger.info("Falling back to parsing governorate from page_content...")
            results = vectorstore._collection.get(ids=sample_ids, include=["documents"])
            for doc in results.get("documents", []):
                if not doc:
                    continue
                text = str(doc)
                # Look for common patterns like "Béja", "Tunis", "Sfax", etc.
                for known in ["Béja", "Tunis", "Sfax", "Sousse", "Ariana", "Ben Arous", "Manouba", 
                             "Nabeul", "Bizerte", "Monastir", "Mahdia", "Kairouan", "Gafsa", "Medenine"]:
                    if known.lower() in text.lower():
                        governorates.add(known)
                        break

        unique_list = sorted(list(governorates))
        if unique_list:
            logger.info(f"Found {len(unique_list)} governorates: {unique_list[:10]}")
        return unique_list

    except Exception as e:
        logger.warning(f"Could not extract governorates: {e}")
        return ["TUNIS", "SFAX", "SOUSSE", "ARIANA", "BEN AROUS", "MANOUBA", "NABEUL",
                "BIZERTE", "MONASTIR", "MAHDIA", "KAIROUAN", "GAFSA", "MEDENINE"]


def get_vectorstore_stats() -> Dict:
    try:
        if vectorstore is None:
            return {"total_documents": 0, "status": "unavailable"}
        return {
            "total_documents": vectorstore._collection.count(),
            "collection_name": Config.COLLECTION_NAME,
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"total_documents": 0, "status": "error", "error": str(e)}


def get_governorate_breakdown() -> pd.DataFrame:
    """Generate governorate breakdown with aggressive parsing."""
    try:
        if vectorstore is None:
            return pd.DataFrame(columns=["Governorate", "Count", "Percentage"])

        MAX_SAMPLE = 2000
        all_ids = vectorstore._collection.get(include=[])["ids"]
        if not all_ids:
            return pd.DataFrame(columns=["Governorate", "Count", "Percentage"])

        sample_ids = all_ids[:MAX_SAMPLE] if len(all_ids) > MAX_SAMPLE else all_ids

        results = vectorstore._collection.get(ids=sample_ids, include=["metadatas", "documents"])

        gov_counts = Counter()
        possible_keys = ["gouvernorat", "governorate", "Gouvernorat", "Governorate", "wilaya"]

        for meta, doc in zip(results.get("metadatas", []), results.get("documents", [])):
            gov = None
            # Try metadata first
            if meta:
                for key in possible_keys:
                    if key in meta and meta[key]:
                        gov = str(meta[key]).strip().title()
                        break

            # Fallback: parse from page_content
            if not gov and doc:
                text = str(doc).lower()
                for known in ["béja", "tunis", "sfax", "sousse", "ariana", "ben arous", "manouba",
                             "nabeul", "bizerte", "monastir", "mahdia", "kairouan", "gafsa", "medenine"]:
                    if known in text:
                        gov = known.title()
                        break

            if gov:
                gov_counts[gov] += 1

        if not gov_counts:
            return pd.DataFrame(columns=["Governorate", "Count", "Percentage"])

        df = pd.DataFrame(list(gov_counts.items()), columns=["Governorate", "Count"])
        df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)

        total = df["Count"].sum()
        df["Percentage"] = (df["Count"] / total * 100).round(1) if total > 0 else 0.0

        return df

    except Exception as e:
        logger.warning(f"Failed to generate governorate breakdown: {e}")
        return pd.DataFrame(columns=["Governorate", "Count", "Percentage"])