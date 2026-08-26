"""
Centralized retriever logic with very robust governorate extraction and breakdown.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import List, Dict

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config

logger = logging.getLogger("tunisia-rag")

# Module-level embedding cache — initialized once, reused across all calls
_embeddings_instance: HuggingFaceEmbeddings = None

def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    return _embeddings_instance


def get_vectorstore(dataset: str = None):
    collection_name = Config.get_collection_name(dataset)
    try:
        vs = Chroma(
            persist_directory=Config.CHROMA_PERSIST_DIR,
            embedding_function=_get_embeddings(),
            collection_name=collection_name,
        )
        return vs
    except Exception as e:
        logger.error(f"Failed to load vectorstore '{collection_name}': {e}")
        return None


def get_retriever(k: int = 8, gouvernorat: str = None, dataset: str = None):
    vectorstore = get_vectorstore(dataset)
    if vectorstore is None:
        raise RuntimeError(f"Vector store for dataset '{dataset or Config.DEFAULT_DATASET}' is not available.")

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


def get_available_governorates(dataset: str = None) -> List[str]:
    """Extract unique governorates with aggressive fallback strategies."""
    vectorstore = get_vectorstore(dataset)
    if vectorstore is None:
        return []

    try:
        all_ids = vectorstore._collection.get(include=[])["ids"]

        BATCH = 2000
        governorates = set()
        possible_keys = ["gouvernorat", "governorate", "Gouvernorat", "Governorate", "wilaya", "région", "region"]

        all_metadatas = []
        all_documents = []
        for i in range(0, len(all_ids), BATCH):
            batch = vectorstore._collection.get(ids=all_ids[i:i + BATCH], include=["metadatas", "documents"])
            all_metadatas.extend(batch.get("metadatas", []))
            all_documents.extend(batch.get("documents", []))

        # Strategy 1: Metadata
        for meta in all_metadatas:
            if not meta:
                continue
            for key in possible_keys:
                if key in meta and meta[key]:
                    gov = str(meta[key]).strip()
                    if gov and len(gov) > 2:
                        governorates.add(gov)
                    break

        # Strategy 2: Parse from page_content
        if len(governorates) < 5:
            logger.info("Falling back to parsing governorate from page_content...")
            for doc in all_documents:
                if not doc:
                    continue
                text = str(doc).lower()
                for known in ["béja", "tunis", "sfax", "sousse", "ariana", "ben arous", "manouba",
                             "nabeul", "bizerte", "monastir", "mahdia", "kairouan", "gafsa", "medenine"]:
                    if known in text:
                        governorates.add(known.title())
                        break

        unique_list = sorted(list(governorates))
        logger.info(f"Found {len(unique_list)} governorates: {unique_list[:15]}")
        return unique_list

    except Exception as e:
        logger.warning(f"Could not extract governorates: {e}")
        return []


def get_vectorstore_stats(dataset: str = None) -> Dict:
    vectorstore = get_vectorstore(dataset)
    if vectorstore is None:
        return {"total_documents": 0, "status": "unavailable"}

    try:
        total = vectorstore._collection.count()

        # Read ingestion stats from the JSON file written by ingest.py
        collection_name = Config.get_collection_name(dataset)
        stats_path = Path(Config.CHROMA_PERSIST_DIR) / f"{collection_name}_stats.json"
        row_count = None
        chunk_count = None
        if stats_path.exists():
            data = json.loads(stats_path.read_text())
            row_count = data.get("source_row_count")
            chunk_count = data.get("chunk_count")

        return {
            "total_documents": chunk_count if chunk_count is not None else total,
            "source_row_count": row_count,
            "collection_name": Config.get_collection_name(dataset),
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"total_documents": 0, "source_row_count": None, "status": "error", "error": str(e)}


def get_governorate_breakdown(dataset: str = None, breakdown_col: str = None) -> pd.DataFrame:
    """Generate a breakdown by the given metadata field.
    
    Args:
        dataset: dataset key (used to open the right collection)
        breakdown_col: canonical metadata field to group by (e.g. 'gouvernorat', 'zone_geo').
                       Falls back to the first entry in Config.DATASET_UI[dataset]['breakdowns'].
    """
    try:
        vectorstore = get_vectorstore(dataset)
        if vectorstore is None:
            return pd.DataFrame(columns=["Value", "Count", "Percentage"])

        # Resolve which field to group by
        if not breakdown_col:
            ui_cfg = Config.DATASET_UI.get(dataset or Config.DEFAULT_DATASET, {})
            breakdowns = ui_cfg.get("breakdowns", [])
            breakdown_col = breakdowns[0]["col"] if breakdowns else "gouvernorat"

        # Fetch ALL IDs then batch-read metadata — no arbitrary cap
        all_ids = vectorstore._collection.get(include=[])["ids"]
        if not all_ids:
            return pd.DataFrame(columns=[breakdown_col, "Count", "Percentage"])

        # Fetch in batches of 2000 to stay memory-friendly on large collections
        BATCH = 2000
        include = ["metadatas", "documents"] if breakdown_col == "gouvernorat" else ["metadatas"]
        all_metadatas = []
        all_documents = []
        for i in range(0, len(all_ids), BATCH):
            batch = vectorstore._collection.get(ids=all_ids[i:i + BATCH], include=include)
            all_metadatas.extend(batch.get("metadatas", []))
            # documents only present when gouvernorat fallback is needed
            docs = batch.get("documents")
            if docs:
                all_documents.extend(docs)
            else:
                all_documents.extend([None] * len(batch.get("metadatas", [])))

        counts = Counter()

        for meta, doc in zip(all_metadatas, all_documents):
            val = None

            if meta:
                val = meta.get(breakdown_col)
                if val:
                    val = str(val).strip()

            # Fallback for gouvernorat only: scan page_content text
            if not val and breakdown_col == "gouvernorat" and doc:
                text = str(doc).lower()
                for known in ["béja", "tunis", "sfax", "sousse", "ariana", "ben arous", "manouba",
                              "nabeul", "bizerte", "monastir", "mahdia", "kairouan", "gafsa", "medenine"]:
                    if known in text:
                        val = known.title()
                        break

            if val:
                counts[val] += 1

        if not counts:
            logger.warning(f"No '{breakdown_col}' data found for dataset '{dataset}'.")
            return pd.DataFrame(columns=[breakdown_col, "Count", "Percentage"])

        df = pd.DataFrame(list(counts.items()), columns=[breakdown_col, "Count"])
        df = df.sort_values(by="Count", ascending=False).reset_index(drop=True)
        total = df["Count"].sum()
        df["Percentage"] = (df["Count"] / total * 100).round(1) if total > 0 else 0.0

        logger.info(f"Breakdown by '{breakdown_col}' for dataset '{dataset}': {len(df)} values")
        return df

    except Exception as e:
        logger.warning(f"Failed to generate breakdown: {e}")
        return pd.DataFrame(columns=["Value", "Count", "Percentage"])