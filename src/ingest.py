#!/usr/bin/env python3
"""
ingest.py - Ingestion pipeline for Tunisian education CSV files.

This script loads the four education CSVs, enriches metadata (especially governorate),
chunks the documents, and stores them in Chroma with optimized settings.
"""

import os
import glob
import logging
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config, logger
import warnings
from tqdm import tqdm

# Silence noisy logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*position_ids.*")

load_dotenv()

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

CSV_DELIMITER = ";"
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180

# ────────────────────────────────────────────────
def ingest_education_csvs() -> None:
    """Main ingestion pipeline."""
    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please place your CSV files in the 'data/' folder.")
        return

    # Find CSV files
    csv_patterns = [
        "Etablissements-publics-enseignement-superieur-en-Tunisie.csv",
        "Les-Universites-Etatiques-Publiques-en-Tunisie.csv",
        "liste-des-etablissements-scolaires-prives.csv",
        "liste-des-etablissements-scolaires-publics.csv",
        "*.csv",   # fallback
    ]

    all_files: List[Path] = []
    for pattern in csv_patterns:
        found = list(data_dir.glob(pattern))
        all_files.extend(found)

    if not all_files:
        logger.error(f"No CSV files found in {data_dir}")
        return

    logger.info(f"Found {len(all_files)} CSV file(s)")

    all_docs = []

    for file_path in all_files:
        logger.info(f"Loading: {file_path.name}")

        try:
            loader = CSVLoader(
                file_path=str(file_path),
                encoding="utf-8",
                csv_args={"delimiter": CSV_DELIMITER},
            )
            docs = loader.load()

            category = _guess_category_from_filename(file_path.name)

            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["category"] = category
                # Improved metadata extraction
                _extract_common_fields_to_metadata(doc)

            logger.info(f"  → Loaded {len(docs):,} rows")
            all_docs.extend(docs)

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    if not all_docs:
        logger.error("No documents were loaded.")
        return

    # ─── Split Documents ────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " - ", " | ", " • "],
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks):,} chunks")

    # ─── Embed & Store ────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    logger.info("Creating / updating Chroma collection...")

    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,          # Use Config now
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 40,      # Faster indexing
            "hnsw:M": 16,
        },
    )

    # Batch indexing with progress bar
    batch_size = 400
    total_chunks = len(chunks)

    with tqdm(total=total_chunks, desc="Indexing", unit="chunk") as pbar:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    logger.info(
        f"✅ Ingestion completed successfully!\n"
        f"   Collection : {Config.COLLECTION_NAME}\n"
        f"   Documents  : {len(all_docs):,}\n"
        f"   Chunks     : {total_chunks:,}\n"
        f"   Storage    : {Config.CHROMA_PERSIST_DIR}"
    )


def _guess_category_from_filename(filename: str) -> str:
    name = filename.lower()
    if "universit" in name or "etatiqu" in name:
        return "universite_publique"
    if "enseignement-superieur" in name:
        return "etablissement_superieur_public"
    if "prives" in name:
        return "ecole_privee"
    if "publics" in name and "scolaires" in name:
        return "ecole_publique"
    return "autre"


def _extract_common_fields_to_metadata(doc) -> None:
    """Improved metadata extraction for real Tunisian CSVs."""
    content = doc.page_content
    lines = [line.strip() for line in content.split("\n") if ":" in line]

    known_keys = {
        "gouvernorat": ["gouvernorat", "governorate", "wilaya", "région"],
        "delegation": ["délégation", "delegation"],
        "type": ["type d'établissement", "type", "category"],
        "nom": ["nom de l'établissement", "nom", "name"],
        "adresse": ["adresse", "address"],
    }

    for line in lines:
        col, _, val = line.partition(":")
        col_clean = col.strip().lower()
        val_clean = val.strip()
        if not val_clean:
            continue

        for key, prefixes in known_keys.items():
            if any(p in col_clean for p in prefixes):
                doc.metadata[key] = val_clean
                break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ingest_education_csvs()