"""
ingest.py

Ingests Tunisian education establishment CSV files into Chroma vector store.
Designed for the four specific files you mentioned (March 2025–2026 versions).

Usage:
    python ingest.py
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

from config import Config, logger
from tqdm import tqdm

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
load_dotenv()

# ────────────────────────────────────────────────
#  Configuration overrides / constants for this dataset
# ────────────────────────────────────────────────

COLLECTION_NAME = "tn_education_etablissements_2025"

# Most Tunisian government CSVs from data.gov.tn use ; as delimiter
CSV_DELIMITER = ";"

# Increase chunk size a bit since school/university records are usually short
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180

# ────────────────────────────────────────────────
def ingest_education_csvs() -> None:
    """
    Load → enrich metadata → chunk → embed → store in Chroma
    """
    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    # Find all relevant CSV files (you can make this list more strict if needed)
    csv_patterns = [
        "Etablissements-publics-enseignement-superieur-en-Tunisie.csv",
        "Les-Universites-Etatiques-Publiques-en-Tunisie.csv",
        "liste-des-etablissements-scolaires-prives.csv",
        "liste-des-etablissements-scolaires-publics.csv",
        "*.csv",  # fallback — in case filenames are slightly different
    ]

    all_files: List[Path] = []
    for pattern in csv_patterns:
        found = list(data_dir.glob(pattern))
        all_files.extend(found)

    if not all_files:
        logger.error(f"No CSV files found in {data_dir}")
        logger.info("Expected files:")
        for p in csv_patterns:
            logger.info(f"  • {p}")
        return

    logger.info(f"Found {len(all_files)} CSV file(s)")

    all_docs = []

    for file_path in all_files:
        logger.info(f"Processing: {file_path.name}")

        try:
            loader = CSVLoader(
                file_path=str(file_path),
                encoding="utf-8",
                csv_args={"delimiter": CSV_DELIMITER},
            )
            docs = loader.load()

            # Enrich metadata with source filename and category inference
            category = _guess_category_from_filename(file_path.name)

            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["category"] = category
                # Try to extract gouvernorat / délégation / type if columns exist
                _extract_common_fields_to_metadata(doc)

            logger.info(f"  → loaded {len(docs):,} rows")
            all_docs.extend(docs)

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    if not all_docs:
        logger.error("No documents were loaded from any file.")
        return

    # ─── Split ────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " - ", " | ", " • ", " "],
        add_start_index=True,
    )

   # ─── Embed & store ────────────────────────────────
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks):,} text chunks")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # "cuda" if you have GPU + proper torch setup
    )

    logger.info("Creating / updating Chroma collection...")

    # 1. Create or get the collection with tuned HNSW params
    #    (lower ef_construction = faster build, but slightly lower quality graph)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,           # required even if pre-embedded
        collection_name=COLLECTION_NAME,
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 40,          # default is ~100–200 → 40 is aggressive speedup
            "hnsw:M": 16,                        # default is usually fine (8–64 range common)
            # Optional: "hnsw:num_threads": 4    # if your machine has multiple cores
        },
    )

    # 2. Add documents in batches with progress bar
    batch_size = 400  # 200–800 is often sweet spot (depends on RAM; lower = safer)
    total_chunks = len(chunks)

    with tqdm(total=total_chunks, desc="Indexing batches", unit="chunk") as pbar:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    logger.info(
        f"Ingestion completed\n"
        f"  Collection:     {COLLECTION_NAME}\n"
        f"  Chunks indexed: {total_chunks:,}\n"
        f"  Storage:        {Config.CHROMA_PERSIST_DIR}"
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
    """
    Try to move important fields from page_content → metadata
    (makes filtering later much easier)
    """
    content = doc.page_content
    lines = [line.strip() for line in content.split("\n") if ":" in line]

    known_keys = {
        "gouvernorat": ["gouvernorat", "gouvernorat :"],
        "delegation": ["délégation", "delegation :"],
        "type": ["type d'établissement", "type :"],
        "nom": ["nom de l'établissement", "nom :"],
        "adresse": ["adresse", "adresse :"],
    }

    for line in lines:
        for key, prefixes in known_keys.items():
            for prefix in prefixes:
                if line.lower().startswith(prefix):
                    value = line[len(prefix):].strip(" :").strip()
                    if value:
                        doc.metadata[key] = value
                    break


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ingest_education_csvs()