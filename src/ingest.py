#!/usr/bin/env python3
"""
ingest.py - Ingestion pipeline for Tunisian open government data (CSV + XLSX).

Supports:
- CSV files with auto-detected delimiters (comma or semicolon)
- XLSX files via pandas + openpyxl
- Per-file category tagging and metadata enrichment (governorate, nom, type…)
- Chunked embedding into a named Chroma collection
"""

import logging
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Config, logger

# Silence noisy logs
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*position_ids.*")

load_dotenv()

# ────────────────────────────────────────────────
# Chunking settings
# ────────────────────────────────────────────────
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 180


# ────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────

def ingest_education_csvs(dataset: str = None) -> None:
    """Main ingestion pipeline — loads all CSV and XLSX files in data/.
    
    Args:
        dataset: Dataset key from Config.DATASETS (e.g. 'education', 'agriculture').
                 Defaults to Config.DEFAULT_DATASET.
    """
    dataset = dataset or Config.DEFAULT_DATASET

    if dataset not in Config.DATASETS:
        logger.error(f"Unknown dataset '{dataset}'. Available: {list(Config.DATASETS.keys())}")
        return

    collection_name = Config.get_collection_name(dataset)
    logger.info(f"Ingesting into dataset='{dataset}' → collection='{collection_name}'")

    data_dir = Path(Config.DATA_DIR)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Please place your data files in the 'data/' folder.")
        return

    csv_files = sorted(data_dir.glob("*.csv"))
    xlsx_files = sorted(data_dir.glob("*.xlsx"))
    all_file_paths = csv_files + xlsx_files

    if not all_file_paths:
        logger.error(f"No CSV or XLSX files found in {data_dir}")
        return

    # Filter to only the files assigned to this dataset
    allowed = Config.DATASET_FILES.get(dataset)
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [f for f in all_file_paths if f.name in allowed_set]
        missing = allowed_set - {f.name for f in all_file_paths}
        if missing:
            logger.warning(f"Expected files not found in data/: {missing}")
        all_file_paths = filtered

    if not all_file_paths:
        logger.error(f"No matching files found for dataset '{dataset}'. Check Config.DATASET_FILES.")
        return

    logger.info(f"Files to ingest ({len(all_file_paths)}): {[f.name for f in all_file_paths]}")

    logger.info(f"Found {len(csv_files)} CSV and {len(xlsx_files)} XLSX file(s)")

    all_docs: List[Document] = []

    for file_path in all_file_paths:
        logger.info(f"Loading: {file_path.name}")
        try:
            if file_path.suffix.lower() == ".xlsx":
                docs = _load_xlsx(file_path)
            else:
                docs = _load_csv(file_path)

            category = _guess_category_from_filename(file_path.name)
            for doc in docs:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["category"] = category
                _extract_common_fields_to_metadata(doc)

            logger.info(f"  → Loaded {len(docs):,} rows")
            all_docs.extend(docs)

        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    if not all_docs:
        logger.error("No documents were loaded.")
        return

    # ─── Deduplicate ────────────────────────────────────────────
    seen: set = set()
    unique_docs: List[Document] = []
    for doc in all_docs:
        key = doc.page_content.strip()
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    logger.info(f"After deduplication: {len(unique_docs):,} documents (removed {len(all_docs) - len(unique_docs):,} duplicates)")
    all_docs = unique_docs

    # ─── Split Documents ────────────────────────────────────────
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " - ", " | ", " • "],
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)
    logger.info(f"Created {len(chunks):,} chunks")

    # ─── Embed & Store ───────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    logger.info("Creating / updating Chroma collection...")
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 40,
            "hnsw:M": 16,
            "source_row_count": len(all_docs),
            "chunk_count": total_chunks,
        },
    )

    batch_size = 400
    total_chunks = len(chunks)
    with tqdm(total=total_chunks, desc="Indexing", unit="chunk") as pbar:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            vectorstore.add_documents(batch)
            pbar.update(len(batch))

    logger.info(
        f"✅ Ingestion completed successfully!\n"
        f"   Collection : {collection_name}\n"
        f"   Documents  : {len(all_docs):,}\n"
        f"   Chunks     : {total_chunks:,}\n"
        f"   Storage    : {Config.CHROMA_PERSIST_DIR}"
    )


# ────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────

def _load_csv(file_path: Path) -> List[Document]:
    """Load a CSV file into Documents, auto-detecting delimiter and encoding."""
    delimiter = _detect_delimiter(file_path)
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(
                file_path, sep=delimiter, encoding=encoding,
                dtype=str, on_bad_lines="skip"
            )
            return _dataframe_to_documents(df, file_path)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {file_path.name} with any supported encoding")


def _load_xlsx(file_path: Path) -> List[Document]:
    """Load an XLSX file into Documents."""
    df = pd.read_excel(file_path, dtype=str)
    return _dataframe_to_documents(df, file_path)


def _dataframe_to_documents(df: pd.DataFrame, file_path: Path) -> List[Document]:
    """Convert a DataFrame to a list of LangChain Documents with pre-extracted metadata."""
    df = df.dropna(how="all").fillna("")

    # Build a column→canonical_field mapping once per file
    col_map = _build_column_map(df.columns.tolist())

    docs: List[Document] = []
    for _, row in df.iterrows():
        content = "\n".join(
            f"{col}: {str(val).strip()}"
            for col, val in row.items()
            if str(val).strip() not in ("", "nan", "None")
        )
        if not content.strip():
            continue

        # Extract metadata directly from the row using the column map
        metadata = _extract_metadata_from_row(row, col_map)
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def _detect_delimiter(file_path: Path) -> str:
    """Return ';' or ',' based on whichever appears more in the first line."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                first_line = f.readline()
            return ";" if first_line.count(";") >= first_line.count(",") else ","
        except UnicodeDecodeError:
            continue
    return ","


# ────────────────────────────────────────────────
# Metadata helpers
# ────────────────────────────────────────────────

# Maps a canonical field name to all known column aliases (lowercased, stripped).
# Order within each list matters: first match wins.
_COLUMN_ALIASES: dict = {
    "gouvernorat": [
        "gouvernorat", "governorate", "wilaya", "région", "region",
        # Arabic
        "الولاية",
        # Regional delegation used as proxy for governorate in school CSVs
        "المندوبية الجهوية للتربية",
        # BAC / stats files use CRE (centre régional d'examen) as region proxy
        "cre",
    ],
    "nom": [
        "nom de l'établissement", "etablissement", "label_fr", "nom",
        "university_fr", "nom de la station", "destination", "label",
        "name",
        # Arabic
        "إسم المؤسّسة", "المؤسسة", "إسم المؤسسة",
    ],
    "nom_ar": [
        "label_ar", "university_ar",
        # Arabic name variants
        "المؤسسة",
    ],
    "type": [
        "type d'établissement", "type de la station", "type", "category",
        "نوع المؤسسة",
        # BAC stats
        "section", "candidature",
    ],
    "delegation": [
        "délégation", "delegation",
        # Arabic — with and without the definite article (ال)
        "معتمدية", "المعتمدية",
    ],
    "adresse": [
        "adresse", "address", "العنوان",
    ],
    "lat": ["lat", "latitude", "خط العرض"],
    "lon": ["lon", "longitude", "خط الطول"],
    "website": ["website", "siteweb"],
    # Transport-specific
    "zone_geo": [
        "zone geographique", "zone géographique",
    ],
    "pays": ["pays", "country"],
    "agence": ["agence", "agency"],
    # Social program numeric fields (stored as strings for metadata)
    "nb_familles": ["عدد العائلات"],
    "nb_enfants":  ["عدد الأطفال"],
    # BAC / stats
    "genre": ["genre", "gender"],
}

# Reverse lookup: lowercased column → canonical field
_ALIAS_LOOKUP: dict = {}
for _field, _aliases in _COLUMN_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_LOOKUP[_alias.lower().strip()] = _field


def _build_column_map(columns: list) -> dict:
    """
    Return {original_col_name: canonical_field} for columns that match a known alias.
    Unrecognized columns are skipped.
    """
    mapping = {}
    for col in columns:
        canonical = _ALIAS_LOOKUP.get(col.lower().strip())
        if canonical and canonical not in mapping.values():
            mapping[col] = canonical
    return mapping


def _extract_metadata_from_row(row, col_map: dict) -> dict:
    """Build a metadata dict from a DataFrame row using the pre-built column map."""
    metadata: dict = {}
    for col, field in col_map.items():
        val = str(row.get(col, "")).strip()
        if val and val not in ("nan", "None", ""):
            # Normalize governorate to uppercase so retriever filters work consistently
            if field == "gouvernorat":
                val = val.upper()
            metadata[field] = val
    return metadata


def _guess_category_from_filename(filename: str) -> str:
    name = filename.lower()
    if "universit" in name or "etatiqu" in name:
        return "universite_publique"
    if "enseignement-superieur" in name or "superieur" in name:
        return "etablissement_superieur_public"
    if "priv" in name and ("scolaire" in name or "etablissement" in name):
        return "ecole_privee"
    if "public" in name and "scolaire" in name:
        return "ecole_publique"
    if "formation" in name and ("agricol" in name or "professionnel" in name):
        return "formation_professionnelle"
    if "bac" in name or "concours" in name or "lycee" in name or "lycées" in name:
        return "statistiques_scolaires"
    if "allocation" in name or "enfant" in name:
        return "programme_social"
    if "transtu" in name or "bus" in name or "station" in name or "arret" in name or "arrêt" in name:
        return "transport_public"
    if "tunisair" in name or "reseau" in name or "réseau" in name:
        return "transport_aerien"
    return "autre"


def _extract_common_fields_to_metadata(doc: Document) -> None:
    """
    Fallback: enrich metadata by re-parsing page_content for files where
    column-map extraction did not populate a governorate (e.g. plain-text rows).
    Only fills fields not already set by _extract_metadata_from_row.
    """
    if "gouvernorat" in doc.metadata:
        return  # already extracted at load time — nothing to do

    content = doc.page_content
    lines = [line.strip() for line in content.split("\n") if ":" in line]

    for line in lines:
        col, _, val = line.partition(":")
        col_clean = col.strip().lower()
        val_clean = val.strip()
        if not val_clean or val_clean in ("nan", "none", ""):
            continue

        canonical = _ALIAS_LOOKUP.get(col_clean)
        if canonical and canonical not in doc.metadata:
            doc.metadata[canonical] = val_clean.upper() if canonical == "gouvernorat" else val_clean


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Tunisia RAG — Ingestion pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=Config.DEFAULT_DATASET,
        choices=list(Config.DATASETS.keys()),
        help=f"Target dataset/collection to ingest into (default: {Config.DEFAULT_DATASET})",
    )
    args = parser.parse_args()
    ingest_education_csvs(dataset=args.dataset)
