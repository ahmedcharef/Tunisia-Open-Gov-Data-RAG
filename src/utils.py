"""
Utility functions for the Tunisia Open Government Data RAG project.
"""

import logging
from src.config import Config
from src.retriever import get_available_governorates

# Fuzzy aliases — maps common misspellings / transliterations / partials to
# the canonical UPPERCASE governorate name used in Chroma metadata.
# Base: lowercase of every canonical name → itself (e.g. "tunis" → "TUNIS")
_FUZZY_MAP: dict = {gov.lower(): gov for gov in Config.GOVERNORATES}

# Extra variants: accent normalization, abbreviations, partial names
_FUZZY_MAP.update({
    "beja":        "BÉJA",
    "béja":        "BÉJA",
    "gabes":       "GABÈS",
    "gabès":       "GABÈS",
    "kebili":      "KÉBILI",
    "kébili":      "KÉBILI",
    "kef":         "LE KEF",
    "le kef":      "LE KEF",
    "sidi bouzid": "SIDI BOUZID",
    "ben arous":   "BEN AROUS",
})


def extract_gouvernorat(query: str, dataset: str = None) -> str | None:
    """Extract a governorate name from a free-text query using fuzzy matching.

    Strategy:
    1. Check against governorates actually indexed in the active collection.
    2. Fall back to the full list of 24 Tunisian governorates via _FUZZY_MAP.

    Returns UPPERCASE to match how governorates are stored in Chroma metadata.
    """
    if not query:
        return None

    query_lower = query.lower().strip()

    # Strategy 1 — match against what's actually in the collection
    available_govs = get_available_governorates(dataset)
    for gov in available_govs:
        if gov.lower() in query_lower:
            return gov  # already uppercase from ingest

    # Strategy 2 — fuzzy map over all 24 governorates
    for key, value in _FUZZY_MAP.items():
        if key in query_lower:
            return value

    return None


def format_source_citation(doc) -> str:
    """Format a clean source citation from document metadata.
    Returns a pipe-separated string or a generic fallback.
    """
    try:
        if not doc or not hasattr(doc, "metadata"):
            return ""

        m = doc.metadata
        parts = []

        if m.get("nom"):
            parts.append(f"🏛️ **{m['nom']}**")
        if m.get("gouvernorat"):
            parts.append(f"📍 {m['gouvernorat'].title()}")
        if m.get("delegation"):
            parts.append(f"🏘️ {m['delegation']}")
        if m.get("type"):
            parts.append(f"📚 {m['type']}")
        elif m.get("category"):
            parts.append(f"📚 {m['category']}")
        if m.get("agence"):
            parts.append(f"🚌 {m['agence']}")
        if m.get("zone_geo"):
            parts.append(f"🌍 {m['zone_geo']}")
        if m.get("pays"):
            parts.append(f"🇹🇳 {m['pays']}")
        if m.get("nb_familles"):
            parts.append(f"👨‍👩‍👧 {m['nb_familles']} familles")
        if m.get("nb_enfants"):
            parts.append(f"👶 {m['nb_enfants']} enfants")
        if m.get("source_file"):
            parts.append(f"📄 {m['source_file']}")

        return " | ".join(parts) if parts else "📄 Source: Official Tunisian Government Data"

    except Exception as e:
        logging.warning(f"Failed to format citation: {e}")
        return "📄 Source: Tunisian Government Data"


def safe_get_metadata(doc, key: str, default: str = "") -> str:
    """Safely extract a metadata value with a fallback default."""
    try:
        if doc and hasattr(doc, "metadata"):
            return str(doc.metadata.get(key, default))
        return default
    except Exception:
        return default
