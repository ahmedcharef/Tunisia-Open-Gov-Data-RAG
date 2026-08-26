"""
Utility functions for the Tunisia Open Government Data RAG project.
"""

import logging
from src.config import Config
from src.retriever import get_available_governorates

# ─── Governorate fuzzy map ───────────────────────────────────────────────────
# Maps every known spelling / transliteration / Arabic form to the canonical
# UPPERCASE governorate name stored in Chroma metadata.
# Base: lowercase of canonical name → itself  (e.g. "tunis" → "TUNIS")
_FUZZY_MAP: dict = {gov.lower(): gov for gov in Config.GOVERNORATES}

# French/English accent normalization and abbreviations
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

# Arabic governorate names  (all 24)
_FUZZY_MAP.update({
    "تونس":         "TUNIS",
    "أريانة":       "ARIANA",
    "منوبة":        "MANOUBA",
    "بن عروس":     "BEN AROUS",
    "نابل":         "NABEUL",
    "زغوان":        "ZAGHOUAN",
    "بنزرت":        "BIZERTE",
    "باجة":         "BÉJA",
    "جندوبة":       "JENDOUBA",
    "الكاف":        "LE KEF",
    "سليانة":       "SILIANA",
    "سوسة":         "SOUSSE",
    "المنستير":     "MONASTIR",
    "المهدية":      "MAHDIA",
    "صفاقس":        "SFAX",
    "القيروان":     "KAIROUAN",
    "القصرين":      "KASSERINE",
    "سيدي بوزيد":  "SIDI BOUZID",
    "قابس":         "GABÈS",
    "مدنين":        "MEDENINE",
    "تطاوين":       "TATAOUINE",
    "قفصة":         "GAFSA",
    "توزر":         "TOZEUR",
    "قبلي":         "KÉBILI",
})


# ─── Category label map ──────────────────────────────────────────────────────
# Maps internal ingest-time category values to user-facing labels.
CATEGORY_LABELS: dict = {
    "universite_publique":            "Public University",
    "etablissement_superieur_public": "Public Higher Education",
    "ecole_privee":                   "Private School",
    "ecole_publique":                 "Public School",
    "formation_professionnelle":      "Vocational Training",
    "statistiques_scolaires":         "Education Statistics",
    "programme_social":               "Social Programme",
    "transport_public":               "Public Transport (TRANSTU)",
    "transport_aerien":               "Air Transport (Tunisair)",
    "autre":                          "Other",
}


def extract_gouvernorat(query: str, dataset: str = None) -> str | None:
    """Extract a governorate name from a free-text query using fuzzy matching.

    Strategy:
    1. Check against governorates actually indexed in the active collection.
    2. Fall back to the full 24-governorate map (_FUZZY_MAP) covering French,
       English, and Arabic names/variants.

    Returns UPPERCASE to match how governorates are stored in Chroma metadata.
    """
    if not query:
        return None

    query_lower = query.lower().strip()

    # Strategy 1 — match against what's actually indexed
    available_govs = get_available_governorates(dataset)
    for gov in available_govs:
        if gov.lower() in query_lower:
            return gov  # already uppercase from ingest

    # Strategy 2 — full fuzzy map (French + English + Arabic)
    for key, value in _FUZZY_MAP.items():
        if key in query_lower:
            return value

    return None


def format_source_citation(doc) -> str:
    """Format a clean source citation from document metadata.
    Returns a pipe-separated string, or a generic fallback if metadata is empty.
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

        # type takes priority; fall back to human-readable category label
        if m.get("type"):
            parts.append(f"📚 {m['type']}")
        elif m.get("category"):
            label = CATEGORY_LABELS.get(m["category"], m["category"])
            parts.append(f"📚 {label}")

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
