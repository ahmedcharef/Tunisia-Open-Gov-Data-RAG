"""
Utility functions for the Tunisia Education RAG project.
Includes helper functions for governorate extraction and source citation formatting.
"""

import logging
from src.retriever import get_available_governorates

def extract_gouvernorat(query: str) -> str | None:
    """Improved governorate extraction with fuzzy matching."""
    if not query:
        return None
    
    query_lower = query.lower().strip()
    available_govs = get_available_governorates()
    
    # Direct match (case insensitive)
    for gov in available_govs:
        if gov.lower() in query_lower:
            return gov
    
    # Partial / fuzzy match (e.g., "sfaxien", "tunisie", "sous")
    fuzzy_map = {
        "tunis": "Tunis",
        "sfax": "Sfax",
        "sousse": "Sousse",
        "ariana": "Ariana",
        "ben arous": "Ben Arous",
        "manouba": "Manouba",
        "nabeul": "Nabeul",
        "bizerte": "Bizerte",
        "monastir": "Monastir",
        "mahdia": "Mahdia",
        "kairouan": "Kairouan",
        "gafsa": "Gafsa",
        "medenine": "Medenine",
        "béja": "Béja",
        "beja": "Béja",
        "jendouba": "Jendouba",
        "kef": "Le Kef",
        "siliana": "Siliana",
        "zaghouan": "Zaghouan",
        "kasserine": "Kasserine",
        "sidi bouzid": "Sidi Bouzid",
        "gabes": "Gabès",
        "gabès": "Gabès",
        "tataouine": "Tataouine",
        "tozeur": "Tozeur",
        "kebili": "Kébili",
    }
    
    for key, value in fuzzy_map.items():
        if key in query_lower:
            return value
    
    return None

def format_source_citation(doc) -> str:
    """
    Format a clean source citation. Returns a string or empty string if no useful metadata.
    """
    try:
        if not doc or not hasattr(doc, 'metadata'):
            return ""

        m = doc.metadata
        parts = []

        # Name
        if m.get("nom"):
            parts.append(f"🏛️ **{m['nom']}**")

        # Governorate
        if m.get("gouvernorat"):
            parts.append(f"📍 {m['gouvernorat'].title()}")

        # Delegation
        if m.get("delegation"):
            parts.append(f"🏘️ {m['delegation']}")

        # Type / section
        if m.get("type"):
            parts.append(f"📚 {m['type']}")
        elif m.get("category"):
            parts.append(f"📚 {m['category']}")

        # Transport extras
        if m.get("agence"):
            parts.append(f"🚌 {m['agence']}")
        if m.get("zone_geo"):
            parts.append(f"🌍 {m['zone_geo']}")
        if m.get("pays"):
            parts.append(f"🇹🇳 {m['pays']}")

        # Social program stats
        if m.get("nb_familles"):
            parts.append(f"👨‍👩‍👧 {m['nb_familles']} familles")
        if m.get("nb_enfants"):
            parts.append(f"👶 {m['nb_enfants']} enfants")

        # Source file
        if m.get("source_file"):
            parts.append(f"📄 {m['source_file']}")

        if parts:
            return " | ".join(parts)
        else:
            return "📄 Source: Official Tunisian Government Data"

    except Exception as e:
        logging.warning(f"Failed to format citation: {e}")
        return "📄 Source: Tunisian Government Data"


def safe_get_metadata(doc, key: str, default: str = "") -> str:
    """Safely extract metadata value with fallback."""
    try:
        if doc and hasattr(doc, 'metadata'):
            return str(doc.metadata.get(key, default))
        return default
    except Exception:
        return default