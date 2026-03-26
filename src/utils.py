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
    Format a clean and attractive source citation with emojis.
    Returns a nicely formatted string or empty string if no useful metadata.
    """
    try:
        if not doc or not hasattr(doc, 'metadata'):
            return ""

        metadata = doc.metadata
        parts = []

        # Main establishment name (bold)
        if metadata.get("nom"):
            parts.append(f"🏛️ **{metadata['nom']}**")
        elif metadata.get("name"):
            parts.append(f"🏛️ **{metadata['name']}**")

        # Governorate
        if metadata.get("gouvernorat"):
            parts.append(f"📍 {metadata['gouvernorat']}")
        elif metadata.get("governorate"):
            parts.append(f"📍 {metadata['governorate']}")

        # Type of establishment
        if metadata.get("type"):
            parts.append(f"📚 {metadata['type']}")
        elif metadata.get("category"):
            parts.append(f"📚 {metadata['category']}")

        # Source file (for traceability)
        if metadata.get("source_file"):
            parts.append(f"📄 {metadata['source_file']}")

        if parts:
            return " | ".join(parts)
        else:
            return "📄 Source: Official Tunisian Education Data"

    except Exception as e:
        # Graceful fallback - never break the UI/CLI due to citation formatting
        logging.warning(f"Failed to format citation: {e}")
        return "📄 Source: Tunisian Government Education Data"


def safe_get_metadata(doc, key: str, default: str = "") -> str:
    """Safely extract metadata value with fallback."""
    try:
        if doc and hasattr(doc, 'metadata'):
            return str(doc.metadata.get(key, default))
        return default
    except Exception:
        return default