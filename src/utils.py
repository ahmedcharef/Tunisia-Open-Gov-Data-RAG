"""
Utility functions for the Tunisia Education RAG project.
Includes helper functions for governorate extraction and source citation formatting.
"""

def extract_gouvernorat(query: str) -> str | None:
    """Extract governorate name from user query using simple keyword matching."""
    if not query:
        return None
    
    query_lower = query.lower()
    governorates = {
        "tunis", "sfax", "sousse", "ariana", "ben arous", "manouba", "nabeul",
        "bizerte", "béja", "jendouba", "kairouan", "kasserine", "gafsa", "medenine",
        "gabes", "kebili", "tataouine", "zaghouan", "siliana", "mahdia", "monastir", "tozeur"
    }
    
    for gov in governorates:
        if gov in query_lower:
            return gov.upper()
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