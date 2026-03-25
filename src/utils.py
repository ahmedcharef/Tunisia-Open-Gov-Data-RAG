"""
Utility functions for the Tunisia Education RAG project.
"""

def extract_gouvernorat(query: str) -> str | None:
    """Detect governorate name in a user query (case-insensitive)."""
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
    Format a nice source citation from document metadata.
    Used in both CLI and Streamlit responses.
    """
    metadata = doc.metadata
    parts = []

    if metadata.get("nom"):
        parts.append(f"**{metadata['nom']}**")
    if metadata.get("gouvernorat"):
        parts.append(metadata["gouvernorat"])
    if metadata.get("type"):
        parts.append(metadata["type"])
    if metadata.get("source_file"):
        parts.append(f"({metadata['source_file']})")

    return " • ".join(parts) if parts else "Source inconnue"