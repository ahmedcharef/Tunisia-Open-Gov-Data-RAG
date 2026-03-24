"""
Tunisia Education RAG - Main Package
A Retrieval-Augmented Generation system over Tunisian open government education data.
"""

__version__ = "0.2.0"
__author__ = "Ahmed Charef"

# Expose key components for clean imports
from .config import Config, logger
from .prompts import (
    get_contextualize_prompt,
    get_qa_prompt,
    get_education_system_prompt,
    contextualize_q_prompt,
    qa_prompt,
)

# Optional: expose main entry points
from .query import main as run_cli

__all__ = [
    "Config",
    "logger",
    "get_contextualize_prompt",
    "get_qa_prompt",
    "get_education_system_prompt",
    "contextualize_q_prompt",
    "qa_prompt",
    "run_cli",
]