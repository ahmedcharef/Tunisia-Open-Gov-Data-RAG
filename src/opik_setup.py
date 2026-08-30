"""
Opik (Comet) observability setup for Tunisia RAG.

Initialises the Opik client once at import time and exposes two helpers:
  - track             : @track decorator for tracing functions (no-op when disabled)
  - get_langchain_tracer : returns an OpikTracer callback for LangChain chains
"""

import logging
from typing import Callable, Optional

from src.config import Config

logger = logging.getLogger("tunisia-rag")

# ── Initialise Opik client ──────────────────────────────────────────────────
_opik_enabled = False
_opik_track = None  # will be set to opik.track if init succeeds

if Config.OPIK_ENABLED:
    try:
        from opik import configure, track as _opik_track
        from opik.integrations.langchain import OpikTracer  # noqa: F401

        configure(
            project_name=Config.OPIK_PROJECT_NAME,
            workspace=Config.OPIK_WORKSPACE,
            use_local=Config.OPIK_USE_LOCAL,
        )

        _opik_enabled = True
        logger.info(f"Opik tracing enabled | project='{Config.OPIK_PROJECT_NAME}'")
    except Exception as exc:
        logger.warning(f"Opik initialisation failed — tracing disabled: {exc}")
else:
    logger.info("Opik tracing disabled (OPIK_ENABLED=false)")


# ── Public helpers ──────────────────────────────────────────────────────────

def track(name: Optional[str] = None, **decorator_kwargs) -> Callable:
    """Decorator that wraps a function in an Opik span.

    Falls back to a transparent no-op when Opik is not enabled / not available,
    so the rest of the codebase never needs to guard against import errors.

    Usage::

        @track(name="my_span")
        def my_function(...): ...

        @track(name="rag_query", entrypoint=True)
        def agent_entry(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        if not _opik_enabled or _opik_track is None:
            return fn  # no-op passthrough
        try:
            span_name = name or fn.__qualname__
            return _opik_track(name=span_name, **decorator_kwargs)(fn)
        except Exception:
            return fn  # graceful fallback

    # Support both @track and @track(name="...") call styles
    if callable(name):
        fn, name = name, None
        return decorator(fn)

    return decorator


def get_langchain_tracer():
    """Return an OpikTracer for LangChain callbacks, or None when disabled.

    Usage::

        tracer = get_langchain_tracer()
        callbacks = [tracer] if tracer else []
        chain.invoke(input, config={"callbacks": callbacks})
    """
    if not _opik_enabled:
        return None
    try:
        from opik.integrations.langchain import OpikTracer
        return OpikTracer()
    except Exception as exc:
        logger.debug(f"Could not create OpikTracer: {exc}")
        return None
