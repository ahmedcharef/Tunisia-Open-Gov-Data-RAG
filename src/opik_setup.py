"""
Opik (Comet) observability setup for Tunisia RAG.

Initialises the Opik client once at import time and exposes two helpers:
  - track                : @track decorator for tracing functions (no-op when disabled)
  - get_langchain_tracer : returns the shared OpikTracer instance for LangChain callbacks
"""

import logging
from typing import Callable, Optional

from src.config import Config

logger = logging.getLogger("tunisia-rag")

# ── Initialise Opik client ──────────────────────────────────────────────────
_opik_enabled = False
_opik_track = None      # opik.track function
_opik_tracer = None     # shared OpikTracer instance (reused across all calls)

if Config.OPIK_ENABLED:
    try:
        from opik import configure, track as _opik_track
        from opik.integrations.langchain import OpikTracer

        configure(
            project_name=Config.OPIK_PROJECT_NAME,
            workspace=Config.OPIK_WORKSPACE,
            use_local=Config.OPIK_USE_LOCAL,
        )

        # Single shared tracer — survives Streamlit re-runs
        _opik_tracer = OpikTracer(project_name=Config.OPIK_PROJECT_NAME)

        _opik_enabled = True
        logger.info(f"Opik tracing enabled | project='{Config.OPIK_PROJECT_NAME}'")
    except Exception as exc:
        logger.warning(f"Opik initialisation failed — tracing disabled: {exc}")
else:
    logger.info("Opik tracing disabled (OPIK_ENABLED=false)")


# ── Public helpers ──────────────────────────────────────────────────────────

def track(name: Optional[str] = None, **decorator_kwargs) -> Callable:
    """Decorator that wraps a function in an Opik span.

    Falls back to a no-op when Opik is disabled so nothing breaks.

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
    """Return the shared OpikTracer instance, or None when disabled.

    The tracer is created once at startup and reused — this is important for
    Streamlit where the script re-runs on every interaction. Using a shared
    instance ensures traces are consistently sent to the same project.

    Usage::

        tracer = get_langchain_tracer()
        callbacks = [tracer] if tracer else []
        chain.invoke(input, config={"callbacks": callbacks})
    """
    return _opik_tracer


def flush_traces():
    """Flush pending traces immediately.

    Call this after a query in Streamlit to ensure traces are sent before
    the script re-runs. Safe to call even when tracing is disabled.
    """
    if _opik_tracer is not None:
        try:
            _opik_tracer.flush()
        except Exception as exc:
            logger.debug(f"Opik flush failed: {exc}")
