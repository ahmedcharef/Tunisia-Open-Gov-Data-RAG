"""
Shared RAG service with multi-dataset and governorate filtering.
"""

from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever, get_hybrid_retriever, rerank_documents
from src.utils import extract_gouvernorat, format_source_citation
from src.opik_setup import track, get_langchain_tracer


class RAGService:
    def __init__(self, dataset: str = None):
        self.dataset = dataset or Config.DEFAULT_DATASET
        self.llm = self._initialize_llm()
        self.contextualize_prompt = get_contextualize_prompt()
        self.qa_prompt = get_qa_prompt()

    def _initialize_llm(self):
        try:
            if Config.LLM_PROVIDER == "openrouter":
                return ChatOpenAI(
                    model=Config.OPENROUTER_MODEL,
                    api_key=Config.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    default_headers={
                        "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
                        "X-Title": "Tunisia Multi-Dataset RAG",
                    },
                )
            else:
                return ChatOllama(
                    model=Config.OLLAMA_MODEL,
                    temperature=Config.TEMPERATURE,
                    num_ctx=32768,
                )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _convert_history(self, chat_history: List) -> List:
        messages = []
        for msg in chat_history:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                role, content = msg
            else:
                continue
            if role in ("user", "human"):
                messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                messages.append(AIMessage(content=content))
        return messages

    @track(name="contextualize_query")
    def _contextualize_query(self, user_input: str, history_messages: List) -> str:
        """Rewrite the user's question into a standalone query using chat history."""
        if not history_messages:
            return user_input
        try:
            tracer = get_langchain_tracer()
            callbacks = [tracer] if tracer else []
            chain = self.contextualize_prompt | self.llm | StrOutputParser()
            return chain.invoke(
                {"input": user_input, "chat_history": history_messages},
                config={"callbacks": callbacks},
            )
        except Exception:
            return user_input  # fallback to original question

    @track(name="rag_query")
    def query(
        self,
        user_input: str,
        chat_history: List = None,
        k: int = 8,
        gouvernorat: str = None,
        dataset: str = None,
    ) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []

        try:
            active_dataset = dataset or self.dataset
            gov = gouvernorat or extract_gouvernorat(user_input, dataset=active_dataset)

            logger.info(f"Query: '{user_input}' | gov={gov} | dataset={active_dataset}")

            history_messages = self._convert_history(chat_history)

            # ── Step 1: Contextualize ──────────────────────────────────
            standalone_query = self._contextualize_query(user_input, history_messages)

            # ── Step 2: Retrieve ──────────────────────────────────────
            retrieve_k = k * Config.RERANK_FACTOR if Config.USE_RERANKER else k
            if Config.USE_HYBRID_SEARCH:
                retriever = get_hybrid_retriever(k=retrieve_k, gouvernorat=gov, dataset=active_dataset)
            else:
                retriever = get_retriever(k=retrieve_k, gouvernorat=gov, dataset=active_dataset)

            context_docs = retriever.invoke(standalone_query)
            logger.info(f"Retrieved {len(context_docs)} candidate docs")

            # ── Step 3: Rerank ────────────────────────────────────────
            if Config.USE_RERANKER:
                context_docs = rerank_documents(standalone_query, context_docs, top_k=k)

            # ── Step 4: Generate answer ───────────────────────────────
            tracer = get_langchain_tracer()
            callbacks = [tracer] if tracer else []

            context_text = "\n\n".join(doc.page_content for doc in context_docs)
            filled_prompt = self.qa_prompt.format_messages(
                context=context_text,
                input=user_input,
                chat_history=history_messages,
            )
            answer = (self.llm | StrOutputParser()).invoke(
                filled_prompt,
                config={"callbacks": callbacks},
            )

            sources = [format_source_citation(doc) for doc in context_docs[:6] if format_source_citation(doc)]

            return {
                "answer": answer,
                "sources": sources,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Query failed for dataset {dataset or self.dataset}: {e}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "success": False,
                "error": str(e),
            }


# ── Opik Agent Playground entrypoint ──────────────────────────────────────
# Must be a module-level function (not a class method) for Opik to detect it.
# Defined after RAGService so the class is fully available.

_default_service: RAGService | None = None


def _get_default_service() -> RAGService:
    global _default_service
    if _default_service is None:
        _default_service = RAGService()
    return _default_service


@track(name="tunisia_rag_agent", entrypoint=True)
def tunisia_rag_agent(query: str, dataset: str = None) -> str:
    """Opik-registered entrypoint for the Tunisia RAG agent.

    This module-level function is what the Opik Agent Playground detects.
    It delegates to RAGService.query and returns just the answer string.
    """
    result = _get_default_service().query(user_input=query, dataset=dataset)
    return result.get("answer", "")
