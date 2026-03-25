# src/rag_service.py
"""
Shared RAG service layer for both CLI and Streamlit interfaces.
Centralizes chain creation and query execution.
"""

from typing import Dict, Any, List, Tuple

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever
from src.utils import extract_gouvernorat, format_source_citation


class RAGService:
    """Shared service for RAG operations."""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.contextualize_prompt = get_contextualize_prompt()
        self.qa_prompt = get_qa_prompt()

    def _initialize_llm(self):
        """Initialize LLM with fallback."""
        try:
            if Config.LLM_PROVIDER == "openrouter":
                return ChatOpenAI(
                    model=Config.OPENROUTER_MODEL,
                    api_key=Config.OPENROUTER_API_KEY,
                    base_url="https://openrouter.ai/api/v1",
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
                        "X-Title": "Tunisia Education RAG",
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

    def create_rag_chain(self, retriever):
        """Create LCEL RAG chain that returns both answer and context."""
        contextualize_chain = self.contextualize_prompt | self.llm | StrOutputParser()

        rag_chain = (
            RunnableParallel({
                "context": contextualize_chain | retriever,
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
            })
            | RunnableParallel({
                "answer": self.qa_prompt | self.llm | StrOutputParser(),
                "context": RunnablePassthrough() | (lambda x: x["context"])
            })
        )
        return rag_chain

    def query(self, user_input: str, chat_history: List[Tuple[str, str]], k: int = 8) -> Dict[str, Any]:
        """Execute a query and return answer + sources."""
        try:
            gov = extract_gouvernorat(user_input)
            retriever = get_retriever(k=k, gouvernorat=gov)

            chain = self.create_rag_chain(retriever)

            result = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            answer = result["answer"]
            context_docs = result.get("context", [])

            # Format citations
            sources = []
            for doc in context_docs[:6]:  # Limit to 6 sources
                citation = format_source_citation(doc)
                if citation:
                    sources.append(citation)

            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "success": False,
                "error": str(e)
            }