"""
Shared RAG service with multi-dataset and governorate filtering.
"""

from typing import Dict, Any, List, Tuple
from operator import itemgetter

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever
from src.utils import extract_gouvernorat, format_source_citation


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

    def create_rag_chain(self, retriever):
        contextualize_chain = self.contextualize_prompt | self.llm | StrOutputParser()

        rag_chain = (
            RunnableParallel({
                "context": contextualize_chain | retriever,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            })
            | RunnableParallel({
                "answer": self.qa_prompt | self.llm | StrOutputParser(),
                "context": itemgetter("context"),
            })
        )
        return rag_chain

    def query(self, user_input: str, chat_history: List = None, k: int = 8, gouvernorat: str = None, dataset: str = None) -> Dict[str, Any]:
        if chat_history is None:
            chat_history = []

        try:
            # Use passed gouvernorat or extract from query
            gov = gouvernorat or extract_gouvernorat(user_input)
            
            logger.info(f"Query: '{user_input}' | Governorate filter: {gov} | Dataset: {dataset or self.dataset}")

            retriever = get_retriever(k=k, gouvernorat=gov, dataset=dataset or self.dataset)

            chain = self.create_rag_chain(retriever)
            history_messages = self._convert_history(chat_history)

            result = chain.invoke({
                "input": user_input,
                "chat_history": history_messages
            })

            answer = result["answer"]
            context_docs = result.get("context", [])

            logger.info(f"Retrieved {len(context_docs)} documents for this query")

            sources = [format_source_citation(doc) for doc in context_docs[:6] if format_source_citation(doc)]

            return {
                "answer": answer,
                "sources": sources,
                "success": True
            }

        except Exception as e:
            logger.error(f"Query failed for dataset {dataset or self.dataset}: {e}")
            return {
                "answer": "An error occurred while processing your question. Please try again.",
                "sources": [],
                "success": False,
                "error": str(e)
            }