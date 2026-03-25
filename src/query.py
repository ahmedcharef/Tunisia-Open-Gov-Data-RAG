#!/usr/bin/env python3
"""
query.py - Tunisian Education RAG (LCEL Version)
Clean, modern implementation without langchain-classic
"""

import argparse
import sys
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever
from src.utils import extract_gouvernorat

load_dotenv()

# Silence noisy warnings
warnings.filterwarnings("ignore", message=".*position_ids.*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ====================== VECTOR STORE ======================
embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=Config.CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=Config.COLLECTION_NAME,
)

# ====================== LLM ======================
def get_llm():
    try:
        if Config.LLM_PROVIDER == "openrouter":
            return ChatOpenAI(
                model=Config.OPENROUTER_MODEL,
                api_key=Config.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                model_kwargs={
                    "extra_headers": {
                        "HTTP-Referer": "https://github.com/ahmedcharef/Tunisia-Open-Gov-Data-RAG",
                        "X-Title": "Tunisia Education RAG",
                    }
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


llm = get_llm()

# ====================== PROMPTS ======================
contextualize_prompt = get_contextualize_prompt()
qa_prompt = get_qa_prompt()

# ====================== LCEL CHAIN BUILDER ======================

def create_rag_chain(retriever):
    """Create a full RAG chain using LCEL."""

    # Step 1: Contextualize the question based on history
    contextualize_chain = (
        contextualize_prompt
        | llm
        | StrOutputParser()
    )

    # Step 2: Main RAG pipeline
    rag_chain = (
        {
            "context": contextualize_chain | retriever,
            "input": RunnablePassthrough(),
            "chat_history": RunnablePassthrough(),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(
        description="Tunisia Education RAG CLI (LCEL)")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        try:
            from src.retriever import get_vectorstore_stats
            stats = get_vectorstore_stats()
            print(f"📊 Total establishments in database: {stats['total_documents']:,}")
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            print("❌ Could not retrieve statistics.")
        return

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        try:
            gov = extract_gouvernorat(args.query)
            retriever = get_retriever(k=args.k, gouvernorat=gov)
            chain = create_rag_chain(retriever)

            answer = chain.invoke({
                "input": args.query,
                "chat_history": chat_history
            })
            print("\n🤖 Response:\n")
            print(answer)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("❌ An error occurred while processing your question.")
        return

    # ====================== Interactive Mode ======================
    print("=" * 80)
    print("   🇹🇳 Tunisia Education RAG - Educational Institutions")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nQuestion (en/fr/ar) > ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 Chat history cleared.")
                continue

            if not user_input:
                continue

            gov = extract_gouvernorat(user_input)
            retriever = get_retriever(k=args.k, gouvernorat=gov)
            chain = create_rag_chain(retriever)

            answer = chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            print(f"\n🤖 Response:\n{answer}\n")

            chat_history.extend([("human", user_input), ("ai", answer)])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\n👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print("❌ An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    main()
