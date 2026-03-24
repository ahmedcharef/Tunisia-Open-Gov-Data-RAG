#!/usr/bin/env python3
"""
query.py - Tunisian Education Establishments RAG
Clean, maintainable version with best practices
"""

import argparse
import sys
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Legacy chains (kept for stability - can migrate to LCEL later)
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import warnings
import logging

# Clean imports from src package
from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt

load_dotenv()

# Silence noisy warnings
warnings.filterwarnings("ignore", message=".*position_ids.*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ====================== LOAD VECTOR STORE ======================
try:
    embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=Config.CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=Config.COLLECTION_NAME,   # Now from Config
    )
    logger.info(f"✅ Vector store loaded | Collection: {Config.COLLECTION_NAME}")
except Exception as e:
    logger.error(f"Failed to load vector store: {e}")
    print("❌ Could not load the database. Please run `python ingest.py` first.")
    sys.exit(1)


# ====================== LLM SETUP ======================
def get_llm():
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


llm = get_llm()
logger.info(f"LLM initialized → {Config.OPENROUTER_MODEL if Config.LLM_PROVIDER == 'openrouter' else Config.OLLAMA_MODEL}")


# ====================== PROMPTS ======================
contextualize_prompt = get_contextualize_prompt()
qa_prompt = get_qa_prompt()


# ====================== RETRIEVER ======================
def get_retriever(k: int = 8, gouvernorat: str = None):
    """Return retriever with optional metadata filter"""
    search_kwargs = {"k": k}
    if gouvernorat:
        search_kwargs["filter"] = {"gouvernorat": {"$eq": gouvernorat.upper()}}

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )


# Create the base retriever once (best practice)
base_retriever = get_retriever(k=Config.RETRIEVER_K)

# Create chains once (avoid rebuilding every query)
history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ====================== HELPER ======================
def extract_gouvernorat(query: str) -> str | None:
    """Detect governorate name in user query"""
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


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(description="Tunisia Education RAG CLI")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K, help="Number of documents to retrieve")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()

    if args.stats:
        count = vectorstore._collection.count()
        print(f"📊 Total establishments in database: {count:,}")
        return

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        gov = extract_gouvernorat(args.query)
        # Use dynamic retriever only for single query
        retriever = get_retriever(k=args.k, gouvernorat=gov)
        temp_history_aware = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        temp_chain = create_retrieval_chain(temp_history_aware, question_answer_chain)

        response = temp_chain.invoke({"input": args.query, "chat_history": chat_history})
        print("\n🤖 Réponse:\n")
        print(response["answer"])
        return

    # ====================== Interactive Mode ======================
    print("=" * 78)
    print("   🇹🇳 Tunisia Education RAG - Établissements scolaires & universitaires")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 78)

    while True:
        try:
            user_input = input("\nQuestion (fr/ar) > ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Au revoir !")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 Historique de conversation effacé.")
                continue

            if not user_input:
                continue

            # Auto-detect governorate
            gov = extract_gouvernorat(user_input)
            current_retriever = get_retriever(k=args.k, gouvernorat=gov)

            # Build chain with current retriever
            current_history_retriever = create_history_aware_retriever(
                llm, current_retriever, contextualize_prompt
            )
            current_rag_chain = create_retrieval_chain(
                current_history_retriever, question_answer_chain
            )

            response = current_rag_chain.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            answer = response["answer"]
            print(f"\n🤖 Réponse:\n{answer}\n")

            chat_history.extend([("human", user_input), ("ai", answer)])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\n👋 Arrêt par l'utilisateur.")
            break
        except Exception as e:
            logger.error(f"Error during query: {e}")
            print("❌ Une erreur est survenue. Veuillez réessayer.")

if __name__ == "__main__":
    main()