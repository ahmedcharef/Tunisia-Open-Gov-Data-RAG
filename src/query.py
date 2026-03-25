#!/usr/bin/env python3
"""
query.py - Tunisian Education Establishments RAG (CLI)
With source citations and robust error handling
"""

import argparse
import sys
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

import warnings
import logging

from src.config import Config, logger
from src.prompts import get_contextualize_prompt, get_qa_prompt
from src.retriever import get_retriever
from src.utils import extract_gouvernorat, format_source_citation

load_dotenv()

# Silence noisy warnings
warnings.filterwarnings("ignore", message=".*position_ids.*UNEXPECTED.*")
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ====================== LLM SETUP ======================
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

# ====================== PROMPTS & BASE CHAINS ======================
contextualize_prompt = get_contextualize_prompt()
qa_prompt = get_qa_prompt()

# Base chains
base_retriever = get_retriever(k=Config.RETRIEVER_K)
history_aware_retriever = create_history_aware_retriever(llm, base_retriever, contextualize_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ====================== MAIN ======================
def main():
    parser = argparse.ArgumentParser(description="Tunisia Education RAG CLI")
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

            history_aware = create_history_aware_retriever(llm, retriever, contextualize_prompt)
            chain = create_retrieval_chain(history_aware, question_answer_chain)

            response = chain.invoke({"input": args.query, "chat_history": chat_history})
            answer = response["answer"]
            print("\n🤖 Réponse:\n")
            print(answer)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("❌ Une erreur est survenue lors du traitement de votre question.")
        return

    # ====================== Interactive Mode ======================
    print("=" * 80)
    print("   🇹🇳 Tunisia Education RAG - Établissements scolaires & universitaires")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nQuestion > ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Au revoir !")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 Historique effacé.")
                continue

            if not user_input:
                continue

            # Process query with robust error handling
            gov = extract_gouvernorat(user_input)
            current_retriever = get_retriever(k=args.k, gouvernorat=gov)

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
            logger.error(f"Unexpected error in interactive mode: {e}")
            print("❌ Une erreur inattendue est survenue. Veuillez réessayer.")

if __name__ == "__main__":
    main()