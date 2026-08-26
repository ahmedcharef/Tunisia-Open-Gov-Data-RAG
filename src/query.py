#!/usr/bin/env python3
"""
query.py - Command Line Interface for Tunisia Multi-Dataset RAG
"""

import argparse
from typing import List, Tuple

from src.config import Config, logger
from src.rag_service import RAGService
from src.retriever import get_vectorstore_stats, get_available_governorates

def main():
    parser = argparse.ArgumentParser(description="Tunisia Multi-Dataset RAG CLI")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--dataset", type=str, default=Config.DEFAULT_DATASET,
                        choices=list(Config.DATASETS.keys()),
                        help="Dataset to use (education, agriculture, etc.)")
    args = parser.parse_args()

    service = RAGService(dataset=args.dataset)

    if args.stats:
        try:
            stats = get_vectorstore_stats(args.dataset)
            print(f"📊 Dataset  : {args.dataset}")
            print(f"   Collection: {stats.get('collection_name', 'N/A')}")
            print(f"   Chunks    : {stats.get('total_documents', 0):,}")
            row_count = stats.get('source_row_count')
            print(f"   Rows      : {row_count:,}" if row_count else "   Rows      : — (re-ingest to compute)")
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            print("❌ Could not retrieve statistics.")
        return

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        try:
            result = service.query(args.query, chat_history, k=args.k)
            print("\n🤖 Response:\n")
            print(result["answer"])

            if result.get("sources"):
                print("\n📚 Sources:")
                for source in result["sources"]:
                    print(f"   • {source}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print("❌ An error occurred while processing your question.")
        return

    # Interactive Mode
    print("=" * 85)
    print(f"   🇹🇳 Tunisia RAG - Dataset: {args.dataset.upper()}")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 85)

    try:
        available_govs = get_available_governorates(args.dataset)
        if available_govs:
            print(f"Available Governorates: {', '.join(available_govs[:10])} ...")
            print("-" * 85)
    except Exception:
        pass

    while True:
        try:
            user_input = input("\nQuestion > ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 Chat history cleared.")
                continue

            if not user_input:
                continue

            result = service.query(user_input, chat_history, k=args.k)

            print(f"\n🤖 Response:\n{result['answer']}\n")

            if result.get("sources"):
                print("📚 Sources:")
                for source in result["sources"]:
                    print(f"   • {source}")
                print("")

            chat_history.extend([("human", user_input), ("ai", result["answer"])])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\n👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("❌ An error occurred. Please try again.")

if __name__ == "__main__":
    main()