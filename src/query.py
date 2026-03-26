#!/usr/bin/env python3
"""
query.py - Command Line Interface for Tunisia Education RAG

Features:
- Single query mode (--query)
- Interactive chat mode
- Dynamic governorate-based filtering
- Source citations
- Database statistics (--stats)
"""

import argparse
import sys
from typing import List, Tuple

from src.config import Config, logger
from src.rag_service import RAGService
from src.retriever import get_vectorstore_stats, get_available_governorates

def main():
    parser = argparse.ArgumentParser(description="Tunisia Education RAG CLI")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K, help="Number of documents to retrieve")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()

    service = RAGService()

    # ====================== STATS MODE ======================
    if args.stats:
        try:
            stats = get_vectorstore_stats()
            print(f"📊 Total establishments in database: {stats.get('total_documents', 0):,}")
            print(f"   Collection     : {stats.get('collection_name', 'N/A')}")
            print(f"   Status         : {stats.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            print("❌ Could not retrieve statistics.")
        return

    # ====================== QUERY MODES ======================
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
            logger.error(f"Error processing single query: {e}")
            print("❌ An error occurred while processing your question.")
        return

    # ====================== INTERACTIVE MODE ======================
    print("=" * 85)
    print("   🇹🇳 Tunisia Education RAG - Educational Institutions")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 85)

    # Show available governorates
    try:
        available_govs = get_available_governorates()
        if available_govs:
            print(f"Available Governorates: {', '.join(available_govs[:12])} ...")
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

            # Process query via shared service
            result = service.query(user_input, chat_history, k=args.k)

            print(f"\n🤖 Response:\n{result['answer']}\n")

            if result.get("sources"):
                print("📚 Sources:")
                for source in result["sources"]:
                    print(f"   • {source}")
                print("")

            # Update history (keep last 10 turns)
            chat_history.extend([("human", user_input), ("ai", result["answer"])])
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]

        except KeyboardInterrupt:
            print("\n\n👋 Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in interactive mode: {e}")
            print("❌ An error occurred. Please try again.")

if __name__ == "__main__":
    main()