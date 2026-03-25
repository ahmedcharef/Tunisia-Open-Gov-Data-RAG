#!/usr/bin/env python3
"""
query.py - Command Line Interface for Tunisia Education RAG

This is the CLI version of the RAG application.
It allows users to query Tunisian educational institutions (schools and universities)
using natural language.

Features:
- Single query mode (--query)
- Interactive chat mode
- Governorate-based filtering
- Source citations
- Database statistics (--stats)

Uses the shared RAGService layer for consistency with the Streamlit UI.
"""

import argparse
import sys
from typing import List, Tuple

from src.config import Config, logger
from src.rag_service import RAGService
from src.retriever import get_vectorstore_stats

def main():
    parser = argparse.ArgumentParser(description="Tunisia Education RAG CLI")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--k", type=int, default=Config.RETRIEVER_K)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    service = RAGService()

    if args.stats:
        stats = get_vectorstore_stats()
        print(f"📊 Total establishments in database: {stats['total_documents']:,}")
        return

    chat_history: List[Tuple[str, str]] = []

    if args.query:
        result = service.query(args.query, chat_history, k=args.k)
        print("\n🤖 Response:\n")
        print(result["answer"])

        if result["sources"]:
            print("\n📚 Sources:")
            for source in result["sources"]:
                print(f"   • {source}")
        return

    # Interactive Mode
    print("=" * 80)
    print("   🇹🇳 Tunisia Education RAG")
    print("   Type 'exit', 'quit', or 'clear' to manage conversation")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nQuestion > ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                chat_history.clear()
                print("🧹 History cleared.")
                continue

            if not user_input:
                continue

            result = service.query(user_input, chat_history, k=args.k)

            print(f"\n🤖 Response:\n{result['answer']}\n")

            if result["sources"]:
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
