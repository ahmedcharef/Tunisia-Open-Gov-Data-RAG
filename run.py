#!/usr/bin/env python3
"""
run.py - Unified launcher for Tunisia Open Government Data RAG

Usage:
    python run.py                            → Interactive CLI (default)
    python run.py cli                        → Interactive CLI
    python run.py cli --query "..."          → Single query, non-interactive
    python run.py ui                         → Streamlit web UI
    python run.py ingest                     → Ingest all files for the default dataset
    python run.py ingest --dataset transport → Ingest a specific dataset
    python run.py --debug                    → Enable debug logging (any mode)
"""

import sys
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(
        description="Tunisia Education RAG - Official Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["cli", "ui", "ingest"],
        default="cli",
        help="Mode to run: 'cli' (default), 'ui', or 'ingest'"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to ingest into (used with 'ingest' mode). See Config.DATASETS for options."
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        print("🛠️  Debug mode enabled\n")

    if args.mode == "ingest":
        print("📥 Starting ingestion pipeline...\n")
        try:
            from src.ingest import ingest_dataset
            ingest_dataset(dataset=args.dataset)
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            sys.exit(1)

    elif args.mode == "cli":
        print("🚀 Starting Tunisia Education RAG (CLI Mode)\n")
        try:
            from src.query import main as run_cli
            run_cli()
        except Exception as e:
            print(f"❌ Failed to start CLI: {e}")
            sys.exit(1)

    elif args.mode == "ui":
        print("🌐 Starting Tunisia Education RAG (Streamlit UI)")
        print("   The browser should open automatically...\n")
        
        try:
            import streamlit.web.cli as stcli
            # Properly set up sys.argv for Streamlit
            sys.argv = ["streamlit", "run", "src/app.py", "--server.headless", "true"]
            stcli.main()
        except ImportError:
            print("❌ Streamlit is not installed. Please run: pip install streamlit")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to start Streamlit UI: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()