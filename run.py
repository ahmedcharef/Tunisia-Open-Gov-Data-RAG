#!/usr/bin/env python3
"""
run.py - Easy entry point for Tunisia Education RAG

Usage:
    python run.py              → Run CLI version (default)
    python run.py ui           → Run Streamlit web UI
    python run.py --debug      → Enable debug logging
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
        choices=["cli", "ui"],
        default="cli",
        help="Mode to run: 'cli' (default) or 'ui'"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        print("🛠️  Debug mode enabled\n")

    if args.mode == "cli":
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