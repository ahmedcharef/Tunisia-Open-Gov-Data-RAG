#!/usr/bin/env python3
"""
run.py - Easy entry point for Tunisia Education RAG

Usage:
    python run.py              → Run CLI version
    python run.py ui           → Run Streamlit web UI
    python run.py --help       → Show help
"""

import sys
import argparse

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

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if args.mode == "cli":
        print("🚀 Starting Tunisia Education RAG (CLI Mode)\n")
        from src.query import main as run_cli
        run_cli()

    elif args.mode == "ui":
        print("🌐 Starting Tunisia Education RAG (Streamlit UI)")
        print("   Open your browser when the server starts...\n")
        
        import streamlit.web.cli as stcli
        import os
        import sys

        # Set Streamlit to run src/app.py
        sys.argv = ["streamlit", "run", "src/app.py", "--server.headless", "true"]
        stcli.main()


if __name__ == "__main__":
    main()