#!/usr/bin/env python3
"""Vera Merchant Bot — main entry point.

Usage:
    python main.py                          # Start the HTTP server (default)
    python main.py --generate-submission    # Generate submission.jsonl
    python main.py --expand-dataset         # Expand seed dataset
    python main.py --host 0.0.0.0 --port 8080  # Custom host/port
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# ---------------------------------------------------------------------------
# Logging setup — configured early so every module inherits the format
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger("vera")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


async def _generate_submission() -> None:
    """Wire up all modules and generate submission.jsonl."""
    from src.composer import Composer
    from src.config import get_llm_config
    from src.context_store import ContextStore
    from src.llm_client import LLMClient
    from src.submission import SubmissionGenerator

    llm_cfg = get_llm_config()
    store = ContextStore()
    llm = LLMClient(
        provider=llm_cfg.provider,
        api_key=llm_cfg.api_key,
        model=llm_cfg.model,
        timeout=60.0,  # 60s for offline submission (no judge time constraint)
    )
    composer = Composer(llm_client=llm, context_store=store)
    generator = SubmissionGenerator(context_store=store, composer=composer)

    logger.info("Loading expanded dataset into context store …")
    await generator.load_expanded_dataset("dataset/expanded")

    logger.info("Generating submission.jsonl …")
    await generator.generate("dataset/expanded/test_pairs.json", "submission.jsonl")
    logger.info("Done.")


def _expand_dataset() -> None:
    """Run the dataset expansion script."""
    from dataset.generate_dataset import main as generate_main

    logger.info("Expanding seed dataset …")
    # Override sys.argv so argparse inside generate_dataset sees the right flags
    original_argv = sys.argv
    sys.argv = [
        "generate_dataset.py",
        "--seed-dir", "dataset",
        "--out", "dataset/expanded",
    ]
    try:
        generate_main()
    finally:
        sys.argv = original_argv
    logger.info("Dataset expansion complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Vera Merchant Bot")
    parser.add_argument(
        "--generate-submission",
        action="store_true",
        help="Generate submission.jsonl from expanded dataset",
    )
    parser.add_argument(
        "--expand-dataset",
        action="store_true",
        help="Expand seed dataset to full 50/200/100 set",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host")
    parser.add_argument("--port", type=int, default=8080, help="Server bind port")
    args = parser.parse_args()

    if args.expand_dataset:
        _expand_dataset()
    elif args.generate_submission:
        asyncio.run(_generate_submission())
    else:
        import uvicorn

        logger.info("Starting Vera Merchant Bot on %s:%d", args.host, args.port)
        uvicorn.run("src.bot:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
