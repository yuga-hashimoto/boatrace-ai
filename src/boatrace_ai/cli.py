"""CLI entrypoint for the boatrace-ai scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path("configs/base.json")


def _placeholder_handler(args: argparse.Namespace) -> int:
    payload = {
        "command": args.command,
        "config": args.config,
        "output_dir": getattr(args, "output_dir", None),
        "race_date": getattr(args, "race_date", None),
        "note": "Pipeline stub. Implement this command in src/boatrace_ai/ next."
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boatrace-ai",
        description="Starter CLI for boat race data collection, training, and prediction."
    )
    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="Collect source data into data/raw.")
    collect.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    collect.add_argument("--output-dir", default="data/raw")
    collect.set_defaults(handler=_placeholder_handler)

    build_dataset = subparsers.add_parser(
        "build-dataset",
        help="Transform raw source data into model-ready tables."
    )
    build_dataset.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    build_dataset.add_argument("--output-dir", default="data/processed")
    build_dataset.set_defaults(handler=_placeholder_handler)

    train = subparsers.add_parser("train", help="Train baseline models.")
    train.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    train.add_argument("--output-dir", default="artifacts/models")
    train.set_defaults(handler=_placeholder_handler)

    predict = subparsers.add_parser("predict", help="Generate daily predictions.")
    predict.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    predict.add_argument("--output-dir", default="artifacts/predictions")
    predict.add_argument("--race-date", default=None)
    predict.set_defaults(handler=_placeholder_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)
