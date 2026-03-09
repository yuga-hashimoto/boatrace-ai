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


def _note_morning_handler(args: argparse.Namespace) -> int:
    from boatrace_ai.note.morning import main as morning_main
    import sys
    argv = []
    if args.race_date:
        argv.extend(["--date", args.race_date])
    if hasattr(args, "output_dir") and args.output_dir:
        argv.extend(["--output-dir", args.output_dir])
    sys.argv = ["boatrace-ai note-morning"] + argv
    morning_main()
    return 0


def _note_evening_handler(args: argparse.Namespace) -> int:
    from boatrace_ai.note.evening import main as evening_main
    import sys
    argv = []
    if args.race_date:
        argv.extend(["--date", args.race_date])
    if hasattr(args, "output_dir") and args.output_dir:
        argv.extend(["--output-dir", args.output_dir])
    sys.argv = ["boatrace-ai note-evening"] + argv
    evening_main()
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
    train.add_argument("--output-dir", default="models")
    train.set_defaults(handler=_placeholder_handler)

    predict = subparsers.add_parser("predict", help="Run prediction for a given date.")
    predict.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    predict.add_argument("--race-date", required=True)
    predict.add_argument("--output-dir", default="data/processed")
    predict.set_defaults(handler=_placeholder_handler)

    # -- note article generation --
    note_morning = subparsers.add_parser(
        "note-morning",
        help="Generate morning prediction article HTML for note.com (paid)."
    )
    note_morning.add_argument("--race-date", default=None, help="Target date (YYYY-MM-DD). Defaults to today.")
    note_morning.add_argument("--output-dir", default=None, help="Output directory override.")
    note_morning.set_defaults(handler=_note_morning_handler)

    note_evening = subparsers.add_parser(
        "note-evening",
        help="Generate evening verification article HTML for note.com (free)."
    )
    note_evening.add_argument("--race-date", default=None, help="Target date (YYYY-MM-DD). Defaults to today.")
    note_evening.add_argument("--output-dir", default=None, help="Output directory override.")
    note_evening.set_defaults(handler=_note_evening_handler)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.handler(args)
