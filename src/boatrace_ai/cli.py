"""CLI entrypoint for the boatrace-ai pipeline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from boatrace_ai.betting import (
    DEFAULT_BETTING_POLICY,
    generate_trifecta_recommendations,
    merge_betting_policy,
    MONTHLY_ROI_BETTING_POLICY,
    REPEATED_ROI_BETTING_POLICY,
    STRUCTURAL_ROI_BETTING_POLICY,
)
from boatrace_ai.collect.history import collect_race_records, refresh_missing_trifecta_odds
from boatrace_ai.collect.official_download import fetch_program_cards_from_official_download, sync_official_download_to_db
from boatrace_ai.collect.official import OfficialBoatraceClient, RaceCard, compact_race_date, restore_race_date
from boatrace_ai.evaluate.backtest import run_holdout_backtest
from boatrace_ai.features.dataset import build_dataset
from boatrace_ai.note.evening import generate_evening_note
from boatrace_ai.note.morning import generate_morning_note
from boatrace_ai.predict.baseline import RacePrediction, predict_race
from boatrace_ai.predict.model import predict_race_with_model
from boatrace_ai.report.live import generate_live_report_message
from boatrace_ai.store.sqlite import import_race_records_to_db
from boatrace_ai.train.model import (
    _attach_fallback_policy,
    find_latest_model,
    load_model_artifact,
    train_win_model,
)


DEFAULT_CONFIG_PATH = Path("configs/base.json")
DEFAULT_TIMEZONE = ZoneInfo("Asia/Tokyo")


def _collect_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    start_date = _resolve_race_date(args.start_date or args.race_date, config)
    end_date = _resolve_race_date(args.end_date, config) if args.end_date else start_date
    summary = collect_race_records(
        output_dir=Path(args.output_dir),
        start_date=start_date,
        end_date=end_date,
        venue_filters=args.venue or config.get("inference", {}).get("venues", []),
        race_numbers=[args.race_no] if args.race_no else None,
        include_beforeinfo=not args.no_beforeinfo,
        include_results=not args.no_results,
        include_odds=not args.no_odds,
        max_workers=args.max_workers,
        skip_existing=not args.force,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        request_retry_backoff_seconds=args.request_retry_backoff_seconds,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _build_dataset_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    output_dir = Path(args.output_dir or config.get("paths", {}).get("processed_dir", "data/processed"))
    output_path = Path(args.dataset_path or output_dir / "entrants.csv")
    input_db = Path(args.input_db) if args.input_db else None
    input_dir = None if input_db else Path(args.input_dir or config.get("paths", {}).get("raw_dir", "data/raw"))
    summary = build_dataset(
        input_dir=input_dir,
        output_path=output_path,
        db_path=input_db,
        start_date=args.start_date,
        end_date=args.end_date,
        venue_filters=args.venue,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _import_db_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    input_dir = Path(args.input_dir or config.get("paths", {}).get("raw_dir", "data/raw"))
    db_path = Path(args.db_path or _default_history_db_path(config))
    summary = import_race_records_to_db(
        input_dir=input_dir,
        db_path=db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        venue_filters=args.venue,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _sync_db_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    end_date = _resolve_race_date(args.race_date, config)
    start_date = (_parse_race_date(end_date) - timedelta(days=max(args.lookback_days - 1, 0))).isoformat()
    output_dir = Path(args.output_dir or config.get("paths", {}).get("raw_dir", "data/raw"))
    venue_filters = args.venue or config.get("inference", {}).get("venues", [])
    collection_summary = collect_race_records(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        venue_filters=venue_filters,
        race_numbers=[args.race_no] if args.race_no else None,
        include_beforeinfo=not args.no_beforeinfo,
        include_results=not args.no_results,
        include_odds=not args.no_odds,
        max_workers=args.max_workers,
        skip_existing=not args.force,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        request_retry_backoff_seconds=args.request_retry_backoff_seconds,
    )
    db_summary = import_race_records_to_db(
        input_dir=output_dir,
        db_path=Path(args.db_path or _default_history_db_path(config)),
        start_date=start_date,
        end_date=end_date,
        venue_filters=venue_filters,
    )
    print(
        json.dumps(
            {
                "sync_start_date": start_date,
                "sync_end_date": end_date,
                "lookback_days": args.lookback_days,
                "collection": collection_summary.to_dict(),
                "db_import": db_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _sync_download_db_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    end_date = _resolve_race_date(args.race_date, config)
    start_date = (_parse_race_date(end_date) - timedelta(days=max(args.lookback_days - 1, 0))).isoformat()
    summary = sync_official_download_to_db(
        output_dir=Path(args.output_dir or config.get("paths", {}).get("raw_dir", "data/raw")),
        db_path=Path(args.db_path or _default_history_db_path(config)),
        start_date=start_date,
        end_date=end_date,
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
    )
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _refresh_odds_handler(args: argparse.Namespace) -> int:
    summary = refresh_missing_trifecta_odds(
        input_dir=Path(args.input_dir),
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        request_retry_backoff_seconds=args.request_retry_backoff_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _train_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    dataset_path = Path(args.dataset_path or Path(config.get("paths", {}).get("processed_dir", "data/processed")) / "entrants.csv")
    train_end_date = args.train_end_date or config.get("dataset", {}).get("train_end_date")
    random_state = int(config.get("model", {}).get("random_state", 42))
    training_guards = config.get("training_guards", {})
    result = train_win_model(
        dataset_path=dataset_path,
        output_dir=Path(args.output_dir),
        train_end_date=train_end_date,
        raw_dir=Path(args.raw_dir or config.get("paths", {}).get("raw_dir", "data/raw")),
        random_state=random_state,
        min_recommendation_roi=(
            args.min_recommendation_roi
            if args.min_recommendation_roi is not None
            else training_guards.get("min_recommendation_roi")
        ),
        min_walk_forward_recommendation_roi=(
            args.min_walk_forward_recommendation_roi
            if args.min_walk_forward_recommendation_roi is not None
            else training_guards.get("min_walk_forward_recommendation_roi")
        ),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


def _backtest_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    dataset_path = Path(args.dataset_path or Path(config.get("paths", {}).get("processed_dir", "data/processed")) / "entrants.csv")
    raw_dir = Path(args.raw_dir or config.get("paths", {}).get("raw_dir", "data/raw"))
    risk_management = config.get("risk_management", {})
    result = run_holdout_backtest(
        dataset_path=dataset_path,
        raw_dir=raw_dir,
        train_end_date=args.train_end_date or config.get("dataset", {}).get("train_end_date"),
        random_state=int(config.get("model", {}).get("random_state", 42)),
        bankroll_mode=args.bankroll_mode,
        starting_bankroll_yen=args.starting_bankroll_yen,
        flat_bet_yen=args.flat_bet_yen,
        kelly_fraction=(
            args.kelly_fraction
            if args.kelly_fraction is not None
            else float(risk_management.get("kelly_fraction", 0.25))
        ),
        kelly_cap_fraction=(
            args.kelly_cap_fraction
            if args.kelly_cap_fraction is not None
            else risk_management.get("kelly_cap_fraction", 0.1)
        ),
        max_bet_yen=args.max_bet_yen,
        max_bet_bankroll_fraction=(
            args.max_bet_bankroll_fraction
            if args.max_bet_bankroll_fraction is not None
            else risk_management.get("max_bet_bankroll_fraction")
        ),
        max_daily_bets=(
            args.max_daily_bets
            if args.max_daily_bets is not None
            else risk_management.get("max_daily_bets")
        ),
        max_daily_investment_yen=(
            args.max_daily_investment_yen
            if args.max_daily_investment_yen is not None
            else risk_management.get("max_daily_investment_yen")
        ),
        max_race_exposure_fraction=(
            args.max_race_exposure_fraction
            if args.max_race_exposure_fraction is not None
            else risk_management.get("max_race_exposure_fraction")
        ),
        daily_stop_loss_yen=(
            args.daily_stop_loss_yen
            if args.daily_stop_loss_yen is not None
            else risk_management.get("daily_stop_loss_yen")
        ),
        daily_take_profit_yen=(
            args.daily_take_profit_yen
            if args.daily_take_profit_yen is not None
            else risk_management.get("daily_take_profit_yen")
        ),
        betting_policy_override=_betting_policy_override_from_args(args),
        clear_derived_filters=bool(getattr(args, "clear_derived_filters", False)),
    )
    payload = result.to_dict()
    output_path = Path(args.output_path) if args.output_path else _default_backtest_output_path(config)
    payload["output_path"] = str(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _predict_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    race_date = _resolve_race_date(args.race_date, config)
    venue_filters = args.venue or config.get("inference", {}).get("venues", [])
    race_no = getattr(args, "race_no", None)

    artifact = _load_prediction_artifact(args.model_path, config)
    betting_policy = _resolve_betting_policy(args, config, artifact)
    client = OfficialBoatraceClient()
    try:
        venues = _select_venues(client.fetch_race_index(race_date), venue_filters)
        if not venues:
            raise ValueError("No active venues matched the current filters.")

        race_numbers = list(range(1, 13)) if args.command == "predict-venue" else ([race_no] if race_no else list(range(1, 13)))
        predictions = _fetch_predictions(
            race_date=race_date,
            venues=venues,
            race_numbers=race_numbers,
            top_k=args.top_k,
            max_workers=args.max_workers,
            artifact=artifact,
        )
    finally:
        client.close()

    recommendations = _build_recommendations(predictions, artifact, betting_policy)
    race_predictions = [_race_prediction_payload(prediction) for prediction in predictions]
    output_path = _write_predictions(
        output_dir=Path(args.output_dir),
        race_date=race_date,
        payload={
            "command": args.command,
            "config": args.config,
            "race_date": race_date,
            "model_name": artifact.get("model_name") if artifact else "baseline_official_card_v1",
            "generated_at": datetime.now(tz=DEFAULT_TIMEZONE).isoformat(),
            "model_metrics": artifact.get("metrics") if artifact else {},
            "betting_policy": betting_policy,
                "filters": {
                    "venue": venue_filters,
                    "race_no": race_no,
                    "top_k": args.top_k,
                },
            "recommendations": [recommendation.to_dict() for recommendation in recommendations],
            "race_predictions": race_predictions,
            "races": [prediction.to_dict() for prediction in predictions],
        },
    )
    print(_format_prediction_summary(predictions, recommendations, output_path))
    return 0


def _predict_live_handler(args: argparse.Namespace) -> int:
    run = _run_live_prediction_job(args, _load_config(Path(args.config)))
    print(
        _format_live_prediction_summary(
            predictions=run["predictions"],
            recommendations=run["recommendations"],
            output_path=run["output_path"],
            now=run["now"],
            candidate_race_count=run["candidate_race_count"],
            skipped=run["skipped"],
        )
    )
    return 0


def _report_live_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    args.with_recommendations = True
    args.require_odds = True
    run = _run_live_prediction_job(args, config)
    output_dir = Path(args.output_dir)
    message, summary = generate_live_report_message(
        race_date=run["race_date"],
        current_payload=run["payload"],
        prediction_dir=output_dir,
        state_path=Path(args.state_path),
        result_max_workers=args.result_max_workers,
        upcoming_limit=args.upcoming_limit,
        settled_limit=args.settled_limit,
        quiet_when_empty=args.quiet_when_empty,
    )
    if message:
        print(message)
    return 0


def _run_live_prediction_job(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    race_date = _resolve_race_date(args.race_date, config)
    venue_filters = args.venue or config.get("inference", {}).get("venues", [])
    artifact = _load_prediction_artifact(args.model_path, config)
    betting_policy = _resolve_betting_policy(args, config, artifact)
    now = _resolve_live_now(getattr(args, "now", None))
    output_dir = Path(args.output_dir)
    cache_path = Path(config.get("paths", {}).get("raw_dir", "data/raw")) / compact_race_date(race_date) / "_program_cards.json"

    cards = fetch_program_cards_from_official_download(
        race_date,
        request_timeout=args.request_timeout,
        cache_path=cache_path,
    )
    cards = _select_venues(cards, venue_filters)
    live_cards = _select_live_cards(cards, race_date, now, args.lookahead_minutes)

    predictions, odds_overrides, skipped = _fetch_live_predictions(
        cards=live_cards,
        artifact=artifact,
        top_k=args.top_k,
        max_workers=args.max_workers,
        request_timeout=args.request_timeout,
        request_max_retries=args.request_max_retries,
        require_odds=args.require_odds,
    )

    recommendations = (
        _build_recommendations(
            predictions,
            artifact,
            betting_policy,
            odds_overrides=odds_overrides,
        )
        if args.with_recommendations
        else []
    )
    race_predictions = [_race_prediction_payload(prediction) for prediction in predictions]
    payload = {
        "command": args.command,
        "config": args.config,
        "race_date": race_date,
        "model_name": artifact.get("model_name") if artifact else "baseline_official_card_v1",
        "generated_at": datetime.now(tz=DEFAULT_TIMEZONE).isoformat(),
        "model_metrics": artifact.get("metrics") if artifact else {},
        "betting_policy": betting_policy,
        "filters": {
            "venue": venue_filters,
            "top_k": args.top_k,
            "lookahead_minutes": args.lookahead_minutes,
            "require_beforeinfo": True,
            "require_odds": args.require_odds,
            "with_recommendations": args.with_recommendations,
            "now": now.isoformat(),
        },
        "candidate_race_count": len(live_cards),
        "ready_race_count": len(predictions),
        "skipped_races": skipped,
        "recommendations": [recommendation.to_dict() for recommendation in recommendations],
        "race_predictions": race_predictions,
        "races": [prediction.to_dict() for prediction in predictions],
    }
    output_path = _write_predictions(
        output_dir=output_dir,
        race_date=race_date,
        payload=payload,
    )
    return {
        "race_date": race_date,
        "payload": payload,
        "output_path": output_path,
        "predictions": predictions,
        "recommendations": recommendations,
        "now": now,
        "candidate_race_count": len(live_cards),
        "skipped": skipped,
    }


def _note_morning_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    race_date = _resolve_race_date(args.race_date, config)
    result = generate_morning_note(
        race_date=race_date,
        prediction_dir=Path(args.prediction_dir or config.get("paths", {}).get("prediction_dir", "artifacts/predictions")),
        output_dir=Path(args.output_dir or config.get("paths", {}).get("note_dir", "artifacts/note")),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _note_evening_handler(args: argparse.Namespace) -> int:
    config = _load_config(Path(args.config))
    race_date = _resolve_race_date(args.race_date, config)
    result = generate_evening_note(
        race_date=race_date,
        prediction_dir=Path(args.prediction_dir or config.get("paths", {}).get("prediction_dir", "artifacts/predictions")),
        output_dir=Path(args.output_dir or config.get("paths", {}).get("note_dir", "artifacts/note")),
        max_workers=args.max_workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boatrace-ai",
        description="Boat race data collection, training, and prediction pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")

    collect = subparsers.add_parser("collect", help="Collect raw official pages into data/raw.")
    collect.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    collect.add_argument("--output-dir", default="data/raw")
    collect.add_argument("--race-date", default=None, help="Single collection date.")
    collect.add_argument("--start-date", default=None, help="Collection start date.")
    collect.add_argument("--end-date", default=None, help="Collection end date.")
    collect.add_argument("--venue", action="append", help="Venue code or name.")
    collect.add_argument("--race-no", type=int, default=None, help="Collect only one race number.")
    collect.add_argument("--no-beforeinfo", action="store_true", help="Skip beforeinfo pages.")
    collect.add_argument("--no-results", action="store_true", help="Skip result pages.")
    collect.add_argument("--no-odds", action="store_true", help="Skip trifecta odds pages.")
    collect.add_argument("--force", action="store_true", help="Refetch even when a complete record already exists.")
    collect.add_argument("--request-timeout", type=float, default=20.0, help="HTTP timeout seconds per request.")
    collect.add_argument("--request-max-retries", type=int, default=3, help="Retry count per request.")
    collect.add_argument("--request-retry-backoff-seconds", type=float, default=1.0, help="Linear retry backoff seconds.")
    collect.add_argument("--max-workers", type=int, default=4, help="Parallel fetch workers.")
    collect.set_defaults(handler=_collect_handler)

    import_db = subparsers.add_parser(
        "import-db",
        help="Import raw JSON race records into the SQLite history store.",
    )
    import_db.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    import_db.add_argument("--input-dir", default="data/raw")
    import_db.add_argument("--db-path", default=None)
    import_db.add_argument("--start-date", default=None)
    import_db.add_argument("--end-date", default=None)
    import_db.add_argument("--venue", action="append", help="Venue code or name.")
    import_db.set_defaults(handler=_import_db_handler)

    sync_db = subparsers.add_parser(
        "sync-db",
        help="Collect a lookback window of races and import them into the SQLite history store.",
    )
    sync_db.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sync_db.add_argument("--output-dir", default="data/raw")
    sync_db.add_argument("--db-path", default=None)
    sync_db.add_argument("--race-date", default=None, help="Sync end date.")
    sync_db.add_argument("--lookback-days", type=int, default=365)
    sync_db.add_argument("--venue", action="append", help="Venue code or name.")
    sync_db.add_argument("--race-no", type=int, default=None, help="Collect only one race number.")
    sync_db.add_argument("--no-beforeinfo", action="store_true", help="Skip beforeinfo pages.")
    sync_db.add_argument("--no-results", action="store_true", help="Skip result pages.")
    sync_db.add_argument("--no-odds", action="store_true", help="Skip trifecta odds pages.")
    sync_db.add_argument("--force", action="store_true", help="Refetch even when a complete record already exists.")
    sync_db.add_argument("--request-timeout", type=float, default=20.0, help="HTTP timeout seconds per request.")
    sync_db.add_argument("--request-max-retries", type=int, default=3, help="Retry count per request.")
    sync_db.add_argument("--request-retry-backoff-seconds", type=float, default=1.0, help="Linear retry backoff seconds.")
    sync_db.add_argument("--max-workers", type=int, default=4, help="Parallel fetch workers.")
    sync_db.set_defaults(handler=_sync_db_handler)

    sync_download_db = subparsers.add_parser(
        "sync-download-db",
        help="Backfill historical races from the official daily download files into raw JSON and SQLite.",
    )
    sync_download_db.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    sync_download_db.add_argument("--output-dir", default="data/raw")
    sync_download_db.add_argument("--db-path", default=None)
    sync_download_db.add_argument("--race-date", default=None, help="Sync end date.")
    sync_download_db.add_argument("--lookback-days", type=int, default=365)
    sync_download_db.add_argument("--request-timeout", type=float, default=30.0)
    sync_download_db.add_argument("--max-workers", type=int, default=4)
    sync_download_db.set_defaults(handler=_sync_download_db_handler)

    build_dataset_cmd = subparsers.add_parser(
        "build-dataset",
        help="Transform raw JSON race records into entrant-level training rows.",
    )
    build_dataset_cmd.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    build_dataset_cmd.add_argument("--input-dir", default="data/raw")
    build_dataset_cmd.add_argument("--input-db", default=None)
    build_dataset_cmd.add_argument("--output-dir", default="data/processed")
    build_dataset_cmd.add_argument("--dataset-path", default=None)
    build_dataset_cmd.add_argument("--start-date", default=None)
    build_dataset_cmd.add_argument("--end-date", default=None)
    build_dataset_cmd.add_argument("--venue", action="append", help="Venue code or name.")
    build_dataset_cmd.set_defaults(handler=_build_dataset_handler)

    refresh_odds = subparsers.add_parser(
        "refresh-odds",
        help="Fetch missing trifecta odds for existing raw race records.",
    )
    refresh_odds.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    refresh_odds.add_argument("--input-dir", default="data/raw")
    refresh_odds.add_argument("--request-timeout", type=float, default=20.0, help="HTTP timeout seconds per request.")
    refresh_odds.add_argument("--request-max-retries", type=int, default=3, help="Retry count per request.")
    refresh_odds.add_argument("--request-retry-backoff-seconds", type=float, default=1.0, help="Linear retry backoff seconds.")
    refresh_odds.add_argument("--max-workers", type=int, default=4, help="Parallel fetch workers.")
    refresh_odds.set_defaults(handler=_refresh_odds_handler)

    train = subparsers.add_parser("train", help="Train a win-probability model and backtest it.")
    train.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    train.add_argument("--dataset-path", default="data/processed/entrants.csv")
    train.add_argument("--raw-dir", default=None, help="Optional raw record directory for odds-aware backtests.")
    train.add_argument("--output-dir", default="artifacts/models")
    train.add_argument("--train-end-date", default=None)
    train.add_argument("--min-recommendation-roi", type=float, default=None, help="Fail if holdout recommendation ROI is below this threshold.")
    train.add_argument("--min-walk-forward-recommendation-roi", type=float, default=None, help="Fail if walk-forward recommendation ROI is below this threshold.")
    train.set_defaults(handler=_train_handler)

    backtest = subparsers.add_parser("backtest", help="Run holdout backtest with bankroll simulation.")
    backtest.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    backtest.add_argument("--dataset-path", default="data/processed/entrants.csv")
    backtest.add_argument("--raw-dir", default=None, help="Raw record directory for odds-aware evaluation.")
    backtest.add_argument("--train-end-date", default=None)
    backtest.add_argument("--bankroll-mode", choices=["flat", "kelly", "kelly_capped", "both", "all"], default="both")
    backtest.add_argument("--starting-bankroll-yen", type=int, default=10000)
    backtest.add_argument("--flat-bet-yen", type=int, default=100)
    backtest.add_argument("--kelly-fraction", type=float, default=None)
    backtest.add_argument("--kelly-cap-fraction", type=float, default=None)
    backtest.add_argument("--max-bet-yen", type=int, default=None)
    backtest.add_argument("--max-bet-bankroll-fraction", type=float, default=None)
    backtest.add_argument("--max-daily-bets", type=int, default=None)
    backtest.add_argument("--max-daily-investment-yen", type=int, default=None)
    backtest.add_argument("--max-race-exposure-fraction", type=float, default=None)
    backtest.add_argument("--daily-stop-loss-yen", type=int, default=None)
    backtest.add_argument("--daily-take-profit-yen", type=int, default=None)
    backtest.add_argument("--betting-preset", choices=["monthly-roi", "repeated-roi", "structural-roi"], default=None)
    backtest.add_argument("--min-expected-value", type=float, default=None, help="Override recommendation EV threshold.")
    backtest.add_argument("--max-per-race", type=int, default=None, help="Override max recommendations per race.")
    backtest.add_argument("--candidate-pool-size", type=int, default=None, help="Override candidate trifecta pool size.")
    backtest.add_argument("--min-probability", type=float, default=None, help="Override minimum model probability.")
    backtest.add_argument("--min-edge", type=float, default=None, help="Override minimum market edge.")
    backtest.add_argument("--min-market-odds", type=float, default=None, help="Override minimum market odds.")
    backtest.add_argument("--max-market-odds", type=float, default=None, help="Override maximum market odds.")
    backtest.add_argument("--min-top-win-probability", type=float, default=None, help="Only keep races where the top win probability exceeds this threshold.")
    backtest.add_argument("--min-win-margin", type=float, default=None, help="Only keep races where the top-vs-second win probability gap exceeds this threshold.")
    backtest.add_argument("--required-first-lane", type=int, default=None, help="Only keep combinations with this first-place lane.")
    backtest.add_argument("--required-second-lane", type=int, default=None, help="Only keep combinations with this second-place lane.")
    backtest.add_argument("--required-third-lane", type=int, default=None, help="Only keep combinations with this third-place lane.")
    backtest.add_argument("--allowed-venues", nargs="+", default=None, help="Only keep recommendations from these venue codes or names.")
    backtest.add_argument("--clear-derived-filters", action="store_true", help="Ignore derived venue and order filters unless they are restated.")
    backtest.add_argument("--output-path", default=None)
    backtest.set_defaults(handler=_backtest_handler)

    predict = subparsers.add_parser("predict", help="Generate race predictions.")
    _add_prediction_arguments(predict, require_venue=False, include_race_no=True)
    predict.set_defaults(handler=_predict_handler)

    predict_venue = subparsers.add_parser(
        "predict-venue",
        help="Generate predictions for all 12 races at a specific venue.",
    )
    _add_prediction_arguments(predict_venue, require_venue=True, include_race_no=False)
    predict_venue.set_defaults(handler=_predict_handler)

    predict_live = subparsers.add_parser(
        "predict-live",
        help="Predict only upcoming races that already have live beforeinfo available.",
    )
    _add_prediction_arguments(predict_live, require_venue=False, include_race_no=False)
    predict_live.add_argument("--lookahead-minutes", type=int, default=90, help="Only target races whose deadlines are within this many minutes from now.")
    predict_live.add_argument("--require-odds", action="store_true", help="Skip races unless trifecta odds are already available.")
    predict_live.add_argument("--with-recommendations", action="store_true", help="Also fetch odds and build recommendation candidates.")
    predict_live.add_argument("--request-timeout", type=float, default=5.0, help="HTTP timeout seconds per request.")
    predict_live.add_argument("--request-max-retries", type=int, default=1, help="Retry count per request.")
    predict_live.add_argument("--now", default=None, help="Optional ISO timestamp override for testing live windows.")
    predict_live.set_defaults(handler=_predict_live_handler)

    report_live = subparsers.add_parser(
        "report-live",
        help="Generate a chat-friendly live report with current bets and newly settled outcomes.",
    )
    _add_prediction_arguments(report_live, require_venue=False, include_race_no=False)
    report_live.add_argument("--lookahead-minutes", type=int, default=90, help="Only target races whose deadlines are within this many minutes from now.")
    report_live.add_argument("--request-timeout", type=float, default=5.0, help="HTTP timeout seconds per request.")
    report_live.add_argument("--request-max-retries", type=int, default=1, help="Retry count per request.")
    report_live.add_argument("--now", default=None, help="Optional ISO timestamp override for testing live windows.")
    report_live.add_argument("--state-path", default="artifacts/reports/live_report_state.json", help="State file used to suppress duplicate settled result notifications.")
    report_live.add_argument("--result-max-workers", type=int, default=4, help="Parallel fetch workers for settled result checks.")
    report_live.add_argument("--upcoming-limit", type=int, default=5, help="Max current recommendations or top predictions to print.")
    report_live.add_argument("--settled-limit", type=int, default=10, help="Max newly settled recommendation outcomes to print.")
    report_live.add_argument("--quiet-when-empty", action="store_true", help="Print nothing when there is no new bet or settled result to report.")
    report_live.set_defaults(handler=_report_live_handler)

    note_morning = subparsers.add_parser("note-morning", help="Generate a morning note article from predictions.")
    note_morning.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    note_morning.add_argument("--race-date", default=None)
    note_morning.add_argument("--prediction-dir", default=None)
    note_morning.add_argument("--output-dir", default=None)
    note_morning.set_defaults(handler=_note_morning_handler)

    note_evening = subparsers.add_parser("note-evening", help="Generate an evening verification note article.")
    note_evening.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    note_evening.add_argument("--race-date", default=None)
    note_evening.add_argument("--prediction-dir", default=None)
    note_evening.add_argument("--output-dir", default=None)
    note_evening.add_argument("--max-workers", type=int, default=4, help="Parallel result fetch workers.")
    note_evening.set_defaults(handler=_note_evening_handler)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    return args.handler(args)


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_race_date(race_date: str | None, config: dict) -> str:
    candidate = race_date or config.get("inference", {}).get("race_date")
    if candidate:
        return restore_race_date(compact_race_date(candidate))
    return datetime.now(tz=DEFAULT_TIMEZONE).date().isoformat()


def _select_venues(entries: list, venue_filters: list[str]) -> list:
    if not venue_filters:
        return entries

    normalized_filters = [str(filter_value).strip() for filter_value in venue_filters if str(filter_value).strip()]
    return [
        entry
        for entry in entries
        if any(
            filter_value == entry.venue_code
            or filter_value.zfill(2) == entry.venue_code
            or filter_value in entry.venue_name
            for filter_value in normalized_filters
        )
    ]


def _load_prediction_artifact(model_path: str | None, config: dict) -> dict | None:
    if model_path:
        return load_model_artifact(Path(model_path))

    model_dir = Path(config.get("paths", {}).get("model_dir", "artifacts/models"))
    latest = find_latest_model(model_dir)
    if latest is None:
        return None
    return load_model_artifact(latest)


def _resolve_betting_policy(args: argparse.Namespace, config: dict, artifact: dict | None) -> dict:
    policy = dict(DEFAULT_BETTING_POLICY)
    policy.update(config.get("betting", {}))
    if artifact and artifact.get("betting_policy"):
        policy.update(artifact["betting_policy"])
    override = _betting_policy_override_from_args(args)
    resolved_policy = merge_betting_policy(
        policy,
        override,
        clear_derived_filters=bool(getattr(args, "clear_derived_filters", False)),
    )
    return _attach_fallback_policy(
        resolved_policy,
        resolved_policy.get("fallback_policy"),
    )


def _betting_policy_override_from_args(args: argparse.Namespace) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    preset_name = getattr(args, "betting_preset", None)
    if preset_name:
        policy.update(_betting_preset(preset_name))
    if getattr(args, "min_expected_value", None) is not None:
        policy["min_expected_value"] = args.min_expected_value
    if getattr(args, "max_per_race", None) is not None:
        policy["max_per_race"] = args.max_per_race
    if getattr(args, "candidate_pool_size", None) is not None:
        policy["candidate_pool_size"] = args.candidate_pool_size
    if getattr(args, "min_probability", None) is not None:
        policy["min_probability"] = args.min_probability
    if getattr(args, "min_edge", None) is not None:
        policy["min_edge"] = args.min_edge
    if getattr(args, "min_market_odds", None) is not None:
        policy["min_market_odds"] = args.min_market_odds
    if getattr(args, "max_market_odds", None) is not None:
        policy["max_market_odds"] = args.max_market_odds
    if getattr(args, "min_top_win_probability", None) is not None:
        policy["min_top_win_probability"] = args.min_top_win_probability
    if getattr(args, "min_win_margin", None) is not None:
        policy["min_win_margin"] = args.min_win_margin
    if getattr(args, "required_first_lane", None) is not None:
        policy["required_first_lane"] = args.required_first_lane
    if getattr(args, "required_second_lane", None) is not None:
        policy["required_second_lane"] = args.required_second_lane
    if getattr(args, "required_third_lane", None) is not None:
        policy["required_third_lane"] = args.required_third_lane
    allowed_venues = _normalize_allowed_venues(getattr(args, "allowed_venues", None))
    if allowed_venues:
        policy["allowed_venues"] = allowed_venues
    return policy


def _normalize_allowed_venues(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            candidate = part.strip()
            if not candidate:
                continue
            normalized.append(candidate.zfill(2) if candidate.isdigit() else candidate)
    return normalized


def _betting_preset(name: str) -> dict[str, Any]:
    normalized = str(name or "").strip().lower()
    if normalized == "monthly-roi":
        return dict(MONTHLY_ROI_BETTING_POLICY)
    if normalized == "repeated-roi":
        return dict(REPEATED_ROI_BETTING_POLICY)
    if normalized == "structural-roi":
        return dict(STRUCTURAL_ROI_BETTING_POLICY)
    raise ValueError(f"Unknown betting preset: {name}")


def _resolve_live_now(value: str | None) -> datetime:
    if value:
        candidate = datetime.fromisoformat(value)
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=DEFAULT_TIMEZONE)
        return candidate.astimezone(DEFAULT_TIMEZONE)
    return datetime.now(tz=DEFAULT_TIMEZONE)


def _select_live_cards(
    cards: list[RaceCard],
    race_date: str,
    now: datetime,
    lookahead_minutes: int,
) -> list[RaceCard]:
    horizon = now + timedelta(minutes=max(0, int(lookahead_minutes)))
    selected: list[RaceCard] = []
    for card in cards:
        deadline_dt = _deadline_to_datetime(race_date, card.deadline)
        if deadline_dt is None:
            continue
        if deadline_dt <= now:
            continue
        if lookahead_minutes > 0 and deadline_dt > horizon:
            continue
        selected.append(card)
    return sorted(selected, key=lambda item: (_deadline_to_datetime(race_date, item.deadline) or horizon, item.venue_code, item.race_no))


def _deadline_to_datetime(race_date: str, deadline: str | None) -> datetime | None:
    if not deadline:
        return None
    return datetime.fromisoformat(f"{restore_race_date(compact_race_date(race_date))}T{deadline}:00+09:00").astimezone(DEFAULT_TIMEZONE)


def _fetch_live_predictions(
    *,
    cards: list[RaceCard],
    artifact: dict | None,
    top_k: int,
    max_workers: int,
    request_timeout: float,
    request_max_retries: int,
    require_odds: bool,
) -> tuple[list[RacePrediction], dict[str, dict[str, float]], list[dict[str, Any]]]:
    if not cards:
        return [], {}, []

    tasks = list(cards)
    if len(tasks) == 1 or max_workers <= 1:
        results = [
            _fetch_live_prediction(
                card=card,
                artifact=artifact,
                top_k=top_k,
                request_timeout=request_timeout,
                request_max_retries=request_max_retries,
                require_odds=require_odds,
            )
            for card in tasks
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
            futures = {
                executor.submit(
                    _fetch_live_prediction,
                    card=card,
                    artifact=artifact,
                    top_k=top_k,
                    request_timeout=request_timeout,
                    request_max_retries=request_max_retries,
                    require_odds=require_odds,
                ): card
                for card in tasks
            }
            for future in as_completed(futures):
                results.append(future.result())

    predictions: list[RacePrediction] = []
    odds_overrides: dict[str, dict[str, float]] = {}
    skipped: list[dict[str, Any]] = []
    for result in results:
        if result["prediction"] is None:
            skipped.append(result["skipped"])
            continue
        prediction = result["prediction"]
        predictions.append(prediction)
        odds_map = result["odds_map"]
        if odds_map:
            odds_overrides[_race_key(prediction.race.date, prediction.race.venue_code, prediction.race.race_no)] = odds_map

    predictions.sort(key=lambda item: (item.race.deadline or "99:99", item.race.venue_code, item.race.race_no))
    skipped.sort(key=lambda item: (item.get("deadline") or "99:99", item.get("venue_code") or "", item.get("race_no") or 0))
    return predictions, odds_overrides, skipped


def _fetch_live_prediction(
    *,
    card: RaceCard,
    artifact: dict | None,
    top_k: int,
    request_timeout: float,
    request_max_retries: int,
    require_odds: bool,
) -> dict[str, Any]:
    race_key = _race_key(card.date, card.venue_code, card.race_no)
    client = OfficialBoatraceClient(timeout=request_timeout, max_retries=request_max_retries)
    try:
        try:
            beforeinfo = client.fetch_beforeinfo(card.date, card.venue_code, card.race_no)
        except Exception:
            return {
                "prediction": None,
                "odds_map": {},
                "skipped": {
                    "race_key": race_key,
                    "venue_code": card.venue_code,
                    "venue_name": card.venue_name,
                    "race_no": card.race_no,
                    "deadline": card.deadline,
                    "reason": "beforeinfo_unavailable",
                },
            }

        odds_map: dict[str, float] = {}
        if require_odds:
            try:
                odds_map = client.fetch_trifecta_odds(card.date, card.venue_code, card.race_no)
            except Exception:
                return {
                    "prediction": None,
                    "odds_map": {},
                    "skipped": {
                        "race_key": race_key,
                        "venue_code": card.venue_code,
                        "venue_name": card.venue_name,
                        "race_no": card.race_no,
                        "deadline": card.deadline,
                        "reason": "odds_unavailable",
                    },
                }
    finally:
        client.close()

    prediction = predict_race_with_model(card, beforeinfo, artifact, top_k=top_k) if artifact else predict_race(card, top_k=top_k)
    return {
        "prediction": prediction,
        "odds_map": odds_map,
        "skipped": None,
    }


def _fetch_predictions(
    race_date: str,
    venues: list,
    race_numbers: list[int],
    top_k: int,
    max_workers: int,
    artifact: dict | None,
) -> list[RacePrediction]:
    tasks = [(venue.venue_code, race_number) for venue in venues for race_number in race_numbers]
    if not tasks:
        return []

    if len(tasks) == 1 or max_workers <= 1:
        predictions = [
            _fetch_and_predict(race_date, venue_code, race_number, top_k, artifact)
            for venue_code, race_number in tasks
        ]
        return sorted(predictions, key=lambda item: (item.race.venue_code, item.race.race_no))

    predictions: list[RacePrediction] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        futures = {
            executor.submit(_fetch_and_predict, race_date, venue_code, race_number, top_k, artifact): (venue_code, race_number)
            for venue_code, race_number in tasks
        }
        for future in as_completed(futures):
            predictions.append(future.result())

    return sorted(predictions, key=lambda item: (item.race.venue_code, item.race.race_no))


def _fetch_and_predict(
    race_date: str,
    venue_code: str,
    race_number: int,
    top_k: int,
    artifact: dict | None,
) -> RacePrediction:
    client = OfficialBoatraceClient()
    try:
        card = client.fetch_race_card(race_date, venue_code, race_number)
        beforeinfo = client.fetch_beforeinfo(race_date, venue_code, race_number)
    finally:
        client.close()

    if artifact:
        return predict_race_with_model(card, beforeinfo, artifact, top_k=top_k)
    return predict_race(card, top_k=top_k)


def _write_predictions(output_dir: Path, race_date: str, payload: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=DEFAULT_TIMEZONE).strftime("%Y%m%dT%H%M%S")
    race_date_key = compact_race_date(race_date)
    output_path = output_dir / f"predictions_{race_date_key}_{timestamp}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _default_backtest_output_path(config: dict) -> Path:
    output_dir = Path(config.get("paths", {}).get("backtest_dir", "artifacts/backtests"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=DEFAULT_TIMEZONE).strftime("%Y%m%dT%H%M%S")
    return output_dir / f"backtest_{timestamp}.json"


def _default_history_db_path(config: dict) -> Path:
    configured = config.get("paths", {}).get("history_db_path")
    if configured:
        return Path(configured)
    external_dir = Path(config.get("paths", {}).get("external_dir", "data/external"))
    return external_dir / "history.sqlite"


def _format_prediction_summary(
    predictions: list[RacePrediction],
    recommendations: list,
    output_path: Path,
) -> str:
    lines = [f"Saved predictions to {output_path}"]

    for prediction in predictions[:5]:
        race = prediction.race
        header = (
            f"[{race.date} {race.venue_name} {race.race_no}R] "
            f"{race.meeting_name} 締切 {race.deadline or '不明'}"
        )
        lines.append(header)
        for index, entrant in enumerate(prediction.entrants[:3], start=1):
            lines.append(
                f"  {index}. {entrant.lane}号艇 {entrant.name} ({entrant.grade}) "
                f"win={entrant.win_probability:.1%} top3={entrant.top3_probability:.1%}"
            )
        if prediction.trifectas:
            trifecta_text = ", ".join(
                f"{'-'.join(str(lane) for lane in trifecta.order)} {trifecta.probability:.1%}"
                for trifecta in prediction.trifectas[:3]
            )
            lines.append(f"  trifecta: {trifecta_text}")

    if recommendations:
        lines.append("Top expected-value bets:")
        for recommendation in recommendations[:5]:
            lines.append(
                "  "
                f"{recommendation.stadium_name}{recommendation.race_number}R "
                f"{recommendation.combination} "
                f"EV={recommendation.expected_value * 100:.0f}% "
                f"p={recommendation.probability:.2f}% "
                f"avg={recommendation.avg_payout:,}円"
            )

    if len(predictions) > 5:
        lines.append(f"... {len(predictions) - 5} more races omitted from console summary.")

    return "\n".join(lines)


def _format_live_prediction_summary(
    *,
    predictions: list[RacePrediction],
    recommendations: list,
    output_path: Path,
    now: datetime,
    candidate_race_count: int,
    skipped: list[dict[str, Any]],
) -> str:
    reason_counts: dict[str, int] = {}
    for item in skipped:
        reason = item.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines = [
        f"Generated live predictions at {now.isoformat()}",
        f"Candidate races: {candidate_race_count}",
        f"Ready races: {len(predictions)}",
        f"Skipped races: {len(skipped)}",
    ]
    if reason_counts:
        summary = ", ".join(f"{reason}={count}" for reason, count in sorted(reason_counts.items()))
        lines.append(f"Skip reasons: {summary}")
    lines.append(_format_prediction_summary(predictions, recommendations, output_path))
    return "\n".join(lines)


def _build_recommendations(
    predictions: list[RacePrediction],
    artifact: dict | None,
    betting_policy: dict,
    odds_overrides: dict[str, dict[str, float]] | None = None,
) -> list:
    payout_model = artifact.get("payout_model") if artifact else None
    recommendations = []
    odds_overrides = odds_overrides or {}
    needs_live_odds = any(
        _race_key(prediction.race.date, prediction.race.venue_code, prediction.race.race_no) not in odds_overrides
        for prediction in predictions
    )
    odds_client = OfficialBoatraceClient() if needs_live_odds else None
    try:
        for prediction in predictions:
            race = prediction.race
            lane_probabilities = {
                entrant.lane: entrant.score
                for entrant in prediction.entrants
            }
            race_key = _race_key(race.date, race.venue_code, race.race_no)
            odds_map = odds_overrides.get(race_key)
            if odds_map is None:
                try:
                    odds_map = odds_client.fetch_trifecta_odds(race.date, race.venue_code, race.race_no) if odds_client else {}
                except Exception:
                    odds_map = {}
            recommendations.extend(
                generate_trifecta_recommendations(
                    race_key=race_key,
                    venue_code=race.venue_code,
                    venue_name=race.venue_name,
                    race_no=race.race_no,
                    lane_probabilities=lane_probabilities,
                    trifecta_probability_map=prediction.trifecta_probability_map,
                    payout_model=payout_model,
                    odds_map=odds_map,
                    policy=betting_policy,
                )
            )
    finally:
        if odds_client:
            odds_client.close()

    recommendations.sort(
        key=lambda item: (-item.expected_value, -item.probability_ratio, item.race_key, item.combination)
    )
    return recommendations


def _race_prediction_payload(prediction: RacePrediction) -> dict:
    ranked = sorted(prediction.entrants, key=lambda item: item.win_probability, reverse=True)
    rank_map = {entrant.lane: index for index, entrant in enumerate(ranked, start=1)}
    race = prediction.race
    return {
        "race_key": _race_key(race.date, race.venue_code, race.race_no),
        "stadium": int(race.venue_code),
        "stadium_name": race.venue_name,
        "race_number": race.race_no,
        "deadline": race.deadline,
        "boats": [
            {
                "boat_number": entrant.lane,
                "racer_name": entrant.name,
                "grade": entrant.grade,
                "win_prob": round(entrant.win_probability * 100, 2),
                "top3_prob": round(entrant.top3_probability * 100, 2),
                "predicted_rank": rank_map[entrant.lane],
            }
            for entrant in sorted(prediction.entrants, key=lambda item: item.lane)
        ],
        "trifectas": [
            {
                "combination": "-".join(str(lane) for lane in trifecta.order),
                "probability": round(trifecta.probability * 100, 2),
            }
            for trifecta in prediction.trifectas
        ],
    }


def _race_key(race_date: str, venue_code: str, race_no: int) -> str:
    return f"{compact_race_date(race_date)}_{str(venue_code).zfill(2)}_{int(race_no):02d}"


def _add_prediction_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_venue: bool,
    include_race_no: bool,
) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default="artifacts/predictions")
    parser.add_argument("--race-date", default=None)
    parser.add_argument("--model-path", default=None, help="Optional trained model artifact path.")
    parser.add_argument(
        "--venue",
        action="append",
        required=require_venue,
        help="Venue code or name. Repeat this option to target multiple venues.",
    )
    if include_race_no:
        parser.add_argument("--race-no", type=int, default=None, help="Race number to predict.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of trifecta candidates to keep.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel fetch workers.")
    parser.add_argument("--betting-preset", choices=["monthly-roi", "repeated-roi", "structural-roi"], default=None)
    parser.add_argument("--min-expected-value", type=float, default=None, help="Override recommendation EV threshold.")
    parser.add_argument("--max-per-race", type=int, default=None, help="Override max recommendations per race.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Override candidate trifecta pool size.")
    parser.add_argument("--min-probability", type=float, default=None, help="Override minimum model probability.")
    parser.add_argument("--min-edge", type=float, default=None, help="Override minimum market edge.")
    parser.add_argument("--min-market-odds", type=float, default=None, help="Override minimum market odds.")
    parser.add_argument("--max-market-odds", type=float, default=None, help="Override maximum market odds.")
    parser.add_argument("--min-top-win-probability", type=float, default=None, help="Only keep races where the top win probability exceeds this threshold.")
    parser.add_argument("--min-win-margin", type=float, default=None, help="Only keep races where the top-vs-second win probability gap exceeds this threshold.")
    parser.add_argument("--required-first-lane", type=int, default=None, help="Only keep combinations with this first-place lane.")
    parser.add_argument("--required-second-lane", type=int, default=None, help="Only keep combinations with this second-place lane.")
    parser.add_argument("--required-third-lane", type=int, default=None, help="Only keep combinations with this third-place lane.")
    parser.add_argument("--allowed-venues", nargs="+", default=None, help="Only keep recommendations from these venue codes or names.")
    parser.add_argument("--clear-derived-filters", action="store_true", help="Ignore derived venue and order filters unless they are restated.")


def _parse_race_date(value: str) -> date:
    normalized = restore_race_date(compact_race_date(value))
    return datetime.fromisoformat(normalized).date()
