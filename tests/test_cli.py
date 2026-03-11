from boatrace_ai.cli import build_parser


def test_predict_command_accepts_race_date():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--race-date",
            "2026-03-10",
            "--venue",
            "24",
            "--race-no",
            "12",
            "--min-probability",
            "0.08",
            "--min-edge",
            "0.02",
        ]
    )

    assert args.command == "predict"
    assert args.race_date == "2026-03-10"
    assert args.venue == ["24"]
    assert args.race_no == 12
    assert args.min_probability == 0.08
    assert args.min_edge == 0.02


def test_collect_command_accepts_force():
    parser = build_parser()
    args = parser.parse_args(
        [
            "collect",
            "--race-date",
            "2026-03-10",
            "--venue",
            "24",
            "--force",
            "--request-timeout",
            "30",
            "--request-max-retries",
            "5",
        ]
    )

    assert args.command == "collect"
    assert args.race_date == "2026-03-10"
    assert args.force is True
    assert args.request_timeout == 30
    assert args.request_max_retries == 5


def test_train_command_accepts_roi_guards():
    parser = build_parser()
    args = parser.parse_args(
        [
            "train",
            "--dataset-path",
            "data/processed/entrants.csv",
            "--min-recommendation-roi",
            "1.2",
            "--min-walk-forward-recommendation-roi",
            "1.2",
        ]
    )

    assert args.command == "train"
    assert args.min_recommendation_roi == 1.2
    assert args.min_walk_forward_recommendation_roi == 1.2


def test_backtest_command_accepts_bankroll_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "backtest",
            "--dataset-path",
            "data/processed/entrants.csv",
            "--train-end-date",
            "2026-03-09",
            "--bankroll-mode",
            "all",
            "--starting-bankroll-yen",
            "50000",
            "--kelly-fraction",
            "0.25",
            "--kelly-cap-fraction",
            "0.1",
            "--max-bet-bankroll-fraction",
            "0.08",
            "--max-daily-bets",
            "5",
            "--max-race-exposure-fraction",
            "0.15",
            "--daily-stop-loss-yen",
            "2000",
        ]
    )

    assert args.command == "backtest"
    assert args.train_end_date == "2026-03-09"
    assert args.bankroll_mode == "all"
    assert args.starting_bankroll_yen == 50000
    assert args.kelly_fraction == 0.25
    assert args.kelly_cap_fraction == 0.1
    assert args.max_bet_bankroll_fraction == 0.08
    assert args.max_daily_bets == 5
    assert args.max_race_exposure_fraction == 0.15
    assert args.daily_stop_loss_yen == 2000


def test_note_morning_command_accepts_race_date():
    parser = build_parser()
    args = parser.parse_args(["note-morning", "--race-date", "2026-03-10"])

    assert args.command == "note-morning"
    assert args.race_date == "2026-03-10"


def test_note_evening_command_accepts_max_workers():
    parser = build_parser()
    args = parser.parse_args(["note-evening", "--race-date", "2026-03-10", "--max-workers", "2"])

    assert args.command == "note-evening"
    assert args.race_date == "2026-03-10"
    assert args.max_workers == 2
