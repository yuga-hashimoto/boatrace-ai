from pathlib import Path
from types import SimpleNamespace

from boatrace_ai import cli
from boatrace_ai.predict.baseline import EntrantPrediction, RacePrediction, TrifectaPrediction
from boatrace_ai.collect.official import RaceCard
from boatrace_ai.cli import _resolve_betting_policy, build_parser


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
            "--betting-preset",
            "monthly-roi",
            "--min-probability",
            "0.08",
            "--min-edge",
            "0.02",
            "--min-market-odds",
            "70",
            "--max-market-odds",
            "120",
            "--min-win-margin",
            "0.3",
            "--required-second-lane",
            "2",
            "--required-third-lane",
            "3",
            "--allowed-venues",
            "20",
            "16",
            "--clear-derived-filters",
        ]
    )

    assert args.command == "predict"
    assert args.race_date == "2026-03-10"
    assert args.venue == ["24"]
    assert args.race_no == 12
    assert args.betting_preset == "monthly-roi"
    assert args.min_probability == 0.08
    assert args.min_edge == 0.02
    assert args.min_market_odds == 70
    assert args.max_market_odds == 120
    assert args.min_win_margin == 0.3
    assert args.required_second_lane == 2
    assert args.required_third_lane == 3
    assert args.allowed_venues == ["20", "16"]
    assert args.clear_derived_filters is True


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
            "--betting-preset",
            "structural-roi",
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
            "--min-win-margin",
            "0.3",
            "--required-second-lane",
            "2",
            "--required-third-lane",
            "3",
            "--allowed-venues",
            "20,16",
            "--clear-derived-filters",
        ]
    )

    assert args.command == "backtest"
    assert args.train_end_date == "2026-03-09"
    assert args.betting_preset == "structural-roi"
    assert args.bankroll_mode == "all"
    assert args.starting_bankroll_yen == 50000
    assert args.kelly_fraction == 0.25
    assert args.kelly_cap_fraction == 0.1
    assert args.max_bet_bankroll_fraction == 0.08
    assert args.max_daily_bets == 5
    assert args.max_race_exposure_fraction == 0.15
    assert args.daily_stop_loss_yen == 2000
    assert args.min_win_margin == 0.3
    assert args.required_second_lane == 2
    assert args.required_third_lane == 3
    assert args.allowed_venues == ["20,16"]
    assert args.clear_derived_filters is True


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


def test_resolve_betting_policy_drops_allowed_venues_on_manual_override():
    args = SimpleNamespace(
        min_expected_value=1.0,
        betting_preset=None,
        max_per_race=None,
        candidate_pool_size=1,
        min_probability=0.06,
        min_edge=None,
        min_market_odds=None,
        max_market_odds=40.0,
        min_top_win_probability=None,
        min_win_margin=0.3,
        required_first_lane=None,
        required_second_lane=2,
        required_third_lane=3,
        allowed_venues=None,
        clear_derived_filters=False,
    )

    policy = _resolve_betting_policy(
        args,
        config={},
        artifact={"betting_policy": {"allowed_venues": ["24"], "min_probability": 0.2}},
    )

    assert "allowed_venues" not in policy
    assert policy["required_second_lane"] == 2
    assert policy["required_third_lane"] == 3
    assert policy["min_win_margin"] == 0.3


def test_resolve_betting_policy_can_clear_stale_structural_filters():
    args = SimpleNamespace(
        min_expected_value=1.0,
        betting_preset=None,
        max_per_race=None,
        candidate_pool_size=1,
        min_probability=0.04,
        min_edge=0.0,
        min_market_odds=30.0,
        max_market_odds=50.0,
        min_top_win_probability=None,
        min_win_margin=None,
        required_first_lane=None,
        required_second_lane=None,
        required_third_lane=None,
        allowed_venues=["20", "16"],
        clear_derived_filters=True,
    )

    policy = _resolve_betting_policy(
        args,
        config={},
        artifact={
            "betting_policy": {
                "allowed_venues": ["24"],
                "required_second_lane": 2,
                "required_third_lane": 3,
                "min_win_margin": 0.3,
            }
        },
    )

    assert policy["allowed_venues"] == ["20", "16"]
    assert policy["min_probability"] == 0.04
    assert policy["min_market_odds"] == 30.0
    assert policy["max_market_odds"] == 50.0
    assert "required_second_lane" not in policy
    assert "required_third_lane" not in policy
    assert "min_win_margin" not in policy


def test_resolve_betting_policy_uses_repeated_roi_preset():
    args = SimpleNamespace(
        min_expected_value=None,
        betting_preset="repeated-roi",
        max_per_race=None,
        candidate_pool_size=None,
        min_probability=None,
        min_edge=None,
        min_market_odds=None,
        max_market_odds=None,
        min_top_win_probability=None,
        min_win_margin=None,
        required_first_lane=None,
        required_second_lane=None,
        required_third_lane=None,
        allowed_venues=["20", "16"],
        clear_derived_filters=True,
    )

    policy = _resolve_betting_policy(
        args,
        config={},
        artifact=None,
    )

    assert policy["min_probability"] == 0.04
    assert policy["min_market_odds"] == 30.0
    assert policy["max_market_odds"] == 50.0
    assert policy["allowed_venues"] == ["20", "16"]


def test_resolve_betting_policy_uses_monthly_roi_preset():
    args = SimpleNamespace(
        min_expected_value=None,
        betting_preset="monthly-roi",
        max_per_race=None,
        candidate_pool_size=None,
        min_probability=None,
        min_edge=None,
        min_market_odds=None,
        max_market_odds=None,
        min_top_win_probability=None,
        min_win_margin=None,
        required_first_lane=None,
        required_second_lane=None,
        required_third_lane=None,
        allowed_venues=None,
        clear_derived_filters=True,
    )

    policy = _resolve_betting_policy(
        args,
        config={},
        artifact=None,
    )

    assert policy["required_first_lane"] == 1
    assert policy["required_second_lane"] == 4
    assert policy["required_third_lane"] == 6
    assert policy["candidate_pool_size"] == 12
    assert policy["min_probability"] == 0.045
    assert policy["max_market_odds"] == 61.0
    assert policy["min_top_win_probability"] == 0.63


def test_resolve_betting_policy_attaches_structural_fallback_for_restrictive_artifact():
    args = SimpleNamespace(
        min_expected_value=None,
        betting_preset=None,
        max_per_race=None,
        candidate_pool_size=None,
        min_probability=None,
        min_edge=None,
        min_market_odds=None,
        max_market_odds=None,
        min_top_win_probability=None,
        min_win_margin=None,
        required_first_lane=None,
        required_second_lane=None,
        required_third_lane=None,
        allowed_venues=None,
        clear_derived_filters=False,
    )

    policy = _resolve_betting_policy(
        args,
        config={},
        artifact={
            "betting_policy": {
                "min_probability": 0.25,
                "candidate_pool_size": 1,
                "allowed_venues": ["15", "20"],
            }
        },
    )

    assert "fallback_policy" in policy
    assert policy["fallback_policy"]["required_second_lane"] == 2
    assert policy["fallback_policy"]["required_third_lane"] == 3


def test_import_db_command_accepts_filters():
    parser = build_parser()
    args = parser.parse_args(
        [
            "import-db",
            "--input-dir",
            "data/raw",
            "--db-path",
            "data/external/history.sqlite",
            "--start-date",
            "2025-03-11",
            "--end-date",
            "2026-03-10",
            "--venue",
            "24",
        ]
    )

    assert args.command == "import-db"
    assert args.db_path == "data/external/history.sqlite"
    assert args.start_date == "2025-03-11"
    assert args.end_date == "2026-03-10"
    assert args.venue == ["24"]


def test_sync_db_command_accepts_lookback_days():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-db",
            "--race-date",
            "2026-03-10",
            "--lookback-days",
            "365",
            "--db-path",
            "data/external/history.sqlite",
        ]
    )

    assert args.command == "sync-db"
    assert args.lookback_days == 365
    assert args.db_path == "data/external/history.sqlite"


def test_sync_download_db_command_accepts_lookback_days():
    parser = build_parser()
    args = parser.parse_args(
        [
            "sync-download-db",
            "--race-date",
            "2026-03-10",
            "--lookback-days",
            "365",
            "--db-path",
            "data/external/history.sqlite",
            "--max-workers",
            "8",
        ]
    )

    assert args.command == "sync-download-db"
    assert args.lookback_days == 365
    assert args.db_path == "data/external/history.sqlite"
    assert args.max_workers == 8


def test_predict_venue_command_targets_all_races(monkeypatch, tmp_path: Path):
    captured: dict[str, list[int]] = {}

    class FakeClient:
        def fetch_race_index(self, race_date: str):
            assert race_date == "2026-03-10"
            return [SimpleNamespace(venue_code="24", venue_name="大村")]

        def close(self) -> None:
            return None

    def fake_fetch_predictions(**kwargs):
        captured["race_numbers"] = kwargs["race_numbers"]
        return []

    monkeypatch.setattr(cli, "OfficialBoatraceClient", FakeClient)
    monkeypatch.setattr(cli, "_fetch_predictions", fake_fetch_predictions)
    monkeypatch.setattr(cli, "_build_recommendations", lambda predictions, artifact, betting_policy: [])

    exit_code = cli.main(
        [
            "predict-venue",
            "--race-date",
            "2026-03-10",
            "--venue",
            "24",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert captured["race_numbers"] == list(range(1, 13))


def test_predict_live_command_accepts_live_filters():
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict-live",
            "--race-date",
            "2026-03-19",
            "--lookahead-minutes",
            "120",
            "--require-odds",
            "--with-recommendations",
            "--request-timeout",
            "8",
            "--request-max-retries",
            "4",
            "--now",
            "2026-03-19T08:00:00+09:00",
        ]
    )

    assert args.command == "predict-live"
    assert args.lookahead_minutes == 120
    assert args.require_odds is True
    assert args.with_recommendations is True
    assert args.request_timeout == 8
    assert args.request_max_retries == 4
    assert args.now == "2026-03-19T08:00:00+09:00"


def test_report_live_command_accepts_report_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "report-live",
            "--race-date",
            "2026-03-19",
            "--lookahead-minutes",
            "60",
            "--state-path",
            "artifacts/reports/live_report_state.json",
            "--result-max-workers",
            "2",
            "--upcoming-limit",
            "3",
            "--settled-limit",
            "7",
            "--quiet-when-empty",
        ]
    )

    assert args.command == "report-live"
    assert args.lookahead_minutes == 60
    assert args.state_path == "artifacts/reports/live_report_state.json"
    assert args.result_max_workers == 2
    assert args.upcoming_limit == 3
    assert args.settled_limit == 7
    assert args.quiet_when_empty is True


def test_select_live_cards_keeps_only_future_window():
    now = cli._resolve_live_now("2026-03-19T08:00:00+09:00")
    cards = [
        SimpleNamespace(venue_code="24", venue_name="大村", race_no=1, deadline="07:59"),
        SimpleNamespace(venue_code="24", venue_name="大村", race_no=2, deadline="08:30"),
        SimpleNamespace(venue_code="24", venue_name="大村", race_no=3, deadline="10:10"),
    ]

    selected = cli._select_live_cards(cards, "2026-03-19", now, 90)

    assert [card.race_no for card in selected] == [2]


def test_predict_live_handler_uses_ready_races(monkeypatch, tmp_path: Path):
    fake_card = RaceCard(
        date="2026-03-19",
        venue_code="24",
        venue_name="大村",
        race_no=11,
        meeting_name="準優勝戦",
        deadline="20:16",
        entrants=[],
    )
    fake_prediction = RacePrediction(
        model_name="test_model",
        race=fake_card,
        entrants=[
            EntrantPrediction(1, "1234", "テスト太郎", "A1", 0.8, 0.6, 0.9, {}),
            EntrantPrediction(2, "1235", "テスト次郎", "A2", 0.2, 0.2, 0.6, {}),
            EntrantPrediction(3, "1236", "テスト三郎", "B1", 0.1, 0.1, 0.4, {}),
        ],
        trifectas=[
            TrifectaPrediction((1, 2, 3), 0.2),
            TrifectaPrediction((1, 3, 2), 0.15),
            TrifectaPrediction((2, 1, 3), 0.1),
        ],
    )

    monkeypatch.setattr(
        cli,
        "fetch_program_cards_from_official_download",
        lambda race_date, request_timeout, cache_path=None: [fake_card],
    )
    monkeypatch.setattr(cli, "_load_prediction_artifact", lambda model_path, config: None)
    monkeypatch.setattr(cli, "_select_live_cards", lambda cards, race_date, now, lookahead_minutes: cards)
    monkeypatch.setattr(cli, "_fetch_live_predictions", lambda **kwargs: ([fake_prediction], {}, []))
    monkeypatch.setattr(cli, "_build_recommendations", lambda predictions, artifact, betting_policy, odds_overrides=None: [])
    monkeypatch.setattr(cli, "_write_predictions", lambda output_dir, race_date, payload: tmp_path / "live.json")
    monkeypatch.setattr(cli, "_format_live_prediction_summary", lambda **kwargs: "ok")

    exit_code = cli.main(
        [
            "predict-live",
            "--race-date",
            "2026-03-19",
            "--lookahead-minutes",
            "120",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0


def test_report_live_handler_prints_report(monkeypatch, tmp_path: Path, capsys):
    monkeypatch.setattr(
        cli,
        "_run_live_prediction_job",
        lambda args, config: {
            "race_date": "2026-03-19",
            "payload": {
                "command": "report-live",
                "recommendations": [],
                "race_predictions": [],
            },
        },
    )
    monkeypatch.setattr(
        cli,
        "generate_live_report_message",
        lambda **kwargs: ("【直前予想】\n大村11R 1-2-3 @12.3倍", {"upcoming_count": 1, "settled_count": 0}),
    )

    exit_code = cli.main(
        [
            "report-live",
            "--race-date",
            "2026-03-19",
            "--output-dir",
            str(tmp_path),
            "--state-path",
            str(tmp_path / "state.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "大村11R" in captured.out
