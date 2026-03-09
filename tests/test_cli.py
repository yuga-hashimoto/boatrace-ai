from boatrace_ai.cli import build_parser


def test_predict_command_accepts_race_date():
    parser = build_parser()
    args = parser.parse_args(["predict", "--race-date", "2026-03-10"])

    assert args.command == "predict"
    assert args.race_date == "2026-03-10"
