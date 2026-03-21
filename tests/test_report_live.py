from datetime import date
from pathlib import Path

from boatrace_ai.report.live import (
    build_settled_report_lines,
    build_upcoming_report_lines,
    load_live_recommendation_index,
    recommendation_key,
)


def test_build_upcoming_report_lines_prefers_recommendations():
    payload = {
        "recommendations": [
            {
                "race_key": "20260319_24_11",
                "stadium_name": "大村",
                "race_number": 11,
                "combination": "1-2-3",
                "market_odds": 12.3,
                "probability": 8.5,
                "expected_value": 1.42,
            }
        ],
        "race_predictions": [
            {"race_key": "20260319_24_11", "deadline": "20:16"},
        ],
    }

    lines = build_upcoming_report_lines(payload, limit=5)

    assert len(lines) == 1
    assert "大村11R" in lines[0]
    assert "1-2-3" in lines[0]
    assert "@12.3倍" in lines[0]


def test_build_settled_report_lines_marks_hits_and_misses():
    recommendations = [
        {
            "race_key": "20260319_24_11",
            "stadium_name": "大村",
            "race_number": 11,
            "combination": "1-2-3",
            "market_odds": 12.3,
        },
        {
            "race_key": "20260319_24_12",
            "stadium_name": "大村",
            "race_number": 12,
            "combination": "1-2-3",
            "market_odds": 18.0,
        },
    ]
    result_map = {
        "20260319_24_11": {"actual_order": "1-2-3", "payoff_map": {"1-2-3": 3130}},
        "20260319_24_12": {"actual_order": "1-3-2", "payoff_map": {"1-3-2": 2140}},
    }

    lines, new_keys = build_settled_report_lines(
        recommendations=recommendations,
        result_map=result_map,
        reported_keys=set(),
        limit=10,
    )

    assert len(lines) == 2
    assert "的中" in lines[0]
    assert "ハズレ" in lines[1]
    assert recommendation_key(recommendations[0]) in new_keys


def test_load_live_recommendation_index_uses_latest_file(tmp_path: Path):
    first = tmp_path / "predictions_20260319_20260319T080000.json"
    second = tmp_path / "predictions_20260319_20260319T081000.json"
    first.write_text(
        '{"command":"report-live","recommendations":[{"race_key":"20260319_24_11","combination":"1-2-3","market_odds":10.0}]}',
        encoding="utf-8",
    )
    second.write_text(
        '{"command":"report-live","recommendations":[{"race_key":"20260319_24_11","combination":"1-2-3","market_odds":12.0}]}',
        encoding="utf-8",
    )

    index = load_live_recommendation_index(tmp_path, target_date=date(2026, 3, 19))

    assert index["20260319_24_11|1-2-3"]["market_odds"] == 12.0
