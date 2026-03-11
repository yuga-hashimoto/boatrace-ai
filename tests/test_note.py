import json
from datetime import date

from boatrace_ai.betting import build_payout_model, generate_trifecta_recommendations
from boatrace_ai.note.evening import compute_summary, update_cumulative, verify_recommendations
from boatrace_ai.note.morning import generate_morning_note


def test_generate_trifecta_recommendations_uses_expected_value():
    payout_model = build_payout_model(
        [
            {
                "race_key": "20260301_24_01",
                "venue_code": "24",
                "trifecta_key": "1-3-2",
                "trifecta_payout_yen": 4200,
            },
            {
                "race_key": "20260302_24_01",
                "venue_code": "24",
                "trifecta_key": "1-3-2",
                "trifecta_payout_yen": 3600,
            },
            {
                "race_key": "20260303_24_01",
                "venue_code": "24",
                "trifecta_key": "1-2-3",
                "trifecta_payout_yen": 980,
            },
        ]
    )

    recommendations = generate_trifecta_recommendations(
        race_key="20260310_24_12",
        venue_code="24",
        venue_name="大村",
        race_no=12,
        lane_probabilities={1: 0.52, 2: 0.18, 3: 0.30},
        payout_model=payout_model,
        policy={"min_expected_value": 1.0, "max_per_race": 2, "candidate_pool_size": 6},
    )

    assert recommendations
    assert recommendations[0].expected_value >= recommendations[-1].expected_value
    assert recommendations[0].avg_payout >= 980


def test_generate_morning_note_writes_html(tmp_path):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction_path = prediction_dir / "predictions_20260310_20260310T010101.json"
    prediction_path.write_text(
        json.dumps(
            {
                "recommendations": [
                    {
                        "race_key": "20260310_24_12",
                        "stadium": 24,
                        "stadium_name": "大村",
                        "race_number": 12,
                        "bet_type": "3連単",
                        "combination": "1-3-2",
                        "probability_ratio": 0.18,
                        "probability": 18.0,
                        "expected_value": 1.42,
                        "avg_payout": 790,
                    }
                ],
                "race_predictions": [
                    {
                        "race_key": "20260310_24_12",
                        "stadium": 24,
                        "stadium_name": "大村",
                        "race_number": 12,
                        "boats": [
                            {"boat_number": 1, "racer_name": "峰竜太", "win_prob": 65.0, "predicted_rank": 1},
                            {"boat_number": 3, "racer_name": "中辻崇人", "win_prob": 20.0, "predicted_rank": 2},
                            {"boat_number": 2, "racer_name": "中村日向", "win_prob": 15.0, "predicted_rank": 3},
                        ],
                    }
                ],
                "model_metrics": {"top1_hit_rate": 0.58},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = generate_morning_note(
        race_date="2026-03-10",
        prediction_dir=prediction_dir,
        output_dir=tmp_path / "note",
    )

    html = (tmp_path / "note" / "morning_20260310.html").read_text(encoding="utf-8")
    assert "期待回収率TOP20" in html
    assert result["prediction_path"] == str(prediction_path)


def test_evening_verification_and_cumulative_update():
    verified = verify_recommendations(
        [
            {
                "race_key": "20260310_24_12",
                "stadium": 24,
                "stadium_name": "大村",
                "race_number": 12,
                "bet_type": "3連単",
                "combination": "1-3-2",
                "probability": 18.0,
                "expected_value": 1.42,
                "avg_payout": 790,
            },
            {
                "race_key": "20260310_24_11",
                "stadium": 24,
                "stadium_name": "大村",
                "race_number": 11,
                "bet_type": "3連単",
                "combination": "1-2-3",
                "probability": 10.0,
                "expected_value": 1.11,
                "avg_payout": 1110,
            },
        ],
        {
            "20260310_24_12": {
                "finish_order": [1, 3, 2],
                "actual_order": "1-3-2",
                "payoff_map": {"1-3-2": 1240},
            }
        },
    )

    summary = compute_summary(verified)
    cumulative = update_cumulative({"daily_results": []}, date(2026, 3, 10), summary)

    assert summary["hits"] == 1
    assert summary["total_bets"] == 2
    assert summary["roi"] == 620.0
    assert cumulative["daily_results"][0]["date"] == "2026-03-10"
