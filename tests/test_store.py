import json
from pathlib import Path

from boatrace_ai.features.dataset import build_dataset
from boatrace_ai.store.sqlite import import_race_records_to_db, iter_race_records_from_db


def test_import_race_records_to_db_and_build_dataset(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "20260310"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record = _make_raw_record("2026-03-10", 12, 1)
    (raw_dir / "24_12.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    db_path = tmp_path / "external" / "history.sqlite"
    summary = import_race_records_to_db(
        input_dir=tmp_path / "raw",
        db_path=db_path,
        start_date="2026-03-10",
        end_date="2026-03-10",
        venue_filters=["24"],
    )

    assert summary["imported_records"] == 1
    assert summary["total_records"] == 1

    records = iter_race_records_from_db(
        db_path=db_path,
        start_date="2026-03-10",
        end_date="2026-03-10",
        venue_filters=["24"],
    )
    assert len(records) == 1
    assert records[0]["race_no"] == 12

    dataset_path = tmp_path / "processed" / "entrants.csv"
    dataset_summary = build_dataset(
        output_path=dataset_path,
        db_path=db_path,
        start_date="2026-03-10",
        end_date="2026-03-10",
        venue_filters=["24"],
    )

    assert dataset_summary["source"] == "db"
    assert dataset_summary["records"] == 1
    assert dataset_summary["rows"] == 3


def _make_raw_record(race_date: str, race_no: int, winner_lane: int) -> dict:
    second_lane = 3 if winner_lane == 1 else 1
    third_lane = 2
    entrants = [
        {
            "lane": 1,
            "racer_id": "4320",
            "grade": "A1",
            "name": "峰竜太",
            "branch": "佐賀",
            "prefecture": "佐賀",
            "age": 40,
            "weight_kg": 52.0,
            "f_count": 0,
            "l_count": 0,
            "average_start_timing": 0.14,
            "national_win_rate": 7.7,
            "national_2ren_rate": 47.6,
            "national_3ren_rate": 73.2,
            "local_win_rate": 8.4,
            "local_2ren_rate": 63.8,
            "local_3ren_rate": 77.7,
            "motor_no": 47,
            "motor_2ren_rate": 34.7,
            "motor_3ren_rate": 49.4,
            "boat_no": 47,
            "boat_2ren_rate": 35.5,
            "boat_3ren_rate": 57.8,
            "recent_starts": [0.11, 0.12],
            "recent_finishes": [1, 2],
        },
        {
            "lane": 2,
            "racer_id": "5043",
            "grade": "A1",
            "name": "中村日向",
            "branch": "香川",
            "prefecture": "香川",
            "age": 27,
            "weight_kg": 52.1,
            "f_count": 0,
            "l_count": 0,
            "average_start_timing": 0.13,
            "national_win_rate": 6.1,
            "national_2ren_rate": 40.9,
            "national_3ren_rate": 54.5,
            "local_win_rate": 7.0,
            "local_2ren_rate": 55.0,
            "local_3ren_rate": 70.0,
            "motor_no": 14,
            "motor_2ren_rate": 30.5,
            "motor_3ren_rate": 42.5,
            "boat_no": 76,
            "boat_2ren_rate": 30.1,
            "boat_3ren_rate": 47.0,
            "recent_starts": [0.15, 0.14],
            "recent_finishes": [4, 5],
        },
        {
            "lane": 3,
            "racer_id": "3876",
            "grade": "A1",
            "name": "中辻崇人",
            "branch": "福岡",
            "prefecture": "福岡",
            "age": 48,
            "weight_kg": 52.0,
            "f_count": 0,
            "l_count": 0,
            "average_start_timing": 0.15,
            "national_win_rate": 7.2,
            "national_2ren_rate": 50.0,
            "national_3ren_rate": 66.0,
            "local_win_rate": 7.8,
            "local_2ren_rate": 58.0,
            "local_3ren_rate": 72.0,
            "motor_no": 22,
            "motor_2ren_rate": 42.0,
            "motor_3ren_rate": 55.0,
            "boat_no": 12,
            "boat_2ren_rate": 40.0,
            "boat_3ren_rate": 60.0,
            "recent_starts": [0.09, 0.10],
            "recent_finishes": [1, 2],
        },
    ]
    beforeinfo = {
        "date": race_date,
        "venue_code": "24",
        "venue_name": "大村",
        "race_no": race_no,
        "weather": {
            "weather": "晴",
            "temperature_c": 10.0,
            "wind_speed_mps": 3.0,
            "water_temperature_c": 13.0,
            "wave_height_cm": 2.0,
            "wind_direction": "is-direction3",
        },
        "entrants": [
            {
                "lane": 1,
                "name": "峰竜太",
                "display_weight_kg": 52.4,
                "exhibition_time": 6.71,
                "tilt": 0.0,
                "adjusted_weight_kg": None,
                "parts_exchange": ["リング×2"],
                "propeller_note": None,
                "start_display_st": 0.11,
            },
            {
                "lane": 2,
                "name": "中村日向",
                "display_weight_kg": 52.1,
                "exhibition_time": 6.76,
                "tilt": -0.5,
                "adjusted_weight_kg": None,
                "parts_exchange": [],
                "propeller_note": None,
                "start_display_st": 0.15,
            },
            {
                "lane": 3,
                "name": "中辻崇人",
                "display_weight_kg": 52.0,
                "exhibition_time": 6.70,
                "tilt": 0.0,
                "adjusted_weight_kg": None,
                "parts_exchange": [],
                "propeller_note": None,
                "start_display_st": 0.09,
            },
        ],
    }
    result = {
        "date": race_date,
        "venue_code": "24",
        "venue_name": "大村",
        "race_no": race_no,
        "technique": "逃げ",
        "weather": beforeinfo["weather"],
        "entrants": [
            {
                "finish_label": "1",
                "finish_position": 1,
                "lane": winner_lane,
                "racer_id": entrants[winner_lane - 1]["racer_id"],
                "name": entrants[winner_lane - 1]["name"],
                "race_time": "1'49\"0",
            },
            {
                "finish_label": "2",
                "finish_position": 2,
                "lane": second_lane,
                "racer_id": entrants[second_lane - 1]["racer_id"],
                "name": entrants[second_lane - 1]["name"],
                "race_time": "1'50\"0",
            },
            {
                "finish_label": "3",
                "finish_position": 3,
                "lane": third_lane,
                "racer_id": entrants[third_lane - 1]["racer_id"],
                "name": entrants[third_lane - 1]["name"],
                "race_time": "1'51\"0",
            },
        ],
        "start_timings": {"1": 0.09, "2": 0.17, "3": 0.14},
        "payouts": [
            {"bet_type": "単勝", "combination": str(winner_lane), "payout_yen": 162, "popularity": 1},
            {
                "bet_type": "3連単",
                "combination": f"{winner_lane}-{second_lane}-{third_lane}",
                "payout_yen": 1120,
                "popularity": 3,
            },
        ],
        "notes": None,
    }
    return {
        "date": race_date,
        "venue_code": "24",
        "venue_name": "大村",
        "race_no": race_no,
        "meeting_name": "開設73周年記念 海の王者決定戦",
        "deadline": "20:45",
        "collected_at": "2026-03-10T12:00:00+09:00",
        "card": {
            "date": race_date,
            "venue_code": "24",
            "venue_name": "大村",
            "race_no": race_no,
            "meeting_name": "開設73周年記念 海の王者決定戦",
            "deadline": "20:45",
            "entrants": entrants,
        },
        "beforeinfo": beforeinfo,
        "result": result,
        "trifecta_odds": {
            f"{winner_lane}-{second_lane}-{third_lane}": 18.2,
            f"{second_lane}-{winner_lane}-{third_lane}": 42.0,
        },
    }
