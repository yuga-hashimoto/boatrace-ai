import json
from pathlib import Path

from boatrace_ai.collect import history
from boatrace_ai.collect.official import RaceIndexEntry


def test_collect_race_records_skips_existing_complete_record(tmp_path, monkeypatch):
    output_dir = tmp_path / "raw"
    date_dir = output_dir / "20260310"
    date_dir.mkdir(parents=True)
    (date_dir / "24_01.json").write_text(
        json.dumps(
            {
                "date": "2026-03-10",
                "venue_code": "24",
                "race_no": 1,
                "card": {"entrants": []},
                "beforeinfo": {"entrants": []},
                "result": {"entrants": []},
                "trifecta_odds": {"1-2-3": 12.3},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(history, "OfficialBoatraceClient", _FakeIndexClient)

    called = {"count": 0}

    def _fake_build_race_record(*args, **kwargs):
        called["count"] += 1
        raise AssertionError("build_race_record should not be called for a complete existing record")

    monkeypatch.setattr(history, "build_race_record", _fake_build_race_record)

    summary = history.collect_race_records(
        output_dir=output_dir,
        start_date="2026-03-10",
        end_date="2026-03-10",
        venue_filters=["24"],
        race_numbers=[1],
        max_workers=1,
    )

    assert summary.written_files == 0
    assert summary.skipped_files == 1
    assert summary.failed_tasks == 0
    assert called["count"] == 0


def test_collect_race_records_tracks_failed_tasks(tmp_path, monkeypatch):
    output_dir = tmp_path / "raw"
    monkeypatch.setattr(history, "OfficialBoatraceClient", _FakeIndexClient)

    def _fake_build_race_record(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(history, "build_race_record", _fake_build_race_record)

    summary = history.collect_race_records(
        output_dir=output_dir,
        start_date="2026-03-10",
        end_date="2026-03-10",
        venue_filters=["24"],
        race_numbers=[1],
        max_workers=1,
    )

    assert summary.written_files == 0
    assert summary.skipped_files == 0
    assert summary.failed_tasks == 1
    assert summary.failed_races == ["20260310_24_01"]


class _FakeIndexClient:
    def __init__(self, *args, **kwargs):
        return None

    def fetch_race_index(self, race_date: str):
        return [
            RaceIndexEntry(
                date=race_date,
                venue_code="24",
                venue_name="大村",
                status="発売中",
                meeting_name="テスト開催",
                meeting_span="3/10-3/10",
                day_label="初日",
            )
        ]

    def close(self):
        return None
