"""SQLite storage for historical race records."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from boatrace_ai.collect.history import iter_race_record_paths, load_race_record
from boatrace_ai.collect.official import compact_race_date, restore_race_date


def init_history_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS race_records (
                race_key TEXT PRIMARY KEY,
                race_date TEXT NOT NULL,
                venue_code TEXT NOT NULL,
                venue_name TEXT NOT NULL,
                race_no INTEGER NOT NULL,
                meeting_name TEXT,
                deadline TEXT,
                has_beforeinfo INTEGER NOT NULL,
                has_result INTEGER NOT NULL,
                has_odds INTEGER NOT NULL,
                collected_at TEXT,
                record_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_race_records_race_date ON race_records (race_date)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_race_records_venue_date ON race_records (venue_code, race_date)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_race_records_venue_date_no ON race_records (venue_code, race_date, race_no)"
        )


def upsert_race_record(db_path: Path, record: dict[str, Any]) -> None:
    normalized = _normalize_record(record)
    init_history_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO race_records (
                race_key,
                race_date,
                venue_code,
                venue_name,
                race_no,
                meeting_name,
                deadline,
                has_beforeinfo,
                has_result,
                has_odds,
                collected_at,
                record_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_key) DO UPDATE SET
                race_date = excluded.race_date,
                venue_code = excluded.venue_code,
                venue_name = excluded.venue_name,
                race_no = excluded.race_no,
                meeting_name = excluded.meeting_name,
                deadline = excluded.deadline,
                has_beforeinfo = excluded.has_beforeinfo,
                has_result = excluded.has_result,
                has_odds = excluded.has_odds,
                collected_at = excluded.collected_at,
                record_json = excluded.record_json
            """,
            (
                normalized["race_key"],
                normalized["race_date"],
                normalized["venue_code"],
                normalized["venue_name"],
                normalized["race_no"],
                normalized["meeting_name"],
                normalized["deadline"],
                normalized["has_beforeinfo"],
                normalized["has_result"],
                normalized["has_odds"],
                normalized["collected_at"],
                normalized["record_json"],
            ),
        )


def import_race_records_to_db(
    input_dir: Path,
    db_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    venue_filters: list[str] | None = None,
) -> dict[str, Any]:
    init_history_db(db_path)
    scanned = 0
    imported = 0
    record_paths = iter_race_record_paths(input_dir)
    for path in record_paths:
        scanned += 1
        record = load_race_record(path)
        if not _record_matches_filters(
            record,
            start_date=start_date,
            end_date=end_date,
            venue_filters=venue_filters,
        ):
            continue
        upsert_race_record(db_path, record)
        imported += 1

    with sqlite3.connect(db_path) as connection:
        total_records = int(connection.execute("SELECT COUNT(*) FROM race_records").fetchone()[0])

    return {
        "input_dir": str(input_dir),
        "db_path": str(db_path),
        "scanned_files": scanned,
        "imported_records": imported,
        "total_records": total_records,
        "start_date": _normalize_date(start_date) if start_date else None,
        "end_date": _normalize_date(end_date) if end_date else None,
        "venue_filters": venue_filters or [],
    }


def iter_race_records_from_db(
    db_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    venue_filters: list[str] | None = None,
) -> list[dict[str, Any]]:
    init_history_db(db_path)
    conditions: list[str] = []
    parameters: list[Any] = []
    if start_date:
        conditions.append("race_date >= ?")
        parameters.append(_normalize_date(start_date))
    if end_date:
        conditions.append("race_date <= ?")
        parameters.append(_normalize_date(end_date))

    query = "SELECT record_json FROM race_records"
    if conditions:
        query = f"{query} WHERE {' AND '.join(conditions)}"
    query = f"{query} ORDER BY race_date, venue_code, race_no"

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()

    records = [json.loads(row[0]) for row in rows]
    if not venue_filters:
        return records
    return [record for record in records if _record_matches_venue(record, venue_filters)]


def _normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    race_date = _normalize_date(str(record["date"]))
    venue_code = str(record["venue_code"]).zfill(2)
    race_no = int(record["race_no"])
    return {
        "race_key": f"{race_date}_{venue_code}_{race_no:02d}",
        "race_date": race_date,
        "venue_code": venue_code,
        "venue_name": record.get("venue_name") or "",
        "race_no": race_no,
        "meeting_name": record.get("meeting_name"),
        "deadline": record.get("deadline"),
        "has_beforeinfo": 1 if record.get("beforeinfo") else 0,
        "has_result": 1 if record.get("result") else 0,
        "has_odds": 1 if record.get("trifecta_odds") else 0,
        "collected_at": record.get("collected_at"),
        "record_json": json.dumps(record, ensure_ascii=False),
    }


def _record_matches_filters(
    record: dict[str, Any],
    *,
    start_date: str | None,
    end_date: str | None,
    venue_filters: list[str] | None,
) -> bool:
    record_date = _normalize_date(str(record["date"]))
    if start_date and record_date < _normalize_date(start_date):
        return False
    if end_date and record_date > _normalize_date(end_date):
        return False
    return _record_matches_venue(record, venue_filters or [])


def _record_matches_venue(record: dict[str, Any], venue_filters: list[str]) -> bool:
    if not venue_filters:
        return True
    venue_code = str(record.get("venue_code", "")).zfill(2)
    venue_name = str(record.get("venue_name", ""))
    normalized_filters = [str(value).strip() for value in venue_filters if str(value).strip()]
    return any(
        candidate == venue_code
        or candidate.zfill(2) == venue_code
        or candidate in venue_name
        for candidate in normalized_filters
    )


def _normalize_date(value: str) -> str:
    return restore_race_date(compact_race_date(value))
