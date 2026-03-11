"""Historical race collection workflow."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from boatrace_ai.collect.official import (
    OfficialBoatraceClient,
    compact_race_date,
    restore_race_date,
)


DEFAULT_TIMEZONE = ZoneInfo("Asia/Tokyo")


@dataclass(frozen=True)
class CollectionSummary:
    start_date: str
    end_date: str
    total_files: int
    written_files: int
    skipped_files: int
    failed_tasks: int
    dates: list[str]
    venues: list[str]
    output_dir: str
    failed_races: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_files": self.total_files,
            "written_files": self.written_files,
            "skipped_files": self.skipped_files,
            "failed_tasks": self.failed_tasks,
            "dates": self.dates,
            "venues": self.venues,
            "output_dir": self.output_dir,
            "failed_races": self.failed_races,
        }


def collect_race_records(
    output_dir: Path,
    start_date: str,
    end_date: str | None = None,
    venue_filters: list[str] | None = None,
    race_numbers: list[int] | None = None,
    include_beforeinfo: bool = True,
    include_results: bool = True,
    include_odds: bool = True,
    max_workers: int = 4,
    skip_existing: bool = True,
    request_timeout: float = 20.0,
    request_max_retries: int = 3,
    request_retry_backoff_seconds: float = 1.0,
) -> CollectionSummary:
    resolved_end_date = end_date or start_date
    race_numbers = race_numbers or list(range(1, 13))
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = 0
    skipped_files = 0
    failed_races: list[str] = []
    seen_dates: list[str] = []
    seen_venues: set[str] = set()

    for race_date in _date_range(start_date, resolved_end_date):
        client = OfficialBoatraceClient(
            timeout=request_timeout,
            max_retries=request_max_retries,
            retry_backoff_seconds=request_retry_backoff_seconds,
        )
        try:
            entries = client.fetch_race_index(race_date)
        finally:
            client.close()

        filtered_entries = _filter_venues(entries, venue_filters or [])
        fallback_venue_codes = _explicit_venue_codes(venue_filters or [])
        task_venues = [entry.venue_code for entry in filtered_entries] or fallback_venue_codes
        tasks = [(race_date, venue_code, race_number) for venue_code in task_venues for race_number in race_numbers]
        if not tasks:
            continue

        seen_dates.append(race_date)
        seen_venues.update(task_venues)
        day_summary = _collect_date_tasks(
            tasks=tasks,
            output_dir=output_dir,
            include_beforeinfo=include_beforeinfo,
            include_results=include_results,
            include_odds=include_odds,
            max_workers=max_workers,
            skip_existing=skip_existing,
            request_timeout=request_timeout,
            request_max_retries=request_max_retries,
            request_retry_backoff_seconds=request_retry_backoff_seconds,
        )
        written_files += int(day_summary["written"])
        skipped_files += int(day_summary["skipped"])
        failed_races.extend(day_summary["failed"])

    return CollectionSummary(
        start_date=start_date,
        end_date=resolved_end_date,
        total_files=written_files + skipped_files,
        written_files=written_files,
        skipped_files=skipped_files,
        failed_tasks=len(failed_races),
        dates=seen_dates,
        venues=sorted(seen_venues),
        output_dir=str(output_dir),
        failed_races=failed_races,
    )


def build_race_record(
    race_date: str,
    venue_code: str,
    race_no: int,
    include_beforeinfo: bool = True,
    include_results: bool = True,
    include_odds: bool = True,
    request_timeout: float = 20.0,
    request_max_retries: int = 3,
    request_retry_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    client = OfficialBoatraceClient(
        timeout=request_timeout,
        max_retries=request_max_retries,
        retry_backoff_seconds=request_retry_backoff_seconds,
    )
    try:
        card = client.fetch_race_card(race_date, venue_code, race_no)
        beforeinfo = client.fetch_beforeinfo(race_date, venue_code, race_no) if include_beforeinfo else None
        result = client.fetch_race_result(race_date, venue_code, race_no) if include_results else None
        trifecta_odds = client.fetch_trifecta_odds(race_date, venue_code, race_no) if include_odds else None
    finally:
        client.close()

    return {
        "date": card.date,
        "venue_code": card.venue_code,
        "venue_name": card.venue_name,
        "race_no": card.race_no,
        "meeting_name": card.meeting_name,
        "deadline": card.deadline,
        "collected_at": datetime.now(tz=DEFAULT_TIMEZONE).isoformat(),
        "card": card.to_dict(),
        "beforeinfo": beforeinfo.to_dict() if beforeinfo else None,
        "result": result.to_dict() if result else None,
        "trifecta_odds": trifecta_odds,
    }


def write_race_record(output_dir: Path, record: dict[str, Any]) -> Path:
    date_key = compact_race_date(record["date"])
    file_dir = output_dir / date_key
    file_dir.mkdir(parents=True, exist_ok=True)
    file_path = file_dir / f"{record['venue_code']}_{int(record['race_no']):02d}.json"
    file_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def load_race_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_race_record_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def refresh_missing_trifecta_odds(
    input_dir: Path,
    max_workers: int = 4,
    request_timeout: float = 20.0,
    request_max_retries: int = 3,
    request_retry_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    record_paths = iter_race_record_paths(input_dir)
    targets: list[Path] = []
    for path in record_paths:
        record = load_race_record(path)
        if not record.get("trifecta_odds"):
            targets.append(path)

    if not targets:
        return {
            "input_dir": str(input_dir),
            "checked_files": len(record_paths),
            "updated_files": 0,
            "remaining_missing": 0,
        }

    updated = 0
    if max_workers <= 1:
        for path in targets:
            if _refresh_odds_for_record(
                path,
                request_timeout=request_timeout,
                request_max_retries=request_max_retries,
                request_retry_backoff_seconds=request_retry_backoff_seconds,
            ):
                updated += 1
    else:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as executor:
            futures = {
                executor.submit(
                    _refresh_odds_for_record,
                    path,
                    request_timeout=request_timeout,
                    request_max_retries=request_max_retries,
                    request_retry_backoff_seconds=request_retry_backoff_seconds,
                ): path
                for path in targets
            }
            for future in as_completed(futures):
                if future.result():
                    updated += 1

    remaining_missing = 0
    for path in targets:
        record = load_race_record(path)
        if not record.get("trifecta_odds"):
            remaining_missing += 1

    return {
        "input_dir": str(input_dir),
        "checked_files": len(record_paths),
        "updated_files": updated,
        "remaining_missing": remaining_missing,
    }


def _collect_date_tasks(
    tasks: list[tuple[str, str, int]],
    output_dir: Path,
    include_beforeinfo: bool,
    include_results: bool,
    include_odds: bool,
    max_workers: int,
    skip_existing: bool,
    request_timeout: float,
    request_max_retries: int,
    request_retry_backoff_seconds: float,
) -> dict[str, Any]:
    pending_tasks: list[tuple[str, str, int]] = []
    skipped = 0
    failed: list[str] = []
    for race_date, venue_code, race_no in tasks:
        if skip_existing and _is_record_complete(
            output_dir=output_dir,
            race_date=race_date,
            venue_code=venue_code,
            race_no=race_no,
            include_beforeinfo=include_beforeinfo,
            include_results=include_results,
            include_odds=include_odds,
        ):
            skipped += 1
            continue
        pending_tasks.append((race_date, venue_code, race_no))

    written = 0
    if not pending_tasks:
        return {"written": 0, "skipped": skipped, "failed": failed}

    if max_workers <= 1:
        for race_date, venue_code, race_no in pending_tasks:
            try:
                record = build_race_record(
                    race_date=race_date,
                    venue_code=venue_code,
                    race_no=race_no,
                    include_beforeinfo=include_beforeinfo,
                    include_results=include_results,
                    include_odds=include_odds,
                    request_timeout=request_timeout,
                    request_max_retries=request_max_retries,
                    request_retry_backoff_seconds=request_retry_backoff_seconds,
                )
            except Exception:
                failed.append(_race_task_key(race_date, venue_code, race_no))
                continue
            write_race_record(output_dir, record)
            written += 1
        return {"written": written, "skipped": skipped, "failed": failed}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(pending_tasks))) as executor:
        futures = {
            executor.submit(
                build_race_record,
                race_date,
                venue_code,
                race_no,
                include_beforeinfo,
                include_results,
                include_odds,
                request_timeout,
                request_max_retries,
                request_retry_backoff_seconds,
            ): (race_date, venue_code, race_no)
            for race_date, venue_code, race_no in pending_tasks
        }
        for future in as_completed(futures):
            race_date, venue_code, race_no = futures[future]
            try:
                record = future.result()
            except Exception:
                failed.append(_race_task_key(race_date, venue_code, race_no))
                continue
            write_race_record(output_dir, record)
            written += 1

    return {"written": written, "skipped": skipped, "failed": failed}


def _refresh_odds_for_record(
    path: Path,
    *,
    request_timeout: float = 20.0,
    request_max_retries: int = 3,
    request_retry_backoff_seconds: float = 1.0,
) -> bool:
    record = load_race_record(path)
    if record.get("trifecta_odds"):
        return False

    client = OfficialBoatraceClient(
        timeout=request_timeout,
        max_retries=request_max_retries,
        retry_backoff_seconds=request_retry_backoff_seconds,
    )
    try:
        odds = client.fetch_trifecta_odds(
            race_date=record["date"],
            venue_code=record["venue_code"],
            race_no=int(record["race_no"]),
        )
    except Exception:
        odds = {}
    finally:
        client.close()

    if not odds:
        return False

    record["trifecta_odds"] = odds
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    days = (end - start).days
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days + 1)]


def _parse_date(value: str) -> date:
    return date.fromisoformat(restore_race_date(compact_race_date(value)))


def _filter_venues(entries: list[Any], venue_filters: list[str]) -> list[Any]:
    if not venue_filters:
        return entries

    normalized_filters = [str(value).strip() for value in venue_filters if str(value).strip()]
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


def _explicit_venue_codes(venue_filters: list[str]) -> list[str]:
    codes: list[str] = []
    for value in venue_filters:
        candidate = str(value).strip()
        if candidate.isdigit():
            codes.append(candidate.zfill(2))
    return sorted(set(codes))


def _record_path_for(output_dir: Path, race_date: str, venue_code: str, race_no: int) -> Path:
    return output_dir / compact_race_date(race_date) / f"{str(venue_code).zfill(2)}_{int(race_no):02d}.json"


def _is_record_complete(
    *,
    output_dir: Path,
    race_date: str,
    venue_code: str,
    race_no: int,
    include_beforeinfo: bool,
    include_results: bool,
    include_odds: bool,
) -> bool:
    path = _record_path_for(output_dir, race_date, venue_code, race_no)
    if not path.exists():
        return False
    try:
        record = load_race_record(path)
    except (json.JSONDecodeError, OSError):
        return False

    if not record.get("card"):
        return False
    if include_beforeinfo and not record.get("beforeinfo"):
        return False
    if include_results and not record.get("result"):
        return False
    if include_odds and not record.get("trifecta_odds"):
        return False
    return True


def _race_task_key(race_date: str, venue_code: str, race_no: int) -> str:
    return f"{compact_race_date(race_date)}_{str(venue_code).zfill(2)}_{int(race_no):02d}"
