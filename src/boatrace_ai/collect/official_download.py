"""Official daily download ingestion for historical BOAT RACE data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
import unicodedata
from urllib import error as urllib_error
from urllib import request as urllib_request
from zoneinfo import ZoneInfo

from boatrace_ai.collect.history import load_race_record, write_race_record
from boatrace_ai.store.sqlite import upsert_race_record


DEFAULT_TIMEZONE = ZoneInfo("Asia/Tokyo")
DOWNLOAD_BASE_URL = "https://www1.mbrace.or.jp/od2"
PROGRAM_HEADER_RE = re.compile(
    r"^(?P<race_no>\d{1,2})R\s+(?P<race_title>.+?)\s+H1800m\s+電話投票締切予定(?P<deadline>\d{2}:\d{2})$"
)
PROGRAM_ENTRANT_RE = re.compile(
    r"^(?P<lane>\d)\s*(?P<racer_id>\d{4})(?P<name>.+?)(?P<age>\d{2})(?P<branch>.{2})(?P<weight>\d{2})(?P<grade>A1|A2|B1|B2)"
    r"\s+(?P<national_win>[\d.]+)\s+(?P<national_2ren>[\d.]+)"
    r"\s+(?P<local_win>[\d.]+)\s+(?P<local_2ren>[\d.]+)"
    r"\s+(?P<motor_no>\d{1,3})\s+(?P<motor_2ren>[\d.]+)"
    r"\s+(?P<boat_no>\d{1,3})\s+(?P<boat_2ren>[\d.]+)(?:\s+.*)?$"
)
RESULT_HEADER_RE = re.compile(
    r"^(?P<race_no>\d{1,2})R\s+(?P<race_title>.+?)\s+H1800m\s+(?P<weather>\S+)\s+風\s+(?P<wind_direction>\S+)\s+(?P<wind_speed>\d+)m\s+波\s+(?P<wave_height>\d+)cm$"
)
RESULT_ENTRANT_RE = re.compile(
    r"^(?P<finish_label>\S{2})\s+(?P<lane>\d)\s+(?P<racer_id>\d{4})\s+(?P<name>.+?)\s+"
    r"(?P<motor_no>\d{1,3})\s+(?P<boat_no>\d{1,3})\s+(?P<exhibition_time>[\d.]+)\s+"
    r"(?P<entry_course>\d)\s+(?P<start_timing>[\d.]+)\s+(?P<race_time>[\d. ]+)$"
)
DAY_LINE_RE = re.compile(
    r"^第\s*(?P<meeting_day>\d+)日\s+"
    r"(?P<year>\d{4})(?:/|年)\s*(?P<month>\d{1,2})(?:/|月)\s*(?P<day>\d{1,2})日?\s+"
    r"ボートレース(?P<venue>.+)$"
)
VENUE_CODE_MAP = {
    "桐生": "01",
    "戸田": "02",
    "江戸川": "03",
    "平和島": "04",
    "多摩川": "05",
    "浜名湖": "06",
    "蒲郡": "07",
    "常滑": "08",
    "津": "09",
    "三国": "10",
    "びわこ": "11",
    "住之江": "12",
    "尼崎": "13",
    "鳴門": "14",
    "丸亀": "15",
    "児島": "16",
    "宮島": "17",
    "徳山": "18",
    "下関": "19",
    "若松": "20",
    "芦屋": "21",
    "福岡": "22",
    "唐津": "23",
    "大村": "24",
}
PAYOUT_TYPES = {
    "単勝": "単勝",
    "3連単": "3連単",
}


@dataclass(frozen=True)
class DownloadSyncSummary:
    start_date: str
    end_date: str
    attempted_dates: int
    downloaded_dates: int
    written_files: int
    updated_files: int
    failed_dates: list[str]
    failed_races: list[str]
    output_dir: str
    db_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "attempted_dates": self.attempted_dates,
            "downloaded_dates": self.downloaded_dates,
            "written_files": self.written_files,
            "updated_files": self.updated_files,
            "failed_dates": self.failed_dates,
            "failed_races": self.failed_races,
            "output_dir": self.output_dir,
            "db_path": self.db_path,
        }


def sync_official_download_to_db(
    *,
    output_dir: Path,
    db_path: Path,
    start_date: str,
    end_date: str,
    max_workers: int = 4,
    request_timeout: float = 30.0,
) -> DownloadSyncSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_files = 0
    updated_files = 0
    failed_dates: list[str] = []
    failed_races: list[str] = []
    downloaded_dates = 0
    race_dates = _date_range(start_date, end_date)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(race_dates) or 1))) as executor:
        futures = {
            executor.submit(fetch_race_records_from_official_download, race_date, request_timeout): race_date
            for race_date in race_dates
        }
        for future in as_completed(futures):
            race_date = futures[future]
            try:
                records = future.result()
            except Exception:
                failed_dates.append(race_date)
                continue
            if not records:
                failed_dates.append(race_date)
                continue

            downloaded_dates += 1
            for record in records:
                race_path = output_dir / compact_date(record["date"]) / f"{record['venue_code']}_{int(record['race_no']):02d}.json"
                existing_record = load_race_record(race_path) if race_path.exists() else None
                merged_record = _merge_race_records(existing_record, record)
                write_race_record(output_dir, merged_record)
                upsert_race_record(db_path, merged_record)
                if existing_record is None:
                    written_files += 1
                else:
                    updated_files += 1
                if not merged_record.get("card") or not merged_record.get("result"):
                    failed_races.append(f"{compact_date(merged_record['date'])}_{merged_record['venue_code']}_{int(merged_record['race_no']):02d}")

    return DownloadSyncSummary(
        start_date=start_date,
        end_date=end_date,
        attempted_dates=len(race_dates),
        downloaded_dates=downloaded_dates,
        written_files=written_files,
        updated_files=updated_files,
        failed_dates=sorted(failed_dates),
        failed_races=sorted(set(failed_races)),
        output_dir=str(output_dir),
        db_path=str(db_path),
    )


def fetch_race_records_from_official_download(race_date: str, request_timeout: float = 30.0) -> list[dict[str, Any]]:
    program_text = _download_archive_text(race_date, kind="B", request_timeout=request_timeout)
    result_text = _download_archive_text(race_date, kind="K", request_timeout=request_timeout)

    if not program_text or not result_text:
        return []

    program_records = parse_program_text(program_text)
    result_records = parse_result_text(result_text)
    race_keys = sorted(set(program_records) | set(result_records))
    now_iso = datetime.now(tz=DEFAULT_TIMEZONE).isoformat()
    records: list[dict[str, Any]] = []

    for race_key in race_keys:
        program = program_records.get(race_key, {})
        result = result_records.get(race_key, {})
        venue_name = program.get("venue_name") or result.get("venue_name")
        if not venue_name:
            continue
        venue_code = venue_code_for_name(venue_name)
        if not venue_code:
            continue
        race_no = int(program.get("race_no") or result.get("race_no"))
        record = {
            "date": race_date,
            "venue_code": venue_code,
            "venue_name": venue_name,
            "race_no": race_no,
            "meeting_name": program.get("meeting_name") or result.get("meeting_name"),
            "deadline": program.get("deadline"),
            "collected_at": now_iso,
            "data_source": "official_download",
            "card": program.get("card"),
            "beforeinfo": None,
            "result": result.get("result"),
            "trifecta_odds": None,
        }
        if record["card"]:
            records.append(record)

    return records


def parse_program_text(text: str) -> dict[tuple[str, int], dict[str, Any]]:
    lines = _normalized_lines(text)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    section_history: list[str] = []
    current_venue_name: str | None = None
    current_meeting_name: str | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.endswith("BGN") or stripped.endswith("END"):
            section_history = []
        if stripped:
            section_history.append(stripped)
            section_history = section_history[-6:]

        day_match = DAY_LINE_RE.match(stripped)
        if day_match:
            current_venue_name = normalize_venue_name(day_match.group("venue"))
            current_meeting_name = _pick_meeting_name(section_history[:-1])
            index += 1
            continue

        header_match = PROGRAM_HEADER_RE.match(stripped)
        if header_match and current_venue_name:
            race_no = int(header_match.group("race_no"))
            deadline = header_match.group("deadline")
            entrants: list[dict[str, Any]] = []
            cursor = index + 1
            while cursor < len(lines):
                candidate = lines[cursor].strip()
                if not candidate:
                    cursor += 1
                    continue
                if DAY_LINE_RE.match(candidate) or PROGRAM_HEADER_RE.match(candidate):
                    break
                entrant_match = PROGRAM_ENTRANT_RE.match(candidate)
                if entrant_match:
                    entrants.append(_program_entrant_from_match(entrant_match))
                cursor += 1

            records[(current_venue_name, race_no)] = {
                "venue_name": current_venue_name,
                "meeting_name": current_meeting_name,
                "race_no": race_no,
                "deadline": deadline,
                "card": {
                    "date": None,
                    "venue_code": venue_code_for_name(current_venue_name),
                    "venue_name": current_venue_name,
                    "race_no": race_no,
                    "meeting_name": current_meeting_name,
                    "deadline": deadline,
                    "entrants": entrants,
                },
            }
            index = cursor
            continue

        index += 1

    return records


def parse_result_text(text: str) -> dict[tuple[str, int], dict[str, Any]]:
    lines = _normalized_lines(text)
    records: dict[tuple[str, int], dict[str, Any]] = {}
    section_history: list[str] = []
    current_venue_name: str | None = None
    current_meeting_name: str | None = None
    current_header: dict[str, Any] | None = None
    current_entrants: list[dict[str, Any]] = []
    current_payouts: list[dict[str, Any]] = []
    current_start_timings: dict[str, float] = {}
    current_technique: str | None = None

    def flush_current() -> None:
        nonlocal current_header, current_entrants, current_payouts, current_start_timings, current_technique
        if not current_header or not current_venue_name:
            return
        race_no = int(current_header["race_no"])
        weather = {
            "weather": current_header["weather"],
            "temperature_c": None,
            "wind_speed_mps": float(current_header["wind_speed"]),
            "water_temperature_c": None,
            "wave_height_cm": float(current_header["wave_height"]),
            "wind_direction": current_header["wind_direction"],
        }
        records[(current_venue_name, race_no)] = {
            "venue_name": current_venue_name,
            "meeting_name": current_meeting_name,
            "race_no": race_no,
            "result": {
                "date": None,
                "venue_code": venue_code_for_name(current_venue_name),
                "venue_name": current_venue_name,
                "race_no": race_no,
                "technique": current_technique,
                "weather": weather,
                "entrants": current_entrants,
                "start_timings": current_start_timings,
                "payouts": current_payouts,
                "notes": None,
            },
        }
        current_header = None
        current_entrants = []
        current_payouts = []
        current_start_timings = {}
        current_technique = None

    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.endswith("BGN") or stripped.endswith("END"):
            section_history = []
        if stripped:
            section_history.append(stripped)
            section_history = section_history[-6:]

        day_match = DAY_LINE_RE.match(stripped)
        if day_match:
            flush_current()
            current_venue_name = normalize_venue_name(day_match.group("venue"))
            current_meeting_name = _pick_meeting_name(section_history[:-1])
            continue

        header_match = RESULT_HEADER_RE.match(stripped)
        if header_match and current_venue_name:
            flush_current()
            current_header = header_match.groupdict()
            continue

        if current_header is None:
            continue

        if stripped.startswith("着 艇 登番"):
            tokens = stripped.split()
            current_technique = tokens[-1] if tokens else None
            continue

        entrant_match = RESULT_ENTRANT_RE.match(stripped)
        if entrant_match:
            entrant = _result_entrant_from_match(entrant_match)
            current_entrants.append(entrant)
            start_timing = entrant.get("start_timing")
            if start_timing is not None:
                current_start_timings[str(entrant["lane"])] = start_timing
            continue

        payout = _parse_payout_line(stripped)
        if payout:
            current_payouts.append(payout)
            continue

    flush_current()
    return records


def normalize_venue_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("ボートレース", "")
    return "".join(normalized.split())


def venue_code_for_name(venue_name: str | None) -> str | None:
    if not venue_name:
        return None
    return VENUE_CODE_MAP.get(normalize_venue_name(venue_name))


def compact_date(race_date: str) -> str:
    return race_date.replace("-", "")


def _normalized_lines(text: str) -> list[str]:
    return [unicodedata.normalize("NFKC", line.rstrip("\r\n")) for line in text.splitlines()]


def _pick_meeting_name(history: list[str]) -> str | None:
    for candidate in reversed(history):
        if (
            not candidate
            or candidate.startswith("第")
            or candidate.startswith("START")
            or candidate.endswith("BGN")
            or candidate.startswith("＊＊＊")
            or candidate.startswith("[")
            or "ボートレース" in candidate
            or "内容については主催者発行" in candidate
        ):
            continue
        return candidate
    return None


def _program_entrant_from_match(match: re.Match[str]) -> dict[str, Any]:
    group = match.groupdict()
    return {
        "lane": int(group["lane"]),
        "racer_id": group["racer_id"],
        "grade": group["grade"],
        "name": _collapse_spaces(group["name"]),
        "branch": group["branch"],
        "prefecture": group["branch"],
        "age": int(group["age"]),
        "weight_kg": float(group["weight"]),
        "f_count": 0,
        "l_count": 0,
        "average_start_timing": None,
        "national_win_rate": float(group["national_win"]),
        "national_2ren_rate": float(group["national_2ren"]),
        "national_3ren_rate": None,
        "local_win_rate": float(group["local_win"]),
        "local_2ren_rate": float(group["local_2ren"]),
        "local_3ren_rate": None,
        "motor_no": int(group["motor_no"]),
        "motor_2ren_rate": float(group["motor_2ren"]),
        "motor_3ren_rate": None,
        "boat_no": int(group["boat_no"]),
        "boat_2ren_rate": float(group["boat_2ren"]),
        "boat_3ren_rate": None,
        "recent_starts": [],
        "recent_finishes": [],
    }


def _result_entrant_from_match(match: re.Match[str]) -> dict[str, Any]:
    group = match.groupdict()
    finish_label = group["finish_label"]
    finish_position = int(finish_label) if finish_label.isdigit() else None
    return {
        "finish_label": finish_label,
        "finish_position": finish_position,
        "lane": int(group["lane"]),
        "racer_id": group["racer_id"],
        "name": _collapse_spaces(group["name"]),
        "race_time": group["race_time"].strip(),
        "exhibition_time": float(group["exhibition_time"]),
        "entry_course": int(group["entry_course"]),
        "start_timing": float(group["start_timing"]),
    }


def _parse_payout_line(line: str) -> dict[str, Any] | None:
    tokens = line.split()
    if len(tokens) < 3:
        return None
    bet_type = _normalize_bet_type(tokens[0])
    if bet_type is None:
        return None
    combination = tokens[1]
    try:
        payout_yen = int(tokens[2])
    except ValueError:
        return None
    popularity = None
    if "人気" in tokens:
        popularity_index = tokens.index("人気")
        if popularity_index + 1 < len(tokens):
            try:
                popularity = int(tokens[popularity_index + 1])
            except ValueError:
                popularity = None
    return {
        "bet_type": bet_type,
        "combination": combination,
        "payout_yen": payout_yen,
        "popularity": popularity,
    }


def _normalize_bet_type(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", value)
    return PAYOUT_TYPES.get(normalized)


def _collapse_spaces(value: str) -> str:
    return "".join(value.split())


def _download_archive_text(race_date: str, *, kind: str, request_timeout: float) -> str | None:
    year_month = race_date[:7].replace("-", "")
    yy_mm_dd = race_date[2:4] + race_date[5:7] + race_date[8:10]
    kind_dir = kind.upper()
    file_prefix = kind.lower()
    url = f"{DOWNLOAD_BASE_URL}/{kind_dir}/{year_month}/{file_prefix}{yy_mm_dd}.lzh"
    try:
        with urllib_request.urlopen(url, timeout=request_timeout) as response:
            archive_bytes = response.read()
    except urllib_error.HTTPError:
        return None
    except urllib_error.URLError:
        return None

    with tempfile.NamedTemporaryFile(suffix=".lzh") as archive_file:
        archive_file.write(archive_bytes)
        archive_file.flush()
        extracted = subprocess.run(
            ["bsdtar", "-xOf", archive_file.name],
            check=True,
            capture_output=True,
        )
    return extracted.stdout.decode("cp932", errors="ignore")


def _merge_race_records(existing: dict[str, Any] | None, downloaded: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return downloaded

    merged = dict(existing)
    merged["meeting_name"] = existing.get("meeting_name") or downloaded.get("meeting_name")
    merged["deadline"] = existing.get("deadline") or downloaded.get("deadline")
    merged["card"] = existing.get("card") or downloaded.get("card")
    merged["beforeinfo"] = existing.get("beforeinfo") or downloaded.get("beforeinfo")
    merged["result"] = existing.get("result") or downloaded.get("result")
    merged["trifecta_odds"] = existing.get("trifecta_odds") or downloaded.get("trifecta_odds")
    merged["data_source"] = existing.get("data_source") or downloaded.get("data_source")
    merged["collected_at"] = existing.get("collected_at") or downloaded.get("collected_at")
    return merged


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return [(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)]
