"""Official BOAT RACE page client and parsers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from time import sleep
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import requests


warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


BASE_URL = "https://www.boatrace.jp"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}
ZENKAKU_TRANSLATION = str.maketrans(
    "０１２３４５６７８９．－",
    "0123456789.-",
)


@dataclass(frozen=True)
class RaceIndexEntry:
    date: str
    venue_code: str
    venue_name: str
    status: str
    meeting_name: str
    meeting_span: str | None
    day_label: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceEntrant:
    lane: int
    racer_id: str
    grade: str
    name: str
    branch: str | None
    hometown: str | None
    age: int | None
    weight_kg: float | None
    f_count: int
    l_count: int
    average_start_timing: float | None
    national_win_rate: float | None
    national_2ren_rate: float | None
    national_3ren_rate: float | None
    local_win_rate: float | None
    local_2ren_rate: float | None
    local_3ren_rate: float | None
    motor_no: int | None
    motor_2ren_rate: float | None
    motor_3ren_rate: float | None
    boat_no: int | None
    boat_2ren_rate: float | None
    boat_3ren_rate: float | None
    recent_starts: list[float]
    recent_finishes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceCard:
    date: str
    venue_code: str
    venue_name: str
    race_no: int
    meeting_name: str
    deadline: str | None
    entrants: list[RaceEntrant]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeforeInfoWeather:
    weather: str | None
    temperature_c: float | None
    wind_speed_mps: float | None
    water_temperature_c: float | None
    wave_height_cm: float | None
    wind_direction: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeforeInfoEntrant:
    lane: int
    name: str
    display_weight_kg: float | None
    exhibition_time: float | None
    tilt: float | None
    propeller_note: str | None
    parts_exchange: list[str]
    adjusted_weight_kg: float | None
    start_display_st: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BeforeInfo:
    date: str
    venue_code: str
    venue_name: str
    race_no: int
    weather: BeforeInfoWeather
    entrants: list[BeforeInfoEntrant]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultEntrant:
    finish_label: str
    finish_position: int | None
    lane: int
    racer_id: str
    name: str
    race_time: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResultPayout:
    bet_type: str
    combination: str
    payout_yen: int | None
    popularity: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RaceResult:
    date: str
    venue_code: str
    venue_name: str
    race_no: int
    technique: str | None
    weather: BeforeInfoWeather | None
    entrants: list[ResultEntrant]
    start_timings: dict[int, float | None]
    payouts: list[ResultPayout]
    notes: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_race_index(html: str, race_date: str) -> list[RaceIndexEntry]:
    soup = BeautifulSoup(html, "lxml")
    entries: list[RaceIndexEntry] = []

    for body in soup.select(".table1 table tbody"):
        row = body.select_one("tr")
        if row is None:
            continue

        racelist_link = row.select_one('a[href*="/owpc/pc/race/racelist"]')
        venue_image = row.select_one("img[alt]")
        meeting_link = row.select_one('a[href*="/owpc/pc/race/raceindex"]')
        if not racelist_link or not venue_image or not meeting_link:
            continue

        query = parse_qs(urlparse(racelist_link["href"]).query)
        venue_code = _first(query.get("jcd")) or ""
        cells = row.find_all("td")
        status = _clean_text(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else ""
        span_and_day = _clean_schedule_bits(cells[7] if len(cells) > 7 else None)

        entries.append(
            RaceIndexEntry(
                date=race_date,
                venue_code=venue_code.zfill(2),
                venue_name=_clean_text(venue_image.get("alt", "")),
                status=status,
                meeting_name=_clean_text(meeting_link.get_text(" ", strip=True)),
                meeting_span=span_and_day[0],
                day_label=span_and_day[1],
            )
        )

    return entries


def parse_race_card(
    html: str,
    race_date: str,
    venue_code: str,
    race_no: int,
) -> RaceCard:
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select(".table1 table")
    if len(tables) < 2:
        raise ValueError("Could not find the race card tables on the page.")

    heading = soup.select_one(".heading2")
    if heading is None:
        raise ValueError("Could not find the race heading on the page.")

    venue_image = heading.select_one(".heading2_area img")
    meeting_name = heading.select_one(".heading2_titleName")

    if venue_image is None or meeting_name is None:
        raise ValueError("Could not parse the race heading metadata.")

    deadline = _parse_deadline(tables[0], race_no)
    entrants = _parse_entrants(tables[1])

    return RaceCard(
        date=race_date,
        venue_code=venue_code.zfill(2),
        venue_name=_clean_text(venue_image.get("alt", "")),
        race_no=race_no,
        meeting_name=_clean_text(meeting_name.get_text(" ", strip=True)),
        deadline=deadline,
        entrants=entrants,
    )


def parse_beforeinfo(
    html: str,
    race_date: str,
    venue_code: str,
    race_no: int,
) -> BeforeInfo:
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select(".table1 table")
    if len(tables) < 3:
        raise ValueError("Could not find the beforeinfo tables on the page.")

    heading = soup.select_one(".heading2")
    if heading is None:
        raise ValueError("Could not find the race heading on the beforeinfo page.")

    venue_image = heading.select_one(".heading2_area img")
    if venue_image is None:
        raise ValueError("Could not parse the venue on the beforeinfo page.")

    start_display_map = _parse_start_display_table(tables[2])
    entrants = _parse_beforeinfo_entrants(tables[1], start_display_map)

    return BeforeInfo(
        date=race_date,
        venue_code=venue_code.zfill(2),
        venue_name=_clean_text(venue_image.get("alt", "")),
        race_no=race_no,
        weather=_parse_weather(soup),
        entrants=entrants,
    )


def parse_race_result(
    html: str,
    race_date: str,
    venue_code: str,
    race_no: int,
) -> RaceResult | None:
    soup = BeautifulSoup(html, "lxml")
    tables = soup.select(".table1 table")
    if len(tables) < 6:
        return None

    result_table = tables[1]
    header_cells = [
        _clean_text(cell.get_text(" ", strip=True))
        for cell in result_table.select("thead th")
    ]
    if not header_cells or "着" not in header_cells:
        return None

    heading = soup.select_one(".heading2")
    if heading is None:
        raise ValueError("Could not find the race heading on the result page.")

    venue_image = heading.select_one(".heading2_area img")
    if venue_image is None:
        raise ValueError("Could not parse the venue on the result page.")

    return RaceResult(
        date=race_date,
        venue_code=venue_code.zfill(2),
        venue_name=_clean_text(venue_image.get("alt", "")),
        race_no=race_no,
        technique=_parse_result_technique(tables[5]),
        weather=_parse_weather(soup),
        entrants=_parse_result_entrants(result_table),
        start_timings=_parse_result_start_timings(tables[2]),
        payouts=_parse_result_payouts(tables[3]),
        notes=_parse_simple_result_note(tables[6]) if len(tables) > 6 else None,
    )


def parse_trifecta_odds(html: str) -> dict[str, float]:
    soup = BeautifulSoup(html, "lxml")
    odds_table = None
    for table in reversed(soup.select(".table1 table")):
        if table.select(".oddsPoint"):
            odds_table = table
            break
    if odds_table is None:
        return {}

    header_cells = odds_table.select("thead th")
    first_lanes = [
        lane
        for lane in (
            _parse_int(_clean_text(cell.get_text(" ", strip=True)))
            for cell in header_cells
        )
        if lane is not None
    ]
    first_lanes = first_lanes[:6]
    if not first_lanes:
        return {}

    expanded_rows = _expand_table_rows(odds_table.select("tbody tr"))
    odds_map: dict[str, float] = {}

    for row in expanded_rows:
        if len(row) < len(first_lanes) * 3:
            continue
        for index, first_lane in enumerate(first_lanes):
            second_lane = _parse_int(row[index * 3])
            third_lane = _parse_int(row[index * 3 + 1])
            odds_value = _parse_measurement(row[index * 3 + 2])
            if (
                second_lane is None
                or third_lane is None
                or odds_value is None
                or first_lane == second_lane
                or third_lane in {first_lane, second_lane}
            ):
                continue
            odds_map[f"{first_lane}-{second_lane}-{third_lane}"] = odds_value

    return odds_map


class OfficialBoatraceClient:
    """Thin client for official BOAT RACE race pages."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        timeout: float = 20.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self._session = session or requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self._primed = False

    def close(self) -> None:
        self._session.close()

    def fetch_race_index(self, race_date: str) -> list[RaceIndexEntry]:
        official_date = compact_race_date(race_date)
        html = self._get_text("/owpc/pc/race/index", {"hd": official_date})
        return parse_race_index(html, restore_race_date(official_date))

    def fetch_race_card(self, race_date: str, venue_code: str, race_no: int) -> RaceCard:
        official_date = compact_race_date(race_date)
        html = self._get_text(
            "/owpc/pc/race/racelist",
            {"hd": official_date, "jcd": str(venue_code).zfill(2), "rno": str(race_no)},
        )
        return parse_race_card(
            html=html,
            race_date=restore_race_date(official_date),
            venue_code=str(venue_code).zfill(2),
            race_no=race_no,
        )

    def fetch_beforeinfo(self, race_date: str, venue_code: str, race_no: int) -> BeforeInfo:
        official_date = compact_race_date(race_date)
        html = self._get_text(
            "/owpc/pc/race/beforeinfo",
            {"hd": official_date, "jcd": str(venue_code).zfill(2), "rno": str(race_no)},
        )
        return parse_beforeinfo(
            html=html,
            race_date=restore_race_date(official_date),
            venue_code=str(venue_code).zfill(2),
            race_no=race_no,
        )

    def fetch_race_result(self, race_date: str, venue_code: str, race_no: int) -> RaceResult | None:
        official_date = compact_race_date(race_date)
        html = self._get_text(
            "/owpc/pc/race/raceresult",
            {"hd": official_date, "jcd": str(venue_code).zfill(2), "rno": str(race_no)},
        )
        return parse_race_result(
            html=html,
            race_date=restore_race_date(official_date),
            venue_code=str(venue_code).zfill(2),
            race_no=race_no,
        )

    def fetch_trifecta_odds(self, race_date: str, venue_code: str, race_no: int) -> dict[str, float]:
        official_date = compact_race_date(race_date)
        html = self._get_text(
            "/owpc/pc/race/odds3t",
            {"hd": official_date, "jcd": str(venue_code).zfill(2), "rno": str(race_no)},
        )
        return parse_trifecta_odds(html)

    def _prime(self, *, force: bool = False) -> None:
        if self._primed and not force:
            return
        response = self._session.get(BASE_URL, timeout=self._timeout)
        response.raise_for_status()
        self._primed = True

    def _get_text(self, path: str, params: dict[str, str]) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self._prime(force=attempt > 1)
                response = self._session.get(
                    urljoin(BASE_URL, path),
                    params=params,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                response.encoding = response.encoding or response.apparent_encoding or "utf-8"
                text = response.text
                if _looks_like_empty_page(text):
                    raise ValueError(f"Received an empty page for {path} with params={params}")
                return text
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                self._primed = False
                if attempt >= self._max_retries:
                    break
                sleep(self._retry_backoff_seconds * attempt)

        raise RuntimeError(f"Failed to fetch {path} after {self._max_retries} attempts") from last_error


def compact_race_date(value: str) -> str:
    candidate = value.strip()
    if re.fullmatch(r"\d{8}", candidate):
        return candidate
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
        return candidate.replace("-", "")
    raise ValueError(f"Unsupported race date format: {value}")


def restore_race_date(value: str) -> str:
    compact = compact_race_date(value)
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"


def _looks_like_empty_page(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) < 64:
        return True
    lowered = normalized.lower()
    return "<html" not in lowered and "<!doctype" not in lowered


def _parse_deadline(table: Any, race_no: int) -> str | None:
    rows = table.select("tr")
    if len(rows) < 2:
        return None

    race_labels = [_clean_text(cell.get_text(" ", strip=True)) for cell in rows[0].find_all(["td", "th"])]
    deadlines = [_clean_text(cell.get_text(" ", strip=True)) for cell in rows[1].find_all(["td", "th"])]

    label = f"{race_no}R"
    if label not in race_labels:
        return None

    index = race_labels.index(label)
    return deadlines[index] if index < len(deadlines) else None


def _parse_entrants(table: Any) -> list[RaceEntrant]:
    rows = table.select("tbody tr")
    entrants: list[RaceEntrant] = []

    for row_index in range(0, len(rows), 4):
        block = rows[row_index : row_index + 4]
        if len(block) < 4:
            continue

        base_cells = block[0].find_all("td")
        if len(base_cells) < 8:
            continue

        entrant = RaceEntrant(
            lane=_parse_int(base_cells[0].get_text()) or 0,
            **_parse_racer_identity(base_cells[2]),
            **_parse_penalty_cell(base_cells[3]),
            **_parse_triplet_stats(
                base_cells[4],
                prefix="national",
                include_number=False,
            ),
            **_parse_triplet_stats(
                base_cells[5],
                prefix="local",
                include_number=False,
            ),
            **_parse_triplet_stats(
                base_cells[6],
                prefix="motor",
                include_number=True,
            ),
            **_parse_triplet_stats(
                base_cells[7],
                prefix="boat",
                include_number=True,
            ),
            recent_starts=_parse_float_cells(block[2].find_all("td")),
            recent_finishes=_parse_int_cells(block[3].find_all("td")),
        )
        entrants.append(entrant)

    return sorted(entrants, key=lambda entrant: entrant.lane)


def _parse_racer_identity(cell: Any) -> dict[str, Any]:
    lines = list(cell.stripped_strings)
    first_line = lines[0] if lines else ""
    second_line = lines[1] if len(lines) > 1 else ""
    location = lines[3] if len(lines) > 3 else ""
    bio = lines[4] if len(lines) > 4 else ""

    racer_match = re.search(
        r"(\d+)\s*/\s*([AB]\d)",
        _normalize_numeric(f"{first_line} {second_line}"),
    )
    racer_id = racer_match.group(1) if racer_match else ""
    grade = racer_match.group(2) if racer_match else ""

    name_links = cell.find_all("a")
    name = _clean_name(name_links[-1].get_text() if name_links else (lines[2] if len(lines) > 2 else ""))

    branch: str | None = None
    hometown: str | None = None
    age: int | None = None
    weight_kg: float | None = None

    if "/" in location:
        branch, hometown = [part.strip() or None for part in location.split("/", 1)]

    age_match = re.search(r"(\d+)歳", _normalize_numeric(bio))
    weight_match = re.search(r"(\d+(?:\.\d+)?)kg", _normalize_numeric(bio))
    age = int(age_match.group(1)) if age_match else None
    weight_kg = float(weight_match.group(1)) if weight_match else None

    return {
        "racer_id": racer_id,
        "grade": grade,
        "name": name,
        "branch": branch,
        "hometown": hometown,
        "age": age,
        "weight_kg": weight_kg,
    }


def _parse_penalty_cell(cell: Any) -> dict[str, Any]:
    text = _clean_text(cell.get_text(" ", strip=True))
    f_match = re.search(r"F(\d+)", _normalize_numeric(text))
    l_match = re.search(r"L(\d+)", _normalize_numeric(text))
    floats = _parse_float_tokens(list(cell.stripped_strings))
    average_start = floats[-1] if floats else None

    return {
        "f_count": int(f_match.group(1)) if f_match else 0,
        "l_count": int(l_match.group(1)) if l_match else 0,
        "average_start_timing": average_start,
    }


def _parse_triplet_stats(cell: Any, prefix: str, include_number: bool) -> dict[str, Any]:
    values = list(cell.stripped_strings)

    if include_number:
        number = _parse_int(values[0]) if values else None
        two_ren = _parse_float(values[1]) if len(values) > 1 else None
        three_ren = _parse_float(values[2]) if len(values) > 2 else None
        return {
            f"{prefix}_no": number,
            f"{prefix}_2ren_rate": two_ren,
            f"{prefix}_3ren_rate": three_ren,
        }

    return {
        f"{prefix}_win_rate": _parse_float(values[0]) if len(values) > 0 else None,
        f"{prefix}_2ren_rate": _parse_float(values[1]) if len(values) > 1 else None,
        f"{prefix}_3ren_rate": _parse_float(values[2]) if len(values) > 2 else None,
    }


def _parse_beforeinfo_entrants(
    table: Any,
    start_display_map: dict[int, float | None],
) -> list[BeforeInfoEntrant]:
    rows = table.select("tbody tr")
    entrants: list[BeforeInfoEntrant] = []

    for row_index in range(0, len(rows), 4):
        block = rows[row_index : row_index + 4]
        if len(block) < 4:
            continue

        base_cells = block[0].find_all("td")
        if len(base_cells) < 8:
            continue

        lane = _parse_int(base_cells[0].get_text()) or 0
        parts_exchange = [_clean_text(text) for text in base_cells[7].stripped_strings]
        entrants.append(
            BeforeInfoEntrant(
                lane=lane,
                name=_clean_name(base_cells[2].get_text(" ", strip=True)),
                display_weight_kg=_parse_weight(base_cells[3].get_text(" ", strip=True)),
                exhibition_time=_parse_float(base_cells[4].get_text(" ", strip=True)),
                tilt=_parse_float(base_cells[5].get_text(" ", strip=True)),
                propeller_note=_clean_text(base_cells[6].get_text(" ", strip=True)) or None,
                parts_exchange=parts_exchange,
                adjusted_weight_kg=_parse_float(block[2].find_all("td")[0].get_text(" ", strip=True)),
                start_display_st=start_display_map.get(lane),
            )
        )

    return sorted(entrants, key=lambda entrant: entrant.lane)


def _parse_start_display_table(table: Any) -> dict[int, float | None]:
    start_display: dict[int, float | None] = {}
    for row in table.select("tr")[2:]:
        lane_node = row.select_one(".table1_boatImage1Number")
        time_node = row.select_one(".table1_boatImage1Time")
        if lane_node is None or time_node is None:
            continue
        lane = _parse_int(lane_node.get_text(" ", strip=True))
        if lane is None:
            continue
        start_display[lane] = _parse_start_value(time_node.get_text(" ", strip=True))
    return start_display


def _parse_weather(soup: BeautifulSoup) -> BeforeInfoWeather | None:
    weather_root = soup.select_one(".weather1")
    if weather_root is None:
        return None

    weather_value: str | None = None
    values: dict[str, str] = {}
    wind_direction: str | None = None

    direction_node = weather_root.select_one(".weather1_bodyUnitImage")
    if direction_node is not None:
        classes = direction_node.get("class", [])
        wind_direction = next((value for value in classes if value.startswith("is-direction")), None)

    for unit in weather_root.select(".weather1_bodyUnit"):
        title_node = unit.select_one(".weather1_bodyUnitLabelTitle")
        value_node = unit.select_one(".weather1_bodyUnitLabelData")
        if title_node is None:
            continue
        title = _clean_text(title_node.get_text(" ", strip=True))
        value = _clean_text(value_node.get_text(" ", strip=True)) if value_node else None
        if title in {"晴", "曇り", "雨", "雪"}:
            weather_value = title
        elif title:
            values[title] = value or ""

    if weather_value is None:
        weather_node = weather_root.select_one(".is-weather .weather1_bodyUnitLabelTitle")
        if weather_node is not None:
            weather_value = _clean_text(weather_node.get_text(" ", strip=True))

    return BeforeInfoWeather(
        weather=weather_value,
        temperature_c=_parse_measurement(values.get("気温")),
        wind_speed_mps=_parse_measurement(values.get("風速")),
        water_temperature_c=_parse_measurement(values.get("水温")),
        wave_height_cm=_parse_measurement(values.get("波高")),
        wind_direction=wind_direction,
    )


def _parse_result_entrants(table: Any) -> list[ResultEntrant]:
    entrants: list[ResultEntrant] = []
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        result_text = _clean_text(cells[0].get_text(" ", strip=True))
        racer_text = _clean_text(cells[2].get_text(" ", strip=True))
        racer_match = re.match(r"(\d+)\s+(.+)", racer_text)
        entrants.append(
            ResultEntrant(
                finish_label=result_text,
                finish_position=_parse_finish_position(result_text),
                lane=_parse_int(cells[1].get_text(" ", strip=True)) or 0,
                racer_id=racer_match.group(1) if racer_match else "",
                name=_clean_name(racer_match.group(2) if racer_match else racer_text),
                race_time=_clean_text(cells[3].get_text(" ", strip=True)) or None,
            )
        )
    return entrants


def _parse_result_start_timings(table: Any) -> dict[int, float | None]:
    timings: dict[int, float | None] = {}
    for row in table.select("tr")[1:]:
        lane_node = row.select_one(".table1_boatImage1Number")
        time_node = row.select_one(".table1_boatImage1TimeInner")
        if lane_node is None or time_node is None:
            continue
        lane = _parse_int(lane_node.get_text(" ", strip=True))
        if lane is None:
            continue
        timings[lane] = _parse_start_value(time_node.get_text(" ", strip=True))
    return timings


def _parse_result_payouts(table: Any) -> list[ResultPayout]:
    payouts: list[ResultPayout] = []
    current_bet_type: str | None = None
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        if len(cells) == 4:
            current_bet_type = _clean_text(cells[0].get_text(" ", strip=True)) or current_bet_type
            combination_cell = cells[1]
            payout_cell = cells[2]
            popularity_cell = cells[3]
        elif len(cells) == 3 and current_bet_type:
            combination_cell = cells[0]
            payout_cell = cells[1]
            popularity_cell = cells[2]
        else:
            continue

        combination = _clean_result_combination(combination_cell)
        payout_yen = _parse_yen(payout_cell.get_text(" ", strip=True))
        popularity = _parse_int(popularity_cell.get_text(" ", strip=True))
        if current_bet_type and combination and payout_yen is not None:
            payouts.append(
                ResultPayout(
                    bet_type=current_bet_type,
                    combination=combination,
                    payout_yen=payout_yen,
                    popularity=popularity,
                )
            )
    return payouts


def _parse_result_technique(table: Any) -> str | None:
    data_cells = table.find_all("td")
    if data_cells:
        return _clean_text(data_cells[0].get_text(" ", strip=True)) or None
    return None


def _parse_simple_result_note(table: Any) -> str | None:
    data_cells = table.find_all("td")
    if data_cells:
        return _clean_text(data_cells[0].get_text(" ", strip=True)) or None
    return None


def _parse_float_cells(cells: list[Any]) -> list[float]:
    values: list[float] = []
    for cell in cells:
        value = _parse_float(cell.get_text(" ", strip=True))
        if value is not None:
            values.append(value)
    return values


def _parse_int_cells(cells: list[Any]) -> list[int]:
    values: list[int] = []
    for cell in cells:
        value = _parse_int(cell.get_text(" ", strip=True))
        if value is not None:
            values.append(value)
    return values


def _parse_float_tokens(tokens: list[str]) -> list[float]:
    values: list[float] = []
    for token in tokens:
        value = _parse_float(token)
        if value is not None:
            values.append(value)
    return values


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    token = _normalize_numeric(value).strip()
    if not token:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    token = _normalize_numeric(value).strip()
    if not token or not re.fullmatch(r"-?\d+", token):
        return None
    return int(token)


def _normalize_numeric(value: str) -> str:
    return value.translate(ZENKAKU_TRANSLATION)


def _parse_start_value(value: str | None) -> float | None:
    if value is None:
        return None
    token = _clean_text(value)
    match = re.search(r"(F\.)?([0-9]+\.[0-9]+|\.[0-9]+)", token)
    if not match:
        return None
    number = float(match.group(2))
    return -number if match.group(1) else number


def _parse_measurement(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", _normalize_numeric(value))
    if not match:
        return None
    return float(match.group(0))


def _parse_weight(value: str | None) -> float | None:
    return _parse_measurement(value)


def _parse_finish_position(value: str | None) -> int | None:
    if value is None:
        return None
    token = _clean_text(value)
    match = re.search(r"\d+", token)
    if match:
        return int(match.group(0))
    return None


def _clean_result_combination(cell: Any) -> str:
    numbers = [_clean_text(node.get_text(" ", strip=True)) for node in cell.select(".numberSet1_number")]
    separators = [_clean_text(node.get_text(" ", strip=True)) for node in cell.select(".numberSet1_text")]
    if numbers:
        parts = [numbers[0]]
        for index, separator in enumerate(separators):
            if index + 1 < len(numbers):
                parts.append(separator)
                parts.append(numbers[index + 1])
        return "".join(parts)
    return _clean_text(cell.get_text(" ", strip=True))


def _parse_yen(value: str | None) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", _normalize_numeric(value))
    if not digits:
        return None
    return int(digits)


def _expand_table_rows(rows: list[Any]) -> list[list[str]]:
    pending: dict[int, tuple[str, int]] = {}
    expanded_rows: list[list[str]] = []

    for row in rows:
        expanded: list[str] = []
        cells = row.find_all(["td", "th"], recursive=False)
        column = 0

        for cell in cells:
            while column in pending:
                value, remaining = pending[column]
                expanded.append(value)
                if remaining <= 1:
                    pending.pop(column)
                else:
                    pending[column] = (value, remaining - 1)
                column += 1

            value = _clean_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan", 1))
            colspan = int(cell.get("colspan", 1))
            for _ in range(colspan):
                expanded.append(value)
                if rowspan > 1:
                    pending[column] = (value, rowspan - 1)
                column += 1

        while column in pending:
            value, remaining = pending[column]
            expanded.append(value)
            if remaining <= 1:
                pending.pop(column)
            else:
                pending[column] = (value, remaining - 1)
            column += 1

        expanded_rows.append(expanded)

    return expanded_rows


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", _normalize_numeric(value.replace("\u3000", " "))).strip()


def _clean_name(value: str) -> str:
    return re.sub(r"\s+", "", _normalize_numeric(value.replace("\u3000", ""))).strip()


def _first(values: list[str] | None) -> str | None:
    if not values:
        return None
    return values[0]


def _clean_schedule_bits(cell: Any) -> tuple[str | None, str | None]:
    if cell is None:
        return None, None
    bits = [_clean_text(text) for text in cell.stripped_strings]
    if not bits:
        return None, None
    if len(bits) == 1:
        return bits[0], None
    return bits[0], bits[1]
