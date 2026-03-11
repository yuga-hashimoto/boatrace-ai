from boatrace_ai.collect.official import (
    OfficialBoatraceClient,
    parse_race_card,
    parse_race_index,
    parse_trifecta_odds,
)
from boatrace_ai.predict.baseline import predict_race


INDEX_HTML = """
<html>
  <body>
    <div class="table1">
      <table>
        <tbody>
          <tr>
            <td><a href="#"><img alt="大村" /></a></td>
            <td class="is-fColor2">1R発売開始前</td>
            <td></td>
            <td></td>
            <td></td>
            <td></td>
            <td><a href="/owpc/pc/race/raceindex?jcd=24&amp;hd=20260310">開設７３周年記念 海の王者決定戦</a></td>
            <td>3/6-3/12<br />５日目</td>
            <td>
              <ul class="textLinks3">
                <li><a href="/owpc/pc/race/racelist?rno=1&amp;jcd=24&amp;hd=20260310">出走表</a></li>
              </ul>
            </td>
          </tr>
          <tr class="is-fBold is-fs15"></tr>
        </tbody>
      </table>
    </div>
  </body>
</html>
"""


def _entrant_block(
    lane: str,
    racer_id: str,
    grade: str,
    name: str,
    branch: str,
    hometown: str,
    age: int,
    weight: float,
    average_start: str,
    national: tuple[str, str, str],
    local: tuple[str, str, str],
    motor: tuple[str, str, str],
    boat: tuple[str, str, str],
    recent_starts: tuple[str, str],
    recent_finishes: tuple[str, str],
) -> str:
    return f"""
    <tr>
      <td rowspan="4">{lane}</td>
      <td rowspan="4"><a href="#"><img alt="" /></a></td>
      <td rowspan="4">
        <div>{racer_id} / <span>{grade}</span></div>
        <div><a href="#">{name}</a></div>
        <div>{branch}/{hometown}<br />{age}歳/{weight:.1f}kg</div>
      </td>
      <td rowspan="4">F0<br />L0<br />{average_start}</td>
      <td rowspan="4">{national[0]}<br />{national[1]}<br />{national[2]}</td>
      <td rowspan="4">{local[0]}<br />{local[1]}<br />{local[2]}</td>
      <td rowspan="4">{motor[0]}<br />{motor[1]}<br />{motor[2]}</td>
      <td rowspan="4">{boat[0]}<br />{boat[1]}<br />{boat[2]}</td>
      <td rowspan="4"></td>
      <td>8</td>
      <td>12</td>
      <td rowspan="4"><a href="#">7R</a></td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td></td>
    </tr>
    <tr>
      <td>{recent_starts[0]}</td>
      <td>{recent_starts[1]}</td>
      <td></td>
    </tr>
    <tr>
      <td>{recent_finishes[0]}</td>
      <td>{recent_finishes[1]}</td>
      <td></td>
    </tr>
    """


CARD_HTML = f"""
<html>
  <body>
    <div class="heading2">
      <div class="heading2_head">
        <div class="heading2_area"><img alt="大村" /></div>
        <div class="heading2_title"><h2 class="heading2_titleName">開設７３周年記念 海の王者決定戦</h2></div>
      </div>
    </div>
    <div class="table1">
      <table>
        <tbody>
          <tr><th>レース</th><th>1R</th><th>2R</th><th>3R</th></tr>
          <tr><td>締切予定時刻</td><td>15:05</td><td>15:35</td><td>16:05</td></tr>
        </tbody>
      </table>
    </div>
    <div class="table1">
      <table>
        <tbody>
          {_entrant_block("１", "4320", "A1", "峰　　　竜太", "佐賀", "佐賀", 40, 52.0, "0.14", ("7.71", "47.67", "73.26"), ("8.47", "63.89", "77.78"), ("47", "34.71", "49.41"), ("47", "35.54", "57.83"), (".18", ".11"), ("３", "３"))}
          {_entrant_block("２", "5043", "A1", "中村　　日向", "香川", "香川", 27, 52.1, "0.13", ("6.11", "40.91", "54.55"), ("7.05", "55.00", "70.00"), ("14", "30.54", "42.51"), ("76", "30.12", "46.99"), (".10", ".17"), ("５", "５"))}
          {_entrant_block("３", "3876", "A1", "中辻　　崇人", "福岡", "福岡", 48, 52.0, "0.15", ("7.20", "50.00", "66.00"), ("7.80", "58.00", "72.00"), ("22", "42.00", "55.00"), ("12", "40.00", "60.00"), (".12", ".13"), ("１", "２"))}
        </tbody>
      </table>
    </div>
  </body>
</html>
"""


ODDS3T_HTML = """
<html>
  <body>
    <div class="table1">
      <table>
        <thead>
          <tr>
            <th class="is-boatColor1">1</th><th colspan="2">峰竜太</th>
            <th class="is-boatColor2">2</th><th colspan="2">中村日向</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2">2</td><td>3</td><td class="oddsPoint">21.3</td>
            <td rowspan="2">1</td><td>3</td><td class="oddsPoint">67.8</td>
          </tr>
          <tr>
            <td>4</td><td class="oddsPoint">28.4</td>
            <td>4</td><td class="oddsPoint">211.4</td>
          </tr>
        </tbody>
      </table>
    </div>
  </body>
</html>
"""


def test_parse_race_index_extracts_active_venue():
    entries = parse_race_index(INDEX_HTML, "2026-03-10")

    assert len(entries) == 1
    assert entries[0].venue_code == "24"
    assert entries[0].venue_name == "大村"
    assert entries[0].status == "1R発売開始前"
    assert entries[0].meeting_name == "開設73周年記念 海の王者決定戦"
    assert entries[0].day_label == "5日目"


def test_parse_race_card_extracts_deadline_and_entrants():
    card = parse_race_card(CARD_HTML, "2026-03-10", "24", 2)

    assert card.deadline == "15:35"
    assert card.venue_name == "大村"
    assert len(card.entrants) == 3
    assert card.entrants[0].racer_id == "4320"
    assert card.entrants[0].grade == "A1"
    assert card.entrants[0].name == "峰竜太"
    assert card.entrants[0].branch == "佐賀"
    assert card.entrants[0].age == 40
    assert card.entrants[0].recent_starts == [0.18, 0.11]
    assert card.entrants[0].recent_finishes == [3, 3]


def test_predict_race_prefers_stronger_recent_form_and_rates():
    card = parse_race_card(CARD_HTML, "2026-03-10", "24", 2)
    prediction = predict_race(card, top_k=3)

    assert prediction.entrants[0].lane == 1
    assert prediction.entrants[1].lane == 3
    assert round(sum(entrant.win_probability for entrant in prediction.entrants), 4) == 1.0
    assert prediction.trifectas[0].order[0] == 1


def test_parse_trifecta_odds_expands_rowspans():
    odds = parse_trifecta_odds(ODDS3T_HTML)

    assert odds["1-2-3"] == 21.3
    assert odds["1-2-4"] == 28.4
    assert odds["2-1-3"] == 67.8
    assert odds["2-1-4"] == 211.4


def test_official_client_retries_after_empty_response():
    session = _FakeSession(
        [
            _FakeResponse("<html><body>prime</body></html>"),
            _FakeResponse(""),
            _FakeResponse("<html><body>prime</body></html>"),
            _FakeResponse(INDEX_HTML),
        ]
    )
    client = OfficialBoatraceClient(
        session=session,
        timeout=0.1,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )

    entries = client.fetch_race_index("2026-03-10")

    assert len(entries) == 1
    assert entries[0].venue_code == "24"
    assert len(session.calls) == 4


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200):
        self.text = text
        self.status_code = status_code
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)

    def close(self):
        return None
