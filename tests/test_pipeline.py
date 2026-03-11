import json
from pathlib import Path

import pytest

from boatrace_ai.collect.official import (
    BeforeInfo,
    BeforeInfoEntrant,
    BeforeInfoWeather,
    RaceCard,
    RaceEntrant,
    parse_beforeinfo,
    parse_race_result,
)
from boatrace_ai.betting import (
    compare_bankroll_strategies,
    generate_trifecta_recommendations,
    simulate_bankroll_strategy,
)
from boatrace_ai.calibration import apply_probability_calibration
from boatrace_ai.evaluate.backtest import run_holdout_backtest
from boatrace_ai.features.dataset import build_dataset
from boatrace_ai.predict.model import predict_race_with_model
from boatrace_ai.train.model import (
    _filter_rows_by_complete_groups,
    _load_race_odds_index,
    _select_recent_training_rows,
    load_model_artifact,
    train_win_model,
)


BEFOREINFO_HTML = """
<html>
  <body>
    <div class="heading2">
      <div class="heading2_head">
        <div class="heading2_area"><img alt="大村" /></div>
      </div>
    </div>
    <div class="table1">
      <table><tbody><tr><th>レース</th><th>1R</th></tr><tr><td>締切予定時刻</td><td>15:05</td></tr></tbody></table>
    </div>
    <div class="table1">
      <table>
        <tbody>
          <tr>
            <td rowspan="4">1</td>
            <td rowspan="4"><a href="#"></a></td>
            <td rowspan="4"><a href="#">峰　　　竜太</a></td>
            <td rowspan="2">52.4kg</td>
            <td rowspan="4">6.70</td>
            <td rowspan="4">0.0</td>
            <td rowspan="4"></td>
            <td rowspan="4"><span>リング×２</span></td>
            <td>R</td>
            <td></td>
          </tr>
          <tr><td>進入</td><td></td></tr>
          <tr><td rowspan="2">0.0</td><td>ST</td><td></td></tr>
          <tr><td>着順</td><td></td></tr>
          <tr>
            <td rowspan="4">2</td>
            <td rowspan="4"><a href="#"></a></td>
            <td rowspan="4"><a href="#">中村　　日向</a></td>
            <td rowspan="2">52.1kg</td>
            <td rowspan="4">6.76</td>
            <td rowspan="4">-0.5</td>
            <td rowspan="4"></td>
            <td rowspan="4"></td>
            <td>R</td>
            <td></td>
          </tr>
          <tr><td>進入</td><td></td></tr>
          <tr><td rowspan="2">0.0</td><td>ST</td><td></td></tr>
          <tr><td>着順</td><td></td></tr>
          <tr>
            <td rowspan="4">3</td>
            <td rowspan="4"><a href="#"></a></td>
            <td rowspan="4"><a href="#">中辻　　崇人</a></td>
            <td rowspan="2">52.0kg</td>
            <td rowspan="4">6.72</td>
            <td rowspan="4">0.0</td>
            <td rowspan="4"></td>
            <td rowspan="4"></td>
            <td>R</td>
            <td></td>
          </tr>
          <tr><td>進入</td><td></td></tr>
          <tr><td rowspan="2">0.0</td><td>ST</td><td></td></tr>
          <tr><td>着順</td><td></td></tr>
        </tbody>
      </table>
    </div>
    <div class="table1">
      <table>
        <tr><th colspan="3">スタート展示</th></tr>
        <tr><th>コース</th><th>並び</th><th>ST</th></tr>
        <tr><td colspan="3"><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type1">1</span><span class="table1_boatImage1Time">.11</span></div></td></tr>
        <tr><td colspan="3"><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type2">2</span><span class="table1_boatImage1Time">.14</span></div></td></tr>
        <tr><td colspan="3"><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type3">3</span><span class="table1_boatImage1Time">.09</span></div></td></tr>
      </table>
    </div>
    <div class="weather1">
      <div class="weather1_bodyUnit is-direction">
        <p class="weather1_bodyUnitImage is-direction3"></p>
        <div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">気温</span><span class="weather1_bodyUnitLabelData">9.0℃</span></div>
      </div>
      <div class="weather1_bodyUnit is-weather">
        <div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">晴</span></div>
      </div>
      <div class="weather1_bodyUnit is-wind">
        <div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">風速</span><span class="weather1_bodyUnitLabelData">3m</span></div>
      </div>
      <div class="weather1_bodyUnit is-waterTemperature">
        <div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">水温</span><span class="weather1_bodyUnitLabelData">13.0℃</span></div>
      </div>
      <div class="weather1_bodyUnit is-wave">
        <div class="weather1_bodyUnitLabel"><span class="weather1_bodyUnitLabelTitle">波高</span><span class="weather1_bodyUnitLabelData">2cm</span></div>
      </div>
    </div>
  </body>
</html>
"""


RACERESULT_HTML = """
<html>
  <body>
    <div class="heading2">
      <div class="heading2_head">
        <div class="heading2_area"><img alt="大村" /></div>
      </div>
    </div>
    <div class="table1"><table><tbody><tr><th>レース</th><th>1R</th></tr><tr><td>締切予定時刻</td><td>15:05</td></tr></tbody></table></div>
    <div class="table1">
      <table>
        <thead><tr><th>着</th><th>枠</th><th>ボートレーサー</th><th>レースタイム</th></tr></thead>
        <tbody>
          <tr><td>１</td><td>1</td><td>4320 峰　　　竜太</td><td>1'49"0</td></tr>
          <tr><td>２</td><td>3</td><td>3876 中辻　　崇人</td><td>1'50"2</td></tr>
          <tr><td>３</td><td>2</td><td>5043 中村　　日向</td><td>1'51"0</td></tr>
        </tbody>
      </table>
    </div>
    <div class="table1">
      <table>
        <tr><th>スタート情報</th></tr>
        <tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type1">1</span><span class="table1_boatImage1TimeInner">.09 逃げ</span></div></td></tr>
        <tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type2">2</span><span class="table1_boatImage1TimeInner">.15</span></div></td></tr>
        <tr><td><div class="table1_boatImage1"><span class="table1_boatImage1Number is-type3">3</span><span class="table1_boatImage1TimeInner">.11</span></div></td></tr>
      </table>
    </div>
    <div class="table1">
      <table>
        <thead><tr><th>勝式</th><th>組番</th><th>払戻金</th><th>人気</th></tr></thead>
        <tbody>
          <tr><td rowspan="2">3連単</td><td><div class="numberSet1_row"><span class="numberSet1_number is-type1">1</span><span class="numberSet1_text">-</span><span class="numberSet1_number is-type3">3</span><span class="numberSet1_text">-</span><span class="numberSet1_number is-type2">2</span></div></td><td><span class="is-payout1">¥1,240</span></td><td>4</td></tr>
          <tr><td>&nbsp;</td><td><span class="is-payout1">&nbsp;</span></td><td>&nbsp;</td></tr>
        </tbody>
        <tbody>
          <tr><td rowspan="2">単勝</td><td><div class="numberSet1_row"><span class="numberSet1_number is-type1">1</span></div></td><td><span class="is-payout1">¥170</span></td><td>&nbsp;</td></tr>
          <tr><td>&nbsp;</td><td><span class="is-payout1">&nbsp;</span></td><td>&nbsp;</td></tr>
        </tbody>
      </table>
    </div>
    <div class="table1"><table><tbody><tr><th>返還</th></tr><tr><td></td></tr></tbody></table></div>
    <div class="table1"><table><tbody><tr><th>決まり手</th></tr><tr><td>逃げ</td></tr></tbody></table></div>
    <div class="table1"><table><tbody><tr><th>備考</th></tr><tr><td></td></tr></tbody></table></div>
  </body>
</html>
"""


def test_parse_beforeinfo_extracts_weather_and_display_times():
    beforeinfo = parse_beforeinfo(BEFOREINFO_HTML, "2026-03-09", "24", 12)

    assert beforeinfo.venue_name == "大村"
    assert beforeinfo.weather.weather == "晴"
    assert beforeinfo.weather.temperature_c == 9.0
    assert beforeinfo.entrants[0].display_weight_kg == 52.4
    assert beforeinfo.entrants[0].parts_exchange == ["リング×2"]
    assert beforeinfo.entrants[2].start_display_st == 0.09


def test_parse_race_result_extracts_finish_and_payouts():
    result = parse_race_result(RACERESULT_HTML, "2026-03-09", "24", 12)

    assert result is not None
    assert result.technique == "逃げ"
    assert result.entrants[0].lane == 1
    assert result.entrants[0].racer_id == "4320"
    assert result.start_timings[1] == 0.09
    assert result.payouts[0].bet_type == "3連単"
    assert result.payouts[0].combination == "1-3-2"
    assert result.payouts[1].bet_type == "単勝"
    assert result.payouts[1].payout_yen == 170


def test_build_dataset_and_train_model_end_to_end(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    dates = ["2026-03-08", "2026-03-09", "2026-03-10"]
    race_counter = 1
    for race_date in dates:
        date_dir = raw_dir / race_date.replace("-", "")
        date_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            winner_lane = 1 if race_counter % 2 else 3
            record = _make_raw_record(race_date, race_counter, winner_lane)
            (date_dir / f"24_{race_counter:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            race_counter += 1

    dataset_path = processed_dir / "entrants.csv"
    dataset_summary = build_dataset(raw_dir, dataset_path)
    assert dataset_summary["rows"] == 36

    training_result = train_win_model(
        dataset_path=dataset_path,
        output_dir=models_dir,
        train_end_date="2026-03-09",
        raw_dir=raw_dir,
        random_state=7,
    )
    assert Path(training_result.model_path).exists()
    assert training_result.train_rows == 24
    assert training_result.test_rows == 12
    assert "top1_hit_rate" in training_result.metrics
    assert "recommendation_roi" in training_result.metrics
    assert "walk_forward_recommendation_roi" in training_result.metrics
    assert "calibration_method" in training_result.metrics
    assert "calibration_oof_raw_brier_score" in training_result.metrics
    assert "calibration_oof_brier_score" in training_result.metrics
    assert training_result.metrics["recommendation_roi"] >= 1.2
    assert training_result.metrics["walk_forward_recommendation_roi"] >= 1.2
    assert "min_expected_value" in training_result.betting_policy
    assert "min_probability" in training_result.betting_policy
    assert "min_edge" in training_result.betting_policy

    artifact = load_model_artifact(Path(training_result.model_path))
    assert "betting_policy" in artifact
    assert "payout_model" in artifact
    assert "calibration_summary" in artifact
    prediction = predict_race_with_model(
        card=_make_race_card("2026-03-11"),
        beforeinfo=_make_beforeinfo("2026-03-11"),
        artifact=artifact,
        top_k=3,
    )
    assert len(prediction.entrants) == 3
    assert round(sum(entrant.win_probability for entrant in prediction.entrants), 4) == 1.0
    assert prediction.trifectas[0].order[0] in {1, 3}


def test_select_recent_training_rows_skips_stale_gap():
    selected = _select_recent_training_rows(
        [
            {"date": "2026-03-01", "race_key": "2026-03-01_24_01"},
            {"date": "2026-03-08", "race_key": "2026-03-08_24_01"},
            {"date": "2026-03-09", "race_key": "2026-03-09_24_01"},
        ]
    )

    assert {row["date"] for row in selected} == {"2026-03-08", "2026-03-09"}


def test_generate_trifecta_recommendations_respects_candidate_pool_size():
    recommendations = generate_trifecta_recommendations(
        race_key="2026-03-10_24_12",
        venue_code="24",
        venue_name="大村",
        race_no=12,
        lane_probabilities={1: 0.6, 2: 0.2, 3: 0.1, 4: 0.05, 5: 0.03, 6: 0.02},
        payout_model=None,
        odds_map={
            "1-2-3": 1.0,
            "1-3-2": 30.0,
        },
        policy={
            "min_expected_value": 1.1,
            "max_per_race": 1,
            "candidate_pool_size": 1,
            "min_probability": 0.0,
            "min_edge": 0.0,
        },
    )

    assert recommendations == []


def test_filter_rows_by_complete_groups_drops_partial_day():
    rows = [
        {"date": "2026-03-08", "venue_code": "24.0", "race_key": "2026-03-08_24_01"},
        {"date": "2026-03-09", "venue_code": "24.0", "race_key": "2026-03-09_24_01"},
        {"date": "2026-03-10", "venue_code": "24.0", "race_key": "2026-03-10_24_01"},
    ]

    filtered = _filter_rows_by_complete_groups(
        rows,
        {("2026-03-08", "24"), ("2026-03-09", "24")},
    )

    assert [row["date"] for row in filtered] == ["2026-03-08", "2026-03-09"]


def test_load_race_odds_index_matches_dataset_race_keys(tmp_path: Path):
    raw_dir = tmp_path / "raw" / "20260310"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record = _make_raw_record("2026-03-10", 1, 1)
    (raw_dir / "24_01.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    odds_index = _load_race_odds_index(tmp_path / "raw")

    assert "2026-03-10_24_01" in odds_index
    assert odds_index["2026-03-10_24_01"]["1-3-2"] == 13.0


def test_run_holdout_backtest_returns_bankroll_summaries(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    dates = ["2026-03-08", "2026-03-09", "2026-03-10"]
    race_counter = 1
    for race_date in dates:
        date_dir = raw_dir / race_date.replace("-", "")
        date_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            winner_lane = 1 if race_counter % 2 else 3
            record = _make_raw_record(race_date, race_counter, winner_lane)
            (date_dir / f"24_{race_counter:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            race_counter += 1

    dataset_path = processed_dir / "entrants.csv"
    build_dataset(raw_dir, dataset_path)

    result = run_holdout_backtest(
        dataset_path=dataset_path,
        raw_dir=raw_dir,
        train_end_date="2026-03-09",
        random_state=7,
        bankroll_mode="all",
        starting_bankroll_yen=10_000,
        kelly_fraction=0.25,
        kelly_cap_fraction=0.1,
        max_race_exposure_fraction=0.2,
    )

    assert result.test_rows == 12
    assert result.metrics["recommendation_roi"] >= 1.2
    assert result.bankroll["flat"]["starting_bankroll_yen"] == 10_000
    assert result.bankroll["flat"]["ending_bankroll_yen"] > 0
    assert result.bankroll["kelly"]["ending_bankroll_yen"] > 0
    assert result.bankroll["kelly_capped"]["ending_bankroll_yen"] > 0
    assert result.bankroll["kelly_capped"]["limits"]["kelly_cap_fraction"] == 0.1
    assert result.bankroll["kelly"]["max_drawdown_rate"] >= 0.0
    assert "calibration_method" in result.metrics
    assert "calibration_oof_brier_score" in result.metrics


def test_train_model_raises_when_roi_guard_fails(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"

    dates = ["2026-03-08", "2026-03-09", "2026-03-10"]
    race_counter = 1
    for race_date in dates:
        date_dir = raw_dir / race_date.replace("-", "")
        date_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(4):
            winner_lane = 1 if race_counter % 2 else 3
            record = _make_raw_record(race_date, race_counter, winner_lane)
            (date_dir / f"24_{race_counter:02d}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            race_counter += 1

    dataset_path = processed_dir / "entrants.csv"
    build_dataset(raw_dir, dataset_path)

    with pytest.raises(ValueError, match="Training thresholds failed"):
        train_win_model(
            dataset_path=dataset_path,
            output_dir=models_dir,
            train_end_date="2026-03-09",
            raw_dir=raw_dir,
            random_state=7,
            min_recommendation_roi=20.0,
        )


def test_compare_bankroll_strategies_supports_capped_kelly():
    race_records = [
        {
            "race_key": "2026-03-10_24_01",
            "date": "2026-03-10",
            "venue_code": "24",
            "venue_name": "大村",
            "race_no": 1,
            "lane_probabilities": {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.05, 5: 0.03, 6: 0.02},
            "odds_map": {"1-2-3": 30.0},
            "actual_trifecta_key": "1-2-3",
            "actual_trifecta_payout_yen": 3000,
        }
    ]
    policy = {
        "min_expected_value": 1.0,
        "max_per_race": 1,
        "candidate_pool_size": 1,
        "min_probability": 0.0,
        "min_edge": 0.0,
    }

    summaries = compare_bankroll_strategies(
        race_records,
        payout_model=None,
        policy=policy,
        starting_bankroll_yen=10_000,
        kelly_fraction=1.0,
        kelly_cap_fraction=0.1,
        mode="all",
    )

    assert set(summaries) == {"flat", "kelly", "kelly_capped"}
    kelly_stake = summaries["kelly"]["race_results"][0]["placed_bets"][0]["stake_yen"]
    capped_stake = summaries["kelly_capped"]["race_results"][0]["placed_bets"][0]["stake_yen"]
    assert kelly_stake > capped_stake
    assert capped_stake <= 1000


def test_simulate_bankroll_strategy_honors_daily_stop_loss():
    race_records = [
        {
            "race_key": "2026-03-10_24_01",
            "date": "2026-03-10",
            "venue_code": "24",
            "venue_name": "大村",
            "race_no": 1,
            "lane_probabilities": {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.05, 5: 0.03, 6: 0.02},
            "odds_map": {"1-2-3": 30.0},
            "actual_trifecta_key": "2-1-3",
            "actual_trifecta_payout_yen": 3000,
        },
        {
            "race_key": "2026-03-10_24_02",
            "date": "2026-03-10",
            "venue_code": "24",
            "venue_name": "大村",
            "race_no": 2,
            "lane_probabilities": {1: 0.6, 2: 0.2, 3: 0.1, 4: 0.05, 5: 0.03, 6: 0.02},
            "odds_map": {"1-2-3": 30.0},
            "actual_trifecta_key": "1-2-3",
            "actual_trifecta_payout_yen": 3000,
        },
    ]
    policy = {
        "min_expected_value": 1.0,
        "max_per_race": 1,
        "candidate_pool_size": 1,
        "min_probability": 0.0,
        "min_edge": 0.0,
    }

    summary = simulate_bankroll_strategy(
        race_records,
        payout_model=None,
        policy=policy,
        strategy="flat",
        starting_bankroll_yen=10_000,
        flat_bet_yen=100,
        daily_stop_loss_yen=100,
    )

    assert summary["bets"] == 1
    assert summary["limit_trigger_counts"]["daily_stop_loss"] == 1
    assert summary["race_results"][1]["skip_reason"] == "daily_stop_loss"


def test_apply_probability_calibration_compresses_extremes():
    calibrated = apply_probability_calibration(
        [0.1, 0.5, 0.9],
        {"method": "platt_logit", "coef": 0.5, "intercept": 0.0},
    )

    assert calibrated[0] > 0.1
    assert calibrated[1] == pytest.approx(0.5)
    assert calibrated[2] < 0.9


def _make_race_card(race_date: str) -> RaceCard:
    return RaceCard(
        date=race_date,
        venue_code="24",
        venue_name="大村",
        race_no=12,
        meeting_name="開設73周年記念 海の王者決定戦",
        deadline="20:45",
        entrants=[
            RaceEntrant(1, "4320", "A1", "峰竜太", "佐賀", "佐賀", 40, 52.0, 0, 0, 0.14, 7.7, 47.6, 73.2, 8.4, 63.8, 77.7, 47, 34.7, 49.4, 47, 35.5, 57.8, [0.11, 0.12], [1, 2]),
            RaceEntrant(2, "5043", "A1", "中村日向", "香川", "香川", 27, 52.1, 0, 0, 0.13, 6.1, 40.9, 54.5, 7.0, 55.0, 70.0, 14, 30.5, 42.5, 76, 30.1, 47.0, [0.15, 0.14], [4, 5]),
            RaceEntrant(3, "3876", "A1", "中辻崇人", "福岡", "福岡", 48, 52.0, 0, 0, 0.15, 7.2, 50.0, 66.0, 7.8, 58.0, 72.0, 22, 42.0, 55.0, 12, 40.0, 60.0, [0.09, 0.10], [1, 2]),
        ],
    )


def _make_beforeinfo(race_date: str) -> BeforeInfo:
    return BeforeInfo(
        date=race_date,
        venue_code="24",
        venue_name="大村",
        race_no=12,
        weather=BeforeInfoWeather("晴", 10.0, 3.0, 13.0, 2.0, "is-direction3"),
        entrants=[
            BeforeInfoEntrant(1, "峰竜太", 52.4, 6.71, 0.0, None, ["リング×2"], 0.0, 0.11),
            BeforeInfoEntrant(2, "中村日向", 52.1, 6.76, -0.5, None, [], 0.0, 0.15),
            BeforeInfoEntrant(3, "中辻崇人", 52.0, 6.70, 0.0, None, [], 0.0, 0.09),
        ],
    )


def _make_raw_record(race_date: str, race_no: int, winner_lane: int) -> dict:
    second_lane = 3 if winner_lane == 1 else 1
    third_lane = 2
    card = _make_race_card(race_date).to_dict()
    beforeinfo = _make_beforeinfo(race_date).to_dict()

    adjusted_times = {
        1: 6.71 if winner_lane == 1 else 6.75,
        2: 6.76,
        3: 6.70 if winner_lane == 3 else 6.74,
    }
    for entrant in beforeinfo["entrants"]:
        entrant["exhibition_time"] = adjusted_times[entrant["lane"]]
        entrant["start_display_st"] = 0.09 if entrant["lane"] == winner_lane else 0.15 + (entrant["lane"] * 0.01)

    result = {
        "date": race_date,
        "venue_code": "24",
        "venue_name": "大村",
        "race_no": race_no,
        "technique": "逃げ" if winner_lane == 1 else "まくり",
        "weather": beforeinfo["weather"],
        "entrants": [
            {"finish_label": "1", "finish_position": 1, "lane": winner_lane, "racer_id": card["entrants"][winner_lane - 1]["racer_id"], "name": card["entrants"][winner_lane - 1]["name"], "race_time": "1'49\"0"},
            {"finish_label": "2", "finish_position": 2, "lane": second_lane, "racer_id": card["entrants"][second_lane - 1]["racer_id"], "name": card["entrants"][second_lane - 1]["name"], "race_time": "1'50\"0"},
            {"finish_label": "3", "finish_position": 3, "lane": third_lane, "racer_id": card["entrants"][third_lane - 1]["racer_id"], "name": card["entrants"][third_lane - 1]["name"], "race_time": "1'51\"0"},
        ],
        "start_timings": {
            "1": 0.09 if winner_lane == 1 else 0.16,
            "2": 0.17,
            "3": 0.08 if winner_lane == 3 else 0.14,
        },
        "payouts": [
            {"bet_type": "単勝", "combination": str(winner_lane), "payout_yen": 150 + race_no, "popularity": 1},
            {"bet_type": "3連単", "combination": f"{winner_lane}-{second_lane}-{third_lane}", "payout_yen": 1000 + race_no * 10, "popularity": 3},
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
        "card": card,
        "beforeinfo": beforeinfo,
        "result": result,
        "trifecta_odds": {
            f"{winner_lane}-{second_lane}-{third_lane}": 12.0 + race_no,
            f"{second_lane}-{winner_lane}-{third_lane}": 40.0 + race_no,
        },
    }
