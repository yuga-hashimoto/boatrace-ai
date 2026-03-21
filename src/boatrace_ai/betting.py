"""Bet recommendation helpers focused on expected value."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import itertools
import math
import statistics
from typing import Any


BET_UNIT_YEN = 100
MIN_POLICY_SELECTION_BETS = 2
MIN_POLICY_SELECTION_BETTING_DAYS = 2
DERIVED_POLICY_FILTER_KEYS = (
    "allowed_venues",
    "required_first_lane",
    "required_second_lane",
    "required_third_lane",
    "min_top_win_probability",
    "min_win_margin",
)
DEFAULT_BETTING_POLICY = {
    "min_expected_value": 1.05,
    "max_per_race": 1,
    "candidate_pool_size": 12,
    "min_probability": 0.0,
    "min_edge": 0.0,
    "min_market_odds": 0.0,
    "max_market_odds": None,
}
STRUCTURAL_ROI_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 1.0,
    "max_per_race": 1,
    "candidate_pool_size": 1,
    "min_probability": 0.065,
    "min_edge": 0.0,
    "min_market_odds": 20.0,
    "max_market_odds": 40.0,
    "required_second_lane": 2,
    "required_third_lane": 3,
    "min_win_margin": 0.3,
}
REPEATED_ROI_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 1.0,
    "max_per_race": 1,
    "candidate_pool_size": 1,
    "min_probability": 0.04,
    "min_edge": 0.0,
    "min_market_odds": 30.0,
    "max_market_odds": 50.0,
}
MONTHLY_ROI_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 0.0,
    "max_per_race": 1,
    "candidate_pool_size": 12,
    "min_probability": 0.0,
    "min_edge": 0.0,
    "min_market_odds": 0.0,
    "max_market_odds": None,
    "required_first_lane": 1,
    "required_second_lane": 4,
    "required_third_lane": 6,
}
SINGLE_DAY_ROI_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 1.0,
    "max_per_race": 1,
    "candidate_pool_size": 1,
    "min_probability": 0.25,
    "min_edge": 0.0,
    "min_market_odds": 50.0,
    "max_market_odds": 120.0,
}
ROI_FOCUSED_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 1.0,
    "max_per_race": 1,
    "candidate_pool_size": 12,
    "min_probability": 0.18,
    "min_edge": 0.0,
    "min_market_odds": 70.0,
    "max_market_odds": 90.0,
    "fallback_policy": {
        **SINGLE_DAY_ROI_BETTING_POLICY,
        "min_expected_value": 24.0,
    },
}
LOSS_AVOIDANCE_BETTING_POLICY = {
    **DEFAULT_BETTING_POLICY,
    "min_expected_value": 1.0,
    "max_per_race": 1,
    "candidate_pool_size": 1,
    "min_probability": 0.2,
    "min_edge": 0.0,
    "min_market_odds": 0.0,
    "max_market_odds": 90.0,
}


@dataclass(frozen=True)
class BetRecommendation:
    race_key: str
    stadium: int
    stadium_name: str
    race_number: int
    bet_type: str
    combination: str
    probability_ratio: float
    probability: float
    expected_value: float
    avg_payout: int
    market_odds: float
    implied_probability: float
    edge: float
    recommended_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def merge_betting_policy(
    base_policy: dict[str, Any] | None,
    override_policy: dict[str, Any] | None,
    *,
    clear_derived_filters: bool = False,
) -> dict[str, Any]:
    resolved = dict(base_policy or {})
    override = dict(override_policy or {})

    if override and "allowed_venues" not in override:
        resolved.pop("allowed_venues", None)

    if clear_derived_filters:
        for key in DERIVED_POLICY_FILTER_KEYS:
            if key not in override:
                resolved.pop(key, None)

    resolved.update(override)
    return resolved


def normalize_probabilities(probabilities: list[float]) -> list[float]:
    observed = [max(float(probability), 1e-6) for probability in probabilities]
    total = sum(observed)
    if total <= 0:
        return [1.0 / len(observed) for _ in observed] if observed else []
    return [probability / total for probability in observed]


def trifecta_probabilities(lane_probabilities: dict[int, float]) -> list[tuple[tuple[int, int, int], float]]:
    lanes = sorted(lane_probabilities)
    probabilities = normalize_probabilities([lane_probabilities[lane] for lane in lanes])
    total = sum(probabilities)
    candidates: list[tuple[tuple[int, int, int], float]] = []

    for first_index, second_index, third_index in itertools.permutations(range(len(lanes)), 3):
        remaining_after_first = total - probabilities[first_index]
        remaining_after_second = remaining_after_first - probabilities[second_index]
        if remaining_after_first <= 0 or remaining_after_second <= 0:
            continue
        probability = (
            probabilities[first_index] / total
            * probabilities[second_index] / remaining_after_first
            * probabilities[third_index] / remaining_after_second
        )
        candidates.append(
            (
                (lanes[first_index], lanes[second_index], lanes[third_index]),
                probability,
            )
        )

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def build_payout_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    race_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        payout = _parse_int(row.get("trifecta_payout_yen"))
        combination = str(row.get("trifecta_key") or "").strip()
        if payout <= 0 or not combination:
            continue
        race_key = str(row.get("race_key") or "")
        if race_key in race_rows:
            continue
        venue_code = _normalize_venue_code(row.get("venue_code"))
        race_rows[race_key] = {
            "venue_code": venue_code,
            "combination": combination,
            "payout_yen": payout,
        }

    global_sum = 0
    global_count = 0
    combo_stats: dict[str, dict[str, float]] = {}
    venue_stats: dict[str, dict[str, float]] = {}
    venue_combo_stats: dict[str, dict[str, dict[str, float]]] = {}

    for race in race_rows.values():
        payout_yen = race["payout_yen"]
        venue_code = race["venue_code"]
        combination = race["combination"]

        global_sum += payout_yen
        global_count += 1

        _update_running_stats(combo_stats, combination, payout_yen)
        _update_running_stats(venue_stats, venue_code, payout_yen)
        venue_combo_stats.setdefault(venue_code, {})
        _update_running_stats(venue_combo_stats[venue_code], combination, payout_yen)

    return {
        "global_average": round(global_sum / global_count) if global_count else 0,
        "combo_stats": combo_stats,
        "venue_stats": venue_stats,
        "venue_combo_stats": venue_combo_stats,
    }


def estimate_trifecta_payout(
    payout_model: dict[str, Any] | None,
    venue_code: str | int,
    combination: str,
) -> int:
    if not payout_model:
        return 0

    normalized_venue = _normalize_venue_code(venue_code)
    global_average = float(payout_model.get("global_average") or 0)
    combo_stats = (payout_model.get("combo_stats") or {}).get(combination, {})
    venue_stats = (payout_model.get("venue_stats") or {}).get(normalized_venue, {})
    venue_combo_stats = (
        (payout_model.get("venue_combo_stats") or {}).get(normalized_venue, {}).get(combination, {})
    )

    estimate = global_average if global_average > 0 else 4500.0
    weight = 16.0
    estimate, weight = _blend_stats(estimate, weight, venue_stats, 10.0)
    estimate, weight = _blend_stats(estimate, weight, combo_stats, 16.0)
    estimate, weight = _blend_stats(estimate, weight, venue_combo_stats, 28.0)
    return max(BET_UNIT_YEN, int(round(estimate)))


def generate_trifecta_recommendations(
    *,
    race_key: str,
    venue_code: str | int,
    venue_name: str,
    race_no: int,
    lane_probabilities: dict[int, float],
    trifecta_probability_map: dict[str, float] | None = None,
    payout_model: dict[str, Any] | None,
    odds_map: dict[str, float] | None = None,
    policy: dict[str, Any] | None = None,
) -> list[BetRecommendation]:
    resolved_policy = dict(DEFAULT_BETTING_POLICY)
    if policy:
        resolved_policy.update(policy)

    fallback_policy = resolved_policy.pop("fallback_policy", None)
    active_policy = resolved_policy
    candidates = _generate_trifecta_candidates(
        race_key=race_key,
        venue_code=venue_code,
        venue_name=venue_name,
        race_no=race_no,
        lane_probabilities=lane_probabilities,
        trifecta_probability_map=trifecta_probability_map,
        payout_model=payout_model,
        odds_map=odds_map,
        policy=resolved_policy,
    )
    if not candidates and fallback_policy:
        fallback_resolved_policy = dict(DEFAULT_BETTING_POLICY)
        fallback_resolved_policy.update(fallback_policy)
        fallback_resolved_policy.pop("fallback_policy", None)
        active_policy = fallback_resolved_policy
        candidates = _generate_trifecta_candidates(
            race_key=race_key,
            venue_code=venue_code,
            venue_name=venue_name,
            race_no=race_no,
            lane_probabilities=lane_probabilities,
            trifecta_probability_map=trifecta_probability_map,
            payout_model=payout_model,
            odds_map=odds_map,
            policy=fallback_resolved_policy,
        )

    max_per_race = max(1, int(active_policy.get("max_per_race", 1)))
    trimmed = candidates[:max_per_race]
    return [
        BetRecommendation(
            race_key=item.race_key,
            stadium=item.stadium,
            stadium_name=item.stadium_name,
            race_number=item.race_number,
            bet_type=item.bet_type,
            combination=item.combination,
            probability_ratio=item.probability_ratio,
            probability=item.probability,
            expected_value=item.expected_value,
            avg_payout=item.avg_payout,
            market_odds=item.market_odds,
            implied_probability=item.implied_probability,
            edge=item.edge,
            recommended_rank=index,
        )
        for index, item in enumerate(trimmed, start=1)
    ]


def _generate_trifecta_candidates(
    *,
    race_key: str,
    venue_code: str | int,
    venue_name: str,
    race_no: int,
    lane_probabilities: dict[int, float],
    trifecta_probability_map: dict[str, float] | None,
    payout_model: dict[str, Any] | None,
    odds_map: dict[str, float] | None,
    policy: dict[str, Any],
) -> list[BetRecommendation]:

    min_expected_value = float(policy.get("min_expected_value", 0.0))
    candidate_pool_size = max(1, int(policy.get("candidate_pool_size", 12)))
    min_probability = float(policy.get("min_probability", 0.0))
    min_edge = float(policy.get("min_edge", 0.0))
    min_market_odds = float(policy.get("min_market_odds", 0.0))
    max_market_odds_value = policy.get("max_market_odds")
    max_market_odds = (
        None
        if max_market_odds_value in (None, "")
        else float(max_market_odds_value)
    )
    min_top_win_probability = float(policy.get("min_top_win_probability", 0.0))
    min_win_margin = float(policy.get("min_win_margin", 0.0))
    required_first_lane = _parse_required_lane(policy.get("required_first_lane"))
    required_second_lane = _parse_required_lane(policy.get("required_second_lane"))
    required_third_lane = _parse_required_lane(policy.get("required_third_lane"))

    venue_int = int(_normalize_venue_code(venue_code))
    allowed_venues = {
        _normalize_venue_code(value)
        for value in (policy.get("allowed_venues") or [])
    }
    if allowed_venues and _normalize_venue_code(venue_code) not in allowed_venues:
        return []
    normalized_lanes = normalize_probabilities(
        [lane_probabilities[lane] for lane in sorted(lane_probabilities)]
    )
    normalized_lane_probabilities = {
        lane: probability
        for lane, probability in zip(sorted(lane_probabilities), normalized_lanes, strict=True)
    }
    sorted_lanes = sorted(
        normalized_lane_probabilities.items(),
        key=lambda item: (-item[1], item[0]),
    )
    top_win_probability = float(sorted_lanes[0][1]) if sorted_lanes else 0.0
    second_win_probability = float(sorted_lanes[1][1]) if len(sorted_lanes) > 1 else 0.0
    win_margin = top_win_probability - second_win_probability
    if top_win_probability < min_top_win_probability or win_margin < min_win_margin:
        return []
    candidates: list[BetRecommendation] = []

    candidate_probabilities = (
        _ranked_probability_map(trifecta_probability_map)
        if trifecta_probability_map
        else [
            ("-".join(str(lane) for lane in order), probability_ratio)
            for order, probability_ratio in trifecta_probabilities(lane_probabilities)
        ]
    )
    for combination, probability_ratio in candidate_probabilities[:candidate_pool_size]:
        order = tuple(int(part) for part in combination.split("-"))
        if not _matches_required_order(
            order,
            required_first_lane=required_first_lane,
            required_second_lane=required_second_lane,
            required_third_lane=required_third_lane,
        ):
            continue
        combination = "-".join(str(lane) for lane in order)
        live_odds = (odds_map or {}).get(combination)
        if live_odds is not None and live_odds > 0:
            avg_payout = max(BET_UNIT_YEN, int(round(live_odds * BET_UNIT_YEN)))
        else:
            avg_payout = estimate_trifecta_payout(payout_model, venue_code, combination)
        market_odds = avg_payout / BET_UNIT_YEN
        implied_probability = (BET_UNIT_YEN / avg_payout) if avg_payout > 0 else 0.0
        edge = probability_ratio - implied_probability
        expected_value = probability_ratio * avg_payout / BET_UNIT_YEN
        if (
            expected_value < min_expected_value
            or probability_ratio < min_probability
            or edge < min_edge
            or market_odds < min_market_odds
            or (max_market_odds is not None and market_odds > max_market_odds)
        ):
            continue
        candidates.append(
            BetRecommendation(
                race_key=race_key,
                stadium=venue_int,
                stadium_name=venue_name,
                race_number=int(race_no),
                bet_type="3連単",
                combination=combination,
                probability_ratio=round(probability_ratio, 6),
                probability=round(probability_ratio * 100, 2),
                expected_value=round(expected_value, 4),
                avg_payout=int(avg_payout),
                market_odds=round(market_odds, 3),
                implied_probability=round(implied_probability, 6),
                edge=round(edge, 6),
                recommended_rank=0,
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.expected_value,
            -item.edge,
            -item.probability_ratio,
            item.combination,
        )
    )
    return candidates


def evaluate_recommendation_strategy(
    race_records: list[dict[str, Any]],
    payout_model: dict[str, Any] | None,
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    recommendations: list[BetRecommendation] = []
    total_bets = 0
    hits = 0
    total_return = 0
    daily_store: dict[str, dict[str, int]] = {}

    for race in race_records:
        race_date = str(race.get("date") or "")
        race_recommendations = generate_trifecta_recommendations(
            race_key=race["race_key"],
            venue_code=race["venue_code"],
            venue_name=race["venue_name"],
            race_no=int(race["race_no"]),
            lane_probabilities=race["lane_probabilities"],
            trifecta_probability_map=race.get("trifecta_probability_map"),
            payout_model=payout_model,
            odds_map=race.get("odds_map"),
            policy=policy,
        )
        recommendations.extend(race_recommendations)
        total_bets += len(race_recommendations)
        if race_date:
            entry = daily_store.setdefault(race_date, {"bets": 0, "hits": 0, "return": 0})
            entry["bets"] += len(race_recommendations)

        actual_combination = str(race.get("actual_trifecta_key") or "").strip()
        actual_payout = _parse_int(race.get("actual_trifecta_payout_yen"))
        for recommendation in race_recommendations:
            if recommendation.combination == actual_combination:
                hits += 1
                total_return += actual_payout
                if race_date:
                    entry = daily_store.setdefault(race_date, {"bets": 0, "hits": 0, "return": 0})
                    entry["hits"] += 1
                    entry["return"] += actual_payout

    investment = total_bets * BET_UNIT_YEN
    roi = ((total_return - investment) / investment) if investment else None
    hit_rate = (hits / total_bets) if total_bets else None
    daily_results = _summarize_daily_results(daily_store)
    daily_rois = [item["roi"] for item in daily_results if item["roi"] is not None]

    return {
        "bets": total_bets,
        "hits": hits,
        "hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
        "investment": investment,
        "return": total_return,
        "profit": total_return - investment,
        "roi": round(roi, 6) if roi is not None else None,
        "days": len({str(race.get("date") or "") for race in race_records if race.get("date")}),
        "betting_days": len(daily_rois),
        "positive_days": sum(1 for value in daily_rois if value > 0),
        "daily_roi_median": round(statistics.median(daily_rois), 6) if daily_rois else None,
        "daily_roi_floor": round(min(daily_rois), 6) if daily_rois else None,
        "daily_results": daily_results,
        "recommendations": [recommendation.to_dict() for recommendation in recommendations],
    }


def simulate_bankroll_strategy(
    race_records: list[dict[str, Any]],
    payout_model: dict[str, Any] | None,
    policy: dict[str, Any] | None,
    *,
    strategy: str,
    starting_bankroll_yen: int = 10_000,
    flat_bet_yen: int = BET_UNIT_YEN,
    kelly_fraction: float = 0.25,
    kelly_cap_fraction: float | None = None,
    max_bet_yen: int | None = None,
    max_bet_bankroll_fraction: float | None = None,
    max_daily_bets: int | None = None,
    max_daily_investment_yen: int | None = None,
    max_race_exposure_fraction: float | None = None,
    daily_stop_loss_yen: int | None = None,
    daily_take_profit_yen: int | None = None,
) -> dict[str, Any]:
    if strategy not in {"flat", "kelly", "kelly_capped"}:
        raise ValueError(f"Unsupported bankroll strategy: {strategy}")
    if starting_bankroll_yen <= 0:
        raise ValueError("starting_bankroll_yen must be positive.")
    _validate_fraction_limit("kelly_fraction", kelly_fraction)
    _validate_fraction_limit("kelly_cap_fraction", kelly_cap_fraction)
    _validate_fraction_limit("max_bet_bankroll_fraction", max_bet_bankroll_fraction)
    _validate_fraction_limit("max_race_exposure_fraction", max_race_exposure_fraction)

    bankroll = int(starting_bankroll_yen)
    peak_bankroll = bankroll
    max_drawdown_rate = 0.0
    max_drawdown_yen = 0
    total_bets = 0
    hits = 0
    total_investment = 0
    total_return = 0
    daily_store: dict[str, dict[str, int]] = {}
    daily_bets_by_date: dict[str, int] = defaultdict(int)
    daily_investment_by_date: dict[str, int] = defaultdict(int)
    daily_return_by_date: dict[str, int] = defaultdict(int)
    limit_trigger_counts: dict[str, int] = defaultdict(int)
    race_results: list[dict[str, Any]] = []

    sorted_races = sorted(
        race_records,
        key=lambda item: (
            str(item.get("date") or ""),
            str(item.get("race_key") or ""),
        ),
    )

    for race in sorted_races:
        race_date = str(race.get("date") or "")
        race_bankroll_start = bankroll
        daily_profit_so_far = daily_return_by_date[race_date] - daily_investment_by_date[race_date]
        if daily_stop_loss_yen is not None and daily_profit_so_far <= -int(daily_stop_loss_yen):
            limit_trigger_counts["daily_stop_loss"] += 1
            race_results.append(
                {
                    "race_key": race["race_key"],
                    "date": race_date,
                    "race_no": int(race["race_no"]),
                    "bets": 0,
                    "investment": 0,
                    "return": 0,
                    "hits": 0,
                    "starting_bankroll_yen": race_bankroll_start,
                    "ending_bankroll_yen": bankroll,
                    "placed_bets": [],
                    "skip_reason": "daily_stop_loss",
                }
            )
            continue
        if daily_take_profit_yen is not None and daily_profit_so_far >= int(daily_take_profit_yen):
            limit_trigger_counts["daily_take_profit"] += 1
            race_results.append(
                {
                    "race_key": race["race_key"],
                    "date": race_date,
                    "race_no": int(race["race_no"]),
                    "bets": 0,
                    "investment": 0,
                    "return": 0,
                    "hits": 0,
                    "starting_bankroll_yen": race_bankroll_start,
                    "ending_bankroll_yen": bankroll,
                    "placed_bets": [],
                    "skip_reason": "daily_take_profit",
                }
            )
            continue

        race_recommendations = generate_trifecta_recommendations(
            race_key=race["race_key"],
            venue_code=race["venue_code"],
            venue_name=race["venue_name"],
            race_no=int(race["race_no"]),
            lane_probabilities=race["lane_probabilities"],
            trifecta_probability_map=race.get("trifecta_probability_map"),
            payout_model=payout_model,
            odds_map=race.get("odds_map"),
            policy=policy,
        )

        placed_bets: list[tuple[BetRecommendation, int]] = []
        reserved_investment = 0
        race_exposure_cap = None
        if max_race_exposure_fraction is not None:
            race_exposure_cap = _round_bet_size(race_bankroll_start * max_race_exposure_fraction)
        for recommendation in race_recommendations:
            if max_daily_bets is not None and daily_bets_by_date[race_date] >= max_daily_bets:
                break

            stake = _resolve_bankroll_stake(
                recommendation=recommendation,
                bankroll_yen=race_bankroll_start,
                strategy=strategy,
                flat_bet_yen=flat_bet_yen,
                kelly_fraction=kelly_fraction,
                kelly_cap_fraction=kelly_cap_fraction,
                max_bet_yen=max_bet_yen,
                max_bet_bankroll_fraction=max_bet_bankroll_fraction,
            )
            if max_daily_investment_yen is not None:
                remaining_daily = max_daily_investment_yen - daily_investment_by_date[race_date]
                stake = min(stake, max(0, remaining_daily))
            if race_exposure_cap is not None:
                remaining_race_exposure = race_exposure_cap - reserved_investment
                stake = min(stake, max(0, remaining_race_exposure))
            available_bankroll = race_bankroll_start - reserved_investment
            stake = min(stake, max(0, available_bankroll))
            stake = _round_bet_size(stake)
            if stake < BET_UNIT_YEN:
                continue

            placed_bets.append((recommendation, stake))
            reserved_investment += stake
            daily_bets_by_date[race_date] += 1
            daily_investment_by_date[race_date] += stake

        race_investment = sum(stake for _, stake in placed_bets)
        bankroll -= race_investment

        actual_combination = str(race.get("actual_trifecta_key") or "").strip()
        actual_payout = _parse_int(race.get("actual_trifecta_payout_yen"))
        race_return = 0
        race_hits = 0
        for recommendation, stake in placed_bets:
            if recommendation.combination != actual_combination or actual_payout <= 0:
                continue
            race_hits += 1
            race_return += int(round(actual_payout * (stake / BET_UNIT_YEN)))

        bankroll += race_return
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown_yen = peak_bankroll - bankroll
        drawdown_rate = (drawdown_yen / peak_bankroll) if peak_bankroll else 0.0
        max_drawdown_yen = max(max_drawdown_yen, drawdown_yen)
        max_drawdown_rate = max(max_drawdown_rate, drawdown_rate)

        total_bets += len(placed_bets)
        hits += race_hits
        total_investment += race_investment
        total_return += race_return
        daily_return_by_date[race_date] += race_return

        if race_date:
            entry = daily_store.setdefault(
                race_date,
                {
                    "bets": 0,
                    "hits": 0,
                    "investment": 0,
                    "return": 0,
                    "ending_bankroll_yen": bankroll,
                },
            )
            entry["bets"] += len(placed_bets)
            entry["hits"] += race_hits
            entry["investment"] += race_investment
            entry["return"] += race_return
            entry["ending_bankroll_yen"] = bankroll

        race_results.append(
            {
                "race_key": race["race_key"],
                "date": race_date,
                "race_no": int(race["race_no"]),
                "bets": len(placed_bets),
                "investment": race_investment,
                "return": race_return,
                "hits": race_hits,
                "starting_bankroll_yen": race_bankroll_start,
                "ending_bankroll_yen": bankroll,
                "placed_bets": [
                    {
                        "combination": recommendation.combination,
                        "stake_yen": stake,
                        "probability_ratio": recommendation.probability_ratio,
                        "market_odds": recommendation.market_odds,
                        "expected_value": recommendation.expected_value,
                    }
                    for recommendation, stake in placed_bets
                ],
                "skip_reason": None,
            }
        )

    roi = ((total_return - total_investment) / total_investment) if total_investment else None
    hit_rate = (hits / total_bets) if total_bets else None
    daily_results = _summarize_bankroll_daily_results(daily_store)

    return {
        "strategy": strategy,
        "starting_bankroll_yen": starting_bankroll_yen,
        "ending_bankroll_yen": bankroll,
        "bankroll_multiple": round(bankroll / starting_bankroll_yen, 6),
        "bets": total_bets,
        "hits": hits,
        "hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
        "investment": total_investment,
        "return": total_return,
        "profit": total_return - total_investment,
        "roi": round(roi, 6) if roi is not None else None,
        "max_drawdown_yen": max_drawdown_yen,
        "max_drawdown_rate": round(max_drawdown_rate, 6),
        "limits": {
            "kelly_fraction": kelly_fraction,
            "kelly_cap_fraction": kelly_cap_fraction,
            "max_bet_yen": max_bet_yen,
            "max_bet_bankroll_fraction": max_bet_bankroll_fraction,
            "max_daily_bets": max_daily_bets,
            "max_daily_investment_yen": max_daily_investment_yen,
            "max_race_exposure_fraction": max_race_exposure_fraction,
            "daily_stop_loss_yen": daily_stop_loss_yen,
            "daily_take_profit_yen": daily_take_profit_yen,
        },
        "limit_trigger_counts": dict(limit_trigger_counts),
        "daily_results": daily_results,
        "race_results": race_results,
    }


def compare_bankroll_strategies(
    race_records: list[dict[str, Any]],
    payout_model: dict[str, Any] | None,
    policy: dict[str, Any] | None,
    *,
    starting_bankroll_yen: int = 10_000,
    flat_bet_yen: int = BET_UNIT_YEN,
    kelly_fraction: float = 0.25,
    kelly_cap_fraction: float | None = 0.1,
    max_bet_yen: int | None = None,
    max_bet_bankroll_fraction: float | None = None,
    max_daily_bets: int | None = None,
    max_daily_investment_yen: int | None = None,
    max_race_exposure_fraction: float | None = None,
    daily_stop_loss_yen: int | None = None,
    daily_take_profit_yen: int | None = None,
    mode: str = "both",
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    if mode in {"flat", "both", "all"}:
        summaries["flat"] = simulate_bankroll_strategy(
            race_records,
            payout_model,
            policy,
            strategy="flat",
            starting_bankroll_yen=starting_bankroll_yen,
            flat_bet_yen=flat_bet_yen,
            kelly_fraction=kelly_fraction,
            kelly_cap_fraction=kelly_cap_fraction,
            max_bet_yen=max_bet_yen,
            max_bet_bankroll_fraction=max_bet_bankroll_fraction,
            max_daily_bets=max_daily_bets,
            max_daily_investment_yen=max_daily_investment_yen,
            max_race_exposure_fraction=max_race_exposure_fraction,
            daily_stop_loss_yen=daily_stop_loss_yen,
            daily_take_profit_yen=daily_take_profit_yen,
        )
    if mode in {"kelly", "both", "all"}:
        summaries["kelly"] = simulate_bankroll_strategy(
            race_records,
            payout_model,
            policy,
            strategy="kelly",
            starting_bankroll_yen=starting_bankroll_yen,
            flat_bet_yen=flat_bet_yen,
            kelly_fraction=kelly_fraction,
            kelly_cap_fraction=kelly_cap_fraction,
            max_bet_yen=max_bet_yen,
            max_bet_bankroll_fraction=max_bet_bankroll_fraction,
            max_daily_bets=max_daily_bets,
            max_daily_investment_yen=max_daily_investment_yen,
            max_race_exposure_fraction=max_race_exposure_fraction,
            daily_stop_loss_yen=daily_stop_loss_yen,
            daily_take_profit_yen=daily_take_profit_yen,
        )
    if mode in {"kelly_capped", "all"}:
        summaries["kelly_capped"] = simulate_bankroll_strategy(
            race_records,
            payout_model,
            policy,
            strategy="kelly_capped",
            starting_bankroll_yen=starting_bankroll_yen,
            flat_bet_yen=flat_bet_yen,
            kelly_fraction=kelly_fraction,
            kelly_cap_fraction=kelly_cap_fraction,
            max_bet_yen=max_bet_yen,
            max_bet_bankroll_fraction=max_bet_bankroll_fraction,
            max_daily_bets=max_daily_bets,
            max_daily_investment_yen=max_daily_investment_yen,
            max_race_exposure_fraction=max_race_exposure_fraction,
            daily_stop_loss_yen=daily_stop_loss_yen,
            daily_take_profit_yen=daily_take_profit_yen,
        )
    return summaries


def iter_betting_policies() -> list[dict[str, Any]]:
    thresholds = [1.0, 1.05, 1.15]
    max_per_race_values = [1]
    candidate_pool_sizes = [1, 3, 6, 12]
    min_probability_values = [0.08, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.35]
    min_edge_values = [0.0, 0.02]
    min_market_odds_values = [0.0, 50.0, 70.0]
    max_market_odds_values = [None, 90.0, 120.0, 200.0]

    policies = [
        {
            "min_expected_value": threshold,
            "max_per_race": max_per_race,
            "candidate_pool_size": candidate_pool_size,
            "min_probability": min_probability,
            "min_edge": min_edge,
            "min_market_odds": min_market_odds,
            "max_market_odds": max_market_odds,
        }
        for threshold in thresholds
        for max_per_race in max_per_race_values
        for candidate_pool_size in candidate_pool_sizes
        for min_probability in min_probability_values
        for min_edge in min_edge_values
        for min_market_odds in min_market_odds_values
        for max_market_odds in max_market_odds_values
        if max_market_odds is None or max_market_odds >= min_market_odds
    ]
    policies.extend(
        [
            dict(MONTHLY_ROI_BETTING_POLICY),
            dict(REPEATED_ROI_BETTING_POLICY),
            {
                **REPEATED_ROI_BETTING_POLICY,
                "min_expected_value": 1.05,
            },
            dict(STRUCTURAL_ROI_BETTING_POLICY),
            {
                **STRUCTURAL_ROI_BETTING_POLICY,
                "min_probability": 0.04,
            },
            {
                **STRUCTURAL_ROI_BETTING_POLICY,
                "max_market_odds": 60.0,
            },
            {
                **STRUCTURAL_ROI_BETTING_POLICY,
                "min_probability": 0.04,
                "max_market_odds": 60.0,
            },
            {
                **STRUCTURAL_ROI_BETTING_POLICY,
                "min_expected_value": 1.2,
            },
        ]
    )
    return policies


def score_betting_summary(summary: dict[str, Any]) -> tuple[float, ...]:
    roi = float(summary["roi"]) if summary.get("roi") is not None else -math.inf
    roi_floor = (
        float(summary["daily_roi_floor"])
        if summary.get("daily_roi_floor") is not None
        else roi
    )
    roi_median = (
        float(summary["daily_roi_median"])
        if summary.get("daily_roi_median") is not None
        else roi
    )
    betting_days = max(1, int(summary.get("betting_days") or 0))
    days = max(1, int(summary.get("days") or betting_days))
    coverage = betting_days / days
    hit_rate = float(summary["hit_rate"]) if summary.get("hit_rate") is not None else 0.0
    profit = float(summary.get("profit") or 0.0)
    bets = float(summary.get("bets") or 0.0)
    return (
        roi_floor,
        coverage,
        roi_median,
        roi,
        hit_rate,
        profit,
        -bets,
    )


def policy_summary_is_active_enough(
    summary: dict[str, Any],
    *,
    min_bets: int = MIN_POLICY_SELECTION_BETS,
    min_betting_days: int = MIN_POLICY_SELECTION_BETTING_DAYS,
) -> bool:
    bets = int(summary.get("bets") or 0)
    if bets < max(1, int(min_bets)):
        return False

    days = max(1, int(summary.get("days") or summary.get("betting_days") or 0))
    required_days = min(max(1, int(min_betting_days)), days)
    betting_days = int(summary.get("betting_days") or 0)
    return betting_days >= required_days


def _ranked_probability_map(probability_map: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(
        (
            (str(combination), float(probability))
            for combination, probability in probability_map.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )


def select_betting_policy(
    validation_races: list[dict[str, Any]],
    payout_model: dict[str, Any] | None,
) -> dict[str, Any]:
    if not validation_races:
        return dict(DEFAULT_BETTING_POLICY)

    candidates: list[tuple[dict[str, Any], dict[str, Any], tuple[float, ...]]] = []

    for policy in iter_betting_policies():
        summary = evaluate_recommendation_strategy(validation_races, payout_model, policy)
        if summary["bets"] == 0 or summary["roi"] is None:
            continue
        candidates.append((policy, summary, score_betting_summary(summary)))

    if not candidates:
        return dict(LOSS_AVOIDANCE_BETTING_POLICY)

    eligible_candidates = [
        candidate
        for candidate in candidates
        if policy_summary_is_active_enough(candidate[1])
    ] or candidates

    best_policy, best_summary, _ = max(
        eligible_candidates,
        key=lambda item: item[2],
    )
    if float(best_summary.get("roi") or 0.0) <= 0.0:
        return dict(LOSS_AVOIDANCE_BETTING_POLICY)
    return dict(best_policy)


def _summarize_daily_results(daily_store: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for race_date in sorted(daily_store):
        entry = daily_store[race_date]
        bets = int(entry.get("bets", 0))
        investment = bets * BET_UNIT_YEN
        total_return = int(entry.get("return", 0))
        hits = int(entry.get("hits", 0))
        roi = ((total_return - investment) / investment) if investment else None
        hit_rate = (hits / bets) if bets else None
        results.append(
            {
                "date": race_date,
                "bets": bets,
                "hits": hits,
                "investment": investment,
                "return": total_return,
                "profit": total_return - investment,
                "hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
                "roi": round(roi, 6) if roi is not None else None,
            }
        )
    return results


def _summarize_bankroll_daily_results(
    daily_store: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for race_date in sorted(daily_store):
        entry = daily_store[race_date]
        investment = int(entry.get("investment", 0))
        total_return = int(entry.get("return", 0))
        hits = int(entry.get("hits", 0))
        bets = int(entry.get("bets", 0))
        roi = ((total_return - investment) / investment) if investment else None
        hit_rate = (hits / bets) if bets else None
        results.append(
            {
                "date": race_date,
                "bets": bets,
                "hits": hits,
                "investment": investment,
                "return": total_return,
                "profit": total_return - investment,
                "hit_rate": round(hit_rate, 6) if hit_rate is not None else None,
                "roi": round(roi, 6) if roi is not None else None,
                "ending_bankroll_yen": int(entry.get("ending_bankroll_yen", 0)),
            }
        )
    return results


def _matches_required_order(
    order: tuple[int, int, int],
    *,
    required_first_lane: int | None,
    required_second_lane: int | None,
    required_third_lane: int | None,
) -> bool:
    first_lane, second_lane, third_lane = order
    if required_first_lane is not None and first_lane != required_first_lane:
        return False
    if required_second_lane is not None and second_lane != required_second_lane:
        return False
    if required_third_lane is not None and third_lane != required_third_lane:
        return False
    return True


def _resolve_bankroll_stake(
    *,
    recommendation: BetRecommendation,
    bankroll_yen: int,
    strategy: str,
    flat_bet_yen: int,
    kelly_fraction: float,
    kelly_cap_fraction: float | None,
    max_bet_yen: int | None,
    max_bet_bankroll_fraction: float | None,
) -> int:
    if bankroll_yen < BET_UNIT_YEN:
        return 0

    if strategy == "flat":
        stake = flat_bet_yen
    else:
        stake = _kelly_stake_yen(
            probability_ratio=recommendation.probability_ratio,
            market_odds=recommendation.market_odds,
            bankroll_yen=bankroll_yen,
            kelly_fraction=kelly_fraction,
            kelly_cap_fraction=kelly_cap_fraction if strategy == "kelly_capped" else None,
        )

    if max_bet_bankroll_fraction is not None:
        stake = min(stake, bankroll_yen * max_bet_bankroll_fraction)
    if max_bet_yen is not None:
        stake = min(stake, max_bet_yen)
    return _round_bet_size(stake)


def _kelly_stake_yen(
    *,
    probability_ratio: float,
    market_odds: float,
    bankroll_yen: int,
    kelly_fraction: float,
    kelly_cap_fraction: float | None = None,
) -> int:
    if bankroll_yen <= 0 or kelly_fraction <= 0:
        return 0
    net_odds = market_odds - 1.0
    if net_odds <= 0:
        return 0

    full_kelly_fraction = ((market_odds * probability_ratio) - 1.0) / net_odds
    clipped_fraction = max(0.0, full_kelly_fraction) * kelly_fraction
    if kelly_cap_fraction is not None:
        clipped_fraction = min(clipped_fraction, kelly_cap_fraction)
    return int(bankroll_yen * clipped_fraction)


def _round_bet_size(value: int | float) -> int:
    if value <= 0:
        return 0
    return int(math.floor(float(value) / BET_UNIT_YEN) * BET_UNIT_YEN)


def _parse_required_lane(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        lane = int(value)
    except (TypeError, ValueError):
        return None
    return lane if 1 <= lane <= 6 else None


def _blend_stats(
    current_average: float,
    current_weight: float,
    stats: dict[str, float],
    max_weight: float,
) -> tuple[float, float]:
    count = float(stats.get("count", 0))
    average = float(stats.get("average", 0))
    if count <= 0 or average <= 0:
        return current_average, current_weight

    applied_weight = min(count, max_weight)
    blended = (current_average * current_weight + average * applied_weight) / (
        current_weight + applied_weight
    )
    return blended, current_weight + applied_weight


def _update_running_stats(store: dict[str, dict[str, float]], key: str, value: int) -> None:
    entry = store.setdefault(key, {"count": 0.0, "sum": 0.0, "average": 0.0})
    entry["count"] += 1
    entry["sum"] += float(value)
    entry["average"] = round(entry["sum"] / entry["count"], 3)


def _normalize_venue_code(value: Any) -> str:
    if value in (None, ""):
        return "00"
    return f"{int(float(value)):02d}"


def _parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _validate_fraction_limit(name: str, value: float | None) -> None:
    if value is None:
        return
    numeric_value = float(value)
    if numeric_value < 0 or numeric_value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")
