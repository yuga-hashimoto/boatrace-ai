"""Baseline scorer for official BOAT RACE race cards."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import itertools
import math
from typing import Any

from boatrace_ai.collect.official import RaceCard, RaceEntrant


LANE_BIAS = {
    1: 1.15,
    2: 0.55,
    3: 0.20,
    4: -0.05,
    5: -0.40,
    6: -0.85,
}
GRADE_BIAS = {
    "A1": 1.00,
    "A2": 0.45,
    "B1": 0.00,
    "B2": -0.65,
}
FEATURE_WEIGHTS = {
    "national_win_rate": 0.90,
    "local_win_rate": 0.40,
    "national_2ren_rate": 0.35,
    "local_2ren_rate": 0.18,
    "motor_2ren_rate": 0.22,
    "boat_2ren_rate": 0.08,
    "average_start_timing": 0.28,
    "recent_form": 0.32,
    "recent_start": 0.10,
}


@dataclass(frozen=True)
class EntrantPrediction:
    lane: int
    racer_id: str
    name: str
    grade: str
    score: float
    win_probability: float
    top3_probability: float
    contributions: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrifectaPrediction:
    order: tuple[int, int, int]
    probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RacePrediction:
    model_name: str
    race: RaceCard
    entrants: list[EntrantPrediction]
    trifectas: list[TrifectaPrediction]
    trifecta_probability_map: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["race"] = self.race.to_dict()
        payload["entrants"] = [entrant.to_dict() for entrant in self.entrants]
        payload["trifectas"] = [trifecta.to_dict() for trifecta in self.trifectas]
        return payload


def predict_race(card: RaceCard, top_k: int = 5) -> RacePrediction:
    entrants = list(card.entrants)
    if len(entrants) < 3:
        raise ValueError("At least 3 entrants are required to build a prediction.")

    normalized = _normalized_feature_map(entrants)
    scores: list[float] = []
    contributions: list[dict[str, float]] = []

    for entrant in entrants:
        component_map = {
            "lane_bias": LANE_BIAS.get(entrant.lane, 0.0),
            "grade_bias": GRADE_BIAS.get(entrant.grade, -0.20),
            "f_penalty": -0.35 * entrant.f_count,
            "l_penalty": -0.15 * entrant.l_count,
        }

        for feature_name, weight in FEATURE_WEIGHTS.items():
            component_map[feature_name] = normalized[feature_name][entrant.lane] * weight

        score = sum(component_map.values())
        scores.append(score)
        contributions.append(component_map)

    weights = _stable_weights(scores)
    win_probabilities = [weight / sum(weights) for weight in weights]
    trifecta_probabilities = _plackett_luce_top3(weights, entrants)

    top3_by_lane = {entrant.lane: 0.0 for entrant in entrants}
    for order, probability in trifecta_probabilities:
        for lane in order:
            top3_by_lane[lane] += probability

    entrant_predictions = [
        EntrantPrediction(
            lane=entrant.lane,
            racer_id=entrant.racer_id,
            name=entrant.name,
            grade=entrant.grade,
            score=round(score, 4),
            win_probability=round(win_probability, 4),
            top3_probability=round(top3_by_lane[entrant.lane], 4),
            contributions={key: round(value, 4) for key, value in contribution.items()},
        )
        for entrant, score, win_probability, contribution in zip(
            entrants,
            scores,
            win_probabilities,
            contributions,
            strict=True,
        )
    ]
    entrant_predictions.sort(key=lambda item: item.win_probability, reverse=True)

    trifectas = [
        TrifectaPrediction(order=order, probability=round(probability, 4))
        for order, probability in trifecta_probabilities[:top_k]
    ]

    return RacePrediction(
        model_name="baseline_official_card_v1",
        race=card,
        entrants=entrant_predictions,
        trifectas=trifectas,
    )


def _normalized_feature_map(entrants: list[RaceEntrant]) -> dict[str, dict[int, float]]:
    raw_features = {
        "national_win_rate": [entrant.national_win_rate for entrant in entrants],
        "local_win_rate": [entrant.local_win_rate for entrant in entrants],
        "national_2ren_rate": [entrant.national_2ren_rate for entrant in entrants],
        "local_2ren_rate": [entrant.local_2ren_rate for entrant in entrants],
        "motor_2ren_rate": [entrant.motor_2ren_rate for entrant in entrants],
        "boat_2ren_rate": [entrant.boat_2ren_rate for entrant in entrants],
        "average_start_timing": [entrant.average_start_timing for entrant in entrants],
        "recent_form": [_recent_form_score(entrant) for entrant in entrants],
        "recent_start": [_recent_start_score(entrant) for entrant in entrants],
    }
    reverse_features = {"average_start_timing", "recent_start"}

    normalized: dict[str, dict[int, float]] = {}
    for feature_name, values in raw_features.items():
        feature_scores = _zscore(values)
        if feature_name in reverse_features:
            feature_scores = [-score for score in feature_scores]
        normalized[feature_name] = {
            entrant.lane: score
            for entrant, score in zip(entrants, feature_scores, strict=True)
        }
    return normalized


def _recent_form_score(entrant: RaceEntrant) -> float | None:
    if not entrant.recent_finishes:
        return None
    quality = [max(0.0, 7.0 - float(finish)) / 6.0 for finish in entrant.recent_finishes]
    return sum(quality) / len(quality)


def _recent_start_score(entrant: RaceEntrant) -> float | None:
    if not entrant.recent_starts:
        return None
    return sum(entrant.recent_starts) / len(entrant.recent_starts)


def _zscore(values: list[float | None]) -> list[float]:
    observed = [value for value in values if value is not None]
    if not observed:
        return [0.0 for _ in values]

    mean = sum(observed) / len(observed)
    variance = sum((value - mean) ** 2 for value in observed) / len(observed)
    stddev = math.sqrt(variance)
    if stddev == 0:
        return [0.0 for _ in values]

    return [((value if value is not None else mean) - mean) / stddev for value in values]


def _stable_weights(scores: list[float]) -> list[float]:
    offset = max(scores)
    return [math.exp(score - offset) for score in scores]


def _plackett_luce_top3(
    weights: list[float],
    entrants: list[RaceEntrant],
) -> list[tuple[tuple[int, int, int], float]]:
    indexed_weights = list(zip(entrants, weights, strict=True))
    total_weight = sum(weights)
    permutations: list[tuple[tuple[int, int, int], float]] = []

    for first, second, third in itertools.permutations(indexed_weights, 3):
        remaining_after_first = total_weight - first[1]
        remaining_after_second = remaining_after_first - second[1]
        if remaining_after_first <= 0 or remaining_after_second <= 0:
            continue

        probability = (
            first[1] / total_weight
            * second[1] / remaining_after_first
            * third[1] / remaining_after_second
        )
        permutations.append(((first[0].lane, second[0].lane, third[0].lane), probability))

    permutations.sort(key=lambda item: item[1], reverse=True)
    return permutations
