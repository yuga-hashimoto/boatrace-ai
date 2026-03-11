"""Prediction using a trained artifact."""

from __future__ import annotations

from typing import Any

import numpy as np

from boatrace_ai.calibration import apply_probability_calibration
from boatrace_ai.collect.official import BeforeInfo, RaceCard
from boatrace_ai.features.dataset import FEATURE_COLUMNS, build_rows_from_record, rows_to_matrix
from boatrace_ai.predict.baseline import EntrantPrediction, RacePrediction, TrifectaPrediction


def predict_race_with_model(
    card: RaceCard,
    beforeinfo: BeforeInfo | None,
    artifact: dict[str, Any],
    top_k: int = 5,
) -> RacePrediction:
    feature_columns = artifact.get("feature_columns") or FEATURE_COLUMNS
    model = artifact["model"]
    record = {
        "date": card.date,
        "venue_code": card.venue_code,
        "venue_name": card.venue_name,
        "race_no": card.race_no,
        "meeting_name": card.meeting_name,
        "deadline": card.deadline,
        "card": card.to_dict(),
        "beforeinfo": beforeinfo.to_dict() if beforeinfo else None,
        "result": None,
    }
    rows = build_rows_from_record(record)
    matrix = np.array(rows_to_matrix(rows, feature_columns), dtype=float)
    raw_probabilities = model.predict_proba(matrix)[:, 1]
    probabilities = apply_probability_calibration(
        raw_probabilities,
        artifact.get("calibrator"),
    )
    normalized = _normalize_probabilities(probabilities.tolist())
    raw_normalized = _normalize_probabilities(raw_probabilities.tolist())
    by_lane = {int(row["lane"]): probability for row, probability in zip(rows, normalized, strict=True)}
    raw_by_lane = {int(row["lane"]): probability for row, probability in zip(rows, raw_normalized, strict=True)}

    entrants = []
    for entrant in card.entrants:
        lane = entrant.lane
        entrants.append(
            EntrantPrediction(
                lane=lane,
                racer_id=entrant.racer_id,
                name=entrant.name,
                grade=entrant.grade,
                score=raw_by_lane[lane],
                win_probability=by_lane[lane],
                top3_probability=0.0,
                contributions={},
            )
        )

    trifecta_probs = _trifecta_probabilities(normalized, [entrant.lane for entrant in card.entrants])
    top3_by_lane = {entrant.lane: 0.0 for entrant in card.entrants}
    for order, probability in trifecta_probs:
        for lane in order:
            top3_by_lane[lane] += probability

    entrants = [
        EntrantPrediction(
            lane=entrant.lane,
            racer_id=entrant.racer_id,
            name=entrant.name,
            grade=entrant.grade,
            score=entrant.score,
            win_probability=entrant.win_probability,
            top3_probability=top3_by_lane[entrant.lane],
            contributions=entrant.contributions,
        )
        for entrant in entrants
    ]
    entrants.sort(key=lambda item: item.win_probability, reverse=True)

    trifectas = [
        TrifectaPrediction(order=order, probability=round(probability, 4))
        for order, probability in trifecta_probs[:top_k]
    ]

    return RacePrediction(
        model_name=artifact.get("model_name", "trained_model"),
        race=card,
        entrants=entrants,
        trifectas=trifectas,
    )


def _normalize_probabilities(probabilities: list[float]) -> list[float]:
    floored = [max(probability, 1e-6) for probability in probabilities]
    total = sum(floored)
    return [value / total for value in floored]


def _trifecta_probabilities(
    probabilities: list[float],
    lanes: list[int],
) -> list[tuple[tuple[int, int, int], float]]:
    total = sum(probabilities)
    candidates: list[tuple[tuple[int, int, int], float]] = []

    for first_index, first_lane in enumerate(lanes):
        for second_index, second_lane in enumerate(lanes):
            if second_index == first_index:
                continue
            for third_index, third_lane in enumerate(lanes):
                if third_index in {first_index, second_index}:
                    continue
                remaining_after_first = total - probabilities[first_index]
                remaining_after_second = remaining_after_first - probabilities[second_index]
                if remaining_after_first <= 0 or remaining_after_second <= 0:
                    continue
                probability = (
                    probabilities[first_index] / total
                    * probabilities[second_index] / remaining_after_first
                    * probabilities[third_index] / remaining_after_second
                )
                candidates.append(((first_lane, second_lane, third_lane), probability))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates
