"""Probability calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


PROBABILITY_FLOOR = 1e-6


def clip_probabilities(probabilities: np.ndarray | list[float]) -> np.ndarray:
    observed = np.asarray(probabilities, dtype=float)
    return np.clip(observed, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)


def apply_probability_calibration(
    probabilities: np.ndarray | list[float],
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    clipped = clip_probabilities(probabilities)
    if not calibrator:
        return clipped

    method = str(calibrator.get("method") or "")
    if method != "platt_logit":
        return clipped

    coef = float(calibrator.get("coef", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    logits = _probabilities_to_logits(clipped)
    calibrated_logits = logits * coef + intercept
    calibrated = 1.0 / (1.0 + np.exp(-calibrated_logits))
    return clip_probabilities(calibrated)


def fit_platt_calibrator(
    raw_probabilities: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    *,
    random_state: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    clipped = clip_probabilities(raw_probabilities)
    observed = np.asarray(labels, dtype=int)
    summary = {
        "method": "platt_logit",
        "rows": int(len(observed)),
        "accepted": False,
        "raw_log_loss": None,
        "raw_brier_score": None,
        "calibrated_log_loss": None,
        "calibrated_brier_score": None,
    }

    if len(observed) == 0 or len(set(observed.tolist())) < 2:
        summary["reason"] = "insufficient_classes"
        return None, summary

    summary["raw_log_loss"] = round(float(log_loss(observed, clipped, labels=[0, 1])), 6)
    summary["raw_brier_score"] = round(float(brier_score_loss(observed, clipped)), 6)

    logits = _probabilities_to_logits(clipped).reshape(-1, 1)
    model = LogisticRegression(random_state=random_state)
    model.fit(logits, observed)
    calibrated = model.predict_proba(logits)[:, 1]

    calibrated_log_loss = round(float(log_loss(observed, calibrated, labels=[0, 1])), 6)
    calibrated_brier_score = round(float(brier_score_loss(observed, calibrated)), 6)
    summary["calibrated_log_loss"] = calibrated_log_loss
    summary["calibrated_brier_score"] = calibrated_brier_score

    if calibrated_log_loss > summary["raw_log_loss"] or calibrated_brier_score > summary["raw_brier_score"]:
        summary["reason"] = "no_improvement"
        return None, summary

    calibrator = {
        "method": "platt_logit",
        "coef": round(float(model.coef_[0][0]), 8),
        "intercept": round(float(model.intercept_[0]), 8),
    }
    summary["accepted"] = True
    return calibrator, summary


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = clip_probabilities(probabilities)
    return np.log(clipped / (1.0 - clipped))
