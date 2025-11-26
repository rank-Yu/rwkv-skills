from __future__ import annotations

"""Compute multiple-choice accuracy from pipeline JSONL 输出。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class MultipleChoicePrediction:
    sample_index: int
    dataset: str
    subject: str | None
    answer: str
    logits: dict[str, float]
    predicted: str
    correct: bool


@dataclass(slots=True)
class MultipleChoiceMetrics:
    accuracy: float
    score_by_subject: dict[str | None, float]
    predictions: list[MultipleChoicePrediction]


def load_predictions(path: str | Path) -> list[MultipleChoicePrediction]:
    path = Path(path)
    predictions: list[MultipleChoicePrediction] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            logits = _extract_logits(payload)
            answer = payload.get("answer")
            dataset = payload.get("dataset", path.stem)
            sample_index = int(payload.get("sample_index", len(predictions)))
            predicted = max(logits, key=logits.get)
            subject = payload.get("subject")
            correct = bool(answer) and predicted == answer
            predictions.append(
                MultipleChoicePrediction(
                    sample_index=sample_index,
                    dataset=dataset,
                    subject=subject,
                    answer=answer,
                    logits=logits,
                    predicted=predicted,
                    correct=correct,
                )
            )
    return predictions


def evaluate_predictions(predictions: Iterable[MultipleChoicePrediction]) -> MultipleChoiceMetrics:
    preds = list(predictions)
    if not preds:
        return MultipleChoiceMetrics(accuracy=0.0, score_by_subject={}, predictions=[])

    subject_totals: dict[str | None, tuple[int, int]] = {}
    correct = 0
    for pred in preds:
        total, hits = subject_totals.get(pred.subject, (0, 0))
        total += 1
        if pred.correct:
            hits += 1
            correct += 1
        subject_totals[pred.subject] = (total, hits)

    score_by_subject = {
        key: hits / total if total else 0.0
        for key, (total, hits) in subject_totals.items()
    }
    accuracy = correct / len(preds) if preds else 0.0
    return MultipleChoiceMetrics(
        accuracy=accuracy,
        score_by_subject=score_by_subject,
        predictions=preds,
    )


def _extract_logits(payload: dict) -> dict[str, float]:
    stage_indices = [
        int(key.removeprefix("logits"))
        for key in payload
        if key.startswith("logits") and key.removeprefix("logits").isdigit()
    ]
    if not stage_indices:
        raise ValueError("未在 JSONL 记录中找到 logitsN 字段")
    stage = max(stage_indices)
    logits = payload.get(f"logits{stage}")
    if not isinstance(logits, dict):
        raise TypeError(f"logits{stage} 字段类型错误: {type(logits)}")
    return {str(k): float(v) for k, v in logits.items()}


__all__ = [
    "MultipleChoicePrediction",
    "MultipleChoiceMetrics",
    "load_predictions",
    "evaluate_predictions",
]
