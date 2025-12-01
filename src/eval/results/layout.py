"""Canonical output layout for evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.eval.scheduler.config import RESULTS_ROOT, DEFAULT_LOG_DIR, DEFAULT_RUN_LOG_DIR
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug


LOGS_ROOT = DEFAULT_RUN_LOG_DIR
SCORES_ROOT = DEFAULT_LOG_DIR


def ensure_results_structure() -> None:
    for path in (RESULTS_ROOT, LOGS_ROOT, SCORES_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def result_basename(dataset_slug: str, *, is_cot: bool, model_name: str) -> str:
    dataset_part = canonical_slug(dataset_slug)
    cot_part = "cot" if is_cot else "nocot"
    model_part = safe_slug(model_name)
    return safe_slug(f"{dataset_part}_{cot_part}_{model_part}")


def jsonl_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    return LOGS_ROOT / f"{result_basename(dataset_slug, is_cot=is_cot, model_name=model_name)}.jsonl"


def scores_path(dataset_slug: str, *, is_cot: bool, model_name: str) -> Path:
    ensure_results_structure()
    return SCORES_ROOT / f"{result_basename(dataset_slug, is_cot=is_cot, model_name=model_name)}.json"


__all__ = [
    "LOGS_ROOT",
    "SCORES_ROOT",
    "ensure_results_structure",
    "result_basename",
    "jsonl_path",
    "scores_path",
    "write_scores_json",
]


def write_scores_json(
    dataset_slug: str,
    *,
    is_cot: bool,
    model_name: str,
    metrics: dict,
    samples: int,
    log_path: Path | str,
    task: str | None = None,
    task_details: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist aggregated metrics as JSON in the canonical scores directory.

    The payload shape intentionally mirrors the documented example:
    {
        "dataset": ..., "model": ..., "cot": bool,
        "metrics": {...}, "samples": int,
        "created_at": iso8601, "log_path": "results/logs/...jsonl",
        "task": "optional task name",
        "task_details": {"task specific breakdowns"},
        ...extra
    }
    """

    ensure_results_structure()
    path = scores_path(dataset_slug, is_cot=is_cot, model_name=model_name)
    payload = {
        "dataset": dataset_slug,
        "model": model_name,
        "cot": bool(is_cot),
        "metrics": metrics,
        "samples": int(samples),
        "created_at": datetime.utcnow().replace(microsecond=False).isoformat() + "Z",
        "log_path": str(log_path),
    }
    if task:
        payload["task"] = task
    if task_details:
        payload["task_details"] = task_details
    if extra:
        payload.update(extra)

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
