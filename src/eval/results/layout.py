"""Canonical output layout for evaluation artifacts."""

from __future__ import annotations

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
]
