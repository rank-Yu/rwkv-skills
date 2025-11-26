from __future__ import annotations

"""Helpers to build stable IDs for dispatcher jobs."""

from pathlib import Path

from .dataset_utils import canonical_slug, safe_slug


def build_run_slug(model_path: Path, dataset_slug: str, *, is_cot: bool) -> str:
    dataset_part = canonical_slug(dataset_slug)
    cot_part = "cot" if is_cot else "nocot"
    base = f"{dataset_part}_{cot_part}_{model_path.stem}"
    return safe_slug(base)


def build_run_log_name(model_path: Path, dataset_slug: str, *, is_cot: bool) -> str:
    return f"{build_run_slug(model_path, dataset_slug, is_cot=is_cot)}.log"


__all__ = ["build_run_slug", "build_run_log_name"]
