from __future__ import annotations

"""Queue construction helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence, Pattern

from .dataset_utils import canonical_slug, safe_slug
from .jobs import JOB_CATALOGUE, JobSpec
from .models import expand_model_paths, filter_model_paths
from .naming import build_run_slug
from .state import CompletedKey, RunningEntry


@dataclass(slots=True)
class QueueItem:
    job_name: str
    job_id: str
    dataset_slug: str
    model_path: Path
    model_slug: str
    dataset_path: Path | None = None


_UNKNOWN_QUESTION_COUNT = 10**9
_EARLY_DATASET_SLUGS = frozenset(
    canonical_slug(slug)
    for slug in (
        "mmlu_test",
        "mmlu_pro_test",
        "gsm8k_test",
        "math_500_test",
        "human_eval_test",
        "mbpp_test",
        "ifeval_test",
        "ceval_test",
    )
)


def build_queue(
    *,
    model_globs: Sequence[str],
    job_order: Sequence[str],
    completed: Collection[CompletedKey],
    failed: Collection[CompletedKey] | None = None,
    running: Collection[str],
    skip_dataset_slugs: Collection[str],
    only_dataset_slugs: Collection[str] | None,
    model_select: str,
    min_param_b: float | None,
    max_param_b: float | None,
    model_name_patterns: Sequence[Pattern[str]] | None = None,
) -> list[QueueItem]:
    model_paths = expand_model_paths(model_globs)
    if not model_paths:
        return []
    filtered_models = filter_model_paths(model_paths, model_select, min_param_b, max_param_b)

    pending: list[QueueItem] = []
    completed_set = set(completed)
    failed_set = set(failed or ())
    skip_datasets = {canonical_slug(slug) for slug in skip_dataset_slugs}
    only_datasets = {canonical_slug(slug) for slug in only_dataset_slugs or []}
    running_set = set(running)
    compiled_patterns = tuple(model_name_patterns or ())

    for job_name in job_order:
        spec = JOB_CATALOGUE.get(job_name)
        if spec is None:
            continue
        for dataset_slug in spec.dataset_slugs:
            canonical_dataset = canonical_slug(dataset_slug)
            if only_datasets and canonical_dataset not in only_datasets:
                continue
            if canonical_dataset in skip_datasets:
                continue
            for model_path in filtered_models:
                model_slug = safe_slug(model_path.stem)
                if compiled_patterns:
                    name = model_path.name
                    stem = model_path.stem
                    if not any(pattern.search(name) or pattern.search(stem) for pattern in compiled_patterns):
                        continue
                key = CompletedKey(
                    job=job_name,
                    model_slug=model_slug,
                    dataset_slug=canonical_dataset,
                    is_cot=spec.is_cot,
                )
                if key in completed_set or key in failed_set:
                    continue
                job_id = f"{spec.id_prefix}{build_run_slug(model_path, canonical_dataset, is_cot=spec.is_cot)}"
                if job_id in running_set:
                    continue
                pending.append(
                    QueueItem(
                        job_name=job_name,
                        job_id=job_id,
                        dataset_slug=canonical_dataset,
                        model_path=model_path,
                        model_slug=model_slug,
                    )
                )
    return pending


def sort_queue_items(
    queue: list[QueueItem],
    *,
    question_counts: Mapping[str, int] | None = None,
    job_priority: Mapping[str, int] | None = None,
) -> list[QueueItem]:
    """Sort pending jobs so smaller datasets & nocot runs go first."""

    if not queue:
        return queue
    counts = question_counts or {}
    priorities = job_priority or {}

    def _key(item: QueueItem) -> tuple[int, int, int, int, str, str]:
        job = JOB_CATALOGUE.get(item.job_name)
        is_cot = job.is_cot if job else False
        questions = counts.get(item.dataset_slug, _UNKNOWN_QUESTION_COUNT)
        job_rank = priorities.get(item.job_name, len(priorities))
        # Prioritise specific datasets: non-CoT runs first, then CoT, then everything else.
        if item.dataset_slug in _EARLY_DATASET_SLUGS:
            dataset_rank = 0 if not is_cot else 1
        else:
            dataset_rank = 2
        nocot_rank = 0 if not is_cot else 1
        return (job_rank, dataset_rank, questions, nocot_rank, item.dataset_slug, item.job_id)

    queue.sort(key=_key)
    return queue


__all__ = ["QueueItem", "build_queue", "sort_queue_items"]
