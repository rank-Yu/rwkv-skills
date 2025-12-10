from __future__ import annotations

"""Utilities for deriving automatic samples-per-task overrides."""

import math
import os

_DEFAULT_AUTO_SAMPLES_CAP = 256


def _compute_auto_samples_cap() -> int:
    raw = os.environ.get("RUN_AUTO_SAMPLES_PER_TASK_MAX")
    if raw is None:
        return _DEFAULT_AUTO_SAMPLES_CAP
    try:
        value = int(raw)
    except (TypeError, ValueError):
        print(
            f"⚠️  RUN_AUTO_SAMPLES_PER_TASK_MAX={raw!r} 无法解析，已回退到 {_DEFAULT_AUTO_SAMPLES_CAP}",
        )
        return _DEFAULT_AUTO_SAMPLES_CAP
    return max(1, value)


_AUTO_SAMPLES_CAP = _compute_auto_samples_cap()


def derive_auto_samples_per_task(batch_size: int | None, questions: int | None) -> int | None:
    """Infer samples-per-task so that batch roughly covers the dataset once."""

    if batch_size is None or not questions or questions <= 0:
        return None
    desired = math.ceil(batch_size / questions)
    if desired <= 1:
        return None
    return min(desired, _AUTO_SAMPLES_CAP)


__all__ = ["derive_auto_samples_per_task"]
