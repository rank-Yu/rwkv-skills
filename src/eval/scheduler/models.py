from __future__ import annotations

"""Model discovery helpers."""

import glob
import os
import re
from pathlib import Path
from typing import Final, Sequence

from .config import REPO_ROOT


MODEL_SELECT_CHOICES: Final[tuple[str, ...]] = ("all", "param-extrema")
MODEL_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<family>[a-z]+[0-9]+)(?P<tag>[a-z]+)?(?P<number>[0-9]+)?$",
    re.IGNORECASE,
)


def expand_model_paths(patterns: Sequence[str]) -> list[Path]:
    matched: set[Path] = set()
    for pattern in patterns:
        if not pattern:
            continue
        expanded = os.path.expanduser(os.path.expandvars(pattern))
        if not os.path.isabs(expanded):
            expanded = str(REPO_ROOT / expanded)
        for candidate in glob.glob(expanded):
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                matched.add(candidate_path.resolve())
    return sorted(matched)


def _normalize_model_identifier(raw: str) -> str:
    if not isinstance(raw, str):
        return raw
    sanitized = raw.replace("_", "-")

    def _decimal_replacer(match: re.Match[str]) -> str:
        whole = match.group("whole")
        frac = match.group("frac")
        return f"{whole}.{frac}b"

    sanitized = re.sub(r"(?P<whole>\d+)-(?P<frac>\d+)b", _decimal_replacer, sanitized)
    match = re.match(r"^rwkv7[a-z0-9]*-(.+)$", sanitized)
    tail = match.group(1) if match else sanitized
    parts = tail.split("-")
    head_parts: list[str] = []
    for part in parts:
        if re.fullmatch(r"\d{8}", part):
            break
        if part.lower().startswith("ctx"):
            break
        head_parts.append(part)
    return "-".join(head_parts) if head_parts else tail


def _extract_param_count(model_short: str) -> float | None:
    if not isinstance(model_short, str):
        return None
    for token in model_short.split("-"):
        if token.lower().endswith("b"):
            number = token[:-1]
            try:
                return float(number)
            except ValueError:
                continue
    return None


def _model_version_sort_key(model: str) -> tuple[str, int, int, str]:
    if not isinstance(model, str) or not model:
        return "", 0, 0, ""
    prefix = model.split("-")[0]
    match = MODEL_PREFIX_PATTERN.match(prefix)
    if not match:
        return prefix, 0, 0, model
    family = match.group("family") or ""
    tag = match.group("tag") or ""
    number = match.group("number") or ""

    if not tag:
        tag_rank = 0
    elif tag == "a":
        tag_rank = 1
    else:
        tag_rank = 2

    try:
        number_rank = int(number) if number else 0
    except ValueError:
        number_rank = 0

    return (family, tag_rank, number_rank, model)


def filter_model_paths(
    model_paths: Sequence[Path],
    strategy: str,
    min_param_b: float | None,
    max_param_b: float | None,
) -> list[Path]:
    entries: list[tuple[float | None, str, Path]] = []
    for path in model_paths:
        normalized = _normalize_model_identifier(path.stem)
        params_val = _extract_param_count(normalized)
        if min_param_b is not None and params_val is not None and params_val < min_param_b:
            continue
        if max_param_b is not None:
            if params_val is None or params_val > max_param_b:
                continue
        entries.append((params_val, normalized, path))

    if strategy == "all":
        return [path for _, _, path in entries]

    if strategy == "param-extrema":
        grouped: dict[float | None, list[tuple[str, Path]]] = {}
        for params_val, normalized, path in entries:
            grouped.setdefault(params_val, []).append((normalized, path))

        selected: set[Path] = set()
        for params_val, models in grouped.items():
            model_map: dict[str, Path] = {}
            for normalized, path in models:
                model_map.setdefault(normalized, path)
            if params_val is None or len(model_map) <= 2:
                selected.update(model_map.values())
                continue
            ordered_models = sorted(model_map.keys(), key=_model_version_sort_key)
            keep = {ordered_models[0], ordered_models[-1]}
            for model_name in keep:
                selected.add(model_map[model_name])
        return [path for _, _, path in entries if path in selected]
    raise ValueError(f"未知的模型选择策略: {strategy!r}")


__all__ = [
    "expand_model_paths",
    "filter_model_paths",
    "MODEL_SELECT_CHOICES",
]
