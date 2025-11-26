from __future__ import annotations

"""Helpers for dataset slug normalisation compatible with the dispatcher."""

from pathlib import Path


DATASET_SLUG_ALIASES: dict[str, str] = {
    "math500_test": "math_500_test",
    "math500": "math_500_test",
    "input_data": "ifeval_test",
    "ceval_exam_test": "ceval_test",
    "mbpp": "mbpp_test",
}

_KNOWN_SPLIT_NAMES = {
    "train",
    "test",
    "validation",
    "val",
    "dev",
    "devtest",
    "main",
    "science",
    "verified",
    "all",
    "text",
}


def safe_slug(text: str) -> str:
    slug_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {".", "_"}:
            slug_chars.append(char)
        else:
            slug_chars.append("_")
    return "".join(slug_chars).replace(".", "_")


def canonical_slug(text: str) -> str:
    slug = safe_slug(text)
    return DATASET_SLUG_ALIASES.get(slug, slug)


def make_dataset_slug(name: str, split: str) -> str:
    return canonical_slug(f"{name}_{split}")


def infer_dataset_slug_from_path(dataset_path: str) -> str:
    path = Path(dataset_path)
    stem = path.stem
    parent = path.parent.name
    lower_stem = stem.lower()
    if lower_stem in _KNOWN_SPLIT_NAMES and parent:
        candidate = f"{parent}_{stem}"
    else:
        candidate = stem
    return canonical_slug(candidate)


__all__ = [
    "DATASET_SLUG_ALIASES",
    "canonical_slug",
    "infer_dataset_slug_from_path",
    "make_dataset_slug",
    "safe_slug",
]
