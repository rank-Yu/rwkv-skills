"""Helpers to load score artifacts and normalise them for the space dashboard."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.eval.results.layout import SCORES_ROOT
from src.eval.scheduler.dataset_utils import canonical_slug
from src.eval.scheduler.jobs import detect_job_from_dataset


ARCH_VERSIONS = ("rwkv7", "rwkv7a", "rwkv7b")
DATA_VERSIONS = ("g0", "g0a", "g0a2", "g0a3", "g0a4", "g0b", "g1", "g1a", "g1a2", "g1a3", "g1a4", "g1b")
NUM_PARAMS = ("0_1b", "0_4b", "1_5b", "2_9b", "7_2b", "13_3b")


@dataclass(slots=True, frozen=True)
class ModelSignature:
    arch: str | None
    data: str | None
    params: str | None
    arch_rank: int | None
    data_rank: int | None
    param_rank: int | None

    def data_key(self) -> int:
        return self.data_rank if self.data_rank is not None else -1


@dataclass(slots=True, frozen=True)
class ScoreEntry:
    dataset: str
    model: str
    metrics: dict[str, Any]
    samples: int
    created_at: datetime
    log_path: str
    cot: bool
    task: str | None
    task_details: dict[str, Any] | None
    path: Path
    domain: str
    extra: dict[str, Any]
    arch_version: str | None
    data_version: str | None
    num_params: str | None


def _normalize_token(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _match_rank(text: str, ordered: tuple[str, ...]) -> tuple[str | None, int | None]:
    normalized = _normalize_token(text)
    best_rank: int | None = None
    best_token: str | None = None
    for idx, token in enumerate(ordered):
        tok_norm = _normalize_token(token)
        if tok_norm and tok_norm in normalized:
            if best_rank is None or idx > best_rank:
                best_rank = idx
                best_token = token.upper()
    return best_token, best_rank


def parse_model_signature(model: str) -> ModelSignature:
    arch, arch_rank = _match_rank(model, ARCH_VERSIONS)
    data, data_rank = _match_rank(model, DATA_VERSIONS)
    params, param_rank = _match_rank(model, NUM_PARAMS)
    return ModelSignature(
        arch=arch,
        data=data,
        params=params,
        arch_rank=arch_rank,
        data_rank=data_rank,
        param_rank=param_rank,
    )


def _parse_created_at(raw: Any, path: Path) -> datetime:
    if isinstance(raw, str):
        try:
            cleaned = raw.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(cleaned)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.utcnow()


def _infer_domain(dataset_slug: str, *, is_cot: bool, task: str | None) -> str:
    slug = canonical_slug(dataset_slug)
    if slug.startswith("mmlu"):
        return "mmlu系列"
    job = detect_job_from_dataset(slug, is_cot=is_cot)
    if job in {"code_human_eval", "code_mbpp"}:
        return "coding系列"
    if job == "instruction_following":
        return "instruction following系列"
    if job in {"free_response", "free_response_judge"}:
        return "math reasoning系列"
    if job in {"multi_choice_plain", "multi_choice_cot"}:
        return "multi-choice系列"
    if task:
        if "code" in task:
            return "coding系列"
        if "instruction" in task:
            return "instruction following系列"
    return "其他"


def load_scores(scores_root: str | Path = SCORES_ROOT, errors: list[str] | None = None) -> list[ScoreEntry]:
    root = Path(scores_root)
    if not root.exists():
        return []
    entries: list[ScoreEntry] = []
    for path in sorted(root.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # noqa: BLE001 - best effort ingest
            msg = f"无法读取分数文件 {path}: {exc}"
            print(f"[space] {msg}", file=sys.stderr)
            if errors is not None:
                errors.append(msg)
            continue

        dataset = canonical_slug(str(payload.get("dataset", "")).strip())
        model = str(payload.get("model", "")).strip()
        if not dataset or not model:
            continue

        metrics = payload.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}

        created_at = _parse_created_at(payload.get("created_at"), path)
        raw_samples = payload.get("samples")
        try:
            samples = int(raw_samples) if raw_samples is not None else 0
        except (TypeError, ValueError):
            msg = f"分数文件 {path} 的 samples 字段无法解析: {raw_samples!r}，已按 0 处理"
            print(f"[space] {msg}", file=sys.stderr)
            if errors is not None:
                errors.append(msg)
            samples = 0
        log_path = str(payload.get("log_path") or "")
        task_details = payload.get("task_details") if isinstance(payload.get("task_details"), dict) else None
        task = str(payload.get("task")).strip() if payload.get("task") else None
        is_cot = bool(payload.get("cot", False))
        domain = _infer_domain(dataset, is_cot=is_cot, task=task)

        sig = parse_model_signature(model)

        known_keys = {
            "dataset",
            "model",
            "metrics",
            "samples",
            "created_at",
            "log_path",
            "task",
            "task_details",
            "cot",
        }
        extra = {k: v for k, v in payload.items() if k not in known_keys}

        entries.append(
            ScoreEntry(
                dataset=dataset,
                model=model,
                metrics=metrics,
                samples=samples,
                created_at=created_at,
                log_path=log_path,
                cot=is_cot,
                task=task,
                task_details=task_details,
                path=path,
                domain=domain,
                extra=extra,
                arch_version=sig.arch,
                data_version=sig.data,
                num_params=sig.params,
            )
        )
    return entries


def pick_latest_model(entries: Iterable[ScoreEntry]) -> str | None:
    by_model: dict[str, list[ScoreEntry]] = {}
    for entry in entries:
        by_model.setdefault(entry.model, []).append(entry)

    latest_model: str | None = None
    best_key: tuple[int, float] | None = None
    for model, items in by_model.items():
        sig = parse_model_signature(model)
        newest_time = max(item.created_at for item in items).timestamp()
        key = (sig.data_key(), newest_time)
        if best_key is None or key > best_key:
            best_key = key
            latest_model = model
    return latest_model


def latest_entries_for_model(entries: Iterable[ScoreEntry], model: str | None) -> list[ScoreEntry]:
    if not model:
        return []
    latest: dict[tuple[str, bool, str | None], ScoreEntry] = {}
    for entry in entries:
        if entry.model != model:
            continue
        key = (entry.dataset, entry.cot, entry.task)
        previous = latest.get(key)
        if previous is None or entry.created_at > previous.created_at:
            latest[key] = entry
    return sorted(
        latest.values(),
        key=lambda item: (item.domain, item.dataset, item.task or ""),
    )


def list_models(entries: Iterable[ScoreEntry]) -> list[str]:
    return sorted({entry.model for entry in entries})


def list_domains(entries: Iterable[ScoreEntry]) -> list[str]:
    return sorted({entry.domain for entry in entries})


__all__ = [
    "ScoreEntry",
    "list_domains",
    "list_models",
    "load_scores",
    "pick_latest_model",
    "latest_entries_for_model",
    "parse_model_signature",
]
