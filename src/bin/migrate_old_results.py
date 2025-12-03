"""Legacy results migrator.

Scans ``results_old`` style JSON dumps (from earlier rwkv-mmlu / rwkv-skills
repos) and rewrites their metrics into the canonical ``results/scores`` layout
so that the current dashboard / scheduler logic can reuse historical runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.eval.results.layout import ensure_results_structure, scores_path
from src.eval.scheduler import jobs as job_catalog
from src.eval.scheduler.config import REPO_ROOT
from src.eval.scheduler.dataset_utils import (
    canonical_slug,
    infer_dataset_slug_from_path,
    safe_slug,
)


LEGACY_PREFIX_MAP: dict[str, tuple[bool, str]] = {
    "single_choice_plain": (False, "multiple_choice"),
    "single_choice_cot": (True, "multiple_choice_cot"),
    "cot_general": (True, "free_response"),
    "cot_llm_judge": (True, "free_response_judge"),
    "instruction_following": (False, "instruction_following"),
}

KNOWN_SLUGS: set[str] = set(
    list(job_catalog.MULTICHOICE_DATASET_SLUGS)
    + list(job_catalog.MATH_DATASET_SLUGS)
    + list(job_catalog.SPECIAL_DATASET_SLUGS)
    + list(job_catalog.CODE_DATASET_SLUGS)
    + list(job_catalog.LLM_JUDGE_DATASET_SLUGS)
    + list(job_catalog.INSTRUCTION_FOLLOWING_DATASET_SLUGS)
)


@dataclass(slots=True)
class LegacyScore:
    dataset_slug: str
    is_cot: bool
    model_basename: str
    model_label: str
    metrics: dict[str, Any]
    samples: int
    log_path: str
    task: str | None
    task_details: dict[str, Any] | None
    created_at: datetime
    source_path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy JSON results into results/scores")
    parser.add_argument(
        "--source",
        default="results_old",
        help="Directory root that contains legacy JSON files (default: results_old)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be migrated without writing scores",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing score JSON if present",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every migrated file instead of only a summary",
    )
    return parser.parse_args()


def _extract_accuracy_by_subject(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, dict):
        return None
    result: dict[str, float] = {}
    for subject, payload in raw.items():
        accuracy: float | None = None
        if isinstance(payload, dict):
            acc = payload.get("accuracy")
            if isinstance(acc, (int, float)):
                accuracy = float(acc)
        elif isinstance(payload, (int, float)):
            accuracy = float(payload)
        if accuracy is not None:
            result[subject] = accuracy
    return result or None


def _metric_from_dict(data: dict[str, Any], key: str) -> float | None:
    if not isinstance(data, dict):
        return None
    payload = data.get(key)
    if isinstance(payload, dict):
        value = payload.get("accuracy") or payload.get("value")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _sanitize_samples(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _parse_timestamp_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            if len(text) == 14:
                try:
                    return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            try:
                epoch = int(text)
                divisor = 1000.0 if len(text) >= 13 else 1.0
                return datetime.fromtimestamp(epoch / divisor, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                return None
        try:
            normalised = text.replace("Z", "+00:00") if text.endswith("Z") else text
            parsed = datetime.fromisoformat(normalised)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _resolve_created_at(raw: Mapping[str, Any], path: Path) -> datetime:
    for key in ("created_at", "timestamp"):
        if key in raw:
            parsed = _parse_timestamp_value(raw.get(key))
            if parsed:
                return parsed
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _resolve_model_label(data: dict[str, Any]) -> tuple[str, str]:
    model_label = str(data.get("model") or "").strip()
    model_path = str(data.get("model_path") or "").strip()
    if model_path:
        model_basename = Path(model_path).stem
    elif model_label:
        model_basename = model_label
    else:
        model_basename = "unknown_model"
    return model_basename, (model_label or model_basename)


def _dataset_slug_from_record(data: dict[str, Any], path: Path) -> str | None:
    slug = data.get("dataset_slug")
    if isinstance(slug, str) and slug.strip():
        return canonical_slug(slug)
    dataset_path = data.get("dataset")
    if isinstance(dataset_path, str) and dataset_path.strip():
        return canonical_slug(infer_dataset_slug_from_path(dataset_path))
    stem = path.stem.lower()
    for known in KNOWN_SLUGS:
        if known and known in stem:
            return canonical_slug(known)
    return None


def _prefix_type(data: dict[str, Any]) -> tuple[str | None, bool | None]:
    benchmark = data.get("benchmark")
    dataset_slug = data.get("dataset_slug")
    if not benchmark or not dataset_slug:
        return None, None
    suffix = f"_{dataset_slug}"
    if not benchmark.endswith(suffix):
        return None, None
    prefix = benchmark[: -len(suffix)]
    mapped = LEGACY_PREFIX_MAP.get(prefix)
    if mapped is None:
        return None, None
    is_cot, task = mapped
    return task, is_cot


def _infer_cot_flag(data: dict[str, Any], path: Path) -> bool | None:
    cot = data.get("cot")
    if isinstance(cot, bool):
        return cot
    name = path.stem.lower()
    if "nocot" in name or "_plain_" in name:
        return False
    if "cot_" in name or name.startswith("cot_") or "_cot_" in name:
        return True
    return None


def _infer_type_from_name(path: Path, dataset_slug: str | None) -> tuple[str | None, bool | None]:
    name = path.name.lower()
    canonical = canonical_slug(dataset_slug or "") if dataset_slug else ""
    if name.startswith("cot_llm_judge"):
        return "free_response_judge", True
    if name.startswith("cot_"):
        if canonical in job_catalog.LLM_JUDGE_DATASET_SLUGS:
            return "free_response_judge", True
        if canonical in job_catalog.MULTICHOICE_DATASET_SLUGS:
            return "multiple_choice_cot", True
        return "free_response", True
    if name.startswith("results_"):
        return "multiple_choice", False
    return None, None


def _build_metrics(task: str | None, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    score_by_subject = _extract_accuracy_by_subject(data.get("score_by_subject"))
    task_details: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}

    if task in {"multiple_choice", "multiple_choice_cot"}:
        acc = data.get("accuracy")
        if isinstance(acc, (int, float)):
            metrics["accuracy"] = float(acc)
        if score_by_subject:
            task_details = {"accuracy_by_subject": score_by_subject}
    elif task == "free_response":
        raw_metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        exact = _metric_from_dict(raw_metrics, "ExactMatchMetric") or data.get("accuracy")
        judge = _metric_from_dict(raw_metrics, "JudgeMetric")
        if isinstance(exact, (int, float)):
            metrics["exact_accuracy"] = float(exact)
        if isinstance(judge, (int, float)):
            metrics["judge_accuracy"] = float(judge)
        if score_by_subject:
            task_details = {"accuracy_by_subject": score_by_subject}
    elif task == "free_response_judge":
        acc = data.get("accuracy")
        judge = _metric_from_dict(data.get("metrics", {}), "JudgeMetric")
        value = judge if isinstance(judge, (int, float)) else acc
        if isinstance(value, (int, float)):
            metrics["judge_accuracy"] = float(value)
        if score_by_subject:
            task_details = {"accuracy_by_subject": score_by_subject}
    elif task == "instruction_following":
        raw_metrics = data.get("metrics") or {}
        inst = raw_metrics.get("InstructionFollowingMetric") if isinstance(raw_metrics, dict) else None
        if isinstance(inst, dict):
            prompt = inst.get("prompt_level", {}).get("strict")
            instr = inst.get("instruction_level", {}).get("strict")
            if isinstance(prompt, (int, float)):
                metrics["prompt_accuracy"] = float(prompt)
            if isinstance(instr, (int, float)):
                metrics["instruction_accuracy"] = float(instr)
            tier0 = inst.get("tier0") if isinstance(inst.get("tier0"), dict) else None
            tier1 = inst.get("tier1") if isinstance(inst.get("tier1"), dict) else None
            details: dict[str, Any] = {}
            if tier0:
                details["tier0_accuracy"] = {
                    key: float(val.get("strict", 0.0))
                    for key, val in tier0.items()
                    if isinstance(val, dict)
                }
            if tier1:
                details["tier1_accuracy"] = {
                    key: float(val.get("strict", 0.0))
                    for key, val in tier1.items()
                    if isinstance(val, dict)
                }
            task_details = details or None
    else:
        acc = data.get("accuracy")
        if isinstance(acc, (int, float)):
            metrics["accuracy"] = float(acc)
        if score_by_subject:
            task_details = {"accuracy_by_subject": score_by_subject}

    return metrics, task_details


def _infer_samples(data: dict[str, Any]) -> int:
    for key in ("total", "num_samples", "samples"):
        if key in data:
            return _sanitize_samples(data.get(key))
    generations = data.get("generations")
    if isinstance(generations, list):
        return len(generations)
    strict_results = data.get("strict_results")
    if isinstance(strict_results, list):
        return len(strict_results)
    score_by_subject = data.get("score_by_subject")
    if isinstance(score_by_subject, dict):
        total = 0
        for payload in score_by_subject.values():
            if isinstance(payload, dict) and isinstance(payload.get("total"), (int, float)):
                total += int(payload["total"])
        if total:
            return total
    return 0


def _log_path_for(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def collect_legacy_scores(root: Path) -> list[LegacyScore]:
    scores: list[LegacyScore] = []
    for path in sorted(root.rglob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, dict) or "model" not in raw:
            continue
        dataset_slug = _dataset_slug_from_record(raw, path)
        if not dataset_slug:
            continue

        task, is_cot = _prefix_type(raw)
        if is_cot is None:
            is_cot = _infer_cot_flag(raw, path)
        if task is None:
            inferred_task, inferred_cot = _infer_type_from_name(path, dataset_slug)
            task = task or inferred_task
            is_cot = inferred_cot if is_cot is None else is_cot

        if task is None or is_cot is None:
            # Try scheduler job detection as a final fallback.
            job_cot = job_catalog.detect_job_from_dataset(dataset_slug, is_cot=True)
            job_nocot = job_catalog.detect_job_from_dataset(dataset_slug, is_cot=False)

            if is_cot is True and job_cot:
                task = job_cot
            elif is_cot is False and job_nocot:
                task = job_nocot
            elif job_cot and not job_nocot:
                task = job_cot
                is_cot = True
            elif job_nocot and not job_cot:
                task = job_nocot
                is_cot = False

        if task is None or is_cot is None:
            continue

        model_basename, model_label = _resolve_model_label(raw)
        metrics, task_details = _build_metrics(task, raw)
        samples = _infer_samples(raw)
        created_at = _resolve_created_at(raw, path)
        scores.append(
            LegacyScore(
                dataset_slug=canonical_slug(dataset_slug),
                is_cot=bool(is_cot),
                model_basename=model_basename,
                model_label=model_label,
                metrics=metrics,
                samples=samples,
                log_path=_log_path_for(path),
                task=task,
                task_details=task_details,
                created_at=created_at,
                source_path=path,
            )
        )
    return scores


def _deduplicate(records: Iterable[LegacyScore]) -> list[LegacyScore]:
    latest: dict[tuple[str, bool, str, str | None], LegacyScore] = {}
    for record in records:
        key = (
            record.dataset_slug,
            record.is_cot,
            safe_slug(record.model_basename),
            record.task,
        )
        existing = latest.get(key)
        if existing is None or record.created_at >= existing.created_at:
            latest[key] = record
    return sorted(latest.values(), key=lambda item: (item.dataset_slug, item.model_basename))


def write_scores(records: Iterable[LegacyScore], *, dry_run: bool, overwrite: bool, verbose: bool) -> tuple[int, int]:
    ensure_results_structure()
    written = 0
    skipped = 0
    for record in records:
        score_file = scores_path(record.dataset_slug, is_cot=record.is_cot, model_name=record.model_basename)
        if score_file.exists() and not overwrite:
            skipped += 1
            if verbose:
                print(f"⚠️  skip existing {score_file}")
            continue
        payload = {
            "dataset": record.dataset_slug,
            "model": record.model_label,
            "cot": bool(record.is_cot),
            "metrics": record.metrics,
            "samples": int(record.samples),
            "created_at": record.created_at.replace(microsecond=False).isoformat().replace("+00:00", "Z"),
            "log_path": record.log_path,
            "task": record.task,
        }
        if record.task_details:
            payload["task_details"] = record.task_details
        if dry_run:
            skipped += 1
            print(f"DRY-RUN would write {score_file} <- {record.source_path}")
            continue
        score_file.parent.mkdir(parents=True, exist_ok=True)
        with score_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        written += 1
        if verbose:
            print(f"✅ wrote {score_file} from {record.source_path}")
    return written, skipped


def main() -> int:
    args = _parse_args()
    legacy_root = Path(args.source)
    if not legacy_root.exists():
        print(f"❌ legacy root {legacy_root} 不存在")
        return 1
    records = collect_legacy_scores(legacy_root)
    if not records:
        print("⚠️ 未发现可迁移的 JSON 文件")
        return 0
    deduped = _deduplicate(records)
    written, skipped = write_scores(deduped, dry_run=args.dry_run, overwrite=args.overwrite, verbose=args.verbose)
    total = len(deduped)
    if args.dry_run:
        print(f"🧪 dry-run: {total} 条结果会被迁移 ({skipped} skipped)")
    else:
        print(f"📦 migrated {written}/{total} results ({skipped} skipped)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
