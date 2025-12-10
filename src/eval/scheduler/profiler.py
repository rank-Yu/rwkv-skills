from __future__ import annotations

"""GPU batch-size probing inspired by rwkv-eval's batch profiler."""

import json
import os
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from filelock import FileLock

from .auto_samples import derive_auto_samples_per_task
from .config import DEFAULT_PYTHON, REPO_ROOT
from .jobs import JobSpec


_BATCH_CANDIDATES_ENV = os.environ.get("RUN_BATCH_CANDIDATES") or os.environ.get(
    "RUN_COT_BATCH_CANDIDATES",
    # limit default probing to 2048 to avoid host fallback overhead when VRAM is flaky
    "4096,2048,1024,512,256,128,64,32,16,8,4,2,1",
)
DEFAULT_COT_BATCH_CANDIDATES = tuple(
    int(value.strip()) for value in _BATCH_CANDIDATES_ENV.split(",") if value.strip()
)
DEFAULT_PROBE_MAX_GENERATE = int(os.environ.get("RUN_PROBE_MAX_GENERATE", "16"))


def _extract_cached_batch(record: Any) -> int | None:
    if isinstance(record, dict):
        value = record.get("batch")
    else:
        value = record
    if isinstance(value, bool):
        return int(value) if value else None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def load_batch_cache(cache_path: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, Mapping):
        return {}

    normalised: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for job_name, job_payload in data.items():
        if not isinstance(job_payload, Mapping):
            continue
        job_map: dict[str, dict[str, dict[str, Any]]] = {}
        for model_slug, model_payload in job_payload.items():
            if not isinstance(model_payload, Mapping):
                continue
            model_map: dict[str, dict[str, Any]] = {}
            for gpu_key, record in model_payload.items():
                entry: dict[str, Any] = {}
                batch_value = _extract_cached_batch(record)
                if batch_value is not None:
                    entry["batch"] = batch_value
                if isinstance(record, Mapping):
                    last_error = record.get("last_error")
                    last_probe = record.get("last_probe")
                    if isinstance(last_error, str) and last_error.strip():
                        entry["last_error"] = last_error.strip()
                    if isinstance(last_probe, (int, float)):
                        entry["last_probe"] = float(last_probe)
                if entry:
                    model_map[str(gpu_key)] = entry
            if model_map:
                job_map[model_slug] = model_map
        if job_map:
            normalised[job_name] = job_map
    return normalised


def save_batch_cache(cache_path: Path, data: Mapping[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(cache_path)


def _batch_cache_lock_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".lock")


@contextmanager
def _batch_cache_lock(cache_path: Path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_batch_cache_lock_path(cache_path)))
    with lock:
        yield


def _merge_batch_cache(
    target: MutableMapping[str, dict[str, dict[str, dict[str, Any]]]],
    source: Mapping[str, Any],
) -> MutableMapping[str, dict[str, dict[str, dict[str, Any]]]]:
    for job_name, job_payload in source.items():
        if not isinstance(job_payload, Mapping):
            continue
        job_target = target.setdefault(job_name, {})
        for model_slug, model_payload in job_payload.items():
            if not isinstance(model_payload, Mapping):
                continue
            model_target = job_target.setdefault(model_slug, {})
            for gpu_key, record in model_payload.items():
                if isinstance(record, Mapping):
                    existing = model_target.get(gpu_key)
                    merged = dict(existing) if isinstance(existing, Mapping) else {}
                    merged.update(record)
                    model_target[gpu_key] = merged
                else:
                    model_target[gpu_key] = record
    return target


def update_batch_cache_locked(
    cache_path: Path,
    mutator: Callable[[MutableMapping[str, dict[str, dict[str, dict[str, Any]]]]], Mapping[str, Any] | None],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Load, mutate, and persist the batch cache under a file lock."""

    with _batch_cache_lock(cache_path):
        current = load_batch_cache(cache_path)
        mutated = mutator(current)
        if isinstance(mutated, Mapping):
            current = dict(mutated)
        save_batch_cache(cache_path, current)
        return current


@dataclass(slots=True)
class BatchProbeResult:
    batch_size: int | None


@dataclass(slots=True)
class BatchProfiler:
    cache_path: Path
    candidates: tuple[int, ...] = field(default_factory=lambda: DEFAULT_COT_BATCH_CANDIDATES)
    probe_max_generate: int = DEFAULT_PROBE_MAX_GENERATE
    command_prefix: tuple[str, ...] = (DEFAULT_PYTHON, "-m")
    _cache: MutableMapping[str, dict[str, dict[str, dict[str, Any]]]] = field(init=False, repr=False)
    _cache_mtime: float | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache = load_batch_cache(self.cache_path)
        self._cache_mtime = self._stat_cache_mtime()

    def _stat_cache_mtime(self) -> float | None:
        try:
            return self.cache_path.stat().st_mtime
        except OSError:
            return None

    def _refresh_cache_if_stale(self) -> None:
        current = self._stat_cache_mtime()
        if current is None and self._cache_mtime is None:
            return
        if current != self._cache_mtime:
            self._cache = load_batch_cache(self.cache_path)
            self._cache_mtime = current

    def _persist_cache(self) -> None:
        self._cache = update_batch_cache_locked(self.cache_path, lambda latest: _merge_batch_cache(latest, self._cache))
        self._cache_mtime = self._stat_cache_mtime()

    def determine_batch_size(
        self,
        *,
        job: JobSpec,
        job_id: str,
        gpu: str,
        dataset_path: Path | None,
        model_path: Path,
        model_slug: str,
        env: dict[str, str],
        dataset_questions: int | None = None,
        samples_per_task_flag: str | None = None,
    ) -> int | None:
        self._refresh_cache_if_stale()
        batch_flag = job.batch_flag
        probe_flag = job.probe_flag
        if not batch_flag or not probe_flag:
            return None

        base_candidates = self.candidates
        if not base_candidates:
            return None

        max_allowed_batch = max(base_candidates)
        question_count = dataset_questions if dataset_questions and dataset_questions > 0 else None
        candidates = self._select_probe_candidates(
            base_candidates,
            question_count=question_count,
            sweep_enabled=bool(samples_per_task_flag),
        )
        if not candidates:
            return None

        job_cache = self._cache.setdefault(job.name, {})
        model_cache = job_cache.setdefault(model_slug, {})
        record = model_cache.get(gpu)
        cached_value = _extract_cached_batch(record)
        if cached_value is not None and cached_value > max_allowed_batch:
            cached_value = max_allowed_batch
            entry = {
                "batch": cached_value,
                "last_probe": time.time(),
            }
            expected_samples = self._expected_samples_per_task(
                cached_value,
                dataset_questions=question_count,
                samples_per_task_flag=samples_per_task_flag,
            )
            if expected_samples is not None:
                entry["samples_per_task"] = expected_samples
            model_cache[gpu] = entry
            record = model_cache[gpu]
            self._persist_cache()
        record_is_dict = isinstance(record, Mapping)
        last_probe = record.get("last_probe") if record_is_dict else None
        last_error = record.get("last_error") if record_is_dict else None
        cached_samples_per_task = None
        if record_is_dict:
            sample_field = record.get("samples_per_task")
            if isinstance(sample_field, (int, float)):
                cached_samples_per_task = int(sample_field)
        cache_is_trustworthy = (
            cached_value is not None and cached_value > 0 and isinstance(last_probe, (int, float)) and (not last_error)
        )
        if cache_is_trustworthy and samples_per_task_flag:
            expected_samples = self._expected_samples_per_task(
                cached_value,
                dataset_questions=question_count,
                samples_per_task_flag=samples_per_task_flag,
            )
            if expected_samples is None or cached_samples_per_task != expected_samples:
                cache_is_trustworthy = False
        if cache_is_trustworthy:
            print(f"🔁 Using cached batch size {cached_value} for {job_id} on cuda:{gpu}.")
            return cached_value

        if job.probe_dataset_required and not dataset_path:
            fallback = min(base_candidates)
            print(
                f"⚠️  Dataset path is required but missing for {job_id}; falling back to batch size {fallback}."
            )
            entry = {
                "batch": fallback,
                "last_error": "dataset path unavailable",
                "last_probe": time.time(),
            }
            expected_samples = self._expected_samples_per_task(
                fallback,
                dataset_questions=question_count,
                samples_per_task_flag=samples_per_task_flag,
            )
            if expected_samples is not None:
                entry["samples_per_task"] = expected_samples
            model_cache[gpu] = entry
            self._persist_cache()
            return fallback

        probe_env = env.copy()
        probe_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        probe_env["CUDA_VISIBLE_DEVICES"] = gpu

        joined = ",".join(str(value) for value in candidates)
        print(f"🔍 Probing batch size for {job_id} on cuda:{gpu} (candidates: {joined})")

        for candidate in candidates:
            command = list(self.command_prefix) + [job.module]
            command.extend(
                [
                    "--model-path",
                    str(model_path),
                    "--device",
                    "cuda:0",
                    batch_flag,
                    str(candidate),
                ]
            )
            derived_samples = None
            expected_samples_value = None
            if samples_per_task_flag:
                derived_samples = derive_auto_samples_per_task(candidate, question_count)
                expected_samples_value = derived_samples if derived_samples is not None else 1
                if derived_samples is not None:
                    command.extend([samples_per_task_flag, str(expected_samples_value)])

            if dataset_path is not None:
                command.extend(["--dataset", str(dataset_path)])
            if job.probe_max_generate_flag:
                command.extend([job.probe_max_generate_flag, str(self.probe_max_generate)])
            if job.probe_extra_args:
                command.extend(job.probe_extra_args)
            if job.extra_args:
                command.extend(job.extra_args)

            start_time = time.time()
            proc = subprocess.run(
                command,
                cwd=str(REPO_ROOT),
                env=probe_env,
                capture_output=True,
                text=True,
            )
            elapsed = max(0.0, time.time() - start_time)

            stdout = (proc.stdout or "").strip()
            stderr = (proc.stderr or "").strip()
            combined_lower = f"{stdout}\n{stderr}".lower()

            if proc.returncode == 0:
                entry = {
                    "batch": candidate,
                    "last_probe": time.time(),
                }
                if samples_per_task_flag:
                    if expected_samples_value is None:
                        expected_samples_value = 1
                    entry["samples_per_task"] = expected_samples_value
                model_cache[gpu] = entry
                self._persist_cache()
                print(f"✅ Batch size {candidate} works for {job_id} on cuda:{gpu} (probe {elapsed:.1f}s).")
                return candidate

            message = stderr or stdout or f"exit code {proc.returncode}"
            if "out of memory" in combined_lower or "cuda oom" in combined_lower:
                print(
                    f"⚠️  Batch size {candidate} hit OOM for {job_id} on cuda:{gpu}; trying smaller candidate."
                )
                model_cache[gpu] = {
                    "last_error": f"oom at {candidate}: {message[:200]}",
                    "last_probe": time.time(),
                }
                self._persist_cache()
                continue

            print(f"❌ Batch size {candidate} probe failed for {job_id} on cuda:{gpu} (exit {proc.returncode}).")
            if stdout:
                print("   probe stdout:")
                for line in stdout.splitlines():
                    print(f"     {line}")
            if stderr:
                print("   probe stderr:")
                for line in stderr.splitlines():
                    print(f"     {line}")
            raise RuntimeError(f"probe failed for {job_id} on cuda:{gpu}: {message}")

    def invalidate_cache(self, job_name: str, model_slug: str, gpu: str, reason: str | None = None) -> None:
        """Mark a cached batch as invalid so the next run will re-probe."""

        self._refresh_cache_if_stale()
        job_cache = self._cache.setdefault(job_name, {})
        model_cache = job_cache.setdefault(model_slug, {})
        entry: dict[str, Any]
        existing = model_cache.get(gpu)
        if isinstance(existing, Mapping):
            entry = dict(existing)
        else:
            entry = {}
        entry.pop("batch", None)
        if reason:
            entry["last_error"] = reason[:200]
        entry["last_probe"] = time.time()
        model_cache[gpu] = entry
        self._persist_cache()

    def _select_probe_candidates(
        self,
        candidates: Sequence[int],
        *,
        question_count: int | None,
        sweep_enabled: bool,
    ) -> tuple[int, ...]:
        ordered = [value for value in candidates if isinstance(value, int) and value > 0]
        if not ordered:
            return tuple()
        if sweep_enabled:
            return tuple(ordered)
        if question_count is None or question_count <= 0:
            return tuple(ordered)
        max_candidate = max(ordered)
        if question_count >= max_candidate:
            return tuple(ordered)
        start = max(1, int(question_count))
        seen: set[int] = set()
        sequence: list[int] = []

        def _add(value: int | None) -> None:
            if value is None or value <= 0 or value in seen:
                return
            sequence.append(value)
            seen.add(value)

        _add(start)
        threshold = start
        fallback = self._previous_power_of_two(start)
        if fallback is not None:
            _add(fallback)
            threshold = fallback
        for candidate in ordered:
            if candidate < threshold:
                _add(candidate)
        return tuple(sequence)

    @staticmethod
    def _previous_power_of_two(value: int) -> int | None:
        if value <= 1:
            return None
        bits = (value - 1).bit_length() - 1
        if bits < 0:
            return None
        return 1 << bits

    def _expected_samples_per_task(
        self,
        batch_size: int | None,
        *,
        dataset_questions: int | None,
        samples_per_task_flag: str | None,
    ) -> int | None:
        if batch_size is None or not samples_per_task_flag:
            return None
        derived = derive_auto_samples_per_task(batch_size, dataset_questions)
        return derived if derived is not None else 1


__all__ = ["BatchProfiler", "load_batch_cache", "save_batch_cache", "update_batch_cache_locked"]
