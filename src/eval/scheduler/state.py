from __future__ import annotations

"""State tracking helpers for dispatcher."""

import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .dataset_utils import canonical_slug, infer_dataset_slug_from_path, safe_slug
from .jobs import JOB_CATALOGUE, JobSpec, detect_job_from_dataset


@dataclass(frozen=True)
class CompletedKey:
    job: str
    model_slug: str
    dataset_slug: str
    is_cot: bool


@dataclass(frozen=True)
class CompletedRecord:
    job_id: str
    key: CompletedKey
    log_path: Path
    dataset_path: str
    model_name: str


@dataclass
class RunningEntry:
    pid: int
    gpu: str | None
    log_path: Path | None = None


def scan_completed_jobs(log_dir: Path) -> tuple[set[CompletedKey], dict[str, CompletedRecord]]:
    completed: set[CompletedKey] = set()
    records: dict[str, CompletedRecord] = {}
    if not log_dir.exists():
        return completed, records

    for json_path in log_dir.glob("*.json"):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, Mapping):
            continue

        model_name = raw.get("model") or raw.get("raw_model")
        dataset_path = raw.get("dataset")
        if not isinstance(model_name, str) or not isinstance(dataset_path, str):
            continue

        dataset_slug = infer_dataset_slug_from_path(dataset_path)
        is_cot = _detect_is_cot(json_path, raw)
        job_name = detect_job_from_dataset(dataset_slug, is_cot)
        if not job_name:
            continue
        key = CompletedKey(
            job=job_name,
            model_slug=safe_slug(model_name),
            dataset_slug=dataset_slug,
            is_cot=is_cot,
        )
        completed.add(key)
        records[json_path.stem] = CompletedRecord(
            job_id=json_path.stem,
            key=key,
            log_path=json_path,
            dataset_path=dataset_path,
            model_name=model_name,
        )
    return completed, records


def _detect_is_cot(log_path: Path, payload: Mapping[str, object]) -> bool:
    if any(
        key in payload
        for key in (
            "cot_generation_template",
            "final_answer_generation_template",
            "cot_generation_input_example",
            "answer_generation_input_example",
        )
    ):
        return True
    return "cot" in log_path.name.lower()


def load_running(pid_dir: Path) -> dict[str, RunningEntry]:
    running: dict[str, RunningEntry] = {}
    if not pid_dir.exists():
        return running
    for pid_file in pid_dir.glob("*.pid"):
        job_id = pid_file.stem
        lines = pid_file.read_text().splitlines()
        if not lines:
            continue
        try:
            pid = int(lines[0])
        except ValueError:
            continue
        gpu = lines[1].strip() if len(lines) > 1 and lines[1].strip() else None
        log_path: Path | None = None
        if len(lines) > 2 and lines[2].strip():
            log_path = Path(lines[2].strip())
        running[job_id] = RunningEntry(pid=pid, gpu=gpu, log_path=log_path)
    return running


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_pid_file(pid_dir: Path, job_id: str, pid: int, gpu: str | None, log_name: str) -> None:
    pid_dir.mkdir(parents=True, exist_ok=True)
    payload = [str(pid), gpu or "", log_name]
    (pid_dir / f"{job_id}.pid").write_text("\n".join(payload), encoding="utf-8")


def stop_job(job_id: str, pid_dir: Path) -> None:
    pid_file = pid_dir / f"{job_id}.pid"
    if not pid_file.exists():
        print(f"ℹ️  {job_id} 未找到 PID 文件")
        return
    lines = pid_file.read_text().splitlines()
    if not lines:
        pid_file.unlink(missing_ok=True)
        return
    try:
        pid = int(lines[0])
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return
    if pid_alive(pid):
        print(f"⏹  停止 {job_id} (pid={pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    else:
        print(f"ℹ️  PID {pid} 已退出")
    pid_file.unlink(missing_ok=True)


def stop_all_jobs(pid_dir: Path) -> None:
    running = load_running(pid_dir)
    if not running:
        return
    for job_id in sorted(running.keys()):
        stop_job(job_id, pid_dir)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def tail_file(path: Path, tail_lines: int) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()[-tail_lines:]
    except Exception:
        return []


def terminate_process(pid: int) -> None:
    if pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


__all__ = [
    "CompletedKey",
    "CompletedRecord",
    "RunningEntry",
    "scan_completed_jobs",
    "load_running",
    "pid_alive",
    "write_pid_file",
    "stop_job",
    "stop_all_jobs",
    "ensure_dirs",
    "tail_file",
]
