from __future__ import annotations

"""Shared helper structures for evaluator pipelines (JSONL 输出 & 调试工具)."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class ProbeConfig:
    """用于评估阶段的轻量化探测：限制样本数/生成长度。"""

    max_samples: int | None = None
    max_tokens: int | None = None

    def apply(self, records: Sequence) -> list:
        if not self.max_samples or self.max_samples <= 0:
            return list(records)
        return list(records)[: max(1, min(self.max_samples, len(records)))]


@dataclass(slots=True)
class StageRecord:
    prompt: str
    output: str | None = None
    finish_reason: str | None = None
    logits: dict[str, float] | None = None


@dataclass(slots=True)
class SampleRecord:
    index: int
    dataset: str
    stages: list[StageRecord]
    metadata: dict = field(default_factory=dict)

    def as_payload(self) -> dict:
        payload = {"sample_index": self.index, "dataset": self.dataset}
        for idx, stage in enumerate(self.stages, start=1):
            payload[f"prompt{idx}"] = stage.prompt
            if stage.output is not None:
                payload[f"output{idx}"] = stage.output
            if stage.finish_reason is not None:
                payload[f"finish_reason{idx}"] = stage.finish_reason
            if stage.logits is not None:
                payload[f"logits{idx}"] = stage.logits
        payload.update(self.metadata)
        return payload


class JsonlStageWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, record: SampleRecord) -> None:
        self._fh.write(json.dumps(record.as_payload(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "JsonlStageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass(slots=True)
class DebugCaptureConfig:
    path: Path
    limit: int
    exit_on_limit: bool = False

    @property
    def enabled(self) -> bool:
        return self.limit > 0 and str(self.path)

    @classmethod
    def from_env(cls) -> DebugCaptureConfig | None:
        raw_path = os.environ.get("RUN_DEBUG_CAPTURE_PATH")
        limit = _env_int(os.environ.get("RUN_DEBUG_CAPTURE_LIMIT"), 0)
        exit_flag = _env_flag(os.environ.get("RUN_DEBUG_CAPTURE_EXIT"))
        if not raw_path or limit <= 0:
            return None
        return cls(path=Path(raw_path).expanduser(), limit=limit, exit_on_limit=exit_flag)


@dataclass(slots=True)
class DebugCaptureBuffer:
    config: DebugCaptureConfig
    records: list[dict] = field(default_factory=list)

    def add(self, record: dict) -> bool:
        if not self.config.enabled:
            return False
        if len(self.records) >= self.config.limit:
            return True
        self.records.append(record)
        return len(self.records) >= self.config.limit

    def flush(self, **metadata) -> None:
        if not self.config.enabled:
            return
        payload = {
            **metadata,
            "limit": self.config.limit,
            "collected": len(self.records),
            "records": self.records,
        }
        target = self.config.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=4)


def _env_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _env_flag(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}


__all__ = [
    "ProbeConfig",
    "StageRecord",
    "SampleRecord",
    "JsonlStageWriter",
    "DebugCaptureConfig",
    "DebugCaptureBuffer",
]
