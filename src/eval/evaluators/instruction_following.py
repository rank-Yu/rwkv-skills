from __future__ import annotations

"""Instruction-following 评估流水线：生成响应 + 导出 JSONL。"""

from dataclasses import dataclass, replace
from pathlib import Path

from src.eval.datasets.data_loader.instruction_following import (
    JsonlInstructionFollowingLoader,
)
from src.eval.datasets.data_struct.instruction_following import (
    InstructionFollowingRecord,
)
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord

DEFAULT_STOP_TOKENS = (0, 261, 24281)
DEFAULT_BAN_TOKEN = 295
DEFAULT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.3,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.99,
    stop_tokens=DEFAULT_STOP_TOKENS,
)


@dataclass(slots=True)
class InstructionFollowingPipelineResult:
    dataset: str
    sample_count: int
    output_path: Path


class InstructionFollowingPipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

    def run(
        self,
        dataset_path: str,
        output_path: str,
        *,
        sampling: SamplingConfig = DEFAULT_SAMPLING,
        batch_size: int = 128,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        enable_think: bool = False,
        stop_tokens: tuple[int, ...] = DEFAULT_STOP_TOKENS,
        ban_tokens: tuple[int, ...] | None = None,
    ) -> InstructionFollowingPipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        if not records:
            return InstructionFollowingPipelineResult(dataset_name, 0, Path(output_path))

        prompts = [self._make_prompt(record.prompt, enable_think) for record in records]
        effective_ban = ban_tokens
        if effective_ban is None:
            effective_ban = () if enable_think else (DEFAULT_BAN_TOKEN,)

        sampling_cfg = replace(sampling, stop_tokens=stop_tokens, ban_tokens=effective_ban)

        outputs = self.engine.generate(
            prompts,
            sampling=sampling_cfg,
            batch_size=max(1, min(batch_size, len(records))),
            progress_desc="Generating instruction-following responses",
        )
        output_by_idx = {item.prompt_index: item for item in outputs}

        writer = JsonlStageWriter(output_path)
        for idx, record in enumerate(records):
            seq = output_by_idx.get(idx)
            if seq is None:
                continue
            cleaned = seq.text.split("</think>")[-1].strip() if enable_think else seq.text.strip()
            metadata = {
                "key": record.key,
                "instruction_ids": record.instruction_ids,
                "kwargs": record.kwargs_list,
                "prompt": record.prompt,
                "response_clean": cleaned,
            }
            stage = StageRecord(
                prompt=prompts[idx],
                output=seq.text,
                finish_reason=seq.finish_reason,
            )
            writer.write(
                SampleRecord(
                    index=idx,
                    dataset=dataset_name,
                    stages=[stage],
                    metadata=metadata,
                )
            )
        writer.close()
        return InstructionFollowingPipelineResult(dataset_name, len(records), Path(output_path))

    def _make_prompt(self, prompt: str, enable_think: bool) -> str:
        suffix = " <think" if enable_think else ""
        return f"User: {prompt}\n\nAssistant:{suffix}"

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[InstructionFollowingRecord], str]:
        loader = JsonlInstructionFollowingLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, Path(dataset_path).stem


__all__ = ["InstructionFollowingPipeline", "InstructionFollowingPipelineResult"]
