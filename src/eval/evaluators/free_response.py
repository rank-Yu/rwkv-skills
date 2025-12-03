from __future__ import annotations

"""Free-form QA 评估流水线：读数据 -> 两阶段生成 -> JSONL 导出。"""

from dataclasses import dataclass
from pathlib import Path

from src.eval.datasets.data_loader.free_answer import JsonlFreeAnswerLoader
from src.eval.datasets.data_struct.free_answer import FreeAnswerRecord
from src.infer.engine import InferenceEngine
from src.infer.model import ModelLoadConfig, load_rwkv_model
from src.infer.sampling import SamplingConfig
from .common import JsonlStageWriter, SampleRecord, StageRecord, detect_resume_state

DEFAULT_COT_PROMPT = """User: <Q>

Assistant: <think"""

DEFAULT_FINAL_PROMPT = """<Q><COT>
Therefore, the answer is \\(\\boxed{"""

DEFAULT_COT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.3,
    top_k=50,
    top_p=0.3,
    alpha_presence=0.5,
    alpha_frequency=0.5,
    alpha_decay=0.99,
    stop_tokens=(0, 261, 24281),
)

DEFAULT_FINAL_SAMPLING = SamplingConfig(
    max_generate_tokens=64,
    temperature=1.0,
    top_k=1,
    top_p=0.3,
    alpha_presence=0.0,
    alpha_frequency=0.0,
    alpha_decay=0.99,
    stop_tokens=(0, 2402, 4910),
)


@dataclass(slots=True)
class FreeResponsePipelineResult:
    dataset: str
    sample_count: int
    output_path: Path


class FreeResponsePipeline:
    def __init__(self, model_config: ModelLoadConfig) -> None:
        self.model, self.tokenizer = load_rwkv_model(model_config)
        self.engine = InferenceEngine(self.model, self.tokenizer)

    def run(
        self,
        dataset_path: str,
        output_path: str,
        *,
        cot_prompt_template: str = DEFAULT_COT_PROMPT,
        final_answer_template: str = DEFAULT_FINAL_PROMPT,
        cot_sampling: SamplingConfig = DEFAULT_COT_SAMPLING,
        final_sampling: SamplingConfig = DEFAULT_FINAL_SAMPLING,
        batch_size: int = 64,
        dataset_name: str | None = None,
        sample_limit: int | None = None,
        pad_to_batch: bool = False,
    ) -> FreeResponsePipelineResult:
        records, resolved_name = self._load_records(dataset_path, sample_limit)
        dataset_name = dataset_name or resolved_name
        if pad_to_batch and records and len(records) < batch_size:
            # 对探测运行：当样本不足以凑满期望的 batch 时重复记录，确保真正以目标 batch 探测显存。
            original_len = len(records)
            repeat = (batch_size + original_len - 1) // original_len
            records = (records * repeat)[:batch_size]
        if not records:
            return FreeResponsePipelineResult(dataset_name, 0, Path(output_path))

        resume = detect_resume_state(output_path)
        start_index = min(resume.next_index, len(records))
        if start_index and len(records):
            remaining = max(len(records) - start_index, 0)
            print(f"⏩ 自由问答恢复运行：已完成 {start_index}/{len(records)}，剩余 {remaining}")
        remaining_records = records[start_index:]
        if not remaining_records:
            return FreeResponsePipelineResult(dataset_name, len(records), Path(output_path))

        writer = JsonlStageWriter(output_path, resume=resume.has_progress)
        cot_prompts = [cot_prompt_template.replace("<Q>", record.question) for record in remaining_records]
        cot_outputs = self.engine.generate(
            cot_prompts,
            sampling=cot_sampling,
            batch_size=batch_size,
            progress_desc="Generating CoT",
        )
        cot_by_idx = {item.prompt_index: item for item in cot_outputs}

        final_prompts: list[str] = []
        for local_idx, _ in enumerate(remaining_records):
            cot_seq = cot_by_idx.get(local_idx)
            cot_text = cot_seq.text if cot_seq else ""
            prompt = final_answer_template.replace("<Q>", cot_prompts[local_idx]).replace("<COT>", cot_text)
            final_prompts.append(prompt)

        final_outputs = self.engine.generate(
            final_prompts,
            sampling=final_sampling,
            batch_size=batch_size,
            progress_desc="Generating answers",
        )
        final_by_idx = {item.prompt_index: item for item in final_outputs}

        for local_idx, record in enumerate(remaining_records):
            global_idx = start_index + local_idx
            cot_seq = cot_by_idx.get(local_idx)
            ans_seq = final_by_idx.get(local_idx)
            if cot_seq is None or ans_seq is None:
                continue
            prediction = ans_seq.text.strip()
            stages = [
                StageRecord(
                    prompt=cot_prompts[local_idx],
                    output=cot_seq.text,
                    finish_reason=cot_seq.finish_reason,
                ),
                StageRecord(
                    prompt=final_prompts[local_idx],
                    output=ans_seq.text,
                    finish_reason=ans_seq.finish_reason,
                ),
            ]
            metadata = {
                "question": record.question,
                "answer": record.answer,
                "prediction": prediction,
                "subject": record.subject,
                "correct_exact": prediction == record.answer,
            }
            writer.write(
                SampleRecord(
                    index=global_idx,
                    dataset=dataset_name,
                    stages=stages,
                    metadata=metadata,
                )
            )
        writer.close()
        return FreeResponsePipelineResult(dataset_name, len(records), Path(output_path))

    def _load_records(
        self, dataset_path: str, sample_limit: int | None
    ) -> tuple[list[FreeAnswerRecord], str]:
        loader = JsonlFreeAnswerLoader(dataset_path)
        dataset = loader.load()
        records = list(dataset)
        if sample_limit is not None and sample_limit > 0:
            records = records[: min(sample_limit, len(records))]
        return records, Path(dataset_path).stem


__all__ = ["FreeResponsePipeline", "FreeResponsePipelineResult"]
