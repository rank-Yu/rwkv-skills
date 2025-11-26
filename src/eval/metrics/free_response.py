from __future__ import annotations

"""Free-form QA metrics：支持 exact match 和 LLM judge。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from openai import OpenAI


@dataclass(slots=True)
class FreeResponseSample:
    sample_index: int
    dataset: str
    question: str
    answer: str
    prediction: str
    subject: str | None
    cot: str | None


@dataclass(slots=True)
class FreeResponseSampleResult:
    sample: FreeResponseSample
    correct_exact: bool
    judge_correct: bool | None = None


@dataclass(slots=True)
class FreeResponseMetrics:
    exact_accuracy: float
    judge_accuracy: float | None
    samples: list[FreeResponseSampleResult]


def load_samples(path: str | Path) -> list[FreeResponseSample]:
    path = Path(path)
    samples: list[FreeResponseSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            payload = json.loads(line)
            dataset = payload.get("dataset", path.stem)
            sample_index = int(payload.get("sample_index", len(samples)))
            question = payload.get("question", "")
            answer = payload.get("answer", "")
            subject = payload.get("subject")
            cot = payload.get("output1")
            prediction = payload.get("prediction") or payload.get("output2") or ""
            samples.append(
                FreeResponseSample(
                    sample_index=sample_index,
                    dataset=dataset,
                    question=question,
                    answer=answer,
                    prediction=prediction.strip(),
                    subject=subject,
                    cot=cot,
                )
            )
    return samples


def evaluate_exact(samples: Iterable[FreeResponseSample]) -> FreeResponseMetrics:
    results: list[FreeResponseSampleResult] = []
    total = 0
    correct = 0
    for sample in samples:
        is_correct = sample.prediction == sample.answer
        results.append(FreeResponseSampleResult(sample, correct_exact=is_correct))
        total += 1
        if is_correct:
            correct += 1
    accuracy = correct / total if total else 0.0
    return FreeResponseMetrics(exact_accuracy=accuracy, judge_accuracy=None, samples=results)


@dataclass(slots=True)
class LLMJudgeConfig:
    api_key: str
    model: str
    base_url: str | None = None
    max_workers: int = 32
    prompt_template: str = (
        "You are a rigorous AI judge. Your task is to evaluate whether a student's "
        "answer is semantically completely equivalent to the reference answer, based on "
        "the provided question and reference answer.\n\nInput:\nQuestion: <Q>\nReference Answer: <REF>\n"
        "Student's Answer: <A>\n\nOutput Format:\nStrictly adhere to the output format: Only output 'True' or 'False'."
    )


class LLMJudge:
    def __init__(self, config: LLMJudgeConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def judge(self, samples: Iterable[FreeResponseSampleResult]) -> None:
        pending = [sample for sample in samples if sample.judge_correct is None]
        if not pending:
            return

        def worker(sample: FreeResponseSampleResult) -> bool:
            prompt = self.config.prompt_template
            prompt = prompt.replace("<Q>", sample.sample.question)
            prompt = prompt.replace("<REF>", sample.sample.answer)
            prompt = prompt.replace("<A>", sample.sample.prediction)
            response = self.client.chat.completions.create(
                model=self.config.model,
                stream=False,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.choices[0].message.content or "").strip()
            if content not in {"True", "False"}:
                raise ValueError(f"LLM judge 输出非法值: {content}")
            return content == "True"

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(worker, sample): sample for sample in pending
            }
            for future in as_completed(futures):
                sample = futures[future]
                sample.judge_correct = future.result()


def evaluate_with_judge(samples: Iterable[FreeResponseSample], judge: LLMJudge) -> FreeResponseMetrics:
    metrics = evaluate_exact(samples)
    judge.judge(metrics.samples)
    total = 0
    correct = 0
    for sample in metrics.samples:
        if sample.judge_correct is None:
            continue
        total += 1
        if sample.judge_correct:
            correct += 1
    judge_accuracy = correct / total if total else None
    metrics.judge_accuracy = judge_accuracy
    return metrics


__all__ = [
    "FreeResponseSample",
    "FreeResponseSampleResult",
    "FreeResponseMetrics",
    "load_samples",
    "evaluate_exact",
    "LLMJudge",
    "LLMJudgeConfig",
    "evaluate_with_judge",
]
