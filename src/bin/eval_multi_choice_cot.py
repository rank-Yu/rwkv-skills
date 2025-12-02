from __future__ import annotations

"""Run chain-of-thought multiple-choice evaluation for RWKV models."""

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence
import uuid

from src.eval.datasets.data_loader.multiple_choice import JsonlMultipleChoiceLoader
from src.eval.metrics.multi_choice import evaluate_predictions, load_predictions
from src.eval.results.layout import jsonl_path, write_scores_json
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.multi_choice import MultipleChoicePipeline, COT_SAMPLING
from src.infer.model import ModelLoadConfig


PROBE_MAX_SAMPLES = 1
PROBE_COT_MAX_TOKENS = 256


def _make_probe_output_path(suffix: str = ".jsonl") -> Path:
    temp_root = Path(tempfile.gettempdir())
    return temp_root / f"rwkv_probe_{uuid.uuid4().hex}{suffix}"


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=True, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV multiple-choice CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation/scoring")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--target-token-format", default=" <LETTER>", help="Token format for answer tokens")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Compatibility flag (no-op).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset)
    except Exception as exc:
        print(f"❌ 数据集准备失败: {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    out_path = _resolve_output_path(str(dataset_path), args.model_path, args.output)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = MultipleChoicePipeline(config, target_token_format=args.target_token_format)

    # Quick validation of dataset readability before heavy model init
    _ = JsonlMultipleChoiceLoader(str(dataset_path)).load()

    probe_only = bool(args.probe_only)
    sample_limit: int | None = args.max_samples
    cot_sampling = COT_SAMPLING
    output_path = out_path
    probe_output_path: Path | None = None
    if probe_only:
        sample_limit = PROBE_MAX_SAMPLES
        cot_sampling = cot_sampling.clamp(PROBE_COT_MAX_TOKENS)
        probe_output_path = _make_probe_output_path(out_path.suffix or ".jsonl")
        output_path = probe_output_path

    result = pipeline.run_chain_of_thought(
        dataset_path=str(dataset_path),
        output_path=str(output_path),
        cot_sampling=cot_sampling,
        batch_size=max(1, args.batch_size),
        sample_limit=sample_limit,
    )

    if probe_only:
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        if probe_output_path:
            probe_output_path.unlink(missing_ok=True)
        return 0

    preds = load_predictions(output_path)
    metrics = evaluate_predictions(preds)
    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics={"accuracy": metrics.accuracy},
        samples=len(preds),
        log_path=out_path,
        task="multiple_choice_cot",
        task_details={"accuracy_by_subject": metrics.score_by_subject},
    )
    print(f"✅ CoT multiple-choice done: {result.sample_count} samples -> {result.output_path}")
    print(f"📊 scores saved: {score_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
