from __future__ import annotations

"""Run chain-of-thought free-form QA evaluation for RWKV models."""

import argparse
import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
import shutil
from typing import Sequence
import uuid

import optuna

from src.eval.metrics.free_response import (
    FreeResponseMetrics,
    compute_pass_at_k,
    evaluate_exact,
    load_samples,
    write_sample_results,
)
from src.eval.results.layout import (
    eval_details_path,
    jsonl_path,
    param_search_records_path,
    param_search_trial_path,
    write_scores_json,
)
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path
from src.eval.evaluators.free_response import (
    FreeResponsePipeline,
    FreeResponsePipelineResult,
    DEFAULT_COT_SAMPLING,
    DEFAULT_FINAL_SAMPLING,
)
from src.infer.model import ModelLoadConfig


PROBE_MIN_SAMPLES = 1
PROBE_COT_MAX_TOKENS = 256
PROBE_FINAL_MAX_TOKENS = 64
DEFAULT_PARAM_SEARCH_TRIALS = 30
PASS_K_LEVELS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

PARAM_SEARCH_DATASETS = {
    "aime24_test",
    "aime25_test",
    "beyond_aime_test",
    "hmmt_feb25_test",
    "brumo25_test",
}


def _round_float(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


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
    parser = argparse.ArgumentParser(description="RWKV free-form CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Disable automatic sampling parameter search even for datasets that enable it by default.",
    )
    parser.add_argument(
        "--param-search",
        action="store_true",
        help="Force-enable sampling parameter search for this run.",
    )
    parser.add_argument(
        "--param-search-trials",
        type=int,
        help=f"Number of Optuna trials to run when parameter search is enabled (default {DEFAULT_PARAM_SEARCH_TRIALS}).",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help="Number of completions to generate per problem (default 1)",
    )
    return parser.parse_args(argv)


def _sampling_config_to_dict(config: SamplingConfig) -> dict[str, object]:
    data = asdict(config)
    normalized: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, tuple):
            normalized[key] = [
                _round_float(item) if isinstance(item, float) else item for item in value
            ]
        elif isinstance(value, float):
            normalized[key] = _round_float(value)
        else:
            normalized[key] = value
    return normalized


def _suggest_sampling_configs(
    trial: optuna.Trial,
    base_cot: SamplingConfig,
    base_final: SamplingConfig,
) -> tuple[SamplingConfig, SamplingConfig]:
    cot = replace(
        base_cot,
        temperature=_round_float(trial.suggest_float("cot_temperature", 0.15, 0.75)),
        top_k=trial.suggest_categorical("cot_top_k", [32, 40, 48, 64, 80, 96]),
        top_p=_round_float(trial.suggest_float("cot_top_p", 0.2, 0.7)),
        alpha_presence=_round_float(trial.suggest_float("cot_alpha_presence", 0.3, 0.8)),
        alpha_frequency=_round_float(trial.suggest_float("cot_alpha_frequency", 0.3, 0.8)),
    )
    final = replace(
        base_final,
        temperature=_round_float(trial.suggest_float("final_temperature", 0.3, 1.2)),
        top_k=trial.suggest_categorical("final_top_k", [1, 2, 4, 8]),
        top_p=_round_float(trial.suggest_float("final_top_p", 0.1, 0.75)),
    )
    return cot, final


def _should_enable_param_search(args: argparse.Namespace, dataset_slug: str) -> bool:
    if getattr(args, "param_search", False):
        return True
    if getattr(args, "no_param_search", False):
        return False
    if os.environ.get("RWKV_SKILLS_DISABLE_PARAM_SEARCH") == "1":
        return False
    return dataset_slug in PARAM_SEARCH_DATASETS


def _resolve_param_search_trials(args: argparse.Namespace) -> int:
    if args.param_search_trials and args.param_search_trials > 0:
        return args.param_search_trials
    env_val = os.environ.get("RWKV_PARAM_SEARCH_TRIALS")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return DEFAULT_PARAM_SEARCH_TRIALS


def _round_params_dict(data: dict[str, object]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, float):
            rounded[key] = _round_float(value)
        else:
            rounded[key] = value
    return rounded


def _run_single_eval(
    pipeline: FreeResponsePipeline,
    dataset_path: str,
    output_path: Path,
    *,
    cot_sampling: SamplingConfig,
    final_sampling: SamplingConfig,
    batch_size: int,
    sample_limit: int | None,
    pad_to_batch: bool,
    samples_per_task: int,
):
    result = pipeline.run(
        dataset_path=dataset_path,
        output_path=str(output_path),
        cot_sampling=cot_sampling,
        final_sampling=final_sampling,
        batch_size=batch_size,
        sample_limit=sample_limit,
        pad_to_batch=pad_to_batch,
        samples_per_task=samples_per_task,
    )
    samples = load_samples(output_path)
    metrics = evaluate_exact(samples)
    return result, metrics


def _run_param_search(
    pipeline: FreeResponsePipeline,
    dataset_path: str,
    base_output_path: Path,
    dataset_slug: str,
    model_name: str,
    *,
    cot_sampling: SamplingConfig,
    final_sampling: SamplingConfig,
    batch_size: int,
    sample_limit: int | None,
    trials: int,
    samples_per_task: int,
) -> tuple[FreeResponsePipelineResult, FreeResponseMetrics, dict[str, object]]:
    if trials <= 0:
        raise ValueError("参数扫描 trials 必须大于 0")

    search_records: list[dict[str, object]] = []
    trial_results: dict[int, tuple[FreeResponsePipelineResult, FreeResponseMetrics, Path]] = {}

    def objective(trial: optuna.Trial) -> float:
        cot_cfg, final_cfg = _suggest_sampling_configs(trial, cot_sampling, final_sampling)
        idx = trial.number + 1
        candidate_path = param_search_trial_path(
            dataset_slug,
            is_cot=True,
            model_name=model_name,
            trial_index=idx,
        )
        candidate_path.unlink(missing_ok=True)
        print(f"🔍 参数扫描 trial {idx}/{trials}: {candidate_path.name}")
        result = pipeline.run(
            dataset_path=dataset_path,
            output_path=str(candidate_path),
            cot_sampling=cot_cfg,
            final_sampling=final_cfg,
            batch_size=batch_size,
            sample_limit=sample_limit,
            samples_per_task=samples_per_task,
        )
        samples = load_samples(candidate_path)
        metrics = evaluate_exact(samples)
        record = {
            "trial": idx,
            "exact_accuracy": metrics.exact_accuracy,
            "samples": len(samples),
            "log_path": str(candidate_path),
            "cot_sampling": _sampling_config_to_dict(cot_cfg),
            "final_sampling": _sampling_config_to_dict(final_cfg),
        }
        search_records.append(record)
        trial_results[trial.number] = (result, metrics, candidate_path)
        trial.set_user_attr("log_path", str(candidate_path))
        trial.set_user_attr("cot_sampling", record["cot_sampling"])
        trial.set_user_attr("final_sampling", record["final_sampling"])
        trial.set_user_attr("samples", len(samples))
        return metrics.exact_accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_trial
    best_tuple = trial_results.get(best.number)
    if not best_tuple:
        raise RuntimeError("Optuna 未返回最佳 trial 结果")
    best_result, best_metrics, best_path = best_tuple
    if base_output_path.exists():
        base_output_path.unlink(missing_ok=True)
    if best_path != base_output_path:
        shutil.copy2(best_path, base_output_path)
    best_result.output_path = base_output_path

    records_path = param_search_records_path(dataset_slug, is_cot=True, model_name=model_name)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as fh:
        for row in search_records:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")

    summary = {
        "enabled": True,
        "trials": trials,
        "best_trial": best.number + 1,
        "best_exact_accuracy": _round_float(best.value),
        "best_params": _round_params_dict(best.params),
        "best_log_path": best.user_attrs.get("log_path"),
        "best_cot_sampling": best.user_attrs.get("cot_sampling"),
        "best_final_sampling": best.user_attrs.get("final_sampling"),
        "records_path": str(records_path),
    }
    return best_result, best_metrics, summary

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
    pipeline = FreeResponsePipeline(config)

    cot_sampling = DEFAULT_COT_SAMPLING.clamp(args.cot_max_tokens)
    final_sampling = DEFAULT_FINAL_SAMPLING.clamp(args.final_max_tokens)
    sample_limit: int | None = args.max_samples
    batch_size = max(1, args.batch_size)
    samples_per_task = max(1, args.samples_per_task)
    if args.probe_only:
        sample_limit = max(args.batch_size, PROBE_MIN_SAMPLES)
        cot_sampling = cot_sampling.clamp(PROBE_COT_MAX_TOKENS)
        final_sampling = final_sampling.clamp(PROBE_FINAL_MAX_TOKENS)
        probe_output_path = _make_probe_output_path(out_path.suffix or ".jsonl")
        probe_samples = samples_per_task if args.samples_per_task > 1 else 1
        result = pipeline.run(
            dataset_path=str(dataset_path),
            output_path=str(probe_output_path),
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pad_to_batch=True,
            samples_per_task=probe_samples,
        )
        print(
            "🧪 probe-only run completed: "
            f"{result.sample_count} sample(s) evaluated with batch {args.batch_size}."
        )
        if probe_output_path:
            probe_output_path.unlink(missing_ok=True)
        return 0

    metrics: FreeResponseMetrics
    param_summary: dict[str, object] | None = None
    if _should_enable_param_search(args, slug):
        param_trials = _resolve_param_search_trials(args)
        result, metrics, param_summary = _run_param_search(
            pipeline,
            dataset_path=str(dataset_path),
            base_output_path=out_path,
            dataset_slug=slug,
            model_name=Path(args.model_path).stem,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            trials=param_trials,
            samples_per_task=samples_per_task,
        )
        output_path = out_path
    else:
        result, metrics = _run_single_eval(
            pipeline,
            dataset_path=str(dataset_path),
            output_path=out_path,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pad_to_batch=False,
            samples_per_task=samples_per_task,
        )
        output_path = out_path

    pass_metrics = compute_pass_at_k(metrics.samples, PASS_K_LEVELS)
    if pass_metrics:
        metrics.pass_at_k = pass_metrics

    eval_path = eval_details_path(slug, is_cot=True, model_name=Path(args.model_path).stem)
    write_sample_results(metrics.samples, eval_path)
    task_details = {"eval_details_path": str(eval_path)}
    if param_summary:
        task_details["param_search"] = param_summary
    metrics_payload = {
        "exact_accuracy": metrics.exact_accuracy,
        "judge_accuracy": metrics.judge_accuracy,
    }
    if metrics.pass_at_k:
        metrics_payload.update(metrics.pass_at_k)

    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=metrics_payload,
        samples=result.sample_count,
        problems=result.problem_count,
        log_path=output_path,
        task="free_response",
        task_details=task_details,
    )
    print(f"✅ CoT free-form done: {result.sample_count} samples -> {result.output_path}")
    if param_summary:
        best_acc = param_summary.get("best_exact_accuracy")
        best_trial = param_summary.get("best_trial")
        best_log = param_summary.get("best_log_path")
        records_path = param_summary.get("records_path")
        if isinstance(best_acc, float):
            print(f"🔍 参数扫描最佳: trial {best_trial} (exact={best_acc:.4f}) -> {best_log}")
        else:
            print(f"🔍 参数扫描最佳: trial {best_trial} -> {best_log}")
        if records_path:
            print(f"    📁 详尽记录: {records_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
