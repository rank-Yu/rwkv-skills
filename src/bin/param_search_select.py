from __future__ import annotations

"""Select the best param-search grid point across multiple benchmarks and promote artifacts.

This script reads per-trial score JSONs under:
  results/param_search/scores/{model}/{benchmark}/trial_{trial}.json

Then it selects the best shared grid point (by summed objective), and promotes the
corresponding trial artifacts into the canonical results layout:
  results/completions, results/eval, results/scores
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from src.eval.results.layout import (
    eval_details_path,
    jsonl_path,
    scores_path,
    write_scores_json_to_path,
)
from src.eval.scheduler.config import REPO_ROOT, RESULTS_ROOT
from src.eval.scheduler.dataset_utils import canonical_slug, safe_slug


DEFAULT_BENCHMARKS = ("gsm8k_test", "hendrycks_math_test")


def _resolve_path(path_value: str | Path) -> Path:
    raw = Path(path_value).expanduser() if not isinstance(path_value, Path) else path_value.expanduser()
    if raw.is_absolute():
        return raw
    return (REPO_ROOT / raw).resolve()


def _objective_from_metrics(metrics: dict[str, Any]) -> float:
    judge = metrics.get("judge_accuracy")
    if isinstance(judge, (int, float)):
        return float(judge)
    exact = metrics.get("exact_accuracy")
    if isinstance(exact, (int, float)):
        return float(exact)
    return 0.0


def _param_key_from_payload(payload: dict[str, Any]) -> str | None:
    task_details = payload.get("task_details")
    if isinstance(task_details, dict):
        trial_info = task_details.get("param_search_trial")
        if isinstance(trial_info, dict):
            params = trial_info.get("params")
            if isinstance(params, dict):
                return json.dumps(params, sort_keys=True, ensure_ascii=False)
    return None


def _iter_trial_score_files(model_name: str, benchmark: str) -> list[Path]:
    model_dir = safe_slug(model_name)
    bench_dir = canonical_slug(benchmark)
    root = RESULTS_ROOT / "param_search" / "scores" / model_dir / bench_dir
    if not root.exists():
        return []
    return sorted(root.glob("trial_*.json"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV param-search selector/promoter")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", help="Ignored (scheduler compatibility)")
    parser.add_argument("--device", help="Ignored (scheduler compatibility)")
    parser.add_argument("--output", help="Optional JSONL marker output path (scheduler compatibility)")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(DEFAULT_BENCHMARKS),
        help="Benchmarks to aggregate (default: gsm8k_test hendrycks_math_test)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite promoted artifacts under results/{completions,eval,scores}.",
    )
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _promote_trial(
    *,
    trial_score_path: Path,
    benchmark: str,
    model_name: str,
    overwrite: bool,
) -> None:
    payload = _load_json(trial_score_path)
    source_completion = _resolve_path(str(payload.get("log_path", "")))
    task_details = payload.get("task_details") if isinstance(payload.get("task_details"), dict) else {}
    eval_details_value = task_details.get("eval_details_path") if isinstance(task_details, dict) else None
    if not isinstance(eval_details_value, str):
        raise ValueError(f"score JSON missing task_details.eval_details_path: {trial_score_path}")
    source_eval = _resolve_path(eval_details_value)

    dest_completion = jsonl_path(benchmark, is_cot=True, model_name=model_name)
    dest_eval = eval_details_path(benchmark, is_cot=True, model_name=model_name)
    dest_score = scores_path(benchmark, is_cot=True, model_name=model_name)

    for src, dst in ((source_completion, dest_completion), (source_eval, dest_eval)):
        if not src.exists():
            raise FileNotFoundError(f"missing source artifact: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)

    # Rewrite score JSON so canonical artifacts are self-contained.
    if dest_score.exists() and not overwrite:
        return

    payload["log_path"] = str(dest_completion)
    if isinstance(payload.get("task_details"), dict):
        payload["task_details"]["eval_details_path"] = str(dest_eval)

    dest_score.parent.mkdir(parents=True, exist_ok=True)
    write_scores_json_to_path(dest_score, payload)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model_name = Path(args.model_path).stem
    benchmarks = tuple(canonical_slug(b) for b in args.benchmarks if b)
    if len(benchmarks) < 2:
        print("❌ 需要至少 2 个 benchmark 才能进行综合选参。")
        return 2

    by_benchmark: dict[str, dict[str, tuple[float, Path]]] = {}
    for bench in benchmarks:
        files = _iter_trial_score_files(model_name, bench)
        if not files:
            print(f"❌ 缺少 param-search trial scores: {bench}")
            return 1
        mapping: dict[str, tuple[float, Path]] = {}
        for path in files:
            payload = _load_json(path)
            metrics = payload.get("metrics")
            if not isinstance(metrics, dict):
                continue
            key = _param_key_from_payload(payload)
            if key is None:
                continue
            mapping[key] = (_objective_from_metrics(metrics), path)
        if not mapping:
            print(f"❌ 未能从 {bench} 解析出任何有效 trial score")
            return 1
        by_benchmark[bench] = mapping

    common_keys = set.intersection(*(set(m.keys()) for m in by_benchmark.values()))
    if not common_keys:
        print("❌ 各 benchmark 的 grid 点没有交集（params key 不一致）")
        return 1

    best_key: str | None = None
    best_sum: float | None = None
    best_detail: dict[str, float] = {}
    best_paths: dict[str, Path] = {}

    for key in sorted(common_keys):
        total = 0.0
        detail: dict[str, float] = {}
        paths: dict[str, Path] = {}
        for bench, mapping in by_benchmark.items():
            score, path = mapping[key]
            total += float(score)
            detail[bench] = float(score)
            paths[bench] = path
        if best_sum is None or total > best_sum:
            best_sum = total
            best_key = key
            best_detail = detail
            best_paths = paths

    if best_key is None or best_sum is None:
        print("❌ 未找到可用的最佳 grid 点")
        return 1

    print("✅ best param-search selection:")
    print(f"    model: {model_name}")
    print(f"    total: {best_sum:.6f}")
    for bench in benchmarks:
        print(f"    {bench}: {best_detail.get(bench, 0.0):.6f}")
    print(f"    params: {best_key}")

    for bench, score_path in best_paths.items():
        _promote_trial(
            trial_score_path=score_path,
            benchmark=bench,
            model_name=model_name,
            overwrite=bool(args.overwrite),
        )

    # Optional marker output for scheduler/debugging.
    if args.output and not os.environ.get("RWKV_SKILLS_JOB_ID"):
        out_path = Path(args.output).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": model_name,
            "benchmarks": list(benchmarks),
            "objective_sum": float(best_sum),
            "objective_by_benchmark": best_detail,
            "params": json.loads(best_key),
        }
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
