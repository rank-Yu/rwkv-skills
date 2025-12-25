from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "results" / "completions"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "completions_datalake"
DEFAULT_GLOB = "*/*.jsonl"


def _json_loads(line: bytes | str) -> dict[str, Any]:
    try:
        import orjson  # type: ignore

        if isinstance(line, str):
            line = line.encode("utf-8", errors="strict")
        return orjson.loads(line)
    except ModuleNotFoundError:
        if isinstance(line, (bytes, bytearray)):
            line = line.decode("utf-8", errors="strict")
        return json.loads(line)


def _json_dumps(obj: Any) -> bytes:
    try:
        import orjson  # type: ignore

        return orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE)
    except ModuleNotFoundError:
        return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8", errors="strict")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield _json_loads(line)


def _pick_logits_token(logits: Any) -> str | None:
    if not isinstance(logits, dict) or not logits:
        return None

    def score(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("-inf")

    token, value = max(logits.items(), key=lambda kv: score(kv[1]))
    if score(value) == float("-inf"):
        return None
    return str(token)


def convert_record_to_emoji_chat(record: dict[str, Any]) -> list[dict[str, str]]:
    chat: list[dict[str, str]] = []

    for idx in (1, 2):
        prompt_key = f"prompt{idx}"
        output_key = f"output{idx}"
        logits_key = f"logits{idx}"

        prompt = record.get(prompt_key)
        if prompt is None and idx == 1:
            prompt = record.get("prompt")

        if prompt is None:
            continue

        chat.append({"😺": str(prompt)})

        assistant: str | None = None
        output = record.get(output_key)
        if output is not None:
            assistant = str(output)
        else:
            assistant = _pick_logits_token(record.get(logits_key))

        chat.append({"🤖": assistant if assistant is not None else ""})

    return chat


def iter_input_files(input_root: Path, pattern: str) -> list[Path]:
    return sorted(p for p in input_root.glob(pattern) if p.is_file())


def convert_file(in_path: Path, out_path: Path, *, max_lines: int | None = None) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    records = 0
    errors = 0
    with tmp_path.open("wb") as out_f:
        for record in _iter_jsonl(in_path):
            records += 1
            if max_lines is not None and records > max_lines:
                break
            try:
                chat = convert_record_to_emoji_chat(record)
                out_f.write(_json_dumps(chat))
            except Exception:  # noqa: BLE001
                errors += 1
                continue

    tmp_path.replace(out_path)
    return records, errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "遍历 results/completions/*/*.jsonl，把每条记录转换成 emoji 对话数组："
            '[{"😺":"prompt"},{"🤖":"output"}, ...]；logits-only 时取最大 logits 的 token 作为输出。'
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT), help="输入目录（默认：results/completions）")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出目录（默认：results/completions_datalake）")
    parser.add_argument("--glob", default=DEFAULT_GLOB, help='相对 input-root 的 glob（默认："*/*.jsonl"）')
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    parser.add_argument("--max-files", type=int, help="最多处理多少个文件（用于调试）")
    parser.add_argument("--max-lines", type=int, help="每个文件最多处理多少行（用于调试）")
    parser.add_argument("--quiet", action="store_true", help="减少日志输出")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    pattern = str(args.glob)

    if not input_root.exists():
        raise SystemExit(f"❌ 输入目录不存在：{input_root}")

    files = iter_input_files(input_root, pattern)
    if args.max_files is not None:
        files = files[: max(0, args.max_files)]

    if not files:
        if not args.quiet:
            print(f"⚠️  没有找到任何输入文件：{input_root}/{pattern}")
        return 0

    total_records = 0
    total_errors = 0
    processed = 0
    skipped = 0

    for in_path in files:
        rel = in_path.relative_to(input_root)
        out_path = output_root / rel
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        processed += 1
        if not args.quiet:
            print(f"➡️  {in_path} -> {out_path}")

        records, errors = convert_file(in_path, out_path, max_lines=args.max_lines)
        total_records += records
        total_errors += errors

    if not args.quiet:
        print(
            "✅ 完成："
            f"files={len(files)} processed={processed} skipped={skipped} "
            f"records={total_records} errors={total_errors} "
            f"out_root={output_root}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

