from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson
import jsonschema
from json import JSONDecodeError
from openai import OpenAI, OpenAIError

from src.eval.results.layout import check_details_path
from src.eval.scheduler.config import REPO_ROOT


CHECKER_FIELD_ANSWER_CORRECT = "answer_correct"
CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR = "instruction_following_error"
CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR = "world_knowledge_error"
CHECKER_FIELD_MATH_ERROR = "math_error"
CHECKER_FIELD_REASONING_LOGIC_ERROR = "reasoning_logic_error"
CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER = "thought_contains_correct_answer"
CHECKER_FIELD_REASON = "reason"
CHECKER_FIELD_COT = "checker_cot"

CHECKER_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        CHECKER_FIELD_ANSWER_CORRECT: {
            "type": "boolean",
            "description": "模型答案实际正确（可能是参考答案不全导致误判）。",
        },
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR: {
            "type": "boolean",
            "description": "指令遵循错误：模型未能正确理解题目意图/格式要求/约束。",
        },
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR: {
            "type": "boolean",
            "description": "世界知识错误：推理过程中引入明显不合理/错误的事实或常识。",
        },
        CHECKER_FIELD_MATH_ERROR: {
            "type": "boolean",
            "description": "数学运算错误：推理中出现不成立的算术/代数/数值计算。",
        },
        CHECKER_FIELD_REASONING_LOGIC_ERROR: {
            "type": "boolean",
            "description": "推理逻辑错误：推理链不严谨/跳步/自相矛盾。",
        },
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER: {
            "type": "boolean",
            "description": "思考过程是否包含正确答案（不一定等同于参考答案）。",
        },
        CHECKER_FIELD_REASON: {
            "type": "string",
            "description": "简要概述判定原因（面向人类阅读，短句即可）。",
        },
        CHECKER_FIELD_COT: {
            "type": "string",
            "description": "分析过程/推理过程（可分点描述，允许较长）。",
        },
    },
    "required": [
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
        CHECKER_FIELD_REASON,
        CHECKER_FIELD_COT,
    ],
}


_PROMPT_TEMPLATE = """你是一个专业的大语言模型评估任务的专家，你需要帮助我分析某大语言模型对以下题目的作答情况。

【题目与模型作答上下文】
{context}

【评估器提取到的答案】
{answer}

【参考答案】
{ref_answer}

请你分析是否有以下情况出现：
1) 答案正确（参考答案不全面导致被判为错题）
2) 指令遵循错误（模型未能正确理解题目意图）
3) 世界知识错误（推理过程中引入了明显不合理的世界知识）
4) 数学运算错误（推理过程中有明显不成立的数学运算）
5) 推理逻辑错误（推理过程不严谨）
6) 思考过程是否包含正确答案（不一定是参考答案）

要求：
- 你必须仅返回一个 JSON 对象，且必须符合固定字段名（不要输出多余字段或额外文本）。
- 字段名固定为：{field_names}
"""


def _load_env_file(path: Path) -> None:
    """Lightweight .env loader (key=value, optional quotes, ignores comments)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _truncate_text(value: str, *, max_chars: int) -> str:
    if not value:
        return ""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    # Keep both head and tail (tail often contains the final answer).
    head = max(1, int(max_chars * 0.7))
    tail = max(1, max_chars - head - 64)
    return f"{value[:head]}\n\n...[truncated {len(value) - head - tail} chars]...\n\n{value[-tail:]}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield orjson.loads(line)


def _load_existing_keys(path: Path) -> set[tuple[str, int, int]]:
    """Return {(dataset_split, sample_index, repeat_index)} already written."""
    if not path.exists():
        return set()
    keys: set[tuple[str, int, int]] = set()
    for row in _iter_jsonl(path):
        split = str(row.get("dataset_split", ""))
        sample_index = int(row.get("sample_index", 0))
        repeat_index = int(row.get("repeat_index", 0))
        keys.add((split, sample_index, repeat_index))
    return keys


def _validate_checker_payload(payload: dict[str, Any]) -> None:
    jsonschema.validate(instance=payload, schema=CHECKER_JSON_SCHEMA)


def _coerce_checker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort coercion for common provider quirks (e.g. 'true'/'false' strings)."""
    normalized: dict[str, Any] = dict(payload)

    def coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    for key in (
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
    ):
        normalized[key] = coerce_bool(normalized.get(key))
    for key in (CHECKER_FIELD_REASON, CHECKER_FIELD_COT):
        value = normalized.get(key)
        normalized[key] = "" if value is None else str(value)
    return normalized


@dataclass(slots=True)
class LLMCheckerConfig:
    api_key: str
    model: str
    base_url: str | None = None
    temperature: float = 0.0
    max_workers: int = 8
    max_prompt_chars: int = 20000
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> LLMCheckerConfig | None:
        api_key = (
            os.environ.get("CHECKER_API_KEY")
            or os.environ.get("LLM_CHECKER_API_KEY")
            or os.environ.get("JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("API_KEY")
        )
        model = (
            os.environ.get("CHECKER_MODEL")
            or os.environ.get("LLM_CHECKER_MODEL")
            or os.environ.get("JUDGE_MODEL")
        )
        base_url = (
            os.environ.get("CHECKER_BASE_URL")
            or os.environ.get("LLM_CHECKER_BASE_URL")
            or os.environ.get("JUDGE_BASE_URL")
            or os.environ.get("LLM_JUDGE_BASE_URL")
            or os.environ.get("API_BASE")
        )
        if not api_key or not model:
            return None
        return cls(api_key=api_key, model=model, base_url=base_url)


def _build_prompt(context: str, answer: str, ref_answer: str) -> str:
    fields = [
        CHECKER_FIELD_COT,
        CHECKER_FIELD_ANSWER_CORRECT,
        CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR,
        CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR,
        CHECKER_FIELD_MATH_ERROR,
        CHECKER_FIELD_REASONING_LOGIC_ERROR,
        CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER,
        CHECKER_FIELD_REASON,
    ]
    return _PROMPT_TEMPLATE.format(
        context=context,
        answer=answer,
        ref_answer=ref_answer,
        field_names=", ".join(fields),
    )


class LLMCheckerFailure(RuntimeError):
    """Raised when the checker cannot obtain/validate a response after retries."""


class _LLMCheckerOutputError(ValueError):
    """Raised when the provider output cannot be parsed/validated as required JSON."""


def _call_llm_checker(client: OpenAI, *, config: LLMCheckerConfig, prompt: str) -> dict[str, Any]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "rwkv_skills_llm_checker",
            "schema": CHECKER_JSON_SCHEMA,
            "strict": True,
        },
    }

    last_exc: Exception | None = None
    for attempt in range(max(1, int(config.max_retries) + 1)):
        try:
            response = client.chat.completions.create(
                model=config.model,
                stream=False,
                temperature=float(config.temperature),
                response_format=response_format,
                messages=[{"role": "user", "content": prompt}],
            )
            content = (response.choices[0].message.content or "").strip()
            data = json.loads(content)
            if not isinstance(data, dict):
                raise _LLMCheckerOutputError("LLM checker output is not a JSON object")
            data = _coerce_checker_payload(data)
            _validate_checker_payload(data)
            return data
        except (
            OpenAIError,
            JSONDecodeError,
            jsonschema.exceptions.ValidationError,
            _LLMCheckerOutputError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            last_exc = exc
            # If the provider doesn't support json_schema response_format, fall back to plain JSON.
            try:
                response = client.chat.completions.create(
                    model=config.model,
                    stream=False,
                    temperature=float(config.temperature),
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                            + "\n\n仅输出 JSON（不要输出任何额外文本），并确保字段类型正确：布尔值必须是 true/false。",
                        }
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise _LLMCheckerOutputError("LLM checker output is not a JSON object")
                data = _coerce_checker_payload(data)
                _validate_checker_payload(data)
                return data
            except (
                OpenAIError,
                JSONDecodeError,
                jsonschema.exceptions.ValidationError,
                _LLMCheckerOutputError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc2:
                last_exc = exc2
                if attempt >= int(config.max_retries):
                    break
                continue

    raise LLMCheckerFailure(f"LLM checker failed after retries: {last_exc}") from last_exc


def run_llm_checker(
    eval_results_path: str | Path,
    *,
    model_name: str,
    config: LLMCheckerConfig | None = None,
) -> Path | None:
    """Run wrong-answer checker over a single eval JSONL file.

    Returns the written check JSONL path, or None if skipped.
    """

    _load_env_file((REPO_ROOT / ".env").resolve())
    cfg = config or LLMCheckerConfig.from_env()
    if cfg is None:
        print("⚠️  LLM checker skipped: missing API_KEY/JUDGE_MODEL (see .env)")
        return None

    eval_path = Path(eval_results_path).expanduser().resolve()
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)

    failed_rows: list[dict[str, Any]] = []
    benchmark_name: str | None = None

    for row in _iter_jsonl(eval_path):
        if benchmark_name is None:
            benchmark_name = str(row.get("benchmark_name", "") or "")
        if bool(row.get("is_passed", False)):
            continue
        failed_rows.append(row)

    if not benchmark_name:
        print(f"⚠️  LLM checker skipped: empty eval file {eval_path}")
        return None

    if not failed_rows:
        print(f"✅ LLM checker: no failed samples for {benchmark_name} ({eval_path.name})")
        return None

    out_path = check_details_path(benchmark_name, model_name=model_name)
    seen = _load_existing_keys(out_path)

    to_check: list[dict[str, Any]] = []
    for row in failed_rows:
        split = str(row.get("dataset_split", "") or "")
        sample_index = int(row.get("sample_index", 0))
        repeat_index = int(row.get("repeat_index", 0))
        key = (split, sample_index, repeat_index)
        if key in seen:
            continue
        to_check.append(row)

    if not to_check:
        print(f"✅ LLM checker: all failed samples already checked -> {out_path}")
        return out_path

    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("ab") as out_f:
        for row in to_check:
            context = _truncate_text(str(row.get("context", "") or ""), max_chars=int(cfg.max_prompt_chars))
            answer = _truncate_text(
                str(row.get("answer", "") or ""),
                max_chars=max(1, int(cfg.max_prompt_chars // 4)),
            )
            ref_answer = _truncate_text(
                str(row.get("ref_answer", "") or ""),
                max_chars=max(1, int(cfg.max_prompt_chars // 4)),
            )
            prompt = _build_prompt(context, answer, ref_answer)

            try:
                checked = _call_llm_checker(client, config=cfg, prompt=prompt)
            except LLMCheckerFailure as exc:
                print(f"⚠️  LLM checker failed; stop early: {exc}")
                return out_path if out_path.exists() else None
            output_row = {
                "benchmark_name": str(row.get("benchmark_name", "") or ""),
                "dataset_split": str(row.get("dataset_split", "") or ""),
                "sample_index": int(row.get("sample_index", 0)),
                "repeat_index": int(row.get("repeat_index", 0)),
                CHECKER_FIELD_COT: str(checked.get(CHECKER_FIELD_COT, "") or ""),
                CHECKER_FIELD_ANSWER_CORRECT: bool(checked.get(CHECKER_FIELD_ANSWER_CORRECT, False)),
                CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR: bool(
                    checked.get(CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR, False)
                ),
                CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR: bool(
                    checked.get(CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR, False)
                ),
                CHECKER_FIELD_MATH_ERROR: bool(checked.get(CHECKER_FIELD_MATH_ERROR, False)),
                CHECKER_FIELD_REASONING_LOGIC_ERROR: bool(
                    checked.get(CHECKER_FIELD_REASONING_LOGIC_ERROR, False)
                ),
                CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER: bool(
                    checked.get(CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER, False)
                ),
                CHECKER_FIELD_REASON: str(checked.get(CHECKER_FIELD_REASON, "") or ""),
            }
            out_f.write(orjson.dumps(output_row, option=orjson.OPT_APPEND_NEWLINE))

    print(f"🧩 LLM checker saved: {out_path} (+{len(to_check)} rows)")
    return out_path


__all__ = [
    "LLMCheckerConfig",
    "run_llm_checker",
    "CHECKER_JSON_SCHEMA",
    "CHECKER_FIELD_COT",
    "CHECKER_FIELD_REASON",
    "CHECKER_FIELD_ANSWER_CORRECT",
    "CHECKER_FIELD_INSTRUCTION_FOLLOWING_ERROR",
    "CHECKER_FIELD_WORLD_KNOWLEDGE_ERROR",
    "CHECKER_FIELD_MATH_ERROR",
    "CHECKER_FIELD_REASONING_LOGIC_ERROR",
    "CHECKER_FIELD_THOUGHT_CONTAINS_CORRECT_ANSWER",
]
