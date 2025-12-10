from __future__ import annotations

"""Gradio space to visualise evaluation scores."""

import csv
import html
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gradio as gr

from src.eval.results.layout import ensure_results_structure
from .data import (
    ARCH_VERSIONS,
    DATA_VERSIONS,
    NUM_PARAMS,
    SPACE_SCORES_ROOT,
    ScoreEntry,
    parse_model_signature,
    latest_entries_for_model,
    list_domains,
    list_models,
    load_scores,
    pick_latest_model,
)


AUTO_MODEL_LABEL = "每档最新（调度策略）"
AUTO_EXCLUDED_PARAMS = {"0_1B", "0_4B"}
DEFAULT_DOMAIN = "全部"
@dataclass(slots=True)
class SelectionState:
    entries: list[ScoreEntry]
    dropdown_value: str
    selected_label: str
    auto_selected: bool
    model_sequence: list[str]
    aggregated_models: list[dict[str, Any]] | None = None
    skipped_small_params: int = 0


def _rank_token(candidates: Sequence[str], value: str | None) -> int | None:
    if not value:
        return None
    low = value.lower()
    try:
        return candidates.index(low)
    except ValueError:
        return None


def _sort_entries(entries: Iterable[ScoreEntry]) -> list[ScoreEntry]:
    def sort_key(item: ScoreEntry) -> tuple[Any, ...]:
        arch_rank = _rank_token(ARCH_VERSIONS, item.arch_version)
        param_rank = _rank_token(NUM_PARAMS, item.num_params)
        data_rank = _rank_token(DATA_VERSIONS, item.data_version)
        return (
            arch_rank if arch_rank is not None else len(ARCH_VERSIONS),
            param_rank if param_rank is not None else len(NUM_PARAMS),
            -(data_rank if data_rank is not None else -1),
            item.domain,
            item.dataset,
            item.task or "",
            item.model,
        )

    return sorted(entries, key=sort_key)


def _snapshot_entry(entry: ScoreEntry) -> dict[str, Any]:
    arch_rank = _rank_token(ARCH_VERSIONS, entry.arch_version)
    data_rank = _rank_token(DATA_VERSIONS, entry.data_version)
    param_rank = _rank_token(NUM_PARAMS, entry.num_params)
    return {
        "model": entry.model,
        "arch": entry.arch_version,
        "data": entry.data_version,
        "params": entry.num_params,
        "arch_rank": arch_rank,
        "data_rank": data_rank,
        "param_rank": param_rank,
        "created": entry.created_at,
        "has_signature": bool(entry.arch_version and entry.num_params),
    }


def _model_snapshots(entries: Iterable[ScoreEntry]) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for entry in entries:
        current = snapshots.get(entry.model)
        if current is None or entry.created_at > current["created"]:
            snapshots[entry.model] = _snapshot_entry(entry)
    return snapshots


def _select_signature_snapshots(entries: Iterable[ScoreEntry]) -> list[dict[str, Any]]:
    snapshots = _model_snapshots(entries)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fallback: list[dict[str, Any]] = []
    for snap in snapshots.values():
        if snap["has_signature"]:
            key = (snap["arch"], snap["params"])
            grouped.setdefault(key, []).append(snap)
        else:
            fallback.append(snap)

    selected: list[dict[str, Any]] = []
    for items in grouped.values():
        best = max(
            items,
            key=lambda snap: (
                snap["data_rank"] if snap["data_rank"] is not None else -1,
                snap["created"].timestamp(),
                snap["model"],
            ),
        )
        selected.append(best)

    selected.sort(
        key=lambda snap: (
            snap["arch_rank"] if snap["arch_rank"] is not None else len(ARCH_VERSIONS),
            snap["param_rank"] if snap["param_rank"] is not None else len(NUM_PARAMS),
            -(snap["data_rank"] if snap["data_rank"] is not None else -1),
            snap["model"],
        )
    )
    fallback.sort(key=lambda snap: (-snap["created"].timestamp(), snap["model"]))
    return selected + fallback


def _latest_entries_for_signatures(entries: Iterable[ScoreEntry]) -> tuple[list[ScoreEntry], list[dict[str, Any]], list[str]]:
    snapshots = _select_signature_snapshots(entries)
    if not snapshots:
        return [], [], []
    ordered_models = []
    seen: set[str] = set()
    for snap in snapshots:
        model = snap["model"]
        if model in seen:
            continue
        seen.add(model)
        ordered_models.append(model)
    combined: list[ScoreEntry] = []
    entry_list = list(entries)
    for model in ordered_models:
        combined.extend(latest_entries_for_model(entry_list, model))
    return _sort_entries(combined), snapshots, ordered_models


def _format_param(token: str | None) -> str:
    if not token:
        return "?"
    return token.replace("_", ".")


def _dataset_base(name: str) -> str:
    for suffix in ("_test", "_eval", "_val"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _method_tag(is_cot: bool) -> str:
    return "cot" if is_cot else "nocot"


def _model_display_name(model: str) -> str:
    sig = parse_model_signature(model)
    arch = (sig.arch or "").lower()
    data = (sig.data or "").lower()
    params = sig.params
    param_label = _format_param(params).lower() if params else ""
    if arch and data and param_label:
        return f"{arch}-{data}-{param_label}"
    parts = model.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return model


def _format_combo_label(snapshot: dict[str, Any]) -> str:
    if snapshot.get("has_signature"):
        arch = snapshot.get("arch") or "未知架构"
        params = _format_param(snapshot.get("params"))
        data = snapshot.get("data") or "?"
        return f"{arch} · {params} → {data}"
    return snapshot.get("model") or "未知模型"


def _summarise_snapshots(snapshots: Sequence[dict[str, Any]]) -> list[str]:
    combo_labels = [_format_combo_label(snap) for snap in snapshots if snap.get("has_signature")]
    extra_labels = [_format_combo_label(snap) for snap in snapshots if not snap.get("has_signature")]

    lines: list[str] = []
    if combo_labels:
        preview = " / ".join(combo_labels[:4])
        if len(combo_labels) > 4:
            preview += f" 等 {len(combo_labels)} 个组合"
        lines.append(f"- 覆盖组合：{preview}")
    if extra_labels:
        preview = " / ".join(extra_labels[:4])
        if len(extra_labels) > 4:
            preview += f" 等 {len(extra_labels)} 个模型"
        lines.append(f"- 其他未解析模型：{preview}")
    return lines


def _load_css() -> tuple[str, str | None]:
    style_path = Path(__file__).parent / "styles" / "space.css"
    if not style_path.exists():
        warning = f"未找到样式文件：{style_path}"
        print(f"[space] {warning}")
        return "", warning
    try:
        return style_path.read_text(encoding="utf-8"), None
    except Exception as exc:  # noqa: BLE001
        warning = f"未加载样式：读取 {style_path.name} 失败 ({exc})"
        print(f"[space] {warning}")
        return "", warning


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return "✓" if value else "✕"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and 0 <= value <= 1:
            return f"{value * 100:.1f}%"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if isinstance(value, int):
            return f"{value:d}"
        return f"{value:.3f}"
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _primary_metric(metrics: dict[str, Any]) -> tuple[str, str] | None:
    # Prefer common accuracy keys in a stable order so judge results surface first.
    preferred = (
        "judge_accuracy",
        "exact_accuracy",
        "accuracy",
        "prompt_accuracy",
        "instruction_accuracy",
        "pass@1",
        "pass@2",
        "pass@5",
        "pass@10",
    )
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, _format_metric_value(value)
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            return key, _format_metric_value(value)
    for key in preferred:
        value = metrics.get(key)
        if value is not None:
            return key, _format_metric_value(value)
    for key, value in metrics.items():
        if value is not None:
            return key, _format_metric_value(value)
    return None


def _html(text: Any) -> str:
    """Escape text for safe HTML rendering inside our custom table."""
    return html.escape(str(text), quote=True)


def _render_summary(
    *,
    all_entries: list[ScoreEntry],
    visible: list[ScoreEntry],
    selection: SelectionState,
    domain_choice: str,
    warnings: Iterable[str] | None = None,
) -> str:
    if not all_entries:
        ensure_results_structure()
        try:
            SPACE_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return f"未找到任何分数文件，期待路径：`{SPACE_SCORES_ROOT}`。运行评测脚本后再刷新即可。"

    benchmark_count = len(
        {(_dataset_base(entry.dataset), _method_tag(entry.cot)) for entry in visible}
    )
    lines = [
        f"- 分数根目录：`{SPACE_SCORES_ROOT}`",
        f"- 当前策略：`{selection.selected_label}`" + ("（按排序规则自动选择）" if selection.auto_selected else ""),
        f"- 已选大领域：{domain_choice}",
        f"- 模型列数：{len(selection.model_sequence)}",
        f"- 基准行数：{benchmark_count}",
        f"- 可见数据集：{len(visible)} / 总分数文件：{len(all_entries)}",
        "- 排序：架构 > 参数量 > data_version（G0→…→G1b）> domain > dataset / task",
    ]
    if selection.selected_label == AUTO_MODEL_LABEL and selection.skipped_small_params:
        lines.append(f"- 已忽略 {selection.skipped_small_params} 个 0.1B / 0.4B 组合（调度策略）")
    if selection.aggregated_models:
        lines.extend(_summarise_snapshots(selection.aggregated_models))
    for warn in warnings or ():
        lines.append(f"⚠️ {warn}")
    return "\n".join(lines)


def _filter_by_domain(entries: Iterable[ScoreEntry], domain: str) -> list[ScoreEntry]:
    if domain == DEFAULT_DOMAIN:
        filtered = list(entries)
    else:
        filtered = [entry for entry in entries if entry.domain == domain]
    return _sort_entries(filtered)


def _prepare_selection(entries: list[ScoreEntry], selection_value: str | None, domain: str) -> SelectionState:
    if not entries:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    if selection_value == AUTO_MODEL_LABEL or selection_value is None:
        combined_entries, snapshots, ordered_models = _latest_entries_for_signatures(entries)
        allowed_models: list[str] = []
        filtered_snapshots: list[dict[str, Any]] = []
        skipped_small = 0
        for snap in snapshots:
            params = snap.get("params")
            if snap.get("has_signature") and params in AUTO_EXCLUDED_PARAMS:
                skipped_small += 1
                continue
            allowed_models.append(snap["model"])
            filtered_snapshots.append(snap)
        if allowed_models:
            snapshots = filtered_snapshots
        else:
            allowed_models = ordered_models
        filtered_entries = [entry for entry in combined_entries if entry.model in allowed_models]
        visible_entries = _filter_by_domain(filtered_entries, domain)
        return SelectionState(
            entries=visible_entries,
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label=AUTO_MODEL_LABEL,
            auto_selected=False,
            model_sequence=allowed_models,
            aggregated_models=snapshots,
            skipped_small_params=skipped_small,
        )

    models = set(list_models(entries))
    target_model = selection_value if selection_value in models else pick_latest_model(entries)
    auto_selected = target_model != selection_value
    if not target_model:
        return SelectionState(
            entries=[],
            dropdown_value=AUTO_MODEL_LABEL,
            selected_label="未检测到模型",
            auto_selected=True,
            model_sequence=[],
            aggregated_models=None,
            skipped_small_params=0,
        )

    latest = latest_entries_for_model(entries, target_model)
    visible_entries = _filter_by_domain(latest, domain)
    return SelectionState(
        entries=visible_entries,
        dropdown_value=target_model,
        selected_label=target_model,
        auto_selected=auto_selected,
        model_sequence=[target_model],
        aggregated_models=None,
        skipped_small_params=0,
    )


def _cell_metric_value(entry: ScoreEntry | None) -> str:
    if entry is None:
        return "—"
    primary = _primary_metric(entry.metrics)
    if not primary:
        return "—"
    return primary[1]


def _build_pivot_table(selection: SelectionState) -> tuple[list[str], list[list[Any]]]:
    headers = ["Benchmark"] + [_model_display_name(model) for model in selection.model_sequence]
    if not selection.entries:
        return headers, []

    row_meta: dict[tuple[str, str], dict[str, Any]] = {}
    grouped: dict[tuple[str, str, str], ScoreEntry] = {}

    for entry in selection.entries:
        base = _dataset_base(entry.dataset)
        method = _method_tag(entry.cot)
        row_key = (base, method)
        meta = row_meta.get(row_key)
        if meta is None:
            primary = _primary_metric(entry.metrics)
            metric_name = primary[0] if primary else None
            row_meta[row_key] = {
                "base": base,
                "method": method,
                "metric": metric_name or "metric",
            }
        elif meta["metric"] == "metric":
            primary = _primary_metric(entry.metrics)
            metric_name = primary[0] if primary else None
            if metric_name:
                meta["metric"] = metric_name

        group_key = (entry.model, base, method)
        current = grouped.get(group_key)
        def _metric_score(item: ScoreEntry | None) -> float | None:
            if item is None:
                return None
            for val in item.metrics.values():
                if isinstance(val, (int, float)):
                    return float(val)
            return None

        cur_score = _metric_score(current)
        new_score = _metric_score(entry)
        if current is None or (new_score is not None and (cur_score is None or new_score > cur_score)) or (
            new_score == cur_score and entry.created_at > current.created_at
        ):
            grouped[group_key] = entry

    ordered_rows = sorted(
        row_meta.values(),
        key=lambda meta: (meta["base"], 0 if meta["method"] == "cot" else 1),
    )

    rows: list[list[Any]] = []
    for meta in ordered_rows:
        row_label = f"{meta['base']}_{meta['method']}"
        row: list[Any] = [row_label]
        for model in selection.model_sequence:
            entry = grouped.get((model, meta["base"], meta["method"]))
            row.append(_cell_metric_value(entry))
        rows.append(row)
    return headers, rows


def _pivot_to_csv(headers: list[str], rows: list[list[Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def _render_pivot_html(headers: list[str], rows: list[list[Any]]) -> str:
    """Render pivot table into a HTML table with predictable column widths.

    要求：
    - Benchmark 列：自适应整段名称，保持单行；
    - 模型列：宽度一致，按表头 / 单元格里的最长字符串估算；
    - 整个表格可以水平滚动，确保所有文本都完整显示。
    """
    if not headers:
        return '<div class="space-table-empty">当前筛选条件下没有数据。</div>'

    def _token_length(token: Any) -> int:
        return len(_html(token))

    col_lengths = [_token_length(title) for title in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if idx < len(col_lengths):
                col_lengths[idx] = max(col_lengths[idx], _token_length(cell))

    benchmark_width = max(col_lengths[0] + 2, 14)
    model_col_width = max(max(col_lengths[1:], default=10) + 2, 10)

    header_cells = "".join(f"<th>{_html(title)}</th>" for title in headers)
    body_rows: list[str] = []
    for row in rows:
        cells: list[str] = []
        for idx, cell in enumerate(row):
            cls = ' class="cell-model"' if idx == 0 else ""
            cells.append(f"<td{cls}>{_html(cell)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    rows_html = "".join(body_rows) if body_rows else '<tr><td colspan="999">当前筛选条件下没有数据。</td></tr>'

    colgroup = (
        '<col class="col-model" />'
        + "".join('<col class="col-metric" />' for _ in headers[1:])
    )

    return f"""
<div class="space-table-title">明细</div>
<div class="space-table-grid" style="--model-col-width: {benchmark_width}ch; --metric-col-width: {model_col_width}ch;">
  <table>
    <colgroup>{colgroup}</colgroup>
    <thead>
      <tr>{header_cells}</tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
""".strip()


def _compute_choices(entries: list[ScoreEntry]) -> tuple[list[str], list[str]]:
    models = [AUTO_MODEL_LABEL] + list_models(entries)
    domains = [DEFAULT_DOMAIN] + list_domains(entries)
    return models, domains


def _initial_payload() -> tuple[list[ScoreEntry], SelectionState, str, list[str]]:
    errors: list[str] = []
    entries = load_scores(errors=errors)
    selection = _prepare_selection(entries, AUTO_MODEL_LABEL, DEFAULT_DOMAIN)
    return entries, selection, DEFAULT_DOMAIN, errors


def _build_dashboard() -> gr.Blocks:
    css, style_warning = _load_css()
    entries, selection, domain, load_errors = _initial_payload()
    model_choices, domain_choices = _compute_choices(entries)
    warnings = load_errors + ([style_warning] if style_warning else [])

    summary = _render_summary(
        all_entries=entries,
        visible=selection.entries,
        selection=selection,
        domain_choice=domain,
        warnings=warnings,
    )
    pivot_headers, pivot_rows = _build_pivot_table(selection)
    pivot_html = _render_pivot_html(pivot_headers, pivot_rows)

    # Gradio 5.50 不再可靠支持 Blocks(elem_id=...) 作为 CSS 锚点，
    # 所以这里用一个带有自定义 class 的 Column 作为样式作用域根节点。
    with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes="space-root"):
            gr.HTML(
                """
<div class="space-card space-hero">
  <div>
    <h1>RWKV Skills · Space</h1>
    <div class="hero-subtitle">以最新分数为基准，快速浏览各评测领域。</div>
  </div>
</div>
"""
            )

            with gr.Row(elem_classes="space-card space-controls space-controls--tight"):
                model_dropdown = gr.Dropdown(
                    label="模型选择",
                    info="默认项会对每个架构 + 参数量组合选取 data_version（G0→…→G1b）最新的模型；手动选择时展示单个模型的最新分数文件。",
                    choices=model_choices,
                    value=AUTO_MODEL_LABEL,
                    scale=3,
                )
                domain_dropdown = gr.Dropdown(
                    label="大领域",
                    choices=domain_choices,
                    value=domain,
                    scale=2,
                )
                refresh_btn = gr.Button("刷新分数", variant="primary", elem_classes="space-refresh-btn", scale=1)

            summary_md = gr.Markdown(summary, elem_classes="space-card space-summary-card")
            table = gr.HTML(pivot_html, elem_classes="space-card space-table-card")
            download_btn = gr.DownloadButton("导出为 CSV", elem_classes="space-export-btn")

            def update_dashboard(selected_model: str, selected_domain: str):
                load_errors: list[str] = []
                entries = load_scores(errors=load_errors)
                model_choices, domain_choices = _compute_choices(entries)
                domain_value = selected_domain if selected_domain in domain_choices else DEFAULT_DOMAIN
                dropdown_value = selected_model if selected_model in model_choices else AUTO_MODEL_LABEL

                selection_state = _prepare_selection(entries, dropdown_value, domain_value)
                warnings = load_errors + ([style_warning] if style_warning else [])

                summary_value = _render_summary(
                    all_entries=entries,
                    visible=selection_state.entries,
                    selection=selection_state,
                    domain_choice=domain_value,
                    warnings=warnings,
                )
                pivot_headers, pivot_rows = _build_pivot_table(selection_state)
                pivot_html = _render_pivot_html(pivot_headers, pivot_rows)
                return (
                    gr.update(choices=model_choices, value=selection_state.dropdown_value),
                    gr.update(choices=domain_choices, value=domain_value),
                    gr.update(value=summary_value),
                    gr.update(value=pivot_html),
                )

            model_dropdown.change(
                update_dashboard,
                inputs=[model_dropdown, domain_dropdown],
                outputs=[model_dropdown, domain_dropdown, summary_md, table],
            )
            domain_dropdown.change(
                update_dashboard,
                inputs=[model_dropdown, domain_dropdown],
                outputs=[model_dropdown, domain_dropdown, summary_md, table],
            )
            refresh_btn.click(
                update_dashboard,
                inputs=[model_dropdown, domain_dropdown],
                outputs=[model_dropdown, domain_dropdown, summary_md, table],
            )

            def export_csv(selected_model: str, selected_domain: str):
                entries = load_scores()
                selection_state = _prepare_selection(entries, selected_model, selected_domain)
                headers, rows = _build_pivot_table(selection_state)
                csv_text = _pivot_to_csv(headers, rows)
                temp_dir = Path(tempfile.mkdtemp(prefix="rwkv_space_"))
                path = temp_dir / "rwkv_scores.csv"
                path.write_text(csv_text, encoding="utf-8")
                return str(path)

            download_btn.click(
                export_csv,
                inputs=[model_dropdown, domain_dropdown],
                outputs=download_btn,
            )

    return demo


def main() -> None:
    demo = _build_dashboard()
    demo.launch()


if __name__ == "__main__":  # pragma: no cover
    main()
