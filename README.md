# RWKV Skills

面向 RWKV7 的推理与评测脚手架，包含批量推理引擎、常见评测数据集的准备器以及一个 GPU 调度器骨架。

## 目录速览
- `src/infer`：RWKV 模型加载、采样策略与连续批量生成引擎。
- `src/eval/datasets`：数据结构定义、JSONL 加载器以及各类数据集的准备脚本。
- `src/eval/evaluators`：多选 / 自由问答 / 指令遵循评测管线（代码生成管线尚未实现）。
- `src/eval/scheduler`：评测任务排队、GPU 侦测与调度的 CLI（现已附带 multi-choice / free-response / instruction-following 入口脚本）。
- `albatross`：上游 RWKV 参考实现子模块。
- `weights`、`data`、`results`（可选）：模型权重、数据集与评测产物的默认存放位置。

## 环境要求
- Python 3.12+，推荐安装 `uv` 以管理依赖。
- NVIDIA GPU（使用 `flashinfer`、`triton` 等依赖），需要与所选 PyTorch 发行版匹配的 CUDA/ROCm。
- Git 子模块：`git submodule update --init --recursive` 确保 `albatross` 就绪。

## 安装
```bash
# 安装依赖（示例：CUDA 12.9，对应 pyproject 中 torch-cu129 可选项）
uv sync --extra torch-cu129

# 开发模式安装，暴露 CLI 入口
uv pip install -e .
```
如需其他 CUDA/CPU 发行版，请改用 `--extra torch-cu126` / `--extra torch-cpu` 等。

## 下载模型权重
`rwkv-download-weights` 会从 Hugging Face 镜像枚举并并发下载 `.pth` 权重：
```bash
rwkv-download-weights /path/to/weights
# 或指定额外仓库：
rwkv-download-weights --repo BlinkDL/rwkv7-g1 --repo your/repo
```
可通过环境变量覆盖默认镜像与 Token（`HF_ENDPOINT`、`HF_TOKEN`）。

## 数据集准备
数据集默认存放在 `data/`。可以直接调用准备器生成 JSONL：
```bash
python - <<'PY'
from pathlib import Path
from src.eval.datasets.data_prepper.data_manager import prepare_dataset

prepare_dataset("mmlu", Path("data"))  # 会生成 data/mmlu/<split>.jsonl
PY
```
支持的数据集别名可通过 `available_*_datasets()` 系列函数查看。

## 评测与推理示例
目前推荐直接调用管线类（调度器缺少实际执行脚本，见下文）：
```python
from src.eval.evaluators.multi_choice import MultipleChoicePipeline
from src.infer.model import ModelLoadConfig

pipeline = MultipleChoicePipeline(ModelLoadConfig(weights_path="weights/rwkv7-*.pth"))
result = pipeline.run_direct(
    dataset_path="data/mmlu/test.jsonl",
    output_path="results/logs/mmlu_direct.jsonl",
)
print(result)
```
自由问答与指令遵循的用法类似，分别使用 `FreeResponsePipeline` 与 `InstructionFollowingPipeline`。

## 调度器 CLI
`rwkv-skills-scheduler` 暴露了一组命令（队列预览、调度、状态、停止、日志轮播）：
```bash
rwkv-skills-scheduler queue
rwkv-skills-scheduler dispatch --run-log-dir results/logs
```
默认模型 glob 在 `src/eval/scheduler/config.py` 中配置（仅指向仓库内 `weights/rwkv7-*.pth`，请按需覆盖）。调度器依赖的入口脚本已提供：
`src/bin/eval_multi_choice.py`、`eval_multi_choice_cot.py`、`eval_free_response.py`、`eval_free_response_judge.py`、`eval_instruction_following.py`。

## 已知缺口 / TODO
- `src/eval/evaluators/coding.py` 为空文件，代码生成评测管线尚未实现（对应 job 已暂时移除）。
- 代码生成相关依赖（EvalPlus 等）已声明，但缺少配套的运行逻辑。

欢迎根据上述缺口补全实现并更新文档。
