# Git 存储与目录说明

## 1. evaluation_output/ 目录

### 各文件作用

| 文件模式 | 作用 | 是否需要 git 追踪 |
|----------|------|-------------------|
| `mcqs_237_<model>_roundN.json` | 第 N 轮生成的题目（每轮覆盖） | 否 |
| `results_237_<model>_roundN.json` | 第 N 轮评估结果（每轮覆盖） | 否 |
| `mcqs_237_<model>_best_{rate}.json` | 历史最高通过率对应轮次的题目 | 是 |
| `results_237_<model>_best_{rate}.json` | 历史最高通过率对应轮次的评估结果 | 是 |
| `closed_loop_progress_<model>.json` | 每轮更新的进度汇总（ephemeral） | 否 |
| `closed_loop_multi_summary.json` | 多模型闭环汇总 | 否 |
| `validate_report.json` | 校验报告 | 否 |

### 方案：只 push 最佳结果

已在根目录 `.gitignore` 与 `evaluation_output/.gitignore` 配置：

- 忽略：所有 round 文件、progress、summary、validate_report 等
- 保留：`*_best_*.json`（最佳题目与最佳评分结果）

因此 `git push` 时只会上传最佳的问题和评分结果，其他轮次文件不会进入版本库。

若 `evaluation_output` 下已有文件曾被 `git add` 追踪，需先执行 `git rm -r --cached evaluation_output/` 再 `git add evaluation_output/`，使新规则生效。

---

## 2. processed_training_data/ 目录

### 各文件作用

| 文件 | 作用 | 可删除？ | 说明 |
|------|------|----------|------|
| `examples.json` | few-shot 示例（generate_mcq 用） | 见下 | 基准示例，可被 `examples_<model>.json` 复制 |
| `examples_<model>.json` | 按模型隔离的示例 | **不建议** | 有成长性；删除会重置进度，需更多轮次恢复 |
| `prompt_rules.json` | 动态 prompt 规则（build_prompt 用） | 是 | `python main.py improve-prompt` 可恢复 |
| `prompt_rules_<model>.json` | 按模型隔离的规则 | 是 | 同上 |
| `sft_data.jsonl` | SFT 训练数据 | **不建议** | 直接加载可加快流程，删除需重新 process-data |
| `dpo_data.jsonl` | DPO 训练数据 | **不建议** | 同上 |
| `improve_examples_failed.json` | improve-examples 失败组合记录 | 是 | 下次 improve-examples 会重新生成 |
| `few_shot_examples.json` | 旧版示例（兼容用） | 是 | 可删除，会 fallback 到空 |
| `cache/` | 缓存（.arrow 等） | 是 | 自动重建 |

### examples_<model>.json 分析

- **成长性**：有。每轮闭环中 improve-examples 会为失败组合补充高分示例，并保留已有组合的示例；因此示例会随轮次累积，覆盖的 (standard, difficulty) 组合增多、质量提升。
- **删除影响**：会降低效率。删除后需从 `examples.json` 重新复制或从空开始，已积累的示例会丢失，需更多轮次才能恢复到类似水平。
- **加载速度**：examples 为 JSON，加载耗时可忽略；删除对单次加载速度影响微乎其微，但会减少整体轮次需求，从而缩短整体流程。

### 结论

- **processed_training_data/* 已被根目录 .gitignore 忽略**，这些文件不会进入 git。
- **建议保留**：`sft_data.jsonl`、`dpo_data.jsonl`、`examples_<model>.json`，以便直接加载和复用示例，加快流程。
- 其他文件可删除，需时通过 `improve-prompt`、`process-data` 等命令重新生成。
