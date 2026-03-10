# 从失败组合改进 Prompt 规则

## 策略（与示例互补）

- **低分因缺示例** → 增加示例：从 raw_data 找同 (std,diff) 且 InceptBench ≥0.85 的题加入 examples；若本批有同组合高分题则直接用本批题作示例。
- **raw_data 无 ≥0.85 的候选** → 构造一条：用本批该 (std,diff) 中得分最高的题（即使 <0.85）作为示例写入，避免该组合完全无示例。
- **有示例仍 <0.85** → 改进 prompt：仅对「examples 中已有该 (std,diff) 示例」的失败组合，从评估反馈生成针对性规则写入 `prompt_rules.json`。

闭环顺序：每轮先 **补示例**（improve-examples），再 **改 prompt 规则**（improve-prompt，且传入 `--examples`，从而只对有示例仍低分的组合加针对性规则）。

## 规则结构概览

- **全局规则**：适用于所有题目，写入 `prompt_rules.json` 的 `global_rules`，在 `build_system_prompt()` 中追加到 system 末尾（“Dynamic global rules” 区块）。
- **针对性规则**：
  - **按 standard**：`by_standard["CCSS.ELA-LITERACY.L.3.1.D"]`，对该标准下任意难度生效。
  - **按 (standard, difficulty)**：`by_standard_difficulty["CCSS.ELA-LITERACY.L.3.1.D|easy"]`，仅对该组合生效。
- 生成单题时，`build_full_prompt()` 会注入：全局规则（system）+ 针对该 standard 与 (standard, difficulty) 的规则（user 末尾 “Reminders for this standard/difficulty”）。

## 规则来源

- 从评估结果中取 **得分 < 0.85** 的题目，读取 InceptBench 返回的 `suggested_improvements` 或 `reasoning`。
- **全局规则**：根据失败反馈中的关键词（如 distractor、guide word、aloud、false claim、peer feedback 等）匹配预设主题，加入对应的一条全局说明，避免重复。
- **标准级规则**：当某标准出现低分时，自动注入 `_STANDARD_SPECIFIC_RULES` 预设规则，仅对该标准生效。
- **针对性规则**：
  - **by_standard**：始终加入所有失败标准的反馈（最大化利用低分建议），不论该 (std,diff) 是否有 examples。
  - **by_standard_difficulty**：仅对 examples 中已有该 (std,diff) 示例的失败组合加入，避免无示例时规则过于泛化。

## 进一步方案：最大化利用低分建议

| 规则层级 | 适用范围 | 更新逻辑 | 示例 |
|----------|----------|----------|------|
| **全局规则** | 所有题目 | 关键词匹配失败反馈 → 加入 `global_rules` | guide words、aloud 定义、false claim、peer feedback、run-on/fragment |
| **标准级规则** | 该标准下所有难度 | 该标准出现低分 → 注入 `_STANDARD_SPECIFIC_RULES` 预设 | L.3.2.G 字典 guide words、RF.3.4.C aloud、RI.3.10 事实准确性、W.3.5 peer feedback |
| **by_standard** | 该标准下所有难度 | 始终加入该标准所有低分题的反馈 | 每标准最多 4 条（含标准级规则+反馈） |
| **by_standard_difficulty** | 该 (标准,难度) 组合 | 仅当 examples 中有该组合 → 加入反馈 | 每组合最多 3 条 |

**新增全局主题关键词**（`scripts/improve_prompt.py` 中 `_GLOBAL_THEME_RULES`）：
- `guide word` / `falls between` → 字典 guide words 题
- `aloud` / `out loud` / `audibly` → 术语定义准确性
- `false claim` / `photosynthesis` / `factually correct` →  passage 事实准确性
- `peer feedback` / `actual issue` / `actual error` → 同伴反馈与草稿一致
- `run-on` / `fragment` / `nonexistent` → 编辑题草稿包含真实错误

**新增标准级规则**（`_STANDARD_SPECIFIC_RULES`）：
- L.3.2.G、RF.3.4.C、RI.3.10、W.3.5 等可扩展

## 文件与用法

| 文件/命令 | 说明 |
|-----------|------|
| `processed_training_data/prompt_rules.json` | 动态规则存储：`global_rules`、`by_standard`、`by_standard_difficulty`。 |
| `python main.py improve-prompt --results ... --mcqs ... [--examples examples.json]` | 根据本次评估结果更新规则；提供 `--examples` 时仅对有示例仍低分的组合加针对性规则。 |
| 闭环 `closed-loop` | 每轮先 improve-examples 再 improve-prompt（传入 `--examples`），先补示例再改 prompt。 |

## 闭环中的顺序

每轮未达标时依次执行：

1. 生成题目  
2. 评估  
3. **补示例**（improve-examples）  
4. **改 prompt 规则**（improve-prompt）  

下一轮生成时，会读取更新后的 `examples.json` 和 `prompt_rules.json`，从而同时从「更多/更好的 few-shot」和「更细的全局/针对性规则」两方面提升题目质量。

## 多模型 / 多终端隔离

**closed-loop** 与 **closed-loop-multi** 均按模型使用独立文件，避免多终端分别跑不同模型时互相覆盖、提示词撕裂：

- 示例：`processed_training_data/examples_<model_slug>.json`（如 `examples_deepseek_reasoner.json`）
- 规则：`processed_training_data/prompt_rules_<model_slug>.json`
- 题目/结果：`evaluation_output/mcqs_237_<model>_*.json`、`results_237_<model>_*.json`

生成阶段通过环境变量 `PROMPT_RULES_PATH` 指定该模型的规则文件；补示例与改 prompt 的 `--output` 也指向上述模型专属路径。首次跑某模型时，若对应文件不存在，会从默认 `examples.json` / `prompt_rules.json` 复制一份作为初始值。
