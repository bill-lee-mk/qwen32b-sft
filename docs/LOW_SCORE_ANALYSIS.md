# 打分低于 0.85 的原因分析与闭环方案

## 闭环流程（不修改题目文件）

1. **生成**：用 few-shot 对 DeepSeek 做指令微调，生成 3 年级 ELA MCQ。
2. **评估**：题目提交 InceptBench API，≥0.85 为合格，目标合格率 ≥95%。
3. **更新示例**：从 `results_240.json` 读取分数，选出 score &lt; 0.85 的题目，取其 `(standard, difficulty)`；在 `raw_data/*.jsonl` 中筛出同 `(standard, difficulty)` 的 MCQ，仅对这些候选做 InceptBench 评分，每个组合保留 1 条 score ≥ 0.85，写入 `examples.json` 并替换该组合下原有示例。

**重要**：不修改 `evaluation_output/mcqs_240.json` 题目本身，只根据打分反馈做「示例是否缺失」与「prompt 是否需调整」的分析与闭环。

---

## 本次分析结果（基于 results_240.json）

- **低分/Error 题目数**：26
- **涉及 (standard, difficulty) 组合数**：26
- **在 examples.json 中【有】该组合高分示例**：17 个 → 有示例仍低分
- **在 examples.json 中【无】该组合高分示例**：9 个 → 缺示例很可能是主因

---

## 原因一：examples 中缺少该组合的高分示例（9 个）

这些组合在 `processed_training_data/examples.json` 里**没有任何**该 `(standard, difficulty)` 的条目，模型没有可模仿的 ≥0.85 示例，生成质量易偏低。

| Standard (简写) | Difficulty |
|-----------------|------------|
| L.3.3.A         | medium     |
| RI.3.8          | hard       |
| RI.3.9          | hard       |
| RL.3.10         | easy       |
| SL.3.1.B        | easy       |
| SL.3.1.B        | hard       |
| SL.3.1.D        | hard       |
| W.3.1.D         | hard       |
| W.3.6           | medium     |

**对应低分/Error 题**：题51(0.83), 题125(0.83), 题132(0.74), 题133(0.83), 题159(error), 题162(error), 题169(0.83), 题194(0.83), 题232(0.83) 等。

**结论**：低分主要因为 **examples 缺该 (standard, difficulty) 的 ≥0.85 示例**。

**解决方案**：

1. 在 **raw_data** 中补充与上述 9 个组合对应的 MCQ（每条含 `standard`、`difficulty`，且为 `messages` 或 DPO `prompt/chosen` 格式，能被 `scripts/improve_examples.py` 解析）。
2. 运行闭环命令，从 raw_data 中只对「失败组合」的候选打分，每个组合保留 1 条 ≥0.85 写入 examples：
   ```bash
   python main.py improve-examples --results evaluation_output/results_240.json --mcqs evaluation_output/mcqs_240.json --output processed_training_data/examples.json
   ```
3. 若当前 **raw_data 下无 `*.jsonl`**，improve-examples 会显示「待评分候选: 0 条」，不会更新 examples。需先往 `raw_data/` 放入对应 jsonl 再跑上述命令。

---

## 原因二：examples 中已有该组合示例，但仍出现低分（17 个）

这些组合在 examples 里**已有**该 `(standard, difficulty)` 的条目（且 improve_examples 只保留过 ≥0.85 的示例），但本批仍出现 score &lt; 0.85 的题目。

| Standard (简写) | Difficulty |
|-----------------|------------|
| L.3.1.E  | hard   |
| L.3.2.A  | easy   |
| L.3.2.D  | medium |
| L.3.2.G  | hard   |
| L.3.5.B  | medium |
| L.3.5.C  | hard   |
| RF.3.3.C | easy   |
| RF.3.3.D | medium |
| RI.3.10  | hard   |
| RI.3.8   | easy   |
| RL.3.5   | hard   |
| RL.3.5   | medium |
| RL.3.9   | hard   |
| W.3.1.B  | easy   |
| W.3.10   | hard   |
| W.3.3.A  | hard   |
| W.3.5    | easy   |

**结论**：低分**不是**因为「完全没有示例」，而更可能是：

- **示例数量不足**：该组合只有 1 条示例，模型未充分学到格式与质量要求。
- **生成 prompt 未强调易扣分点**：如答案与解析一致、simple vs progressive 时态、单复数表述、题干与选项匹配等，需在 system/user prompt 中更明确。
- **题目本身难度/表述**：个别题存在事实错误、答案标错、题干歧义等，需通过 prompt 规则约束生成行为。

**解决方案（建议优先顺序）**：

1. **加强生成 prompt**  
   - 在 `data_processing/build_prompt.py` 的 system prompt 中继续强化现有规则（如：答案与 explanation 必须一致、不用 “Which choices” 表单选、时态/术语与标准一致、不引用图片除非提供 image_url 等）。  
   - 可根据 InceptBench 的 `internal_reasoning` / `suggested_improvements` 提炼成简短规则加入 prompt。

2. **增加该组合的示例条数**  
   - 对上述 17 个组合，在 raw_data 中多提供几条同 `(standard, difficulty)` 的 MCQ，再运行 improve-examples，使每个组合在 examples 中保留 ≥2 条 ≥0.85 的示例（若 improve_examples 支持 `max_per_pair` 可设为 2）。

3. **可选：把 InceptBench 反馈写回 prompt**  
   - 对反复低分的 (standard, difficulty)，可将典型扣分原因（如 factual_accuracy、clarity_precision）写成 1～2 条「禁止/必须」规则加入该组合的 prompt 模板。

---

## 如何复现本分析

不修改题目，仅做对比分析：

```bash
python scripts/analyze_low_scores_vs_examples.py
```

脚本会：

- 从 `evaluation_output/results_240.json` 和 `evaluation_output/mcqs_240.json` 读入分数与题目；
- 从 `processed_training_data/examples.json` 提取所有 `(standard, difficulty)`；
- 输出：低分/Error 题数量、有/无示例的组合列表、每道低分题的题号/分数/standard/difficulty/id 及是否「有示例」。

---

## 小结

| 情况 | 组合数 | 主要原因 | 建议动作 |
|------|--------|----------|----------|
| examples 中**无**该组合示例 | 9 | 缺 ≥0.85 示例 | 向 raw_data 补充对应 jsonl，再运行 improve-examples |
| examples 中**有**该组合示例 | 17 | 示例数少 / prompt 未约束易扣分点 | 加强 build_prompt 规则、增加该组合示例数，必要时加入 InceptBench 反馈规则 |

不修改 `mcqs_240.json` 的前提下，先完成「缺示例」的 9 个组合的示例补充与闭环，再针对「有示例仍低分」的 17 个组合做 prompt 与示例数量优化，可系统提升合格率 toward ≥95%。
