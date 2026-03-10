# 生成题校验出错根因分析

## 处理策略：修复或构造，不丢弃

送评题目总数与 **(standard, difficulty) 组合** 保持一致。校验不通过的题目会依次：

1. **自动修复**：`fix_mcq` 处理题干复数、选项重复、解析未提及正确选项、无图却提看图等。
2. **激进修复**：`repair_aggressively` 补全缺失字段、修正 answer 与选项一致、多轮 `fix_mcq`。
3. **仍不通过则构造**：`build_minimal_valid_mcq` 为该组合生成一条满足校验的最小合法题，保证该组合不丢失。

生成脚本（`--all-combinations`）按组合下标存储结果，最终输出条数 = 计划组合数，不再丢弃任何组合。

---

## 问题现象

`evaluation_output/mcqs_240.json` 中部分题目在运行 `scripts/validate_mcq.py` 时校验不通过，主要报错为：

- **duplicate_option_text**：选项 A/B/C/D 中存在两条及以上文字完全相同
- **stem_says_choices_plural_may_imply_multiple_answers**：题干使用 “Which choices/options” 等复数表述，易被理解为多选
- **stem_references_image_but_no_image_url**：题干提到看图/用图但未提供 `image_url`

## 根因结论：**Prompt 未被系统遵守，而非缺示例**

### 1. 示例情况

对当前所有校验失败题目对应的 `(standard, difficulty)` 在 `processed_training_data/examples.json` 中做了统计与校验：

| 情况 | 数量 | 说明 |
|------|------|------|
| 该组合有 1 条示例且该示例**通过校验** | 多数 | 示例中无重复选项、无复数题干、无未提供图片的看图表述 |
| 该组合**无示例**（0 条） | 2 个 | L.3.1.D easy、RL.3.7 easy |

结论：**校验出错并不是因为“缺高质量示例”**。有示例的组合里示例本身是合规的；无示例的组合则完全依赖 system/user prompt，仍出现重复选项或题干/图片问题，说明约束未被稳定执行。

### 2. Prompt 行为

- 约束已写在 **system prompt** 的 “GLOBAL CONSTRAINTS” 中（选项互异、单数表述、不引用未提供图片）。
- 但生成时模型**没有在输出前做稳定自检**，容易出现：
  - 写错一个干扰项导致与正确项相同（duplicate_option_text）
  - 沿用 “Which choices correctly...” 等表述（plural stem）
  - 题干写 “The illustration shows...” 却无 `image_url`（image reference）

因此根因是：**约束只在 system 里出现一次，且未在“即将输出”的上下文中再次强调，导致模型未系统性地在生成前自检。**

## 已采取的针对性措施（系统性修改 Prompt）

1. **在 system 的 GLOBAL CONSTRAINTS 中**
   - 明确增加“不得在无图时引用看图/用图”的约束。
   - 增加一句 **“Before outputting your JSON, do a final check: (1) A≠B≠C≠D, (2) stem 用单数 Which choice/option, (3) 无 image 引用则无 image_url”**，把“输出前自检”写进系统约束。

2. **在每次请求的 user prompt 末尾**
   - 增加固定句 **`_USER_VERIFICATION_REMINDER`**，在**每次**生成前再次提醒：
     - 选项 A/B/C/D 必须互不相同；
     - 题干使用单数 “Which choice/option”，禁止 “Which choices/options”；
     - 不得在无 `image_url` 时说 “look at the picture” / “use the image”。

这样既保留“全局约束”的集中说明，又在**临近生成**的 user 里重复关键约束，促使模型在输出前做一次自检，减少同类校验错误。

## 后续建议

- **重新生成并校验**：用当前 prompt 重新跑 `generate_mcq` 生成一批题，再跑 `validate_mcq`，观察 duplicate_option_text / plural stem / image 相关错误是否明显下降。
- **无示例组合**：L.3.1.D easy、RL.3.7 easy 若仍易出错，可考虑为这两组补充 1 条通过校验的示例，或在该组合的 user prompt 中再写一句针对性的“选项互异 + 单数题干”提醒。
- **持续监控**：若某类错误仍反复出现，可在 `build_user_prompt` 中针对该类错误再加一句更短、更刺眼的提醒（例如 “CRITICAL: A,B,C,D must be 4 different strings.”）。
