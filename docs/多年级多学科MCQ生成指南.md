# 多年级多学科 MCQ 自动生成与闭环改进指南

## 这个系统做什么？

**用大语言模型自动生成 K-12 各年级、各学科的选择题（MCQ），并通过"生成→评估→改进→再生成"的闭环不断提高题目质量。**

### 核心思想

```
人工写好"出题规则"（提示词）
       ↓
大语言模型按规则批量出题
       ↓
InceptBench 自动评分（0~1 分）
       ↓
得分 ≥ 0.85 的好题 → 存为"范例"，下一轮作为参考
得分 < 0.85 的差题 → 提取反馈，写入"改进规则"
       ↓
带着更好的范例和规则，重新出题
       ↓
循环多轮，直到通过率达标
```

 **Few-Shot + Prompt 工程闭环优化**：
- **Few-Shot**：给模型看几道高分范例题，让它模仿质量
- **Prompt 工程**：根据失败反馈不断优化提示词规则
- **闭环**：生成 → 评估 → 改进 → 再生成，自动迭代

---

## 覆盖范围

### 年级（Grade）

| 值 | 含义 | 标准数 |
|----|------|--------|
| K | 幼儿园（Kindergarten） | 381 |
| 1-8 | 小学 1-5 年级 + 初中 6-8 年级 | 8,220 |
| 10, 11 | 高中（有明确年级标注的课程） | 129 |
| HS | 高中（未标注具体年级的课程） | 220 |
| AP | 大学先修课程（Advanced Placement） | 5,788 |
| SAT | SAT 备考 | 114 |

### 学科（Subject）

| 缩写 | 全称 | 覆盖年级 |
|------|------|----------|
| ELA | 英语语言艺术（Language） | K-8, 10-11, AP |
| READ | 阅读理解（Reading） | 2-7, AP |
| MATH | 数学（Math） | 1-8, AP |
| SCI | 科学（Science） | K-8, HS, AP |
| USHIST | 美国历史（American History） | K-7, AP |
| WHIST | 世界历史（World History） | K-6, 8, AP |
| SS | 社会学（Social Studies） | 5-8, HS, AP |
| MUS | 音乐（Music） | K-8 |
| ART | 视觉艺术（Visual Arts） | K-8 |
| BIO | 生物学（Biology） | AP |
| CHEM | 化学（Chemistry） | AP |
| CS | 计算机科学（Computer Science） | AP |
| ECON | 经济学（Economics） | AP |
| ENG | 英语写作与修辞（English） | AP |
| GOV | 政治学（Government） | AP |
| HGEO | 人文地理（Human Geography） | AP |
| SAT | SAT 备考（SAT Prep） | SAT |

共 **89 个有效的（年级, 学科）组合**，覆盖 **14,852 个教学标准**，每个标准 × 3 个难度（easy/medium/hard）= **44,556 个题目组合**。

### 难度（Difficulty）

每个标准固定生成 3 个难度等级的题目：
- **easy**：基础概念，直接考查
- **medium**：需要理解和应用
- **hard**：需要分析、综合判断

---

## 快速上手

### 前提条件

```bash
# 需要设置 API 密钥（至少一个）
export DEEPSEEK_API_KEY=sk-xxx          # DeepSeek 模型
export KIMI_API_KEY=sk-xxx              # Kimi（月之暗面）模型
export INCEPTBENCH_API_KEY=xxx          # InceptBench 评估（必须）
```

### 最常用的命令

#### 1. 运行闭环（Grade 3 英语，与之前完全一致）

```bash
python main.py closed-loop \
  --model deepseek-reasoner \
  --pass-rate-target 0 \
  --max-rounds 10 \
  --run-id 0213_2 \
  --workers 10
```

不传 `--grade` 和 `--subject` 时，默认就是 Grade 3 ELA（英语语言艺术）。

#### 2. 运行闭环（其他年级/学科）

```bash
# AP 美国历史
python main.py closed-loop \
  --model deepseek-reasoner \
  --grade AP \
  --subject USHIST \
  --pass-rate-target 0 \
  --max-rounds 10 \
  --run-id ap_ushist_01 \
  --workers 10

# 5 年级数学
python main.py closed-loop \
  --model deepseek-reasoner \
  --grade 5 \
  --subject MATH \
  --pass-rate-target 0 \
  --max-rounds 10 \
  --run-id g5_math_01 \
  --workers 10

# 幼儿园科学
python main.py closed-loop \
  --model deepseek-reasoner \
  --grade K \
  --subject SCI \
  --pass-rate-target 0 \
  --max-rounds 5 \
  --run-id k_sci_01 \
  --workers 10
```

#### 3. 先试水再全量（推荐用于大规模学科）

标准数量较多的学科（如 AP USHIST 有 3,774 个组合），每轮全量生成耗时太长。
`--pilot-batch` 可以先用小批量跑闭环积累范例和规则，最后自动全量生成：

```bash
# AP 美国历史：每轮试水 50 题 × 5 轮，最后全量 3774 题
python main.py closed-loop \
  --model deepseek-reasoner \
  --grade AP --subject USHIST \
  --pilot-batch 50 \
  --pass-rate-target 0 \
  --max-rounds 5 \
  --run-id ap_ushist_01 \
  --workers 10
```

执行流程：
```
试水第 1 轮：随机抽 50 题 → 评分 → 积累范例 + 规则
试水第 2 轮：再抽 50 题（带新范例）→ 评分 → 更新
...
试水第 5 轮：再抽 50 题 → 评分 → 更新
全量生成：  用积累的范例和规则，一次出完 3774 题 → 评分
```

- **不设 `--pilot-batch`**：每轮都是全量（适合标准数少的学科，如 Grade 3 ELA 只有 237 题）
- **设了 `--pilot-batch 50`**：前 N 轮每轮只随机抽 50 题试水，最后一轮自动全量

各学科标准数参考：

| 学科 | 标准数 | 每轮全量题数 | 建议 |
|------|--------|------------|------|
| Grade 3 ELA | 79 | 237 | 不需要 pilot |
| Grade 5 MATH | 204 | 612 | 可选 pilot |
| AP BIO | 615 | 1,845 | 建议 --pilot-batch 50-100 |
| AP USHIST | 1,258 | 3,774 | 强烈建议 --pilot-batch 50 |

#### 4. 只生成题目（不跑闭环）

```bash
python scripts/generate_mcq.py \
  --model deepseek-reasoner \
  --grade AP \
  --subject BIO \
  --all-combinations \
  --output evaluation_output/mcqs_ap_bio.json \
  --workers 10
```

#### 5. 只评估已有题目

```bash
python main.py evaluate \
  --input evaluation_output/mcqs_ap_bio.json \
  --output evaluation_output/results_ap_bio.json \
  --parallel 20
```

#### 6. 查看可用的年级和学科组合

输入任意无效组合即可看到完整列表：

```bash
python main.py closed-loop --grade 0 --subject XXX --model deepseek-chat
```

---

## 闭环每一轮做了什么？

```
第 N 轮
  ├── [1/4] 生成：调用大模型，为每个（标准, 难度）组合生成 1 道选择题
  ├── [2/4] 评估：提交到 InceptBench 评分，得到每道题的分数
  ├── [3/4] 补示例：找到高分题（≥ 0.85）作为下一轮的 few-shot 范例
  └── [4/4] 改提示词：从低分题的反馈中提取规则，写入提示词
```

### 通过率计算

- 每道题 InceptBench 评分 0~1 分
- 得分 ≥ 0.85 算"通过"
- 通过率 = 通过题数 / 总题数 × 100%
- 目标通过率默认 95%（可通过 `--pass-rate-target` 设置，设为 0 表示跑满轮数取最高）

---

## 关键参数说明

| 参数 | 含义 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | 生成题目的大模型 | deepseek-chat | deepseek-reasoner, kimi-k2.5 |
| `--grade` | 年级 | 3 | K, 1-8, 10, 11, AP, HS, SAT |
| `--subject` | 学科缩写 | ELA | MATH, SCI, USHIST 等 |
| `--pilot-batch` | 试水批量（每轮题数） | 无（全量） | 50 表示每轮试水 50 题 |
| `--pass-rate-target` | 通过率目标（%） | 95 | 0 表示不设目标 |
| `--max-rounds` | 最大循环轮数 | 10 | |
| `--workers` | 生成阶段并发线程数 | 按模型自动 | deepseek-reasoner 建议 10-15 |
| `--run-id` | 运行批次标识 | 无 | 不同批次互不覆盖 |
| `--parallel` | 评估阶段并发数 | 20 | |

---

## 新学科的冷启动

对于从未跑过闭环的新学科（如 AP 生物），系统会自动冷启动：

1. **第 1 轮**：没有范例题，依赖课程元数据（学习目标、评估范围、常见误解）引导模型出题
2. **第 1 轮评估后**：高分题自动成为范例，低分反馈写入规则
3. **第 2 轮起**：进入正常闭环

AP 课程的优势：课程数据中包含丰富的元数据（学习目标、评估范围、常见误解），这些信息会自动注入提示词，帮助模型在没有范例的情况下也能出较高质量的题目。

---

## 输出文件

运行闭环后，在 `evaluation_output/` 目录下会生成：

| 文件 | 内容 |
|------|------|
| `mcqs_237_<model>_best_<rate>.json` | 最高通过率对应的题目 |
| `results_237_<model>_best_<rate>.json` | 最高通过率对应的评估结果 |
| `log_237_<model>.json` | 综合日志（每轮耗时、token 用量、通过率等） |

在 `processed_training_data/` 目录下：

| 文件 | 内容 |
|------|------|
| `examples_<model>_<run_id>.json` | 积累的高分范例题（few-shot 示例） |
| `prompt_rules_<model>_<run_id>.json` | 积累的改进规则 |

---

## 分享提示词给他人

如果只想把调好的提示词分享给别人，不需要整个代码仓库：

```bash
# 导出提示词包（Grade 3 ELA，DeepSeek 模型的最佳提示词）
python scripts/export_standalone_prompt.py \
  --examples processed_training_data/examples_deepseek-reasoner_0213_2.json \
  --prompt-rules processed_training_data/prompt_rules_deepseek-reasoner_0213_2.json \
  --output prompt_bundle_deepseek.json

# 导出 AP 美国历史的提示词包
python scripts/export_standalone_prompt.py \
  --grade AP \
  --subject USHIST \
  --output prompt_bundle_ap_ushist.json
```

对方只需要两个文件即可生成题目：
1. `prompt_bundle_xxx.json`（提示词包）
2. `scripts/run_with_bundle.py`（独立运行脚本）

```bash
pip install openai
export DEEPSEEK_API_KEY=sk-xxx
python run_with_bundle.py \
  --bundle prompt_bundle_deepseek.json \
  --all \
  --model deepseek-reasoner \
  --output mcqs.json
```
