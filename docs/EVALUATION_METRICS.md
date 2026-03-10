# 评估指标与无分数题处理

## 评估 API 环境变量配置

**所有 Key/Token 均从环境变量读取，请勿在代码或配置文件中硬编码。**

### 主配置（第一套）

| 环境变量 | 说明 |
|----------|------|
| `INCEPTBENCH_API_KEY` 或 `INCEPTBENCH_TOKEN` | 主 token，用于 inceptbench.api.inceptlabs.ai |
| `INCEPTBENCH_URL` | 可选，主 API 地址，默认 `https://inceptbench.api.inceptlabs.ai/2.3.0/evaluate` |

### 第二套配置（主配置失败时自动降级）

| 环境变量 | 说明 |
|----------|------|
| `EVALUATOR_TOKEN` | 第二套 token（URL 固定在代码中为 api.inceptbench.com） |

### 设置示例

```bash
# 主配置（必填）
export INCEPTBENCH_API_KEY="你的主token"

# 第二套（可选，主配置失败时自动切换）
export EVALUATOR_TOKEN="你的第二套token"
```

也可在运行前一次性设置：

```bash
INCEPTBENCH_API_KEY="主token" EVALUATOR_TOKEN="第二套token" python main.py evaluate --input mcqs.json
```

或写入 `~/.bashrc` / `~/.profile` 持久生效：

```bash
echo 'export EVALUATOR_TOKEN="你的第二套token"' >> ~/.bashrc
source ~/.bashrc
```

---

## 定义

| 名称 | 含义 |
|------|------|
| **总评估数** | 提交给评分服务（InceptBench）的题目数，即 `mcqs` 文件中的题目条数。 |
| **有效分数数** | 评分服务返回了数值分数（`overall_score` 或 `evaluations.*.overall.score`）的题目数。 |
| **无分数题** | 总评估数 − 有效分数数。即请求失败、超时或服务返回错误导致没有数值分数的题目。 |

## 通过率分母为什么是「有效分数数」而不是「总评估数」？

- **默认通过率**：`通过数 / 有效分数数`。  
  无分数题既不算通过也不算不通过，无法赋予 0/1，因此不参与通过率计算；分母用「有效分数数」更合理。
- **若将无分数按不通过计**：`通过数 / 总评估数` 会在汇总中以「若将无分数题按不通过计」单独给出，便于对比。

## 无分数题：区分服务端问题与题目问题

评估结果中会为每条无分数题记录：

- **reason**：接口返回的错误信息（如 `Could not save evaluation result to the DB: 'overall'`）。
- **classification**：
  - **service**：判定为评分服务端问题（如 DB 写入失败、500/503、超时、`overall` 字段缺失等）。建议重试或联系评分方。
  - **question_or_request**：判定为题目或请求问题（如格式不符、校验失败等）。建议检查题目内容或请求 payload。

输出与排查方式：

- 汇总中会打印「无分数题目明细」及每条判定（服务端 / 题目/请求）。
- `results_240.json` 中的 **error_details** 数组含全部无分数题明细。
- 若存在无分数题，会额外写出 **results_240_errors.json**（或 `*_errors.json`），便于单独打开排查。

## 建议流程

1. 若有 **service** 类无分数题：先重跑评估（或对这几题单独重试）；若持续出现，联系 InceptBench 服务方。
2. 若有 **question_or_request** 类：检查对应题目的 JSON（题干、选项、答案、解析等）是否符合 InceptBench 要求，必要时修正题目或生成逻辑。

## 闭环输出文件（closed-loop）

| 文件 | 说明 |
|------|------|
| `evaluation_output/mcqs_237_<model>_roundN.json` | 第 N 轮生成的题目（79 标准×3 难度=237 题） |
| `evaluation_output/results_237_<model>_roundN.json` | 第 N 轮评估结果 |
| `evaluation_output/mcqs_237_<model>_best_{rate}.json` | 历史最高通过率对应轮次的题目（如 best_94_5.json 表示 94.5% 通过率） |
| `evaluation_output/results_237_<model>_best_{rate}.json` | 历史最高通过率对应轮次的评估结果 |
| `evaluation_output/closed_loop_progress_<model>.json` | 每轮更新的进度汇总（轮次、通过率、最佳路径等），可随时查看 |

**轮次保存**：每一轮都会写入 `round1.json`、`round2.json`、…、`roundN.json`；最佳轮次会额外复制到 `best_{rate}.json`。下次运行时会覆盖同名的 roundN 文件，仅保留本次运行各轮。

**批次隔离**：若本次 10 轮、下次 20 轮，默认会互相覆盖：examples、prompt_rules 会承接；round1–10 会被新 run 覆盖；best 会被更高分覆盖。需隔离时使用 `--run-id <id>`，例如 `--run-id exp1`，则所有路径加此后缀（如 `mcqs_237_deepseek-reasoner_exp1_round1.json`），不同批次互不影响。

**查看方式**：运行 `closed-loop` 时，每轮结束会更新 `closed_loop_progress_<model>.json`。`Ctrl+C` 中断或跑满后，控制台会打印汇总；也可直接 `cat evaluation_output/closed_loop_progress_<model>.json` 查看。使用 `--log-file <路径>` 可将运行日志同时写入文件。
