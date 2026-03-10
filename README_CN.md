# K-12 ELA 题目生成 API 服务

[English](README.md) | 中文

通过 REST API 生成高质量的 K-12 ELA（英语语言艺术）题目 — 支持 MCQ（单选）、MSQ（多选）、Fill-in（填空）三种题型，后端通过 [OpenRouter](https://openrouter.ai) 调用 Gemini 3 Pro 模型（默认）。

服务采用完整的提示工程流水线：基于课程标准的 few-shot 示例、针对性提示规则、迭代式闭环优化，在 [InceptBench](https://benchmark.inceptbench.com) 评估中最大化题目质量。

## 特性

- **全题型支持**：MCQ（单选）、MSQ（多选）、Fill-in（填空）
- **1–12年级全覆盖**：完整的 Common Core ELA 课程标准，每个年级配有独立的 few-shot 示例库和提示规则
- **InceptBench 兼容输出**：响应格式与 InceptBench 评估 API 直接兼容
- **OpenRouter 统一接入**：通过一个 API Key 访问 Gemini 3 Pro、DeepSeek、Kimi、GPT、Claude 等模型
- **精细化提示工程**：按 `(标准, 难度, 题型)` 筛选 few-shot 示例 + 按 `(标准, 难度, 题型)` 注入针对性规则
- **闭环优化**：生成 → InceptBench 评估 → 更新提示规则 → 下一轮使用改进后的提示
- **轻量部署**：无需 GPU — 任何有 Python 3.10+ 的机器即可运行

## 快速部署

### 方式 A：Docker 部署（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. 构建镜像（约200MB，无 GPU 依赖）
docker build -t ela-question-generator .

# 3. 启动服务（只需两个 Key）
docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=你的_openrouter_api_key \
  -e INCEPTBENCH_API_KEY=你的_inceptbench_api_key \
  ela-question-generator

# 4. 验证服务
curl http://localhost:8000/health
```

### 方式 B：直接运行 Python

```bash
# 1. 克隆仓库
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. 安装轻量依赖（无需 torch/transformers）
pip install -r requirements-api.txt

# 3. 设置 API Key
export OPENROUTER_API_KEY=你的_openrouter_api_key
export INCEPTBENCH_API_KEY=你的_inceptbench_api_key

# 4. 启动服务
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 验证服务是否正常

```bash
curl http://localhost:8000/health

# 预期响应
{
  "status": "healthy",
  "default_model": "or/gemini-3-pro",
  "loaded_grades": ["1","2","3","4","5","6","7","8","9","10","11","12"]
}
```

## API 接口一览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 — 返回已加载年级和默认模型 |
| `/models` | GET | 列出所有可用 OpenRouter 模型 |
| `/grades/{grade}/combinations` | GET | 查询某年级的所有（标准, 难度, 题型）组合 |
| `/generate` | POST | 生成单道题目 |
| `/generate-all` | POST | 批量生成某年级的全部组合 |
| `/docs` | GET | 交互式 Swagger UI 文档 |

### 生成单道题目

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "5",
    "standard": "CCSS.ELA-LITERACY.L.5.1.A",
    "difficulty": "medium",
    "type": "mcq"
  }'
```

响应格式（InceptBench 兼容）：

```json
{
  "generated_content": [{
    "id": "L.5.1.A-conjunctions-medium-001",
    "request": {
      "grade": "5", "subject": "ELA", "type": "mcq",
      "difficulty": "medium", "locale": "en-US",
      "skills": {
        "lesson_title": "K-12 ELA",
        "substandard_id": "CCSS.ELA-LITERACY.L.5.1.A",
        "substandard_description": "..."
      }
    },
    "content": {
      "question": "Which conjunction best completes the sentence?...",
      "answer": "B",
      "answer_options": [
        {"key": "A", "text": "..."},
        {"key": "B", "text": "..."},
        {"key": "C", "text": "..."},
        {"key": "D", "text": "..."}
      ],
      "answer_explanation": "Option B is correct because..."
    },
    "image_url": [],
    "metadata": {"generated_question_id": "..."},
    "verbose": false
  }]
}
```

### 批量生成某年级全部题目

```bash
curl -X POST http://localhost:8000/generate-all \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "5",
    "subject": "ELA",
    "type": "all",
    "workers": 10
  }'
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `grade` | string | （必填） | 年级（1-12） |
| `subject` | string | `"ELA"` | 学科 |
| `type` | string | `"all"` | 题型：`mcq`/`msq`/`fill-in`/`all` |
| `model` | string | `or/gemini-3-pro` | 指定模型（见可用模型列表） |
| `workers` | int | `10` | 并发线程数（1-50） |

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `OPENROUTER_API_KEY` | 是 | — | OpenRouter API Key（[openrouter.ai](https://openrouter.ai) 注册获取） |
| `INCEPTBENCH_API_KEY` | 是 | — | InceptBench 评估 API Token |
| `DEFAULT_MODEL` | 否 | `or/gemini-3-pro` | 默认生成模型 |
| `PRELOAD_GRADES` | 否 | `1,2,...,12` | 启动时预加载的年级（逗号分隔） |
| `EVALUATOR_TOKEN` | 否 | — | 备用 InceptBench Token（api.inceptbench.com） |

## 可用模型

所有模型通过 [OpenRouter](https://openrouter.ai) 统一接入，只需一个 `OPENROUTER_API_KEY`。

| 模型 ID | OpenRouter 模型 | 说明 |
|---------|----------------|------|
| `or/gemini-3-pro` | `google/gemini-3-pro-preview` | **默认** — Gemini 3 Pro，质量最佳 |
| `or/deepseek-v3.2` | `deepseek/deepseek-chat-v3-0324` | DeepSeek V3.2 |
| `or/kimi-k2.5` | `moonshotai/kimi-k2.5` | Kimi K2.5 |
| `or/glm-5` | `z-ai/glm-5` | GLM-5（智谱） |
| `or/gpt-5.2` | `openai/gpt-5.2` | GPT-5.2 |
| `or/claude-sonnet` | `anthropic/claude-sonnet-4.6` | Claude Sonnet 4.6 |
| `or/gemini-3-flash` | `google/gemini-3-flash-preview` | Gemini 3 Flash（快速、低成本） |

指定其他模型时，在请求中传入 `model` 参数：

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "8",
    "standard": "CCSS.ELA-LITERACY.L.8.1.A",
    "difficulty": "hard",
    "type": "fill-in",
    "model": "or/deepseek-v3.2"
  }'
```

## 项目结构

```
qwen32b-sft/
├── api_service/
│   ├── fastapi_app.py          # FastAPI 应用入口
│   ├── remote_generator.py     # OpenRouter 模型调用与生成逻辑
│   └── schemas.py              # 请求/响应数据模型
├── data_processing/
│   ├── build_prompt.py         # 提示构建（few-shot + 规则注入）
│   ├── analyze_dimensions.py   # 年级/标准/难度组合分析
│   └── select_examples.py      # 示例选择与验证
├── evaluation/
│   └── inceptbench_client.py   # InceptBench 评估客户端
├── scripts/
│   ├── generate_questions.py   # 核心生成逻辑
│   └── validate_mcq.py         # MCQ 验证与修复
├── data/
│   └── curriculum_standards.json   # CCSS 课程标准数据
├── processed_training_data/        # Few-shot 示例 + 提示规则
│   ├── {N}_ELA_examples.json       # 各年级 few-shot 示例（1-12）
│   └── {N}_ELA_prompt_rules_*.json # 各年级提示规则
├── evaluation_output/              # 最佳评估结果
├── docs/                           # 报告与文档
├── requirements-api.txt        # 轻量 API 依赖
├── Dockerfile                  # 容器构建文件
└── .env.example                # 环境变量模板
```

## 第三方完整部署指南

以下为在全新服务器上从零部署 API 服务的详细步骤。

### 前提条件

- Python 3.10+ 或 Docker
- OpenRouter API Key — 在 [openrouter.ai](https://openrouter.ai) 注册获取
- InceptBench 评估 Token — 在 [InceptBench](https://benchmark.inceptbench.com) 获取

### 步骤一：获取代码

```bash
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft
```

### 步骤二（Docker 方式）

```bash
docker build -t ela-question-generator .

docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=你的_api_key \
  -e INCEPTBENCH_API_KEY=你的_inceptbench_key \
  --restart unless-stopped \
  ela-question-generator

docker logs -f ela-api
```

### 步骤二（Python 方式）

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements-api.txt

export OPENROUTER_API_KEY=你的_api_key
export INCEPTBENCH_API_KEY=你的_inceptbench_key

# 前台运行
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000

# 或后台运行
nohup uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### 步骤三：验证

```bash
# 1. 健康检查
curl http://localhost:8000/health

# 2. 查看可用模型
curl http://localhost:8000/models

# 3. 查看 5 年级所有组合
curl http://localhost:8000/grades/5/combinations

# 4. 生成一道题（5年级 MCQ）
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "5",
    "standard": "CCSS.ELA-LITERACY.L.5.1.A",
    "difficulty": "medium",
    "type": "mcq"
  }'

# 5. 生成一道填空题（8年级）
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "8",
    "standard": "CCSS.ELA-LITERACY.L.8.1.A",
    "difficulty": "hard",
    "type": "fill-in"
  }'

# 6. 批量生成（5年级全部组合）
curl -X POST http://localhost:8000/generate-all \
  -H "Content-Type: application/json" \
  -d '{"grade": "5", "workers": 10}'
```

### 步骤四：生产环境建议

```bash
pip install gunicorn
gunicorn api_service.fastapi_app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `429 Too Many Requests` | OpenRouter 限流 | 降低 `workers` 参数（建议 10-20） |
| 启动时加载慢 | 预加载 12 个年级 | 设置 `PRELOAD_GRADES=3,5` 只加载需要的年级 |
| 生成超时 | 网络或模型服务慢 | 内置自动重试机制，可调整 `max_retries` |
| 评估失败 | InceptBench Token 无效 | 检查 `INCEPTBENCH_API_KEY` 是否正确 |

## 评估

题目通过 [InceptBench](https://benchmark.inceptbench.com) API 进行评估：

- **API 地址**: `https://inceptbench.api.inceptlabs.ai/2.3.3/evaluate`
- **认证方式**: Bearer Token，通过 `INCEPTBENCH_API_KEY` 环境变量配置
- **备用地址**: `https://api.inceptbench.com/evaluate`，通过 `EVALUATOR_TOKEN` 环境变量配置

评估系统对每道题在 0–1 分之间打分，评估维度包括课程标准对齐度、难度适当性、答案正确性和解释质量。

## 许可证

MIT
