# K-12 ELA 题目生成 API 服务

[English](README.md) | 中文

通过 REST API 生成高质量的 K-12 ELA（英语语言艺术）题目 — 支持 MCQ（单选）、MSQ（多选）、Fill-in（填空）三种题型，后端由远程 LLM 模型（Fireworks AI、DeepSeek、Kimi 等）驱动。

服务采用完整的提示工程流水线：基于课程标准的 few-shot 示例、针对性提示规则、迭代式闭环优化，在 [InceptBench](https://benchmark.inceptbench.com) 评估中最大化题目质量。

## 特性

- **全题型支持**：MCQ（单选）、MSQ（多选）、Fill-in（填空）
- **1–12年级全覆盖**：完整的 Common Core ELA 课程标准，每个年级配有独立的 few-shot 示例库
- **InceptBench 兼容输出**：响应格式与 InceptBench 评估 API 直接兼容
- **多模型后端**：Fireworks AI（kimi-k2.5、DeepSeek、GLM-5 等）、DeepSeek 直连、Kimi 直连、Gemini
- **精细化提示工程**：按 `(标准, 难度, 题型)` 筛选 few-shot 示例 + 按 `(标准, 难度, 题型)` 注入针对性规则
- **轻量部署**：无需 GPU — 任何有 Python 3.10+ 的机器即可运行

## 快速部署

### 方式 A：Docker 部署（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. 构建镜像（约200MB，无 GPU 依赖）
docker build -t ela-question-generator .

# 3. 启动服务
docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e FIREWORKS_API_KEY=你的_fireworks_api_key \
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
export FIREWORKS_API_KEY=你的_fireworks_api_key

# 4. 启动服务
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 验证服务是否正常

```bash
# 健康检查
curl http://localhost:8000/health

# 预期响应
{
  "status": "ok",
  "default_model": "fw/kimi-k2.5",
  "loaded_grades": ["1","2","3","4","5","6","7","8","9","10","11","12"]
}
```

## API 接口一览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 — 返回已加载年级和默认模型 |
| `/models` | GET | 列出所有可用模型（Fireworks + 直连） |
| `/grades/{grade}/combinations` | GET | 查询某年级的所有（标准, 难度, 题型）组合 |
| `/generate` | POST | 生成单道题目 |
| `/generate-all` | POST | 批量生成某年级的全部组合 |
| `/docs` | GET | 交互式 Swagger UI 文档 |

### 生成单道题目

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "3",
    "standard": "CCSS.ELA-LITERACY.L.3.1.A",
    "difficulty": "medium",
    "type": "mcq"
  }'
```

响应格式（InceptBench 兼容）：

```json
{
  "generated_content": [{
    "id": "L.3.1.A-pronoun-function-medium-001",
    "request": {
      "grade": "3", "subject": "ELA", "type": "mcq",
      "difficulty": "medium", "locale": "en-US",
      "skills": {
        "lesson_title": "K-12 ELA",
        "substandard_id": "CCSS.ELA-LITERACY.L.3.1.A",
        "substandard_description": "..."
      }
    },
    "content": {
      "question": "Which choice best describes the function of...",
      "answer": "C",
      "answer_options": [
        {"key": "A", "text": "..."},
        {"key": "B", "text": "..."},
        {"key": "C", "text": "..."},
        {"key": "D", "text": "..."}
      ],
      "answer_explanation": "Option C is correct because..."
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
    "grade": "3",
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
| `type` | string | `"all"` | 题型：`mcq`/`msq`/`fill_in`/`all` |
| `model` | string | 服务默认值 | 覆盖默认模型 |
| `workers` | int | `10` | 并发线程数 |
| `max_retries` | int | `3` | 每道题最大重试次数 |

## 环境变量

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `FIREWORKS_API_KEY` | 是 | — | Fireworks AI 的 API Key |
| `DEFAULT_MODEL` | 否 | `fw/kimi-k2.5` | 默认生成模型 |
| `PRELOAD_GRADES` | 否 | `1,2,...,12` | 启动时预加载的年级（逗号分隔） |
| `DEEPSEEK_API_KEY` | 否 | — | 使用 DeepSeek 模型时需要 |
| `KIMI_API_KEY` | 否 | — | 使用 Kimi 直连时需要 |
| `GEMINI_API_KEY` | 否 | — | 使用 Gemini 模型时需要 |

## 可用模型

### 通过 Fireworks AI（推荐）

| 模型 ID | 说明 |
|---------|------|
| `fw/kimi-k2.5` | Kimi K2.5（默认推荐） |
| `fw/deepseek-v3.2` | DeepSeek V3.2 |
| `fw/deepseek-r1` | DeepSeek R1（推理增强） |
| `fw/glm-5` | GLM-5（智谱） |
| `fw/gpt-oss-120b` | GPT-OSS 120B |
| `fw/qwen3-235b` | Qwen3 235B |

### 直连 API

| 模型 ID | 需要的 API Key |
|---------|---------------|
| `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `kimi-latest` | `KIMI_API_KEY` |
| `gemini-3-flash-preview` | `GEMINI_API_KEY` |

## 项目结构（API 服务相关）

```
qwen32b-sft/
├── api_service/
│   ├── fastapi_app.py          # FastAPI 应用入口
│   ├── remote_generator.py     # 远程模型调用与生成逻辑
│   └── schemas.py              # 请求/响应数据模型
├── data_processing/
│   ├── build_prompt.py         # 提示构建（few-shot + 规则注入）
│   ├── analyze_dimensions.py   # 年级/标准/难度组合分析
│   └── select_examples.py      # JSON 解析工具
├── evaluation/
│   └── inceptbench_client.py   # InceptBench 格式转换
├── scripts/
│   ├── generate_questions.py   # 核心生成逻辑
│   └── validate_mcq.py         # MCQ 验证与修复
├── data/
│   └── curriculum_standards.json   # CCSS 课程标准数据
├── processed_training_data/        # Few-shot 示例 + 提示规则
│   ├── {N}_ELA_examples.json       # 各年级 few-shot 示例（1-12）
│   └── {N}_ELA_prompt_rules_*.json # 各年级提示规则
├── requirements-api.txt        # 轻量 API 依赖
└── Dockerfile                  # 容器构建文件
```

## 第三方完整部署指南

以下为在全新服务器上从零部署 API 服务的详细步骤。

### 前提条件

- Python 3.10+ 或 Docker
- Fireworks AI 的 API Key（在 [fireworks.ai](https://fireworks.ai) 注册获取）
- 网络可访问 Fireworks AI API

### 步骤一：获取代码

```bash
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft
```

### 步骤二（Docker 方式）

```bash
# 构建镜像
docker build -t ela-question-generator .

# 启动容器
docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e FIREWORKS_API_KEY=你的_api_key \
  -e DEFAULT_MODEL=fw/kimi-k2.5 \
  -e PRELOAD_GRADES=1,2,3,4,5,6,7,8,9,10,11,12 \
  --restart unless-stopped \
  ela-question-generator

# 查看启动日志
docker logs -f ela-api
```

### 步骤二（Python 方式）

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements-api.txt

# 设置环境变量
export FIREWORKS_API_KEY=你的_api_key
export DEFAULT_MODEL=fw/kimi-k2.5
export PRELOAD_GRADES=1,2,3,4,5,6,7,8,9,10,11,12

# 启动服务（前台运行）
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

# 3. 查看 3 年级所有组合
curl http://localhost:8000/grades/3/combinations

# 4. 生成一道题
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "3",
    "standard": "CCSS.ELA-LITERACY.L.3.1.A",
    "difficulty": "medium",
    "type": "mcq"
  }'

# 5. 批量生成（3年级全部组合）
curl -X POST http://localhost:8000/generate-all \
  -H "Content-Type: application/json" \
  -d '{"grade": "3", "workers": 10}'
```

### 步骤四：生产环境建议

```bash
# 使用 gunicorn + uvicorn workers 提升并发（可选）
pip install gunicorn
gunicorn api_service.fastapi_app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300

# 或在 Docker 中指定多进程
docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e FIREWORKS_API_KEY=你的_api_key \
  ela-question-generator \
  gunicorn api_service.fastapi_app:app \
    -w 4 -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 --timeout 300
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| `429 Too Many Requests` | Fireworks 限流 | 降低 `workers` 参数（建议 10-20） |
| 启动时加载慢 | 预加载 12 个年级 | 设置 `PRELOAD_GRADES=3,5` 只加载需要的年级 |
| 模型不可用 | 未设置对应 API Key | 检查环境变量，或使用默认的 `fw/kimi-k2.5` |
| 生成超时 | 网络或模型服务慢 | 内置自动重试机制，可调整 `max_retries` |

## 闭环优化（开发用）

用于迭代提升题目质量的闭环流水线（生成 → 评估 → 更新规则 → 下一轮使用改进后的提示）：

```bash
# 对 1-3 年级运行 3 轮闭环优化
MODELS='fw/kimi-k2.5' GRADES='1,2,3' ROUNDS=3 bash scripts/run_matrix.sh
```

## 许可证

MIT
