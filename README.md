# K-12 ELA Question Generator API

English | [中文](README_CN.md)

Generate high-quality K-12 ELA (English Language Arts) questions — MCQ, MSQ, and Fill-in — via a REST API powered by [OpenRouter](https://openrouter.ai) (Gemini 3 Pro by default).

The service uses a full prompt-engineering pipeline: few-shot examples, curriculum-aligned prompt rules, and iterative closed-loop refinement to maximize question quality on the [InceptBench](https://benchmark.inceptbench.com) evaluation.

## Features

- **All question types**: MCQ (multiple-choice), MSQ (multi-select), Fill-in
- **Grades 1–12**: Full Common Core ELA coverage with per-grade few-shot examples and prompt rules
- **InceptBench-compatible output**: Response format matches InceptBench evaluation API directly
- **OpenRouter integration**: Access to Gemini 3 Pro, DeepSeek, Kimi, GPT, Claude and more via a single API key
- **Prompt engineering pipeline**: Few-shot examples filtered by `(standard, difficulty, type)` + targeted prompt rules
- **Closed-loop refinement**: Generate → Evaluate via InceptBench → Update prompt rules → Repeat
- **Lightweight deployment**: No GPU required — runs on any machine with Python 3.10+

## Quick Start

### Option A: Docker (Recommended)

```bash
# 1. Clone
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. Build image (~200MB, no GPU dependencies)
docker build -t ela-question-generator .

# 3. Run (only 2 keys needed)
docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_openrouter_api_key \
  -e INCEPTBENCH_API_KEY=your_inceptbench_api_key \
  ela-question-generator

# 4. Verify
curl http://localhost:8000/health
```

### Option B: Direct Python

```bash
# 1. Clone
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. Install lightweight dependencies (no torch/transformers needed)
pip install -r requirements-api.txt

# 3. Set API keys
export OPENROUTER_API_KEY=your_openrouter_api_key
export INCEPTBENCH_API_KEY=your_inceptbench_api_key

# 4. Start server
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000
```

### Verify

```bash
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "default_model": "or/gemini-3-pro",
  "loaded_grades": ["1","2","3","4","5","6","7","8","9","10","11","12"]
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns loaded grades and default model |
| `/models` | GET | List available OpenRouter models |
| `/grades/{grade}/combinations` | GET | List all (standard, difficulty, type) combinations for a grade |
| `/generate` | POST | Generate a single question |
| `/generate-all` | POST | Batch-generate all combinations for a grade |
| `/docs` | GET | Interactive Swagger UI documentation |

### Generate a Single Question

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

Response (InceptBench format):

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

### Batch-Generate All Combinations for a Grade

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

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key ([openrouter.ai](https://openrouter.ai)) |
| `INCEPTBENCH_API_KEY` | Yes | — | InceptBench evaluation API token |
| `DEFAULT_MODEL` | No | `or/gemini-3-pro` | Default generation model |
| `PRELOAD_GRADES` | No | `1,2,...,12` | Grades to preload at startup |
| `EVALUATOR_TOKEN` | No | — | Fallback InceptBench token (api.inceptbench.com) |

## Available Models

All models are accessed through [OpenRouter](https://openrouter.ai) with a single `OPENROUTER_API_KEY`.

| Model ID | OpenRouter Model | Description |
|----------|-----------------|-------------|
| `or/gemini-3-pro` | `google/gemini-3-pro-preview` | **Default** — Gemini 3 Pro, best quality |
| `or/deepseek-v3.2` | `deepseek/deepseek-chat-v3-0324` | DeepSeek V3.2 |
| `or/kimi-k2.5` | `moonshotai/kimi-k2.5` | Kimi K2.5 |
| `or/glm-5` | `z-ai/glm-5` | GLM-5 (Zhipu) |
| `or/gpt-5.2` | `openai/gpt-5.2` | GPT-5.2 |
| `or/claude-sonnet` | `anthropic/claude-sonnet-4.6` | Claude Sonnet 4.6 |
| `or/gemini-3-flash` | `google/gemini-3-flash-preview` | Gemini 3 Flash (fast, cheaper) |

To use a different model, pass the `model` parameter in the request:

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

## Project Structure

```
qwen32b-sft/
├── api_service/
│   ├── fastapi_app.py          # FastAPI application
│   ├── remote_generator.py     # OpenRouter model generation logic
│   └── schemas.py              # Request/response models
├── data_processing/
│   ├── build_prompt.py         # Prompt construction (few-shot + rules)
│   ├── analyze_dimensions.py   # Grade/standard/difficulty combinations
│   └── select_examples.py      # Example selection & validation
├── evaluation/
│   └── inceptbench_client.py   # InceptBench evaluation client
├── scripts/
│   ├── generate_questions.py   # Core generation logic
│   └── validate_mcq.py         # MCQ validation/repair
├── data/
│   └── curriculum_standards.json   # CCSS curriculum data
├── processed_training_data/        # Few-shot examples + prompt rules
│   ├── {N}_ELA_examples.json       # Per-grade examples (1-12)
│   └── {N}_ELA_prompt_rules_*.json # Per-grade prompt rules
├── evaluation_output/              # Best evaluation results
├── docs/                           # Reports & documentation
├── requirements-api.txt        # Lightweight API dependencies
├── Dockerfile                  # Container build file
└── .env.example                # Environment variable template
```

## Full Deployment Guide

Step-by-step instructions for deploying on a fresh server.

### Prerequisites

- Python 3.10+ or Docker
- OpenRouter API key — sign up at [openrouter.ai](https://openrouter.ai)
- InceptBench evaluation token — obtain at [InceptBench](https://benchmark.inceptbench.com)

### Step 1: Get the Code

```bash
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft
```

### Step 2a: Docker Deployment

```bash
docker build -t ela-question-generator .

docker run -d \
  --name ela-api \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=your_api_key \
  -e INCEPTBENCH_API_KEY=your_inceptbench_key \
  --restart unless-stopped \
  ela-question-generator

docker logs -f ela-api
```

### Step 2b: Python Deployment

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements-api.txt

export OPENROUTER_API_KEY=your_api_key
export INCEPTBENCH_API_KEY=your_inceptbench_key

# Foreground
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000

# Or background
nohup uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### Step 3: Verify

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# View Grade 5 combinations
curl http://localhost:8000/grades/5/combinations

# Generate a Grade 5 MCQ
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "5",
    "standard": "CCSS.ELA-LITERACY.L.5.1.A",
    "difficulty": "medium",
    "type": "mcq"
  }'

# Generate a Grade 8 Fill-in (hard)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "grade": "8",
    "standard": "CCSS.ELA-LITERACY.L.8.1.A",
    "difficulty": "hard",
    "type": "fill-in"
  }'

# Batch-generate all Grade 5 combinations
curl -X POST http://localhost:8000/generate-all \
  -H "Content-Type: application/json" \
  -d '{"grade": "5", "workers": 10}'
```

### Step 4: Production

```bash
pip install gunicorn
gunicorn api_service.fastapi_app:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `429 Too Many Requests` | OpenRouter rate limit | Lower `workers` (recommend 10-20) |
| Slow startup | Preloading 12 grades | Set `PRELOAD_GRADES=3,5` to load only needed grades |
| Generation timeout | Network or model latency | Built-in retry; adjust `max_retries` |
| Evaluation fails | Invalid InceptBench token | Check `INCEPTBENCH_API_KEY` |

## Evaluation

Questions are evaluated using the [InceptBench](https://benchmark.inceptbench.com) API:

- **API URL**: `https://inceptbench.api.inceptlabs.ai/2.3.3/evaluate`
- **Authentication**: Bearer token via `INCEPTBENCH_API_KEY`
- **Fallback**: `https://api.inceptbench.com/evaluate` via `EVALUATOR_TOKEN`

Each question is scored 0–1 based on curriculum alignment, difficulty appropriateness, answer correctness, and explanation quality.

## License

MIT
