# K-12 ELA Question Generator API

English | [中文](README_CN.md)

Generate high-quality K-12 ELA (English Language Arts) questions — MCQ, MSQ, and Fill-in — via a REST API backed by remote LLM models through [OpenRouter](https://openrouter.ai) (Gemini, DeepSeek, Kimi, etc.) and other providers.

The service uses a full prompt-engineering pipeline: few-shot examples, curriculum-aligned prompt rules, and iterative closed-loop refinement to maximize question quality on the [InceptBench](https://benchmark.inceptbench.com) evaluation.

## Features

- **All question types**: MCQ (multiple-choice), MSQ (multi-select), Fill-in
- **Grades 1–12**: Full Common Core ELA coverage with per-grade few-shot examples and prompt rules
- **InceptBench-compatible output**: Response format matches InceptBench evaluation API directly
- **Multiple model backends**: OpenRouter (Gemini 3 Pro, DeepSeek, Kimi, GPT, Claude, etc.), Fireworks AI, direct API (DeepSeek, Kimi, Gemini)
- **Prompt engineering pipeline**: Few-shot examples filtered by `(standard, difficulty, type)` + targeted prompt rules by `(standard, difficulty, type)`
- **Closed-loop refinement**: Generate → Evaluate via InceptBench → Update prompt rules → Next round with improved prompts
- **Lightweight deployment**: No GPU required — runs on any machine with Python 3.10+

## Quick Start

### Option A: Docker (Recommended)

```bash
# 1. Clone
git clone https://github.com/bill-lee-mk/qwen32b-sft.git
cd qwen32b-sft

# 2. Build image (~200MB, no GPU dependencies)
docker build -t ela-question-generator .

# 3. Run
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns loaded grades and default model |
| `/models` | GET | List available models (OpenRouter + Fireworks + direct) |
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
| `OPENROUTER_API_KEY` | Yes | — | OpenRouter API key (default model backend) |
| `INCEPTBENCH_API_KEY` | Yes | — | InceptBench evaluation API token |
| `DEFAULT_MODEL` | No | `or/gemini-3-pro` | Default generation model |
| `PRELOAD_GRADES` | No | `1,2,...,12` | Grades to preload at startup |
| `FIREWORKS_API_KEY` | No | — | Required if using Fireworks AI models |
| `DEEPSEEK_API_KEY` | No | — | Required if using DeepSeek direct |
| `KIMI_API_KEY` | No | — | Required if using Kimi direct |
| `GEMINI_API_KEY` | No | — | Required if using Gemini direct |
| `EVALUATOR_TOKEN` | No | — | Fallback InceptBench token (api.inceptbench.com) |

## Available Models

### Via OpenRouter (recommended)

| Model ID | OpenRouter Model | Description |
|----------|-----------------|-------------|
| `or/gemini-3-pro` | `google/gemini-3-pro-preview` | Gemini 3 Pro (default, best quality) |
| `or/deepseek-v3.2` | `deepseek/deepseek-chat-v3-0324` | DeepSeek V3.2 |
| `or/kimi-k2.5` | `moonshotai/kimi-k2.5` | Kimi K2.5 |
| `or/glm-5` | `z-ai/glm-5` | GLM-5 (Zhipu) |
| `or/gpt-5.2` | `openai/gpt-5.2` | GPT-5.2 |
| `or/claude-sonnet` | `anthropic/claude-sonnet-4.6` | Claude Sonnet 4.6 |
| `or/gemini-3-flash` | `google/gemini-3-flash-preview` | Gemini 3 Flash (fast, cheaper) |

### Via Fireworks AI

| Model ID | Description |
|----------|-------------|
| `fw/kimi-k2.5` | Kimi K2.5 |
| `fw/deepseek-v3.2` | DeepSeek V3.2 |
| `fw/glm-5` | GLM-5 |

### Direct API

| Model ID | Requires |
|----------|----------|
| `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `kimi-latest` | `KIMI_API_KEY` |
| `gemini-3-flash-preview` | `GEMINI_API_KEY` |

## Project Structure

```
qwen32b-sft/
├── api_service/
│   ├── fastapi_app.py          # FastAPI application
│   ├── remote_generator.py     # Remote model generation logic
│   └── schemas.py              # Request/response models
├── data_processing/
│   ├── build_prompt.py         # Prompt construction (few-shot + rules)
│   ├── analyze_dimensions.py   # Grade/standard/difficulty combinations
│   └── select_examples.py      # Example selection & validation
├── evaluation/
│   └── inceptbench_client.py   # InceptBench evaluation client
├── scripts/
│   ├── generate_questions.py   # Core generation logic
│   ├── run_matrix.sh           # Batch run orchestrator
│   └── validate_mcq.py         # MCQ validation/repair
├── data/
│   └── curriculum_standards.json   # CCSS curriculum data
├── processed_training_data/        # Few-shot examples + prompt rules
│   ├── {N}_ELA_examples.json       # Per-grade examples (1-12)
│   └── {N}_ELA_prompt_rules_*.json # Per-grade per-model prompt rules
├── evaluation_output/              # Best evaluation results per model
├── docs/                           # Reports & documentation
├── requirements-api.txt        # Lightweight API dependencies
├── Dockerfile                  # Container build file
└── .env.example                # Environment variable template
```

## Closed-Loop Refinement (Development)

For iterating on question quality, the project includes a closed-loop pipeline:

```bash
# Run 5 cycles of generate → evaluate → improve-prompt for all grades
export OPENROUTER_API_KEY=your_key
export INCEPTBENCH_API_KEY=your_key
MODELS='or/gemini-3-pro' GRADES='1,2,3' CYCLES=5 bash scripts/run_matrix.sh
```

Each cycle: generates all questions → evaluates via InceptBench API → extracts feedback → updates targeted prompt rules → next cycle uses improved prompts.

## Evaluation

Questions are evaluated using the [InceptBench](https://benchmark.inceptbench.com) API:

- **API URL**: `https://inceptbench.api.inceptlabs.ai/2.3.3/evaluate`
- **Authentication**: Bearer token via `INCEPTBENCH_API_KEY`
- **Fallback**: `https://api.inceptbench.com/evaluate` via `EVALUATOR_TOKEN`

The evaluation scores each question on a 0–1 scale based on curriculum alignment, difficulty appropriateness, answer correctness, and explanation quality.

## License

MIT
