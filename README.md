# K-12 ELA Question Generator API

English | [中文](README_CN.md)

Generate high-quality K-12 ELA (English Language Arts) questions — MCQ, MSQ, and Fill-in — via a REST API backed by remote LLM models (Fireworks AI, DeepSeek, Kimi, etc.).

The service uses a full prompt-engineering pipeline: few-shot examples, curriculum-aligned prompt rules, and iterative closed-loop refinement to maximize question quality on the [InceptBench](https://benchmark.inceptbench.com) evaluation.

## Features

- **All question types**: MCQ (multiple-choice), MSQ (multi-select), Fill-in
- **Grades 1–12**: Full Common Core ELA coverage with per-grade few-shot examples
- **InceptBench-compatible output**: Response format matches InceptBench evaluation API directly
- **Multiple model backends**: Fireworks AI (kimi-k2.5, DeepSeek, GLM-5, etc.), DeepSeek direct, Kimi direct, Gemini
- **Prompt engineering pipeline**: Few-shot examples filtered by `(standard, difficulty, type)` + targeted prompt rules by `(standard, difficulty, type)`
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
  -e FIREWORKS_API_KEY=your_fireworks_api_key \
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

# 3. Set API key
export FIREWORKS_API_KEY=your_fireworks_api_key

# 4. Start server
uvicorn api_service.fastapi_app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns loaded grades and default model |
| `/models` | GET | List available models (Fireworks + direct) |
| `/grades/{grade}/combinations` | GET | List all (standard, difficulty, type) combinations for a grade |
| `/generate` | POST | Generate a single question |
| `/generate-all` | POST | Batch-generate all combinations for a grade |
| `/docs` | GET | Interactive Swagger UI documentation |

### Generate a Single Question

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

Response (InceptBench format):

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

### Batch-Generate All Combinations for a Grade

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

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FIREWORKS_API_KEY` | Yes | — | Fireworks AI API key |
| `DEFAULT_MODEL` | No | `fw/kimi-k2.5` | Default generation model |
| `PRELOAD_GRADES` | No | `1,2,...,12` | Grades to preload at startup |
| `DEEPSEEK_API_KEY` | No | — | Required if using DeepSeek models |
| `KIMI_API_KEY` | No | — | Required if using Kimi direct |
| `GEMINI_API_KEY` | No | — | Required if using Gemini models |

## Available Models

### Via Fireworks AI (recommended)

| Model ID | Description |
|----------|-------------|
| `fw/kimi-k2.5` | Kimi K2.5 (default) |
| `fw/deepseek-v3.2` | DeepSeek V3.2 |
| `fw/deepseek-r1` | DeepSeek R1 (reasoning) |
| `fw/glm-5` | GLM-5 (Zhipu) |
| `fw/gpt-oss-120b` | GPT-OSS 120B |
| `fw/qwen3-235b` | Qwen3 235B |

### Direct API

| Model ID | Requires |
|----------|----------|
| `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `kimi-latest` | `KIMI_API_KEY` |
| `gemini-3-flash-preview` | `GEMINI_API_KEY` |

## Project Structure (API Service)

```
qwen32b-sft/
├── api_service/
│   ├── fastapi_app.py          # FastAPI application
│   ├── remote_generator.py     # Remote model generation logic
│   └── schemas.py              # Request/response models
├── data_processing/
│   ├── build_prompt.py         # Prompt construction (few-shot + rules)
│   ├── analyze_dimensions.py   # Grade/standard/difficulty combinations
│   └── select_examples.py      # JSON parsing utilities
├── evaluation/
│   └── inceptbench_client.py   # InceptBench format conversion
├── scripts/
│   ├── generate_questions.py   # Core generation logic
│   └── validate_mcq.py         # MCQ validation/repair
├── data/
│   └── curriculum_standards.json   # CCSS curriculum data
├── processed_training_data/        # Few-shot examples + prompt rules
│   ├── {N}_ELA_examples.json       # Per-grade examples (1-12)
│   └── {N}_ELA_prompt_rules_*.json # Per-grade prompt rules
├── requirements-api.txt        # Lightweight API dependencies
└── Dockerfile                  # Container build file
```

## Closed-Loop Refinement (Development)

For iterating on question quality, the project includes a closed-loop pipeline:

```bash
# Run 3 rounds of generate → evaluate → improve-prompt for grades 1-3
MODELS='fw/kimi-k2.5' GRADES='1,2,3' ROUNDS=3 bash scripts/run_matrix.sh
```

Each round: generates all questions → evaluates via InceptBench API → extracts feedback → updates targeted prompt rules → next round uses improved prompts.

## License

MIT
