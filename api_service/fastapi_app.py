# -*- coding: utf-8 -*-
"""FastAPI 服务 — 供 InceptBench 调用的题目生成 API"""
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .remote_generator import RemoteGenerator
from .schemas import (
    CombinationItem,
    CombinationsResponse,
    GenerateAllProgress,
    GenerateAllRequest,
    GenerateRequest,
    HealthResponse,
    InceptBenchResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

generator: RemoteGenerator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    default_model = os.environ.get("DEFAULT_MODEL", "fw/kimi-k2.5")
    generator = RemoteGenerator(default_model=default_model)
    preload = os.environ.get("PRELOAD_GRADES", "1,2,3,4,5,6,7,8,9,10,11,12")
    for g in preload.split(","):
        g = g.strip()
        if g:
            generator.load_grade(g, "ELA")
    logger.info(f"Service ready — default model: {default_model}, loaded grades: {generator.loaded_grades}")
    yield
    generator = None


app = FastAPI(
    title="K-12 ELA Question Generator API",
    description="生成 K-12 ELA 题目（MCQ/MSQ/Fill-in），返回 InceptBench 兼容格式",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _to_inceptbench_response(questions: list) -> dict:
    """将内部格式的题目列表转为 InceptBench 响应。"""
    from evaluation.inceptbench_client import to_inceptbench_payload
    payload = to_inceptbench_payload(questions)
    return payload


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "K-12 ELA Question Generator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /generate": "生成单道题目",
            "POST /generate-all": "批量生成某年级全部题目",
            "GET /health": "健康检查",
            "GET /models": "可用模型列表",
            "GET /grades/{grade}/combinations": "查看某年级的全部组合",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if generator:
        return HealthResponse(
            status="healthy",
            default_model=generator.default_model,
            loaded_grades=generator.loaded_grades,
        )
    return HealthResponse(status="unhealthy", default_model="N/A", loaded_grades=[])


@app.get("/models")
async def list_models():
    from scripts.generate_questions import FIREWORKS_MODEL_MAP
    return {
        "default": generator.default_model if generator else "fw/kimi-k2.5",
        "fireworks": {f"fw/{k}": v for k, v in FIREWORKS_MODEL_MAP.items()},
        "direct": ["deepseek-chat", "deepseek-reasoner", "kimi-latest", "gemini-3-flash-preview"],
    }


@app.get("/grades/{grade}/combinations", response_model=CombinationsResponse)
async def get_combinations(grade: str, subject: str = "ELA", type: str = "all"):
    if not generator:
        raise HTTPException(503, "Service not ready")
    combos = generator.get_combinations(grade, subject, type)
    return CombinationsResponse(
        grade=grade,
        subject=subject,
        total=len(combos),
        combinations=[CombinationItem(standard=s, difficulty=d, type=t) for s, d, t in combos],
    )


@app.post("/generate", response_model=InceptBenchResponse)
async def generate_question(req: GenerateRequest):
    """生成单道题目，返回 InceptBench 兼容格式。"""
    if not generator:
        raise HTTPException(503, "Service not ready")
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: generator.generate_one(
            grade=req.grade, standard=req.standard,
            difficulty=req.difficulty, question_type=req.type,
            subject=req.subject, model=req.model,
        ),
    )
    if result.get("error"):
        raise HTTPException(500, result["error"])
    payload = _to_inceptbench_response([result])
    return payload


@app.post("/generate-all", response_model=GenerateAllProgress)
async def generate_all(req: GenerateAllRequest):
    """批量生成某年级的全部 (standard x difficulty x type) 组合。"""
    if not generator:
        raise HTTPException(503, "Service not ready")
    import asyncio
    loop = asyncio.get_event_loop()
    t0 = time.time()
    results, failed = await loop.run_in_executor(
        None,
        lambda: generator.generate_batch(
            grade=req.grade, subject=req.subject,
            question_type=req.type, model=req.model,
            workers=req.workers,
        ),
    )
    payload = _to_inceptbench_response(results)
    total = len(results) + failed
    return GenerateAllProgress(
        generated_content=payload["generated_content"],
        total=total,
        completed=len(results),
        failed=failed,
        elapsed_seconds=round(time.time() - t0, 1),
    )


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
