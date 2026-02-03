# -*- coding: utf-8 -*-

"""
FastAPI API服务
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import uvicorn
from datetime import datetime
import logging

from .model_loader import MCQGenerator
from .schemas import (
    MCQRequest, MCQResponse, BatchMCQRequest, BatchMCQResponse,
    HealthResponse, ConfigResponse
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型实例
generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期（替代已弃用的 on_event）"""
    global generator
    model_path = os.environ.get("MODEL_PATH", "models/final_model")
    try:
        generator = MCQGenerator(model_path=model_path)
        generator.load_model()
        logger.info("模型加载完成，API服务已就绪")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise
    yield
    if generator:
        del generator
        generator = None
        logger.info("模型资源已清理")


# 创建FastAPI应用
app = FastAPI(
    title="K-12 ELA MCQ Generator API",
    description="基于全参数微调Qwen2.5-32B的K-12 ELA选择题生成API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    """根端点"""
    return {
        "service": "K-12 ELA MCQ Generator API",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "/generate": "POST - 生成单个MCQ",
            "/batch-generate": "POST - 批量生成MCQ",
            "/health": "GET - 健康检查",
            "/config": "GET - 配置信息"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    if generator and generator.model_loaded:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            device=generator.device,
            model_name=generator.model_name
        )
    return HealthResponse(
        status="unhealthy",
        model_loaded=False,
        device="unknown",
        model_name="unknown"
    )

@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """获取配置信息"""
    if generator:
        return ConfigResponse(
            model_path=generator.model_path,
            default_parameters=generator.default_params,
            device=str(generator.device),
            model_name=generator.model_name
        )
    return ConfigResponse(
        model_path="unknown",
        default_parameters={},
        device="unknown",
        model_name="unknown"
    )

@app.post("/generate", response_model=MCQResponse)
async def generate_mcq(request: MCQRequest):
    """生成单个MCQ"""
    if not generator or not generator.model_loaded:
        raise HTTPException(status_code=503, detail="服务未就绪，模型未加载")
    
    try:
        start_time = datetime.now()
        
        # 生成MCQ
        result = await generator.generate_async(request)
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        result.processing_time = processing_time
        
        return result
        
    except Exception as e:
        logger.error(f"生成MCQ失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/batch-generate", response_model=BatchMCQResponse)
async def batch_generate_mcq(request: BatchMCQRequest):
    """批量生成MCQ"""
    if not generator or not generator.model_loaded:
        raise HTTPException(status_code=503, detail="服务未就绪，模型未加载")
    
    # 限制批量大小
    if len(request.requests) > 20:
        raise HTTPException(status_code=400, detail="批量大小不能超过20")
    
    try:
        start_time = datetime.now()
        
        # 批量生成
        results = []
        for req in request.requests:
            try:
                result = await generator.generate_async(req)
                results.append(result)
            except Exception as e:
                logger.error(f"批量生成中单个失败: {e}")
                # 可以添加错误处理逻辑
        
        # 创建响应
        response = BatchMCQResponse(
            batch_id=request.batch_id or f"batch_{int(datetime.now().timestamp())}",
            results=results,
            total_questions=len(results),
            generated_at=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"批量生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量生成失败: {str(e)}")

@app.post("/template/generate")
async def generate_from_template(
    grade: str = "3",
    standard: str = "CCSS.ELA-LITERACY.L.3.1.E",
    difficulty: str = "medium",
    subject: str = "ELA",
    include_think_chain: bool = False
):
    """使用模板生成MCQ"""
    request = MCQRequest(
        grade=grade,
        standard=standard,
        difficulty=difficulty,
        subject=subject,
        include_think_chain=include_think_chain
    )
    
    return await generate_mcq(request)

def run_api_server(host: str = "0.0.0.0", port: int = 8000, model_path: str = None):
    """运行API服务器。model_path 会覆盖环境变量 MODEL_PATH"""
    if model_path is not None:
        os.environ["MODEL_PATH"] = model_path
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    run_api_server()
