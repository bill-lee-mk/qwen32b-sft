# -*- coding: utf-8 -*-
"""API 请求/响应模型 — InceptBench 兼容格式"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class GenerateRequest(BaseModel):
    """单题生成请求"""
    grade: str = Field(description="年级 (1-12)")
    standard: str = Field(description="课标 ID, e.g. CCSS.ELA-LITERACY.L.3.1.A")
    difficulty: str = Field(default="medium", description="easy / medium / hard")
    type: str = Field(default="mcq", description="mcq / msq / fill-in")
    subject: str = Field(default="ELA", description="学科")
    model: str = Field(default="fw/kimi-k2.5", description="生成模型")


class GenerateAllRequest(BaseModel):
    """全组合批量生成请求"""
    grade: str = Field(description="年级 (1-12)")
    subject: str = Field(default="ELA", description="学科")
    type: str = Field(default="all", description="题型: all / mcq / msq / fill-in")
    model: str = Field(default="fw/kimi-k2.5", description="生成模型")
    workers: int = Field(default=10, ge=1, le=50, description="并发数")


class InceptBenchSkills(BaseModel):
    lesson_title: str
    substandard_id: str
    substandard_description: str


class InceptBenchItemRequest(BaseModel):
    grade: str
    subject: str
    type: str
    difficulty: str
    locale: str = "en-US"
    skills: InceptBenchSkills


class InceptBenchContent(BaseModel):
    question: str
    answer: str
    answer_options: Optional[List[Dict[str, str]]] = None
    answer_explanation: str
    acceptable_answers: Optional[List[str]] = None


class InceptBenchItem(BaseModel):
    id: str
    request: InceptBenchItemRequest
    content: InceptBenchContent
    image_url: List[Any] = []
    metadata: Dict[str, Any] = {}
    verbose: bool = False


class InceptBenchResponse(BaseModel):
    """InceptBench 兼容的响应格式"""
    generated_content: List[InceptBenchItem]


class GenerateAllProgress(BaseModel):
    """全组合生成的进度响应"""
    generated_content: List[InceptBenchItem]
    total: int
    completed: int
    failed: int
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    default_model: str
    loaded_grades: List[str]


class CombinationItem(BaseModel):
    standard: str
    difficulty: str
    type: str


class CombinationsResponse(BaseModel):
    grade: str
    subject: str
    total: int
    combinations: List[CombinationItem]
