# -*- coding: utf-8 -*-
"""API 请求/响应模型"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class MCQRequest(BaseModel):
    """MCQ 生成请求"""
    grade: str = Field(default="3", description="年级")
    standard: str = Field(default="CCSS.ELA-LITERACY.L.3.1.E", description="标准ID")
    difficulty: str = Field(default="medium", description="难度")
    subject: str = Field(default="ELA", description="科目")
    include_think_chain: bool = Field(default=False, description="是否包含思考链")


class MCQResponse(BaseModel):
    """MCQ 生成响应"""
    question_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    answer_options: Optional[Dict[str, str]] = None
    answer_explanation: Optional[str] = None
    difficulty: Optional[str] = None
    processing_time: Optional[float] = None
    raw_response: Optional[str] = None


class BatchMCQRequest(BaseModel):
    """批量 MCQ 请求"""
    requests: List[MCQRequest]
    batch_id: Optional[str] = None


class BatchMCQResponse(BaseModel):
    """批量 MCQ 响应"""
    batch_id: str
    results: List[MCQResponse]
    total_questions: int
    generated_at: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    device: str
    model_name: str


class ConfigResponse(BaseModel):
    """配置信息响应"""
    model_path: str
    default_parameters: Dict[str, Any]
    device: str
    model_name: str
