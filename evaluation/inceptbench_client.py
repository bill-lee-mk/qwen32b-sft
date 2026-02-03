# -*- coding: utf-8 -*-
"""
InceptBench 评估客户端（占位实现）
用于通过 InceptBench API 评估 MCQ 生成质量。
完整实现需根据 InceptBench 官方 API 文档补充。
"""
import os
from typing import Dict, Any, Optional


class InceptBenchEvaluator:
    """InceptBench MCQ 评估器"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("INCEPTBENCH_API_KEY")
        if not self.api_key:
            import warnings
            warnings.warn(
                "未设置 InceptBench API Key。"
                "请通过 --api-key 或环境变量 INCEPTBENCH_API_KEY 提供。"
            )

    def evaluate_mcq(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个 MCQ 的质量。

        Args:
            question_data: 包含 question, answer, answer_options 等的字典

        Returns:
            评估结果，包含 overall_score 等字段

        注意：此为占位实现。完整实现需调用 InceptBench API，例如：
            - 发送 question_data 到评估端点
            - 解析返回的评分和反馈
        """
        # 占位：返回基于本地规则的简单评分
        return {
            "overall_score": 0.0,
            "status": "stub",
            "message": (
                "InceptBench 评估为占位实现。"
                "请根据 InceptBench API 文档实现完整 evaluate_mcq 逻辑，"
                "或配置 INCEPTBENCH_API_KEY 后接入真实 API。"
            ),
            "input_keys": list(question_data.keys()) if question_data else [],
        }
