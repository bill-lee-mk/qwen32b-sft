# -*- coding: utf-8 -*-
"""
InceptBench 评估客户端（占位实现）
用于通过 InceptBench API 评估 MCQ 生成质量。
完整实现需根据 InceptBench 官方 API 文档补充。
"""
import os
from typing import Dict, Any, Optional

# InceptBench 期望的 MCQ 必填字段
INCEPTBENCH_MCQ_FIELDS = [
    "id", "type", "question", "answer", "answer_options",
    "answer_explanation", "difficulty"
]


def normalize_for_inceptbench(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 MCQ 数据归一化为 InceptBench API 期望的格式。
    确保必填字段存在，answer_options 为 {"A":..., "B":..., "C":..., "D":...}，
    answer 为 A/B/C/D。
    """
    out = {}
    for k in INCEPTBENCH_MCQ_FIELDS:
        v = question_data.get(k)
        if v is None:
            if k == "type":
                out[k] = "mcq"
            elif k == "difficulty":
                out[k] = "medium"
            else:
                out[k] = ""
        else:
            out[k] = v
    opts = out.get("answer_options")
    if isinstance(opts, dict):
        normalized_opts = {}
        for letter in "ABCD":
            normalized_opts[letter] = opts.get(letter, opts.get(letter.lower(), ""))
        out["answer_options"] = normalized_opts
    ans = str(out.get("answer", "")).upper().strip()
    out["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
    return out


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
        # 归一化格式后再评估
        normalized = normalize_for_inceptbench(question_data)
        # 占位：返回基于本地规则的简单评分
        return {
            "overall_score": 0.0,
            "status": "stub",
            "message": (
                "InceptBench 评估为占位实现。"
                "请根据 InceptBench API 文档实现完整 evaluate_mcq 逻辑，"
                "或配置 INCEPTBENCH_API_KEY 后接入真实 API。"
            ),
            "input_keys": list(normalized.keys()) if normalized else [],
        }
