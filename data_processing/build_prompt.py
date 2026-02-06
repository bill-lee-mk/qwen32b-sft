# -*- coding: utf-8 -*-
"""
构建 prompt（build_prompt）

将示例拼成给闭源模型的 prompt。
"""
import json
from typing import Dict, List, Optional

MCQ_SCHEMA = """
{
  "id": "唯一ID",
  "type": "mcq",
  "question": "题目文本",
  "answer": "A/B/C/D",
  "answer_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_explanation": "正确答案解析",
  "difficulty": "easy/medium/hard"
}
"""


def build_system_prompt(include_think_chain: bool = False) -> str:
    """构建 system prompt"""
    base = """You are an expert K-12 ELA (English Language Arts) MCQ designer for Alpha School, Grade 3.

Your task: Generate one multiple-choice question (MCQ) that assesses the given standard at the specified difficulty.

Rules:
1. The question must directly assess the provided CCSS ELA standard.
2. Difficulty must match the request (easy/medium/hard).
3. Provide exactly 4 answer options (A, B, C, D) with exactly one correct answer.
4. Distractors should be plausible but clearly incorrect.
5. Include a brief answer_explanation for why the correct answer is right.
6. Clarity: If the correct answer is a phrase (e.g., "will paint"), avoid "Which word..." in the stem; use "Which choice...", "Which words...", or "Which option..." instead.
7. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += "\n\n8. You may optionally include <think>...</think> before the JSON, but the output MUST end with a complete JSON object."
    else:
        base += "\n\n8. Output ONLY the JSON object."
    base += f"\n\nOutput schema:\n{MCQ_SCHEMA.strip()}"
    return base


def build_examples_text(examples: List[Dict], include_think_chain: bool = False) -> str:
    """将示例格式化为 prompt 中的文本"""
    parts = []
    for i, s in enumerate(examples, 1):
        user = s.get("user_prompt", "")
        mcq = s.get("mcq_json", {})
        if not user or not mcq:
            continue
        assistant_json = json.dumps(mcq, ensure_ascii=False, indent=2)
        if include_think_chain:
            assistant = f"<think>\n[Reasoning...]\n</think>\n{assistant_json}"
        else:
            assistant = assistant_json
        parts.append(f"Example {i}:\nUser: {user}\n\nAssistant:\n{assistant}")
    return "\n\n---\n\n".join(parts) if parts else ""


def build_user_prompt(
    grade: str = "3",
    standard: str = "CCSS.ELA-LITERACY.L.3.1.E",
    standard_description: Optional[str] = None,
    difficulty: str = "medium",
    subject: str = "ELA",
) -> str:
    """构建 user prompt"""
    desc = standard_description or ""
    return f"""Generate one MCQ for Grade {grade} {subject}.

Standard: {standard}
{f'Description: {desc}' if desc else ''}
Difficulty: {difficulty}

Return only the JSON object."""


def build_full_prompt(
    grade: str,
    standard: str,
    difficulty: str,
    examples: List[Dict],
    subject: str = "ELA",
    standard_description: Optional[str] = None,
    include_think_chain: bool = False,
) -> tuple:
    """构建完整 prompt，返回 (system, user)"""
    system = build_system_prompt(include_think_chain=include_think_chain)
    user = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=standard_description,
        difficulty=difficulty,
        subject=subject,
    )
    if examples:
        examples_text = build_examples_text(examples, include_think_chain=include_think_chain)
        system += f"\n\nHere are some examples:\n\n{examples_text}\n\n---\n\n"
    return system, user
