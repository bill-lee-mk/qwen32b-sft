# -*- coding: utf-8 -*-
"""
Few-shot Prompt 构建器

设计原则：
1. System：明确任务、输出格式、K-12 ELA 约束
2. Few-shot：2-5 个 (user, assistant) 对，展示输入→输出
3. 输出格式：严格 JSON，便于解析和 InceptBench 打分
"""
import json
from typing import Any, Dict, List, Optional

# InceptBench 要求的 MCQ JSON 结构
MCQ_SCHEMA = """
{
  "id": "唯一ID，如 standard_difficulty_001",
  "type": "mcq",
  "question": "题目文本",
  "answer": "A/B/C/D",
  "answer_options": {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"},
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
6. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += """

7. You may optionally include a <think>...</think> block before the JSON to show your reasoning, but the final output MUST end with a complete JSON object."""
    else:
        base += """

7. Output ONLY the JSON object. No <think>, no explanation outside the JSON."""
    base += f"""

Output schema (strict):
{MCQ_SCHEMA.strip()}
"""
    return base


def build_few_shot_examples(few_shot_samples: List[Dict], include_think_chain: bool = False) -> str:
    """
    将 few-shot 样本格式化为 prompt 中的示例。
    每个样本: {user_prompt, mcq_json}
    """
    parts = []
    for i, s in enumerate(few_shot_samples, 1):
        user = s.get("user_prompt", "")
        mcq = s.get("mcq_json", {})
        if not user or not mcq:
            continue
        assistant_json = json.dumps(mcq, ensure_ascii=False, indent=2)
        if include_think_chain:
            assistant = f"<think>\n[Reasoning about the standard and question design...]\n</think>\n{assistant_json}"
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
    """构建单次生成的 user prompt"""
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
    few_shot_samples: List[Dict],
    subject: str = "ELA",
    standard_description: Optional[str] = None,
    include_think_chain: bool = False,
) -> str:
    """
    构建完整 prompt（用于 API 模型，如 Gemini/GPT-4）。
    返回格式取决于后端：部分 API 需要 [{"role":"system","content":...}, ...]
    """
    system = build_system_prompt(include_think_chain=include_think_chain)
    user_for_gen = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=standard_description,
        difficulty=difficulty,
        subject=subject,
    )
    if few_shot_samples:
        examples = build_few_shot_examples(few_shot_samples, include_think_chain=include_think_chain)
        system += f"\n\nHere are some examples:\n\n{examples}\n\n---\n\n"
    return system, user_for_gen


def format_for_openai_api(system: str, user: str) -> List[Dict[str, str]]:
    """格式化为 OpenAI API 的 messages"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def format_for_gemini_api(system: str, user: str) -> str:
    """格式化为 Gemini 单轮 prompt（system + user 合并）"""
    return f"{system}\n\n{user}"
