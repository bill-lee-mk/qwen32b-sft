# -*- coding: utf-8 -*-
"""
构建 prompt（build_prompt）

将示例拼成给闭源模型的 prompt。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STANDARD_DESCRIPTIONS_PATH = _PROJECT_ROOT / "data" / "standard_descriptions.json"
_standard_descriptions_cache: Optional[Dict[str, str]] = None


def load_standard_descriptions() -> Dict[str, str]:
    """加载 CCSS 标准描述（用于提升生成质量和标准对齐）"""
    global _standard_descriptions_cache
    if _standard_descriptions_cache is not None:
        return _standard_descriptions_cache
    if _STANDARD_DESCRIPTIONS_PATH.exists():
        with open(_STANDARD_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
            _standard_descriptions_cache = json.load(f)
    else:
        _standard_descriptions_cache = {}
    return _standard_descriptions_cache


def get_standard_description(standard: str) -> Optional[str]:
    """获取单个标准的描述"""
    return load_standard_descriptions().get(standard)

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
1. The question MUST directly assess the EXACT skill described in the standard—not a related or adjacent skill. Match the standard precisely.
2. Difficulty must match the request (easy/medium/hard).
3. Provide exactly 4 answer options (A, B, C, D) with exactly ONE correct answer. No ambiguity.
4. Distractors should be plausible but clearly incorrect.
5. answer_explanation MUST accurately describe ONLY the correct option and why it is correct. Do NOT reference wrong options or incorrect rules.
6. The answer key (answer field) MUST match the content of the correct option in answer_options. Verify consistency.
7. Clarity: NEVER use "Which word..." when the correct answer is a phrase (e.g., "will paint", "has been"). Use "Which choice...", "Which option...", or "Which words..." instead.
8. Stem–option match: If the stem has a blank (e.g. "a feeling of ______"), each option must be a phrase that can be inserted into that blank to form a grammatical sentence. Do NOT use full clauses (e.g. "disappointment was heavy") as options for a single blank; use phrases (e.g. "disappointment").
9. answer_explanation must state the correct option letter (e.g. "Option B is correct because...") and describe why that option is right; the explanation must not contradict the correct option text.
10. For roots/affixes (L.3.4.B, L.3.4.C, RF.3.3): verify Latin vs Greek roots and affix meanings are factually correct. RF.3.3.B is specifically Latin suffixes (e.g. -tion, -able, -ible, -ment from Latin); do NOT use -ful or -ly for RF.3.3.B (they are not Latin suffixes).
11. Single correct answer: For items like \"Which word in the sentence is [an abstract noun/a verb/...]?\", ensure the sentence contains exactly ONE word that fits the criterion. Avoid sentences where two or more options could be correct (e.g. two abstract nouns in the same sentence).
12. No unreferenced stimuli: Do not say \"Look at the picture\" or \"Use the image\" unless you actually provide image_url. If no image is used, keep the item self-contained with text only.
13. Tense and time consistency: If the stem or sentence uses time cues (e.g. \"yesterday\", \"present tense\"), ensure the correct answer and all options are consistent with that cue; avoid conflicting directives.
14. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += "\n\n15. You may optionally include <think>...</think> before the JSON, but the output MUST end with a complete JSON object."
    else:
        base += "\n\n15. Output ONLY the JSON object."
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
    desc_line = f"Description: {desc}\n" if desc else ""
    return f"""Generate one MCQ for Grade {grade} {subject}.

Standard: {standard}
{desc_line}Difficulty: {difficulty}

Return only the JSON object. The question MUST assess exactly the skill described above—not a related skill."""


def build_full_prompt(
    grade: str,
    standard: str,
    difficulty: str,
    examples: List[Dict],
    subject: str = "ELA",
    standard_description: Optional[str] = None,
    include_think_chain: bool = False,
    use_standard_descriptions: bool = True,
) -> tuple:
    """构建完整 prompt，返回 (system, user)"""
    desc = standard_description
    if desc is None and use_standard_descriptions:
        desc = get_standard_description(standard)
    system = build_system_prompt(include_think_chain=include_think_chain)
    user = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=desc,
        difficulty=difficulty,
        subject=subject,
    )
    if examples:
        examples_text = build_examples_text(examples, include_think_chain=include_think_chain)
        system += f"\n\nHere are some examples:\n\n{examples_text}\n\n---\n\n"
    return system, user
