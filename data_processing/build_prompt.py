# -*- coding: utf-8 -*-
"""
构建 prompt（build_prompt）

将示例拼成给闭源模型的 prompt。
支持动态规则：全局规则（所有题目）+ 针对性规则（按 standard 或 (standard, difficulty)）。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STANDARD_DESCRIPTIONS_PATH = _PROJECT_ROOT / "data" / "standard_descriptions.json"
_PROMPT_RULES_PATH = _PROJECT_ROOT / "processed_training_data" / "prompt_rules.json"
_standard_descriptions_cache: Optional[Dict[str, str]] = None
_prompt_rules_cache: Optional[Dict] = None


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


def load_prompt_rules() -> Dict:
    """
    加载动态 prompt 规则（来自失败分析/闭环）。
    结构: { "global_rules": [], "by_standard": {"std": []}, "by_standard_difficulty": {"std|diff": []} }
    """
    global _prompt_rules_cache
    if _prompt_rules_cache is not None:
        return _prompt_rules_cache
    if _PROMPT_RULES_PATH.exists():
        try:
            with open(_PROMPT_RULES_PATH, "r", encoding="utf-8") as f:
                _prompt_rules_cache = json.load(f)
        except Exception:
            _prompt_rules_cache = {}
    else:
        _prompt_rules_cache = {}
    return _prompt_rules_cache


def get_global_rules() -> List[str]:
    """返回全局规则列表（适用于所有题目）"""
    rules = load_prompt_rules()
    return list(rules.get("global_rules") or [])


def get_targeted_rules(standard: str, difficulty: str) -> List[str]:
    """返回针对 (standard, difficulty) 的规则：by_standard[standard] + by_standard_difficulty['std|diff']"""
    rules = load_prompt_rules()
    out: List[str] = []
    by_std = rules.get("by_standard") or {}
    by_key = rules.get("by_standard_difficulty") or {}
    out.extend(by_std.get(standard) or [])
    key = f"{standard}|{difficulty}"
    out.extend(by_key.get(key) or [])
    return out

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

--- GLOBAL CONSTRAINTS (apply to every MCQ; violation causes validation failure) ---
• Option uniqueness: The four options A, B, C, and D must all have different text. No two options may be identical. Duplicate option text will cause validation to fail.
• Single-answer wording: For single-answer MCQs, use singular wording only. Use "Which choice..." or "Which option...". Do NOT use "Which choices..." or "Which options...". Plural wording will trigger validation failure.
• No unreferenced images: Do not say "look at the picture" or "use the image" or "based on the image" in the stem unless you actually provide image_url; otherwise validation will fail.
Before outputting your JSON, do a final check: (1) A ≠ B ≠ C ≠ D in text, (2) stem uses singular "Which choice/option", (3) no image reference without image_url.
--- END GLOBAL CONSTRAINTS ---

Rules:
1. The question MUST directly assess the EXACT skill described in the standard—not a related or adjacent skill. Match the standard precisely.
2. Difficulty must match the request (easy/medium/hard).
3. Provide exactly 4 answer options (A, B, C, D) with exactly ONE correct answer. No ambiguity. Every option must be distinct (see Global Constraints above).
4. For single-answer MCQs use singular wording only (see Global Constraints above).
5. Distractors should be plausible but clearly incorrect.
6. answer_explanation MUST accurately describe ONLY the correct option and why it is correct. Use correct grammatical terms (e.g. comparative adverb vs comparative adjective) and check that the word actually has that function in the sentence. Do NOT reference wrong options or incorrect rules.
7. The answer key (answer field) MUST match the content of the correct option in answer_options. Verify consistency.
8. Clarity: NEVER use "Which word..." when the correct answer is a phrase (e.g., "will paint", "has been"). Use "Which choice...", "Which option...", or "Which words..." instead.
9. Stem–option match: If the stem has a blank (e.g. "a feeling of ______"), each option must be a phrase that can be inserted into that blank to form a grammatical sentence. Do NOT use full clauses (e.g. "disappointment was heavy") as options for a single blank; use phrases (e.g. "disappointment").
10. answer_explanation must state the correct option letter (e.g. "Option B is correct because...") and describe why that option is right; the explanation must not contradict the correct option text.
11. For roots/affixes (L.3.4.B, L.3.4.C, RF.3.3): verify Latin vs Greek roots and affix meanings are factually correct. RF.3.3.B is specifically Latin suffixes (e.g. -tion, -able, -ible, -ment from Latin); do NOT use -ful or -ly for RF.3.3.B (they are not Latin suffixes). For RF.3.3.B focus on decoding (identify base + suffix or which word uses the suffix), not standalone \"What does -ment mean?\" semantic questions.
12. RF.3.3.A (prefixes): Avoid pure recall (e.g. \"What does un- mean?\"). Use the prefix in a word or sentence context so the task involves application, not memorization alone.
13. Single correct answer: For items like \"Which word in the sentence is [an abstract noun/a verb/...]?\", ensure the sentence contains exactly ONE word that fits the criterion. Avoid sentences where two or more options could be correct (e.g. two abstract nouns in the same sentence).
14. No unreferenced stimuli: Do not say \"Look at the picture\" or \"Use the image\" unless you actually provide image_url. If no image is used, keep the item self-contained with text only.
15. Tense and time consistency: If the stem or sentence uses time cues (e.g. \"yesterday\", \"present tense\"), ensure the correct answer and all options are consistent with that cue; avoid conflicting directives.
16. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += "\n\n17. You may optionally include <think>...</think> before the JSON, but the output MUST end with a complete JSON object."
    else:
        base += "\n\n17. Output ONLY the JSON object."
    global_rules = get_global_rules()
    if global_rules:
        base += "\n\n--- Dynamic global rules (from failure analysis) ---\n"
        for r in global_rules:
            base += f"• {r}\n"
        base += "--- END dynamic global rules ---"
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


# 每次生成前必读的校验提醒（写入 user prompt，确保模型在输出前看到）
_USER_VERIFICATION_REMINDER = (
    "Before returning the JSON: verify (1) options A, B, C, and D all have different text—no duplicates; "
    "(2) the stem uses singular wording (e.g. \"Which choice...\" or \"Which option...\"), not \"Which choices/options\"; "
    "(3) do not say \"look at the picture\" or \"use the image\" unless you provide image_url."
)


def build_user_prompt(
    grade: str = "3",
    standard: str = "CCSS.ELA-LITERACY.L.3.1.E",
    standard_description: Optional[str] = None,
    difficulty: str = "medium",
    subject: str = "ELA",
    targeted_rules: Optional[List[str]] = None,
) -> str:
    """构建 user prompt。targeted_rules 为针对该 (standard, difficulty) 的额外提醒。"""
    desc = standard_description or ""
    desc_line = f"Description: {desc}\n" if desc else ""
    text = f"""Generate one MCQ for Grade {grade} {subject}.

Standard: {standard}
{desc_line}Difficulty: {difficulty}

Return only the JSON object. The question MUST assess exactly the skill described above—not a related skill.

{_USER_VERIFICATION_REMINDER}"""
    if targeted_rules:
        text += "\n\n--- Reminders for this standard/difficulty ---\n"
        for r in targeted_rules:
            text += f"• {r}\n"
        text += "--- END reminders ---"
    return text


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
    """构建完整 prompt，返回 (system, user)。会注入全局规则与针对该 (standard, difficulty) 的规则。"""
    desc = standard_description
    if desc is None and use_standard_descriptions:
        desc = get_standard_description(standard)
    system = build_system_prompt(include_think_chain=include_think_chain)
    targeted = get_targeted_rules(standard, difficulty)
    user = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=desc,
        difficulty=difficulty,
        subject=subject,
        targeted_rules=targeted if targeted else None,
    )
    if examples:
        examples_text = build_examples_text(examples, include_think_chain=include_think_chain)
        system += f"\n\nHere are some examples:\n\n{examples_text}\n\n---\n\n"
    return system, user
