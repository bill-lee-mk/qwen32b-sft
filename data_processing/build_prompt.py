# -*- coding: utf-8 -*-
"""
构建 prompt（build_prompt）

将示例拼成给闭源模型的 prompt。
支持动态规则：全局规则（所有题目）+ 针对性规则（按 standard 或 (standard, difficulty)）。
支持多年级/学科：从 curriculum_standards.json 加载标准描述与课程元数据。
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STANDARD_DESCRIPTIONS_PATH = _PROJECT_ROOT / "data" / "standard_descriptions.json"
_CURRICULUM_STANDARDS_PATH = _PROJECT_ROOT / "data" / "curriculum_standards.json"
_PROMPT_RULES_PATH = _PROJECT_ROOT / "processed_training_data" / "prompt_rules.json"
_standard_descriptions_cache: Optional[Dict[str, str]] = None
_curriculum_standards_cache: Optional[Dict[str, Dict]] = None
# 按路径缓存，支持 PROMPT_RULES_PATH 环境变量（多模型闭环时每模型独立 prompt_rules）
_prompt_rules_cache: Dict[str, Dict] = {}


def load_curriculum_standards() -> Dict[str, Dict]:
    """加载 curriculum_standards.json（含完整元数据）。"""
    global _curriculum_standards_cache
    if _curriculum_standards_cache is not None:
        return _curriculum_standards_cache
    if _CURRICULUM_STANDARDS_PATH.exists():
        with open(_CURRICULUM_STANDARDS_PATH, "r", encoding="utf-8") as f:
            _curriculum_standards_cache = json.load(f)
    else:
        _curriculum_standards_cache = {}
    return _curriculum_standards_cache


def load_standard_descriptions() -> Dict[str, str]:
    """加载标准描述。优先从 curriculum_standards.json 提取，fallback 到 standard_descriptions.json。"""
    global _standard_descriptions_cache
    if _standard_descriptions_cache is not None:
        return _standard_descriptions_cache
    curriculum = load_curriculum_standards()
    if curriculum:
        _standard_descriptions_cache = {
            sid: info["standard_description"]
            for sid, info in curriculum.items()
            if info.get("standard_description")
        }
    if not _standard_descriptions_cache:
        if _STANDARD_DESCRIPTIONS_PATH.exists():
            with open(_STANDARD_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
                _standard_descriptions_cache = json.load(f)
        else:
            _standard_descriptions_cache = {}
    return _standard_descriptions_cache


def get_standard_description(standard: str) -> Optional[str]:
    """获取单个标准的描述"""
    return load_standard_descriptions().get(standard)


def get_curriculum_metadata(standard: str) -> Optional[Dict]:
    """获取标准的完整课程元数据（learning_objectives, assessment_boundaries, common_misconceptions）。"""
    return load_curriculum_standards().get(standard)


def load_prompt_rules() -> Dict:
    """
    加载动态 prompt 规则（来自失败分析/闭环）。
    结构: { "global_rules": [], "by_standard": {"std": []}, "by_standard_difficulty": {"std|diff": []} }
    若设置环境变量 PROMPT_RULES_PATH，则从该路径读取（用于多模型闭环时每模型独立规则，避免互相覆盖）。
    """
    global _prompt_rules_cache
    path = Path(os.environ.get("PROMPT_RULES_PATH", str(_PROMPT_RULES_PATH))).resolve()
    key = str(path)
    if key in _prompt_rules_cache:
        return _prompt_rules_cache[key]
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    _prompt_rules_cache[key] = data
    return data


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

QUESTION_TYPES = ("mcq", "msq", "fill-in")

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

MSQ_SCHEMA = """
{
  "id": "唯一ID",
  "type": "msq",
  "question": "题目文本（必须明确指示 'Select ALL that apply' 或 'Choose all correct answers'）",
  "answer": "A,C（逗号分隔的所有正确选项字母，按字母顺序排列）",
  "answer_options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "answer_explanation": "解释为什么每个正确选项是对的，以及每个错误选项为什么不对",
  "difficulty": "easy/medium/hard"
}
"""

FILLIN_SCHEMA = """
{
  "id": "唯一ID",
  "type": "fill-in",
  "question": "题目文本（包含一个需要填入答案的空白，用 ______ 标记）",
  "answer": "正确答案文本",
  "acceptable_answers": ["正确答案的其他可接受写法（可选）"],
  "answer_explanation": "正确答案解析",
  "difficulty": "easy/medium/hard"
}
"""

_SCHEMA_BY_TYPE = {
    "mcq": MCQ_SCHEMA,
    "msq": MSQ_SCHEMA,
    "fill-in": FILLIN_SCHEMA,
}


def _build_system_prompt_mcq(grade: str, subject: str, include_think_chain: bool) -> str:
    base = f"""You are an expert K-12 {subject} MCQ designer for Alpha School, Grade {grade}.

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
    return base


def _build_system_prompt_msq(grade: str, subject: str, include_think_chain: bool) -> str:
    base = f"""You are an expert K-12 {subject} question designer for Alpha School, Grade {grade}.

Your task: Generate one multiple-select question (MSQ) that assesses the given standard at the specified difficulty. MSQ questions have MULTIPLE correct answers (2 or 3 out of 4 options).

--- GLOBAL CONSTRAINTS (apply to every MSQ; violation causes validation failure) ---
• Option uniqueness: All options must have different text. Duplicate option text will cause validation to fail.
• Multi-answer wording: The stem MUST clearly indicate multiple answers are expected. Use phrases like "Select ALL that apply", "Choose all correct answers", or "Which of the following are correct? (Select all that apply)".
• No unreferenced images: Do not reference images unless you provide image_url.
Before outputting your JSON, do a final check: (1) all options have different text, (2) stem clearly indicates multiple selection, (3) answer field lists ALL correct options separated by commas.
--- END GLOBAL CONSTRAINTS ---

Rules:
1. The question MUST directly assess the EXACT skill described in the standard.
2. Difficulty must match the request (easy/medium/hard).
3. Provide exactly 4 answer options (A, B, C, D). At least 2 and at most 3 must be correct.
4. The "answer" field must list ALL correct option letters separated by commas in alphabetical order (e.g. "A,C" or "A,B,D").
5. Distractors should be plausible but clearly incorrect when analyzed carefully.
6. answer_explanation must explain why EACH correct option is right AND why each incorrect option is wrong.
7. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += "\n\n8. You may optionally include <think>...</think> before the JSON, but the output MUST end with a complete JSON object."
    else:
        base += "\n\n8. Output ONLY the JSON object."
    return base


def _build_system_prompt_fillin(grade: str, subject: str, include_think_chain: bool) -> str:
    base = f"""You are an expert K-12 {subject} question designer for Alpha School, Grade {grade}.

Your task: Generate one fill-in-the-blank question that assesses the given standard at the specified difficulty. Students must type the correct answer (no options provided).

--- GLOBAL CONSTRAINTS (apply to every fill-in question; violation causes validation failure) ---
• The question MUST contain a clear blank (use "______" to mark it) where the student types their answer.
• The "answer" field must contain the single best correct answer text.
• Optionally provide "acceptable_answers" array with alternative correct spellings, phrasings, or forms.
• No unreferenced images: Do not reference images unless you provide image_url.
Before outputting your JSON, do a final check: (1) question contains a blank, (2) answer is a reasonable text that fills the blank, (3) acceptable_answers covers common variations.
--- END GLOBAL CONSTRAINTS ---

Rules:
1. The question MUST directly assess the EXACT skill described in the standard.
2. Difficulty must match the request (easy/medium/hard).
3. The blank should test a specific skill—avoid overly open-ended blanks with too many valid answers.
4. The "answer" field contains the primary correct answer. Keep it concise (1-3 words typically).
5. "acceptable_answers" should list reasonable alternative correct responses (e.g. different tenses, abbreviations, synonyms that are equally correct).
6. answer_explanation must explain why the answer is correct and what skill it tests.
7. Do NOT include answer_options (this is not a multiple-choice question).
8. Return ONLY a valid JSON object. No markdown, no extra text."""
    if include_think_chain:
        base += "\n\n9. You may optionally include <think>...</think> before the JSON, but the output MUST end with a complete JSON object."
    else:
        base += "\n\n9. Output ONLY the JSON object."
    return base


def build_system_prompt(grade: str = "3", subject: str = "ELA",
                        include_think_chain: bool = False,
                        question_type: str = "mcq") -> str:
    """构建 system prompt（支持任意 grade/subject/question_type）"""
    qtype = question_type.lower().strip() if question_type else "mcq"
    if qtype == "msq":
        base = _build_system_prompt_msq(grade, subject, include_think_chain)
    elif qtype == "fill-in":
        base = _build_system_prompt_fillin(grade, subject, include_think_chain)
    else:
        base = _build_system_prompt_mcq(grade, subject, include_think_chain)

    global_rules = get_global_rules()
    if global_rules:
        base += "\n\n--- Dynamic global rules (from failure analysis) ---\n"
        for r in global_rules:
            base += f"• {r}\n"
        base += "--- END dynamic global rules ---"
    schema = _SCHEMA_BY_TYPE.get(qtype, MCQ_SCHEMA)
    base += f"\n\nOutput schema:\n{schema.strip()}"
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


_USER_VERIFICATION_REMINDER = {
    "mcq": (
        "Before returning the JSON: verify (1) options A, B, C, and D all have different text—no duplicates; "
        "(2) the stem uses singular wording (e.g. \"Which choice...\" or \"Which option...\"), not \"Which choices/options\"; "
        "(3) do not say \"look at the picture\" or \"use the image\" unless you provide image_url."
    ),
    "msq": (
        "Before returning the JSON: verify (1) all options have different text—no duplicates; "
        "(2) the stem clearly says \"Select ALL that apply\" or similar multi-select wording; "
        "(3) the answer field lists ALL correct letters separated by commas in alphabetical order; "
        "(4) at least 2 and at most 3 options are correct."
    ),
    "fill-in": (
        "Before returning the JSON: verify (1) the question contains a clear blank (______); "
        "(2) the answer is concise and directly fills the blank; "
        "(3) acceptable_answers covers reasonable alternative correct responses; "
        "(4) do NOT include answer_options (this is not multiple-choice)."
    ),
}


def _build_curriculum_guidance(standard: str) -> str:
    """当课程元数据可用时，构建 CURRICULUM GUIDANCE 段落。"""
    meta = get_curriculum_metadata(standard)
    if not meta:
        return ""
    lo = meta.get("learning_objectives") or []
    ab = meta.get("assessment_boundaries") or []
    cm = meta.get("common_misconceptions") or []
    if not lo and not ab and not cm:
        return ""
    parts = ["\n--- CURRICULUM GUIDANCE (use to improve question quality) ---"]
    if lo:
        parts.append("Learning Objectives:")
        for item in lo:
            parts.append(f"  • {item}")
    if ab:
        parts.append("Assessment Boundaries:")
        for item in ab:
            parts.append(f"  • {item}")
    if cm:
        parts.append("Common Misconceptions (use as distractor inspiration):")
        for item in cm:
            parts.append(f"  • {item}")
    parts.append("--- END CURRICULUM GUIDANCE ---")
    return "\n".join(parts)


_TYPE_LABELS = {
    "mcq": "MCQ (multiple-choice, single correct answer)",
    "msq": "MSQ (multiple-select, 2-3 correct answers)",
    "fill-in": "fill-in-the-blank (student types the answer)",
}


def build_user_prompt(
    grade: str = "3",
    standard: str = "CCSS.ELA-LITERACY.L.3.1.E",
    standard_description: Optional[str] = None,
    difficulty: str = "medium",
    subject: str = "ELA",
    targeted_rules: Optional[List[str]] = None,
    question_type: str = "mcq",
) -> str:
    """构建 user prompt。targeted_rules 为针对该 (standard, difficulty) 的额外提醒。"""
    qtype = question_type.lower().strip() if question_type else "mcq"
    type_label = _TYPE_LABELS.get(qtype, _TYPE_LABELS["mcq"])
    desc = standard_description or ""
    desc_line = f"Description: {desc}\n" if desc else ""
    reminder = _USER_VERIFICATION_REMINDER.get(qtype, _USER_VERIFICATION_REMINDER["mcq"])
    text = f"""Generate one {type_label} question for Grade {grade} {subject}.

Standard: {standard}
{desc_line}Difficulty: {difficulty}

Return only the JSON object. The question MUST assess exactly the skill described above—not a related skill.

{reminder}"""
    curriculum_guidance = _build_curriculum_guidance(standard)
    if curriculum_guidance:
        text += "\n" + curriculum_guidance
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
    question_type: str = "mcq",
) -> tuple:
    """构建完整 prompt，返回 (system, user)。会注入全局规则与针对该 (standard, difficulty) 的规则。"""
    qtype = question_type.lower().strip() if question_type else "mcq"
    desc = standard_description
    if desc is None and use_standard_descriptions:
        desc = get_standard_description(standard)
    system = build_system_prompt(grade=grade, subject=subject,
                                include_think_chain=include_think_chain,
                                question_type=qtype)
    targeted = get_targeted_rules(standard, difficulty)
    user = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=desc,
        difficulty=difficulty,
        subject=subject,
        targeted_rules=targeted if targeted else None,
        question_type=qtype,
    )
    if examples:
        same_type = [e for e in examples if (e.get("mcq_json", {}).get("type") or "mcq") == qtype]
        ex_to_use = same_type if same_type else examples
        examples_text = build_examples_text(ex_to_use, include_think_chain=include_think_chain)
        system += f"\n\nHere are some examples:\n\n{examples_text}\n\n---\n\n"
    return system, user
