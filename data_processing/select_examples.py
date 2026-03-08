# -*- coding: utf-8 -*-
"""
筛选示例（select_examples）

从 raw_data 中筛选少量高质量 MCQ 样本，用于闭源模型的 prompt 示范。
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MCQ_REQUIRED_FIELDS = ["id", "type", "question", "answer", "answer_explanation", "difficulty"]
MCQ_OPTION_FIELDS = ["answer_options"]
MSQ_REQUIRED_FIELDS = ["id", "type", "question", "answer", "answer_options", "answer_explanation", "difficulty"]
FILLIN_REQUIRED_FIELDS = ["id", "type", "question", "answer", "answer_explanation", "difficulty"]


def _repair_json_control_chars(s: str) -> str:
    """转义 JSON 字符串值内的非法控制字符（如 LLM 输出的真实换行/制表符）。"""
    out = []
    in_str = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_str and i + 1 < len(s):
            out.append(c)
            out.append(s[i + 1])
            i += 2
            continue
        if c == '"':
            in_str = not in_str
        if in_str:
            if c == '\n':
                out.append('\\n')
            elif c == '\r':
                out.append('\\r')
            elif c == '\t':
                out.append('\\t')
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def _fix_unescaped_inner_quotes(text: str) -> str:
    """修复 JSON 字符串值内部未转义的双引号。"""
    current = text
    for _ in range(20):
        try:
            json.loads(current)
            return current
        except json.JSONDecodeError as e:
            if "Expecting ',' delimiter" not in str(e) and "Expecting ':' delimiter" not in str(e):
                return current
            pos = e.pos
            quote_pos = current.rfind('"', 0, pos)
            if quote_pos < 0 or quote_pos == 0:
                return current
            if quote_pos > 0 and current[quote_pos - 1] == '\\':
                return current
            current = current[:quote_pos] + '\\' + current[quote_pos:]
    return current


def extract_json_from_text(text: str) -> Optional[Dict]:
    """从文本中提取第一个可解析的完整 JSON 对象。

    跳过思考链等非 JSON 文本中的花括号，持续扫描直到找到有效 JSON。
    自动修复字符串内的非法控制字符（换行等）和未转义的内嵌双引号。
    """
    pos = 0
    while pos < len(text):
        start = text.find("{", pos)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start : i + 1]
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        pass
                    repaired = _repair_json_control_chars(raw)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
                    try:
                        return json.loads(_fix_unescaped_inner_quotes(repaired))
                    except json.JSONDecodeError:
                        pass
                    pos = i + 1
                    break
        else:
            return None
    return None


def _is_placeholder(mcq: Dict) -> bool:
    """检测模板/占位题：模型未生成真实内容，而是输出了元描述。"""
    q = str(mcq.get("question", "")).lower()
    opts = mcq.get("answer_options", {})
    opt_vals = " ".join(str(v).lower() for v in opts.values()) if isinstance(opts, dict) else ""
    placeholder_q = ("demonstrate the skill" in q) or ("skill described in" in q and "select all" in q)
    placeholder_opts = ("correct choice" in opt_vals and "distractor" in opt_vals) or \
                       ("matches the standard" in opt_vals) or \
                       ("a correct" in opt_vals and "an incorrect" in opt_vals) or \
                       ("correct answer" in opt_vals and "incorrect answer" in opt_vals and
                        opt_vals.count("correct answer") >= 2)
    return placeholder_q or placeholder_opts


def is_valid_mcq(mcq: Dict) -> Tuple[bool, str]:
    """校验题目是否符合 InceptBench 要求（支持 mcq/msq/fill-in）"""
    if not isinstance(mcq, dict):
        return False, "not a dict"
    if _is_placeholder(mcq):
        return False, "placeholder/template question detected (no real content)"
    qtype = str(mcq.get("type", "mcq")).lower().strip()

    if qtype == "fill-in":
        missing = [k for k in FILLIN_REQUIRED_FIELDS if k not in mcq]
        if missing:
            return False, f"missing fields: {missing}"
        answer = str(mcq.get("answer", "")).strip()
        if not answer:
            return False, "answer must not be empty for fill-in"
        question = str(mcq.get("question", ""))
        q_lower = question.lower()
        has_blank_indicator = (
            "___" in question
            or "blank" in q_lower
            or "type the" in q_lower
            or "type a " in q_lower
            or "type your" in q_lower
            or "write the" in q_lower
            or "write a " in q_lower
            or "enter the" in q_lower
            or "fill in" in q_lower
            or "identify the" in q_lower
            or "identify and type" in q_lower
        )
        if not has_blank_indicator:
            return False, "fill-in question should contain a blank (______ or mention 'blank') or a typing instruction (e.g. 'Type the...')"
        # acceptable_answers 硬校验
        aa = mcq.get("acceptable_answers", [])
        if not isinstance(aa, list):
            aa = [str(aa)] if aa else []
        answer_lower = answer.lower().strip()
        aa_lower = [str(a).lower().strip() for a in aa]
        if answer_lower not in aa_lower:
            return False, f"fill-in answer '{answer}' not found in acceptable_answers {aa}"

        # --- 硬校验: AA-passage 一致性 ---
        # 当问题明确要求 "from the text/passage/story" 时，单词级 AA 条目必须出现在 passage 中
        # 排除 "from the options"/"from the choices" 等非 passage 引用
        # 仅检查单词条目（≤1词），多词短语交给评估方判断
        _from_text_cues = ["word from the text", "word from the passage", "word from the story",
                           "word from the paragraph", "phrase from the text", "phrase from the passage",
                           "phrase from the story"]
        _exclude_cues = ["from the options", "from the choices", "from the list",
                         "from the sentence", "add -ing", "add -ed", "add -s",
                         "with an -ed", "with -ed", "-ed ending", "-ing ending",
                         "with an -ing", "with -ing", "-s ending"]
        requires_text_word = (any(cue in q_lower for cue in _from_text_cues)
                              and not any(exc in q_lower for exc in _exclude_cues))
        if requires_text_word:
            import re as _re_aa
            passage_text = ""
            for line in question.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                low = stripped.lower()
                is_instruction = (low.startswith("type ") or low.startswith("fill in")
                                  or low.startswith("question:") or low.startswith("directions:"))
                if is_instruction or stripped.startswith("**"):
                    continue
                cleaned = _re_aa.sub(r'__{2,}', ' ', stripped)
                passage_text += " " + _re_aa.sub(r'[,;:!?\.\'\"\*\(\)\[\]\u201c\u201d\u2018\u2019]', ' ', cleaned.lower())
            passage_text = " " + _re_aa.sub(r'\s+', ' ', passage_text) + " "
            if len(passage_text) > 40:
                for entry in aa:
                    entry_clean = str(entry).strip().lower().strip('"\'.,;:!? ')
                    if not entry_clean or len(entry_clean.split()) > 1:
                        continue
                    if f" {entry_clean} " not in passage_text:
                        return False, f"fill-in AA entry '{entry}' not found in passage text (question requires word from text)"

        # --- 硬校验: 引用 passage 必须存在 ---
        # 仅对阅读类标准 (RL/RI) 严格检查，写作类标准 (W) 问题可能自然较短
        _ref_cues = ["based on the text", "based on the passage", "according to the text",
                      "according to the passage", "read the passage", "read the text",
                      "read the story", "read the paragraph"]
        refs_passage = any(cue in q_lower for cue in _ref_cues)
        std_code = str(mcq.get("standard", ""))
        is_reading_standard = any(std_code.startswith(p) for p in
                                   ["CCSS.ELA-LITERACY.RL", "CCSS.ELA-LITERACY.RI"])
        if refs_passage and is_reading_standard:
            non_instruction_text = ""
            for line in question.split("\n"):
                stripped = line.strip()
                if stripped and "______" not in stripped and not stripped.startswith("**"):
                    non_instruction_text += stripped + " "
            if len(non_instruction_text.split()) < 20:
                return False, "fill-in references a text/passage but no substantial passage found in question (need >=20 words)"

        # --- 硬校验: 答案泄露检测 ---
        # 仅当答案单词直接紧邻 blank（前后 2 词内）才拦截
        # 排除 Word Bank / 选项提示 / 结论标签等合理的教学辅助
        if "______" in question and answer_lower and len(answer_lower.split()) == 1:
            blank_idx = question.lower().find("______")
            if blank_idx >= 0:
                text_before = question[:blank_idx]
                text_after = question[blank_idx + 6:]
                import re as _re_giveaway
                text_after_clean = _re_giveaway.split(
                    r'(?i)\(?\s*(?:word\s*bank|type\s+|choose\s+from|options?\s*:|conclusion\s|option\s)',
                    text_after, maxsplit=1)[0]
                context_before = text_before.split()[-2:]
                context_after = text_after_clean.split()[:2]
                nearby = " ".join(context_before + context_after).lower()
                nearby_clean = _re_giveaway.sub(r'[,;:!?\.\'\"\*\(\)\[\]]', ' ', nearby)
                if f" {answer_lower} " in f" {nearby_clean} ":
                    return False, f"fill-in answer '{answer}' appears adjacent to the blank (answer giveaway)"

        return True, ""

    if qtype == "msq":
        missing = [k for k in MSQ_REQUIRED_FIELDS if k not in mcq]
        if missing:
            return False, f"missing fields: {missing}"
        opts = mcq.get("answer_options")
        if not isinstance(opts, dict):
            return False, "answer_options must be dict"
        keys = set(str(k).upper().strip() for k in opts.keys())
        if keys != {"A", "B", "C", "D"}:
            return False, f"answer_options keys must be A,B,C,D, got {keys}"
        ans_raw = str(mcq.get("answer", "")).upper().strip()
        ans_letters = sorted(set(l.strip() for l in ans_raw.replace(" ", "").split(",") if l.strip()))
        if len(ans_letters) < 2:
            return False, f"msq answer must have at least 2 correct options, got {ans_raw}"
        if len(ans_letters) > 3:
            return False, f"msq answer should have at most 3 correct options, got {ans_raw}"
        for l in ans_letters:
            if l not in ("A", "B", "C", "D"):
                return False, f"msq answer letters must be A/B/C/D, got {l}"
            if l not in opts:
                return False, f"answer {l} not in answer_options"
        return True, ""

    # MCQ (default)
    required = MCQ_REQUIRED_FIELDS + MCQ_OPTION_FIELDS
    missing = [k for k in required if k not in mcq]
    if missing:
        return False, f"missing fields: {missing}"
    opts = mcq.get("answer_options")
    if not isinstance(opts, dict):
        return False, "answer_options must be dict"
    keys = set(str(k).upper().strip() for k in opts.keys())
    if keys != {"A", "B", "C", "D"}:
        return False, f"answer_options keys must be A,B,C,D, got {keys}"
    ans = str(mcq.get("answer", "")).upper().strip()
    if ans not in ("A", "B", "C", "D"):
        return False, f"answer must be A/B/C/D, got {ans}"
    if ans not in opts:
        return False, f"answer {ans} not in answer_options"
    return True, ""


def load_jsonl(path: str, max_lines: Optional[int] = None) -> List[Dict]:
    """加载 jsonl 文件"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return samples


def extract_user_prompt_from_messages(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def extract_assistant_response(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def extract_mcq_json_from_assistant(content: str) -> Optional[Dict]:
    json_obj = extract_json_from_text(content)
    if not json_obj:
        return None
    ok, _ = is_valid_mcq(json_obj)
    return json_obj if ok else None


def extract_standard_from_user(user_text: str) -> Optional[str]:
    import re
    m = re.search(r"(CCSS\.ELA-LITERACY\.[A-Z0-9.-]+)", user_text)
    if m:
        return m.group(1)
    m = re.search(r"Standard ID:\s*([^\s\n]+)", user_text, re.I)
    if m:
        return m.group(1).strip()
    return None


def process_messages_file(samples: List[Dict]) -> List[Dict]:
    results = []
    for s in samples:
        if "messages" not in s:
            continue
        msgs = s.get("messages", [])
        if len(msgs) < 3:
            continue
        user = extract_user_prompt_from_messages(msgs)
        assistant = extract_assistant_response(msgs)
        if not user or not assistant:
            continue
        mcq = extract_mcq_json_from_assistant(assistant)
        if not mcq:
            continue
        standard = extract_standard_from_user(user) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user,
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "messages",
            "score": 1.0,
        })
    return results


def process_dpo_file(samples: List[Dict]) -> List[Dict]:
    results = []
    for s in samples:
        if "prompt" not in s or "chosen" not in s:
            continue
        prompt = s.get("prompt", [])
        chosen = s.get("chosen", {})
        if isinstance(chosen, dict):
            content = chosen.get("content", "")
        else:
            content = str(chosen)
        if not content.strip():
            continue
        mcq = extract_mcq_json_from_assistant(content)
        if not mcq:
            continue
        user_prompt = ""
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                user_prompt = m.get("content", "")
        if not user_prompt:
            user_prompt = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
        meta = s.get("metadata", {})
        chosen_score = meta.get("chosen_score", 0.5)
        standard = meta.get("standard") or extract_standard_from_user(user_prompt) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user_prompt[:2000],
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "dpo",
            "score": float(chosen_score),
        })
    return results


def select_diverse_examples(candidates: List[Dict], n: int = 5, seed: int = 42) -> List[Dict]:
    """从候选中选出多样性好的示例"""
    random.seed(seed)
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    seen = set()
    selected = []
    for c in sorted_candidates:
        if len(selected) >= n:
            break
        key = (c["standard"], c["difficulty"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(c)
    for c in sorted_candidates:
        if len(selected) >= n:
            break
        if c not in selected:
            selected.append(c)
    return selected[:n]


def run(
    input_dir: str = "raw_data",
    output: str = "processed_training_data/examples.json",
    n: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """从 raw_data 筛选示例并保存到 output"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    all_candidates = []
    for fpath in sorted(input_dir.glob("*.jsonl")):
        samples = load_jsonl(str(fpath))
        if not samples:
            continue
        if "messages" in samples[0]:
            candidates = process_messages_file(samples)
        elif "prompt" in samples[0] and "chosen" in samples[0]:
            candidates = process_dpo_file(samples)
        else:
            continue
        all_candidates.extend(candidates)
        print(f"  {fpath.name}: 提取 {len(candidates)} 条候选")

    if not all_candidates:
        print("未找到有效候选样本")
        return []

    selected = select_diverse_examples(all_candidates, n=n, seed=seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(selected)} 条示例到: {output_path}")
    return selected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="筛选示例（select_examples）")
    parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    parser.add_argument("--output", default="processed_training_data/examples.json", help="输出 JSON 路径")
    parser.add_argument("-n", type=int, default=5, help="示例数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(input_dir=args.input_dir, output=args.output, n=args.n, seed=args.seed)
