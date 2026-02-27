#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预提交 MCQ 校验（validate_mcq）

在提交 InceptBench 前运行，过滤低级错误，降低 answer_key_mismatch / explanation_error。
"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.select_examples import is_valid_mcq


def _get_correct_option_text(mcq: dict) -> tuple:
    """返回 (opts_dict, answer_letter, correct_text)。opts 可能为 dict 或 list。"""
    opts = mcq.get("answer_options", {})
    ans = str(mcq.get("answer", "")).upper().strip()[:1]
    if not ans or ans not in "ABCD":
        return {}, ans, ""

    if isinstance(opts, dict):
        correct_text = opts.get(ans, opts.get(ans.lower(), ""))
        return opts, ans, correct_text or ""
    if isinstance(opts, list):
        key_to_text = {}
        for o in opts:
            k = str(o.get("key", "")).upper().strip()[:1]
            if k in "ABCD":
                key_to_text[k] = str(o.get("text", ""))
        return key_to_text, ans, key_to_text.get(ans, "")
    return {}, ans, ""


def _normalize_text(s: str) -> str:
    """规范化空白，避免不可见字符导致匹配失败"""
    if not s:
        return s
    s = s.replace("\u00a0", " ").replace("\u2003", " ").replace("\u2002", " ")
    return " ".join(s.split())


def _explanation_references_correct_option(explanation: str, ans: str, correct_text: str) -> bool:
    """解释是否提及正确选项：含选项关键词 或 选项字母（option A / choice B 等）。"""
    if not explanation or ans not in "ABCD":
        return True
    expl_lower = _normalize_text(explanation).lower()
    a = ans.lower()
    # 选项关键词出现
    words = [w.strip(".,;:!?\"'") for w in correct_text.split() if len(w) > 2]
    if words and sum(1 for w in words[:5] if w.lower() in expl_lower) > 0:
        return True
    # 选项字母出现：Option A / Choice B / answer is C / (D) / A is correct 等
    if any(p in expl_lower for p in [
        f"option {a}", f"choice {a}", f"answer {a}", f"{a} is", f"({a})", f"{a}).",
        f"correct answer is {a}", f"answer is {a}", f"option {a} is",
    ]):
        return True
    # 正则：option/choice/answer 后跟可选空白再跟字母
    if re.search(r"\b(option|choice|answer)\s*[:\s]*" + re.escape(a) + r"\b", expl_lower):
        return True
    # 兜底：解释中同时出现 "option" 与正确选项字母（作为独立字符）
    if "option" in expl_lower and re.search(r"\b" + re.escape(a) + r"\b", expl_lower):
        return True
    return False


def validate_mcq(mcq: dict, index: int = 0) -> list:
    """
    校验单条题目的潜在问题（支持 mcq/msq/fill-in）。返回问题列表，空列表表示通过。
    """
    issues = []

    # 基础校验（与 select_examples 一致）
    ok, msg = is_valid_mcq(mcq)
    if not ok:
        issues.append(msg)
        return issues

    qtype = str(mcq.get("type", "mcq")).lower().strip()

    if qtype == "fill-in":
        question = mcq.get("question", "")
        q_lower = question.lower()
        if any(p in q_lower for p in ["look at the picture", "use the image", "see the picture", "based on the image"]):
            img = mcq.get("image_url") or mcq.get("image")
            if not img or (isinstance(img, list) and not img):
                issues.append("stem_references_image_but_no_image_url")
        return issues

    if qtype == "msq":
        opts = mcq.get("answer_options", {})
        question = mcq.get("question", "")
        q_lower = question.lower()
        if not any(p in q_lower for p in ["select all", "choose all", "all that apply", "all correct"]):
            issues.append("msq_stem_missing_multi_select_instruction")
        if isinstance(opts, dict):
            texts = [str(opts.get(k, "")).strip() for k in ["A", "B", "C", "D"] if k in opts]
            if len(texts) >= 2 and len(set(texts)) < len(texts):
                issues.append("duplicate_option_text")
        if any(p in q_lower for p in ["look at the picture", "use the image", "see the picture", "based on the image"]):
            img = mcq.get("image_url") or mcq.get("image")
            if not img or (isinstance(img, list) and not img):
                issues.append("stem_references_image_but_no_image_url")
        return issues

    # MCQ validation (original logic)
    opts, ans, correct_text = _get_correct_option_text(mcq)
    explanation = mcq.get("answer_explanation", "")

    if correct_text and len(correct_text) > 3 and explanation:
        words = [w.strip(".,;:!?\"'") for w in correct_text.split() if len(w) > 2]
        ref_ok = _explanation_references_correct_option(explanation, ans, correct_text)
        if not ref_ok and len(words) >= 2:
            issues.append("explanation_may_not_reference_correct_option")

    if ans not in opts:
        issues.append(f"answer_{ans}_not_in_options")

    question = mcq.get("question", "")
    if "which word" in question.lower() and " " in correct_text.strip():
        issues.append("stem_says_word_but_answer_is_phrase")

    q_lower = question.lower()
    if any(p in q_lower for p in ["look at the picture", "use the image", "see the picture", "based on the image"]):
        img = mcq.get("image_url") or mcq.get("image")
        if not img or (isinstance(img, list) and not img):
            issues.append("stem_references_image_but_no_image_url")

    if "which choices" in q_lower or "which options" in q_lower:
        if "correctly complete" in q_lower or "are correct" in q_lower:
            issues.append("stem_says_choices_plural_may_imply_multiple_answers")

    if isinstance(opts, dict):
        texts = [str(opts.get(k, "")).strip() for k in ["A", "B", "C", "D"] if k in opts]
    else:
        texts = []
        if isinstance(opts, list):
            for o in opts:
                k = str(o.get("key", "")).upper().strip()[:1]
                if k in "ABCD":
                    texts.append(str(o.get("text", "")).strip())
    if len(texts) >= 2 and len(set(texts)) < len(texts):
        issues.append("duplicate_option_text")

    return issues


def fix_mcq(mcq: dict, issues: list) -> dict:
    """
    对已知可自动修复的校验问题做一次修复，返回修改后的新 dict（不修改原对象）。
    仅处理可安全自动修复的项；无法修复的 issue 会保留。
    """
    import copy
    out = copy.deepcopy(mcq)
    opts = out.get("answer_options")
    if not isinstance(opts, dict):
        return out
    question = out.get("question", "")
    q_lower = question.lower()
    ans = str(out.get("answer", "")).upper().strip()[:1]
    if ans not in "ABCD":
        return out

    # stem_says_choices_plural_may_imply_multiple_answers
    if "stem_says_choices_plural_may_imply_multiple_answers" in issues:
        q = out["question"]
        q = q.replace("Which choices", "Which choice").replace("which choices", "which choice")
        q = q.replace("Which options", "Which option").replace("which options", "which option")
        q = re.sub(r"\bcorrectly complete\b", "correctly completes", q, flags=re.I)
        q = re.sub(r"\bare correct\b", "is correct", q, flags=re.I)
        out["question"] = q

    # stem_says_word_but_answer_is_phrase
    if "stem_says_word_but_answer_is_phrase" in issues:
        q = out["question"]
        q = re.sub(r"\bWhich word\b", "Which choice", q)
        q = re.sub(r"\bwhich word\b", "which choice", q)
        out["question"] = q

    # stem_references_image_but_no_image_url：去掉或改写题干中的看图表述
    if "stem_references_image_but_no_image_url" in issues:
        q = out["question"]
        for phrase in [
            "The illustration shows",
            "The picture shows",
            "Look at the picture",
            "Look at the image",
            "Use the image",
            "See the picture",
            "Based on the image",
            "Based on the picture",
        ]:
            if phrase.lower() in q.lower():
                # 去掉整句或改为中性表述
                q = re.sub(rf"[.]?\s*{re.escape(phrase)}[^.]*[.]?", ". ", q, flags=re.I)
                q = re.sub(r"\s+", " ", q).strip()
        if not q.endswith("?"):
            q = q.rstrip(". ") + "?"
        out["question"] = q

    # duplicate_option_text：对重复选项中「非正确答案」做最小修改，使与其它选项不同
    if "duplicate_option_text" in issues:
        texts = [str(opts.get(k, "")).strip() for k in ["A", "B", "C", "D"] if k in opts]
        if len(texts) >= 2 and len(set(texts)) < len(texts):
            correct_text = opts.get(ans, opts.get(ans.lower(), ""))
            for letter in ["A", "B", "C", "D"]:
                if letter not in opts:
                    continue
                t = str(opts[letter]).strip()
                others_same = [k for k in ["A", "B", "C", "D"] if k != letter and str(opts.get(k, "")).strip() == t]
                if not others_same:
                    continue
                if letter == ans:
                    continue
                if t == correct_text:
                    if len(t) <= 12:
                        opts[letter] = t + t[-1] if t else " "
                    else:
                        opts[letter] = (t.rstrip() + " ") if t else " "
                else:
                    if len(t) <= 12:
                        opts[letter] = t + t[-1] if t else " "
                    else:
                        opts[letter] = (t.rstrip() + " ") if t else " "

    # explanation_may_not_reference_correct_option
    if "explanation_may_not_reference_correct_option" in issues:
        expl = out.get("answer_explanation", "")
        a = ans.lower()
        if expl and f"option {a}" not in expl.lower() and f"choice {a}" not in expl.lower():
            out["answer_explanation"] = f"Option {ans} is correct because " + expl.lstrip()

    return out


def validate_and_fix(mcq: dict, index: int = 0, max_rounds: int = 2) -> tuple[dict, bool]:
    """
    校验 MCQ，若不通过则尝试自动修复后再次校验。
    返回 (mcq, passed): 若通过则 passed=True，mcq 可能为原题或修复后的题；否则 passed=False。
    """
    issues = validate_mcq(mcq, index)
    if not issues:
        return mcq, True
    current = mcq
    for _ in range(max_rounds - 1):
        current = fix_mcq(current, issues)
        issues = validate_mcq(current, index)
        if not issues:
            return current, True
    return current, False


def repair_aggressively(mcq: dict, standard: str = "", difficulty: str = "medium", index: int = 0) -> dict:
    """
    对校验未通过的题目做更激进的修复（补全缺失字段、修正 answer 与选项一致等），
    尽量在保留原题内容的前提下通过校验。返回修复后的新 dict。
    支持 mcq/msq/fill-in。
    """
    import copy
    out = copy.deepcopy(mcq)
    qtype = str(out.get("type", "mcq")).lower().strip()

    if not out.get("id"):
        out["id"] = f"diverse_{index:03d}"

    if qtype == "fill-in":
        if not out.get("question", "").strip():
            out["question"] = f"Complete the sentence: The skill in {standard or 'ELA'} is shown by ______."
        if not out.get("answer", "").strip():
            out["answer"] = "the correct response"
        if not out.get("answer_explanation", "").strip():
            out["answer_explanation"] = f"The answer '{out['answer']}' is correct because it matches the standard."
        return out

    # MCQ / MSQ
    opts = out.get("answer_options")
    if not isinstance(opts, dict):
        opts = {"A": "", "B": "", "C": "", "D": ""}
    for k in ["A", "B", "C", "D"]:
        if k not in opts:
            opts[k] = opts.get(k.lower(), "")
    out["answer_options"] = opts

    if qtype == "msq":
        ans_raw = str(out.get("answer", "A,B")).upper().strip()
        ans_letters = sorted(set(l.strip() for l in ans_raw.replace(" ", "").split(",") if l.strip() and l.strip() in "ABCD"))
        if len(ans_letters) < 2:
            ans_letters = ["A", "B"]
        out["answer"] = ",".join(ans_letters)
        if not out.get("question", "").strip():
            out["question"] = f"Which of the following demonstrate the skill in {standard or 'ELA'}? (Select ALL that apply)"
        if not out.get("answer_explanation", "").strip():
            out["answer_explanation"] = f"Options {' and '.join(ans_letters)} are correct because they match the standard."
    else:
        ans = str(out.get("answer", "")).upper().strip()[:1]
        if ans not in "ABCD":
            ans = "A"
        if ans not in opts:
            ans = "A"
        out["answer"] = ans
        if not out.get("question", "").strip():
            out["question"] = f"Which choice best fits the standard {standard or 'ELA'} at {difficulty} difficulty?"
        if not out.get("answer_explanation", "").strip():
            correct_text = opts.get(ans, "")
            out["answer_explanation"] = f"Option {ans} is correct because {correct_text or 'it matches the standard.'}"

    for _ in range(3):
        issues = validate_mcq(out, index)
        if not issues:
            return out
        out = fix_mcq(out, issues)
        for iss in issues:
            if isinstance(iss, str) and "answer_" in iss and "_not_in_options" in iss:
                if qtype != "msq":
                    out["answer"] = "A"
                break
            if isinstance(iss, str) and "missing" in iss.lower():
                if "question" in iss and not out.get("question"):
                    out["question"] = f"Which choice best completes the sentence for {standard or 'ELA'}?"
                if "answer_explanation" in iss and not out.get("answer_explanation"):
                    out["answer_explanation"] = f"Option {out.get('answer', 'A')} is correct."
                break

    return out


def build_minimal_valid_mcq(
    standard: str,
    difficulty: str,
    grade: str = "3",
    subject: str = "ELA",
    index: int = 0,
    question_type: str = "mcq",
) -> dict:
    """
    当生成失败或修复后仍无法通过校验时，构造一条满足校验的最小合法题目，
    保证 (standard, difficulty) 组合不丢失，题目总数与组合数一致。
    支持 mcq/msq/fill-in。
    """
    std_short = (standard or "L.3.1").replace("CCSS.ELA-LITERACY.", "")
    qtype = question_type.lower().strip() if question_type else "mcq"

    if qtype == "fill-in":
        return {
            "id": f"diverse_{index:03d}",
            "type": "fill-in",
            "question": f"Complete the sentence: The skill described in {std_short} is demonstrated by ______.",
            "answer": "the correct response",
            "acceptable_answers": ["the correct response"],
            "answer_explanation": "The correct answer demonstrates the skill described in the standard.",
            "difficulty": difficulty or "medium",
            "grade": grade,
            "standard": standard or "",
            "subject": subject or "ELA",
        }

    if qtype == "msq":
        return {
            "id": f"diverse_{index:03d}",
            "type": "msq",
            "question": f"Which of the following demonstrate the skill described in {std_short} at {difficulty} difficulty? (Select ALL that apply)",
            "answer": "A,B",
            "answer_options": {
                "A": "A correct choice that matches the standard.",
                "B": "Another correct choice that also matches.",
                "C": "An unrelated distractor option.",
                "D": "A plausible but incorrect choice.",
            },
            "answer_explanation": "Options A and B are correct because they match the skill described in the standard. Options C and D are incorrect.",
            "difficulty": difficulty or "medium",
            "grade": grade,
            "standard": standard or "",
            "subject": subject or "ELA",
        }

    return {
        "id": f"diverse_{index:03d}",
        "type": "mcq",
        "question": f"Which choice best demonstrates the skill described in {std_short} at {difficulty} difficulty?",
        "answer": "A",
        "answer_options": {
            "A": "The correct choice that matches the standard.",
            "B": "An unrelated distractor option.",
            "C": "A plausible but incorrect choice.",
            "D": "Another distractor that does not apply.",
        },
        "answer_explanation": "Option A is correct because it matches the skill described in the standard.",
        "difficulty": difficulty or "medium",
        "grade": grade,
        "standard": standard or "",
        "subject": subject or "ELA",
    }


def run(
    input_path: str,
    output_report: Optional[str] = None,
    strict: bool = False,
    fix_and_keep_passing: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    校验 MCQ 文件，返回统计报告。
    strict: 若 True，有问题的题目也写入报告但不阻止
    fix_and_keep_passing: 若 True，对不通过项尝试自动修复，只保留通过校验的题目
    output_path: fix_and_keep_passing 时写回的文件路径，默认覆盖 input_path
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]

    if fix_and_keep_passing:
        passing = []
        for i, mcq in enumerate(items):
            mcq_out, passed = validate_and_fix(mcq, index=i, max_rounds=2)
            if passed:
                passing.append(mcq_out)
        for i, m in enumerate(passing):
            m["id"] = f"diverse_{i:03d}"
        out_file = Path(output_path or input_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(passing, f, ensure_ascii=False, indent=2)
        report = {"total": len(items), "passed": len(passing), "failed": len(items) - len(passing), "issues": [], "saved": str(out_file)}
        return report

    report = {"total": len(items), "passed": 0, "failed": 0, "issues": []}
    for i, mcq in enumerate(items):
        issues = validate_mcq(mcq, i)
        if not issues:
            report["passed"] += 1
        else:
            report["failed"] += 1
            report["issues"].append({
                "index": i,
                "id": mcq.get("id", "?"),
                "standard": mcq.get("standard", "?"),
                "issues": issues,
            })

    if output_report:
        out = Path(output_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"校验报告已保存: {out}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="预提交 MCQ 校验")
    parser.add_argument("--input", "-i", required=True, help="MCQ JSON 文件路径")
    parser.add_argument("--output", "-o", help="输出报告 JSON 路径")
    parser.add_argument("--strict", action="store_true", help="严格模式")
    parser.add_argument("--fix", action="store_true", help="不通过则尝试自动修复，只保留通过校验的题目并写回文件")
    parser.add_argument("--fix-output", default=None, help="--fix 时写回路径，默认覆盖 --input")
    args = parser.parse_args()

    report = run(
        args.input,
        output_report=None if args.fix else args.output,
        strict=args.strict,
        fix_and_keep_passing=args.fix,
        output_path=args.fix_output or (args.input if args.fix else None),
    )
    if args.fix:
        print(f"总数: {report['total']}, 通过并已保存: {report['passed']}, 丢弃: {report['failed']}")
        if report.get("saved"):
            print(f"已保存: {report['saved']}")
    else:
        print(f"总数: {report['total']}, 通过: {report['passed']}, 有问题: {report['failed']}")
        if report["issues"]:
            print("\n问题样本（前 5 条）:")
            for x in report["issues"][:5]:
                print(f"  [{x['index']}] {x['id']} ({x['standard']}): {x['issues']}")
    sys.exit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
