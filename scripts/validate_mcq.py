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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.select_examples import is_valid_mcq


def _get_correct_option_text(mcq: dict) -> tuple[dict, str, str]:
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


def _explanation_references_correct_option(explanation: str, ans: str, correct_text: str) -> bool:
    """解释是否提及正确选项：含选项关键词 或 选项字母（option A / choice B 等）。"""
    if not explanation or ans not in "ABCD":
        return True
    expl_lower = explanation.lower()
    # 选项关键词出现
    words = [w.strip(".,;:!?\"'") for w in correct_text.split() if len(w) > 2]
    if words and sum(1 for w in words[:5] if w.lower() in expl_lower) > 0:
        return True
    # 选项字母出现：Option A / Choice B / answer is C / (D) / A is correct 等
    a = ans.lower()
    if any(p in expl_lower for p in [
        f"option {a}", f"choice {a}", f"answer {a}", f"{a} is", f"({a})", f"{a}).",
        f"correct answer is {a}", f"answer is {a}",
    ]):
        return True
    # 正则：option/choice 后跟可选空格再跟字母
    if re.search(r"\b(option|choice|answer)\s*[:\s]*\s*" + re.escape(a) + r"\b", expl_lower):
        return True
    return False


def validate_mcq(mcq: dict, index: int = 0) -> list[str]:
    """
    校验单条 MCQ 的潜在问题。返回问题列表，空列表表示通过。
    """
    issues = []

    # 基础校验（与 select_examples 一致）
    ok, msg = is_valid_mcq(mcq)
    if not ok:
        issues.append(msg)
        return issues

    opts, ans, correct_text = _get_correct_option_text(mcq)
    explanation = mcq.get("answer_explanation", "")

    # 1. 解析是否提及正确选项（启发式：选项关键词 或 选项字母 出现即可）
    if correct_text and len(correct_text) > 3 and explanation:
        words = [w.strip(".,;:!?\"'") for w in correct_text.split() if len(w) > 2]
        ref_ok = _explanation_references_correct_option(explanation, ans, correct_text)
        if not ref_ok and len(words) >= 2:
            issues.append("explanation_may_not_reference_correct_option")

    # 2. 检查 answer 键与选项键一致
    if ans not in opts:
        issues.append(f"answer_{ans}_not_in_options")

    # 3. 题干与选项一致性：若问 "Which word..." 但正确选项是短语
    question = mcq.get("question", "")
    if "which word" in question.lower() and " " in correct_text.strip():
        issues.append("stem_says_word_but_answer_is_phrase")

    return issues


def run(input_path: str, output_report: str | None = None, strict: bool = False) -> dict:
    """
    校验 MCQ 文件，返回统计报告。
    strict: 若 True，有问题的题目也写入报告但不阻止
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]

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
    args = parser.parse_args()

    report = run(args.input, output_report=args.output, strict=args.strict)
    print(f"总数: {report['total']}, 通过: {report['passed']}, 有问题: {report['failed']}")
    if report["issues"]:
        print("\n问题样本（前 5 条）:")
        for x in report["issues"][:5]:
            print(f"  [{x['index']}] {x['id']} ({x['standard']}): {x['issues']}")
    sys.exit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
