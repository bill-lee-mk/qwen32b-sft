#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从评估结果中的失败题提取 InceptBench 的 suggested_improvements / reasoning，
聚合成：全局规则（适用所有题）+ 针对性规则（按 standard 或 (standard, difficulty)），
写入 processed_training_data/prompt_rules.json，供 build_prompt 注入。
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _score_from_result(r: dict) -> float | None:
    if not r:
        return None
    s = r.get("overall_score")
    if s is not None and isinstance(s, (int, float)):
        return float(s)
    for ev in (r.get("evaluations") or {}).values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        s = (inc.get("overall") or {}).get("score")
        if s is not None:
            return float(s)
    return None


def _get_overall(r: dict) -> dict:
    """从单条 result 取 evaluations.*.inceptbench_new_evaluation.overall"""
    for ev in (r.get("evaluations") or {}).values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        ov = inc.get("overall")
        if ov:
            return ov
    return {}


def extract_failure_feedback(
    results_path: str,
    mcqs_path: str,
    threshold: float = 0.85,
) -> list[tuple[str, str, str | None, str | None]]:
    """
    返回 [(standard, difficulty, suggested_improvements, reasoning), ...] 仅对得分 < threshold 的题目。
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)
    scores = results.get("scores", [])
    result_list = results.get("results") or []
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    out = []
    for i in range(len(items)):
        s = scores[i] if i < len(scores) else None
        if s is None and i < len(result_list):
            s = _score_from_result(result_list[i])
        if s is not None and float(s) >= threshold:
            continue
        m = items[i]
        std = m.get("standard", "unknown")
        diff = m.get("difficulty", "medium")
        ov = _get_overall(result_list[i]) if i < len(result_list) else {}
        sug = ov.get("suggested_improvements")
        reason = ov.get("reasoning") or ov.get("internal_reasoning") or ""
        if isinstance(sug, str) and sug.strip():
            out.append((std, diff, sug.strip(), reason[:500] if reason else None))
        elif reason:
            out.append((std, diff, None, reason[:500]))
        else:
            out.append((std, diff, None, None))
    return out


# 关键词 → 全局规则文案（当失败反馈中出现该主题时，加入这条全局规则）
_GLOBAL_THEME_RULES = [
    (["distractor", "non-word", "implausible", "plausible"], "Use only plausible, real-word distractors; avoid non-words or invented forms."),
    (["option", "duplicate", "identical", "same text"], "Ensure all four options A/B/C/D have different text; no duplicate options."),
    (["explanation", "answer_explanation", "contradict", "wrong option"], "answer_explanation must describe only the correct option and must not reference or contradict wrong options."),
    (["plural", "Which choices", "Which options"], "For single-answer MCQs use singular wording (e.g. 'Which choice...'), not 'Which choices/options'."),
    (["image", "picture", "stimulus"], "Do not refer to an image or picture in the stem unless you provide image_url."),
    (["tense", "time cue", "yesterday", "present tense"], "Keep tense and time cues consistent between stem and options."),
    (["blank", "stem", "option", "grammatical"], "If the stem has a blank, each option must be a phrase that fits the blank grammatically; avoid full clauses as options for a single blank."),
]


def _normalize_snippet(t: str, max_len: int = 200) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        t = t[: max_len - 3] + "..."
    return t


def load_example_keys(examples_path: str | Path) -> set[tuple[str, str]]:
    """从 examples.json 加载已有示例的 (standard, difficulty) 集合。用于仅对有示例仍低分的组合加针对性规则。"""
    path = Path(examples_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()
    out = set()
    for item in data if isinstance(data, list) else []:
        std = item.get("standard")
        diff = item.get("difficulty")
        if std and diff:
            out.add((str(std), str(diff)))
    return out


def aggregate_rules(
    feedback_list: list[tuple[str, str, str | None, str | None]],
    max_global: int = 5,
    max_per_standard: int = 2,
    max_per_standard_difficulty: int = 3,
    only_targeted_keys: set[tuple[str, str]] | None = None,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    将 (std, diff, suggested_improvements, reasoning) 聚合成：
    - global_rules: 列表（来自所有失败反馈的主题）
    - by_standard / by_standard_difficulty: 仅当 only_targeted_keys 为 None 或 (std,diff) 在 only_targeted_keys 时加入
    """
    global_rules: list[str] = []
    by_standard: dict[str, list[str]] = defaultdict(list)
    by_standard_difficulty: dict[str, list[str]] = defaultdict(list)

    # 收集所有失败片段用于全局主题检测；针对性规则仅对 only_targeted_keys 内的 (std,diff) 加入
    all_snippets: list[str] = []
    for std, diff, sug, reason in feedback_list:
        if std == "unknown":
            continue
        if sug:
            snip = _normalize_snippet(sug, 180)
        elif reason:
            snip = _normalize_snippet(reason, 180)
        else:
            continue
        all_snippets.append(snip)
        key = f"{std}|{diff}"
        if only_targeted_keys is None or (std, diff) in only_targeted_keys:
            if snip not in by_standard_difficulty[key]:
                by_standard_difficulty[key].append(snip)
            if snip not in by_standard[std]:
                by_standard[std].append(snip)

    # 全局规则：按主题匹配
    snip_lower = " ".join(all_snippets).lower()
    for keywords, rule_text in _GLOBAL_THEME_RULES:
        if any(kw in snip_lower for kw in keywords) and rule_text not in global_rules:
            global_rules.append(rule_text)
            if len(global_rules) >= max_global:
                break

    # 截断
    for k in list(by_standard.keys()):
        by_standard[k] = by_standard[k][:max_per_standard]
    for k in list(by_standard_difficulty.keys()):
        by_standard_difficulty[k] = by_standard_difficulty[k][:max_per_standard_difficulty]

    return global_rules, dict(by_standard), dict(by_standard_difficulty)


def merge_into_prompt_rules(
    new_global: list[str],
    new_by_standard: dict[str, list[str]],
    new_by_standard_difficulty: dict[str, list[str]],
    rules_path: str | Path,
    max_global_total: int = 10,
    max_per_key_total: int = 5,
) -> dict:
    """与现有 prompt_rules.json 合并，去重并限制条数，写回。返回合并后的完整结构。"""
    path = Path(rules_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = {"global_rules": [], "by_standard": {}, "by_standard_difficulty": {}}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    def _dedupe_append(current: list, new: list, cap: int) -> list:
        seen = {s.strip().lower(): s for s in current}
        for s in new:
            k = s.strip().lower()[:120]
            if k not in seen:
                seen[k] = s.strip()
        return list(seen.values())[:cap]

    global_rules = _dedupe_append(existing.get("global_rules") or [], new_global, max_global_total)
    by_standard = dict(existing.get("by_standard") or {})
    for std, rules in new_by_standard.items():
        by_standard[std] = _dedupe_append(by_standard.get(std) or [], rules, max_per_key_total)
    by_standard_difficulty = dict(existing.get("by_standard_difficulty") or {})
    for key, rules in new_by_standard_difficulty.items():
        by_standard_difficulty[key] = _dedupe_append(
            by_standard_difficulty.get(key) or [], rules, max_per_key_total
        )

    out = {
        "global_rules": global_rules,
        "by_standard": by_standard,
        "by_standard_difficulty": by_standard_difficulty,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def run(
    results_path: str,
    mcqs_path: str,
    rules_output: str = "processed_training_data/prompt_rules.json",
    threshold: float = 0.85,
    max_global: int = 5,
    max_per_standard: int = 2,
    max_per_standard_difficulty: int = 3,
    examples_path: str | Path | None = None,
) -> dict:
    """
    从 results + mcqs 提取低分反馈 → 聚合全局/针对性规则 → 合并写入 prompt_rules.json。
    若提供 examples_path：仅对「该 (std,diff) 在 examples 中已有示例」的失败组合加针对性规则（有示例仍低分 → 改 prompt）。
    """
    feedback = extract_failure_feedback(results_path, mcqs_path, threshold)
    if not feedback:
        return {"updated": False, "reason": "no_failures", "feedback_count": 0}

    only_targeted_keys = load_example_keys(examples_path) if examples_path else None

    global_rules, by_std, by_key = aggregate_rules(
        feedback,
        max_global=max_global,
        max_per_standard=max_per_standard,
        max_per_standard_difficulty=max_per_standard_difficulty,
        only_targeted_keys=only_targeted_keys,
    )
    merged = merge_into_prompt_rules(
        global_rules, by_std, by_key,
        rules_path=rules_output,
    )
    return {
        "updated": True,
        "feedback_count": len(feedback),
        "global_rules_count": len(merged["global_rules"]),
        "by_standard_count": len(merged["by_standard"]),
        "by_standard_difficulty_count": len(merged["by_standard_difficulty"]),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="从失败评估结果改进 prompt 规则")
    p.add_argument("--results", required=True, help="评估结果 JSON")
    p.add_argument("--mcqs", required=True, help="MCQ JSON")
    p.add_argument("--output", default="processed_training_data/prompt_rules.json", help="prompt_rules 输出路径")
    p.add_argument("--threshold", type=float, default=0.85, help="低于此分视为失败")
    p.add_argument("--max-global", type=int, default=5, help="本轮最多新增全局规则条数")
    p.add_argument("--max-per-standard", type=int, default=2, help="每个 standard 最多保留条数")
    p.add_argument("--max-per-standard-difficulty", type=int, default=3, help="每个 (std,diff) 最多保留条数")
    p.add_argument("--examples", default=None, help="examples.json 路径；提供则仅对有示例仍低分的组合加针对性规则")
    args = p.parse_args()
    report = run(
        results_path=args.results,
        mcqs_path=args.mcqs,
        rules_output=args.output,
        threshold=args.threshold,
        max_global=args.max_global,
        max_per_standard=args.max_per_standard,
        max_per_standard_difficulty=args.max_per_standard_difficulty,
        examples_path=args.examples,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("updated"):
        print(f"已更新 prompt 规则: 全局 {report.get('global_rules_count', 0)} 条, "
              f"by_standard {report.get('by_standard_count', 0)} 个, "
              f"by_standard_difficulty {report.get('by_standard_difficulty_count', 0)} 个")
    return 0 if report.get("updated") or report.get("reason") == "no_failures" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
