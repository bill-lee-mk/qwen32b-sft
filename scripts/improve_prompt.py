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
        if s is None:
            s = (ev.get("overall") or {}).get("score")  # 第二套格式
        if s is not None:
            return float(s)
    return None


def _get_overall(r: dict) -> dict:
    """从单条 result 取 evaluations.*.overall（兼容主配置与第二套格式）"""
    for ev in (r.get("evaluations") or {}).values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        ov = inc.get("overall") or ev.get("overall")
        if ov:
            return ov
    return {}


def extract_failure_feedback(
    results_path: str,
    mcqs_path: str,
    threshold: float = 0.85,
) -> list[tuple[str, str, str, str | None, str | None]]:
    """
    返回 [(standard, difficulty, qtype, suggested_improvements, reasoning), ...]
    仅对得分 < threshold 的题目。qtype 为 mcq/msq/fill-in。
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
        qtype = m.get("type", "mcq")
        ov = _get_overall(result_list[i]) if i < len(result_list) else {}
        sug = ov.get("suggested_improvements")
        reason = ov.get("reasoning") or ov.get("internal_reasoning") or ""
        if isinstance(sug, str) and sug.strip():
            out.append((std, diff, qtype, sug.strip(), reason[:500] if reason else None))
        elif reason:
            out.append((std, diff, qtype, None, reason[:500]))
        else:
            out.append((std, diff, qtype, None, None))
    return out


def extract_dimension_feedback(
    results_path: str,
    mcqs_path: str,
    threshold: float = 0.85,
) -> list[dict]:
    """
    解析维度级分数，返回每个失败题的详细维度信息。
    返回 [{"standard", "difficulty", "qtype", "overall_score",
           "failed_dims": {"dim_name": {"score": float, "reasoning": str}}}]
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)
    result_list = results.get("results") or []
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    out = []
    for i in range(min(len(items), len(result_list))):
        score = _score_from_result(result_list[i])
        if score is not None and score >= threshold:
            continue
        m = items[i]
        failed_dims = {}
        evals = result_list[i].get("evaluations") or {}
        if isinstance(evals, dict):
            for _key, ev in evals.items():
                inc = ev.get("inceptbench_new_evaluation") or {}
                if not isinstance(inc, dict):
                    continue
                for dim_name, dim_data in inc.items():
                    if dim_name == "overall" or not isinstance(dim_data, dict):
                        continue
                    ds = dim_data.get("score")
                    if ds is not None and isinstance(ds, (int, float)) and ds < threshold:
                        failed_dims[dim_name] = {
                            "score": ds,
                            "reasoning": (dim_data.get("reasoning") or "")[:400],
                        }
                break
        if failed_dims:
            out.append({
                "standard": m.get("standard", "unknown"),
                "difficulty": m.get("difficulty", "medium"),
                "qtype": m.get("type", "mcq"),
                "overall_score": score,
                "failed_dims": failed_dims,
            })
    return out


# 维度级规则：当某维度低分时，根据 reasoning 中的关键词自动匹配精准规则
_DIMENSION_THEME_RULES: dict[str, list[tuple[list[str], str]]] = {
    "factual_accuracy": [
        (["acceptable_answers", "contradict", "conflict", "inconsisten", "mismatch"],
         "CRITICAL for fill-in: Every entry in acceptable_answers must be consistent with what the question asks AND what answer_explanation states. Perform a FINAL CONSISTENCY CHECK before outputting JSON."),
        (["not in the text", "not present", "not in the passage", "not in the story", "synonyms not"],
         "For fill-in: If the question says 'from the text/passage/story', acceptable_answers must ONLY contain words that appear VERBATIM in the text. Do NOT add synonyms not found in the passage."),
        (["overstate", "always", "never", "incorrectly claims", "inaccurately", "misrepresent"],
         "For fill-in: answer_explanation must be factually precise. Do NOT overstate grammar rules. Do NOT claim the answer is the ONLY correct response while listing alternatives."),
        (["capitali", "spacing", "incorrect form", "double-space"],
         "For fill-in: Every acceptable_answers entry must use correct capitalization and spacing for its position in the sentence. Do NOT include entries with wrong case or extra spaces."),
        (["incorrect", "wrong", "not support", "passage does not"],
         "For fill-in: Do NOT include acceptable_answers entries that are factually wrong or unsupported by the passage. Before finalizing, verify each AA entry against the passage text."),
        (["word bank", "metadata", "student-facing"],
         "For fill-in: If a word bank is provided, acceptable_answers must match EXACTLY what the student sees. Do NOT include entries that differ from the word bank."),
        (["punctuation", "form", "incorrect punctuation"],
         "For fill-in punctuation questions: acceptable_answers must use correct punctuation forms. Double-check quotation marks, commas, and sentence-ending punctuation."),
    ],
    "educational_accuracy": [
        (["trivial", "copy", "already visible", "pre-attempt", "giveaway"],
         "For fill-in: The answer must NOT be directly copyable from the question text. The student should need to apply a skill to produce the answer."),
        (["not the specific", "not accurate", "does not match"],
         "For fill-in: Ensure the educational content precisely matches the skill described in the standard. Do not test a related but different skill."),
        (["generic", "overly broad", "omit", "missing", "valid synonym"],
         "For fill-in: acceptable_answers must include common valid synonyms demanded by the prompt. Do NOT use overly generic verbs. Do NOT omit obviously valid alternatives."),
        (["contains the exact answer", "answer word", "explicitly states"],
         "For fill-in: The passage or question stem must NOT contain the exact answer word in a way that makes the task trivially obvious. If the answer appears in the passage, the student must identify WHY it is correct, not simply locate it."),
    ],
    "difficulty_alignment": [
        (["labeled", "hard", "but", "straightforward", "low", "cognitive", "single-step", "single step"],
         "For fill-in hard: The task MUST genuinely require multi-step reasoning, inference, or synthesis (≥3 cognitive steps). If a student can answer by simply locating a word in the passage, it is NOT hard. Self-check: count cognitive steps needed."),
        (["labeled", "medium", "but", "simple", "recall"],
         "For fill-in medium: The task should require at least one inference step beyond direct word retrieval from text (≥2 cognitive steps)."),
        (["too easy", "below", "does not match", "above"],
         "For fill-in: Difficulty label must match actual cognitive demand. Count how many reasoning steps the student needs: easy=1, medium=2, hard≥3."),
    ],
    "clarity_precision": [
        (["ambig", "unclear", "confus", "vague"],
         "For fill-in: Ensure the question and blank are unambiguous. The student should clearly understand what to type."),
        (["duplicate", "already present", "repeat"],
         "For fill-in punctuation questions: Do NOT require the student to retype a word already visible in the question. The blank should test ONLY the punctuation or corrected form."),
        (["too many valid", "multiple valid", "open-ended"],
         "For fill-in: Ensure the blank has a limited set of valid answers. If too many words could fit, narrow the question or add context."),
    ],
    "passage_reference": [
        (["not provided", "external", "unstated"],
         "For fill-in: If the question references a text/passage, the FULL text MUST be included in the question field. Do NOT reference unstated texts."),
    ],
    "mastery_learning_alignment": [
        (["not aligned", "does not assess", "different skill"],
         "For fill-in: The question MUST directly assess the EXACT skill described in the standard, not a related or broader skill."),
        (["pure recall", "memoriz", "rote", "no application", "no reasoning"],
         "For fill-in: Even easy questions must require the student to demonstrate understanding through reading and applying a concept. Avoid questions answerable by memorizing a single phrase (e.g., 'The main idea tells the reader what the text is mostly ______' with answer 'about')."),
        (["recall only", "no meaningful", "trivial"],
         "For fill-in: Design questions that require the student to engage with passage content, not just recall definitions or complete well-known phrases."),
    ],
    "curriculum_alignment": [
        (["mismatch", "misalign", "better aligned", "wrong standard", "different standard"],
         "For fill-in: Verify the question tests the SPECIFIC cognitive action described in the standard. Common mistake: testing main idea (RI.x.2) when the standard is about reasons/evidence (RI.x.8), or testing grammar (L.x.1) when the standard is about writing process (W.x.5)."),
        (["fluency", "RF", "infer", "not reading fluency"],
         "For Reading Foundational (RF) standards: ensure the question tests the EXACT decoding/fluency skill, not inference or comprehension."),
    ],
}


# 关键词 → 全局规则文案（当失败反馈中出现该主题时，加入这条全局规则，适用所有题目）
_GLOBAL_THEME_RULES = [
    (["distractor", "non-word", "implausible", "plausible"], "Use only plausible, real-word distractors; avoid non-words or invented forms."),
    (["option", "duplicate", "identical", "same text"], "Ensure all four options A/B/C/D have different text; no duplicate options."),
    (["explanation", "answer_explanation", "contradict", "wrong option"], "answer_explanation must describe only the correct option and must not reference or contradict wrong options."),
    (["plural", "Which choices", "Which options"], "For single-answer MCQs use singular wording (e.g. 'Which choice...'), not 'Which choices/options'."),
    (["image", "picture", "stimulus"], "Do not refer to an image or picture in the stem unless you provide image_url."),
    (["tense", "time cue", "yesterday", "present tense"], "Keep tense and time cues consistent between stem and options."),
    (["blank", "stem", "option", "grammatical"], "If the stem has a blank, each option must be a phrase that fits the blank grammatically; avoid full clauses as options for a single blank."),
    # 低分题反馈扩展：字典/guide words、术语定义、事实准确性、同伴反馈
    (["guide word", "falls between", "between the guide"], "For dictionary/guide-word tasks (L.3.2.G): ensure only one option falls between the guide words; adjust guide words or options so exactly one choice fits."),
    (["aloud", "out loud", "audibly", "speaking loudly"], "For fluency terms (e.g. RF.3.4.C): use accurate definitions: 'aloud' means 'out loud' or 'audibly,' not 'speaking loudly'."),
    (["false claim", "false statement", "inaccurate", "photosynthesis", "factually correct"], "Ensure all passage content is factually accurate; avoid false or misleading claims in stems or passages."),
    (["peer feedback", "reflect", "actual issue", "actual error", "draft"], "For peer feedback/editing items: ensure the feedback describes an actual error present in the draft; avoid describing errors that don't exist."),
    (["run-on", "fragment", "nonexistent", "doesn't exist", "presume a nonexistent"], "For editing tasks: ensure the draft text actually contains the error students are asked to fix; or rephrase the question to not presume a nonexistent error."),
]

# 标准级规则（当该标准出现低分时，自动注入到 by_standard，仅对该标准生效）
# 格式：standard_id -> [rule1, rule2, ...]，规则在 aggregate_rules 中按标准注入
_STANDARD_SPECIFIC_RULES: dict[str, list[str]] = {
    "CCSS.ELA-LITERACY.L.3.2.G": [
        "For dictionary guide-word tasks: ensure only one option falls between the guide words; if multiple options fit, revise guide words or replace options.",
    ],
    "CCSS.ELA-LITERACY.RF.3.4.C": [
        "Define 'aloud' correctly as 'out loud' or 'audibly,' not 'speaking loudly.' Ensure answer_explanation uses accurate terminology.",
    ],
    "CCSS.ELA-LITERACY.RI.3.10": [
        "Ensure passage content is factually accurate; avoid false claims (e.g., about photosynthesis: plants make their own food, not 'create').",
    ],
    "CCSS.ELA-LITERACY.W.3.5": [
        "For peer feedback items: ensure the feedback reflects an actual error in the draft; or ensure the draft contains the error (run-on, fragment) the question asks to fix.",
    ],
}


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
    feedback_list: list[tuple[str, str, str, str | None, str | None]],
    max_global: int = 5,
    max_per_standard: int = 2,
    max_per_standard_difficulty: int = 3,
    only_targeted_keys: set[tuple[str, str]] | None = None,
    dimension_feedback: list[dict] | None = None,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    将 (std, diff, qtype, suggested_improvements, reasoning) 聚合成：
    - global_rules: 列表（来自所有失败反馈的主题 + 维度级规则）
    - by_standard / by_standard_difficulty_type: key 为 'std|diff|type'，
      仅当 only_targeted_keys 为 None 或 (std,diff) 在 only_targeted_keys 时加入
    """
    global_rules: list[str] = []
    by_standard: dict[str, list[str]] = defaultdict(list)
    by_standard_difficulty: dict[str, list[str]] = defaultdict(list)

    all_snippets: list[str] = []
    failed_standards: set[str] = set()
    for std, diff, qtype, sug, reason in feedback_list:
        if std == "unknown":
            continue
        failed_standards.add(std)
        if sug:
            snip = _normalize_snippet(sug, 180)
        elif reason:
            snip = _normalize_snippet(reason, 180)
        else:
            continue
        all_snippets.append(snip)
        key = f"{std}|{diff}|{qtype}"
        if snip not in by_standard[std]:
            by_standard[std].append(snip)
        if only_targeted_keys is None or (std, diff) in only_targeted_keys:
            if snip not in by_standard_difficulty[key]:
                by_standard_difficulty[key].append(snip)

    # 全局规则：按主题匹配（原有逻辑）
    snip_lower = " ".join(all_snippets).lower()
    for keywords, rule_text in _GLOBAL_THEME_RULES:
        if any(kw in snip_lower for kw in keywords) and rule_text not in global_rules:
            global_rules.append(rule_text)
            if len(global_rules) >= max_global:
                break

    # --- 维度级规则：从 dimension_feedback 中按维度+关键词精准匹配 ---
    if dimension_feedback:
        dim_reasoning_pool: dict[str, list[str]] = defaultdict(list)
        for item in dimension_feedback:
            for dim_name, dim_info in item.get("failed_dims", {}).items():
                reasoning = dim_info.get("reasoning", "")
                if reasoning:
                    dim_reasoning_pool[dim_name].append(reasoning.lower())
                std = item.get("standard", "unknown")
                diff = item.get("difficulty", "medium")
                qtype = item.get("qtype", "mcq")
                key = f"{std}|{diff}|{qtype}"
                if only_targeted_keys is None or (std, diff) in only_targeted_keys:
                    dim_rule_snip = _normalize_snippet(
                        f"[{dim_name}] {reasoning}", 200
                    )
                    if dim_rule_snip not in by_standard_difficulty.get(key, []):
                        by_standard_difficulty[key] = by_standard_difficulty.get(key, [])
                        by_standard_difficulty[key].append(dim_rule_snip)

        for dim_name, reasoning_list in dim_reasoning_pool.items():
            combined = " ".join(reasoning_list)
            dim_rules = _DIMENSION_THEME_RULES.get(dim_name, [])
            for keywords, rule_text in dim_rules:
                if any(kw in combined for kw in keywords):
                    if rule_text not in global_rules:
                        global_rules.append(rule_text)

    # 标准级规则：按失败标准注入（置于列表前部，优先保留）
    for std in failed_standards:
        for rule in _STANDARD_SPECIFIC_RULES.get(std, []):
            if rule not in by_standard[std]:
                by_standard[std].insert(0, rule)

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
    max_per_key_total: int = 3,
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
        seen = {s.strip().lower()[:120]: s for s in current}
        for s in new:
            k = s.strip().lower()[:120]
            if k not in seen:
                seen[k] = s.strip()
        vals = list(seen.values())
        return vals[-cap:] if len(vals) > cap else vals

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
    max_global: int = 8,
    max_per_standard: int = 4,
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

    dim_feedback = extract_dimension_feedback(results_path, mcqs_path, threshold)

    only_targeted_keys = load_example_keys(examples_path) if examples_path else None

    global_rules, by_std, by_key = aggregate_rules(
        feedback,
        max_global=max_global,
        max_per_standard=max_per_standard,
        max_per_standard_difficulty=max_per_standard_difficulty,
        only_targeted_keys=only_targeted_keys,
        dimension_feedback=dim_feedback,
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
    p.add_argument("--max-global", type=int, default=8, help="本轮最多新增全局规则条数")
    p.add_argument("--max-per-standard", type=int, default=4, help="每个 standard 最多保留条数（含标准级规则+反馈）")
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
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report.get("updated"):
        print(f"已更新 prompt 规则: 全局 {report.get('global_rules_count', 0)} 条, "
              f"by_standard {report.get('by_standard_count', 0)} 个, "
              f"by_standard_difficulty {report.get('by_standard_difficulty_count', 0)} 个", flush=True)
    return 0 if report.get("updated") or report.get("reason") == "no_failures" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
