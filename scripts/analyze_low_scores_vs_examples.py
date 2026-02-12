#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析「打分低于 0.85」的原因：是否因 examples.json 中缺少该 (standard, difficulty) 的高分（≥0.85）示例。

不修改题目文件；仅基于评估结果与 examples 做对比分析。
"""
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_overall_score(result: dict) -> Optional[float]:
    """从单条评估结果中取出 overall_score"""
    if not result:
        return None
    s = result.get("overall_score")
    if s is not None and isinstance(s, (int, float)):
        return float(s)
    evals = result.get("evaluations") or {}
    for ev in evals.values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        overall = inc.get("overall") or {}
        s = overall.get("score")
        if s is not None:
            return float(s)
    return None


def main():
    results_path = PROJECT_ROOT / "evaluation_output/results_237.json"
    mcqs_path = PROJECT_ROOT / "evaluation_output/mcqs_237.json"
    examples_path = PROJECT_ROOT / "processed_training_data/examples.json"

    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)
    if isinstance(mcqs, dict):
        mcqs = [mcqs]
    with open(examples_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    # 所有在 examples 中出现的 (standard, difficulty) → 视为「有 ≥0.85 示例」（improve_examples 只保留达标项）
    example_pairs = set()
    for e in examples:
        std = e.get("standard") or ""
        diff = e.get("difficulty") or "medium"
        example_pairs.add((std, diff))

    # 从评估结果中取出：每道题的 index → (score, standard, difficulty, id)
    result_list = results_data.get("results") or []
    low_score_items = []   # [(index, score, standard, difficulty, id), ...]
    failed_pairs = set()   # (standard, difficulty) 出现过低分或 error 的组合

    for i in range(len(mcqs)):
        m = mcqs[i]
        std = m.get("standard", "unknown")
        diff = m.get("difficulty", "medium")
        qid = m.get("id", f"index_{i}")
        if i < len(result_list):
            r = result_list[i]
            score = extract_overall_score(r)
        else:
            score = None
        if score is None or score < 0.85:
            low_score_items.append((i + 1, score, std, diff, qid))
            failed_pairs.add((std, diff))

    # 分类：该组合是否有高分示例
    pairs_with_example = failed_pairs & example_pairs
    pairs_without_example = failed_pairs - example_pairs

    # 输出报告
    print("=" * 60)
    print("打分低于 0.85 的原因分析：是否缺少 examples 高分示例")
    print("=" * 60)
    print(f"\n低分/Error 题目数: {len(low_score_items)}")
    print(f"涉及 (standard, difficulty) 组合数: {len(failed_pairs)}")
    print(f"  - 在 examples.json 中【有】该组合的高分示例: {len(pairs_with_example)} 个")
    print(f"  - 在 examples.json 中【无】该组合的高分示例: {len(pairs_without_example)} 个")

    print("\n--- 无高分示例的组合（缺示例很可能是主因）---")
    for (std, diff) in sorted(pairs_without_example):
        short = std.replace("CCSS.ELA-LITERACY.", "") if std else "?"
        print(f"  ({short}, {diff})")

    print("\n--- 有高分示例但仍出现低分的组合（原因可能在别处：模型未学好/题目难度/单例不足等）---")
    for (std, diff) in sorted(pairs_with_example):
        short = std.replace("CCSS.ELA-LITERACY.", "") if std else "?"
        print(f"  ({short}, {diff})")

    print("\n--- 低分/Error 题目明细（题号、分数、standard、difficulty、id）---")
    for idx, score, std, diff, qid in sorted(low_score_items, key=lambda x: (x[2], x[3], x[0])):
        s = f"{score:.2f}" if score is not None else "error"
        short = std.replace("CCSS.ELA-LITERACY.", "") if std else "?"
        has_ex = "有示例" if (std, diff) in example_pairs else "无示例"
        print(f"  题{idx}: score={s} ({short}, {diff}) id={qid} [{has_ex}]")

    print("\n=== 结论 ===")
    if pairs_without_example:
        print(f"有 {len(pairs_without_example)} 个 (standard, difficulty) 在 examples 中没有任何高分示例，")
        print("这些组合下生成题得分低，很可能是「缺 ≥0.85 示例」导致。建议：对这类组合运行 improve-examples 从 raw_data 中筛出达标示例并入 examples。")
    if pairs_with_example:
        print(f"有 {len(pairs_with_example)} 个组合在 examples 中已有高分示例但仍出现低分，")
        print("原因可能包括：示例数量少、模型未充分模仿、或题目本身难度/表述问题；可考虑增加该组合的示例条数或检查生成 prompt。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
