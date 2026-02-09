#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
闭环：从评估结果中识别失败题 → 在 raw_data 中找同 (standard,difficulty) → InceptBench 评分 → 保留 ≥0.85 作 few-shot → 更新 examples.json

用法:
  python main.py improve-examples --results evaluation_output/results_240.json --mcqs evaluation_output/mcqs_240.json --output processed_training_data/examples.json
"""
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.select_examples import (
    load_jsonl,
    process_dpo_file,
    process_messages_file,
)
from evaluation.inceptbench_client import InceptBenchEvaluator, normalize_for_inceptbench


def load_raw_data_by_standard_difficulty(raw_data_dir: str) -> dict:
    """从 raw_data 加载 MCQ，按 (standard, difficulty) 分组"""
    raw_dir = Path(raw_data_dir)
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    if not raw_dir.exists():
        return {}

    by_key = defaultdict(list)
    for fpath in sorted(raw_dir.glob("*.jsonl")):
        samples = load_jsonl(str(fpath))
        if not samples:
            continue
        if "messages" in samples[0]:
            candidates = process_messages_file(samples)
        elif "prompt" in samples[0] and "chosen" in samples[0]:
            candidates = process_dpo_file(samples)
        else:
            continue
        for c in candidates:
            key = (c["standard"], c["difficulty"])
            by_key[key].append(c)
    return dict(by_key)


def extract_failed_standard_difficulty(
    results_path: str,
    mcqs_path: str,
    threshold: float = 0.85,
) -> set[tuple[str, str]]:
    """从评估结果中提取得分 < threshold 的 (standard, difficulty)"""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)

    scores = results.get("scores", [])
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    failed = set()
    for i, s in enumerate(scores):
        if i < len(items) and isinstance(s, (int, float)) and float(s) < threshold:
            m = items[i]
            std = m.get("standard", "unknown")
            diff = m.get("difficulty", "medium")
            failed.add((std, diff))
    return failed


def run(
    results_path: str,
    mcqs_path: str,
    raw_data_dir: str = "raw_data",
    examples_output: str = "processed_training_data/examples.json",
    threshold: float = 0.85,
    max_per_pair: int = 1,
    max_candidates_per_pair: int = 5,
    parallel: int = 20,
    timeout: int = 180,
    api_key: str | None = None,
) -> dict:
    """
    闭环：失败题 (standard,difficulty) → raw_data 候选 → InceptBench 评分 → 保留 ≥0.85 的 1-2 条 → 更新 examples
    """
    import os
    api_key = api_key or os.environ.get("INCEPTBENCH_API_KEY") or os.environ.get("INCEPTBENCH_TOKEN")
    if not api_key:
        return {"error": "未设置 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN"}

    failed = extract_failed_standard_difficulty(results_path, mcqs_path, threshold)
    print(f"失败 (standard,difficulty) 组合: {len(failed)} 个")

    by_key = load_raw_data_by_standard_difficulty(raw_data_dir)
    print(f"raw_data 中 (standard,difficulty) 组合: {len(by_key)} 个")

    # 仅对失败组合的候选做 InceptBench 评分
    to_evaluate = []
    max_cand = max_candidates_per_pair  # 避免闭包问题
    for key in failed:
        if key[0] == "unknown":
            continue
        for c in by_key.get(key, [])[:max_cand]:  # 每组合最多试 N 条
            mcq = c.get("mcq_json", {})
            if not mcq:
                continue
            norm = normalize_for_inceptbench(mcq)
            norm["grade"] = "3"
            norm["standard"] = key[0]
            norm["subject"] = "ELA"
            norm["difficulty"] = key[1]
            to_evaluate.append((key, c, norm))

    print(f"待评分候选: {len(to_evaluate)} 条")

    evaluator = InceptBenchEvaluator(api_key=api_key, timeout=timeout)
    scored: list[tuple[tuple[str, str], dict, float]] = []

    total = len(to_evaluate)
    print(f"  [{parallel} 并行] 开始评分...")

    if parallel <= 1:
        for idx, (key, c, norm) in enumerate(to_evaluate):
            r = evaluator.evaluate_mcq(norm)
            s = r.get("overall_score")
            if s is None and "evaluations" in r:
                ev = next(iter(r.get("evaluations", {}).values()), {})
                s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
            if isinstance(s, (int, float)):
                scored.append((key, c, float(s)))
            print(f"  [{idx + 1}/{total}] 题{idx + 1}: score={s:.2f}" if isinstance(s, (int, float)) else f"  [{idx + 1}/{total}] 题{idx + 1}: error")
    else:
        import threading
        scored_lock = threading.Lock()
        done_count = [0]  # 用 list 以便闭包修改

        def _eval(item_with_idx):
            idx, (key, c, norm) = item_with_idx
            try:
                r = evaluator.evaluate_mcq(norm)
                s = r.get("overall_score")
                if s is None and "evaluations" in r:
                    ev = next(iter(r.get("evaluations", {}).values()), {})
                    s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
                return (idx, key, c, float(s) if isinstance(s, (int, float)) else None)
            except Exception as e:
                return (idx, key, c, None)

        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_eval, (i, x)): i for i, x in enumerate(to_evaluate)}
            results_by_idx = {}
            for fut in as_completed(futures):
                i, key, c, s = fut.result()
                results_by_idx[i] = (key, c, s)
                with scored_lock:
                    done_count[0] += 1
                    d = done_count[0]
                    status = f"score={s:.2f}" if s is not None else "error"
                    print(f"  [{d}/{total}] 题{i + 1}: {status}")

        for i in range(total):
            if i in results_by_idx:
                key, c, s = results_by_idx[i]
                if s is not None:
                    scored.append((key, c, s))

    # 每个 (standard,difficulty) 保留 1-2 条 score >= threshold
    kept_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for key, c, s in scored:
        if s >= threshold:
            kept_by_key[key].append((c, s))
    for key in kept_by_key:
        kept_by_key[key].sort(key=lambda x: -x[1])
        kept_by_key[key] = [c for c, _ in kept_by_key[key][:max_per_pair]]

    print(f"达标 (≥{threshold}) 的示例: {sum(len(v) for v in kept_by_key.values())} 条")

    # 加载现有 examples，用新达标项替换失败组合的示例
    examples_path = Path(examples_output)
    if not examples_path.is_absolute():
        examples_path = PROJECT_ROOT / examples_path
    existing = []
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 新示例：每个 (standard,difficulty) 1-2 条
    new_examples = []
    for key, items in kept_by_key.items():
        for item in items:
            new_examples.append({
                "user_prompt": item["user_prompt"],
                "mcq_json": item["mcq_json"],
                "difficulty": key[1],
                "standard": key[0],
            })

    # 保留原有示例中 (standard,difficulty) 不在 kept_by_key 中的
    replaced_keys = set(kept_by_key.keys())
    for e in existing:
        k = (e.get("standard"), e.get("difficulty"))
        if k not in replaced_keys:
            new_examples.append(e)

    unique = new_examples

    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已更新 examples: {examples_path} ({len(unique)} 条)")

    return {
        "failed_pairs": len(failed),
        "evaluated": len(to_evaluate),
        "kept": sum(len(v) for v in kept_by_key.values()),
        "examples_count": len(unique),
    }
