#!/usr/bin/env python3
"""将 raw_data/inceptbench_highscore_grade{N}_ela.json 转为管线格式并更新种子文件。

转换格式：
  raw_data:  {id, type, question, answer, answer_explanation, difficulty, standard, substandard_description, score, answer_options, ...}
  pipeline:  {user_prompt, mcq_json, standard, substandard_description, difficulty, type, score}

合并策略（与 enrich_examples_from_db.py 一致）：
  - 对每个 (standard, difficulty, type) 组合
  - 若新分数 > 旧分数 → 替换
  - 若旧无此组合 → 填补（score >= 0.85）
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import build_user_prompt, get_standard_description

RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
SEED_DIR = PROJECT_ROOT / "processed_training_data"
MIN_SCORE = 0.85
MAX_PER_COMBO = 2


def _raw_to_mcq_json(item: dict) -> dict:
    """将 raw_data 格式转为 mcq_json 字段"""
    qtype = item.get("type", "mcq")
    mcq = {
        "type": qtype,
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "answer_explanation": item.get("answer_explanation", ""),
        "difficulty": item.get("difficulty", "medium"),
    }
    if qtype in ("mcq", "msq"):
        opts = item.get("answer_options", [])
        if isinstance(opts, dict):
            opts = [{"key": k, "text": v} for k, v in sorted(opts.items())]
        mcq["answer_options"] = opts
    elif qtype == "fill-in":
        mcq["acceptable_answers"] = item.get("acceptable_answers", [])
    if item.get("context"):
        mcq["context"] = item["context"]
    return mcq


def _raw_to_pipeline(item: dict, grade: str) -> dict:
    """将单条 raw_data 转换为管线 seed 格式"""
    standard = item.get("standard", "")
    difficulty = item.get("difficulty", "medium")
    qtype = item.get("type", "mcq")
    desc = item.get("substandard_description") or get_standard_description(standard) or ""

    user_prompt = build_user_prompt(
        grade=grade,
        standard=standard,
        standard_description=desc,
        difficulty=difficulty,
        subject="ELA",
        question_type=qtype,
    )
    mcq_json = _raw_to_mcq_json(item)

    return {
        "user_prompt": user_prompt,
        "mcq_json": mcq_json,
        "standard": standard,
        "substandard_description": desc,
        "difficulty": difficulty,
        "type": qtype,
        "score": item.get("score", 0),
    }


def _build_index(items: list) -> dict:
    """建立 (standard, difficulty, type) → [(score, index), ...] 的索引，按分数降序"""
    idx = {}
    for i, item in enumerate(items):
        key = (item.get("standard", ""), item.get("difficulty", ""), item.get("type", "mcq"))
        score = item.get("score", 0) or 0
        idx.setdefault(key, []).append((score, i))
    for key in idx:
        idx[key].sort(key=lambda x: -x[0])
    return idx


def process_grade(grade: int) -> tuple:
    """处理单个年级，每组合保留 top MAX_PER_COMBO 条，返回 (final_count, replaced, filled, skipped)"""
    grade_str = str(grade)
    raw_path = RAW_DATA_DIR / f"inceptbench_highscore_grade{grade}_ela.json"
    seed_path = SEED_DIR / f"{grade}_ELA_examples.json"

    if not raw_path.exists():
        return (0, 0, 0, 0)

    raw_data = json.load(open(raw_path, encoding="utf-8"))

    raw_by_combo = {}
    for item in raw_data:
        key = (item.get("standard", ""), item.get("difficulty", ""), item.get("type", "mcq"))
        score = item.get("score", 0) or 0
        if score < MIN_SCORE:
            continue
        raw_by_combo.setdefault(key, []).append(item)
    for key in raw_by_combo:
        raw_by_combo[key].sort(key=lambda x: -(x.get("score", 0) or 0))
        raw_by_combo[key] = raw_by_combo[key][:MAX_PER_COMBO]

    if seed_path.exists():
        seed_data = json.load(open(seed_path, encoding="utf-8"))
    else:
        seed_data = []

    seed_idx = _build_index(seed_data)

    replaced = 0
    filled = 0
    skipped = 0

    for key, raw_items in raw_by_combo.items():
        cur_entries = seed_idx.get(key, [])

        for rank, raw_item in enumerate(raw_items):
            raw_score = raw_item.get("score", 0) or 0
            new_entry = _raw_to_pipeline(raw_item, grade_str)

            if rank < len(cur_entries):
                cur_score, cur_i = cur_entries[rank]
                if raw_score > cur_score:
                    seed_data[cur_i] = new_entry
                    cur_entries[rank] = (raw_score, cur_i)
                    replaced += 1
            else:
                seed_data.append(new_entry)
                cur_entries.append((raw_score, len(seed_data) - 1))
                filled += 1

        seed_idx[key] = cur_entries

    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump(seed_data, f, ensure_ascii=False, indent=2)

    return (len(seed_data), replaced, filled, skipped)


def main():
    print(f"{'Grade':>5} | {'原种子':>6} | {'替换':>4} | {'填补':>4} | {'最终':>5}")
    print("-" * 45)

    total_replaced = 0
    total_filled = 0

    for grade in range(1, 13):
        seed_path = SEED_DIR / f"{grade}_ELA_examples.json"
        orig = len(json.load(open(seed_path))) if seed_path.exists() else 0
        final, replaced, filled, skipped = process_grade(grade)
        total_replaced += replaced
        total_filled += filled
        print(f"{grade:>5} | {orig:>6} | {replaced:>4} | {filled:>4} | {final:>5}")

    print("-" * 45)
    print(f"总计: 替换 {total_replaced} 条, 填补 {total_filled} 条")
    print("\n完成! 种子文件已更新到 processed_training_data/{N}_ELA_examples.json")


if __name__ == "__main__":
    main()
