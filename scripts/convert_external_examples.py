# -*- coding: utf-8 -*-
"""
将 data/external_examples_hierarchical.json 转换为项目内部格式，
按年级拆分输出到 processed_training_data/{grade}_ELA_examples.json。

每个 (standard, difficulty) 组合保留得分最高的 1 条作为种子示例。

用法:
    python scripts/convert_external_examples.py
    python scripts/convert_external_examples.py --top-k 2       # 每组合保留 2 条
    python scripts/convert_external_examples.py --dry-run        # 只统计不写文件
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import build_user_prompt, get_standard_description


INPUT_PATH = PROJECT_ROOT / "data" / "external_examples_hierarchical.json"
OUTPUT_DIR = PROJECT_ROOT / "processed_training_data"


def convert_answer_options(opts_list: list) -> dict:
    """[{key, text}, ...] → {A: text, B: text, ...}"""
    return {opt["key"]: opt["text"] for opt in opts_list}


def convert_one(item: dict, grade: str) -> dict:
    """将外部格式的单条示例转为项目内部 examples.json 格式。"""
    mr = item["model_response"]
    std = item["standard_id"]
    diff = item["difficulty"]
    subject = item.get("subject", "ELA").upper()
    desc = get_standard_description(std)

    user_prompt = build_user_prompt(
        grade=grade,
        standard=std,
        standard_description=desc,
        difficulty=diff,
        subject=subject,
    )

    mcq_json = {
        "id": mr.get("id", item.get("item_id", "")),
        "type": "mcq",
        "answer": mr["answer"],
        "question": mr["question"],
        "difficulty": diff,
        "answer_options": convert_answer_options(mr["answer_options"]),
        "answer_explanation": mr.get("answer_explanation", ""),
    }

    return {
        "user_prompt": user_prompt,
        "mcq_json": mcq_json,
        "difficulty": diff,
        "standard": std,
    }


def main():
    parser = argparse.ArgumentParser(description="转换外部高分示例为项目种子示例")
    parser.add_argument("--input", default=str(INPUT_PATH), help="输入文件路径")
    parser.add_argument("--top-k", type=int, default=1, help="每个 (standard, difficulty) 保留前 k 条最高分示例")
    parser.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    grades = sorted(data.keys(), key=lambda x: int(x))
    print(f"输入文件: {args.input}")
    print(f"年级: {grades}")
    print(f"每 (standard, difficulty) 保留: top-{args.top_k}")
    print()

    total_in = 0
    total_out = 0

    for grade in grades:
        by_key = defaultdict(list)

        for diff in ["easy", "medium", "hard"]:
            items = data[grade].get("ELA", {}).get("mcq", {}).get(diff, [])
            for item in items:
                total_in += 1
                std = item.get("standard_id", "")
                score = item.get("score", 0)
                by_key[(std, diff)].append((score, item))

        examples = []
        for (std, diff), scored_items in sorted(by_key.items()):
            scored_items.sort(key=lambda x: -x[0])
            for score, item in scored_items[: args.top_k]:
                examples.append(convert_one(item, grade))

        total_out += len(examples)
        out_path = OUTPUT_DIR / f"{grade}_ELA_examples.json"

        print(f"  Grade {grade:>2}: {sum(len(v) for v in by_key.values()):>4} 条输入 → "
              f"{len(by_key):>3} 组合 → {len(examples):>3} 条种子示例 → {out_path.name}")

        if not args.dry_run:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(examples, f, ensure_ascii=False, indent=2)

    print()
    print(f"总计: {total_in} 条输入 → {total_out} 条种子示例")
    if not args.dry_run:
        print(f"已写入 {OUTPUT_DIR}/{{grade}}_ELA_examples.json")
    else:
        print("(dry-run 模式，未写入文件)")


if __name__ == "__main__":
    main()
