#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出独立提示词包：预构建 237 个 (standard, difficulty) 的完整 prompt，供他人直接使用。

用法: python scripts/export_standalone_prompt.py [--output prompt_bundle.json]

输出: 单一 JSON 文件，含所有预构建的 system+user prompt。他人用 run_with_bundle.py 即可生成，无需本仓库。
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import build_full_prompt
from data_processing.analyze_dimensions import analyze_dimensions, build_diverse_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation_output/prompt_bundle.json")
    parser.add_argument("--examples", default=None)
    parser.add_argument("--prompt-rules", default=None)
    args = parser.parse_args()

    if args.prompt_rules:
        os.environ["PROMPT_RULES_PATH"] = str(Path(args.prompt_rules).resolve())

    examples_path = args.examples or (PROJECT_ROOT / "processed_training_data" / "examples.json")
    examples = []
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

    dims = analyze_dimensions(str(PROJECT_ROOT / "raw_data"))
    plan = build_diverse_plan(dims, n=9999, all_combinations=True)
    if len(plan) < 200:
        # 无 raw_data 时从 standard_descriptions 构建 79×3
        from data_processing.build_prompt import load_standard_descriptions
        stds = list(load_standard_descriptions().keys())
        plan = [(s, d) for s in stds for d in ("easy", "medium", "hard")]

    prompts = {}
    for standard, difficulty in plan:
        filtered = [e for e in examples if e.get("standard") == standard and e.get("difficulty") == difficulty]
        system, user = build_full_prompt(grade="3", standard=standard, difficulty=difficulty, examples=filtered, subject="ELA")
        key = f"{standard}|{difficulty}"
        prompts[key] = {"system": system, "user": user}

    bundle = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "description": "K-12 ELA MCQ 独立提示词包。含 237 个 (standard,difficulty) 的完整 system+user prompt。配合 run_with_bundle.py 使用，无需本仓库。",
        "prompts": prompts,
        "plan": plan,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"已导出到: {out} ({len(prompts)} 个 prompt)")
    print("分享: 此文件 + scripts/run_with_bundle.py，对方 pip install openai 后即可用")


if __name__ == "__main__":
    main()
