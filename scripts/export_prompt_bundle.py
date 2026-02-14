#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出微调好的提示词为单一 JSON 包，便于分享给他人使用。

用法:
  python scripts/export_prompt_bundle.py --output prompt_bundle.json
  python scripts/export_prompt_bundle.py --examples processed_training_data/examples_deepseek-reasoner_0213_2.json --output prompt_bundle.json

导出的 JSON 包含：system 模板、规则、标准描述、few-shot 示例。
他人只需此 JSON + run_with_bundle.py + API Key 即可生成题目。
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import (
    build_system_prompt,
    build_examples_text,
    build_user_prompt,
    get_standard_description,
    get_targeted_rules,
    get_global_rules,
)
from data_processing.build_prompt import load_standard_descriptions
from data_processing.build_prompt import load_prompt_rules


def main():
    parser = argparse.ArgumentParser(description="导出提示词为单一 JSON 包")
    parser.add_argument("--examples", default=None, help="示例 JSON 路径，默认 processed_training_data/examples.json")
    parser.add_argument("--prompt-rules", default=None, help="规则 JSON 路径，默认 processed_training_data/prompt_rules.json")
    parser.add_argument("--output", default="prompt_bundle.json", help="输出 bundle JSON 路径")
    args = parser.parse_args()

    examples_path = args.examples or (PROJECT_ROOT / "processed_training_data" / "examples.json")
    rules_path = args.prompt_rules or (PROJECT_ROOT / "processed_training_data" / "prompt_rules.json")

    if args.prompt_rules:
        os.environ["PROMPT_RULES_PATH"] = str(Path(args.prompt_rules).resolve())

    examples = []
    if Path(examples_path).exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)

    standard_descriptions = load_standard_descriptions()
    prompt_rules = load_prompt_rules()
    system_prompt_base = build_system_prompt(include_think_chain=False)

    bundle = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "description": "K-12 ELA MCQ 生成提示词包，含 system 模板、规则、标准描述、few-shot 示例。配合 run_with_bundle.py 使用。",
        "system_prompt_base": system_prompt_base,
        "standard_descriptions": standard_descriptions,
        "prompt_rules": prompt_rules,
        "examples": examples,
        "user_prompt_reminder": "Before returning the JSON: verify (1) options A, B, C, and D all have different text—no duplicates; (2) the stem uses singular wording (e.g. \"Which choice...\" or \"Which option...\"), not \"Which choices/options\"; (3) do not say \"look at the picture\" or \"use the image\" unless you provide image_url.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"已导出到: {out_path}")
    print(f"  示例数: {len(examples)}")
    print(f"  标准描述数: {len(standard_descriptions)}")
    print(f"\n分享给他人时，提供此文件 + scripts/run_with_bundle.py，对方只需: pip install openai && 设置 API Key")


if __name__ == "__main__":
    main()
