#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot MCQ 生成脚本（闭源 API 模型）

用法:
  # 1. 先筛选 few-shot 样本
  python -m data_processing.few_shot_selector --input-dir raw_data -n 5

  # 2. 使用 Gemini 生成
  python scripts/few_shot_generate.py --provider gemini --api-key $GEMINI_API_KEY

  # 3. 使用 OpenAI 生成
  python scripts/few_shot_generate.py --provider openai --api-key $OPENAI_API_KEY

  # 4. 指定输出用于 InceptBench 评估
  python scripts/few_shot_generate.py --provider gemini --output evaluation_output/mcqs.json
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.few_shot_prompt import build_full_prompt
from data_processing.few_shot_selector import extract_json_from_text, is_valid_mcq
from evaluation.inceptbench_client import normalize_for_inceptbench


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    """调用 Gemini API"""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("请安装: pip install google-generativeai")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    return response.text


def call_openai(messages: list, api_key: str, model: str = "gpt-4o") -> str:
    """调用 OpenAI API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装: pip install openai")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def parse_mcq_from_response(text: str) -> dict | None:
    """从模型回复中解析 MCQ JSON"""
    obj = extract_json_from_text(text)
    if not obj:
        return None
    ok, _ = is_valid_mcq(obj)
    return obj if ok else None


def main():
    parser = argparse.ArgumentParser(description="Few-shot MCQ 生成（闭源 API）")
    parser.add_argument("--provider", choices=["gemini", "openai"], required=True, help="API 提供商")
    parser.add_argument("--api-key", default=None, help="API Key（也可用环境变量 GEMINI_API_KEY / OPENAI_API_KEY）")
    parser.add_argument("--model", default=None, help="模型名，默认 gemini-2.0-flash 或 gpt-4o")
    parser.add_argument("--few-shot-path", default=None, help="few-shot 样本 JSON 路径")
    parser.add_argument("--grade", default="3", help="年级")
    parser.add_argument("--standard", default="CCSS.ELA-LITERACY.L.3.1.E", help="标准 ID")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--output", default=None, help="输出 JSON 路径（单条 MCQ）")
    parser.add_argument("--batch", type=int, default=1, help="生成条数")
    parser.add_argument("--include-think-chain", action="store_true", help="是否要求输出 <think>")
    args = parser.parse_args()

    # API Key
    if args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        model = args.model or "gemini-2.0-flash"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        model = args.model or "gpt-4o"
    if not api_key:
        print("错误: 请提供 --api-key 或设置环境变量")
        sys.exit(1)

    # 加载 few-shot
    few_shot_path = args.few_shot_path or (PROJECT_ROOT / "processed_training_data" / "few_shot_examples.json")
    if not Path(few_shot_path).exists():
        print(f"警告: few-shot 文件不存在 {few_shot_path}，将使用零样本")
        few_shot_samples = []
    else:
        with open(few_shot_path, "r", encoding="utf-8") as f:
            few_shot_samples = json.load(f)
        print(f"已加载 {len(few_shot_samples)} 条 few-shot 样本")

    # 构建 prompt
    system, user = build_full_prompt(
        grade=args.grade,
        standard=args.standard,
        difficulty=args.difficulty,
        few_shot_samples=few_shot_samples,
        include_think_chain=args.include_think_chain,
    )

    results = []
    for i in range(args.batch):
        if args.provider == "gemini":
            full_prompt = f"{system}\n\n{user}"
            raw = call_gemini(full_prompt, api_key, model)
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            raw = call_openai(messages, api_key, model)

        mcq = parse_mcq_from_response(raw)
        if mcq:
            normalized = normalize_for_inceptbench(mcq)
            results.append(normalized)
            print(f"生成 {i+1}/{args.batch}: {normalized.get('id', 'unknown')}")
        else:
            print(f"生成 {i+1}/{args.batch}: 解析失败")
            print(f"  raw: {raw[:200]}...")

    if not results:
        print("未成功生成任何 MCQ")
        sys.exit(1)

    # 输出
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results[0], f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {out_path}")
    else:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
