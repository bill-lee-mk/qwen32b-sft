#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 MCQ（generate_mcq）

用闭源 API（Gemini/OpenAI/DeepSeek）生成 MCQ。

用法:
  # 1. 先筛选示例
  python main.py select-examples -n 5

  # 2. 生成 MCQ
  python scripts/generate_mcq.py --provider gemini --output evaluation_output/mcqs.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import build_full_prompt
from data_processing.select_examples import extract_json_from_text, is_valid_mcq
from evaluation.inceptbench_client import normalize_for_inceptbench


def call_gemini(prompt: str, api_key: str, model: str = "gemini-3-flash-preview") -> str:
    """调用 Gemini API"""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("请安装: pip install google-generativeai")
    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)
    return response.text


def call_openai(messages: list, api_key: str, model: str = "gpt-4o", base_url: str | None = None) -> str:
    """调用 OpenAI 兼容 API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装: pip install openai")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def call_deepseek(messages: list, api_key: str, model: str = "deepseek-chat") -> str:
    """调用 DeepSeek API"""
    return call_openai(messages, api_key, model, base_url="https://api.deepseek.com")


def parse_mcq(text: str) -> dict | None:
    """从模型回复中解析 MCQ JSON"""
    obj = extract_json_from_text(text)
    if not obj:
        return None
    ok, _ = is_valid_mcq(obj)
    return obj if ok else None


def main():
    parser = argparse.ArgumentParser(description="生成 MCQ（generate_mcq）")
    parser.add_argument("--provider", choices=["gemini", "openai", "deepseek"], required=True, help="API 提供商")
    parser.add_argument("--api-key", default=None, help="API Key（环境变量: GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--model", default=None, help="模型名")
    parser.add_argument("--examples", default=None, help="示例 JSON 路径，默认 processed_training_data/examples.json")
    parser.add_argument("--grade", default="3", help="年级")
    parser.add_argument("--standard", default="CCSS.ELA-LITERACY.L.3.1.E", help="标准 ID")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--batch", type=int, default=1, help="生成条数")
    parser.add_argument("--include-think-chain", action="store_true", help="是否要求输出 <think>")
    args = parser.parse_args()

    # API Key 与默认模型
    if args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        model = args.model or "gemini-3-flash-preview"
    elif args.provider == "deepseek":
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        model = args.model or "deepseek-chat"
    else:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        model = args.model or "gpt-4o"
    if not api_key:
        print("错误: 请提供 --api-key 或设置环境变量")
        sys.exit(1)

    # 加载示例（默认 examples.json，兼容旧版 few_shot_examples.json）
    examples_path = args.examples or (PROJECT_ROOT / "processed_training_data" / "examples.json")
    if not Path(examples_path).exists():
        fallback = PROJECT_ROOT / "processed_training_data" / "few_shot_examples.json"
        if fallback.exists():
            examples_path = fallback
            print(f"使用兼容路径: {examples_path}")
    if not Path(examples_path).exists():
        print(f"警告: 示例文件不存在，将使用零样本")
        examples = []
    else:
        with open(examples_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        print(f"已加载 {len(examples)} 条示例")

    # 构建 prompt
    system, user = build_full_prompt(
        grade=args.grade,
        standard=args.standard,
        difficulty=args.difficulty,
        examples=examples,
        include_think_chain=args.include_think_chain,
    )

    results = []
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for i in range(args.batch):
        if args.provider == "gemini":
            full_prompt = f"{system}\n\n{user}"
            raw = call_gemini(full_prompt, api_key, model)
        elif args.provider == "deepseek":
            raw = call_deepseek(messages, api_key, model)
        else:
            raw = call_openai(messages, api_key, model)

        mcq = parse_mcq(raw)
        if mcq:
            normalized = normalize_for_inceptbench(mcq)
            normalized["grade"] = args.grade
            normalized["standard"] = args.standard
            normalized["subject"] = "ELA"
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
