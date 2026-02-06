#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Few-shot MCQ 生成脚本（闭源 API 模型）

用法:
  # 1. 先筛选 few-shot 样本
  python -m data_processing.few_shot_selector --input-dir raw_data -n 5

  # 2. 使用 Gemini 3 生成（默认 gemini-3-flash-preview，免费额度内可用）
  python scripts/few_shot_generate.py --provider gemini

  # 3. 使用 Gemini 3 Pro（更强推理能力）
  python scripts/few_shot_generate.py --provider gemini --model gemini-3-pro-preview

  # 4. 使用 OpenAI 生成
  python scripts/few_shot_generate.py --provider openai --api-key $OPENAI_API_KEY

  # 5. 使用 DeepSeek 生成（按量计费，无免费额度）
  python scripts/few_shot_generate.py --provider deepseek --api-key $DEEPSEEK_API_KEY

  # 6. 指定输出用于 InceptBench 评估
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
    """调用 OpenAI 兼容 API（OpenAI、DeepSeek 等）"""
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
    """调用 DeepSeek API（OpenAI 兼容接口）"""
    return call_openai(
        messages=messages,
        api_key=api_key,
        model=model,
        base_url="https://api.deepseek.com",
    )


def parse_mcq_from_response(text: str) -> dict | None:
    """从模型回复中解析 MCQ JSON"""
    obj = extract_json_from_text(text)
    if not obj:
        return None
    ok, _ = is_valid_mcq(obj)
    return obj if ok else None


def main():
    parser = argparse.ArgumentParser(description="Few-shot MCQ 生成（闭源 API）")
    parser.add_argument("--provider", choices=["gemini", "openai", "deepseek"], required=True, help="API 提供商")
    parser.add_argument("--api-key", default=None, help="API Key（环境变量: GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY）")
    parser.add_argument("--model", default=None, help="模型名，Gemini 默认 gemini-3-flash-preview，OpenAI 默认 gpt-4o，DeepSeek 默认 deepseek-chat")
    parser.add_argument("--few-shot-path", default=None, help="few-shot 样本 JSON 路径")
    parser.add_argument("--grade", default="3", help="年级")
    parser.add_argument("--standard", default="CCSS.ELA-LITERACY.L.3.1.E", help="标准 ID")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--output", default=None, help="输出 JSON 路径（单条 MCQ）")
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
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for i in range(args.batch):
        if args.provider == "gemini":
            full_prompt = f"{system}\n\n{user}"
            raw = call_gemini(full_prompt, api_key, model)
        elif args.provider == "deepseek":
            raw = call_deepseek(messages, api_key, model)
        else:
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
