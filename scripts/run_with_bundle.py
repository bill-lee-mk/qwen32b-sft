#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立运行脚本：仅需 prompt_bundle.json + openai，无需本仓库其他代码。

用法:
  pip install openai
  export DEEPSEEK_API_KEY=sk-xxx
  python scripts/run_with_bundle.py --bundle evaluation_output/prompt_bundle.json --standard CCSS.ELA-LITERACY.L.3.1.E --difficulty medium --model deepseek-reasoner

  # 生成全部 237 题
  python scripts/run_with_bundle.py --bundle prompt_bundle.json --all --model deepseek-reasoner --output mcqs.json
"""
import argparse
import json
import os
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("请安装: pip install openai")
    exit(1)

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "KIMI_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
}


def _provider(model: str):
    m = (model or "").lower()
    if "deepseek" in m:
        return "deepseek"
    if "kimi" in m or "moonshot" in m:
        return "kimi"
    return "openai"


def _call_api(messages, model: str, api_key: str, base_url: str | None):
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    kwargs = {"model": model, "messages": messages}
    if "reasoner" in model.lower():
        kwargs["max_tokens"] = 8192
        kwargs["temperature"] = 0.7
    elif "k2" in model.lower():
        kwargs["max_tokens"] = 8192
        kwargs["temperature"] = 1.0
    else:
        kwargs["max_tokens"] = 8192
        kwargs["temperature"] = 0.7
    return client.chat.completions.create(**kwargs)


def _parse_mcq(text: str) -> dict | None:
    s = (text or "").strip()
    for m in ("<think>", "```json", "```"):
        if m in s:
            if s.startswith("<think>") and "</think>" in s:
                s = s[s.index("</think>") + 7 :].strip()
            for part in s.split("```"):
                if "{" in part:
                    start = part.find("{")
                    depth = 0
                    for i in range(start, len(part)):
                        if part[i] == "{":
                            depth += 1
                        elif part[i] == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    return json.loads(part[start : i + 1])
                                except json.JSONDecodeError:
                                    pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="prompt_bundle.json 路径")
    p.add_argument("--standard", default=None)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--model", default="deepseek-reasoner")
    p.add_argument("--api-key", default=None)
    p.add_argument("--output", default="mcqs.json")
    p.add_argument("--all", action="store_true", help="生成全部 237 题")
    args = p.parse_args()

    with open(args.bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    prompts = bundle.get("prompts", {})
    plan = bundle.get("plan", [])

    prov = _provider(args.model)
    base_url, env_key = PROVIDERS[prov]
    api_key = args.api_key or os.environ.get(env_key)
    if not api_key:
        print(f"错误: 请设置 {env_key} 或 --api-key")
        exit(1)

    if args.all:
        results = []
        for i, (std, diff) in enumerate(plan):
            key = f"{std}|{diff}"
            if key not in prompts:
                print(f"  跳过 {key} (无 prompt)")
                continue
            sys, usr = prompts[key]["system"], prompts[key]["user"]
            messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
            try:
                r = _call_api(messages, args.model, api_key, base_url)
                raw = (r.choices[0].message.content or "").strip()
                mcq = _parse_mcq(raw)
                if mcq:
                    mcq["grade"] = "3"
                    mcq["standard"] = std
                    mcq["subject"] = "ELA"
                    mcq["difficulty"] = diff
                    mcq["id"] = f"diverse_{i:03d}"
                    results.append(mcq)
                    print(f"  [{i+1}/237] {std} {diff} OK")
                else:
                    print(f"  [{i+1}/237] {std} {diff} 解析失败")
            except Exception as e:
                print(f"  [{i+1}/237] {std} {diff} 错误: {e}")
            time.sleep(0.3)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存 {len(results)} 题到 {args.output}")
    else:
        if not args.standard:
            print("错误: 单题模式需 --standard")
            exit(1)
        key = f"{args.standard}|{args.difficulty}"
        if key not in prompts:
            print(f"错误: 无此组合的 prompt: {key}")
            exit(1)
        sys, usr = prompts[key]["system"], prompts[key]["user"]
        messages = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        r = _call_api(messages, args.model, api_key, base_url)
        raw = (r.choices[0].message.content or "").strip()
        mcq = _parse_mcq(raw)
        if mcq:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(mcq, f, ensure_ascii=False, indent=2)
            print(f"已保存到 {args.output}")
        else:
            print("解析失败，原始输出:")
            print(raw[:500])


if __name__ == "__main__":
    main()
