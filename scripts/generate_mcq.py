#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 MCQ（generate_mcq）

用闭源 API（Gemini/OpenAI/DeepSeek）生成 MCQ。

用法:
  # 1. 先筛选示例
  python main.py select-examples -n 5

  # 2. 生成 MCQ（单条）
  python scripts/generate_mcq.py --provider gemini --output evaluation_output/mcqs.json

  # 3. 多样化批量生成（20 条，覆盖不同难度/标准，多线程）
  python scripts/generate_mcq.py --provider gemini --diverse 20 --output evaluation_output/mcqs.json

  # 4. 使用 Kimi (Moonshot) 生成（需配置 KIMI_API_KEY，新用户有免费额度）
  python scripts/generate_mcq.py --provider kimi --output evaluation_output/mcqs.json
  python scripts/generate_mcq.py --provider kimi --all-combinations --output evaluation_output/mcqs_240.json
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.analyze_dimensions import analyze_dimensions, build_diverse_plan
from data_processing.build_prompt import build_full_prompt
from data_processing.select_examples import extract_json_from_text, is_valid_mcq
from evaluation.inceptbench_client import normalize_for_inceptbench
from scripts.validate_mcq import validate_and_fix


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


def _parse_retry_seconds(err_msg: str) -> int:
    """从 429 错误信息解析建议等待秒数"""
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", err_msg, re.I)
    if m:
        return min(int(float(m.group(1))) + 5, 120)  # 多等 5s，最多 120s
    return 60


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


# Kimi (Moonshot) 官方 API：https://platform.moonshot.cn ，OpenAI 兼容，需配置 API Key；新用户有免费额度
# 可选模型：kimi-latest（推荐）、moonshot-v1-8k、kimi-k2-0905-preview、kimi-k2-turbo-preview 等
# 用量规则（以控制台为准）：并发数=3（同时请求数）、RPM=20（每分钟请求数）、TPM=500000（每分钟 token）、TPD=500000（每日 token）
# 本仓库：Kimi 默认 --workers 3，不超过并发；若触发限频可加 --workers 2 或 1
KIMI_API_BASE = "https://api.moonshot.cn/v1"
KIMI_MAX_CONCURRENCY = 3  # 与平台「并发数: 3」一致


def call_kimi(messages: list, api_key: str, model: str = "kimi-latest") -> str:
    """调用 Kimi (Moonshot) API，OpenAI 兼容接口"""
    return call_openai(messages, api_key, model, base_url=KIMI_API_BASE)


def parse_mcq(text: str) -> dict | None:
    """从模型回复中解析 MCQ JSON"""
    obj = extract_json_from_text(text)
    if not obj:
        return None
    ok, _ = is_valid_mcq(obj)
    return obj if ok else None


def _filter_examples_for_standard_difficulty(examples: list, standard: str, difficulty: str) -> list:
    """仅保留与当前 (standard, difficulty) 匹配的示例，避免 prompt 超出 64K 上下文"""
    return [e for e in examples if e.get("standard") == standard and e.get("difficulty") == difficulty]


def _validate_and_keep_passing(results: list) -> tuple[list, dict]:
    """
    对每条生成题做校验，不通过则尝试自动修复；只保留最终通过校验的题目，并重新编号 id。
    返回 (通过校验的题目列表, {"fixed": 修复后通过数, "dropped": 仍不通过被丢弃数})
    """
    from scripts.validate_mcq import validate_mcq

    passing = []
    fixed_count = 0
    dropped_count = 0
    for idx, mcq in enumerate(results):
        had_issues_before = len(validate_mcq(mcq, idx)) > 0
        mcq_out, passed = validate_and_fix(mcq, index=idx, max_rounds=2)
        if passed:
            if had_issues_before:
                fixed_count += 1
            passing.append(mcq_out)
        else:
            dropped_count += 1
    for i, m in enumerate(passing):
        m["id"] = f"diverse_{i:03d}"
    return passing, {"fixed": fixed_count, "dropped": dropped_count}


def _generate_one(
    standard: str,
    difficulty: str,
    examples: list,
    provider: str,
    api_key: str,
    model: str,
    grade: str = "3",
    index: int = 0,
    max_retries: int = 3,
) -> dict | None:
    """生成单条 MCQ，供多线程调用。遇 429 自动重试。"""
    filtered = _filter_examples_for_standard_difficulty(examples, standard, difficulty)
    system, user = build_full_prompt(
        grade=grade,
        standard=standard,
        difficulty=difficulty,
        examples=filtered,
        subject="ELA",
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(max_retries):
        try:
            if provider == "gemini":
                raw = call_gemini(f"{system}\n\n{user}", api_key, model)
            elif provider == "deepseek":
                raw = call_deepseek(messages, api_key, model)
            elif provider == "kimi":
                raw = call_kimi(messages, api_key, model)
            else:
                raw = call_openai(messages, api_key, model)
            mcq = parse_mcq(raw)
            if mcq:
                out = normalize_for_inceptbench(mcq)
                out["grade"] = grade
                out["standard"] = standard
                out["subject"] = "ELA"
                out["difficulty"] = difficulty
                out["id"] = f"diverse_{index:03d}"  # 统一 ID 避免冲突
                return out
            return None
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
            if is_429 and attempt < max_retries - 1:
                wait_s = _parse_retry_seconds(err_str)
                print(f"  [429] {standard} {difficulty}: 等待 {wait_s}s 后重试 ({attempt+1}/{max_retries})")
                time.sleep(wait_s)
            else:
                print(f"  [WARN] {standard} {difficulty}: {e}")
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description="生成 MCQ（generate_mcq）")
    parser.add_argument("--provider", choices=["gemini", "openai", "deepseek", "kimi"], required=True, help="API 提供商")
    parser.add_argument("--api-key", default=None, help="API Key（环境变量: GEMINI_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY / KIMI_API_KEY）")
    parser.add_argument("--model", default=None, help="模型名")
    parser.add_argument("--examples", default=None, help="示例 JSON 路径，默认 processed_training_data/examples.json")
    parser.add_argument("--grade", default="3", help="年级")
    parser.add_argument("--standard", default="CCSS.ELA-LITERACY.L.3.1.E", help="标准 ID")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--batch", type=int, default=1, help="生成条数（同参数重复）")
    parser.add_argument("--diverse", type=int, default=None, help="多样化生成 N 条，覆盖不同难度/标准")
    parser.add_argument("--all-combinations", action="store_true", help="遍历生成全部 (standard,difficulty) 组合（如 240 条）")
    parser.add_argument("--workers", type=int, default=None, help="--diverse 时的并行线程数（Gemini 默认 2，Kimi 默认 3 以匹配并发限制，其他 10）")
    parser.add_argument("--input-dir", default="raw_data", help="--diverse 时分析 raw_data 的目录")
    parser.add_argument("--include-think-chain", action="store_true", help="是否要求输出 <think>")
    args = parser.parse_args()

    # API Key 与默认模型
    if args.provider == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        model = args.model or "gemini-3-flash-preview"
    elif args.provider == "deepseek":
        api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        model = args.model or "deepseek-chat"
    elif args.provider == "kimi":
        api_key = args.api_key or os.environ.get("KIMI_API_KEY")
        model = args.model or "kimi-latest"
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

    # 多样化批量生成（多线程）
    if args.diverse or args.all_combinations:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute():
            input_dir = PROJECT_ROOT / input_dir
        dims = analyze_dimensions(input_dir=str(input_dir))
        n = args.diverse if args.diverse else None
        plan = build_diverse_plan(dims, n=n or 9999, all_combinations=args.all_combinations)
        n = len(plan)
        workers = args.workers
        if workers is None:
            workers = 2 if args.provider == "gemini" else (KIMI_MAX_CONCURRENCY if args.provider == "kimi" else 10)
        workers = min(workers, n)
        if args.provider == "kimi":
            workers = min(workers, KIMI_MAX_CONCURRENCY)  # 不超过平台并发数
        print(f"多样化生成 {n} 条，{workers} 线程并行，覆盖 {len(set(p[0] for p in plan))} 个标准、{len(set(p[1] for p in plan))} 个难度")
        if args.provider == "gemini" and workers > 2:
            print("  提示: Gemini 免费版约 5 次/分钟，建议 --workers 2；遇 429 会自动重试")
        if args.provider == "kimi":
            print("  提示: Kimi 并发≤3、RPM=20；若遇限频可 --workers 2 或 1")
        if n >= 50 and len(examples) < 8:
            print("  建议: 可先运行 python main.py select-examples -n 8 以增加示例覆盖")
        print(f"  计划: {plan[:5]}..." if len(plan) > 5 else f"  计划: {plan}")

        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _generate_one,
                    standard=s,
                    difficulty=d,
                    examples=examples,
                    provider=args.provider,
                    api_key=api_key,
                    model=model,
                    grade=args.grade,
                    index=i,
                ): (i, s, d)
                for i, (s, d) in enumerate(plan)
            }
            done = 0
            for fut in as_completed(futures):
                i, s, d = futures[fut]
                try:
                    mcq = fut.result()
                    if mcq:
                        results.append(mcq)
                        done += 1
                        print(f"生成 {done}/{n}: {mcq.get('id', '?')} ({s} {d})")
                    else:
                        print(f"生成失败: {s} {d}")
                except Exception as e:
                    print(f"生成异常 {s} {d}: {e}")

        if not results:
            print("未成功生成任何 MCQ")
            sys.exit(1)
        # 校验并自动修复，只保留通过校验的题目
        results, validated_stats = _validate_and_keep_passing(results)
        if validated_stats["dropped"]:
            print(f"  [校验] 丢弃未通过校验: {validated_stats['dropped']} 题")
        if validated_stats["fixed"]:
            print(f"  [校验] 自动修复后通过: {validated_stats['fixed']} 题")
        if not results:
            print("无题目通过校验，不写入文件")
            sys.exit(1)
    else:
        # 单参数模式（原有逻辑）
        filtered = _filter_examples_for_standard_difficulty(examples, args.standard, args.difficulty)
        system, user = build_full_prompt(
            grade=args.grade,
            standard=args.standard,
            difficulty=args.difficulty,
            examples=filtered,
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
            elif args.provider == "kimi":
                raw = call_kimi(messages, api_key, model)
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

        # 单条/批量模式：校验并自动修复，只保留通过校验的题目
        results, validated_stats = _validate_and_keep_passing(results)
        if validated_stats["dropped"]:
            print(f"  [校验] 丢弃未通过校验: {validated_stats['dropped']} 题")
        if validated_stats["fixed"]:
            print(f"  [校验] 自动修复后通过: {validated_stats['fixed']} 题")

    if not results:
        print("未成功生成任何 MCQ 或无一题通过校验")
        sys.exit(1)

    # 输出：仅保存通过校验的题目
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results[0], f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"已保存到: {out_path}（仅通过校验的 {len(results)} 题）")
    else:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
