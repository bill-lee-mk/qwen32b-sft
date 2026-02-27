#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立运行脚本：仅需 prompt_bundle.json + openai，无需本仓库其他代码。

用法:
  pip install openai
  export DEEPSEEK_API_KEY=sk-xxx
  python scripts/run_with_bundle.py --bundle evaluation_output/prompt_bundle.json --standard CCSS.ELA-LITERACY.L.3.1.E --difficulty medium --model deepseek-reasoner

  # 生成全部 237 题（默认并发：DeepSeek-reasoner 8、Kimi 15；遇 429 可 --workers 5 或 1）
  python scripts/run_with_bundle.py --bundle prompt_bundle.json --all --model deepseek-reasoner --output mcqs.json
  python scripts/run_with_bundle.py --bundle prompt_bundle_kimi-k2.5.json --all --model kimi-k2.5 --output mcqs.json
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("请安装: pip install openai")
    exit(1)

# === 直连 API 默认并发 ===
DEEPSEEK_REASONER_WORKERS = 20  # DeepSeek 无硬性 RPM 限制，reasoner 单条 ~30-60s
DEEPSEEK_CHAT_WORKERS = 30      # chat 响应更快，可更高并发
KIMI_DEFAULT_WORKERS = 50       # Kimi Tier1: 50 并发 / 200 RPM
KIMI_MIN_INTERVAL = 0.3         # Kimi 请求间隔(s)，50×0.3=15s 发 50 请求 ≈ 200 RPM
# === Fireworks 企业账户默认并发 ===
FIREWORKS_DEFAULT_WORKERS = 50  # 企业账户无硬性并发限制
_kimi_rate_lock = threading.Lock()
_kimi_last_call = 0.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDERS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "kimi": ("https://api.moonshot.cn/v1", "KIMI_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "openai": (None, "OPENAI_API_KEY"),
}
FIREWORKS_MODEL_MAP = {
    "deepseek-r1":       "accounts/fireworks/models/deepseek-r1",
    "deepseek-v3.2":     "accounts/fireworks/models/deepseek-v3p2",
    "kimi-k2.5":         "accounts/fireworks/models/kimi-k2p5",
    "glm-5":             "accounts/fireworks/models/glm-5",
    "gpt-oss-120b":      "accounts/fireworks/models/gpt-oss-120b",
    "qwen3-235b":        "accounts/fireworks/models/qwen3-235b-a22b",
}


def _provider(model: str):
    m = (model or "").lower()
    if m.startswith("fw/") or m.startswith("fireworks/"):
        return "fireworks"
    if "deepseek" in m:
        return "deepseek"
    if "kimi" in m or "moonshot" in m:
        return "kimi"
    return "openai"


def _resolve_fw_model(model: str) -> str:
    """fw/deepseek-r1 → accounts/fireworks/models/deepseek-r1"""
    short = model.replace("fw/", "").replace("fireworks/", "").strip()
    return FIREWORKS_MODEL_MAP.get(short, f"accounts/fireworks/models/{short}")


def _kimi_rate_wait():
    """Kimi 限速：保证请求间隔 ≥ KIMI_MIN_INTERVAL，满足 200 RPM。"""
    global _kimi_last_call
    with _kimi_rate_lock:
        now = time.time()
        wait = _kimi_last_call + KIMI_MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
        _kimi_last_call = time.time()


def _call_api(messages, model: str, api_key: str, base_url: str | None, max_retries: int = 3, rate_limit_kimi: bool = False):
    """调用 API，Kimi 遇 429 时重试（等待 120s）。rate_limit_kimi=True 时对 Kimi 做 RPM 限速。"""
    prov = _provider(model)
    if rate_limit_kimi and prov == "kimi":
        _kimi_rate_wait()
    actual_model = _resolve_fw_model(model) if prov == "fireworks" else model
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    m = model.lower()
    temp = 1.0 if ("kimi" in m and "k2" in m) else 0.7
    max_tok = 4096 if prov == "fireworks" else 8192
    kwargs = {"model": actual_model, "messages": messages, "max_tokens": max_tok, "temperature": temp}
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_429 = "429" in err_str or "quota" in err_str or "rate" in err_str
            if is_429 and attempt < max_retries - 1:
                wait_s = 120 if prov == "kimi" else 30
                hint = " 建议降低 --workers（如 5 或 1）" if prov == "kimi" else ""
                print(f"  [429] 限流，{wait_s}s 后重试 ({attempt + 1}/{max_retries}){hint}", flush=True)
                time.sleep(wait_s)
            else:
                raise last_err
    raise last_err


def _parse_mcq(text: str) -> dict | None:
    """优先使用 generate_questions 的解析逻辑（更健壮），否则用内置 fallback。"""
    if not (text or "").strip():
        return None
    # 本仓库内运行时复用 generate_questions 的完整解析
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from data_processing.select_examples import extract_json_from_text, is_valid_mcq
        from scripts.generate_questions import parse_mcq as _parse_impl
        return _parse_impl(text)
    except ImportError:
        pass
    # 独立运行时的 fallback：与 generate_questions 逻辑对齐
    s = text.strip()
    if s.startswith("<think>") and "</think>" in s:
        s = s[s.index("</think>") + 7 :].strip()
    for marker in ("```json", "```"):
        if marker in s:
            parts = s.split(marker, 2)
            if len(parts) >= 2:
                s = parts[1].strip()
                if s.endswith("```"):
                    s = s[:-3].strip()
                break
    # 提取 JSON（支持纯 JSON、无 <think>/``` 的情况）
    def _extract(s):
        start = s.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        pass
        return None
    obj = _extract(s)
    if not obj and "```" in s:
        for part in s.split("```"):
            if "{" in part:
                obj = _extract(part)
                if obj:
                    break
    # 修复截断的 JSON（Kimi 等可能因 max_tokens 截断）
    if not obj and "{" in s:
        start = s.find("{")
        base = s[start:].rstrip()
        for suffix in ('"}', "}"):
            try:
                repaired = base + suffix
                obj = json.loads(repaired)
                if obj:
                    break
            except json.JSONDecodeError:
                continue
    if not obj or not isinstance(obj, dict):
        return None
    # 补全必填字段
    if "type" not in obj:
        obj["type"] = "mcq"
    # 兼容 answer_options 为 list
    opts = obj.get("answer_options")
    if isinstance(opts, list):
        d = {}
        for o in opts:
            k = str(o.get("key", o.get("letter", ""))).upper().strip()[:1]
            t = o.get("text", o.get("content", o.get("value", "")))
            if k in "ABCD":
                d[k] = str(t)
        if set(d.keys()) == {"A", "B", "C", "D"}:
            obj["answer_options"] = d
    if not isinstance(obj.get("answer_options"), dict) or set(obj["answer_options"].keys()) != {"A", "B", "C", "D"}:
        return None
    ans = str(obj.get("answer", "")).upper().strip()[:1]
    if ans not in ("A", "B", "C", "D"):
        return None
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="prompt_bundle.json 路径")
    p.add_argument("--standard", default=None)
    p.add_argument("--difficulty", default="medium")
    p.add_argument("--model", default="deepseek-reasoner")
    p.add_argument("--api-key", default=None)
    p.add_argument("--output", default="mcqs.json")
    p.add_argument("--all", action="store_true", help="生成全部 237 题")
    p.add_argument("--workers", type=int, default=None, help="并行数（默认 Fireworks 50、Kimi 50、DeepSeek-reasoner 20、DeepSeek-chat 30）")
    args = p.parse_args()

    with open(args.bundle, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    prompts = bundle.get("prompts", {})
    plan = bundle.get("plan", [])

    prov = _provider(args.model)
    base_url, env_key = PROVIDERS[prov]
    api_key = args.api_key or os.environ.get(env_key)
    if prov == "kimi" and not api_key:
        api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        hint = f"{env_key} 或 MOONSHOT_API_KEY" if prov == "kimi" else env_key
        print(f"错误: 请设置 {hint} 或 --api-key")
        exit(1)

    if args.all:
        workers = args.workers
        if workers is None:
            if prov == "fireworks":
                workers = FIREWORKS_DEFAULT_WORKERS          # 企业账户，50 并发
            elif prov == "deepseek":
                workers = DEEPSEEK_REASONER_WORKERS if "reasoner" in args.model.lower() else DEEPSEEK_CHAT_WORKERS
            elif prov == "kimi":
                workers = KIMI_DEFAULT_WORKERS               # Tier1: 50
            else:
                workers = 10
        workers = min(workers, len(plan))
        rate_limit_kimi = prov == "kimi" and workers > 1

        results_by_idx = [None] * len(plan)
        total_tokens = 0
        total_elapsed = 0.0
        std_short = lambda s: (s or "").replace("CCSS.ELA-LITERACY.", "")[:12]
        done_count = 0
        print_lock = threading.Lock()

        def _gen_one(item):
            i, (std, diff) = item
            key = f"{std}|{diff}"
            if key not in prompts:
                return (i, None, 0.0, 0, "跳过", None)
            syst, usr = prompts[key]["system"], prompts[key]["user"]
            messages = [{"role": "system", "content": syst}, {"role": "user", "content": usr}]
            try:
                t0 = time.time()
                r = _call_api(messages, args.model, api_key, base_url, rate_limit_kimi=rate_limit_kimi)
                elapsed = time.time() - t0
                u = getattr(r, "usage", None)
                tok = (getattr(u, "total_tokens", 0) or getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)) if u else 0
                msg = r.choices[0].message
                raw = (getattr(msg, "content", None) or "").strip()
                if not raw and prov == "kimi":
                    raw = (getattr(msg, "reasoning_content", None) or "").strip()
                mcq = _parse_mcq(raw)
                if mcq:
                    mcq["grade"] = bundle.get("grade", "3")
                    mcq["standard"] = std
                    mcq["subject"] = bundle.get("subject", "ELA")
                    mcq["difficulty"] = diff
                    mcq["id"] = f"diverse_{i:03d}"
                    return (i, mcq, elapsed, tok, "OK", None)
                return (i, None, elapsed, tok, "解析失败", None)
            except Exception as e:
                elapsed = time.time() - t0
                return (i, None, elapsed, 0, "错误", str(e)[:36])

        if prov == "kimi":
            print(f"  Kimi: 首次请求前等待 5s，{workers} 并发，限速 {KIMI_MIN_INTERVAL}s/请求（遇 429 可试 --workers 5 或 1）", flush=True)
            time.sleep(5)
        else:
            print(f"  {workers} 并发", flush=True)

        t_start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_gen_one, (i, p)): i for i, p in enumerate(plan)}
            for fut in as_completed(futures):
                i, mcq, elapsed, tok, st, err = fut.result()
                std, diff = plan[i]
                total_elapsed += elapsed
                total_tokens += tok
                if mcq:
                    results_by_idx[i] = mcq
                    done_count += 1
                with print_lock:
                    if st == "跳过":
                        print(f"  [{(i+1):>3}/237] {std_short(std):<12} {diff:<6} 跳过 (无 prompt)", flush=True)
                    elif st == "错误":
                        print(f"  [{(i+1):>3}/237] {std_short(std):<12} {diff:<6} {'错误':<6} {elapsed:>5.1f}s  {'-':>6}    {err or ''}", flush=True)
                    else:
                        print(f"  [{(i+1):>3}/237] {std_short(std):<12} {diff:<6} {st:<6} {elapsed:>5.1f}s  {tok:>6}tok", flush=True)
        results = [r for r in results_by_idx if r is not None]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        wall = time.time() - t_start
        print(f"已保存 {len(results)} 题到 {args.output}  墙钟 {wall:.1f}s  总 token {total_tokens}", flush=True)
    else:
        if not args.standard:
            print("错误: 单题模式需 --standard")
            exit(1)
        key = f"{args.standard}|{args.difficulty}"
        if key not in prompts:
            print(f"错误: 无此组合的 prompt: {key}")
            exit(1)
        syst, usr = prompts[key]["system"], prompts[key]["user"]
        messages = [{"role": "system", "content": syst}, {"role": "user", "content": usr}]
        t0 = time.time()
        r = _call_api(messages, args.model, api_key, base_url)
        elapsed = time.time() - t0
        u = getattr(r, "usage", None)
        tok = (getattr(u, "total_tokens", 0) or getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0)) if u else 0
        msg = r.choices[0].message
        raw = (getattr(msg, "content", None) or "").strip()
        if not raw and prov == "kimi":
            raw = (getattr(msg, "reasoning_content", None) or "").strip()
        mcq = _parse_mcq(raw)
        if mcq:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(mcq, f, ensure_ascii=False, indent=2)
            print(f"已保存到 {args.output}  耗时 {elapsed:.1f}s  {tok} tok")
        else:
            print(f"解析失败 (耗时 {elapsed:.1f}s  {tok} tok)，原始输出:")
            print(raw[:500])


if __name__ == "__main__":
    main()
