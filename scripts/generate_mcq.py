#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 MCQ（generate_mcq）

用闭源 API（Gemini/OpenAI/DeepSeek）生成 MCQ。

用法:
  # 1. 先筛选示例
  python main.py select-examples -n 5

  # 2. 生成 MCQ（单条，--model 指定具体模型，默认 deepseek-chat）
  python scripts/generate_mcq.py --model deepseek-chat --output evaluation_output/mcqs.json
  python scripts/generate_mcq.py --model kimi-latest --output evaluation_output/mcqs.json

  # 3. 全组合生成（输出默认 evaluation_output/mcqs_237_<model>.json，79 标准×3 难度=237 题）
  python scripts/generate_mcq.py --model deepseek-chat --all-combinations
  python scripts/generate_mcq.py --model gemini-3-flash-preview --all-combinations --output evaluation_output/mcqs.json
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.analyze_dimensions import analyze_dimensions, build_diverse_plan
from data_processing.build_prompt import build_full_prompt
from data_processing.select_examples import extract_json_from_text, is_valid_mcq
from evaluation.inceptbench_client import normalize_for_inceptbench
from scripts.validate_mcq import (
    build_minimal_valid_mcq,
    repair_aggressively,
    validate_and_fix,
    validate_mcq,
)


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


def _extract_kimi_429_message(err_str: str) -> str:
    """从 Kimi 429 错误中提取关键信息（如 TPD rate limit, current: X, limit: Y）"""
    # 格式: ... request reached organization TPD rate limit, current: 1501520, limit: 1500000
    m = re.search(r"request reached organization\s+(.+)", err_str)
    if m:
        return m.group(1).strip()
    return err_str


def call_openai(messages: list, api_key: str, model: str = "gpt-4o", base_url: str | None = None, temperature: float = 0.7, max_tokens: int = 1024) -> tuple[str, dict | None]:
    """调用 OpenAI 兼容 API，返回 (content, usage_dict)。Kimi 等 OpenAI 兼容接口均返回 usage。"""
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
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    usage = None
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", None),
        }
        if hasattr(u, "prompt_cache_hit_tokens"):
            usage["prompt_cache_hit_tokens"] = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        if hasattr(u, "prompt_cache_miss_tokens"):
            usage["prompt_cache_miss_tokens"] = getattr(u, "prompt_cache_miss_tokens", 0) or 0
    return content, usage


def call_deepseek(messages: list, api_key: str, model: str = "deepseek-chat") -> tuple[str, dict | None]:
    """调用 DeepSeek API。reasoner 模型会返回 reasoning_content + content，取 content 作为最终答案用于解析 MCQ。
    返回 (content, usage_dict)。usage_dict 含 prompt_tokens, completion_tokens, prompt_cache_hit_tokens, prompt_cache_miss_tokens。
    参数由 _get_generation_params(deepseek) 控制。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装: pip install openai")
    params = _get_generation_params("deepseek", model)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
    )
    msg = response.choices[0].message
    content = (getattr(msg, "content", None) or "").strip()
    # deepseek-reasoner 的最终答案在 content；若为空则尝试 reasoning_content（部分版本或流式可能不同）
    if not content and getattr(msg, "reasoning_content", None):
        content = (msg.reasoning_content or "").strip()
    usage = None
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", None),
        }
        # DeepSeek 缓存字段（命中 0.1 元/百万，未命中 1 元/百万）
        if hasattr(u, "prompt_cache_hit_tokens"):
            usage["prompt_cache_hit_tokens"] = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        if hasattr(u, "prompt_cache_miss_tokens"):
            usage["prompt_cache_miss_tokens"] = getattr(u, "prompt_cache_miss_tokens", 0) or 0
    return content, usage


# Kimi (Moonshot) 官方 API：https://platform.moonshot.cn ，OpenAI 兼容，需配置 API Key
KIMI_API_BASE = "https://api.moonshot.cn/v1"
KIMI_MAX_CONCURRENCY = 50  # Tier1 并发 50
# Tier1：RPM 200 → 间隔 0.3s，并发 50
KIMI_MIN_INTERVAL = 0.3
KIMI_MAX_CONCURRENT = 50
# 429 后重试等待（秒），Kimi 配额重置较慢，需较长等待
KIMI_429_WAIT_SECONDS = 120
_kimi_rate_lock = threading.Lock()
_kimi_last_call_time = 0.0
_kimi_sem = threading.Semaphore(max(1, KIMI_MAX_CONCURRENT))


def _is_local_model(model: str) -> bool:
    """是否为本地模型：'local' 或指向本地目录的路径（如 models/qwen3-32B/final_model）"""
    m = (model or "").strip()
    if m.lower() == "local":
        return True
    if not m:
        return False
    if m.startswith("models/") or m.startswith("/"):
        return True
    p = PROJECT_ROOT / m
    if "/" in m and p.exists() and p.is_dir():
        return True
    return False


def _model_to_provider(model: str) -> str:
    """从模型名推断 API 提供商：local / deepseek / kimi / gemini / openai"""
    if _is_local_model(model):
        return "local"
    m = (model or "").strip().lower()
    if m.startswith("deepseek-") or m == "deepseek":
        return "deepseek"
    if m.startswith("kimi-") or m.startswith("moonshot-") or m == "kimi":
        return "kimi"
    if m.startswith("gemini-") or m == "gemini":
        return "gemini"
    return "openai"


def _get_generation_params(provider: str, model: str) -> dict:
    """
    按 provider 和 model 返回生成参数，避免混淆。
    - DeepSeek: temperature=0.7, max_tokens=8192
    - Kimi: k2 系列仅支持 temperature=1，其余 0.7；max_tokens=8192
    - Local: temperature=0.7, max_tokens=4096
    - OpenAI/Gemini: temperature=0.7, max_tokens=1024
    """
    m = (model or "").strip().lower()
    if provider == "deepseek":
        return {"temperature": 0.7, "max_tokens": 8192}
    if provider == "kimi":
        temp = 1.0 if "k2" in m else 0.7  # kimi-k2.5 等仅支持 temperature=1
        return {"temperature": temp, "max_tokens": 8192}
    if provider == "local":
        return {"temperature": 0.7, "max_tokens": 4096}
    return {"temperature": 0.7, "max_tokens": 1024}


def _get_api_key_for_model(model: str) -> str | None:
    """根据模型名返回对应环境变量中的 API Key；本地模型返回 dummy。"""
    provider = _model_to_provider(model)
    if provider == "local":
        return "dummy"
    env_map = {"deepseek": "DEEPSEEK_API_KEY", "kimi": "KIMI_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
    return os.environ.get(env_map[provider])


def call_kimi(messages: list, api_key: str, model: str = "kimi-latest") -> tuple[str, dict | None]:
    """调用 Kimi (Moonshot) API，OpenAI 兼容接口。限速：KIMI_MIN_INTERVAL 控制间隔，KIMI_MAX_CONCURRENT 控制并发。返回 (content, usage) 同 DeepSeek。"""
    global _kimi_last_call_time
    with _kimi_sem:
        with _kimi_rate_lock:
            now = time.time()
            if _kimi_last_call_time == 0.0:
                time.sleep(5)  # 首次请求前稍等，避免冷启动 429
            wait = _kimi_last_call_time + KIMI_MIN_INTERVAL - now
            if wait > 0:
                time.sleep(wait)
            _kimi_last_call_time = time.time()
        params = _get_generation_params("kimi", model)
        return call_openai(messages, api_key, model, base_url=KIMI_API_BASE, temperature=params["temperature"], max_tokens=params["max_tokens"])


def _local_api_bases() -> list[str]:
    """解析 LOCAL_API_BASE：逗号分隔多个地址时做轮询，提升多卡利用率。"""
    raw = os.environ.get("LOCAL_API_BASE", "http://127.0.0.1:8000").strip()
    parts = [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]
    if not parts:
        parts = ["http://127.0.0.1:8000"]
    return [f"{p}/v1" if not p.endswith("/v1") else p for p in parts]


_local_api_bases_cache: list[str] | None = None
_local_api_round_robin_index = 0
_local_api_round_robin_lock = threading.Lock()


def _get_local_api_base() -> str:
    """本地模型 API 根地址；多地址时轮询，便于 8 实例 8 卡打满。"""
    global _local_api_bases_cache, _local_api_round_robin_index
    if _local_api_bases_cache is None:
        _local_api_bases_cache = _local_api_bases()
    bases = _local_api_bases_cache
    with _local_api_round_robin_lock:
        idx = _local_api_round_robin_index % len(bases)
        _local_api_round_robin_index += 1
        return bases[idx]


def _check_local_api_reachable() -> None:
    """本地模型：在开始批量生成前检查 serve-api 是否可达；多地址时检查全部。"""
    bases = _local_api_bases()
    import urllib.request
    for base in bases:
        health_url = base.rstrip("/").replace("/v1", "") + "/health"
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as _:
                pass
        except Exception as e:
            raise RuntimeError(
                f"本地 API 不可达: {health_url}\n"
                f"错误: {e}\n"
                "请先启动 serve-api。多卡时可起 8 个实例并设 LOCAL_API_BASE=http://127.0.0.1:8000,8001,...,8007"
            ) from e


# 本地模型单条超时（秒）；32B 长 prompt 单条可能 15–30 分钟，默认 30 分钟，可用环境变量 LOCAL_API_TIMEOUT 覆盖
def _local_api_timeout() -> float:
    try:
        return float(os.environ.get("LOCAL_API_TIMEOUT", "1800").strip())
    except ValueError:
        return 1800.0


def call_local(messages: list, base_url: str, model: str = "local") -> str:
    """调用本地 serve-api 的 OpenAI 兼容 /v1/chat/completions（需先启动 python main.py serve-api --model <path>）。
    参数由 _get_generation_params(local) 控制。"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装: pip install openai")
    params = _get_generation_params("local", model)
    client = OpenAI(api_key="dummy", base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            timeout=_local_api_timeout(),
        )
    except Exception as e:
        err = str(e).strip()
        if "connection" in err.lower() or "refused" in err.lower() or "connect" in err.lower():
            raise RuntimeError(
                f"连接本地 API 失败 ({base_url})：{err}\n"
                "请确认：1) serve-api 已启动 (python main.py serve-api --model <path>)；"
                "2) 本机访问用 LOCAL_API_BASE=http://127.0.0.1:8000；远程访问用 http://<服务器IP>:8000"
            ) from e
        if "timeout" in err.lower() or "timed out" in err.lower():
            raise RuntimeError(
                f"本地 API 请求超时（单条限 {int(_local_api_timeout())}s）。可设置 LOCAL_API_TIMEOUT=3600 延长或降低 --workers"
            ) from e
        raise
    return (response.choices[0].message.content or "").strip()


def _normalize_parsed_mcq(obj: dict) -> dict | None:
    """将解析出的对象规范化为 MCQ 格式，兼容 kimi-k2.5 等模型的多种输出格式。"""
    if not isinstance(obj, dict):
        return None
    # 补全必填字段默认值
    if "type" not in obj:
        obj["type"] = "mcq"
    if "id" not in obj and "question" in obj:
        obj["id"] = "parsed"
    # 兼容 answer_options 为 list 格式：[{key:"A", text:"..."}, ...]
    opts = obj.get("answer_options")
    if isinstance(opts, list):
        d = {}
        for o in opts:
            k = str(o.get("key", o.get("letter", ""))).upper().strip()[:1]
            t = o.get("text", o.get("content", o.get("value", "")))
            if k in "ABCD":
                d[k] = str(t)
        if set(d.keys()) == {"A", "B", "C", "D"}:
            obj = {**obj, "answer_options": d}
    # 兼容 answer 别名
    if "answer" not in obj and "correct_answer" in obj:
        obj["answer"] = obj.pop("correct_answer", "")
    # 兼容 options 别名
    if "answer_options" not in obj and "options" in obj:
        o = obj["options"]
        if isinstance(o, dict) and set(str(k).upper()[:1] for k in o) >= {"A", "B", "C", "D"}:
            obj["answer_options"] = {k: str(v) for k, v in o.items()}
    return obj


def parse_mcq(text: str) -> dict | None:
    """从模型回复中解析 MCQ JSON。支持 <think>、```json 等包裹，兼容多种输出格式。"""
    if not (text or "").strip():
        return None
    s = text.strip()
    # 去掉 <think>...</think> 包裹
    if s.startswith("<think>") and "</think>" in s:
        idx = s.index("</think>") + 7
        s = s[idx:].strip()
    # 去掉 ```json ... ``` 或 ``` ... ``` 包裹
    for marker in ("```json", "```"):
        if marker in s:
            parts = s.split(marker, 2)
            if len(parts) >= 2:
                s = parts[1].strip()
                if s.endswith("```"):
                    s = s[:-3].strip()
                break
    obj = extract_json_from_text(s)
    # 若首次失败，尝试从 ```json 块内提取（可能有多段文本）
    if not obj and "```" in s:
        for part in s.split("```"):
            if "{" in part:
                obj = extract_json_from_text(part)
                if obj:
                    break
    # 若仍失败，尝试修复截断的 JSON（如 Kimi 输出被 max_tokens 截断）
    if not obj and "{" in s:
        start = s.find("{")
        base = s[start:].rstrip()
        for suffix in ('"}', '}'):
            try:
                repaired = base + suffix
                obj = json.loads(repaired)
                if obj:
                    break
            except json.JSONDecodeError:
                continue
    if not obj:
        return None
    obj = _normalize_parsed_mcq(obj)
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
    （用于单条/批量模式，不要求与组合一一对应）
    """
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


def _validate_and_repair_keep_all(
    results_by_index: list,
    plan: list,
    grade: str = "3",
    subject: str = "ELA",
) -> tuple[list, dict]:
    """
    与 plan 一一对应：results_by_index[i] 为第 i 个 (standard, difficulty) 的生成结果（可为 None）。
    对未生成或校验不通过的题目进行修复或构造，保证输出题目数 = len(plan)，且每条通过校验。
    返回 (题目列表, {"fixed": 修复后通过数, "repaired": 激进修复后通过数, "constructed": 构造的最小合法题数})
    """
    n = len(plan)
    assert len(results_by_index) == n
    fixed_count = 0
    repaired_count = 0
    constructed_count = 0
    out = [None] * n
    for i in range(n):
        standard, difficulty = plan[i]
        mcq = results_by_index[i]
        if mcq is None:
            out[i] = build_minimal_valid_mcq(standard, difficulty, grade=grade, subject=subject, index=i)
            constructed_count += 1
            continue
        had_issues = len(validate_mcq(mcq, i)) > 0
        mcq_out, passed = validate_and_fix(mcq, index=i, max_rounds=2)
        if passed:
            if had_issues:
                fixed_count += 1
            out[i] = mcq_out
            out[i]["id"] = f"diverse_{i:03d}"
            continue
        mcq_out = repair_aggressively(mcq_out, standard=standard, difficulty=difficulty, index=i)
        if not validate_mcq(mcq_out, i):
            out[i] = mcq_out
            out[i]["id"] = f"diverse_{i:03d}"
            repaired_count += 1
        else:
            out[i] = build_minimal_valid_mcq(standard, difficulty, grade=grade, subject=subject, index=i)
            constructed_count += 1
    for i in range(n):
        out[i]["id"] = f"diverse_{i:03d}"
    return out, {"fixed": fixed_count, "repaired": repaired_count, "constructed": constructed_count}


def _generate_one(
    standard: str,
    difficulty: str,
    examples: list,
    provider: str,
    api_key: str,
    model: str,
    grade: str = "3",
    subject: str = "ELA",
    index: int = 0,
    max_retries: int = 3,
) -> tuple[dict | None, float, str | None, dict | None]:
    """生成单条 MCQ，供多线程调用。遇 429 自动重试。返回 (mcq_or_none, 耗时秒, 异常信息_or_None, usage_dict_or_None)。"""
    start = time.time()
    filtered = _filter_examples_for_standard_difficulty(examples, standard, difficulty)
    system, user = build_full_prompt(
        grade=grade,
        standard=standard,
        difficulty=difficulty,
        examples=filtered,
        subject=subject,
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(max_retries):
        try:
            usage = None
            if provider == "gemini":
                raw = call_gemini(f"{system}\n\n{user}", api_key, model)
            elif provider == "deepseek":
                raw, usage = call_deepseek(messages, api_key, model)
            elif provider == "kimi":
                raw, usage = call_kimi(messages, api_key, model)
            elif provider == "local":
                raw = call_local(messages, _get_local_api_base(), model)
            else:
                p = _get_generation_params(provider, model)
                raw, usage = call_openai(messages, api_key, model, temperature=p["temperature"], max_tokens=p["max_tokens"])
            mcq = parse_mcq(raw)
            elapsed = time.time() - start
            if mcq:
                out = normalize_for_inceptbench(mcq)
                out["grade"] = grade
                out["standard"] = standard
                out["subject"] = subject
                out["difficulty"] = difficulty
                out["id"] = f"diverse_{index:03d}"
                return (out, elapsed, None, usage)
            if provider == "kimi" and raw:
                debug_path = PROJECT_ROOT / "evaluation_output" / "debug_kimi_raw.txt"
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(raw)
                print(f"  [解析失败] 已保存完整输出到 {debug_path}", flush=True)
            return (None, elapsed, None, usage)
        except Exception as e:
            err_str = str(e)
            is_429 = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
            if is_429 and attempt < max_retries - 1:
                wait_s = KIMI_429_WAIT_SECONDS if provider == "kimi" else _parse_retry_seconds(err_str)
                msg = _extract_kimi_429_message(err_str) if provider == "kimi" else err_str
                print(f"  [429] {standard} {difficulty}: {msg}", flush=True)
                time.sleep(wait_s)
            else:
                elapsed = time.time() - start
                msg = _extract_kimi_429_message(err_str) if (provider == "kimi" and is_429) else err_str
                print(f"  [WARN] {standard} {difficulty}: {msg}", flush=True)
                return (None, elapsed, err_str, None)
    elapsed = time.time() - start
    return (None, elapsed, None, None)


def main():
    parser = argparse.ArgumentParser(description="生成 MCQ（generate_mcq）")
    parser.add_argument("--model", default="deepseek-chat", help="具体模型名，如 deepseek-chat / deepseek-reasoner / kimi-latest / gemini-3-flash-preview / gpt-4o（据此前缀选 API）")
    parser.add_argument("--api-key", default=None, help="API Key（未设时从环境变量按模型推断：DEEPSEEK_API_KEY / KIMI_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY）")
    parser.add_argument("--examples", default=None, help="示例 JSON 路径，默认 processed_training_data/examples.json")
    parser.add_argument("--grade", default="3", help="年级（K, 1-12, AP, HS, SAT）")
    parser.add_argument("--subject", default="ELA", help="学科缩写（ELA, MATH, SCI, USHIST 等）")
    parser.add_argument("--standard", default="CCSS.ELA-LITERACY.L.3.1.E", help="标准 ID")
    parser.add_argument("--difficulty", default="medium", help="难度")
    parser.add_argument("--output", default=None, help="输出 JSON 路径（--all-combinations 未指定时默认 evaluation_output/mcqs_237_<model>.json）")
    parser.add_argument("--batch", type=int, default=1, help="生成条数（同参数重复）")
    parser.add_argument("--diverse", type=int, default=None, help="多样化生成 N 条，覆盖不同难度/标准")
    parser.add_argument("--all-combinations", action="store_true", help="遍历生成全部 (standard,difficulty) 组合（79 标准×3 难度=237 题）")
    parser.add_argument("--workers", type=int, default=None, help="并行线程数（Gemini 2，Kimi 10，本地 8，DeepSeek/其他 API 10）")
    parser.add_argument("--input-dir", default="raw_data", help="--diverse 时分析 raw_data 的目录")
    parser.add_argument("--include-think-chain", action="store_true", help="是否要求输出 <think>")
    args = parser.parse_args()

    model = (args.model or "deepseek-chat").strip()
    provider = _model_to_provider(model)
    api_key = args.api_key or _get_api_key_for_model(model)
    if provider != "local" and not api_key:
        print("错误: 请提供 --api-key 或设置环境变量（按模型推断: deepseek-* -> DEEPSEEK_API_KEY, kimi-* -> KIMI_API_KEY, gemini-* -> GEMINI_API_KEY, 其他 -> OPENAI_API_KEY）")
        sys.exit(1)
    if provider == "local":
        print("使用模型: 本地 (serve-api)，确保已启动: python main.py serve-api --model <模型路径>")
    else:
        print(f"使用模型: {model}")

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
        from data_processing.analyze_dimensions import analyze_dimensions_from_curriculum, validate_grade_subject
        err = validate_grade_subject(args.grade, args.subject)
        if err:
            print(err)
            sys.exit(1)
        dims_curriculum = analyze_dimensions_from_curriculum(args.grade, args.subject)
        if dims_curriculum["total"] > 0:
            dims = dims_curriculum
        else:
            input_dir = Path(args.input_dir)
            if not input_dir.is_absolute():
                input_dir = PROJECT_ROOT / input_dir
            dims = analyze_dimensions(input_dir=str(input_dir))
        n = args.diverse if args.diverse else None
        plan = build_diverse_plan(dims, n=n or 9999, all_combinations=args.all_combinations)
        n = len(plan)
        workers = args.workers
        if workers is None:
            # 生成并发默认：Gemini 2，Kimi 50，本地 8，DeepSeek-reasoner 6（单条慢，适度并发），DeepSeek-chat/其他 API 10
            if provider == "deepseek" and "reasoner" in (model or "").lower():
                workers = 6
            else:
                workers = 2 if provider == "gemini" else (KIMI_MAX_CONCURRENCY if provider == "kimi" else (8 if provider == "local" else 10))
        workers = min(workers, n)
        if provider == "kimi":
            workers = min(workers, KIMI_MAX_CONCURRENCY)  # 不超过平台并发数
        print(f"多样化生成 {n} 条，{workers} 线程并行，覆盖 {len(set(p[0] for p in plan))} 个标准、{len(set(p[1] for p in plan))} 个难度")
        if provider == "local":
            print("  提示: 本地默认 8 并发（8 卡机可保持队列）；单进程 serve-api 串行推理仅用 1–2 卡，若需用满 8 卡可起 8 个 serve-api 实例并做负载均衡")
        if provider == "gemini" and workers > 2:
            print("  提示: Gemini 免费版约 5 次/分钟，建议 --workers 2；遇 429 会自动重试")
        if provider == "kimi":
            print(f"  提示: Kimi {KIMI_MAX_CONCURRENT} 并发、间隔 {KIMI_MIN_INTERVAL}s；429 后等 2 分钟重试")
        if n >= 50 and len(examples) < 8:
            print("  建议: 可先运行 python main.py select-examples -n 8 以增加示例覆盖")
        print(f"  计划: {plan[:5]}..." if len(plan) > 5 else f"  计划: {plan}")
        if provider == "local":
            _check_local_api_reachable()
            bases = _local_api_bases()
            if len(bases) > 1:
                print(f"  提示: 本地多实例轮询（{len(bases)} 个地址），8 卡可打满；首条可能 1–3 分钟")
            else:
                print("  提示: 本地串行推理，首条约 1–3 分钟，后续按队列依次返回；8 卡机默认 --workers 8 保持队列")
                print("  若 10+ 分钟无一条输出，请看 serve-api 终端是否出现 [推理] 开始/完成 日志；首条 32B 长 prompt 可能需 10–15 分钟")
        # 未指定 --output 时，按模型名区分输出文件（便于对比不同模型生成结果）
        if args.all_combinations and args.output is None:
            model_slug = model.replace(".", "_").replace("/", "_")
            args.output = str(PROJECT_ROOT / "evaluation_output" / f"mcqs_237_{model_slug}.json")

        # 与评估一致：按本题集最大「题号+(标准,难度)」长度固定 label 宽度；耗时列固定宽度对齐
        label_width = max(
            len(f"题{i+1:>3} ({(s or '').replace('CCSS.ELA-LITERACY.', '')}, {d})")
            for i, (s, d) in enumerate(plan)
        )
        time_width = 8  # 耗时列宽度，如 "  12.3s"、" 120.5s"
        # 按组合下标存储，保证题目总数与 (standard, difficulty) 组合一致；未生成或校验不通过则修复或构造，不丢弃
        results_by_index = [None] * n
        generation_details = [None] * n  # 供 usage 文件写入（闭环综合日志使用）
        usage_agg = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 0, "n_calls": 0}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    _generate_one,
                    standard=s,
                    difficulty=d,
                    examples=examples,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    grade=args.grade,
                    subject=args.subject,
                    index=i,
                ): (i, s, d)
                for i, (s, d) in enumerate(plan)
            }
            done = 0
            for fut in as_completed(futures):
                i, s, d = futures[fut]
                try:
                    mcq, elapsed, err_msg, usage = fut.result()
                    if usage:
                        usage_agg["n_calls"] += 1
                        usage_agg["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
                        usage_agg["completion_tokens"] += usage.get("completion_tokens", 0) or 0
                        usage_agg["prompt_cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0) or 0
                        usage_agg["prompt_cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0) or 0
                    std_short = (s or "").replace("CCSS.ELA-LITERACY.", "")
                    tokens = (usage.get("prompt_tokens", 0) or 0) + (usage.get("completion_tokens", 0) or 0) if usage else 0
                    generation_details[i] = {"index": i + 1, "standard": std_short, "difficulty": d, "tokens": tokens, "elapsed_s": round(elapsed, 1)}
                    elapsed_s = f"{elapsed:.1f}s".rjust(time_width)
                    label = f"题{i+1:>3} ({std_short}, {d})".ljust(label_width)
                    tok = ""
                    if usage:
                        pt = (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0)
                        tok = f"  {pt}token"
                    if mcq:
                        results_by_index[i] = mcq
                        done += 1
                        print(f"[生成] [{done:>3}/{n}] {label}  {elapsed_s}{tok}", flush=True)
                    elif err_msg:
                        short = "请求超时" if "timeout" in err_msg.lower() or "timed out" in err_msg.lower() else err_msg[:28]
                        print(f"[生成] 异常: {label}  {elapsed_s}{tok}  {short}，将修复或构造", flush=True)
                    else:
                        print(f"[生成] 失败: {label}  {elapsed_s}{tok}，将修复或构造", flush=True)
                except Exception as e:
                    std_short = (s or "").replace("CCSS.ELA-LITERACY.", "")
                    label = f"题{i+1:>3} ({std_short}, {d})".ljust(label_width)
                    print(f"[生成] 异常: {label}  {'—':>{time_width}}  {str(e)[:40]}…，将修复或构造", flush=True)

        # 校验不通过则修复或构造，保证输出题目数 = n、组合与 plan 一致
        results, validated_stats = _validate_and_repair_keep_all(
            results_by_index, plan, grade=args.grade, subject=args.subject
        )
        if validated_stats.get("fixed"):
            print(f"[生成] [校验] 自动修复后通过: {validated_stats['fixed']} 题")
        if validated_stats.get("repaired"):
            print(f"[生成] [校验] 激进修复后通过: {validated_stats['repaired']} 题")
        if validated_stats.get("constructed"):
            print(f"[生成] [校验] 未通过则构造最小合法题: {validated_stats['constructed']} 题（保持组合与总数一致）")
        # 记录 token 用量并估算费用（DeepSeek/Kimi/OpenAI 等 OpenAI 兼容接口均返回 usage）
        if usage_agg["n_calls"] > 0 and provider in ("deepseek", "kimi", "openai"):
            hit = usage_agg["prompt_cache_hit_tokens"]
            miss = usage_agg["prompt_cache_miss_tokens"]
            out = usage_agg["completion_tokens"]
            # 定价（元/百万 token）；Kimi 约 4/16，DeepSeek-chat 2/8、reasoner 4/16
            if provider == "kimi":
                hit_yuan, miss_yuan, out_yuan = 1.0, 4.0, 16.0  # 缓存1/未命中4，输出16
            elif provider == "deepseek":
                is_reasoner = "reasoner" in model.lower()
                hit_yuan, miss_yuan = (0.5, 2.0) if not is_reasoner else (1.0, 4.0)
                out_yuan = 16.0 if is_reasoner else 8.0
            else:  # openai 等
                hit_yuan, miss_yuan, out_yuan = 0, 0, 0  # 不估算，仅记录 token
            cost_in = (hit * hit_yuan + miss * miss_yuan) / 1e6 if (hit_yuan or miss_yuan) else 0
            cost_out = out * out_yuan / 1e6 if out_yuan else 0
            total_tok = usage_agg["prompt_tokens"] + usage_agg["completion_tokens"]
            cost_str = f"，估算 ¥{cost_in + cost_out:.2f}" if (cost_in + cost_out) > 0 else ""
            print(f"[生成] [用量] 本批 {n} 题总 token: {total_tok:,}（输入 {usage_agg['prompt_tokens']:,}，缓存命中 {hit:,}；输出 {out:,}）{cost_str}")
        # 由 --output 路径推导 usage 路径（mcqs_xxx.json -> log_xxx_usage.json），写入 evaluation_output 供闭环综合日志使用
        usage_out = str(args.output).replace("mcqs_", "log_", 1).replace(".json", "_usage.json") if args.output else None
        if usage_out:
            est_cost = 0.0
            if usage_agg["n_calls"] > 0 and provider in ("deepseek", "kimi", "openai"):
                hit = usage_agg["prompt_cache_hit_tokens"]
                miss = usage_agg["prompt_cache_miss_tokens"]
                out = usage_agg["completion_tokens"]
                if provider == "kimi":
                    hit_yuan, miss_yuan, out_yuan = 1.0, 4.0, 16.0
                elif provider == "deepseek":
                    is_reasoner = "reasoner" in model.lower()
                    hit_yuan, miss_yuan = (0.5, 2.0) if not is_reasoner else (1.0, 4.0)
                    out_yuan = 16.0 if is_reasoner else 8.0
                else:
                    hit_yuan, miss_yuan, out_yuan = 0, 0, 0
                est_cost = (hit * hit_yuan + miss * miss_yuan) / 1e6 + out * out_yuan / 1e6
            token_out = {
                "prompt_tokens": usage_agg["prompt_tokens"],
                "completion_tokens": usage_agg["completion_tokens"],
                "prompt_cache_hit_tokens": usage_agg["prompt_cache_hit_tokens"],
                "prompt_cache_miss_tokens": usage_agg["prompt_cache_miss_tokens"],
                "estimated_cost_cny": round(est_cost, 3),
            }
            generation = [g for g in generation_details if g is not None]
            usage_data = {"usage_agg": token_out, "generation": generation}
            out_usage = Path(usage_out)
            out_usage.parent.mkdir(parents=True, exist_ok=True)
            with open(out_usage, "w", encoding="utf-8") as f:
                json.dump(usage_data, f, ensure_ascii=False, indent=2)
    else:
        # 单参数模式（原有逻辑）
        filtered = _filter_examples_for_standard_difficulty(examples, args.standard, args.difficulty)
        system, user = build_full_prompt(
            grade=args.grade,
            standard=args.standard,
            difficulty=args.difficulty,
            examples=filtered,
            subject=args.subject,
            include_think_chain=args.include_think_chain,
        )

        results = []
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for i in range(args.batch):
            if provider == "gemini":
                full_prompt = f"{system}\n\n{user}"
                raw = call_gemini(full_prompt, api_key, model)
            elif provider == "deepseek":
                raw, _ = call_deepseek(messages, api_key, model)
            elif provider == "kimi":
                raw, _ = call_kimi(messages, api_key, model)
            elif provider == "local":
                raw = call_local(messages, _get_local_api_base(), model)
            else:
                p = _get_generation_params(provider, model)
                raw, _ = call_openai(messages, api_key, model, temperature=p["temperature"], max_tokens=p["max_tokens"])

            mcq = parse_mcq(raw)
            if mcq:
                normalized = normalize_for_inceptbench(mcq)
                normalized["grade"] = args.grade
                normalized["standard"] = args.standard
                normalized["subject"] = args.subject
                results.append(normalized)
                print(f"生成 {i+1}/{args.batch}: {normalized.get('id', 'unknown')}", flush=True)
            else:
                print(f"生成 {i+1}/{args.batch}: 解析失败", flush=True)
                print(f"  raw: {raw[:200]}...", flush=True)
                if provider == "kimi" and raw:
                    debug_path = PROJECT_ROOT / "evaluation_output" / "debug_kimi_raw.txt"
                    debug_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write(raw)
                    print(f"  已保存完整输出到 {debug_path} 便于排查", flush=True)

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
        print(f"[生成] 已保存到: {out_path}（仅通过校验的 {len(results)} 题）")
    else:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
