#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_data 原始数据综合分析脚本

分析结构、token 分布、MCQ JSON 完整性、缺失字段比例等。
输出综合报告，供 docs/RAW_DATA_ANALYSIS.md 参考。

用法:
  python -m data_processing.analyze_raw_data
  python -m data_processing.analyze_raw_data --input-dir raw_data
  python -m data_processing.analyze_raw_data --sample 5000  # 大文件抽样
  python -m data_processing.analyze_raw_data --output report.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 与训练配置一致的长度参数（用于分析时的超长统计）
SFT_MAX_LENGTH = 2048
DPO_MAX_LENGTH = 1024
DPO_MAX_PROMPT_LENGTH = 512

# MCQ JSON 必填字段
MCQ_REQUIRED_FIELDS = ["id", "type", "question", "answer", "answer_options", "answer_explanation", "difficulty"]


def _get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)
    except Exception:
        return None


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, len(text) // 4)


def load_jsonl(path: str, sample_size: Optional[int] = None, seed: int = 42) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                samples.append({"_parse_error": True})
    if sample_size and len(samples) > sample_size:
        random.seed(seed)
        samples = random.sample(samples, sample_size)
    return samples


def extract_json_from_text(text: str) -> Optional[Dict]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def analyze_messages(samples: List[Dict], tokenizer) -> Dict:
    """分析 messages 格式，统计 JSON 完整性、缺失字段等"""
    total = len(samples)
    valid = [s for s in samples if not s.get("_parse_error") and "messages" in s]
    n_valid = len(valid)

    stats = {
        "total": total,
        "parse_errors": total - n_valid,
        "valid": n_valid,
        "no_messages": 0,
        "no_assistant": 0,
        "empty_assistant": 0,
        "no_json": 0,
        "has_json_incomplete": 0,
        "missing_fields": Counter(),
        "missing_fields_samples": 0,
        "answer_options_invalid": 0,
        "answer_invalid": 0,
        "has_think": 0,
        "token_lengths": {"system": [], "user": [], "assistant": [], "total": []},
        "over_sft_max": 0,
    }

    for s in valid:
        msgs = s.get("messages", [])
        if not msgs:
            stats["no_messages"] += 1
            continue
        last = msgs[-1] if msgs else {}
        if last.get("role") != "assistant":
            stats["no_assistant"] += 1
            continue
        content = last.get("content", "") or ""
        if not content.strip():
            stats["empty_assistant"] += 1
            continue

        if "<think>" in content and "</think>" in content:
            stats["has_think"] += 1

        total_tok = 0
        for m in msgs:
            role = m.get("role", "?")
            c = m.get("content", "") or ""
            n = _count_tokens(c, tokenizer)
            total_tok += n
            if role in stats["token_lengths"]:
                stats["token_lengths"][role].append(n)
        stats["token_lengths"]["total"].append(total_tok)
        if total_tok > SFT_MAX_LENGTH:
            stats["over_sft_max"] += 1

        json_obj = extract_json_from_text(content)
        if not json_obj:
            stats["no_json"] += 1
            continue

        missing = [k for k in MCQ_REQUIRED_FIELDS if k not in json_obj]
        if missing:
            stats["has_json_incomplete"] += 1
            stats["missing_fields_samples"] += 1
            for k in missing:
                stats["missing_fields"][k] += 1

        opts = json_obj.get("answer_options")
        if isinstance(opts, dict):
            keys = set(str(k).upper() for k in opts.keys())
            if keys != {"A", "B", "C", "D"}:
                stats["answer_options_invalid"] += 1
        ans = str(json_obj.get("answer", "")).upper().strip()
        if ans and ans not in ("A", "B", "C", "D"):
            stats["answer_invalid"] += 1

    return stats


def analyze_dpo(samples: List[Dict], tokenizer) -> Dict:
    """分析 DPO 格式"""
    total = len(samples)
    valid = [s for s in samples if not s.get("_parse_error") and "prompt" in s and "chosen" in s and "rejected" in s]
    n_valid = len(valid)

    stats = {
        "total": total,
        "parse_errors": total - n_valid,
        "valid": n_valid,
        "no_prompt": 0,
        "no_chosen": 0,
        "no_rejected": 0,
        "empty_chosen": 0,
        "empty_rejected": 0,
        "score_order_wrong": 0,
        "score_gap_small": 0,
        "token_lengths": {"prompt": [], "chosen": [], "rejected": []},
        "over_prompt_max": 0,
        "over_response_max": 0,
    }

    for s in valid:
        prompt = s.get("prompt", [])
        chosen = s.get("chosen", {})
        rejected = s.get("rejected", {})
        meta = s.get("metadata", {})

        prompt_text = "".join(m.get("content", "") for m in prompt if isinstance(m, dict))
        chosen_text = chosen.get("content", "") if isinstance(chosen, dict) else ""
        rejected_text = rejected.get("content", "") if isinstance(rejected, dict) else ""

        pt = _count_tokens(prompt_text, tokenizer)
        ct = _count_tokens(chosen_text, tokenizer)
        rt = _count_tokens(rejected_text, tokenizer)

        stats["token_lengths"]["prompt"].append(pt)
        stats["token_lengths"]["chosen"].append(ct)
        stats["token_lengths"]["rejected"].append(rt)

        if pt > DPO_MAX_PROMPT_LENGTH:
            stats["over_prompt_max"] += 1
        if ct > DPO_MAX_LENGTH or rt > DPO_MAX_LENGTH:
            stats["over_response_max"] += 1

        if not prompt:
            stats["no_prompt"] += 1
        if not chosen or not isinstance(chosen, dict):
            stats["no_chosen"] += 1
        elif not (chosen.get("content") or "").strip():
            stats["empty_chosen"] += 1
        if not rejected or not isinstance(rejected, dict):
            stats["no_rejected"] += 1
        elif not (rejected.get("content") or "").strip():
            stats["empty_rejected"] += 1

        cs = meta.get("chosen_score")
        rs = meta.get("rejected_score")
        if cs is not None and rs is not None:
            gap = float(cs) - float(rs)
            if gap <= 0:
                stats["score_order_wrong"] += 1
            elif gap < 0.05:
                stats["score_gap_small"] += 1

    return stats


def _percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    s = sorted(arr)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def _summarize(arr: List[float]) -> Dict:
    if not arr:
        return {}
    return {
        "min": min(arr),
        "p50": _percentile(arr, 50),
        "p95": _percentile(arr, 95),
        "max": max(arr),
        "mean": sum(arr) / len(arr),
    }


def main():
    parser = argparse.ArgumentParser(description="raw_data 综合分析")
    parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    parser.add_argument("--sample", type=int, default=None, help="大文件抽样条数（默认全量）")
    parser.add_argument("--output", help="输出 JSON 报告路径")
    parser.add_argument("--max-per-file", type=int, default=50000, help="单文件最大分析条数")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    input_dir = base / args.input_dir
    if not input_dir.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return 1

    paths = list(input_dir.glob("*.jsonl"))
    if not paths:
        print("未找到 .jsonl 文件")
        return 1

    tokenizer = _get_tokenizer()
    tokenizer_desc = "Qwen tokenizer" if tokenizer else "字符估算 (~4 char/token)"

    all_results = []
    for path in sorted(paths):
        samples = load_jsonl(str(path), sample_size=args.sample)
        if len(samples) > args.max_per_file:
            samples = random.sample(samples, args.max_per_file)
        total_lines = sum(1 for _ in open(path, encoding="utf-8") if _.strip())

        fmt = "messages" if samples and "messages" in samples[0] else "dpo"
        if fmt == "dpo" and samples and not ("prompt" in samples[0] and "chosen" in samples[0]):
            fmt = "unknown"

        result = {
            "file": path.name,
            "format": fmt,
            "total_lines": total_lines,
            "analyzed": len(samples),
            "tokenizer": tokenizer_desc,
        }

        if fmt == "messages":
            stats = analyze_messages(samples, tokenizer)
            n = stats["valid"]
            result["messages"] = {
                "total": stats["total"],
                "parse_errors": stats["parse_errors"],
                "valid": n,
                "no_json_count": stats["no_json"],
                "no_json_ratio": round(stats["no_json"] / n * 100, 2) if n else 0,
                "missing_fields_count": stats["missing_fields_samples"],
                "missing_fields_ratio": round(stats["missing_fields_samples"] / n * 100, 2) if n else 0,
                "missing_fields_detail": dict(stats["missing_fields"]),
                "answer_options_invalid": stats["answer_options_invalid"],
                "answer_invalid": stats["answer_invalid"],
                "has_think": stats["has_think"],
                "over_sft_max": stats["over_sft_max"],
                "over_sft_max_ratio": round(stats["over_sft_max"] / n * 100, 2) if n else 0,
                "token_lengths": {k: _summarize(v) for k, v in stats["token_lengths"].items() if v},
            }
        elif fmt == "dpo":
            stats = analyze_dpo(samples, tokenizer)
            n = stats["valid"]
            result["dpo"] = {
                "total": stats["total"],
                "parse_errors": stats["parse_errors"],
                "valid": n,
                "over_prompt_max": stats["over_prompt_max"],
                "over_prompt_max_ratio": round(stats["over_prompt_max"] / n * 100, 2) if n else 0,
                "over_response_max": stats["over_response_max"],
                "over_response_max_ratio": round(stats["over_response_max"] / n * 100, 2) if n else 0,
                "token_lengths": {k: _summarize(v) for k, v in stats["token_lengths"].items() if v},
            }

        all_results.append(result)

    # 打印摘要
    for r in all_results:
        print(f"\n=== {r['file']} ===")
        print(f"格式: {r['format']} | 总行: {r['total_lines']} | 分析: {r['analyzed']}")
        if "messages" in r:
            m = r["messages"]
            print(f"缺少 JSON 比例: {m['no_json_ratio']}% ({m['no_json_count']}/{m['valid']})")
            print(f"缺少字段比例: {m['missing_fields_ratio']}% ({m['missing_fields_count']}/{m['valid']})")
            if m["missing_fields_detail"]:
                print(f"  缺失字段详情: {m['missing_fields_detail']}")
            print(f"超 SFT max 比例: {m['over_sft_max_ratio']}%")
        if "dpo" in r:
            d = r["dpo"]
            print(f"prompt 超 512 比例: {d['over_prompt_max_ratio']}%")
            print(f"response 超 1024 比例: {d['over_response_max_ratio']}%")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
