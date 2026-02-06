# -*- coding: utf-8 -*-
"""
筛选示例（select_examples）

从 raw_data 中筛选少量高质量 MCQ 样本，用于闭源模型的 prompt 示范。
"""
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MCQ_REQUIRED_FIELDS = ["id", "type", "question", "answer", "answer_options", "answer_explanation", "difficulty"]


def extract_json_from_text(text: str) -> Optional[Dict]:
    """从文本中提取第一个完整 JSON 对象"""
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


def is_valid_mcq(mcq: Dict) -> Tuple[bool, str]:
    """校验 MCQ 是否符合 InceptBench 要求"""
    if not isinstance(mcq, dict):
        return False, "not a dict"
    missing = [k for k in MCQ_REQUIRED_FIELDS if k not in mcq]
    if missing:
        return False, f"missing fields: {missing}"
    opts = mcq.get("answer_options")
    if not isinstance(opts, dict):
        return False, "answer_options must be dict"
    keys = set(str(k).upper().strip() for k in opts.keys())
    if keys != {"A", "B", "C", "D"}:
        return False, f"answer_options keys must be A,B,C,D, got {keys}"
    ans = str(mcq.get("answer", "")).upper().strip()
    if ans not in ("A", "B", "C", "D"):
        return False, f"answer must be A/B/C/D, got {ans}"
    if ans not in opts:
        return False, f"answer {ans} not in answer_options"
    return True, ""


def load_jsonl(path: str, max_lines: Optional[int] = None) -> List[Dict]:
    """加载 jsonl 文件"""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return samples


def extract_user_prompt_from_messages(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def extract_assistant_response(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def extract_mcq_json_from_assistant(content: str) -> Optional[Dict]:
    json_obj = extract_json_from_text(content)
    if not json_obj:
        return None
    ok, _ = is_valid_mcq(json_obj)
    return json_obj if ok else None


def extract_standard_from_user(user_text: str) -> Optional[str]:
    import re
    m = re.search(r"(CCSS\.ELA-LITERACY\.[A-Z0-9.-]+)", user_text)
    if m:
        return m.group(1)
    m = re.search(r"Standard ID:\s*([^\s\n]+)", user_text, re.I)
    if m:
        return m.group(1).strip()
    return None


def process_messages_file(samples: List[Dict]) -> List[Dict]:
    results = []
    for s in samples:
        if "messages" not in s:
            continue
        msgs = s.get("messages", [])
        if len(msgs) < 3:
            continue
        user = extract_user_prompt_from_messages(msgs)
        assistant = extract_assistant_response(msgs)
        if not user or not assistant:
            continue
        mcq = extract_mcq_json_from_assistant(assistant)
        if not mcq:
            continue
        standard = extract_standard_from_user(user) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user,
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "messages",
            "score": 1.0,
        })
    return results


def process_dpo_file(samples: List[Dict]) -> List[Dict]:
    results = []
    for s in samples:
        if "prompt" not in s or "chosen" not in s:
            continue
        prompt = s.get("prompt", [])
        chosen = s.get("chosen", {})
        if isinstance(chosen, dict):
            content = chosen.get("content", "")
        else:
            content = str(chosen)
        if not content.strip():
            continue
        mcq = extract_mcq_json_from_assistant(content)
        if not mcq:
            continue
        user_prompt = ""
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                user_prompt = m.get("content", "")
        if not user_prompt:
            user_prompt = " ".join(m.get("content", "") for m in prompt if isinstance(m, dict))
        meta = s.get("metadata", {})
        chosen_score = meta.get("chosen_score", 0.5)
        standard = meta.get("standard") or extract_standard_from_user(user_prompt) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user_prompt[:2000],
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "dpo",
            "score": float(chosen_score),
        })
    return results


def select_diverse_examples(candidates: List[Dict], n: int = 5, seed: int = 42) -> List[Dict]:
    """从候选中选出多样性好的示例"""
    random.seed(seed)
    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    seen = set()
    selected = []
    for c in sorted_candidates:
        if len(selected) >= n:
            break
        key = (c["standard"], c["difficulty"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(c)
    for c in sorted_candidates:
        if len(selected) >= n:
            break
        if c not in selected:
            selected.append(c)
    return selected[:n]


def run(
    input_dir: str = "raw_data",
    output: str = "processed_training_data/examples.json",
    n: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """从 raw_data 筛选示例并保存到 output"""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    all_candidates = []
    for fpath in sorted(input_dir.glob("*.jsonl")):
        samples = load_jsonl(str(fpath))
        if not samples:
            continue
        if "messages" in samples[0]:
            candidates = process_messages_file(samples)
        elif "prompt" in samples[0] and "chosen" in samples[0]:
            candidates = process_dpo_file(samples)
        else:
            continue
        all_candidates.extend(candidates)
        print(f"  {fpath.name}: 提取 {len(candidates)} 条候选")

    if not all_candidates:
        print("未找到有效候选样本")
        return []

    selected = select_diverse_examples(all_candidates, n=n, seed=seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(selected)} 条示例到: {output_path}")
    return selected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="筛选示例（select_examples）")
    parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    parser.add_argument("--output", default="processed_training_data/examples.json", help="输出 JSON 路径")
    parser.add_argument("-n", type=int, default=5, help="示例数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(input_dir=args.input_dir, output=args.output, n=args.n, seed=args.seed)
