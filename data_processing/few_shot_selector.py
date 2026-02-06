# -*- coding: utf-8 -*-
"""
Few-shot 高质量样本筛选器

从 raw_data 中筛选少量高质量 MCQ 样本，用于闭源模型的 few-shot 提示。
筛选策略：
1. 完整性：MCQ JSON 含全部必填字段
2. 格式正确：answer_options 为 A/B/C/D，answer 为 A/B/C/D
3. 质量信号：DPO 数据优先选 chosen_score 高的；messages 数据选无缺失字段的
4. 多样性：覆盖不同 standard、difficulty
"""
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# InceptBench 兼容的 MCQ 必填字段
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
    """
    校验 MCQ 是否符合 InceptBench 要求。
    Returns: (是否有效, 错误信息)
    """
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
    """从 messages 中提取 user 的最后一条（或唯一一条）"""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def extract_assistant_response(messages: List[Dict]) -> str:
    """从 messages 中提取 assistant 的最后一条"""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "").strip()
    return ""


def extract_mcq_json_from_assistant(content: str) -> Optional[Dict]:
    """从 assistant 回复中提取 MCQ JSON（可含 <think>）"""
    json_obj = extract_json_from_text(content)
    if not json_obj:
        return None
    ok, _ = is_valid_mcq(json_obj)
    return json_obj if ok else None


def process_messages_file(samples: List[Dict]) -> List[Dict]:
    """
    处理 messages 格式，返回高质量样本列表。
    每个样本: {user_prompt, mcq_json, difficulty, standard, source}
    """
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
        # 尝试从 user 中解析 standard
        standard = extract_standard_from_user(user) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user,
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "messages",
            "score": 1.0,  # messages 无显式分数，默认高质量
        })
    return results


def extract_standard_from_user(user_text: str) -> Optional[str]:
    """从 user prompt 中提取 Standard ID（如 CCSS.ELA-LITERACY.L.3.1.E）"""
    import re
    # 常见模式
    m = re.search(r"(CCSS\.ELA-LITERACY\.[A-Z0-9.-]+)", user_text)
    if m:
        return m.group(1)
    m = re.search(r"Standard ID:\s*([^\s\n]+)", user_text, re.I)
    if m:
        return m.group(1).strip()
    return None


def process_dpo_file(samples: List[Dict]) -> List[Dict]:
    """
    处理 DPO 格式，返回高质量样本。
    chosen_score 越高越优先。
    """
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
        # 拼接 prompt 为 user 文本（取 system+user 最后一条 user）
        user_prompt = ""
        for m in prompt:
            if isinstance(m, dict) and m.get("role") == "user":
                user_prompt = m.get("content", "")
        if not user_prompt:
            user_prompt = " ".join(
                m.get("content", "") for m in prompt if isinstance(m, dict)
            )
        meta = s.get("metadata", {})
        chosen_score = meta.get("chosen_score", 0.5)
        standard = meta.get("standard") or extract_standard_from_user(user_prompt) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        results.append({
            "user_prompt": user_prompt[:2000],  # 截断避免过长
            "mcq_json": mcq,
            "difficulty": difficulty,
            "standard": standard,
            "source": "dpo",
            "score": float(chosen_score),
        })
    return results


def select_diverse_few_shot(
    candidates: List[Dict],
    n: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    从候选中选出多样性好的 few-shot 样本。
    策略：按 score 排序，再按 (standard, difficulty) 去重，保证覆盖不同组合。
    """
    random.seed(seed)
    # 按 score 降序
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
    # 若不足 n 个，补充剩余
    for c in sorted_candidates:
        if len(selected) >= n:
            break
        if c not in selected:
            selected.append(c)
    return selected[:n]


def run(
    input_dir: str = "raw_data",
    output_path: str = "processed_training_data/few_shot_examples.json",
    n_examples: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    主入口：从 raw_data 筛选 few-shot 样本并保存。
    """
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

    selected = select_diverse_few_shot(all_candidates, n=n_examples, seed=seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(selected)} 条 few-shot 样本到: {output_path}")
    return selected


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="筛选 few-shot 高质量样本")
    parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    parser.add_argument("--output", default="processed_training_data/few_shot_examples.json", help="输出 JSON 路径")
    parser.add_argument("-n", "--n-examples", type=int, default=5, help="few-shot 样本数量")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        input_dir=args.input_dir,
        output_path=args.output,
        n_examples=args.n_examples,
        seed=args.seed,
    )
