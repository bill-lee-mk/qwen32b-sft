# -*- coding: utf-8 -*-
"""
分析 raw_data 中的维度分布（难度、学科、标准等）

用于 diverse 生成时确定要覆盖的 (standard, difficulty) 组合。
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .select_examples import (
    extract_json_from_text,
    extract_standard_from_user,
    is_valid_mcq,
    load_jsonl,
)


def _extract_mcq_and_meta(sample: Dict) -> Optional[Tuple[Dict, str, str, Optional[str]]]:
    """从单条样本提取 (mcq, standard, difficulty, subject)"""
    mcq = None
    standard = "unknown"
    difficulty = "medium"
    subject = "ELA"

    if "messages" in sample:
        msgs = sample.get("messages", [])
        if len(msgs) < 3:
            return None
        user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                user = m.get("content", "")
                break
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                mcq = extract_json_from_text(content)
                if mcq and is_valid_mcq(mcq)[0]:
                    standard = extract_standard_from_user(user) or "unknown"
                    difficulty = mcq.get("difficulty", "medium")
                    return (mcq, standard, difficulty, subject)
                break
        return None

    if "prompt" in sample and "chosen" in sample:
        chosen = sample.get("chosen", {})
        if isinstance(chosen, dict):
            content = chosen.get("content", "")
        else:
            content = str(chosen)
        mcq = extract_json_from_text(content)
        if not mcq or not is_valid_mcq(mcq)[0]:
            return None
        meta = sample.get("metadata", {})
        standard = meta.get("standard") or "unknown"
        user = ""
        for m in sample.get("prompt", []):
            if isinstance(m, dict) and m.get("role") == "user":
                user = m.get("content", "")
                break
        if standard == "unknown":
            standard = extract_standard_from_user(user) or "unknown"
        difficulty = mcq.get("difficulty", "medium")
        return (mcq, standard, difficulty, subject)

    return None


def analyze_dimensions(
    input_dir: str = "raw_data",
    max_per_file: int = 50000,
) -> Dict:
    """
    分析 raw_data 中的维度分布。

    返回:
        {
            "difficulties": {"easy": 100, "medium": 200, "hard": 80},
            "standards": {"CCSS.ELA-LITERACY.L.3.1.E": 50, ...},
            "subjects": {"ELA": 380},
            "combinations": [("L.3.1.E", "easy"), ("L.3.1.E", "medium"), ...],
            "total": 380,
        }
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        return {
            "difficulties": {"easy": 0, "medium": 0, "hard": 0},
            "standards": {},
            "subjects": {"ELA": 0},
            "combinations": [],
            "total": 0,
        }

    difficulties = Counter()
    standards = Counter()
    subjects = Counter()
    combinations = set()

    for fpath in sorted(input_path.glob("*.jsonl")):
        samples = load_jsonl(str(fpath), max_lines=max_per_file)
        if not samples:
            continue
        for s in samples:
            out = _extract_mcq_and_meta(s)
            if not out:
                continue
            _, standard, difficulty, subject = out
            difficulties[difficulty] += 1
            standards[standard] += 1
            subjects[subject] += 1
            combinations.add((standard, difficulty))

    return {
        "difficulties": dict(difficulties),
        "standards": dict(standards),
        "subjects": dict(subjects),
        "combinations": sorted(combinations),
        "total": sum(difficulties.values()),
    }


def build_diverse_plan(
    dimensions: Dict,
    n: int = 20,
    seed: int = 42,
    all_combinations: bool = False,
) -> List[Tuple[str, str]]:
    """
    根据维度分布构建生成计划，返回 [(standard, difficulty), ...]。

    all_combinations=True 时返回全部组合；否则取 n 条并尽量在标准、难度上均匀分布。
    """
    import random

    combos = dimensions.get("combinations", [])
    combos = [(s, d) for s, d in combos if s != "unknown"]
    if not combos:
        # 无 raw_data 时使用默认组合
        defaults = [
            ("CCSS.ELA-LITERACY.L.3.1.E", "easy"),
            ("CCSS.ELA-LITERACY.L.3.1.E", "medium"),
            ("CCSS.ELA-LITERACY.L.3.1.E", "hard"),
            ("CCSS.ELA-LITERACY.L.3.1.F", "easy"),
            ("CCSS.ELA-LITERACY.L.3.1.F", "medium"),
            ("CCSS.ELA-LITERACY.L.3.1.F", "hard"),
            ("CCSS.ELA-LITERACY.SL.3.1.A", "easy"),
            ("CCSS.ELA-LITERACY.SL.3.1.A", "medium"),
            ("CCSS.ELA-LITERACY.SL.3.1.A", "hard"),
            ("CCSS.ELA-LITERACY.SL.3.6", "easy"),
            ("CCSS.ELA-LITERACY.SL.3.6", "medium"),
            ("CCSS.ELA-LITERACY.SL.3.6", "hard"),
            ("CCSS.ELA-LITERACY.L.3.2.A", "medium"),
            ("CCSS.ELA-LITERACY.L.3.2.B", "medium"),
            ("CCSS.ELA-LITERACY.RF.3.3.A", "medium"),
            ("CCSS.ELA-LITERACY.RF.3.3.B", "medium"),
            ("CCSS.ELA-LITERACY.RI.3.1", "medium"),
            ("CCSS.ELA-LITERACY.RL.3.1", "medium"),
            ("CCSS.ELA-LITERACY.W.3.1.A", "medium"),
            ("CCSS.ELA-LITERACY.L.3.4.A", "medium"),
        ]
        combos = defaults

    if all_combinations or (n and n >= len(combos)):
        return combos[:n] if n else combos

    # 按 (difficulty, standard) 分组，轮询取以增加多样性
    by_diff = defaultdict(list)
    for s, d in combos:
        if s != "unknown":
            by_diff[d].append((s, d))
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            random.seed(seed)
            random.shuffle(by_diff[d])

    plan = []
    seen = set()
    round_idx = 0
    max_rounds = max(n, 100)
    for _ in range(max_rounds):
        if len(plan) >= n:
            break
        for d in ["easy", "medium", "hard"]:
            if len(plan) >= n:
                break
            lst = by_diff.get(d, [])
            if not lst:
                continue
            s, d_val = lst[round_idx % len(lst)]
            if (s, d_val) not in seen:
                seen.add((s, d_val))
                plan.append((s, d_val))
        round_idx += 1

    # 不足 n 时从 combos 补充
    for s, d in combos:
        if len(plan) >= n:
            break
        if (s, d) not in seen and s != "unknown":
            seen.add((s, d))
            plan.append((s, d))

    return plan[:n]


def run(
    input_dir: str = "raw_data",
    output: Optional[str] = None,
) -> Dict:
    """分析维度并打印/保存"""
    dims = analyze_dimensions(input_dir=input_dir)
    print(f"\n=== raw_data 维度统计 ===")
    print(f"总样本数: {dims['total']}")
    print(f"难度分布: {dims['difficulties']}")
    print(f"学科分布: {dims['subjects']}")
    print(f"标准数量: {len(dims['standards'])}")
    if dims["standards"]:
        top = sorted(dims["standards"].items(), key=lambda x: -x[1])[:10]
        print(f"  前10: {dict(top)}")
    print(f"(standard, difficulty) 组合数: {len(dims['combinations'])}")
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dims, f, ensure_ascii=False, indent=2)
        print(f"\n已保存: {out_path}")
    return dims
