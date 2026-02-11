#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
闭环：从评估结果中识别失败题 → 在 raw_data 中找同 (standard,difficulty) → InceptBench 评分 → 保留 ≥0.85 作 few-shot → 更新 examples.json

用法:
  python main.py improve-examples --results evaluation_output/results_240.json --mcqs evaluation_output/mcqs_240.json --output processed_training_data/examples.json
"""
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_processing.build_prompt import build_user_prompt, get_standard_description
from data_processing.select_examples import (
    load_jsonl,
    process_dpo_file,
    process_messages_file,
)
from evaluation.inceptbench_client import InceptBenchEvaluator, normalize_for_inceptbench


def load_raw_data_by_standard_difficulty(raw_data_dir: str) -> dict:
    """从 raw_data 加载 MCQ，按 (standard, difficulty) 分组"""
    raw_dir = Path(raw_data_dir)
    if not raw_dir.is_absolute():
        raw_dir = PROJECT_ROOT / raw_dir
    if not raw_dir.exists():
        return {}

    by_key = defaultdict(list)
    for fpath in sorted(raw_dir.glob("*.jsonl")):
        samples = load_jsonl(str(fpath))
        if not samples:
            continue
        if "messages" in samples[0]:
            candidates = process_messages_file(samples)
        elif "prompt" in samples[0] and "chosen" in samples[0]:
            candidates = process_dpo_file(samples)
        else:
            continue
        for c in candidates:
            key = (c["standard"], c["difficulty"])
            by_key[key].append(c)
    return dict(by_key)


def _score_from_result(r: dict) -> float | None:
    """从单条评估结果取出分数"""
    if not r:
        return None
    s = r.get("overall_score")
    if s is not None and isinstance(s, (int, float)):
        return float(s)
    for ev in (r.get("evaluations") or {}).values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        s = (inc.get("overall") or {}).get("score")
        if s is not None:
            return float(s)
    return None


def collect_batch_high_scorers(
    results_path: str,
    mcqs_path: str,
    failed: set[tuple[str, str]],
    threshold: float = 0.85,
) -> dict[tuple[str, str], list[tuple[dict, float]]]:
    """
    从本批评估结果中，为每个失败组合找同 (standard,difficulty) 且 score>=threshold 的题目，
    构造成 example 条目（user_prompt + mcq_json），无需再调 InceptBench。
    返回 key -> [(example_dict, score), ...]，每个 key 至多 1 条。
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)
    scores = results.get("scores", [])
    result_list = results.get("results") or []
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    by_key: dict[tuple[str, str], list[tuple[dict, float]]] = defaultdict(list)
    for i in range(len(items)):
        m = items[i]
        std = m.get("standard", "unknown")
        diff = m.get("difficulty", "medium")
        key = (std, diff)
        if key not in failed:
            continue
        s = scores[i] if i < len(scores) else None
        if s is None and i < len(result_list):
            s = _score_from_result(result_list[i])
        if s is None or not isinstance(s, (int, float)) or float(s) < threshold:
            continue
        desc = get_standard_description(std)
        user_prompt = build_user_prompt(grade="3", standard=std, difficulty=diff, standard_description=desc, subject="ELA")
        mcq_norm = normalize_for_inceptbench(m)
        mcq_norm["grade"] = "3"
        mcq_norm["standard"] = std
        mcq_norm["subject"] = "ELA"
        mcq_norm["difficulty"] = diff
        example = {"user_prompt": user_prompt, "mcq_json": mcq_norm}
        by_key[key].append((example, float(s)))
    for key in by_key:
        by_key[key].sort(key=lambda x: -x[1])
        by_key[key] = by_key[key][:1]
    return dict(by_key)


def collect_batch_best_fallback(
    results_path: str,
    mcqs_path: str,
    keys: set[tuple[str, str]],
) -> dict[tuple[str, str], list[tuple[dict, float]]]:
    """
    当 raw_data 与本批均无 ≥0.85 候选时：从本批中取该 (standard,difficulty) 得分最高的题
    构造成一条示例（即使分数 < 0.85），作为「构造示例」补入，避免该组合完全无示例。
    返回 key -> [(example_dict, score), ...]，每个 key 至多 1 条。
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)
    scores = results.get("scores", [])
    result_list = results.get("results") or []
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    by_key: dict[tuple[str, str], list[tuple[dict, float]]] = defaultdict(list)
    for i in range(len(items)):
        m = items[i]
        std = m.get("standard", "unknown")
        diff = m.get("difficulty", "medium")
        key = (std, diff)
        if key not in keys:
            continue
        s = scores[i] if i < len(scores) else None
        if s is None and i < len(result_list):
            s = _score_from_result(result_list[i])
        if s is None:
            s = 0.0
        else:
            s = float(s)
        desc = get_standard_description(std)
        user_prompt = build_user_prompt(grade="3", standard=std, difficulty=diff, standard_description=desc, subject="ELA")
        mcq_norm = normalize_for_inceptbench(m)
        mcq_norm["grade"] = "3"
        mcq_norm["standard"] = std
        mcq_norm["subject"] = "ELA"
        mcq_norm["difficulty"] = diff
        example = {"user_prompt": user_prompt, "mcq_json": mcq_norm}
        by_key[key].append((example, s))
    for key in by_key:
        by_key[key].sort(key=lambda x: -x[1])
        by_key[key] = by_key[key][:1]
    return dict(by_key)


def extract_failed_standard_difficulty(
    results_path: str,
    mcqs_path: str,
    threshold: float = 0.85,
) -> set[tuple[str, str]]:
    """从评估结果中提取得分 < threshold 或 error（无分数）的 (standard, difficulty)"""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    with open(mcqs_path, "r", encoding="utf-8") as f:
        mcqs = json.load(f)

    scores = results.get("scores", [])
    result_list = results.get("results") or []
    items = mcqs if isinstance(mcqs, list) else [mcqs]
    failed = set()
    for i in range(len(items)):
        m = items[i]
        std = m.get("standard", "unknown")
        diff = m.get("difficulty", "medium")
        s = scores[i] if i < len(scores) else None
        if s is None and i < len(result_list):
            s = _score_from_result(result_list[i])
        if s is None or (isinstance(s, (int, float)) and float(s) < threshold):
            failed.add((std, diff))
    return failed


def _is_server_error(r: dict) -> bool:
    """判断是否为服务端错误（可重试）。如 DB 写入失败、HTTP 5xx、超时等"""
    if not r:
        return True
    msg = str(r.get("message", "")).lower()
    errors = r.get("errors") or []
    if isinstance(errors, str):
        errors = [errors]
    err_str = " ".join(str(e).lower() for e in errors)
    body = str(r.get("response_body", "")).lower()
    combined = f"{msg} {err_str} {body}"
    patterns = (
        "could not save", "save evaluation", "db", "database", "500", "503",
        "timeout", "服务端", "server error", "internal",
    )
    return any(p in combined for p in patterns)


def run(
    results_path: str,
    mcqs_path: str,
    raw_data_dir: str = "raw_data",
    examples_output: str = "processed_training_data/examples.json",
    threshold: float = 0.85,
    max_per_pair: int = 1,
    max_candidates_per_pair: int = 0,
    parallel: int = 50,
    timeout: int = 180,
    api_key: str | None = None,
    retry_delay: int = 60,
    max_retries: int = 3,
    failed_output: str | None = "processed_training_data/improve_examples_failed.json",
) -> dict:
    """
    闭环：失败题 (standard,difficulty) → raw_data 候选 → InceptBench 评分 → 保留 ≥0.85 的 1-2 条 → 更新 examples
    服务端错误会记录并延迟重试，超过 max_retries 次后写入 failed_output 供后续处理。
    """
    import os
    api_key = api_key or os.environ.get("INCEPTBENCH_API_KEY") or os.environ.get("INCEPTBENCH_TOKEN")
    if not api_key:
        return {"error": "未设置 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN"}

    failed = extract_failed_standard_difficulty(results_path, mcqs_path, threshold)
    print(f"失败 (standard,difficulty) 组合: {len(failed)} 个")

    by_key = load_raw_data_by_standard_difficulty(raw_data_dir)
    print(f"raw_data 中 (standard,difficulty) 组合: {len(by_key)} 个")

    # 本批高分 fallback：从 results+mcqs 中取同 (std,diff) 且 score>=threshold 的题作为示例，无需再评
    batch_kept = collect_batch_high_scorers(results_path, mcqs_path, failed, threshold)
    kept_by_key: dict[tuple[str, str], list[tuple[dict, float]]] = defaultdict(list)
    for k, lst in batch_kept.items():
        kept_by_key[k].extend(lst)
    if kept_by_key:
        print(f"本批高分补充示例: {len(kept_by_key)} 个组合已从本批题目中取高分题作为示例")

    # 构建每组合的候选列表（仅对尚无示例的失败组合从 raw_data 取候选；提前停止：某组合一经达标即不再评估该组合其余候选）
    # max_candidates_per_pair=0 表示不限制
    max_cand = max_candidates_per_pair
    pending_by_key: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for key in failed:
        if key[0] == "unknown":
            continue
        if key in kept_by_key:
            continue
        candidates = by_key.get(key, [])
        slice_end = len(candidates) if max_cand <= 0 else min(max_cand, len(candidates))
        norm_list = []
        for c in candidates[:slice_end]:
            mcq = c.get("mcq_json", {})
            if not mcq:
                continue
            norm = normalize_for_inceptbench(mcq)
            norm["grade"] = "3"
            norm["standard"] = key[0]
            norm["subject"] = "ELA"
            norm["difficulty"] = key[1]
            norm_list.append((c, norm))
        if norm_list:
            pending_by_key[key] = norm_list

    total_candidates = sum(len(v) for v in pending_by_key.values())
    pair_total_count = {k: len(v) for k, v in pending_by_key.items()}  # 每组合总候选数
    print(f"待评分候选（含提前停止节约）: 最多 {total_candidates} 条，实际按需评估")

    evaluator = InceptBenchEvaluator(api_key=api_key, timeout=timeout)
    scored: list[tuple[tuple[str, str], dict, float]] = []

    def _fmt_key(key):
        std_short = key[0].replace("CCSS.ELA-LITERACY.", "") if key[0] else "?"
        return f"({std_short}, {key[1]})"

    def _extract_score(r):
        s = r.get("overall_score")
        if s is None and "evaluations" in r:
            ev = next(iter(r.get("evaluations", {}).values()), {})
            s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
        return float(s) if isinstance(s, (int, float)) else None

    def _fmt_api_response(r) -> str:
        """score=0 时格式化为简短摘要，默认打印在日志后面"""
        if not r:
            return ""
        parts = []
        if r.get("status"):
            parts.append(f"status={r.get('status')}")
        if r.get("message"):
            parts.append(f"msg={str(r.get('message'))[:100]}")
        if r.get("errors"):
            parts.append(f"errors={r.get('errors')}")
        if "evaluations" in r:
            for k, v in list(r["evaluations"].items())[:1]:
                inc = (v or {}).get("inceptbench_new_evaluation") or {}
                ov = inc.get("overall") or {}
                if ov:
                    score = ov.get("score")
                    reason = (ov.get("internal_reasoning") or ov.get("reasoning") or "")[:80]
                    parts.append(f"score={score}" + (f" reason={reason}..." if reason else ""))
                if inc.get("overall_rating"):
                    parts.append(f"rating={inc.get('overall_rating')}")
        if "response_body" in r and r.get("response_body"):
            parts.append(f"body={str(r.get('response_body'))[:80]}...")
        return " | API: " + " ".join(str(p) for p in parts) if parts else ""

    import threading
    lock = threading.Lock()
    done_total = [0]
    round_num = [0]
    server_error_keys: set[tuple[str, str]] = set()
    failed_records: list[dict] = []
    retry_count = 0

    while pending_by_key:
        round_num[0] += 1
        # 本轮：每个未达标组合取 1 个候选（跳过因服务端错误延后的组合）
        this_round: list[tuple[tuple[str, str], dict, dict]] = []
        to_remove = []
        for key, lst in list(pending_by_key.items()):
            if key in server_error_keys:
                continue
            if not lst:
                to_remove.append(key)
                continue
            c, norm = lst.pop(0)
            this_round.append((key, c, norm))
            if not lst:
                to_remove.append(key)
        for k in to_remove:
            del pending_by_key[k]

        # 若本轮为空但仍有待处理项（全部为服务端错误延后），则等待后重试
        if not this_round and pending_by_key:
            retry_count += 1
            if retry_count > max_retries:
                for key in list(server_error_keys):
                    lst = pending_by_key.get(key, [])
                    err_msg = "服务端错误，已超过最大重试次数"
                    failed_records.append({
                        "standard": key[0],
                        "difficulty": key[1],
                        "retries": retry_count,
                        "remaining_candidates": len(lst),
                    })
                    del pending_by_key[key]
                server_error_keys.clear()
                retry_count = 0
                print(f"\n  已达最大重试次数 {max_retries}，{len(failed_records)} 个组合写入失败记录")
                continue
            print(f"\n  等待 {retry_delay}s 后重试服务端错误组合（第 {retry_count}/{max_retries} 次）...")
            time.sleep(retry_delay)
            server_error_keys.clear()
            continue

        if not this_round:
            break

        n_round = len(this_round)
        print(f"\n  第 {round_num[0]} 轮: {n_round} 个组合各评 1 条 [{parallel} 并行]")

        def _eval(item):
            key, c, norm = item
            t0 = time.time()
            try:
                r = evaluator.evaluate_mcq(norm)
                s = _extract_score(r)
                return (key, c, s, time.time() - t0, r)
            except Exception as e:
                return (key, c, None, time.time() - t0, {"status": "error", "message": str(e)})

        if parallel <= 1:
            for key, c, norm in this_round:
                t0 = time.time()
                r = evaluator.evaluate_mcq(norm)
                s = _extract_score(r)
                elapsed = time.time() - t0
                if s is not None:
                    scored.append((key, c, s))
                    if s >= threshold:
                        kept_by_key[key].append((c, s))
                        if key in pending_by_key:
                            del pending_by_key[key]
                else:
                    if _is_server_error(r):
                        with lock:
                            if key not in pending_by_key:
                                pending_by_key[key] = []
                            pending_by_key[key].insert(0, (c, norm))
                            server_error_keys.add(key)
                with lock:
                    done_total[0] += 1
                    d = done_total[0]
                    status = f"score={s:.2f}" if s is not None else "error"
                    ord_num = round_num[0]
                    total_k = pair_total_count.get(key, 0)
                    ord_str = f" 该组合第{ord_num}/{total_k}" if total_k else ""
                    stop_str = " → 达标，跳过剩余" if s is not None and s >= threshold else ""
                    api_suffix = _fmt_api_response(r) if (s is None or (isinstance(s, (int, float)) and float(s) == 0)) else ""
                    print(f"  [{d}] {status} {_fmt_key(key)}{ord_str}{stop_str} 耗时 {elapsed:.1f}s{api_suffix}")
        else:
            with ThreadPoolExecutor(max_workers=parallel) as ex:
                futures = {ex.submit(_eval, x): x for x in this_round}
                for fut in as_completed(futures):
                    key, c, s, elapsed, r = fut.result()
                    _, _, norm = futures[fut]
                    if s is not None:
                        scored.append((key, c, s))
                        if s >= threshold:
                            with lock:
                                kept_by_key[key].append((c, s))
                                if key in pending_by_key:
                                    del pending_by_key[key]
                    else:
                        # 服务端错误：放回候选，延后重试
                        if _is_server_error(r):
                            with lock:
                                if key not in pending_by_key:
                                    pending_by_key[key] = []
                                pending_by_key[key].insert(0, (c, norm))
                                server_error_keys.add(key)
                    with lock:
                        done_total[0] += 1
                        d = done_total[0]
                        status = f"score={s:.2f}" if s is not None else "error"
                        ord_num = round_num[0]
                        total_k = pair_total_count.get(key, 0)
                        ord_str = f" 该组合第{ord_num}/{total_k}" if total_k else ""
                        stop_str = " → 达标，跳过剩余" if s is not None and s >= threshold else ""
                        api_suffix = _fmt_api_response(r) if (s is None or (isinstance(s, (int, float)) and float(s) == 0)) else ""
                        print(f"  [{d}] {status} {_fmt_key(key)}{ord_str}{stop_str} 耗时 {elapsed:.1f}s{api_suffix}")

    # 每个 (standard,difficulty) 保留 1-2 条，按分数排序后截断
    for key in kept_by_key:
        kept_by_key[key].sort(key=lambda x: -x[1])
        kept_by_key[key] = [c for c, _ in kept_by_key[key][:max_per_pair]]

    kept_count = sum(len(v) for v in kept_by_key.values())
    saved = total_candidates - done_total[0]
    print(f"\n达标 (≥{threshold}) 的示例: {kept_count} 条")
    print(f"实际评估: {done_total[0]} 条（提前停止节约 {saved} 条）")
    no_pass = failed - set(kept_by_key.keys())
    if no_pass:
        # 原始数据与本批均无 ≥threshold 候选时：用本批该组合最高分题构造一条示例（即使 < 0.85）
        constructed = collect_batch_best_fallback(results_path, mcqs_path, no_pass)
        for k, lst in constructed.items():
            kept_by_key[k].extend(lst)
        if constructed:
            print(f"未找到达标候选的组合中，已用本批最高分构造示例: {len(constructed)} 个")
        for k in no_pass - set(constructed.keys()):
            print(f"  组合 {_fmt_key(k)} 本批无题目，无法构造示例")

    # 加载现有 examples，用新达标项替换失败组合的示例
    examples_path = Path(examples_output)
    if not examples_path.is_absolute():
        examples_path = PROJECT_ROOT / examples_path
    existing = []
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    # 新示例：每个 (standard,difficulty) 1-2 条
    new_examples = []
    for key, items in kept_by_key.items():
        for item in items:
            new_examples.append({
                "user_prompt": item["user_prompt"],
                "mcq_json": item["mcq_json"],
                "difficulty": key[1],
                "standard": key[0],
            })

    # 保留原有示例中 (standard,difficulty) 不在 kept_by_key 中的
    replaced_keys = set(kept_by_key.keys())
    for e in existing:
        k = (e.get("standard"), e.get("difficulty"))
        if k not in replaced_keys:
            new_examples.append(e)

    unique = new_examples

    examples_path.parent.mkdir(parents=True, exist_ok=True)
    with open(examples_path, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)
    print(f"已更新 examples: {examples_path} ({len(unique)} 条)")

    if failed_records and failed_output:
        failed_path = Path(failed_output)
        if not failed_path.is_absolute():
            failed_path = PROJECT_ROOT / failed_path
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_records, f, ensure_ascii=False, indent=2)
        print(f"服务端错误超重试的组合已写入: {failed_path} ({len(failed_records)} 个，可稍后重跑本命令重试)")

    return {
        "failed_pairs": len(failed),
        "evaluated": done_total[0],
        "kept": sum(len(v) for v in kept_by_key.values()),
        "examples_count": len(unique),
        "server_error_failed": len(failed_records),
    }
