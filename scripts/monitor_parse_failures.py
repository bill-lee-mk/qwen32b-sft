#!/usr/bin/env python3
"""
监控 parse_failures_*.jsonl，实时报告失败模式。
用法:
  python scripts/monitor_parse_failures.py [--interval 30] [--threshold 5]
"""
import json
import time
import glob
import argparse
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "evaluation_output"


def analyze_failures(path: Path, since_line: int = 0) -> tuple[list[dict], dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < since_line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                pass

    reasons = Counter()
    combos = defaultdict(list)
    for e in entries:
        r = e.get("reason", "unknown")[:100]
        reasons[r] += 1
        key = f"{e.get('standard', '?')}|{e.get('difficulty', '?')}"
        combos[key].append(r)

    repeated = {}
    for key, reason_list in combos.items():
        if len(reason_list) >= 3:
            top_reason = Counter(reason_list).most_common(1)[0]
            repeated[key] = {"count": len(reason_list), "top_reason": top_reason[0], "top_count": top_reason[1]}

    return entries, {
        "total_new": len(entries),
        "reasons": dict(reasons.most_common(10)),
        "repeated_combos": repeated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60, help="检查间隔（秒）")
    parser.add_argument("--threshold", type=int, default=5, help="新增失败数达到此值时报警")
    parser.add_argument("--model", default="or_gemini-3-pro", help="模型标识")
    args = parser.parse_args()

    log_path = EVAL_DIR / f"parse_failures_{args.model}.jsonl"
    if not log_path.exists():
        print(f"日志文件不存在: {log_path}")
        return

    last_line_count = sum(1 for _ in open(log_path))
    last_report_count = last_line_count
    print(f"开始监控: {log_path}")
    print(f"当前已有 {last_line_count} 条记录，从此处开始监控新增")
    print(f"检查间隔: {args.interval}s，报警阈值: {args.threshold} 条新增")
    print("=" * 60)

    while True:
        time.sleep(args.interval)
        try:
            current_count = sum(1 for _ in open(log_path))
        except Exception:
            continue

        new_count = current_count - last_report_count
        if new_count <= 0:
            continue

        if new_count >= args.threshold:
            entries, report = analyze_failures(log_path, since_line=last_report_count)
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}] 新增 {report['total_new']} 条失败记录（总计 {current_count}）")

            if report["reasons"]:
                print("  失败原因:")
                for r, c in report["reasons"].items():
                    print(f"    {c:3d}x  {r[:90]}")

            if report["repeated_combos"]:
                print("  反复失败的组合（同一标准×难度 ≥3 次）:")
                for key, info in report["repeated_combos"].items():
                    print(f"    {key}: {info['count']}次, 主因: {info['top_reason'][:70]}")

            print()
            last_report_count = current_count

        last_line_count = current_count


if __name__ == "__main__":
    main()
