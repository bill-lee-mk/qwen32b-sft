#!/usr/bin/env python3
"""实时监控题目生成的日志和错误。

功能:
  1. 增量扫描 evaluation_output/parse_failures_*.jsonl 中的新错误
  2. 扫描所有终端日志文件中的错误模式
  3. 自动分类统计、发出告警
  4. 生成修复建议

用法:
  python scripts/monitor_logs.py                # 单次扫描当前状态
  python scripts/monitor_logs.py --watch        # 持续监控（每30秒刷新）
  python scripts/monitor_logs.py --watch -i 10  # 每10秒刷新
  python scripts/monitor_logs.py --fix          # 扫描 + 输出修复建议
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation_output"
TERMINALS_DIR = Path(
    os.environ.get(
        "CURSOR_TERMINALS",
        Path.home() / ".cursor/projects/home-ubuntu-lilei-projects-qwen32b-sft/terminals",
    )
)

ERROR_PATTERNS = [
    (r"\[解析失败\]\s*(.+?)(?:\s*\(第\d+次)", "parse_fail", "解析失败"),
    (r"429|Too Many Requests|rate.?limit", "rate_limit", "限流 429"),
    (r"timed?\s*out|timeout|deadline", "timeout", "超时"),
    (r"500|502|503|internal.?server", "server_error", "服务端错误"),
    (r"content.?filter|content.?management|moderation", "content_filter", "内容过滤"),
    (r"connection.?(?:reset|refused|error)", "conn_error", "连接错误"),
    (r"no healthy upstream", "upstream", "上游不可用"),
    (r"missing.?fields?.*answer_explanation", "missing_explanation", "缺少 answer_explanation"),
    (r"answer_options keys must be", "bad_options", "选项格式错误"),
]

KNOWN_FIXES = {
    "missing_explanation": "已在 _normalize_parsed_mcq 中添加 explanation→answer_explanation 别名映射，无需额外修复。",
    "bad_options": "已在 _normalize_parsed_mcq 中添加选项 key 规范化（含数字→字母映射），此错误表示模型真的只给了<4个选项，重试可解决。",
    "rate_limit": "降低 workers 数或增加 API 调用间隔。检查 RPM 限制。",
    "timeout": "可能是模型推理较慢（如 Grok-4, Gemini 3 Pro），增大超时时间或换用更快的模型。",
    "content_filter": "模型拒绝生成，可能是误触发内容审核。调整 prompt 或换模型。",
    "parse_fail": "检查 parse_failures_*.jsonl 中的 raw_preview 字段，确认模型输出是否被截断或含有非 JSON 内容。",
}


def scan_parse_failure_logs() -> list[dict]:
    """读取所有 parse_failures_*.jsonl 文件。"""
    entries = []
    for f in sorted(EVAL_DIR.glob("parse_failures_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def scan_terminal_logs() -> dict[str, list[dict]]:
    """扫描终端日志文件，提取错误行。"""
    results: dict[str, list[dict]] = {}
    if not TERMINALS_DIR.exists():
        return results
    for f in sorted(TERMINALS_DIR.glob("*.txt")):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        errors = []
        for line_no, line in enumerate(content.splitlines(), 1):
            for pattern, err_type, label in ERROR_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    errors.append({
                        "line": line_no,
                        "type": err_type,
                        "label": label,
                        "text": line.strip()[:200],
                    })
                    break
        if errors:
            results[f.name] = errors
    return results


def classify_failures(entries: list[dict]) -> dict:
    """对解析失败进行分类统计。"""
    by_reason = Counter()
    by_model = Counter()
    by_difficulty = Counter()
    by_standard = Counter()
    recent = []
    for e in entries:
        reason = e.get("reason", "unknown")
        by_reason[reason] += 1
        by_model[e.get("model", "?")] += 1
        by_difficulty[e.get("difficulty", "?")] += 1
        by_standard[e.get("standard", "?")] += 1
    recent = entries[-10:] if entries else []
    return {
        "total": len(entries),
        "by_reason": dict(by_reason.most_common(20)),
        "by_model": dict(by_model.most_common(20)),
        "by_difficulty": dict(by_difficulty.most_common()),
        "recent": recent,
    }


def classify_terminal_errors(terminal_errors: dict[str, list[dict]]) -> dict:
    """汇总终端错误。"""
    type_counts = Counter()
    for fname, errors in terminal_errors.items():
        for e in errors:
            type_counts[e["label"]] += 1
    return {
        "total": sum(type_counts.values()),
        "by_type": dict(type_counts.most_common()),
        "terminals": {k: len(v) for k, v in terminal_errors.items()},
    }


def suggest_fixes(failure_stats: dict, terminal_stats: dict) -> list[str]:
    """根据错误分布生成修复建议。"""
    suggestions = []
    reasons = failure_stats.get("by_reason", {})
    for reason_text, count in reasons.items():
        matched = False
        for key, fix in KNOWN_FIXES.items():
            if key in reason_text.lower().replace(" ", "_"):
                suggestions.append(f"[{count}次] {reason_text}: {fix}")
                matched = True
                break
        if not matched and count >= 3:
            suggestions.append(f"[{count}次] {reason_text}: 需要进一步分析 raw_preview 确定根因。")

    term_types = terminal_stats.get("by_type", {})
    for label, count in term_types.items():
        for key, fix in KNOWN_FIXES.items():
            if key.replace("_", " ") in label.lower() or label.lower() in KNOWN_FIXES:
                suggestions.append(f"[终端-{count}次] {label}: {fix}")
                break
    return suggestions


def print_report(failure_stats: dict, terminal_stats: dict, fixes: list[str], watch_mode: bool = False):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    w = 70
    print(f"\n{'=' * w}")
    if watch_mode:
        print(f"  日志监控报告  {ts}  (Ctrl+C 退出)")
    else:
        print(f"  日志监控报告  {ts}")
    print(f"{'=' * w}")

    total_f = failure_stats.get("total", 0)
    print(f"\n📋 解析失败日志 (parse_failures_*.jsonl): {total_f} 条")
    if total_f > 0:
        print("\n  按原因:")
        for reason, cnt in failure_stats.get("by_reason", {}).items():
            pct = cnt / total_f * 100
            bar = "█" * int(pct / 3)
            print(f"    {reason:<45} {cnt:>4} ({pct:>5.1f}%) {bar}")
        print("\n  按模型:")
        for model, cnt in failure_stats.get("by_model", {}).items():
            print(f"    {model:<45} {cnt:>4}")
        print("\n  按难度:")
        for diff, cnt in failure_stats.get("by_difficulty", {}).items():
            print(f"    {diff:<10} {cnt:>4}")

        print("\n  最近10条:")
        for e in failure_stats.get("recent", []):
            preview = (e.get("raw_preview", "") or "")[:80].replace("\n", "\\n")
            print(f"    [{e.get('ts','')}] {e.get('model','?')} | {e.get('standard','?')} {e.get('difficulty','?')} | {e.get('reason','?')}")
            if preview:
                print(f"      raw: {preview}")

    total_t = terminal_stats.get("total", 0)
    print(f"\n📺 终端日志错误: {total_t} 条")
    if total_t > 0:
        for label, cnt in terminal_stats.get("by_type", {}).items():
            print(f"    {label:<30} {cnt:>4}")
        print("  终端分布:")
        for fname, cnt in terminal_stats.get("terminals", {}).items():
            print(f"    {fname:<20} {cnt:>4} 条错误")

    if fixes:
        print(f"\n🔧 修复建议:")
        for i, fix in enumerate(fixes, 1):
            print(f"  {i}. {fix}")

    if total_f == 0 and total_t == 0:
        print("\n  ✓ 当前无异常，一切正常。")

    print(f"\n{'=' * w}\n")


def main():
    parser = argparse.ArgumentParser(description="实时日志监控")
    parser.add_argument("--watch", action="store_true", help="持续监控模式")
    parser.add_argument("-i", "--interval", type=int, default=30, help="监控间隔（秒）")
    parser.add_argument("--fix", action="store_true", help="输出修复建议")
    args = parser.parse_args()

    if args.watch:
        print(f"开始持续监控，每 {args.interval} 秒刷新... (Ctrl+C 退出)")
        try:
            while True:
                entries = scan_parse_failure_logs()
                terminal_errors = scan_terminal_logs()
                fs = classify_failures(entries)
                ts = classify_terminal_errors(terminal_errors)
                fixes = suggest_fixes(fs, ts)
                os.system("clear" if os.name != "nt" else "cls")
                print_report(fs, ts, fixes, watch_mode=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n监控已停止。")
    else:
        entries = scan_parse_failure_logs()
        terminal_errors = scan_terminal_logs()
        fs = classify_failures(entries)
        ts = classify_terminal_errors(terminal_errors)
        fixes = suggest_fixes(fs, ts) if args.fix else []
        print_report(fs, ts, fixes)


if __name__ == "__main__":
    main()
