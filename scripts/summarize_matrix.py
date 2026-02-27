#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总 ELA 模型×年级分数矩阵。

扫描 run_matrix.sh 产出的 results_*.json，提取 pass_rate / avg_score，
打印终端表格 + 保存 JSON。

用法:
  python scripts/summarize_matrix.py                        # 默认目录
  python scripts/summarize_matrix.py --dir evaluation_output/matrix
  python scripts/summarize_matrix.py --metric avg_score     # 用平均分代替通过率
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

GRADE_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

WINNER_ABBR = {
    "fw/deepseek-r1": "R1", "fw/deepseek-v3.2": "V3.2", "fw/kimi-k2.5": "Kimi",
    "fw/glm-5": "GLM5", "fw/gpt-oss-120b": "GPT", "fw/qwen3-235b": "Qwen",
}


def _display_width(s: str) -> int:
    """计算字符串的终端显示宽度（中文/全角字符占 2 列）。"""
    import unicodedata
    w = 0
    for ch in s:
        cat = unicodedata.east_asian_width(ch)
        w += 2 if cat in ("W", "F") else 1
    return w


def _pad(s: str, width: int, align: str = "left") -> str:
    """按显示宽度填充空格，支持 left / right 对齐。"""
    dw = _display_width(s)
    pad = max(0, width - dw)
    return (s + " " * pad) if align == "left" else (" " * pad + s)


def _short_model(name: str) -> str:
    return name.replace("fw_", "fw/")


def scan_results(result_dir: str, subject: str = "ELA"):
    """扫描 results_{model}_{grade}_{subject}.json，返回 {model: {grade: {metric: val}}}"""
    data: dict[str, dict[str, dict]] = {}
    pattern = re.compile(r"^results_(.+?)_([^_]+)_" + re.escape(subject) + r"\.json$")
    for fname in sorted(os.listdir(result_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        model_slug = m.group(1)
        grade = m.group(2)
        fpath = os.path.join(result_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            info = {
                "pass_rate": res.get("pass_rate", 0),
                "avg_score": round(res.get("avg_score", 0) * 100, 1),
                "pass_count": res.get("pass_count", 0),
                "total": res.get("valid_score_count", res.get("total", 0)),
            }
        except Exception:
            info = {"pass_rate": -1, "avg_score": -1, "pass_count": 0, "total": 0}
        model_name = _short_model(model_slug)
        data.setdefault(model_name, {})[grade] = info
    return data


def print_matrix(data: dict, metric: str = "pass_rate", subject: str = "ELA"):
    """打印终端矩阵表格。"""
    if not data:
        print("没有找到结果文件。请先运行 bash scripts/run_matrix.sh")
        return

    grades = [g for g in GRADE_ORDER if any(g in v for v in data.values())]
    models = sorted(data.keys())

    col_w = 9
    model_w = max(len(m) for m in models) + 2
    metric_label = "pass_rate %" if metric == "pass_rate" else "avg_score %"

    header = _pad("Model", model_w) + "".join(f"{g:>{col_w}}" for g in grades) + f"  {'AVG':>6}"
    total_w = _display_width(header)
    sep = "─" * total_w

    print()
    print(f"  {subject} 分数矩阵 ({metric_label})")
    print(f"  {sep}")
    print(f"  {header}")
    print(f"  {sep}")

    model_avgs = {}
    grade_bests: dict[str, tuple[float, str]] = {}

    for model in models:
        row = "  " + _pad(model, model_w)
        vals = []
        for g in grades:
            info = data[model].get(g)
            if info is None:
                row += f"{'--':>{col_w}}"
            else:
                v = info[metric]
                if v < 0:
                    row += f"{'ERR':>{col_w}}"
                else:
                    row += f"{v:>{col_w}.1f}"
                    vals.append(v)
                    best_val, _ = grade_bests.get(g, (-1, ""))
                    if v > best_val:
                        grade_bests[g] = (v, model)
        avg = sum(vals) / len(vals) if vals else 0
        model_avgs[model] = avg
        row += f"  {avg:>6.1f}"
        print(row)

    print(f"  {sep}")

    best_label = "BEST"
    winner_label = "WINNER"
    best_row = f"  {best_label:<{model_w}}"
    winner_row = f"  {winner_label:<{model_w}}"
    for g in grades:
        if g in grade_bests:
            bv, bm = grade_bests[g]
            best_row += f"{bv:>{col_w}.1f}"
            short = WINNER_ABBR.get(bm, bm.split("/")[-1][:col_w])
            winner_row += f"{short:>{col_w}}"
        else:
            best_row += f"{'--':>{col_w}}"
            winner_row += f"{'--':>{col_w}}"
    print(best_row)
    print(winner_row)
    print(f"  {sep}")

    print()
    print("  模型平均分排名:")
    for rank, (model, avg) in enumerate(sorted(model_avgs.items(), key=lambda x: -x[1]), 1):
        print(f"    {rank}. {model:<{model_w}} {avg:.1f}%")
    print()


def save_matrix_json(data: dict, output: str, metric: str = "pass_rate"):
    """保存矩阵数据为 JSON。"""
    grades = [g for g in GRADE_ORDER if any(g in v for v in data.values())]
    models = sorted(data.keys())
    matrix = {}
    for model in models:
        row = {}
        for g in grades:
            info = data[model].get(g)
            row[g] = info[metric] if info and info[metric] >= 0 else None
        vals = [v for v in row.values() if v is not None]
        row["avg"] = round(sum(vals) / len(vals), 1) if vals else None
        matrix[model] = row

    out = {"metric": metric, "grades": grades, "models": models, "matrix": matrix}
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  矩阵已保存: {output}")


def main():
    p = argparse.ArgumentParser(description="汇总 ELA 模型×年级分数矩阵")
    p.add_argument("--dir", default="evaluation_output/matrix", help="结果文件目录")
    p.add_argument("--subject", default="ELA", help="学科")
    p.add_argument("--metric", default="pass_rate", choices=["pass_rate", "avg_score"], help="展示指标")
    p.add_argument("--output", default=None, help="矩阵 JSON 输出路径（默认 <dir>/score_matrix.json）")
    args = p.parse_args()

    if not os.path.isdir(args.dir):
        print(f"错误: 目录不存在: {args.dir}")
        print("请先运行: bash scripts/run_matrix.sh")
        sys.exit(1)

    data = scan_results(args.dir, args.subject)
    print_matrix(data, args.metric, args.subject)

    out_path = args.output or os.path.join(args.dir, "score_matrix.json")
    save_matrix_json(data, out_path, args.metric)


if __name__ == "__main__":
    main()
