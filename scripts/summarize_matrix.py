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
    "fw/deepseek-r1": "R1", "fw/deepseek-v3.2": "V3.2", "fw/deepseek-v3_2": "DSv3",
    "fw/kimi-k2.5": "Kimi", "fw/kimi-k2_5": "Kimi",
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
    """扫描结果文件，返回 {model: {grade: {metric: val}}}

    支持两种文件名格式：
    1. matrix 目录: results_{model}_{grade}_{subject}.json
    2. best 文件:   results_{grade}_{subject}_{model}_best_{score}.json
    """
    data: dict[str, dict[str, dict]] = {}
    # 格式1: results_{model}_{grade}_{subject}.json (matrix 目录输出)
    pat1 = re.compile(r"^results_(.+?)_([^_]+)_" + re.escape(subject) + r"\.json$")
    # 格式2: results_{grade}_{subject}_{model}_best_{score}.json (闭环 best 输出)
    pat2 = re.compile(
        r"^results_(\d+)_" + re.escape(subject) + r"_(.+?)_best_[\d_]+\.json$"
    )
    for fname in sorted(os.listdir(result_dir)):
        m1 = pat1.match(fname)
        m2 = pat2.match(fname)
        if m1:
            model_slug, grade = m1.group(1), m1.group(2)
            model_slug = re.sub(r"_matrix$", "", model_slug)
        elif m2:
            grade, model_slug = m2.group(1), m2.group(2)
            model_slug = re.sub(r"_matrix$", "", model_slug)
        else:
            continue
        fpath = os.path.join(result_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            breakdown = res.get("breakdown", {})
            if not breakdown:
                mcqs_fname = fname.replace("results_", "mcqs_")
                mcqs_fpath = os.path.join(result_dir, mcqs_fname)
                breakdown = _compute_breakdown_from_result(res, mcqs_fpath)
            info = {
                "pass_rate": res.get("pass_rate", 0),
                "avg_score": round(res.get("avg_score", 0) * 100, 1),
                "pass_count": res.get("pass_count", 0),
                "total": res.get("valid_score_count", res.get("total", 0)),
                "breakdown": breakdown,
            }
        except Exception:
            info = {"pass_rate": -1, "avg_score": -1, "pass_count": 0, "total": 0, "breakdown": {}}
        model_name = _short_model(model_slug)
        existing = data.get(model_name, {}).get(grade)
        if existing and existing.get("pass_rate", -1) >= info["pass_rate"]:
            continue
        data.setdefault(model_name, {})[grade] = info
    return data


def _compute_breakdown_from_result(res, mcqs_path=None):
    """从旧格式结果文件（无 breakdown 字段）中补算拆解。
    如果 mcqs_path 存在，从中获取题型信息。
    """
    details = res.get("evaluation_details", [])
    scores = res.get("scores", [])
    if not details and not scores:
        return {}

    mcq_items = []
    if mcqs_path and os.path.exists(mcqs_path):
        try:
            with open(mcqs_path, "r", encoding="utf-8") as f:
                mcq_items = json.load(f)
        except Exception:
            pass

    from collections import defaultdict
    buckets_type = defaultdict(list)
    buckets_diff = defaultdict(list)
    buckets_td = defaultdict(list)

    if details:
        for i, d in enumerate(details):
            s = d.get("score")
            if s is None:
                continue
            qtype = d.get("type")
            if not qtype and i < len(mcq_items):
                qtype = mcq_items[i].get("type", "mcq")
            qtype = qtype or "mcq"
            diff = d.get("difficulty", "medium")
            buckets_type[qtype].append(s)
            buckets_diff[diff].append(s)
            buckets_td[f"{qtype}|{diff}"].append(s)
    elif scores and mcq_items:
        for i, s in enumerate(scores):
            if s is None or not isinstance(s, (int, float)):
                continue
            q = mcq_items[i] if i < len(mcq_items) else {}
            qtype = q.get("type", "mcq")
            diff = q.get("difficulty", "medium")
            buckets_type[qtype].append(s)
            buckets_diff[diff].append(s)
            buckets_td[f"{qtype}|{diff}"].append(s)

    def _summarize(bucket):
        out = {}
        for key in sorted(bucket.keys()):
            vals = bucket[key]
            n = len(vals)
            passed = sum(1 for v in vals if v >= 0.85)
            out[key] = {
                "count": n,
                "pass_count": passed,
                "pass_rate": round(100 * passed / n, 1) if n else 0,
                "avg_score": round(sum(vals) / n, 4) if n else 0,
            }
        return out

    return {
        "by_type": _summarize(buckets_type),
        "by_difficulty": _summarize(buckets_diff),
        "by_type_difficulty": _summarize(buckets_td),
    }


def print_unified_matrix(data: dict, metric: str = "pass_rate", subject: str = "ELA"):
    """打印嵌套式矩阵：按题型分组，每组内显示模型行+难度子行，底部 ★Best 标注冠军。"""
    if not data:
        print("没有找到结果文件。请先运行 bash scripts/run_matrix.sh")
        return

    USE_COLOR = sys.stdout.isatty()

    def C(text, code):
        if not USE_COLOR:
            return text
        return f"\033[{code}m{text}\033[0m"

    def fmt_val(v):
        if v is None:
            return "   --"
        s = f"{v:5.1f}"
        if not USE_COLOR:
            return s
        if v >= 95:
            return C(s, "92")
        if v < 75:
            return C(s, "91")
        if v < 85:
            return C(s, "93")
        return s

    grades = [g for g in GRADE_ORDER if any(g in v for v in data.values())]
    models = sorted(data.keys())
    if not grades or not models:
        print("没有可用数据。")
        return

    MW = max((len(m) for m in models), default=8) + 2
    MW = max(MW, 14)
    CW = 5
    DIFF_ORDER = ["easy", "medium", "hard"]

    TYPE_SECTIONS = [
        ("MCQ", "mcq"),
        ("MSQ", "msq"),
        ("Fill-in", "fill-in"),
        ("Overall", None),
    ]

    def get_val(model, grade, ft=None, fd=None):
        info = data[model].get(grade)
        if info is None:
            return None
        if ft is None and fd is None:
            return info.get(metric)
        bd = info.get("breakdown", {})
        if ft is not None and fd is not None:
            key = f"{ft}|{fd}"
            btd = bd.get("by_type_difficulty", {}).get(key)
            if btd:
                return btd.get("pass_rate")
            bt = bd.get("by_type", {}).get(ft)
            bf = bd.get("by_difficulty", {}).get(fd)
            if bt and bf:
                return None
            return None
        if ft is not None:
            bt = bd.get("by_type", {}).get(ft)
            return bt.get("pass_rate") if bt else None
        if fd is not None:
            bf = bd.get("by_difficulty", {}).get(fd)
            return bf.get("pass_rate") if bf else None
        return None

    def avg_val(model, ft=None, fd=None):
        vals = [get_val(model, g, ft, fd) for g in grades]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 1) if vals else None

    cols = grades + ["AVG"]
    table_w = MW + 1 + len(cols) * (CW + 1)

    print()
    print(C(f"  {subject} Pass Rate (%) 嵌套矩阵", "1"))
    print(f'  {"═" * table_w}')
    ch = " " * (MW + 1)
    for c in cols:
        lbl = f"G{c}" if c != "AVG" else "AVG"
        ch += f"{lbl:>{CW}} "
    print(C(f"  {ch}", "2"))
    print(f'  {"═" * table_w}')

    for si, (slabel, ft) in enumerate(TYPE_SECTIONS):
        if si > 0:
            print(f'  {"─" * table_w}')
        print(C(f"  {slabel}", "96;1"))

        for model in models:
            line = f"  {_pad(model, MW, 'right')} "
            for c in cols:
                v = avg_val(model, ft) if c == "AVG" else get_val(model, c, ft)
                line += fmt_val(v) + " "
            print(line)

            diff_ft = ft  # ft=None 时表示 Overall，取总难度
            for di, diff in enumerate(DIFF_ORDER):
                prefix = "└" if di == len(DIFF_ORDER) - 1 else "├"
                dlabel = f"{prefix}{diff[0].upper()}"
                dline = f"  {_pad(dlabel, MW, 'right')} "
                for c in cols:
                    v = avg_val(model, diff_ft, diff) if c == "AVG" else get_val(model, c, diff_ft, diff)
                    dline += fmt_val(v) + " "
                print(C(dline, "2"))

        def _best_row(label, val_ft, val_fd):
            """打印一行 ★Best / 难度 Best，val_ft/val_fd 指定取值维度。"""
            row = "  "
            if USE_COLOR:
                row += f"\033[95m{_pad(label, MW, 'right')}\033[0m "
            else:
                row += f"{_pad(label, MW, 'right')} "
            for c in cols:
                best_v = -1
                best_m = None
                for model in models:
                    v = avg_val(model, val_ft, val_fd) if c == "AVG" else get_val(model, c, val_ft, val_fd)
                    if v is not None and v > best_v:
                        best_v = v
                        best_m = model
                if best_m:
                    abbr = WINNER_ABBR.get(best_m, best_m.split("/")[-1][:CW])
                    cell = f"{abbr:>{CW}}"
                    if USE_COLOR:
                        row += f"\033[95m{cell}\033[0m "
                    else:
                        row += cell + " "
                else:
                    row += f"{'--':>{CW}} "
            print(row)

        _best_row("★Best", ft, None)
        for di, diff in enumerate(DIFF_ORDER):
            prefix = "└" if di == len(DIFF_ORDER) - 1 else "├"
            _best_row(f"{prefix}★{diff[0].upper()}", ft, diff)

    print(f'  {"═" * table_w}')

    # 模型排名
    print()
    print("  模型平均分排名 (Overall):")
    model_avgs = {}
    for model in models:
        avg = avg_val(model, None, None)
        model_avgs[model] = avg if avg is not None else 0
    for rank, (model, avg) in enumerate(
        sorted(model_avgs.items(), key=lambda x: -x[1]), 1
    ):
        print(f"    {rank}. {model:<{MW}} {avg:.1f}%")

    # 图例
    print()
    if USE_COLOR:
        print(
            f"  颜色: \033[92m≥95\033[0m  ≥85  \033[93m75-84\033[0m  \033[91m<75\033[0m   -- 无数据"
        )
    else:
        print("  标记: ≥95优秀  ≥85良好  75-84一般  <75需改进  -- 无数据")
    print("  ★Best = 该列最高分模型")
    abbr_items = []
    for m in models:
        a = WINNER_ABBR.get(m, m.split("/")[-1][:CW])
        abbr_items.append(f"{a}={m}")
    if abbr_items:
        print(f'  缩写: {", ".join(abbr_items)}')
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

    # 按题型拆解矩阵
    type_matrix = {}
    diff_matrix = {}
    for model in models:
        for g in grades:
            bd = data[model].get(g, {}).get("breakdown", {})
            for t, info in bd.get("by_type", {}).items():
                type_matrix.setdefault(t, {}).setdefault(model, {})[g] = info
            for d, info in bd.get("by_difficulty", {}).items():
                diff_matrix.setdefault(d, {}).setdefault(model, {})[g] = info

    out = {
        "metric": metric,
        "grades": grades,
        "models": models,
        "matrix": matrix,
        "by_type": type_matrix,
        "by_difficulty": diff_matrix,
    }
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
    print_unified_matrix(data, args.metric, args.subject)

    out_path = args.output or os.path.join(args.dir, "score_matrix.json")
    save_matrix_json(data, out_path, args.metric)


if __name__ == "__main__":
    main()
