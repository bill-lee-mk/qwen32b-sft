#!/usr/bin/env python3
"""
quick_validate.py — 快速验证 prompt 规则对低分题的效果

只针对 best 中的低分题（精确到每一题的 standard×difficulty×type），
执行：找弱项 → 仅生成这些题 → 仅评分这些题 → 对比新旧分数

Usage:
  # 查看低分题（不花钱）
  python scripts/quick_validate.py --grade 2 --model or/gemini-3-pro --dry-run

  # 快速验证（仅生成+评估低分题）
  python scripts/quick_validate.py --grade 2 --model or/gemini-3-pro

  # 自定义阈值
  python scripts/quick_validate.py --grade 7 --model or/gemini-3-pro --threshold 90
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _find_best_files(grade, model, run_id=None):
    model_slug = model.replace(".", "_").replace("/", "_")
    suffix = f"_{run_id}" if run_id else ""
    pattern = os.path.join(
        PROJECT_ROOT, "evaluation_output",
        f"mcqs_{grade}_ELA_{model_slug}{suffix}_best_*.json"
    )
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return None, None
    best_mcqs = matches[0]
    best_results = best_mcqs.replace("mcqs_", "results_")
    if not os.path.exists(best_results):
        return best_mcqs, None
    return best_mcqs, best_results


def _find_prompt_rules(grade, model, run_id=None):
    model_slug = model.replace(".", "_").replace("/", "_")
    suffix = f"_{run_id}" if run_id else ""
    return os.path.join(
        PROJECT_ROOT, "processed_training_data",
        f"{grade}_ELA_prompt_rules_{model_slug}{suffix}.json"
    )


def _find_examples(grade, model, run_id=None):
    model_slug = model.replace(".", "_").replace("/", "_")
    suffix = f"_{run_id}" if run_id else ""
    return os.path.join(
        PROJECT_ROOT, "processed_training_data",
        f"{grade}_ELA_examples_{model_slug}{suffix}.json"
    )


def _load_weak_items(mcqs_path, results_path, threshold=0.85):
    with open(mcqs_path) as f:
        mcqs = json.load(f)
    with open(results_path) as f:
        res = json.load(f)
    results = res.get("results", [])
    weak = []
    for i, (r, q) in enumerate(zip(results, mcqs)):
        score = r.get("overall_score", 1.0)
        if score < threshold:
            feedback = r.get("feedback", r.get("message", ""))
            weak.append((i, score, q, str(feedback)[:200]))
    return weak, mcqs, res


def _print_analysis(weak_items):
    if not weak_items:
        print("  没有低于阈值的题目！")
        return

    print(f"\n{'='*70}")
    print(f"  低分题共 {len(weak_items)} 题")
    print(f"{'='*70}")

    by_type = Counter()
    by_diff = Counter()
    by_td = Counter()
    for _, score, q, _ in weak_items:
        t = q.get("type", "mcq")
        d = q.get("difficulty", "medium")
        by_type[t] += 1
        by_diff[d] += 1
        by_td[f"{t}×{d}"] += 1

    print("\n  按题型:")
    for t, c in by_type.most_common():
        print(f"    {t:10s}: {c} 题")
    print("  按难度:")
    for d, c in by_diff.most_common():
        print(f"    {d:10s}: {c} 题")

    print(f"\n  {'分数':>5s} | {'题型':7s} | {'难度':6s} | {'标准'}")
    print(f"  {'─'*60}")
    for _, score, q, fb in sorted(weak_items, key=lambda x: x[1]):
        std = q.get("standard", "?").replace("CCSS.ELA-LITERACY.", "")
        print(f"  {score:.2f} | {q.get('type','?'):7s} | {q.get('difficulty','?'):6s} | {std}")


def _generate_targeted(weak_items, grade, model, run_id, workers):
    """只生成低分题对应的 (standard, difficulty, type) 组合。"""
    from scripts.generate_questions import (
        _generate_one, _model_to_provider, _get_api_key_for_model,
        _validate_and_repair_keep_all,
    )

    provider = _model_to_provider(model)
    api_key = _get_api_key_for_model(model)

    examples_path = _find_examples(grade, model, run_id)
    examples = []
    if os.path.exists(examples_path):
        with open(examples_path) as f:
            examples = json.load(f)

    rules_path = _find_prompt_rules(grade, model, run_id)
    if os.path.exists(rules_path):
        os.environ["PROMPT_RULES_PATH"] = rules_path

    plan = []
    for _, _, q, _ in weak_items:
        plan.append((
            q.get("standard", ""),
            q.get("difficulty", "medium"),
            q.get("type", "mcq"),
        ))

    n = len(plan)
    print(f"  精确生成 {n} 题（仅低分组合）")
    print(f"  provider: {provider}, model: {model}")
    print(f"  prompt rules: {rules_path}")

    results_by_index = [None] * n
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=min(workers, n)) as ex:
        futures = {}
        for i, (s, d, qt) in enumerate(plan):
            fut = ex.submit(
                _generate_one,
                standard=s, difficulty=d, examples=examples,
                provider=provider, api_key=api_key, model=model,
                grade=grade, subject="ELA", index=i, question_type=qt,
            )
            futures[fut] = (i, s, d, qt)
            if i < workers:
                time.sleep(0.1)

        done = 0
        for fut in as_completed(futures):
            i, s, d, qt = futures[fut]
            try:
                mcq, elapsed, err_msg, usage = fut.result()
                std_short = (s or "").replace("CCSS.ELA-LITERACY.", "")
                if mcq:
                    results_by_index[i] = mcq
                    done += 1
                    print(f"  [{done:>3d}/{n}] {std_short:20s} {d:6s} {qt:7s} {elapsed:.1f}s ✓")
                else:
                    print(f"  [{done:>3d}/{n}] {std_short:20s} {d:6s} {qt:7s} {elapsed:.1f}s ✗ {(err_msg or '')[:40]}")
            except Exception as e:
                print(f"  异常: {str(e)[:60]}")

    results, stats = _validate_and_repair_keep_all(
        results_by_index, plan, grade=grade, subject="ELA"
    )
    if stats.get("constructed"):
        print(f"  校验: {stats['constructed']} 题构造了最小合法题")

    elapsed = time.time() - t0
    print(f"  生成完成: {done}/{n} 题 ({elapsed:.0f}s)")
    return results


def _evaluate_items(items, parallel=20):
    from evaluation.inceptbench_client import InceptBenchEvaluator

    evaluator = InceptBenchEvaluator()
    results = [None] * len(items)
    total = len(items)
    t0 = time.time()

    def _eval_one(idx_item):
        idx, item = idx_item
        attempt = 0
        while True:
            attempt += 1
            r = evaluator.evaluate_mcq(item)
            is_err = r.get("status") == "error"
            s = r.get("overall_score")
            has_score = isinstance(s, (int, float))
            if not is_err and has_score and float(s) > 0:
                return idx, r
            if is_err and attempt < 5:
                time.sleep(min(10 * attempt, 60))
                continue
            if attempt >= 3:
                return idx, r
            time.sleep(5)

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = {pool.submit(_eval_one, (i, item)): i for i, item in enumerate(items)}
        done = 0
        for future in as_completed(futs):
            idx, r = future.result()
            results[idx] = r
            done += 1
            score = r.get("overall_score", 0)
            item = items[idx]
            std = item.get("standard", "").replace("CCSS.ELA-LITERACY.", "")
            diff = item.get("difficulty", "?")
            qtype = item.get("type", "?")
            marker = "✓" if score >= 0.85 else "✗"
            print(f"  [{done:>3d}/{total}] {std:20s} {diff:6s} {qtype:7s} : {score:.2f} {marker}")

    return results


def _compare_results(weak_items, new_items, new_results, threshold=0.85):
    """对比新旧分数，返回 (improved_indices, still_failing_indices)。"""
    print(f"\n{'='*70}")
    print(f"  对比结果 (阈值 {threshold*100:.0f}%)")
    print(f"{'='*70}")
    print(f"  {'标准':25s} | {'难度':6s} | {'题型':7s} | {'旧分':>5s} → {'新分':>5s} | {'变化'}")
    print(f"  {'─'*75}")

    improved = degraded = unchanged = newly_passed = 0
    improved_indices = []
    still_failing = []

    for i, (orig_idx, old_score, q, _) in enumerate(weak_items):
        std = q.get("standard", "").replace("CCSS.ELA-LITERACY.", "")
        diff = q.get("difficulty", "?")
        qtype = q.get("type", "?")

        if i < len(new_results) and new_results[i] is not None:
            new_score = new_results[i].get("overall_score", 0)
            delta = new_score - old_score
            if delta > 0.01:
                arrow = f"↑ +{delta:.2f}"
                improved += 1
                if new_score >= threshold:
                    arrow += " ✓ PASS"
                    newly_passed += 1
                    improved_indices.append(i)
                else:
                    still_failing.append(i)
            elif delta < -0.01:
                arrow = f"↓ {delta:.2f}"
                degraded += 1
                still_failing.append(i)
            else:
                arrow = "  ─"
                unchanged += 1
                still_failing.append(i)
            print(f"  {std:25s} | {diff:6s} | {qtype:7s} | {old_score:.2f} → {new_score:.2f} | {arrow}")
        else:
            print(f"  {std:25s} | {diff:6s} | {qtype:7s} | {old_score:.2f} → {'N/A':>5s} |")
            still_failing.append(i)

    print(f"\n  汇总: ↑{improved} 改善  ↓{degraded} 下降  ─{unchanged} 不变")
    print(f"  新通过: {newly_passed}/{len(weak_items)} 题 ({newly_passed/max(len(weak_items),1)*100:.0f}%)")
    return improved_indices, still_failing


def _update_best(best_mcqs_path, best_results_path, weak_items, new_items,
                 new_results, improved_indices, all_mcqs, all_res):
    """将分数更高的新题替换进 best 文件，更新 pass_rate。"""
    updated_mcqs = list(all_mcqs)
    updated_results = list(all_res.get("results", []))
    replaced = 0

    for i in improved_indices:
        orig_idx = weak_items[i][0]
        if i < len(new_items) and i < len(new_results) and new_results[i] is not None:
            updated_mcqs[orig_idx] = new_items[i]
            updated_results[orig_idx] = new_results[i]
            replaced += 1

    total = len(updated_mcqs)
    pass_count = sum(1 for r in updated_results
                     if isinstance(r.get("overall_score"), (int, float)) and r["overall_score"] >= 0.85)
    new_rate = round(100 * pass_count / total, 1) if total else 0

    old_rate = all_res.get("pass_rate", 0)
    rate_str = str(new_rate).replace(".", "_")

    new_mcqs_path = best_mcqs_path.rsplit("_best_", 1)[0] + f"_best_{rate_str}.json"
    new_results_path = best_results_path.rsplit("_best_", 1)[0] + f"_best_{rate_str}.json"

    with open(new_mcqs_path, "w", encoding="utf-8") as f:
        json.dump(updated_mcqs, f, ensure_ascii=False, indent=2)

    updated_res = dict(all_res)
    updated_res["results"] = updated_results
    updated_res["pass_rate"] = new_rate
    updated_res["pass_count"] = pass_count
    updated_res["total"] = total
    with open(new_results_path, "w", encoding="utf-8") as f:
        json.dump(updated_res, f, ensure_ascii=False, indent=2)

    # 更新 progress 文件
    progress_pattern = os.path.join(
        PROJECT_ROOT, "evaluation_output",
        f"closed_loop_progress_*{os.path.basename(best_mcqs_path).split('_best_')[0].replace('mcqs_', '')}*.json"
    )
    for pf in glob.glob(progress_pattern):
        try:
            with open(pf) as f:
                prog = json.load(f)
            if new_rate > prog.get("best_pass_rate", 0):
                prog["best_pass_rate"] = new_rate
                prog["best_mcqs_path"] = new_mcqs_path
                prog["best_results_path"] = new_results_path
                prog["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                with open(pf, "w", encoding="utf-8") as f:
                    json.dump(prog, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if new_mcqs_path != best_mcqs_path and os.path.exists(best_mcqs_path):
        try:
            os.remove(best_mcqs_path)
        except Exception:
            pass
    if new_results_path != best_results_path and os.path.exists(best_results_path):
        try:
            os.remove(best_results_path)
        except Exception:
            pass

    print(f"\n  [update-best] 替换 {replaced} 题, 通过率 {old_rate}% → {new_rate}%")
    print(f"  新 best: {os.path.basename(new_mcqs_path)}")
    return new_rate


def _update_rules(still_failing_indices, weak_items, new_items, new_results,
                  grade, model, run_id, best_mcqs_path, best_results_path):
    """对仍低分的题，调用 improve_prompt 自动更新规则。"""
    if not still_failing_indices:
        print("\n  [update-rules] 所有题目均已通过，无需更新规则")
        return

    tmp_mcqs = []
    tmp_results = []
    for i in still_failing_indices:
        if i < len(new_items) and i < len(new_results) and new_results[i] is not None:
            tmp_mcqs.append(new_items[i])
            tmp_results.append(new_results[i])

    if not tmp_mcqs:
        print("\n  [update-rules] 没有可用于更新的题目")
        return

    tmp_mcqs_path = os.path.join(PROJECT_ROOT, "evaluation_output", "_qv_tmp_mcqs.json")
    tmp_results_path = os.path.join(PROJECT_ROOT, "evaluation_output", "_qv_tmp_results.json")

    with open(tmp_mcqs_path, "w", encoding="utf-8") as f:
        json.dump(tmp_mcqs, f, ensure_ascii=False, indent=2)

    tmp_res_data = {
        "results": tmp_results,
        "pass_rate": 0,
        "pass_count": sum(1 for r in tmp_results if r.get("overall_score", 0) >= 0.85),
        "total": len(tmp_results),
    }
    with open(tmp_results_path, "w", encoding="utf-8") as f:
        json.dump(tmp_res_data, f, ensure_ascii=False, indent=2)

    rules_path = _find_prompt_rules(grade, model, run_id)
    examples_path = _find_examples(grade, model, run_id)

    print(f"\n  [update-rules] 基于 {len(tmp_mcqs)} 题仍低分的反馈更新 prompt 规则...")

    from scripts.improve_prompt import run as improve_prompt_run
    report = improve_prompt_run(
        results_path=tmp_results_path,
        mcqs_path=tmp_mcqs_path,
        rules_output=rules_path,
        threshold=0.85,
        max_global=3,
        max_per_standard=2,
        max_per_standard_difficulty=3,
        examples_path=examples_path if os.path.exists(examples_path) else None,
    )

    for p in [tmp_mcqs_path, tmp_results_path]:
        try:
            os.remove(p)
        except Exception:
            pass

    if report.get("updated"):
        print(f"  规则已更新: 全局 {report.get('global_rules_count', 0)} 条, "
              f"by_standard {report.get('by_standard_count', 0)} 个, "
              f"by_standard_difficulty {report.get('by_standard_difficulty_count', 0)} 个")
    else:
        print(f"  未更新: {report.get('reason', '')} (反馈数 {report.get('feedback_count', 0)})")



def _run_multi(args):
    """处理 --grade all 和多模型的批量运行。"""
    grades = list(range(1, 13)) if args.grade.lower() == "all" else [g.strip() for g in args.grade.split(",")]
    models = [m.strip() for m in args.model.split(",")]

    all_summary = []
    total_start = time.time()

    for model in models:
        for grade in grades:
            grade = str(grade)
            best_mcqs, best_results = _find_best_files(grade, model, args.run_id)
            if not best_mcqs or not best_results:
                print(f"\n  跳过 G{grade} {model}: 无 best 文件")
                continue

            # 先看有多少低分题
            threshold = args.threshold / 100.0
            weak_items, all_mcqs, _ = _load_weak_items(best_mcqs, best_results, threshold)
            if not weak_items and not args.dry_run:
                rate = round(100 * (len(all_mcqs) - len(weak_items)) / len(all_mcqs), 1) if all_mcqs else 0
                print(f"\n  G{grade:>2s} {model}: {rate}% — 已全部通过，跳过")
                all_summary.append({"grade": grade, "model": model, "rate": rate, "weak": 0, "improved": 0, "status": "已通过"})
                continue

            # 构造子 args
            sub_args = argparse.Namespace(
                grade=grade, model=model, run_id=args.run_id,
                threshold=args.threshold, workers=args.workers,
                parallel=args.parallel, dry_run=args.dry_run,
                update_best=args.update_best, update_rules=args.update_rules,
                auto=args.auto, rounds=args.rounds,
            )
            if sub_args.auto:
                sub_args.update_best = True
                sub_args.update_rules = True
            if sub_args.rounds > 1:
                sub_args.update_best = True
                sub_args.update_rules = True

            # 保存 sys.argv 防止子调用干扰
            try:
                _run_single_grade(sub_args, all_summary)
            except Exception as e:
                print(f"\n  G{grade} {model} 出错: {e}")
                all_summary.append({"grade": grade, "model": model, "status": f"错误: {e}"})

    elapsed = time.time() - total_start
    print(f"\n{'#'*70}")
    print(f"  全年级汇总 ({len(all_summary)} 个, 总耗时 {elapsed/60:.1f} 分钟)")
    print(f"{'#'*70}")
    print(f"  {'年级':>4s} | {'模型':20s} | {'低分':>4s} | {'新通过':>5s} | {'通过率':>6s} | 状态")
    print(f"  {'─'*70}")
    for s in sorted(all_summary, key=lambda x: (x.get("model",""), int(x.get("grade","0")))):
        rate = str(s.get("new_rate", s.get("rate", "?")))
        print(f"  G{str(s['grade']):>2s} | {s.get('model','?'):20s} | {s.get('weak',0):>4d} | {s.get('improved',0):>5d} | {rate:>5s}% | {s.get('status','')}")


def _run_single_grade(args, summary_list):
    """对单个年级运行多轮快速验证（从 main 或 _run_multi 调用）。"""
    threshold = args.threshold / 100.0
    total_start = time.time()
    round_history = []

    for rnd in range(1, args.rounds + 1):
        print(f"\n{'#'*70}")
        if args.rounds > 1:
            print(f"  第 {rnd}/{args.rounds} 轮")
        print(f"  快速验证: Grade {args.grade} | {args.model} | 阈值 {args.threshold}%")
        print(f"{'#'*70}")

        best_mcqs, best_results = _find_best_files(args.grade, args.model, args.run_id)
        if not best_mcqs or not best_results:
            print(f"  错误: 找不到 Grade {args.grade} {args.model} 的 best 文件")
            return

        print(f"  best: {os.path.basename(best_mcqs)}")

        rules_path = _find_prompt_rules(args.grade, args.model, args.run_id)
        if os.path.exists(rules_path):
            with open(rules_path) as f:
                rules = json.load(f)
            print(f"  rules: {len(rules.get('global_rules',[]))}g + {len(rules.get('by_standard',{}))}s + {len(rules.get('by_standard_difficulty',{}))}sd")

        weak_items, all_mcqs, all_res = _load_weak_items(best_mcqs, best_results, threshold)
        total = len(all_mcqs)
        passed = total - len(weak_items)
        current_rate = round(100 * passed / total, 1)
        print(f"  {total} 题, 通过 {passed} ({current_rate}%), 低分 {len(weak_items)}")

        if rnd == 1:
            _print_analysis(weak_items)

        if args.dry_run:
            print("  [dry-run] 跳过")
            summary_list.append({"grade": args.grade, "model": args.model, "rate": current_rate, "weak": len(weak_items), "improved": 0, "status": "dry-run"})
            return

        if not weak_items:
            print("  已全部通过！")
            round_history.append({"round": rnd, "rate": current_rate, "weak": 0, "improved": 0})
            break

        print(f"\n  [1/2] 生成 {len(weak_items)} 题")
        new_items = _generate_targeted(weak_items, args.grade, args.model, args.run_id, args.workers)

        print(f"\n  [2/2] 评估 {len(new_items)} 题")
        t0 = time.time()
        new_results = _evaluate_items(new_items, args.parallel)
        print(f"  评估完成 ({time.time()-t0:.0f}s)")

        improved_indices, still_failing = _compare_results(
            weak_items, new_items, new_results, threshold
        )

        newly_passed = len(improved_indices)
        round_history.append({
            "round": rnd, "rate": current_rate,
            "weak": len(weak_items), "improved": newly_passed,
            "still_failing": len(still_failing),
        })

        if args.update_best and improved_indices:
            new_rate = _update_best(
                best_mcqs, best_results, weak_items, new_items,
                new_results, improved_indices, all_mcqs, all_res
            )
            round_history[-1]["new_rate"] = new_rate

        if args.update_rules and still_failing:
            print(f"\n  [update-rules] {len(still_failing)} 题仍低分")
            _update_rules(
                still_failing, weak_items, new_items, new_results,
                args.grade, args.model, args.run_id, best_mcqs, best_results
            )

        if not still_failing:
            print(f"\n  第 {rnd} 轮后全部通过！")
            break

    # 汇总
    last = round_history[-1] if round_history else {}
    final_rate = last.get("new_rate", last.get("rate", current_rate))
    total_improved = sum(h.get("improved", 0) for h in round_history)
    total_weak = round_history[0].get("weak", 0) if round_history else 0

    if len(round_history) > 1:
        elapsed = time.time() - total_start
        print(f"\n  多轮汇总 ({len(round_history)} 轮, {elapsed/60:.1f} 分钟)")
        print(f"  {'轮次':>4s} | {'低分':>5s} | {'通过':>5s} | {'仍低分':>5s} | 通过率")
        print(f"  {'─'*45}")
        for h in round_history:
            rate = h.get("new_rate", h.get("rate", "?"))
            print(f"  R{h['round']:>3d} | {h.get('weak',0):>5d} | {h.get('improved',0):>5d} | {h.get('still_failing',0):>5d} | {rate}%")

    summary_list.append({
        "grade": args.grade, "model": args.model,
        "rate": str(final_rate), "new_rate": str(final_rate),
        "weak": total_weak, "improved": total_improved,
        "status": f"R{len(round_history)}完成",
    })


if __name__ == "__main__":
    args = None
    parsed = argparse.ArgumentParser(description="快速验证 prompt 规则对低分题的效果")
    parsed.add_argument("--grade", required=True, help="年级 (1-12, 逗号分隔, 或 all)")
    parsed.add_argument("--model", default="or/gemini-3-pro", help="模型名（逗号分隔可跑多模型）")
    parsed.add_argument("--run-id", default="matrix", help="run-id")
    parsed.add_argument("--threshold", type=float, default=85, help="低分阈值 (百分数)")
    parsed.add_argument("--workers", type=int, default=30, help="生成并行数")
    parsed.add_argument("--parallel", type=int, default=20, help="评估并行数")
    parsed.add_argument("--dry-run", action="store_true", help="仅分析，不生成和评估")
    parsed.add_argument("--update-best", action="store_true", help="将分数更高的新题替换进 best 文件")
    parsed.add_argument("--update-rules", action="store_true", help="对仍低分的题自动更新 prompt 规则")
    parsed.add_argument("--auto", action="store_true", help="等同于 --update-best --update-rules")
    parsed.add_argument("--rounds", type=int, default=1, help="每个年级的迭代轮数")
    args = parsed.parse_args()

    if args.auto:
        args.update_best = True
        args.update_rules = True
    if args.rounds > 1 and not (args.update_best and args.update_rules):
        args.update_best = True
        args.update_rules = True

    if args.grade.lower() == "all" or "," in args.grade or "," in args.model:
        _run_multi(args)
    else:
        summary = []
        _run_single_grade(args, summary)
