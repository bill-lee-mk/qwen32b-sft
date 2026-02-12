
## 8. 主程序入口

### `main.py`


"""
主程序入口
支持数据处理、训练、API服务等多种功能
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 默认 MCQ/结果路径（与 closed-loop 默认一致，用于判断是否按模型名生成路径）
_DEFAULT_MCQS = "evaluation_output/mcqs_237.json"
_DEFAULT_RESULTS = "evaluation_output/results_237.json"


def _run_closed_loop_one_model(project_root, model, args, use_model_specific_paths=False, run_id=None):
    """
    对单个模型跑闭环：生成 → 评估 → 未达标则补示例/改 prompt → 重复至达标或达最大轮数。
    use_model_specific_paths=True 时使用 examples_<model_slug>.json 与 prompt_rules_<model_slug>.json，避免多模型闭环时互相覆盖、提示词撕裂。
    每轮保存到 mcqs_237_<model>_roundN.json / results_237_<model>_roundN.json；刷新历史最高时复制到 _best_{rate}.json（如 best_94_5.json 表示 94.5% 通过率）。
    run_id 不为空时，所有路径加上 _<run_id> 后缀，实现不同批次完全隔离（示例、提示词、题目、结果互不覆盖）。
    每轮结束后写入 evaluation_output/closed_loop_progress_<model>.json，便于中断后查看汇总。
    返回 dict: model, model_slug, final_pass_rate, best_pass_rate, best_round, pass_count, n_valid, n_submitted, round_reached, target_reached, error
    """
    model = (model or "deepseek-chat").strip()
    model_slug = model.replace(".", "_").replace("/", "_")
    run_suffix = f"_{run_id}" if run_id else ""
    use_default_paths = args.mcqs == _DEFAULT_MCQS and args.results == _DEFAULT_RESULTS
    if use_default_paths:
        base_mcqs = os.path.join(project_root, "evaluation_output", f"mcqs_237_{model_slug}{run_suffix}")
        base_results = os.path.join(project_root, "evaluation_output", f"results_237_{model_slug}{run_suffix}")
        progress_path = os.path.join(project_root, "evaluation_output", f"closed_loop_progress_{model_slug}{run_suffix}.json")
    else:
        base_mcqs = (os.path.join(project_root, args.mcqs) if not os.path.isabs(args.mcqs) else args.mcqs).replace(".json", "")
        base_results = (os.path.join(project_root, args.results) if not os.path.isabs(args.results) else args.results).replace(".json", "")
        if run_suffix:
            base_mcqs += run_suffix
            base_results += run_suffix
        progress_path = os.path.join(project_root, "evaluation_output", f"closed_loop_progress_{model_slug}{run_suffix}.json")
    # 当前最佳文件路径（含通过率，如 best_94_5.json），启动时从已有文件恢复；兼容旧格式 best.json
    _existing_mcqs = glob.glob(f"{base_mcqs}_best_*.json")
    _existing_results = glob.glob(f"{base_results}_best_*.json")
    current_best_mcqs_path = _existing_mcqs[0] if _existing_mcqs else (f"{base_mcqs}_best.json" if os.path.exists(f"{base_mcqs}_best.json") else None)
    current_best_results_path = _existing_results[0] if _existing_results else (f"{base_results}_best.json" if os.path.exists(f"{base_results}_best.json") else None)
    if use_model_specific_paths:
        examples_path = os.path.join(project_root, "processed_training_data", f"examples_{model_slug}{run_suffix}.json")
        prompt_rules_path = os.path.join(project_root, "processed_training_data", f"prompt_rules_{model_slug}{run_suffix}.json")
        # 首次运行：从默认文件复制一份，避免空文件
        _default_examples = os.path.join(project_root, "processed_training_data", "examples.json")
        _default_rules = os.path.join(project_root, "processed_training_data", "prompt_rules.json")
        if not os.path.exists(examples_path) and os.path.exists(_default_examples):
            shutil.copy2(_default_examples, examples_path)
            print(f"  已为模型 {model} 复制初始示例: {examples_path}")
        if not os.path.exists(prompt_rules_path) and os.path.exists(_default_rules):
            shutil.copy2(_default_rules, prompt_rules_path)
            print(f"  已为模型 {model} 复制初始 prompt 规则: {prompt_rules_path}")
    else:
        examples_path = os.path.join(project_root, args.examples) if not os.path.isabs(args.examples) else args.examples
        prompt_rules_path = os.path.join(project_root, "processed_training_data", "prompt_rules.json")
    target = getattr(args, "pass_rate_target", 95.0)
    max_rounds = getattr(args, "max_rounds", 10)
    parallel = getattr(args, "parallel", 20)
    no_target = (target is None or target <= 0)  # 不设目标时跑满 max_rounds，取最终/最高通过率
    log_file = getattr(args, "log_file", None)
    def _log(msg):
        print(msg)
        if log_file:
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
    result = {
        "model": model,
        "model_slug": model_slug,
        "final_pass_rate": None,
        "best_pass_rate": None,
        "best_round": None,
        "pass_count": None,
        "n_valid": None,
        "n_submitted": None,
        "round_reached": 0,
        "target_reached": False,
        "error": None,
    }
    best_pass_rate_seen = -1.0
    best_round_seen = 0

    def _write_progress(rnd, pr, br, bpr, m_path, r_path):
        best_m = current_best_mcqs_path
        best_r = current_best_results_path
        prog = {
            "model": model,
            "model_slug": model_slug,
            "round_reached": rnd,
            "pass_rate": pr,
            "best_round": br,
            "best_pass_rate": bpr,
            "best_mcqs_path": os.path.relpath(best_m, project_root) if best_m and use_default_paths else (best_m or ""),
            "best_results_path": os.path.relpath(best_r, project_root) if best_r and use_default_paths else (best_r or ""),
            "current_mcqs_path": os.path.relpath(m_path, project_root) if use_default_paths else m_path,
            "current_results_path": os.path.relpath(r_path, project_root) if use_default_paths else r_path,
            "updated_at": datetime.now().isoformat(),
        }
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(prog, f, indent=2, ensure_ascii=False)

    def _print_summary():
        rel_progress = os.path.relpath(progress_path, project_root)
        _log("\n" + "=" * 60)
        _log("闭环汇总（可随时查看 " + rel_progress + "）")
        _log("=" * 60)
        _log(f"模型: {model}")
        _log(f"当前轮次: {result.get('round_reached', 0)}")
        _log(f"本轮通过率: {result.get('final_pass_rate')}%")
        _log(f"历史最高: {result.get('best_pass_rate')}% @ 第{result.get('best_round', 0)}轮")
        if current_best_mcqs_path and current_best_results_path:
            rel_best_m = os.path.relpath(current_best_mcqs_path, project_root)
            rel_best_r = os.path.relpath(current_best_results_path, project_root)
            _log(f"最佳题目: {rel_best_m}")
            _log(f"最佳结果: {rel_best_r}")
        else:
            _log("最佳题目/结果: 暂无（未完成首轮评估）")
        _log("=" * 60)

    try:
        for round_num in range(1, max_rounds + 1):
            mcqs_path = f"{base_mcqs}_round{round_num}.json"
            results_path = f"{base_results}_round{round_num}.json"
            result["round_reached"] = round_num
            if round_num == 1 and no_target:
                _log(f"\n  (未设通过率目标，将跑满 {max_rounds} 轮，取最终/历史最高通过率)")
            _log(f"\n========== [{model}] 闭环 第 {round_num}/{max_rounds} 轮 ==========")
            gen_cmd = [sys.executable, os.path.join(project_root, "scripts", "generate_mcq.py"), "--model", model, "--all-combinations", "--output", mcqs_path, "--examples", examples_path]
            if getattr(args, "workers", None) is not None:
                gen_cmd.extend(["--workers", str(args.workers)])
            _log(f"  [1/4] 生成: {' '.join(gen_cmd)}")
            gen_env = None
            if use_model_specific_paths:
                gen_env = {**os.environ, "PROMPT_RULES_PATH": prompt_rules_path}
            r = subprocess.run(gen_cmd, cwd=project_root, env=gen_env)
            if r.returncode != 0:
                result["error"] = f"生成失败 exit={r.returncode}"
                _log(f"  生成失败 exit={r.returncode}")
                return result
            eval_cmd = [sys.executable, os.path.join(project_root, "main.py"), "evaluate", "--input", mcqs_path, "--output", results_path, "--parallel", str(parallel)]
            _log(f"  [2/4] 评估: ... --parallel {parallel}")
            r = subprocess.run(eval_cmd, cwd=project_root)
            if r.returncode != 0:
                result["error"] = f"评估失败 exit={r.returncode}"
                _log(f"  评估失败 exit={r.returncode}")
                return result
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    out_data = json.load(f)
            except Exception as e:
                result["error"] = str(e)
                return result
            scores = out_data.get("scores") or []
            n_valid = out_data.get("valid_score_count")
            if n_valid is None:
                n_valid = sum(1 for s in scores if s is not None and isinstance(s, (int, float)))
            if not n_valid:
                result["final_pass_rate"] = 0.0
                result["pass_count"] = 0
                result["n_valid"] = 0
                result["n_submitted"] = out_data.get("total") or len(scores)
                result["error"] = "无有效分数"
                _log("  无有效分数，结束闭环")
                return result
            pass_count = out_data.get("pass_count") or sum(1 for s in scores if s is not None and float(s) >= 0.85)
            pass_rate = 100.0 * pass_count / n_valid
            n_submitted = out_data.get("total") or out_data.get("total_submitted") or len(scores)
            result["final_pass_rate"] = round(pass_rate, 2)
            result["pass_count"] = pass_count
            result["n_valid"] = n_valid
            result["n_submitted"] = n_submitted
            if pass_rate > best_pass_rate_seen:
                best_pass_rate_seen = pass_rate
                best_round_seen = round_num
                rate_str = str(round(pass_rate, 1)).replace(".", "_")
                new_mcqs_best = f"{base_mcqs}_best_{rate_str}.json"
                new_results_best = f"{base_results}_best_{rate_str}.json"
                for old in glob.glob(f"{base_mcqs}_best_*.json"):
                    if old != new_mcqs_best and os.path.exists(old):
                        os.remove(old)
                for old in glob.glob(f"{base_results}_best_*.json"):
                    if old != new_results_best and os.path.exists(old):
                        os.remove(old)
                for old_path in [f"{base_mcqs}_best.json", f"{base_results}_best.json"]:
                    if os.path.exists(old_path):
                        os.remove(old_path)
                shutil.copy2(mcqs_path, new_mcqs_best)
                shutil.copy2(results_path, new_results_best)
                current_best_mcqs_path = new_mcqs_best
                current_best_results_path = new_results_best
                _log(f"  已保存最佳: {os.path.relpath(new_mcqs_best, project_root)}")
            result["best_pass_rate"] = round(best_pass_rate_seen, 2)
            result["best_round"] = best_round_seen
            n_error = n_submitted - n_valid
            if n_error:
                _log(f"  无分数题: {n_error} 题")
            _log(f"  通过率(>=0.85): {pass_count}/{n_valid} ({pass_rate:.1f}%)（总提交 {n_submitted}）" + (f"  历史最高: {best_pass_rate_seen:.1f}% @ 第{best_round_seen}轮" if best_round_seen != round_num else ""))
            _write_progress(round_num, round(pass_rate, 2), best_round_seen, round(best_pass_rate_seen, 2), mcqs_path, results_path)
            # 仅当明确设了目标（target>0）且达标时才提前结束；target<=0 时跑满 max_rounds
            if target > 0 and pass_rate >= target:
                result["target_reached"] = True
                _log(f"\n  已达目标通过率 {pass_rate:.1f}% >= {target}%，闭环结束")
                _print_summary()
                return result
            imp_cmd = [sys.executable, os.path.join(project_root, "main.py"), "improve-examples", "--results", results_path, "--mcqs", mcqs_path, "--output", examples_path, "--raw-data-dir", args.raw_data_dir, "--parallel", str(parallel)]
            _log(f"  [3/4] 补示例: ... --parallel {parallel}")
            subprocess.run(imp_cmd, cwd=project_root)
            imp_prompt_cmd = [sys.executable, os.path.join(project_root, "scripts", "improve_prompt.py"), "--results", results_path, "--mcqs", mcqs_path, "--output", prompt_rules_path, "--examples", examples_path]
            _log(f"  [4/4] 改 prompt 规则")
            subprocess.run(imp_prompt_cmd, cwd=project_root)
            if round_num == max_rounds:
                _log(f"\n  已达最大轮数 {max_rounds}，最终通过率 {pass_rate:.1f}%，历史最高 {best_pass_rate_seen:.1f}% @ 第{best_round_seen}轮")
                _print_summary()
    except KeyboardInterrupt:
        _log("\n[用户中断] 已打印汇总，最佳题目与结果见上方。")
        _print_summary()
        raise
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-32B K-12 ELA MCQ生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 数据处理
  python main.py process-data
  
  # 筛选示例（闭源模型用）
  python main.py select-examples -n 5
  
  # SFT训练
  python main.py train-sft
  
  # DPO训练
  python main.py train-dpo
  
  # 完整训练
  python main.py train-all
  
  # 导出最终模型（SFT+DPO 分别训练后执行）
  python main.py merge-model
  
  # 启动API服务
  python main.py serve-api
  
  # 评估模型
  python main.py evaluate --input sample_questions.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 数据处理命令
    process_parser = subparsers.add_parser("process-data", help="处理训练数据")
    
    # 筛选示例（用于闭源模型）
    select_parser = subparsers.add_parser("select-examples", help="从 raw_data 筛选示例")
    # 分析 raw_data 维度（难度、学科、标准）
    analyze_dims_parser = subparsers.add_parser("analyze-dimensions", help="统计 raw_data 中难度、学科、标准分布")
    analyze_dims_parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    analyze_dims_parser.add_argument("--output", help="输出 JSON 报告路径")
    select_parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    select_parser.add_argument("--output", default="processed_training_data/examples.json", help="输出 JSON 路径")
    select_parser.add_argument("-n", type=int, default=5, help="示例数量")
    process_parser.add_argument("--input-dir", default="raw_data", help="原始数据目录")
    process_parser.add_argument("--output-dir", default="processed_training_data", help="输出数据目录")
    
    # SFT训练命令
    sft_parser = subparsers.add_parser("train-sft", help="SFT训练")
    sft_parser.add_argument("--config", default="configs/training_config.yaml", help="配置文件")
    sft_parser.add_argument("--data", default="processed_training_data/sft_data.jsonl", help="训练数据")
    
    # DPO训练命令
    dpo_parser = subparsers.add_parser("train-dpo", help="DPO训练")
    dpo_parser.add_argument("--config", default="configs/training_config.yaml", help="配置文件")
    dpo_parser.add_argument("--data", default="processed_training_data/dpo_data.jsonl", help="训练数据")
    dpo_parser.add_argument("--sft-model", help="SFT模型路径")
    
    # 完整训练命令
    train_all_parser = subparsers.add_parser("train-all", help="完整训练流水线")
    train_all_parser.add_argument("--config", default="configs/training_config.yaml", help="配置文件")
    
    # 合并/导出最终模型命令（SFT+DPO 分别训练后使用）
    merge_parser = subparsers.add_parser("merge-model", help="将 DPO 模型导出到 final_model 目录")
    merge_parser.add_argument("--config", default="configs/training_config.yaml", help="配置文件")
    merge_parser.add_argument("--source", help="源模型路径（默认用 config 中的 dpo_model，若不存在则用 sft_model）")
    
    # API服务命令
    api_parser = subparsers.add_parser("serve-api", help="启动API服务")
    api_parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    api_parser.add_argument("--port", type=int, default=8000, help="端口号")
    api_parser.add_argument("--model", default="models/final_model", help="模型路径")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--input", required=True, help="输入文件或目录")
    eval_parser.add_argument("--output", help="输出文件")
    eval_parser.add_argument("--api-key", help="InceptBench token（默认从 INCEPTBENCH_API_KEY 环境变量读取）")
    eval_parser.add_argument("--debug", action="store_true", help="打印请求 payload，便于排查 500 错误")
    eval_parser.add_argument("--timeout", type=int, default=180, help="单题超时秒数（默认 180，约 2 分钟/题）")
    eval_parser.add_argument("--parallel", type=int, default=20, help="并行提交数（一次只能提交一题，默认 20 进程并行）")

    # 预提交 MCQ 校验（提升 InceptBench 通过率）
    validate_parser = subparsers.add_parser("validate-mcq", help="预提交 MCQ 校验")
    validate_parser.add_argument("--input", "-i", required=True, help="MCQ JSON 文件路径")
    validate_parser.add_argument("--output", "-o", help="校验报告 JSON 路径")

    # 闭环：失败题 → raw_data 候选 InceptBench 评分 → 保留 ≥0.85 作 few-shot → 更新 examples
    improve_parser = subparsers.add_parser("improve-examples", help="根据评估结果更新 few-shot 示例")
    improve_parser.add_argument("--results", required=True, help="评估结果 JSON（含 scores）")
    improve_parser.add_argument("--mcqs", required=True, help="被评估的 MCQ JSON")
    improve_parser.add_argument("--raw-data-dir", default="raw_data", help="raw_data 目录")
    improve_parser.add_argument("--output", default="processed_training_data/examples.json", help="输出 examples 路径")
    improve_parser.add_argument("--threshold", type=float, default=0.85, help="通过分数阈值")
    improve_parser.add_argument("--max-per-pair", type=int, default=1, help="每个 (standard,difficulty) 保留示例数")
    improve_parser.add_argument("--max-candidates-per-pair", type=int, default=0, help="每个组合最多试评的候选数，0=不限制取全部（增加命中率）")
    improve_parser.add_argument("--parallel", type=int, default=20, help="InceptBench 并行评分数")
    improve_parser.add_argument("--timeout", type=int, default=180, help="单题超时秒数")
    improve_parser.add_argument("--retry-delay", type=int, default=60, help="服务端错误时等待秒数后再重试")
    improve_parser.add_argument("--max-retries", type=int, default=3, help="服务端错误最大重试次数")

    # 闭环自动化：生成 → 评估 → 若通过率 < 目标则 improve-examples → 重复直到达标或达最大轮数
    loop_parser = subparsers.add_parser("closed-loop", help="闭环：生成→评估→未达标则补示例/改prompt→重复")
    loop_parser.add_argument("--model", default="deepseek-chat", help="生成 MCQ 使用的模型：API 模型名（deepseek-chat/kimi-latest/gpt-4o）或 local/本地路径（如 models/qwen3-32B/final_model，需先启动 serve-api）")
    loop_parser.add_argument("--mcqs", default="evaluation_output/mcqs_237.json", help="MCQ 输出路径（默认会改为 evaluation_output/mcqs_237_<model>.json，79 标准×3 难度=237 题）")
    loop_parser.add_argument("--results", default="evaluation_output/results_237.json", help="评估结果路径（默认会改为 evaluation_output/results_237_<model>.json）")
    loop_parser.add_argument("--examples", default="processed_training_data/examples.json", help="few-shot 示例路径（每轮 improve 会更新）")
    loop_parser.add_argument("--pass-rate-target", type=float, default=95.0, help="通过率目标（百分数，默认 95）；设为 0 表示不设目标，跑满 --max-rounds 后取最终/最高通过率")
    loop_parser.add_argument("--max-rounds", type=int, default=10, help="最大循环轮数")
    loop_parser.add_argument("--raw-data-dir", default="raw_data", help="improve-examples 使用的 raw_data 目录")
    loop_parser.add_argument("--workers", type=int, default=None, help="生成阶段并行数（DeepSeek/API 默认 10，本地 8，Kimi 10）")
    loop_parser.add_argument("--parallel", type=int, default=20, help="评估阶段 InceptBench 并行数（默认 20）")
    loop_parser.add_argument("--log-file", default=None, help="将运行日志同时写入该文件（便于中途停止后查看完整输出）")
    loop_parser.add_argument("--run-id", default=None, help="运行批次 ID；指定后 examples/prompt_rules/mcqs/results 均加此后缀，不同批次互不覆盖（如 --run-id exp1）")

    # 多模型闭环：对多个模型分别跑闭环，最后汇总各模型通过率并保存 JSON
    multi_parser = subparsers.add_parser("closed-loop-multi", help="多模型闭环：对多个模型分别跑闭环，汇总通过率并保存 JSON")
    multi_parser.add_argument("--models", required=True, help="逗号分隔的模型列表，如 deepseek-chat,local,kimi-latest 或 models/qwen3-32B/final_model")
    multi_parser.add_argument("--mcqs", default=_DEFAULT_MCQS, help="MCQ 路径模板（默认按模型名生成 mcqs_237_<model_slug>.json）")
    multi_parser.add_argument("--results", default=_DEFAULT_RESULTS, help="结果路径模板")
    multi_parser.add_argument("--examples", default="processed_training_data/examples.json", help="few-shot 示例路径（多模型共用，每轮更新）")
    multi_parser.add_argument("--pass-rate-target", type=float, default=95.0, help="通过率目标（百分数）；设为 0 表示不设目标，跑满 --max-rounds 取最终/最高通过率")
    multi_parser.add_argument("--max-rounds", type=int, default=10, help="每个模型最大循环轮数")
    multi_parser.add_argument("--raw-data-dir", default="raw_data", help="improve-examples 使用的 raw_data 目录")
    multi_parser.add_argument("--workers", type=int, default=None, help="生成阶段并行数")
    multi_parser.add_argument("--parallel", type=int, default=20, help="评估阶段并行数（默认 20）")
    multi_parser.add_argument("--summary-output", default=None, help="汇总 JSON 输出路径（默认 evaluation_output/closed_loop_multi_summary.json）")
    multi_parser.add_argument("--log-file", default=None, help="将运行日志同时写入该文件（便于中途停止后查看完整输出）")
    multi_parser.add_argument("--run-id", default=None, help="运行批次 ID；指定后各模型 examples/prompt_rules/mcqs/results 均加此后缀，不同批次互不覆盖")

    # 从失败组合改进 prompt 规则（全局 + 按 standard / (standard,difficulty)）
    improve_prompt_parser = subparsers.add_parser("improve-prompt", help="从评估结果提取失败反馈，更新全局/针对性 prompt 规则")
    improve_prompt_parser.add_argument("--results", required=True, help="评估结果 JSON")
    improve_prompt_parser.add_argument("--mcqs", required=True, help="MCQ JSON")
    improve_prompt_parser.add_argument("--output", default="processed_training_data/prompt_rules.json", help="prompt_rules 输出路径")
    improve_prompt_parser.add_argument("--threshold", type=float, default=0.85, help="低于此分视为失败")
    improve_prompt_parser.add_argument("--max-global", type=int, default=5, help="本轮最多新增全局规则条数")
    improve_prompt_parser.add_argument("--max-per-standard", type=int, default=2, help="每个 standard 最多保留条数")
    improve_prompt_parser.add_argument("--max-per-standard-difficulty", type=int, default=3, help="每个 (std,diff) 最多保留条数")
    improve_prompt_parser.add_argument("--examples", default=None, help="examples.json 路径；提供则仅对有示例仍低分的组合加针对性规则")
    improve_parser.add_argument("--failed-output", default="processed_training_data/improve_examples_failed.json", help="服务端错误超重试的组合记录路径，空不写入")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行命令
    if args.command == "process-data":
        from data_processing.data_processor import main as process_data_main
        project_root = os.path.dirname(os.path.abspath(__file__))
        input_dir = getattr(args, 'input_dir', 'raw_data')
        output_dir = getattr(args, 'output_dir', 'processed_training_data')
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(project_root, input_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(project_root, output_dir)
        process_data_main(input_dir=input_dir, output_dir=output_dir)
    
    elif args.command == "select-examples":
        from data_processing.select_examples import run as select_examples_run
        project_root = os.path.dirname(os.path.abspath(__file__))
        input_dir = getattr(args, 'input_dir', 'raw_data')
        output = getattr(args, 'output', 'processed_training_data/examples.json')
        n = getattr(args, 'n', 5)
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(project_root, input_dir)
        if not os.path.isabs(output):
            output = os.path.join(project_root, output)
        select_examples_run(input_dir=input_dir, output=output, n=n)

    elif args.command == "analyze-dimensions":
        from data_processing.analyze_dimensions import run as analyze_dims_run
        project_root = os.path.dirname(os.path.abspath(__file__))
        input_dir = getattr(args, 'input_dir', 'raw_data')
        if not os.path.isabs(input_dir):
            input_dir = os.path.join(project_root, input_dir)
        analyze_dims_run(input_dir=input_dir, output=getattr(args, 'output', None))
        
    elif args.command == "train-sft":
        from training.full_finetune import main as train_sft_main
        # full_finetune.main() 期望 args 有 sft_only/dpo_only/sft_model/local_rank，
        # 但 sft_parser 未定义这些参数，需在此补全，避免 AttributeError
        args.sft_only = True
        args.dpo_only = False
        args.sft_model = getattr(args, 'sft_model', None)
        args.local_rank = getattr(args, 'local_rank', -1)
        train_sft_main(args)
        
    elif args.command == "train-dpo":
        from training.full_finetune import main as train_dpo_main
        # dpo_parser 有 --sft-model，但无 sft_only/dpo_only/local_rank
        args.sft_only = False
        args.dpo_only = True
        args.sft_model = getattr(args, 'sft_model', None)
        args.local_rank = getattr(args, 'local_rank', -1)
        train_dpo_main(args)
        
    elif args.command == "train-all":
        # 运行完整训练流水线
        print("开始完整训练流水线...")
        
        # 1. 数据处理
        print("\n=== 数据处理 ===")
        from data_processing.data_processor import main as process_data_main
        project_root = os.path.dirname(os.path.abspath(__file__))
        process_data_main(
            input_dir=os.path.join(project_root, "raw_data"),
            output_dir=os.path.join(project_root, "processed_training_data")
        )
        
        # 2. SFT训练
        print("\n=== SFT训练 ===")
        from training.full_finetune import main as train_sft_main
        sys.argv = [sys.argv[0], "--sft-only", "--config", args.config]
        train_sft_main()
        
        # 3. DPO训练
        print("\n=== DPO训练 ===")
        from training.full_finetune import main as train_dpo_main
        sys.argv = [sys.argv[0], "--dpo-only", "--config", args.config, "--sft-model", "models/qwen3-32B/sft_model"]
        train_dpo_main()
        
        # 4. 合并/导出最终模型
        print("\n=== 导出最终模型 ===")
        from training.full_finetune import FullParameterFinetuner, load_config_from_yaml
        config = load_config_from_yaml(args.config)
        finetuner = FullParameterFinetuner(config)
        if finetuner.merge_and_save_final_model(config.dpo_config.output_dir):
            print("最终模型保存在:", config.final_model_dir)
        else:
            print("警告: 导出最终模型失败")
        
    elif args.command == "merge-model":
        from training.full_finetune import FullParameterFinetuner, load_config_from_yaml
        config = load_config_from_yaml(args.config)
        # --source 指定源目录（如 dpo_model 或 checkpoint 路径），默认用 config 中的 dpo_model
        source = getattr(args, 'source', None)
        if not source:
            source = config.dpo_config.output_dir
        finetuner = FullParameterFinetuner(config)
        if finetuner.merge_and_save_final_model(source):
            print("最终模型已导出到:", config.final_model_dir)
        else:
            print("导出失败，请检查 DPO 或 SFT 模型路径是否存在")
            sys.exit(1)
        
    elif args.command == "serve-api":
        from api_service.fastapi_app import run_api_server
        run_api_server(host=args.host, port=args.port, model_path=args.model)
        
    elif args.command == "validate-mcq":
        from scripts.validate_mcq import run as validate_mcq_run
        project_root = os.path.dirname(os.path.abspath(__file__))
        inp = args.input
        if not os.path.isabs(inp):
            inp = os.path.join(project_root, inp)
        report = validate_mcq_run(inp, output_report=getattr(args, 'output', None))
        print(f"总数: {report['total']}, 通过: {report['passed']}, 有问题: {report['failed']}")
        if report["issues"]:
            print("\n问题样本（前 5 条）:")
            for x in report["issues"][:5]:
                print(f"  [{x['index']}] {x['id']} ({x['standard']}): {x['issues']}")
        sys.exit(0 if report["failed"] == 0 else 1)

    elif args.command == "improve-examples":
        from scripts.improve_examples import run as improve_examples_run
        project_root = os.path.dirname(os.path.abspath(__file__))
        results = args.results if os.path.isabs(args.results) else os.path.join(project_root, args.results)
        mcqs = args.mcqs if os.path.isabs(args.mcqs) else os.path.join(project_root, args.mcqs)
        raw_dir = getattr(args, "raw_data_dir", "raw_data")
        raw_dir = raw_dir if os.path.isabs(raw_dir) else os.path.join(project_root, raw_dir)
        out = getattr(args, "output", "processed_training_data/examples.json")
        out = out if os.path.isabs(out) else os.path.join(project_root, out)
        report = improve_examples_run(
            results_path=results,
            mcqs_path=mcqs,
            raw_data_dir=raw_dir,
            examples_output=out,
            threshold=args.threshold,
            max_per_pair=args.max_per_pair,
            max_candidates_per_pair=getattr(args, "max_candidates_per_pair", 0),
            parallel=args.parallel,
            timeout=args.timeout,
            retry_delay=getattr(args, "retry_delay", 60),
            max_retries=getattr(args, "max_retries", 3),
            failed_output=getattr(args, "failed_output", "processed_training_data/improve_examples_failed.json"),
        )
        if "error" in report:
            print(f"错误: {report['error']}")
            sys.exit(1)
        print(f"闭环完成: 失败组合 {report.get('failed_pairs', 0)} 个, 评分 {report.get('evaluated', 0)} 条, 保留 {report.get('kept', 0)} 条, examples 共 {report.get('examples_count', 0)} 条")

    elif args.command == "improve-prompt":
        from scripts.improve_prompt import run as improve_prompt_run
        project_root = os.path.dirname(os.path.abspath(__file__))
        results = args.results if os.path.isabs(args.results) else os.path.join(project_root, args.results)
        mcqs = args.mcqs if os.path.isabs(args.mcqs) else os.path.join(project_root, args.mcqs)
        out = getattr(args, "output", "processed_training_data/prompt_rules.json")
        out = out if os.path.isabs(out) else os.path.join(project_root, out)
        report = improve_prompt_run(
            results_path=results,
            mcqs_path=mcqs,
            rules_output=out,
            threshold=args.threshold,
            max_global=getattr(args, "max_global", 5),
            max_per_standard=getattr(args, "max_per_standard", 2),
            max_per_standard_difficulty=getattr(args, "max_per_standard_difficulty", 3),
            examples_path=getattr(args, "examples", None),
        )
        if report.get("updated"):
            print(f"已更新 prompt 规则: 全局 {report.get('global_rules_count', 0)} 条, by_standard {report.get('by_standard_count', 0)} 个, by_standard_difficulty {report.get('by_standard_difficulty_count', 0)} 个")
        else:
            print(f"未更新: {report.get('reason', '')} (失败反馈数 {report.get('feedback_count', 0)})")

    elif args.command == "evaluate":
        from evaluation.inceptbench_client import InceptBenchEvaluator, to_inceptbench_payload
        
        evaluator = InceptBenchEvaluator(api_key=args.api_key, timeout=getattr(args, "timeout", 180))
        
        # 评估输入文件或目录
        if os.path.isdir(args.input):
            # 评估目录中的所有文件
            results = []
            for file_name in os.listdir(args.input):
                if file_name.endswith('.json'):
                    file_path = os.path.join(args.input, file_name)
                    with open(file_path, 'r') as f:
                        question_data = json.load(f)
                    if getattr(args, 'debug', False):
                        payload = to_inceptbench_payload(question_data)
                        print(f"=== {file_name} 请求 payload（--debug）===")
                        print(json.dumps(payload, indent=2, ensure_ascii=False))
                    result = evaluator.evaluate_mcq(question_data)
                    results.append(result)
                    print(f"评估 {file_name}: {result.get('overall_score', 'N/A')}")
        else:
            # 评估单个文件（支持单个 MCQ 或 MCQ 数组）
            with open(args.input, 'r') as f:
                question_data = json.load(f)
            items = question_data if isinstance(question_data, list) else [question_data]
            timeout = getattr(args, "timeout", 180)
            parallel = getattr(args, "parallel", 20)

            if len(items) == 1:
                # 单题：一次提交
                evaluator = InceptBenchEvaluator(api_key=args.api_key, timeout=timeout)
                if getattr(args, 'debug', False):
                    payload = to_inceptbench_payload(question_data)
                    print("=== 请求 payload（--debug）===")
                    print(json.dumps(payload, indent=2, ensure_ascii=False))
                    print("=============================")
                result = evaluator.evaluate_mcq(question_data)
                print(f"评估结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                # 多题：一次只能提交一题，--parallel 个进程并行
                from concurrent.futures import ThreadPoolExecutor, as_completed
                n_total = len(items)
                # 按本题集最大「题号+(标准,难度)」长度固定 label 宽度，使 score/error 列对齐
                label_width = max(
                    len(f"题{i+1:>3} ({(q.get('standard') or '').replace('CCSS.ELA-LITERACY.', '')}, {q.get('difficulty') or 'medium'})")
                    for i, q in enumerate(items)
                )
                print(f"[评估] {n_total} 题，每次 1 题，{parallel} 并行，超时 {timeout}s/题")
                evaluator = InceptBenchEvaluator(api_key=args.api_key, timeout=timeout)
                results_by_idx = {}
                done = 0
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futures = {ex.submit(evaluator.evaluate_mcq, q): i for i, q in enumerate(items)}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        q = items[i]
                        std = (q.get("standard") or "").replace("CCSS.ELA-LITERACY.", "")
                        diff = q.get("difficulty") or "medium"
                        try:
                            r = fut.result()
                            results_by_idx[i] = r
                            s = r.get("overall_score")
                            if s is None and "evaluations" in r:
                                ev = next(iter(r.get("evaluations", {}).values()), {})
                                s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
                            done += 1
                            progress = f"{done:>3}/{n_total}"
                            label = f"题{i+1:>3} ({std}, {diff})".ljust(label_width)
                            if isinstance(s, (int, float)):
                                sf = float(s)
                                status = f"score={sf:.2f}"
                                if sf < 0.85:
                                    status += "  X"
                                print(f"[评估] [{progress}] {label}: {status}")
                            else:
                                err_reason = r.get("message") or r.get("status") or ""
                                if r.get("errors"):
                                    err_reason = err_reason or str(r.get("errors"))[:80]
                                if not err_reason and "evaluations" in r:
                                    for ev in (r.get("evaluations") or {}).values():
                                        if ev.get("errors"):
                                            err_reason = str(ev.get("errors"))[:80]
                                            break
                                status = "error (原因: " + (err_reason[:100] if err_reason else "未知") + ")"
                                print(f"[评估] [{progress}] {label}: {status}")
                        except Exception as e:
                            results_by_idx[i] = {"overall_score": 0.0, "status": "error", "message": str(e)}
                            done += 1
                            progress = f"{done:>3}/{n_total}"
                            label = f"题{i+1:>3} ({std}, {diff})".ljust(label_width)
                            print(f"[评估] [{progress}] {label}: error (原因: {e})")

                # 总评估数 = 提交的题目数；有效题目数 = 拿到数值分数的题目数；无分数题 = 总评估数 - 有效题目数
                # scores 与 items 按下标对齐，无分数题为 None，便于下游 improve-examples 等按 index 取分
                n_submitted = len(items)
                scores = [None] * n_submitted
                error_details = []  # 无分数题目明细，用于排查是题目问题还是评分服务问题
                for i in range(len(items)):
                    r = results_by_idx.get(i, {})
                    q = items[i]
                    s = r.get("overall_score")
                    if s is None and "evaluations" in r:
                        ev = next(iter(r.get("evaluations", {}).values()), {})
                        s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
                    if isinstance(s, (int, float)):
                        scores[i] = float(s)
                    else:
                        reason = r.get("message") or r.get("status") or ""
                        if r.get("errors"):
                            reason = reason or str(r.get("errors"))[:200]
                        if not reason and "evaluations" in r:
                            for ev in (r.get("evaluations") or {}).values():
                                if ev.get("errors"):
                                    reason = str(ev.get("errors"))[:200]
                                    break
                        reason = (reason or "未知").strip()
                        # 判定：服务端问题（DB/500/超时等） vs 题目或请求问题
                        reason_lower = reason.lower()
                        if any(x in reason_lower for x in (
                            "could not save", "save evaluation", "db", "database", "500", "503", "timeout",
                            "服务端", "server error", "internal", "overall"
                        )):
                            classification = "service"
                        else:
                            classification = "question_or_request"
                        error_details.append({
                            "index": i,
                            "question_id": q.get("id"),
                            "standard": q.get("standard"),
                            "difficulty": q.get("difficulty"),
                            "reason": reason[:300],
                            "classification": classification,
                        })

                n_valid = sum(1 for s in scores if s is not None)
                n_error = n_submitted - n_valid

                if n_valid:
                    valid_scores = [s for s in scores if s is not None]
                    avg = sum(valid_scores) / len(valid_scores)
                    passed = sum(1 for s in scores if s is not None and s >= 0.85)
                    print(f"\n[评估] === 汇总 ===")
                    print(f"[评估] 总评估数: {n_submitted}（提交给评分服务的题目数）")
                    print(f"[评估] 有效分数数: {n_valid}（拿到数值分数的题目数）")
                    if n_error:
                        print(f"[评估] 无分数: {n_error} 题（见下方明细，需区分服务端问题与题目问题）")
                    print(f"[评估] 通过率(>=0.85，按有效题目): {passed}/{n_valid} ({100*passed/n_valid:.1f}%)")
                    if n_submitted > n_valid:
                        print(f"[评估] 若将无分数题按不通过计: {passed}/{n_submitted} ({100*passed/n_submitted:.1f}%)")
                    print(f"[评估] 平均分: {avg:.2f}")
                    if error_details:
                        print(f"\n[评估] 无分数题目明细（建议先排查「服务端」再重试或联系评分方）:")
                        for e in error_details:
                            std_short = (e.get("standard") or "").replace("CCSS.ELA-LITERACY.", "")
                            kind = "服务端" if e.get("classification") == "service" else "题目/请求"
                            print(f"[评估]   题{e['index']+1:>3} ({std_short}, {e.get('difficulty','')}): {e.get('reason','')[:80]}... 判定: {kind}")
                else:
                    print("\n[评估] 无有效分数")

                if args.output:
                    out_data = {
                        "total": n_submitted,
                        "total_submitted": n_submitted,
                        "valid_score_count": n_valid,
                        "error_count": n_error,
                        "scores": scores,
                        "results": [results_by_idx.get(i) for i in range(len(items))],
                    }
                    if error_details:
                        out_data["error_details"] = error_details
                    if n_valid:
                        out_data["avg_score"] = round(sum(valid_scores) / n_valid, 2)
                        out_data["pass_count"] = passed
                        out_data["pass_rate"] = round(100 * out_data["pass_count"] / n_valid, 1)
                        out_data["pass_rate_if_errors_as_fail"] = round(100 * out_data["pass_count"] / n_submitted, 1)
                    with open(args.output, 'w') as f:
                        json.dump(out_data, f, indent=2, ensure_ascii=False)
                    print(f"[评估] 已保存: {args.output}")
                    if error_details:
                        err_path = args.output.replace(".json", "_errors.json")
                        if err_path == args.output:
                            err_path = os.path.join(os.path.dirname(args.output), "evaluation_errors.json")
                        with open(err_path, 'w') as f:
                            json.dump({"total_submitted": n_submitted, "error_count": n_error, "error_details": error_details}, f, indent=2, ensure_ascii=False)
                        print(f"[评估] 无分数题明细已保存: {err_path}（可据此排查服务端 vs 题目问题）")

    elif args.command == "closed-loop":
        project_root = os.path.dirname(os.path.abspath(__file__))
        # 使用按模型隔离的 examples/prompt_rules，避免多终端分别跑不同模型时互相覆盖
        _run_closed_loop_one_model(project_root, getattr(args, "model", "deepseek-chat"), args, use_model_specific_paths=True, run_id=getattr(args, "run_id", None))

    elif args.command == "closed-loop-multi":
        project_root = os.path.dirname(os.path.abspath(__file__))
        models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
        if not models:
            print("错误: --models 不能为空，例如 --models deepseek-chat,local,kimi-latest")
            sys.exit(1)
        pass_rate_target = getattr(args, "pass_rate_target", 95.0)
        max_rounds = getattr(args, "max_rounds", 10)
        results_list = []
        for model in models:
            one = _run_closed_loop_one_model(project_root, model, args, use_model_specific_paths=True, run_id=getattr(args, "run_id", None))
            results_list.append(one)
        # 汇总：控制台表格 + JSON 文件
        summary = {
            "generated_at": datetime.now().isoformat(),
            "pass_rate_target": pass_rate_target,
            "max_rounds": max_rounds,
            "models": results_list,
        }
        # 控制台打印
        print("\n" + "=" * 72)
        print("多模型闭环汇总（通过率比较）")
        print("=" * 72)
        print(f"{'模型':<28} {'最终':>6} {'最高':>6} {'达标':>4} {'轮数':>4} {'通过/有效/提交':>16}  备注")
        print("-" * 72)
        for r in results_list:
            rate = r.get("final_pass_rate")
            rate_s = f"{rate:.1f}%" if rate is not None else "N/A"
            best = r.get("best_pass_rate")
            best_s = f"{best:.1f}%" if best is not None else "N/A"
            ok = "是" if r.get("target_reached") else "否"
            rd = r.get("round_reached") or 0
            cnt = f"{r.get('pass_count') or 0}/{r.get('n_valid') or 0}/{r.get('n_submitted') or 0}"
            err = (r.get("error") or "")[:18]
            print(f"{r['model']:<28} {rate_s:>6} {best_s:>6} {ok:>4} {rd:>4} {cnt:>16}  {err}")
        print("=" * 72)
        # 保存 JSON
        out_path = getattr(args, "summary_output", None)
        if not out_path:
            out_path = os.path.join(project_root, "evaluation_output", "closed_loop_multi_summary.json")
        else:
            out_path = os.path.join(project_root, out_path) if not os.path.isabs(out_path) else out_path
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n汇总已保存: {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()