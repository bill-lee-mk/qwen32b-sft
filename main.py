
## 8. 主程序入口

### `main.py`


"""
主程序入口
支持数据处理、训练、API服务等多种功能
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    improve_parser.add_argument("--max-candidates-per-pair", type=int, default=5, help="每个组合最多试评的候选数（减少可加快）")
    improve_parser.add_argument("--parallel", type=int, default=20, help="InceptBench 并行评分数")
    improve_parser.add_argument("--timeout", type=int, default=180, help="单题超时秒数")
    
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
            max_candidates_per_pair=getattr(args, "max_candidates_per_pair", 5),
            parallel=args.parallel,
            timeout=args.timeout,
        )
        if "error" in report:
            print(f"错误: {report['error']}")
            sys.exit(1)
        print(f"闭环完成: 失败组合 {report.get('failed_pairs', 0)} 个, 评分 {report.get('evaluated', 0)} 条, 保留 {report.get('kept', 0)} 条, examples 共 {report.get('examples_count', 0)} 条")

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
                print(f"评估 {len(items)} 题，每次 1 题，{parallel} 并行，超时 {timeout}s/题")
                evaluator = InceptBenchEvaluator(api_key=args.api_key, timeout=timeout)
                results_by_idx = {}
                done = 0
                with ThreadPoolExecutor(max_workers=parallel) as ex:
                    futures = {ex.submit(evaluator.evaluate_mcq, q): i for i, q in enumerate(items)}
                    for fut in as_completed(futures):
                        i = futures[fut]
                        try:
                            r = fut.result()
                            results_by_idx[i] = r
                            s = r.get("overall_score")
                            if s is None and "evaluations" in r:
                                ev = next(iter(r.get("evaluations", {}).values()), {})
                                s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
                            done += 1
                            status = f"score={s:.2f}" if isinstance(s, (int, float)) else "error"
                            print(f"  [{done}/{len(items)}] 题{i+1}: {status}")
                        except Exception as e:
                            results_by_idx[i] = {"overall_score": 0.0, "status": "error", "message": str(e)}
                            done += 1
                            print(f"  [{done}/{len(items)}] 题{i+1}: error {e}")

                scores = []
                for i in range(len(items)):
                    r = results_by_idx.get(i, {})
                    s = r.get("overall_score")
                    if s is None and "evaluations" in r:
                        ev = next(iter(r.get("evaluations", {}).values()), {})
                        s = (ev.get("inceptbench_new_evaluation") or {}).get("overall", {}).get("score")
                    if isinstance(s, (int, float)):
                        scores.append(float(s))

                if scores:
                    avg = sum(scores) / len(scores)
                    passed = sum(1 for s in scores if s >= 0.85)
                    print(f"\n=== 汇总 ===")
                    print(f"平均分: {avg:.2f}")
                    print(f"通过率(>=0.85): {passed}/{len(scores)} ({100*passed/len(scores):.1f}%)")
                else:
                    print("\n无有效分数")

                if args.output:
                    out_data = {
                        "total": len(items),
                        "scores": scores,
                        "results": [results_by_idx.get(i) for i in range(len(items))],
                    }
                    if scores:
                        out_data["avg_score"] = round(sum(scores) / len(scores), 2)
                        out_data["pass_count"] = sum(1 for s in scores if s >= 0.85)
                        out_data["pass_rate"] = round(100 * out_data["pass_count"] / len(scores), 1)
                    with open(args.output, 'w') as f:
                        json.dump(out_data, f, indent=2, ensure_ascii=False)
                    print(f"已保存: {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()