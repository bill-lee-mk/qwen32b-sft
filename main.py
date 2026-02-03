
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
  
  # SFT训练
  python main.py train-sft
  
  # DPO训练
  python main.py train-dpo
  
  # 完整训练
  python main.py train-all
  
  # 启动API服务
  python main.py serve-api
  
  # 评估模型
  python main.py evaluate --input sample_questions.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 数据处理命令
    process_parser = subparsers.add_parser("process-data", help="处理训练数据")
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
    
    # API服务命令
    api_parser = subparsers.add_parser("serve-api", help="启动API服务")
    api_parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    api_parser.add_argument("--port", type=int, default=8000, help="端口号")
    api_parser.add_argument("--model", default="models/final_model", help="模型路径")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--input", required=True, help="输入文件或目录")
    eval_parser.add_argument("--output", help="输出文件")
    eval_parser.add_argument("--api-key", help="InceptBench API密钥")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # 执行命令
    if args.command == "process-data":
        from data_processing.data_processor import main as process_data_main
        process_data_main()
        
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
        process_data_main()
        
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
        
        print("\n=== 训练完成 ===")
        print("最终模型保存在: models/final_model")
        
    elif args.command == "serve-api":
        # 设置模型路径环境变量
        os.environ["MODEL_PATH"] = args.model
        
        from api_service.fastapi_app import run_api_server
        run_api_server(host=args.host, port=args.port)
        
    elif args.command == "evaluate":
        from evaluation.inceptbench_client import InceptBenchEvaluator
        
        evaluator = InceptBenchEvaluator(api_key=args.api_key)
        
        # 评估输入文件或目录
        if os.path.isdir(args.input):
            # 评估目录中的所有文件
            results = []
            for file_name in os.listdir(args.input):
                if file_name.endswith('.json'):
                    file_path = os.path.join(args.input, file_name)
                    with open(file_path, 'r') as f:
                        question_data = json.load(f)
                    result = evaluator.evaluate_mcq(question_data)
                    results.append(result)
                    
                    print(f"评估 {file_name}: {result.get('overall_score', 'N/A')}")
        else:
            # 评估单个文件
            with open(args.input, 'r') as f:
                question_data = json.load(f)
            result = evaluator.evaluate_mcq(question_data)
            print(f"评估结果: {json.dumps(result, indent=2)}")
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()