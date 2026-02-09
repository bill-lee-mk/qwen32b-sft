# -*- coding: utf-8 -*-

"""
全参数微调主入口
支持SFT + DPO两阶段训练
"""
import os
import argparse
import logging
import warnings
import torch
from typing import Optional

# 分布式训练时，非主进程禁用 datasets/huggingface 进度条，避免 8 份重复输出
if int(os.environ.get("LOCAL_RANK", 0)) != 0:
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

# 抑制 TensorFlow 日志（TensorBoard 会拉取 TF，触发 oneDNN 等 print）
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# 抑制 DeepSpeed+Trainer 下 lr_scheduler.step() 与 optimizer.step() 顺序警告
warnings.filterwarnings(
    "ignore",
    message=r"Detected call of.*lr_scheduler.*before.*optimizer",
    category=UserWarning,
)
# 抑制 google.api_core Python 版本 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")
# 抑制 torch_dtype 弃用提醒（来自第三方库）
warnings.filterwarnings("ignore", message=r".*torch_dtype.*deprecated.*dtype.*")
# 抑制 tokenizer PAD/BOS/EOS 对齐提示（各 rank 重复打印）
warnings.filterwarnings("ignore", message=r".*tokenizer has new PAD/BOS/EOS tokens.*")
# 抑制 accelerate 梯度累积步数不一致提示（DeepSpeed 会覆盖）
warnings.filterwarnings("ignore", message=r".*Gradient accumulation steps mismatch.*")

from .config import TrainingPipelineConfig, load_config_from_yaml
from .sft_trainer import SFTTrainer
from .dpo_trainer import DPOTrainerWrapper

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 降低 transformers/httpx 日志级别，减少 "tokenizer has new PAD"、HTTP Request 等重复输出
for _name in ("transformers", "httpx", "transformers.tokenization_utils_base"):
    logging.getLogger(_name).setLevel(logging.ERROR)


class FullParameterFinetuner:
    """全参数微调器"""
    
    def __init__(self, config: TrainingPipelineConfig):
        self.config = config
        
        # 检查CUDA可用性
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，将使用CPU训练（速度会很慢）")
        
        # 创建必要的目录
        os.makedirs("models", exist_ok=True)
        os.makedirs(self.config.final_model_dir, exist_ok=True)
    
    def run_sft(self) -> Optional[str]:
        """运行SFT训练"""
        if not self.config.sft_enabled:
            logger.info("SFT训练已禁用")
            return None
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("SFT 训练阶段")
        try:
            # 初始化SFT训练器
            sft_trainer = SFTTrainer(self.config.sft_config, self.config.model_config)
            
            # 运行SFT训练
            sft_result = sft_trainer.run(self.config.sft_data_path)
            
            sft_model_path = self.config.sft_config.output_dir
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(f"SFT 完成，输出: {sft_model_path}")
            
            return sft_model_path
            
        except Exception as e:
            logger.error(f"SFT训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_dpo(self, sft_model_path: Optional[str] = None):
        """运行DPO训练"""
        if not self.config.dpo_enabled:
            logger.info("DPO训练已禁用")
            return None
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info("DPO 训练阶段")
        try:
            # 初始化DPO训练器
            dpo_trainer = DPOTrainerWrapper(self.config.dpo_config, self.config.model_config)
            
            # 运行DPO训练（使用SFT模型或基础模型）
            model_path = sft_model_path or self.config.model_config.model_name
            dpo_result = dpo_trainer.run(self.config.dpo_data_path, model_path)
            
            dpo_model_path = self.config.dpo_config.output_dir
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(f"DPO 完成，输出: {dpo_model_path}")
            
            return dpo_model_path
            
        except Exception as e:
            logger.error(f"DPO训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_model_source(self, base_path: str) -> Optional[str]:
        """在目录中查找模型：优先根目录，否则用最新 checkpoint"""
        if not base_path or not os.path.exists(base_path):
            return None
        # 根目录有模型文件
        for name in ("model.safetensors", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "pytorch_model.bin", "config.json"):
            if os.path.isfile(os.path.join(base_path, name)):
                return base_path
        # 查找最新 checkpoint（按步数排序）
        import glob
        import re
        checkpoints = glob.glob(os.path.join(base_path, "checkpoint-*"))
        if checkpoints:
            def step_num(p):
                m = re.search(r"checkpoint-(\d+)$", p)
                return int(m.group(1)) if m else 0
            return max(checkpoints, key=step_num)
        return None

    def merge_and_save_final_model(self, dpo_model_path: Optional[str] = None):
        """将训练好的模型导出到 final_model 目录（复制，非参数合并）"""
        logger.info("导出最终模型...")
        
        # 确定源路径：优先 DPO，否则 SFT
        base_path = dpo_model_path or self.config.dpo_config.output_dir
        final_source_path = self._find_model_source(base_path)
        if not final_source_path:
            base_path = self.config.sft_config.output_dir
            final_source_path = self._find_model_source(base_path)
        if not final_source_path:
            logger.error("没有可用的训练模型（请检查 dpo_model 或 sft_model 目录）")
            return False
        
        logger.info(f"使用模型: {final_source_path}")
        
        try:
            import shutil
            
            if os.path.exists(self.config.final_model_dir):
                shutil.rmtree(self.config.final_model_dir)
            
            shutil.copytree(final_source_path, self.config.final_model_dir)
            
            # 添加配置文件
            config_info = {
                "training_pipeline": {
                    "sft_enabled": self.config.sft_enabled,
                    "dpo_enabled": self.config.dpo_enabled,
                    "sft_model_path": self.config.sft_config.output_dir if self.config.sft_enabled else None,
                    "dpo_model_path": self.config.dpo_config.output_dir if self.config.dpo_enabled else None,
                },
                "model": {
                    "base_model": self.config.model_config.model_name,
                    "fine_tuned_for": "K-12 ELA MCQ Generation",
                    "training_date": os.path.getmtime(final_source_path),
                }
            }
            
            import json
            config_path = os.path.join(self.config.final_model_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(config_info, f, indent=2)
            
            logger.info(f"最终模型已保存到: {self.config.final_model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")
            return False
    
    def run(self):
        """运行完整的训练流水线"""
        logger.info("开始全参数微调训练流水线")
        
        # 1. SFT训练
        sft_model_path = self.run_sft()
        
        # 2. DPO训练（在SFT模型基础上）
        dpo_model_path = self.run_dpo(sft_model_path)
        
        # 3. 保存最终模型
        success = self.merge_and_save_final_model(dpo_model_path)
        
        if success:
            logger.info("训练流水线完成!")
            logger.info(f"最终模型位置: {self.config.final_model_dir}")
        else:
            logger.error("训练流水线失败")
        
        return success


def main(args=None):
    """
    训练主函数
    支持：
      1）--sft-only       只跑 SFT
      2）--dpo-only       只跑 DPO（可配合 --sft-model）
      3）默认             跑完整流水线 SFT + DPO
    """
    # 1. 解析命令行参数（兼容 deepspeed 的 --local_rank）
    if args is None:
        parser = argparse.ArgumentParser(description="全参数微调（SFT + DPO）")
        parser.add_argument(
            "--config",
            type=str,
            default="configs/training_config.yaml",
            help="配置文件路径"
        )
        parser.add_argument(
            "--data",
            type=str,
            default=None,
            help="（可选）训练数据路径，通常使用 config 里的路径"
        )
        parser.add_argument(
            "--sft-only",
            action="store_true",
            help="仅运行 SFT 阶段"
        )
        parser.add_argument(
            "--dpo-only",
            action="store_true",
            help="仅运行 DPO 阶段"
        )
        parser.add_argument(
            "--sft-model",
            type=str,
            default=None,
            help="DPO 阶段使用的 SFT 模型路径（为空则用基础模型）"
        )
        # Deepspeed 会传这个参数，必须接受，否则 argparse 会报错
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="deepspeed / torch.distributed 用的本地 rank（训练逻辑里可以不用显式处理）"
        )
        args = parser.parse_args()

    # 2. 加载配置
    config = load_config_from_yaml(args.config)

    # 使用 getattr 防御性编程：main.py 子命令传入的 args 可能缺少这些属性
    sft_only = getattr(args, 'sft_only', False)
    dpo_only = getattr(args, 'dpo_only', False)

    # 2.1 命令行 --data 覆盖 config 中的路径
    data_path = getattr(args, 'data', None)
    if data_path:
        if sft_only:
            config.sft_data_path = data_path
        elif dpo_only:
            config.dpo_data_path = data_path

    # 3. 根据模式分支处理（仅主进程打印，避免多 GPU 重复）
    def _log_main(msg: str):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(msg)

    if sft_only:
        _log_main("SFT 训练开始")
        sft_trainer = SFTTrainer(config.sft_config, config.model_config)
        sft_trainer.run(config.sft_data_path)
        _log_main(f"SFT 完成，模型: {config.sft_config.output_dir}")
        return

    if dpo_only:
        _log_main("DPO 训练开始")
        dpo_trainer = DPOTrainerWrapper(config.dpo_config, config.model_config)
        model_path = getattr(args, 'sft_model', None) or config.model_config.model_name
        dpo_trainer.run(config.dpo_data_path, model_path)
        _log_main(f"DPO 完成，模型: {config.dpo_config.output_dir}")
        return

    _log_main("训练流水线开始（SFT + DPO）")
    finetuner = FullParameterFinetuner(config)
    finetuner.run()
    _log_main("训练流水线完成")


if __name__ == "__main__":
    main()
