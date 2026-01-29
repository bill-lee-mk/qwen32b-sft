# -*- coding: utf-8 -*-

"""
全参数微调主入口
支持SFT + DPO两阶段训练
"""
import os
import logging
import torch
from typing import Optional

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
        
        logger.info("=" * 50)
        logger.info("开始SFT训练阶段")
        logger.info("=" * 50)
        
        try:
            # 初始化SFT训练器
            sft_trainer = SFTTrainer(self.config.sft_config, self.config.model_config)
            
            # 运行SFT训练
            sft_result = sft_trainer.run(self.config.sft_data_path)
            
            # 返回SFT模型路径
            sft_model_path = self.config.sft_config.output_dir
            logger.info(f"SFT训练完成，模型路径: {sft_model_path}")
            
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
        
        logger.info("=" * 50)
        logger.info("开始DPO训练阶段")
        logger.info("=" * 50)
        
        try:
            # 初始化DPO训练器
            dpo_trainer = DPOTrainerWrapper(self.config.dpo_config, self.config.model_config)
            
            # 运行DPO训练（使用SFT模型或基础模型）
            model_path = sft_model_path or self.config.model_config.model_name
            dpo_result = dpo_trainer.run(self.config.dpo_data_path, model_path)
            
            # 返回DPO模型路径
            dpo_model_path = self.config.dpo_config.output_dir
            logger.info(f"DPO训练完成，模型路径: {dpo_model_path}")
            
            return dpo_model_path
            
        except Exception as e:
            logger.error(f"DPO训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_and_save_final_model(self, dpo_model_path: Optional[str] = None):
        """合并并保存最终模型"""
        logger.info("保存最终模型...")
        
        # 确定最终模型路径
        if dpo_model_path and os.path.exists(dpo_model_path):
            final_source_path = dpo_model_path
            logger.info(f"使用DPO模型作为最终模型: {final_source_path}")
        elif self.config.sft_config.output_dir and os.path.exists(self.config.sft_config.output_dir):
            final_source_path = self.config.sft_config.output_dir
            logger.info(f"使用SFT模型作为最终模型: {final_source_path}")
        else:
            logger.error("没有可用的训练模型")
            return False
        
        try:
            # 复制模型文件到最终目录
            import shutil
            
            # 如果目标目录已存在，先删除
            if os.path.exists(self.config.final_model_dir):
                shutil.rmtree(self.config.final_model_dir)
            
            # 复制模型文件
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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="全参数微调Qwen模型用于MCQ生成")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="配置文件路径")
    parser.add_argument("--sft-only", action="store_true",
                       help="仅运行SFT训练")
    parser.add_argument("--dpo-only", action="store_true",
                       help="仅运行DPO训练（需要已训练的SFT模型）")
    parser.add_argument("--sft-model", type=str,
                       help="DPO训练的SFT模型路径")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config_from_yaml(args.config)
    
    # 根据参数调整配置
    if args.sft_only:
        config.dpo_enabled = False
    elif args.dpo_only:
        config.sft_enabled = False
    
    # 初始化训练器
    trainer = FullParameterFinetuner(config)
    
    # 运行训练
    if args.dpo_only and args.sft_model:
        # 仅运行DPO训练
        dpo_model_path = trainer.run_dpo(args.sft_model)
        trainer.merge_and_save_final_model(dpo_model_path)
    else:
        # 运行完整流水线
        trainer.run()


if __name__ == "__main__":
    main()
