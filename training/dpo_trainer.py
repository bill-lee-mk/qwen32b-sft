# -*- coding: utf-8 -*-

"""
DPO（直接偏好优化）训练器
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from datasets import Dataset
import os
import logging
from typing import Optional

from .config import DPOTrainingConfig, ModelConfig
from data_processing.dataset_formatter import create_dpo_dataset

logger = logging.getLogger(__name__)


class DPOTrainerWrapper:
    """DPO训练器封装"""
    
    def __init__(self, config: DPOTrainingConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model_and_tokenizer(self, model_path: Optional[str] = None):
        """加载模型和tokenizer"""
        logger.info("加载模型和tokenizer...")
        
        # 确定torch数据类型
        if self.model_config.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logger.info("使用bfloat16精度")
        elif self.model_config.torch_dtype == "float16":
            torch_dtype = torch.float16
            logger.info("使用float16精度")
        else:
            torch_dtype = torch.float32
            logger.info("使用float32精度")
        
        # 加载tokenizer
        tokenizer_name = self.model_config.tokenizer_name or self.model_config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.model_config.trust_remote_code,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型（从SFT模型或基础模型）
        model_path = model_path or self.model_config.model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=self.model_config.device_map,
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.model_revision
        )
        
        # 启用梯度检查点
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点")
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """准备DPO数据集"""
        logger.info(f"准备DPO数据集: {data_path}")
        
        dataset = create_dpo_dataset(
            self.tokenizer,
            data_path,
            max_length=self.config.max_length
        )
        
        logger.info(f"数据集大小: {len(dataset)}")
        return dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """执行DPO训练"""
        logger.info("开始DPO训练...")
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy=self.config.evaluation_strategy,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            report_to=self.config.report_to,
            deepspeed=self.config.deepspeed,
            save_strategy="steps",
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            load_best_model_at_end=False,
        )
        
        # 创建DPOTrainer
        self.trainer = DPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
        )
        
        # 训练
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        self.trainer.save_state()
        
        logger.info(f"DPO训练完成! 模型保存在: {self.config.output_dir}")
        
        return train_result
    
    def run(self, data_path: str, model_path: Optional[str] = None):
        """运行完整的DPO训练流程"""
        # 1. 加载模型和tokenizer
        self.load_model_and_tokenizer(model_path)
        
        # 2. 准备数据集
        train_dataset = self.prepare_dataset(data_path)
        
        # 3. 训练
        train_result = self.train(train_dataset)
        
        return train_result
