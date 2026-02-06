# -*- coding: utf-8 -*-

"""
SFT（监督微调）训练器
"""
import math
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import Dataset
import os
from typing import Optional, Dict, Any
import logging

from .config import SFTTrainingConfig, ModelConfig
from data_processing.dataset_formatter import create_sft_dataset

logger = logging.getLogger(__name__)


class StepTimingCallback(TrainerCallback):
    """每步完成后将耗时加入训练日志"""
    def __init__(self):
        self._step_start = None
        self._last_step_duration = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self._last_step_duration = time.perf_counter() - self._step_start

    def on_log(self, args, state, control, logs=None, **kwargs):
        """将本步耗时加入日志输出"""
        if logs is not None and self._last_step_duration is not None:
            logs["step_duration_sec"] = round(self._last_step_duration, 3)


class SFTTrainer:
    """SFT训练器"""
    
    def __init__(self, config: SFTTrainingConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 初始化模型和tokenizer
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model_and_tokenizer(self):
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
        
        # 如果使用DeepSpeed，device_map应该为None（让DeepSpeed管理）
        device_map = None if self.config.deepspeed else self.model_config.device_map

        # 重要：设置Flash Attention配置
        flash_attention_config = {
            "use_flash_attention_2": False,    # 禁用Flash Attention 2
            "use_flash_attention_3": True,     # 启用Flash Attention 3
        }
        
        # 加载模型，启用Flash Attention 3
        logger.info("启用Flash Attention 3...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,    # DeepSpeed时设为None
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.model_revision,
            attn_implementation="flash_attention_3" if self.model_config.use_flash_attention else "eager",  
        )
        
        # 检查Flash Attention可否导入
        if self.model_config.use_flash_attention:
            try:
                # from flash_attn import flash_attn_qkvpacked_func
                import flash_attn_interface
                logger.info(f"Flash Attention 3 可用: {flash_attn_interface.__file__}")
            except ImportError:
                logger.warning("Flash Attention 3不可用，请从:/home/ubuntu/flash-attention/hopper/ 安装")
        
        
        # 启用梯度检查点
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点")
        
        
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
        
        return self.model, self.tokenizer
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """准备训练数据集"""
        logger.info(f"准备SFT数据集: {data_path}")
        
        dataset = create_sft_dataset(
            self.tokenizer,
            data_path,
            max_length=getattr(self.config, "max_length", 2048)
        )
        
        logger.info(f"数据集大小: {len(dataset)}")
        return dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """执行训练"""
        logger.info("开始SFT训练...")
        # 打印训练参数与总步数计算公式（仅主进程）
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            per_device = self.config.per_device_train_batch_size
            grad_accum = self.config.gradient_accumulation_steps
            eff_batch = per_device * world_size * grad_accum
            steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
            total_steps_est = steps_per_epoch * self.config.num_train_epochs
            logger.info("=" * 50 + " SFT 训练参数 " + "=" * 50)
            for k, v in vars(self.config).items():
                logger.info(f"  {k}: {v}")
            logger.info("-" * 50)
            logger.info("总步数计算公式:")
            logger.info(f"  有效batch = per_device_batch_size × WORLD_SIZE × gradient_accumulation_steps")
            logger.info(f"            = {per_device} × {world_size} × {grad_accum} = {eff_batch}")
            logger.info(f"  每epoch步数 = ceil(样本数 / 有效batch) = ceil({len(train_dataset)} / {eff_batch}) = {steps_per_epoch}")
            logger.info(f"  预估总步数 = 每epoch步数 × num_train_epochs = {steps_per_epoch} × {self.config.num_train_epochs} = {total_steps_est}")
            logger.info("=" * 50)
        
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
            eval_strategy=self.config.eval_strategy,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim=self.config.optim,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            group_by_length=self.config.group_by_length,
            report_to=self.config.report_to,
            deepspeed=self.config.deepspeed,
            save_strategy="steps",
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        
        # 数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建Trainer（添加每步耗时 callback）
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[StepTimingCallback()],
        )
        
        # 训练
        train_result = self.trainer.train()
        
        # 保存最终模型（会保存到output_dir，而不是checkpoint目录）
        self.trainer.save_model()
        self.trainer.save_state()
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # 打印总步数与每步耗时（仅主进程）
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            total_steps = self.trainer.state.global_step
            train_runtime = metrics.get("train_runtime", 0)
            time_per_step = train_runtime / total_steps if total_steps > 0 else 0
            logger.info("=" * 50 + " SFT 训练统计 " + "=" * 50)
            logger.info(f"  总步数: {total_steps}")
            logger.info(f"  总耗时: {train_runtime:.1f} 秒 ({train_runtime/3600:.2f} 小时)")
            logger.info(f"  每步平均耗时: {time_per_step:.3f} 秒")
            logger.info("=" * 50)
        
        # 检查checkpoint目录（用于信息提示）
        # 注意：如果save_total_limit=2，旧的checkpoint应该已经被自动清理
        # checkpoint目录包含训练中间状态，可用于恢复训练
        import glob
        checkpoint_dirs = glob.glob(os.path.join(self.config.output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            logger.info(f"发现 {len(checkpoint_dirs)} 个checkpoint目录")
            logger.info("最终模型已保存到output_dir，checkpoint目录可保留用于恢复训练，或手动删除")
        
        logger.info(f"SFT训练完成! 最终模型保存在: {self.config.output_dir}")
        logger.info(f"注意：checkpoint目录（如checkpoint-2000）包含训练中间状态，用于恢复训练")
        logger.info(f"用于后续DPO训练的应该是: {self.config.output_dir}（最终模型）")
        
        return train_result
    
    def run(self, data_path: str):
        """运行完整的SFT训练流程"""
        # 1. 加载模型和tokenizer
        self.load_model_and_tokenizer()
        
        # 2. 准备数据集
        train_dataset = self.prepare_dataset(data_path)
        
        # 3. 训练
        train_result = self.train(train_dataset)
        
        return train_result
