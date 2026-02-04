# -*- coding: utf-8 -*-

"""
DPO（直接偏好优化）训练器
"""

import os
os.environ["TENSORBOARD_LOGGING_DIR"] = "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/dpo_model/logs"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # TrainingArguments,
)
from trl import DPOTrainer, DPOConfig 
from datasets import Dataset
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
        
        # 如果使用DeepSpeed，device_map应该为None
        # device_map = None if self.config.deepspeed else self.model_config.device_map
        
        device_map = None
        
        
        # 加载模型（从SFT模型或基础模型）
        model_path = model_path or self.model_config.model_name
        
        # Load the active model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,    # DeepSpeed时设为None
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.model_revision,
            attn_implementation="flash_attention_3" if self.model_config.use_flash_attention else "eager", 
        )
        
        
        # Load the reference model for DeepSpeed ZeRO-3
        # 关键优化：将ref_model offload到CPU，避免OOM
        # 因为ref_model在训练时不需要梯度，可以放在CPU上
        logger.info("Loading reference model (offloaded to CPU to save GPU memory)...")
        if self.config.deepspeed:
            # 使用DeepSpeed时，ref_model可以offload到CPU
            # DeepSpeed会自动处理CPU-GPU数据传输
            ref_device_map = {"": "cpu"}  # 强制加载到CPU
            logger.info("Reference model will be offloaded to CPU (DeepSpeed will handle transfers)")
        else:
            # 不使用DeepSpeed时，如果显存足够可以放在GPU
            # 但为了安全，还是建议放在CPU
            ref_device_map = {"": "cpu"}
            logger.info("Reference model offloaded to CPU to prevent OOM")
        
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=ref_device_map,  # 加载到CPU，避免OOM
            trust_remote_code=self.model_config.trust_remote_code,
            revision=self.model_config.model_revision,
            attn_implementation="flash_attention_3" if self.model_config.use_flash_attention else "eager", 
        )
        
        # Ensure reference is in eval mode (不需要梯度)
        self.ref_model.eval()
        # 禁用ref_model的梯度计算，进一步节省显存
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        
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
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=int(self.config.warmup_steps),
            logging_steps=int(self.config.logging_steps),
            save_steps=self.config.save_steps,
            
            beta=float(self.config.beta),
            loss_type=self.config.loss_type,
            max_length=self.config.max_length,
            max_prompt_length=int(self.config.max_prompt_length),
                        
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
            #remove_unused_columns=self.config.remove_unused_columns,
            
            remove_unused_columns=False,
            gradient_checkpointing=True, # Critical for 32B
            
            report_to=self.config.report_to,
            deepspeed=self.config.deepspeed,
            save_strategy="steps",
            # logging_dir=os.path.join(self.config.output_dir, "logs"),
            load_best_model_at_end=False,
            
            # logging_steps=1,
            
            ddp_timeout=7200, # Increase to 2 hours (ZeRO-3 offload需要更长时间)
            dataloader_pin_memory=False,  # 禁用pin_memory，减少显存占用
            dataloader_persistent_workers=False,  # 禁用持久化workers，避免死锁
        )
        
        # 创建DPOTrainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            
            # ref_model=None,
            
            args=training_args,
            train_dataset=train_dataset,
            
            # model_init_kwargs={"device_map": "auto"} # This helps TRL shard properly
            
        )
        
        # 训练（添加异常处理和checkpoint恢复）
        try:
            train_result = self.trainer.train(resume_from_checkpoint=None)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                logger.error("训练过程中发生OOM错误，建议：")
                logger.error("1. 检查ref_model是否已offload到CPU")
                logger.error("2. 减少batch_size或增加gradient_accumulation_steps")
                logger.error("3. 检查DeepSpeed ZeRO-3配置")
                logger.error("4. 查看最近的checkpoint是否可以恢复训练")
                # 尝试从最新checkpoint恢复
                import glob
                checkpoint_dirs = sorted(glob.glob(os.path.join(self.config.output_dir, "checkpoint-*")), 
                                       key=lambda x: int(x.split("-")[-1]))
                if checkpoint_dirs:
                    latest_checkpoint = checkpoint_dirs[-1]
                    logger.info(f"尝试从最新checkpoint恢复: {latest_checkpoint}")
                    # 这里可以添加恢复逻辑，但需要用户确认
            raise
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # 保存模型
        try:
            self.trainer.save_model()
            self.trainer.save_state()
            logger.info(f"DPO训练完成! 模型保存在: {self.config.output_dir}")
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            logger.warning("训练已完成，但模型保存失败，请检查checkpoint目录")
            # 不抛出异常，因为训练已经完成
        
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
