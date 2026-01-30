# -*- coding: utf-8 -*-

"""
训练配置文件
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    model_revision: Optional[str] = None
    tokenizer_name: Optional[str] = None
    torch_dtype: str = "bfloat16"  # bfloat16, float16, float32
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    flash_attention_type: str = "flash_attention_3"


@dataclass
class SFTTrainingConfig:
    """SFT训练配置"""
    output_dir: str = "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model/"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # 32B模型需要小批量
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 2
    eval_strategy: str = "no"
    fp16: bool = False
    bf16: bool = True
    fp16_full_eval=False # 禁用评估时的FP16
    bf16_full_eval=True  # 使用BF16进行评估
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    group_by_length: bool = True
    report_to: str = "none"  # "wandb", "tensorboard", "none"
    deepspeed: Optional[str] = None  # deepspeed配置文件路径


@dataclass
class DPOTrainingConfig:
    """DPO训练配置"""
    output_dir: str = "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/dpo_model/"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-6
    warmup_steps: int = 50
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int = 2
    eval_strategy: str = "no"
    fp16: bool = True
    bf16: bool = False
    beta: float = 0.1  # DPO beta参数
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, IPO, KTO
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 1024
    max_prompt_length: int = 512
    seed: int = 42
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = True
    report_to: str = "none"
    deepspeed: Optional[str] = None


@dataclass
class TrainingPipelineConfig:
    """训练流水线配置"""
    sft_enabled: bool = True
    dpo_enabled: bool = True
    sft_data_path: str = "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl"
    dpo_data_path: str = "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl"
    sft_config: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    dpo_config: DPOTrainingConfig = field(default_factory=DPOTrainingConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    final_model_dir: str = "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/final_model"
    use_checkpoint: bool = False
    checkpoint_dir: Optional[str] = None


def load_config_from_yaml(config_path: str) -> TrainingPipelineConfig:
    """从YAML文件加载配置"""
    import yaml
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 将字典转换为配置对象
    model_config = ModelConfig(**config_dict.get('model_config', {}))
    sft_config = SFTTrainingConfig(**config_dict.get('sft_config', {}))
    dpo_config = DPOTrainingConfig(**config_dict.get('dpo_config', {}))
    
    pipeline_config = TrainingPipelineConfig(
        sft_enabled=config_dict.get('sft_enabled', True),
        dpo_enabled=config_dict.get('dpo_enabled', True),
        sft_data_path=config_dict.get('sft_data_path', '/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl'),
        dpo_data_path=config_dict.get('dpo_data_path', '/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl'),
        sft_config=sft_config,
        dpo_config=dpo_config,
        model_config=model_config,
        final_model_dir=config_dict.get('final_model_dir', '/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/final_model')
    )
    
    return pipeline_config
