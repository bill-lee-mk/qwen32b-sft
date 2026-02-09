# -*- coding: utf-8 -*-

"""
DPO（直接偏好优化）训练器
"""
import math
import os
import time
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

from transformers import TrainerCallback
from .config import DPOTrainingConfig, ModelConfig
from data_processing.dataset_formatter import create_dpo_dataset

logger = logging.getLogger(__name__)


class SavePathCallback(TrainerCallback):
    """保存 checkpoint 时打印写入路径"""
    def on_save(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            save_path = os.path.abspath(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"))
            logger.info(f"Writing checkpoint-{state.global_step} model shards to: {save_path}")


class StepTimingCallback(TrainerCallback):
    """每步完成后将耗时加入训练日志，并同步 GPU 缓存以消除 DeepSpeed 警告"""
    def __init__(self):
        self._step_start = None
        self._last_step_duration = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._step_start is not None:
            self._last_step_duration = time.perf_counter() - self._step_start
        # 同步各 rank 的 GPU 缓存，消除 "pytorch allocator cache flushes" 警告
        try:
            from deepspeed.accelerator import get_accelerator
            get_accelerator().empty_cache()
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """增强日志：epoch进度、步数进度（第一位）、每步耗时（放最后）"""
        if logs is None:
            return
        step_progress_val = None
        # 1. epoch 显示为 当前/总数
        if "epoch" in logs and hasattr(args, "num_train_epochs") and args.num_train_epochs:
            logs["epoch"] = f"{logs['epoch']}/{args.num_train_epochs}"
        # 2. 步数进度：当前步/总步数（放第一位）
        if hasattr(state, "global_step") and hasattr(state, "max_steps") and state.max_steps > 0:
            step_progress_val = f"{state.global_step}/{state.max_steps}"
            logs["step_progress"] = step_progress_val
        # 3. 每步耗时放在最后：先移除再最后添加
        if self._last_step_duration is not None:
            logs.pop("step_duration_sec", None)
            logs["step_duration_sec"] = round(self._last_step_duration, 3)
        # 4. 将 step_progress 移到第一位
        if step_progress_val is not None:
            logs.pop("step_progress", None)
            new_logs = {"step_progress": step_progress_val}
            new_logs.update(logs)
            logs.clear()
            logs.update(new_logs)


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
        is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0
        if is_main:
            logger.info("加载模型和tokenizer...")
        
        # 确定torch数据类型
        if self.model_config.torch_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            if is_main:
                logger.info("使用bfloat16精度")
        elif self.model_config.torch_dtype == "float16":
            torch_dtype = torch.float16
            if is_main:
                logger.info("使用float16精度")
        else:
            torch_dtype = torch.float32
            if is_main:
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
        
        # Load the active model (dtype 替代已弃用的 torch_dtype)
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
        if is_main:
            logger.info("Loading reference model (offloaded to CPU to save GPU memory)...")
        if self.config.deepspeed:
            ref_device_map = {"": "cpu"}
            if is_main:
                logger.info("Reference model will be offloaded to CPU (DeepSpeed will handle transfers)")
        else:
            ref_device_map = {"": "cpu"}
            if is_main:
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
                import flash_attn_interface
                if is_main:
                    logger.info("Flash Attention 3 可用")
            except ImportError:
                if is_main:
                    logger.warning("Flash Attention 3不可用，请从:/home/ubuntu/flash-attention/hopper/ 安装")
        
        # 启用梯度检查点
        if self.model_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if is_main:
                logger.info("已启用梯度检查点")
        
        if is_main:
            logger.info(f"模型参数量: {self.model.num_parameters():,}")
        
        return self.model, self.tokenizer
    
    def prepare_dataset(
        self,
        data_path: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        num_proc: int = 1,
    ) -> Dataset:
        """准备DPO数据集。cache_dir 指定时，tokenize 结果缓存，后续训练秒级加载。"""
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"准备DPO数据集: {data_path}" + (f"（缓存: {cache_dir}）" if cache_dir else ""))
        
        dataset = create_dpo_dataset(
            self.tokenizer,
            data_path,
            max_length=self.config.max_length,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_proc=num_proc,
        )
        
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"数据集大小: {len(dataset)}")
        return dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """执行DPO训练"""
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
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
        
        # 打印实际用到的训练参数（与 SFT 一致，仅主进程）
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            per_device = training_args.per_device_train_batch_size
            grad_accum = training_args.gradient_accumulation_steps
            eff_batch = per_device * world_size * grad_accum
            steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
            total_steps_est = steps_per_epoch * training_args.num_train_epochs
            logger.info("=" * 50 + " DPO 训练参数 " + "=" * 50)
            for k, v in vars(training_args).items():
                logger.info(f"  {k}: {v}")
            logger.info("-" * 50)
            logger.info("总步数计算公式:")
            logger.info(f"  有效batch = per_device_batch_size × WORLD_SIZE × gradient_accumulation_steps")
            logger.info(f"            = {per_device} × {world_size} × {grad_accum} = {eff_batch}")
            logger.info(f"  每epoch步数 = ceil(样本数 / 有效batch) = ceil({len(train_dataset)} / {eff_batch}) = {steps_per_epoch}")
            logger.info(f"  预估总步数 = 每epoch步数 × num_train_epochs = {steps_per_epoch} × {training_args.num_train_epochs} = {total_steps_est}")
            logger.info("=" * 50)
        
        # 创建DPOTrainer（添加每步耗时 callback）
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[StepTimingCallback(), SavePathCallback()],
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
            final_path = os.path.abspath(self.config.output_dir)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                metrics = train_result.metrics
                total_steps = self.trainer.state.global_step
                train_runtime = metrics.get("train_runtime", 0)
                time_per_step = train_runtime / total_steps if total_steps > 0 else 0
                logger.info(
                    f"DPO 完成 | 步数: {total_steps} | 耗时: {train_runtime/3600:.2f}h | "
                    f"每步: {time_per_step:.2f}s | 输出: {self.config.output_dir}"
                )
            self.trainer.save_model()
            self.trainer.save_state()
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            logger.warning("训练已完成，但模型保存失败，请检查checkpoint目录")
            # 不抛出异常，因为训练已经完成
        
        return train_result
    
    def run(
        self,
        data_path: str,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        num_proc: int = 1,
    ):
        """运行完整的DPO训练流程。cache_dir 指定时，tokenize 结果会缓存，后续训练直接加载。"""
        from pathlib import Path
        
        # 1. 加载模型和tokenizer
        self.load_model_and_tokenizer(model_path)
        
        # 2. 准备数据集（默认缓存在数据同目录下的 cache/）
        if cache_dir is None:
            cache_dir = str(Path(data_path).parent / "cache")
        train_dataset = self.prepare_dataset(
            data_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            num_proc=num_proc,
        )
        
        # 3. 训练
        train_result = self.train(train_dataset)
        
        return train_result
