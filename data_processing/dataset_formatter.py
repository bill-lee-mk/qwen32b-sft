# -*- coding: utf-8 -*-

"""
数据集格式化工具
将处理后的数据转换为训练需要的格式
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset
from transformers import AutoTokenizer
import torch


class DatasetFormatter:
    """数据集格式化器"""
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 设置填充token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_sft_dataset(self, data_path: str) -> Dataset:
        """格式化SFT数据集"""
        
        def tokenize_function(examples):
            """SFT数据tokenize函数"""
            # 最大长度调整
            max_len = min(self.max_length, 2048)
            
            # Tokenize文本
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            
            # 对于因果语言模型，标签就是输入ID
            # 关键：padding 位置必须设为 -100，否则 loss 会错误地学习预测 padding
            tokenized["labels"] = tokenized["input_ids"].clone()
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            tokenized["labels"][tokenized["attention_mask"] == 0] = -100
            
            return tokenized
        
        # 读取数据
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        
        # 创建Dataset
        dataset = Dataset.from_list(samples)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=1, # Force single-process per rank to avoid "Broken Pipe"
        )
        
        return tokenized_dataset
    
    def format_dpo_dataset(
        self,
        data_path: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        num_proc: int = 1,
    ) -> Dataset:
        """格式化DPO数据集。use_cache=True 时，首次 tokenize 后写入缓存，后续直接加载。"""
        
        def tokenize_dpo_function(examples):
            """DPO数据tokenize函数"""
            max_len = self.max_length
            
            batch_size = len(examples["prompt"])
            
            # Tokenize prompts
            prompt_tokens = self.tokenizer(
                examples["prompt"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            
            # Tokenize chosen responses
            chosen_tokens = self.tokenizer(
                examples["chosen"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            
            # Tokenize rejected responses
            rejected_tokens = self.tokenizer(
                examples["rejected"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )
            
            # return {
            #     "input_ids": prompt_tokens["input_ids"],
            #     "attention_mask": prompt_tokens["attention_mask"],
            #     "chosen_input_ids": chosen_tokens["input_ids"],
            #     "chosen_attention_mask": chosen_tokens["attention_mask"],
            #     "rejected_input_ids": rejected_tokens["input_ids"],
            #     "rejected_attention_mask": rejected_tokens["attention_mask"],
            # }
        
            return {
                "prompt_input_ids": prompt_tokens["input_ids"],
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
            }
        
        # 读取数据
        samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                samples.append(sample)
        
        dataset = Dataset.from_list(samples)
        
        # 缓存路径：{data_path 同目录}/cache/dpo_{stem}_max_length_{max_length}.arrow
        cache_path = None
        if use_cache and cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            stem = Path(data_path).stem
            cache_path = str(Path(cache_dir) / f"dpo_{stem}_max_length{self.max_length}.arrow")
        
        tokenized_dataset = dataset.map(
            tokenize_dpo_function,
            batched=True,
            cache_file_name=cache_path,
            load_from_cache_file=use_cache and cache_path is not None,
            num_proc=num_proc,
        )
        
        return tokenized_dataset


def create_sft_dataset(tokenizer, data_path, max_length=2048):
    """创建SFT数据集"""
    formatter = DatasetFormatter(tokenizer, max_length)
    return formatter.format_sft_dataset(data_path)


def create_dpo_dataset(
    tokenizer,
    data_path: str,
    max_length: int = 2048,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    num_proc: int = 1,
) -> Dataset:
    """创建DPO数据集。cache_dir 指定时，tokenize 结果会缓存到该目录，后续训练直接加载。"""
    formatter = DatasetFormatter(tokenizer, max_length)
    return formatter.format_dpo_dataset(
        data_path,
        cache_dir=cache_dir,
        use_cache=use_cache,
        num_proc=num_proc,
    )

