# -*- coding: utf-8 -*-

"""
数据集格式化工具
将处理后的数据转换为训练需要的格式
"""
import json
from typing import Dict, List, Any
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
            tokenized["labels"] = tokenized["input_ids"].clone()
            
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
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def format_dpo_dataset(self, data_path: str) -> Dataset:
        """格式化DPO数据集"""
        
        def tokenize_dpo_function(examples):
            """DPO数据tokenize函数"""
            max_len = min(self.max_length, 1024)
            
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
            
            return {
                "input_ids": prompt_tokens["input_ids"],
                "attention_mask": prompt_tokens["attention_mask"],
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
        
        # 创建Dataset
        dataset = Dataset.from_list(samples)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            tokenize_dpo_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset


def create_sft_dataset(tokenizer, data_path, max_length=2048):
    """创建SFT数据集"""
    formatter = DatasetFormatter(tokenizer, max_length)
    return formatter.format_sft_dataset(data_path)


def create_dpo_dataset(tokenizer, data_path, max_length=1024):
    """创建DPO数据集"""
    formatter = DatasetFormatter(tokenizer, max_length)
    return formatter.format_dpo_dataset(data_path)

