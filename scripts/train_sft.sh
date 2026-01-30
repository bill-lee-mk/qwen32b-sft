# -*- coding: utf-8 -*-


#!/bin/bash

# SFT训练脚本
echo "开始SFT训练..."

# 检查数据文件
if [ ! -f "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl" ]; then
    echo "错误: 找不到SFT训练数据"
    echo "请先运行数据处理脚本: ./scripts/process_data.sh"
    exit 1
fi

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8个GPU

# 运行SFT训练
python -m training.full_finetune \
    --config configs/training_config.yaml \
    --sft-only

# 检查训练结果
if [ -d "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model" ] && [ -f "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model/pytorch_model.bin" ]; then
    echo "SFT训练完成!"
    echo "模型保存在: /home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model"
else
    echo "SFT训练失败!"
    exit 1
fi
