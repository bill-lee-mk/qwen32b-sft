# -*- coding: utf-8 -*-

#!/bin/bash

# DPO训练脚本（使用DeepSpeed）
echo "开始DPO训练（使用DeepSpeed）..."

# 检查数据文件
if [ ! -f "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl" ]; then
    echo "错误: 找不到DPO训练数据"
    echo "请先运行数据处理脚本: ./scripts/process_data.sh"
    exit 1
fi

# 检查SFT模型
if [ ! -d "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model/checkpoint-2000/" ]; then
    echo "警告: 找不到SFT模型，将使用基础模型进行DPO训练"
    SFT_MODEL="/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/"
else
    SFT_MODEL="/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model/checkpoint-2000/"
    echo "使用SFT模型: $SFT_MODEL"
fi

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8个GPU

# 运行DPO训练
if [ -z "$SFT_MODEL" ]; then
    echo "使用基础模型进行DPO训练!"
    # 使用DeepSpeed启动DPO训练（会自动使用所有可见GPU）
    deepspeed --num_gpus=8 --module training.full_finetune \
        --config configs/training_config.yaml \
        --dpo-only      
else
    echo "使用SFT模型进行DPO训练!"
    # 使用DeepSpeed启动DPO训练（会自动使用所有可见GPU）
    deepspeed --num_gpus=8 --module training.full_finetune \
        --config configs/training_config.yaml \
        --dpo-only \
        --sft-model "$SFT_MODEL"
fi

# 检查训练结果
if [ -d "/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/dpo_model" ] ; then
    echo "DPO训练完成!"
    echo "模型保存在: /home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/dpo_model"
else
    echo "DPO训练失败!"
    exit 1
fi