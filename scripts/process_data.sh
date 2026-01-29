# -*- coding: utf-8 -*-

#!/bin/bash

# 数据处理脚本
echo "开始处理训练数据..."

# 创建必要的目录
#mkdir -p raw_data
#mkdir -p processed_training_data

# 检查原始数据文件是否存在
if [ ! -f "/home/ubuntu/lilei/projects/qwen32b-sft/raw_data/ela_reasoning_medium_hard_hiquality_loquality_few_shot_20260119.jsonl" ] || [ ! -f "/home/ubuntu/lilei/projects/qwen32b-sft/raw_data/ela_eval230_dpo_grade3_mcq_expanded_20260119.jsonl" ]; then
    echo "错误: 找不到原始数据文件"
    echo "请将原始数据文件放在 raw_data/ 目录下"
    exit 1
fi

# 运行数据处理
python -m data_processing.data_processor

# 检查处理结果
if [ -f "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl" ] && [ -f "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl" ]; then
    echo "数据处理完成!"
    echo "SFT数据: processed_training_data/sft_data.jsonl"
    echo "DPO数据: processed_training_data/dpo_data.jsonl"
else
    echo "数据处理失败!"
    exit 1
fi

