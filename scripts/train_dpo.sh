#!/bin/bash
# DPO训练脚本（使用DeepSpeed）
echo "开始DPO训练（使用DeepSpeed）..."

# 检查数据文件
if [ ! -f "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl" ]; then
    echo "错误: 找不到DPO训练数据"
    echo "请先运行数据处理脚本: ./scripts/process_data.sh"
    exit 1
fi

# 检查SFT模型（优先使用最终模型，如果没有则使用checkpoint）
SFT_MODEL_DIR="/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/sft_model"
if [ -d "$SFT_MODEL_DIR" ] && { [ -f "$SFT_MODEL_DIR/model.safetensors" ] || [ -f "$SFT_MODEL_DIR/model.safetensors.index.json" ] || [ -f "$SFT_MODEL_DIR/pytorch_model.bin" ]; }; then
    # 使用最终保存的模型（推荐）
    SFT_MODEL="$SFT_MODEL_DIR"
    echo "使用SFT最终模型: $SFT_MODEL"
elif [ -d "$SFT_MODEL_DIR/checkpoint-2000" ]; then
    # 如果没有最终模型，使用checkpoint（不推荐，但可用）
    SFT_MODEL="$SFT_MODEL_DIR/checkpoint-2000"
    echo "警告: 使用checkpoint模型（建议使用最终模型）: $SFT_MODEL"
else
    echo "警告: 找不到SFT模型，将使用基础模型进行DPO训练"
    SFT_MODEL=""
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

# 检查训练结果（DeepSpeed/大模型可能保存为分片格式 model-00001-of-00002.safetensors 或 checkpoint-XXX/）
DPO_MODEL_DIR="/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B/dpo_model"
HAS_MODEL=false
[ -f "$DPO_MODEL_DIR/model.safetensors" ] && HAS_MODEL=true
[ -f "$DPO_MODEL_DIR/model.safetensors.index.json" ] && HAS_MODEL=true
[ -f "$DPO_MODEL_DIR/model-00001-of-00002.safetensors" ] && HAS_MODEL=true
[ -f "$DPO_MODEL_DIR/pytorch_model.bin" ] && HAS_MODEL=true
ls -d "$DPO_MODEL_DIR"/checkpoint-* >/dev/null 2>&1 && HAS_MODEL=true

if [ -d "$DPO_MODEL_DIR" ] && [ "$HAS_MODEL" = true ]; then
    echo "DPO训练完成!"
    echo "模型保存在: $DPO_MODEL_DIR"
else
    echo "DPO训练失败!"
    exit 1
fi