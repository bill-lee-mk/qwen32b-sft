#!/bin/bash
# Flash Attention 3训练脚本
echo "使用Flash Attention 3进行训练..."

# 设置Flash Attention环境变量
export FLASH_ATTENTION_FORCE_BUILD=1
export FLASH_ATTENTION_INSTALL_FORCE_BUILD=1
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"  # 支持多种GPU架构
export MAX_JOBS=4

# 检查Flash Attention
echo "检查Flash Attention安装..."
python -c "import flash_attn_3; print(f'Flash Attention路径: {flash_attn_3.__path__}')"

# 检查编译路径
if [ -d "/home/ubuntu/flash-attention/hopper" ]; then
    echo "✅ Flash Attention 3已编译在: /home/ubuntu/flash-attention/hopper"
    export PYTHONPATH="/home/ubuntu/flash-attention:$PYTHONPATH"
fi

# 设置Python路径
export PYTHONPATH="/home/ubuntu/lilei/projects/qwen32b-sft:$PYTHONPATH"

# 进入项目目录
cd /home/ubuntu/lilei/projects/qwen32b-sft

# 检查数据
if [ ! -f "processed_training_data/sft_data.jsonl" ]; then
    echo "错误: 找不到训练数据"
    exit 1
fi

echo "开始训练，启用Flash Attention 3..."
echo "=================================="

# 运行训练
python main.py train-sft \
    --config configs/training_config.yaml \
    --data processed_training_data/sft_data.jsonl

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练完成!"
    echo "模型保存在: models/sft_model"
else
    echo "❌ 训练失败"
    exit 1
fi

