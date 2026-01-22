#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2


cd /projects/qwen3_sft


# tmux / nohup 启动

deepspeed --num_gpus=8 train.py \
          --model_path /home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B \
          --train_file /home/ubuntu/lilei/projects/qwen32b-sft/data/splits/train.jsonl \
          --val_file /home/ubuntu/lilei/projects/qwen32b-sft/data/splits/val.jsonl \
          --output_dir /home/ubuntu/lilei/projects/qwen32b-sft/outputs/qwen3_sft/exp01_full_sft \
          --deepspeed_config /home/ubuntu/lilei/projects/qwen32b-sft/configs/ds_config.json