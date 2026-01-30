# -*- coding: utf-8 -*-

#!/usr/bin/env python3
"""
启用Flash Attention 3的配置脚本
"""
import torch
import os
import sys

def check_flash_attention():
    """检查Flash Attention支持"""
    print("=== Flash Attention 3支持检查 ===")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法使用Flash Attention")
        return False
    
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查Flash Attention
    try:
        import flash_attn_3
        print(f"✅ Flash Attention已安装: {flash_attn.__name__}")
        
        # 检查编译路径
        print(f"Flash Attention编译路径: {flash_attn.__path__}")
        
        return True
    except ImportError:
        print("❌ Flash Attention未安装")
        print("安装命令: pip install flash-attn --no-build-isolation")
        return False

def optimize_for_flash_attention():
    """优化配置以使用Flash Attention 3"""
    print("\n=== 优化Flash Attention配置 ===")
    
    # 设置环境变量
    env_vars = {
        "FLASH_ATTENTION_FORCE_BUILD": "1",
        "FLASH_ATTENTION_INSTALL_FORCE_BUILD": "1",
        "TORCH_CUDA_ARCH_LIST": "8.0 8.6 8.9 9.0",  # 支持多种架构
        "MAX_JOBS": "4",  # 限制编译作业数
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"设置 {key}={value}")
    
    return env_vars

def create_flash_attention_config():
    """创建Flash Attention配置文件"""
    config_content = """# Flash Attention 3 配置
flash_attention:
  enabled: true
  type: "flash_attention_3"  # 注意：transformers中Flash Attention 3仍使用此标签
  dtype: "bfloat16"
  causal: true
  dropout: 0.0
  deterministic: false
  backend: "CUTLASS"  # Flash Attention 3使用的后端

training_optimizations:
  use_gradient_checkpointing: true
  use_cpu_offload: false
  mixed_precision: "bf16"
  gradient_accumulation_steps: 8
  max_grad_norm: 1.0

hardware_config:
  cuda_arch: "8.0"  # A100等Ampere架构
  memory_efficient: true
  kernel_tiling: true
"""
    
    config_path = "configs/flash_attention_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\n✅ 已创建Flash Attention配置文件: {config_path}")
    return config_path

def main():
    """主函数"""
    print("Flash Attention 3 配置助手")
    print("=" * 50)
    
    # 检查支持
    if not check_flash_attention():
        sys.exit(1)
    
    # 优化配置
    optimize_for_flash_attention()
    
    # 创建配置文件
    config_path = create_flash_attention_config()
    
    print("\n=== 下一步 ===")
    print("1. 确保在训练配置中启用Flash Attention")
    print("2. 运行训练时，模型会自动使用Flash Attention 3")
    print("3. 监控GPU内存使用和训练速度")
    
    return True

if __name__ == "__main__":
    main()

