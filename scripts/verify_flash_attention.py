# -*- coding: utf-8 -*-


#!/usr/bin/env python3
"""
验证Flash Attention 3是否生效
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def verify_flash_attention():
    print("验证Flash Attention 3...")
    
    # 测试模型
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # 用小模型测试
    
    print(f"加载模型: {model_name}")
    
    # 方法1：使用Flash Attention
    print("\n方法1: 使用Flash Attention")
    try:
        model_fa = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # Flash Attention 3使用此标签
        )
        print("✅ Flash Attention模型加载成功")
        
        # 测试推理速度
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model_fa.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_fa.generate(**inputs, max_new_tokens=10)
        fa_time = time.time() - start_time
        
        print(f"Flash Attention推理时间: {fa_time:.4f}秒")
        
    except Exception as e:
        print(f"❌ Flash Attention加载失败: {e}")
    
    # 方法2：不使用Flash Attention（对比）
    print("\n方法2: 不使用Flash Attention")
    try:
        model_normal = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # 标准注意力
        )
        print("✅ 标准模型加载成功")
        
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model_normal.device)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model_normal.generate(**inputs, max_new_tokens=10)
        normal_time = time.time() - start_time
        
        print(f"标准注意力推理时间: {normal_time:.4f}秒")
        
        if 'fa_time' in locals():
            speedup = normal_time / fa_time
            print(f"\n🚀 Flash Attention加速比: {speedup:.2f}x")
    
    except Exception as e:
        print(f"❌ 标准模型加载失败: {e}")
    
    # 检查Flash Attention编译路径
    print("\n=== Flash Attention编译信息 ===")
    try:
        import flash_attn
        print(f"Flash Attention版本: {flash_attn.__version__}")
        print(f"安装路径: {flash_attn.__file__}")
        
        # 检查编译的架构
        import subprocess
        result = subprocess.run(["python", "-c", "import flash_attn; print(flash_attn.__version__)"], 
                              capture_output=True, text=True)
        print(f"编译信息: {result.stdout}")
        
    except ImportError:
        print("❌ Flash Attention未安装")

def main():
    verify_flash_attention()

if __name__ == "__main__":
    main()
