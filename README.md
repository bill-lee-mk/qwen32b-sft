# Qwen3-32B K-12 ELA MCQ生成器

基于Qwen2.5-32B-Instruct模型全参数微调的K-12 ELA（英语语言艺术）选择题生成系统。

## 特性

- **全参数微调**：更新模型所有参数，获得最佳性能
- **两阶段训练**：SFT（监督微调）+ DPO（直接偏好优化）
- **高质量数据**：基于教育领域专家标注的高质量MCQ示例
- **完整API**：提供RESTful API服务，支持单条和批量生成

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/qwen-mcq-generator.git
cd qwen-mcq-generator

# 安装依赖
pip install -r requirements.txt

# 安装Flash Attention（可选，提升训练速度）
pip install flash-attn --no-build-isolation