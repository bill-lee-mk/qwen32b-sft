# Qwen32b-SFT 仓库完整分析报告

## 一、仓库用途

本仓库是一个 **K-12 ELA（英语语言艺术）选择题（MCQ）生成系统**，基于 Qwen3-32B 模型进行全参数微调，用于生成符合教育标准的英语选择题。

### 核心功能
1. **两阶段训练**：SFT（监督微调） + DPO（直接偏好优化）
2. **数据处理**：支持指令跟随格式和 DPO 偏好对格式
3. **API 服务**：提供 RESTful API 进行 MCQ 生成
4. **模型评估**：通过 InceptBench 评估生成质量（接口已定义但实现缺失）

---

## 二、目录结构与模块关系

```
qwen32b-sft/
├── main.py                 # 主入口，CLI 命令分发
├── download_model.py       # 从 HuggingFace 下载 Qwen3-32B
├── configs/                # 配置文件
│   ├── training_config.yaml
│   └── deepspeed_config.json
├── data_processing/       # 数据处理
│   ├── data_processor.py   # 原始数据 → SFT/DPO 格式
│   └── dataset_formatter.py # 文本 → tokenized Dataset
├── training/               # 训练逻辑
│   ├── config.py           # 配置 dataclass
│   ├── full_finetune.py    # 训练入口（SFT/DPO 调度）
│   ├── sft_trainer.py      # SFT 训练器
│   └── dpo_trainer.py      # DPO 训练器
├── api_service/            # API 服务（依赖缺失）
│   └── fastapi_app.py
├── evaluation/              # 评估（实现缺失）
│   └── __init__.py
├── scripts/                 # 脚本
│   ├── process_data.sh
│   ├── train_sft.sh
│   ├── train_dpo.sh
│   ├── train_with_flash_attention.sh
│   ├── enable_flash_attention.py
│   └── verify_flash_attention.py
├── raw_data/                # 原始数据（需用户提供）
├── processed_training_data/ # 处理后数据
└── models/                  # 模型存储
    └── qwen3-32B/
        ├── sft_model/
        ├── dpo_model/
        └── final_model/
```

---

## 三、执行依赖顺序与数据流

### 1. 完整流水线执行顺序

```
1. 下载模型
   download_model.py
   → models/qwen3-32B/

2. 准备原始数据
   将 .jsonl 放入 raw_data/

3. 数据处理
   python main.py process-data
   或 scripts/process_data.sh
   → data_processing.data_processor.main()
   → processed_training_data/sft_data.jsonl
   → processed_training_data/dpo_data.jsonl

4. SFT 训练
   python main.py train-sft
   或 deepspeed --num_gpus=8 --module training.full_finetune --sft-only
   → training.full_finetune.main()
   → SFTTrainer.run()
   → models/qwen3-32B/sft_model/

5. DPO 训练（依赖 SFT 输出）
   python main.py train-dpo --sft-model <path>
   或 scripts/train_dpo.sh
   → DPOTrainerWrapper.run()
   → models/qwen3-32B/dpo_model/

6. 合并最终模型（在 train-all 中自动完成）
   FullParameterFinetuner.merge_and_save_final_model()
   → models/qwen3-32B/final_model/

7. 启动 API（依赖 final_model）
   python main.py serve-api --model models/final_model
   → api_service.fastapi_app.run_api_server()
```

### 2. 模块调用关系

```
main.py
├── process-data → data_processing.data_processor.main()
├── train-sft    → training.full_finetune.main() → SFTTrainer
├── train-dpo    → training.full_finetune.main() → DPOTrainerWrapper
├── train-all    → 依次: process_data → SFT → DPO → merge
├── serve-api    → api_service.fastapi_app
└── evaluate     → evaluation.inceptbench_client.InceptBenchEvaluator  [缺失]

SFTTrainer
├── config.py (SFTTrainingConfig, ModelConfig)
├── data_processing.dataset_formatter.create_sft_dataset()
└── transformers.Trainer + DataCollatorForLanguageModeling

DPOTrainerWrapper
├── config.py (DPOTrainingConfig, ModelConfig)
├── data_processing.dataset_formatter.create_dpo_dataset()
└── trl.DPOTrainer + DPOConfig
```

---

## 四、关键依赖版本与兼容性

### 1. 核心版本（requirements.txt）

| 包名 | 版本 | 用途 |
|------|------|------|
| transformers | 5.0.0 | 模型加载、Trainer、attn_implementation |
| trl | 0.27.1 | DPOTrainer, DPOConfig |
| torch | 2.10.0 | 深度学习 |
| deepspeed | 0.18.5 | ZeRO-3 分布式训练 |
| datasets | 4.5.0 | Dataset |
| accelerate | 1.12.0 | 分布式 |
| peft | 0.18.1 | （本仓库未使用 PEFT） |
| flash_attn_3 | 3.0.0b1 | Flash Attention 3 |

### 2. 版本兼容性要点

- **transformers 5.0.0**：支持 `attn_implementation="flash_attention_3"`，与当前用法一致。
- **trl 0.27.1**：DPOTrainer 使用 `DPOConfig`（非旧版 `TrainingArguments`），与 `dpo_trainer.py` 一致。
- **TRL DPO 数据格式**：需要 `prompt_input_ids`、`chosen_input_ids`、`rejected_input_ids`，`dataset_formatter.py` 已正确输出。
- **flash_attn_3**：需与 transformers 5.0 的 `flash_attention_3` 后端配合；脚本中 `import flash_attn_interface` 需确认由 `flash_attn_3` 包提供。

### 3. 已知问题

1. **packaging 路径硬编码**（requirements.txt 第 79 行）：
   ```
   packaging @ file:///home/task_176104877067765/conda-bld/packaging_1761049113113/work
   ```
   建议改为：`packaging>=24.0` 或移除该行使用默认版本。

2. **绝对路径硬编码**：多处使用 `/home/ubuntu/lilei/projects/qwen32b-sft/`，迁移环境需修改：
   - `configs/training_config.yaml`
   - `data_processing/data_processor.py` (DataConfig, main)
   - `training/config.py`
   - `scripts/*.sh`
   - `dpo_trainer.py` (TENSORBOARD_LOGGING_DIR)

---

## 五、缺失模块与潜在错误

### 1. API 服务依赖缺失

`api_service/fastapi_app.py` 引用了不存在的模块：

```python
from .model_loader import MCQGenerator   # 不存在
from .schemas import (                   # 不存在
    MCQRequest, MCQResponse, ...
)
```

**影响**：`python main.py serve-api` 会 `ImportError`。

**建议**：实现 `api_service/model_loader.py` 和 `api_service/schemas.py`，或暂时注释/移除相关导入和逻辑。

### 2. 评估模块缺失

`main.py` 第 130 行：

```python
from evaluation.inceptbench_client import InceptBenchEvaluator
```

`evaluation/inceptbench_client.py` 不存在，`evaluation/` 下仅有 `__init__.py` 和 `.gitkeep`。

**影响**：`python main.py evaluate --input xxx` 会 `ImportError`。

### 3. main.py 与 full_finetune 参数传递

`main.py` 中：

```python
elif args.command == "train-sft":
    train_sft_main(args)   # 传入 subparser 的 args
```

`full_finetune.main()` 会访问 `args.sft_only`、`args.dpo_only`、`args.sft_model`。当从 `train-sft` 子命令进入时，这些属性可能不存在，可能触发 `AttributeError`。

**建议**：使用 `getattr(args, 'sft_only', False)` 等安全访问。

### 4. train-all 中 sft-model 路径

`main.py` 第 116 行：

```python
train_dpo_main()  # 通过 sys.argv 传参
sys.argv = ["train-dpo", "--config", args.config, "--sft-model", "models/sft_model"]
```

实际 SFT 输出在 `models/qwen3-32B/sft_model/`，且 `train_dpo.sh` 使用 `checkpoint-2000/`。若目录结构不同，DPO 可能加载错误 checkpoint。

---

## 六、数据格式说明

### 1. 原始数据格式

**指令跟随格式**（用于 SFT）：
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "question_id": "...",
  "recipe_id": "..."
}
```

**DPO 格式**：
```json
{
  "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "chosen": {"content": "高质量回答"},
  "rejected": {"content": "低质量回答"},
  "metadata": {"chosen_score": 0.9, "rejected_score": 0.5}
}
```

### 2. 处理后格式

- **SFT**：`{"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...", "metadata": {...}}`
- **DPO**：`{"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}`

---

## 七、DeepSpeed 配置

`configs/deepspeed_config.json` 使用 **ZeRO Stage 3**，并开启 CPU offload，适合 32B 模型在有限显存下训练。

---

## 八、建议修复清单

1. 修复 `requirements.txt` 中 `packaging` 的路径依赖。
2. 实现或补齐 `api_service/model_loader.py` 和 `api_service/schemas.py`。
3. 实现 `evaluation/inceptbench_client.py`，或从 `main.py` 中移除 evaluate 命令。
4. 将硬编码绝对路径改为相对路径或环境变量。
5. 在 `full_finetune.main()` 中用 `getattr` 安全访问 `args` 属性。
6. 统一 `train-all` 与脚本中 SFT 模型路径（含 checkpoint 目录）。
