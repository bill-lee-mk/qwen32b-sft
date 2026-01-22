# -*- coding: utf-8 -*-

import time
import math
import torch
from datasets import load_dataset
from transformers import (
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments,
                          Trainer,
                          DataCollatorForLanguageModeling,
                          TrainerCallback
                          )
 

MODEL_PATH = "/models/qwen3-32B"
TRAIN_FILE = "/data/splits/train.jsonl"
VAL_FILE = "/data/splits/val.jsonl"
OUTPUT_DIR = "/outputs/qwen3_sft/exp01_full_sft"

# ================= Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ================= Dataset =================
def preprocess(example):
    text = example["text"]
    return tokenizer(
                     text,
                     truncation=True,
                     max_length=512,
                     padding=False,
                     )


datasets = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE,})


tokenized = datasets.map(
                         preprocess,
                         batched=True,
                         remove_columns=datasets["train"].column_names,
                         num_proc=8,
                         )


# =================  token / step estimation =================
def count_tokens(dataset):
    total = 0
    for ex in dataset:
        total += len(ex["input_ids"])
    return total


train_tokens = count_tokens(tokenized["train"])


seq_len = 512
micro_batch = 1
grad_accum = 16
dp = torch.cuda.device_count()


tokens_per_step = micro_batch * grad_accum * dp * seq_len
steps_per_epoch = math.ceil(train_tokens / tokens_per_step)


print("===== Training Estimation =====")
print(f"Total tokens: {train_tokens}")
print(f"Tokens per step: {tokens_per_step}")
print(f"Estimated steps per epoch: {steps_per_epoch}")
print("===============================")


# ================= Model =================
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,)


# 开启 gradient checkpoint
model.gradient_checkpointing_enable()
model.config.use_cache = False


# ================= Data Collator =================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)


# ================= Training Args =================
training_args = TrainingArguments(
                                  output_dir=OUTPUT_DIR,
                                  overwrite_output_dir=True,
                                    
                                    
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  gradient_accumulation_steps=16,
                                    
                                    
                                  num_train_epochs=1,
                                  learning_rate=2e-5,
                                  warmup_ratio=0.03,
                                  lr_scheduler_type="cosine",
                                    
                                    
                                  logging_steps=5,
                                  save_steps=100,
                                  save_total_limit=3,
                                    
                                    
                                  evaluation_strategy="steps",
                                  eval_steps=200,
                                    
                                    
                                  bf16=True,
                                  fp16=False,
                                    
                                    
                                  deepspeed="configs/ds_config.json",
                                    
                                    
                                  report_to=["tensorboard"],
                                  logging_dir=f"{OUTPUT_DIR}/logs",
                                    
                                    
                                  ddp_find_unused_parameters=False,
                                  )


# ================= Trainer =================
trainer = Trainer(
                  model=model,
                  args=training_args,
                  train_dataset=tokenized["train"],
                  eval_dataset=tokenized["validation"],
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  )


class ThroughputCallback(TrainerCallback):
    def __init__(self, tokens_per_step):
        self.tokens_per_step = tokens_per_step
        self.last_time = None
    
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.last_time = time.time()
    
    
    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        step_time = now - self.last_time
        tps = self.tokens_per_step / step_time


if state.global_step % 10 == 0:
    print(f"[Step {state.global_step}] step_time={step_time:.2f}s, tokens/s={tps:.0f}")

trainer.add_callback(ThroughputCallback(tokens_per_step))

# ================= Train =================
if __name__ == "__main__":
    start = time.time()
    trainer.train()
    end = time.time()


print("Training finished.")
print(f"Total training time: {(end-start)/60:.2f} minutes")


trainer.save_model(f"{OUTPUT_DIR}/final_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

































