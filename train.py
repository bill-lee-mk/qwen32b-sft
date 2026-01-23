# -*- coding: utf-8 -*-

import os
import time
import math
import argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

# --------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/home/ubuntu/lilei/projects/qwen32b-sft/models/qwen3-32B")
parser.add_argument("--train_file", type=str, default="/home/ubuntu/lilei/projects/qwen32b-sft/data/splits/train.jsonl")
parser.add_argument("--val_file", type=str, default="/home/ubuntu/lilei/projects/qwen32b-sft/data/splits/val.jsonl")
parser.add_argument("--output_dir", type=str, default="/home/ubuntu/lilei/projects/qwen32b-sft/outputs/qwen32b-sft/")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--deepspeed_config", type=str, default="/home/ubuntu/lilei/projects/qwen32b-sft/configs/ds_config.json")
args = parser.parse_args()

# --------- Prepare tokenizer ----------
print("Loading tokenizer from", args.model_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# --------- Load dataset ----------
print("Loading dataset...")
# Expect JSON lines with field "text". If your data has other fields change accordingly.
data_files = {}
if os.path.exists(args.train_file):
    data_files["train"] = args.train_file
else:
    raise FileNotFoundError(f"Train file not found: {args.train_file}")
if os.path.exists(args.val_file):
    data_files["validation"] = args.val_file
else:
    print("Warning: val file not found, continuing without validation")
datasets = load_dataset("json", data_files=data_files)

# Tokenize
def preprocess_fn(examples):
    texts = examples["text"]
    return tokenizer(texts, truncation=True, max_length=args.max_length, padding=False)

print("Tokenizing (this can take a while)...")
tokenized = datasets.map(preprocess_fn, batched=True, remove_columns=datasets["train"].column_names, num_proc=8)

# Convert to torch tensors lazily (Trainer will collate)
# --------- Model load ----------
print("Loading model (bf16, low_cpu_mem_usage)...")
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
# Important for ZeRO training
model.gradient_checkpointing_enable()
model.config.use_cache = False
# If tokenizer was extended, resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# --------- Data collator (causal LM) ----------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --------- Estimation: total tokens, tokens/step, steps/epoch ----------
def estimate_tokens_and_steps(tokenized_dataset, seq_len, micro_batch, grad_accum, dp):
    # sum lengths of input_ids
    total_tokens = 0
    # For memory safety, iterate in streaming if needed
    for ids in tokenized_dataset["input_ids"]:
        total_tokens += len(ids)
    tokens_per_step = micro_batch * grad_accum * dp * seq_len
    steps_per_epoch = math.ceil(total_tokens / tokens_per_step)
    return total_tokens, tokens_per_step, steps_per_epoch

dp = torch.cuda.device_count() if torch.cuda.is_available() else 1
total_tokens, tokens_per_step, steps_per_epoch = estimate_tokens_and_steps(tokenized["train"], args.max_length, args.per_device_train_batch_size, args.gradient_accumulation_steps, dp)

print("===== Training Estimation =====")
print(f"Total tokens (train): {total_tokens}")
print(f"Tokens per step: {tokens_per_step}")
print(f"Estimated steps per epoch: {steps_per_epoch}")
print("===============================")

# --------- Throughput callback ----------
class ThroughputCallback(TrainerCallback):
    def __init__(self, tokens_per_step, log_every_n_steps=10):
        self.tokens_per_step = tokens_per_step
        self.log_every_n_steps = log_every_n_steps
        self._step_start_time = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._step_start_time = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self._step_start_time is None:
            return
        step_time = time.time() - self._step_start_time
        tokens_per_sec = self.tokens_per_step / step_time if step_time > 0 else float("inf")
        if state.global_step % self.log_every_n_steps == 0:
            print(f"[Throughput] step={state.global_step} step_time={step_time:.3f}s tokens/s={tokens_per_sec:.0f}")

# --------- Training arguments ----------
training_args = TrainingArguments(
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    evaluation_strategy="steps" if "validation" in tokenized else "no",
    eval_steps=200,
    bf16=True,
    fp16=False,
    deepspeed=args.deepspeed_config,
    report_to=["tensorboard"],
    logging_dir=f"{args.output_dir}/logs",
    ddp_find_unused_parameters=False,
)

# --------- Trainer ----------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[ThroughputCallback(tokens_per_step=tokens_per_step, log_every_n_steps=10)],
)

# --------- Train ----------
if __name__ == "__main__":
    start = time.time()
    trainer.train()
    end = time.time()
    print("Training finished.")
    print(f"Total training time: {(end - start)/60:.2f} minutes")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
