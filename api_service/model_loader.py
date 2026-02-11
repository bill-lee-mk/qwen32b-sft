# -*- coding: utf-8 -*-
"""模型加载与推理"""
import os
import asyncio
from typing import Optional
from .schemas import MCQRequest, MCQResponse


class MCQGenerator:
    """MCQ 生成器，加载微调后的 Qwen 模型进行推理"""

    def __init__(self, model_path: str = "models/final_model"):
        self.model_path = os.path.abspath(model_path)
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device = "cpu"
        self.model_name = "unknown"
        self.default_params = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        }

    def load_model(self):
        """加载模型和 tokenizer"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"模型路径不存在: {self.model_path}\n"
                "请先完成训练，或指定正确的 --model 路径"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.device = str(next(self.model.parameters()).device)
        self.model_name = os.path.basename(self.model_path)
        self.model_loaded = True

    async def generate_async(self, request: MCQRequest) -> MCQResponse:
        """异步生成 MCQ（在线程池中执行同步推理）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_sync, request)

    def _generate_sync(self, request: MCQRequest) -> MCQResponse:
        """同步生成 MCQ"""
        import torch
        import json
        import re

        if not self.model_loaded:
            raise RuntimeError("模型未加载")

        prompt = self._build_prompt(request)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.default_params["max_new_tokens"],
                temperature=self.default_params["temperature"],
                top_p=self.default_params["top_p"],
                do_sample=self.default_params["do_sample"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        # 尝试解析 JSON
        result = MCQResponse(raw_response=response_text)
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                result.question_id = data.get("id")
                result.question = data.get("question")
                result.answer = data.get("answer")
                result.answer_options = data.get("answer_options")
                result.answer_explanation = data.get("answer_explanation")
                result.difficulty = data.get("difficulty", request.difficulty)
        except (json.JSONDecodeError, KeyError):
            pass

        return result

    def _build_prompt(self, request: MCQRequest) -> str:
        """构建生成 prompt"""
        system = (
            f"You are an expert grade {request.grade} ELA MCQ designer. "
            f"Generate a clear MCQ at '{request.difficulty}' difficulty. "
            "Return a valid JSON object with: id, type, question, answer, "
            "answer_options (A,B,C,D), answer_explanation, difficulty."
        )
        user = (
            f"Generate one MCQ for {request.subject}, standard {request.standard}, "
            f"difficulty: {request.difficulty}. "
            + ("Include <think>...</think> chain." if request.include_think_chain else "Return only the JSON.")
        )
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """按给定完整 prompt 生成续写，用于 OpenAI 兼容的 /v1/chat/completions。"""
        import torch
        if not self.model_loaded:
            raise RuntimeError("模型未加载")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
