# -*- coding: utf-8 -*-
"""远程模型生成器 — 复用现有 prompt 工程流水线，通过远程 API 生成题目。"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data_processing.analyze_dimensions import (
    analyze_dimensions_from_curriculum,
    build_diverse_plan,
)
from data_processing.build_prompt import build_full_prompt
from evaluation.inceptbench_client import normalize_for_inceptbench, to_inceptbench_payload
from scripts.generate_questions import (
    FIREWORKS_API_BASE,
    _filter_examples_for_standard_difficulty,
    _get_api_key_for_model,
    _get_generation_params,
    _model_to_provider,
    _resolve_fireworks_model,
    call_deepseek,
    call_gemini,
    call_kimi,
    call_openai,
    parse_mcq,
)

logger = logging.getLogger(__name__)

QUESTION_TYPES = ("mcq", "msq", "fill-in")


class RemoteGenerator:
    """通过远程 API 生成 K-12 题目，复用完整的 prompt 工程流水线。"""

    def __init__(self, default_model: str = "fw/kimi-k2.5"):
        self.default_model = default_model
        self._examples: Dict[str, list] = {}
        self._loaded_grades: set[str] = set()

    def load_grade(self, grade: str, subject: str = "ELA") -> None:
        """预加载指定年级的 few-shot 示例到内存。"""
        key = f"{grade}_{subject}"
        if key in self._examples:
            return
        examples_path = _PROJECT_ROOT / "processed_training_data" / f"{grade}_{subject}_examples.json"
        if examples_path.exists():
            with open(examples_path, "r", encoding="utf-8") as f:
                self._examples[key] = json.load(f)
            logger.info(f"Loaded {len(self._examples[key])} examples for grade {grade} {subject}")
        else:
            self._examples[key] = []
            logger.warning(f"No examples file at {examples_path}")
        self._loaded_grades.add(grade)

    def load_all_grades(self, subject: str = "ELA") -> None:
        """预加载 1-12 年级的 few-shot 示例。"""
        for g in range(1, 13):
            self.load_grade(str(g), subject)

    @property
    def loaded_grades(self) -> list[str]:
        return sorted(self._loaded_grades)

    def _get_examples(self, grade: str, subject: str = "ELA") -> list:
        key = f"{grade}_{subject}"
        if key not in self._examples:
            self.load_grade(grade, subject)
        return self._examples.get(key, [])

    def _setup_model(self, model: str) -> Tuple[str, str, str]:
        """返回 (provider, api_key, resolved_model)。"""
        provider = _model_to_provider(model)
        api_key = _get_api_key_for_model(model)
        if not api_key:
            env_hint = {
                "fireworks": "FIREWORKS_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "kimi": "KIMI_API_KEY",
                "gemini": "GEMINI_API_KEY",
                "openai": "OPENAI_API_KEY",
            }
            raise ValueError(f"Missing API key: set {env_hint.get(provider, 'API_KEY')} env var")
        return provider, api_key, model

    def _call_model(self, provider: str, api_key: str, model: str,
                    messages: list) -> Tuple[str, Optional[dict]]:
        """调用远程模型，返回 (raw_text, usage_dict)。"""
        if provider == "gemini":
            system = messages[0]["content"] if messages else ""
            user = messages[1]["content"] if len(messages) > 1 else ""
            return call_gemini(f"{system}\n\n{user}", api_key, model), None
        if provider == "fireworks":
            p = _get_generation_params(provider, model)
            fw_model = _resolve_fireworks_model(model)
            return call_openai(messages, api_key, fw_model,
                               base_url=FIREWORKS_API_BASE,
                               temperature=p["temperature"],
                               max_tokens=p["max_tokens"])
        if provider == "deepseek":
            return call_deepseek(messages, api_key, model)
        if provider == "kimi":
            return call_kimi(messages, api_key, model)
        p = _get_generation_params(provider, model)
        return call_openai(messages, api_key, model,
                           temperature=p["temperature"],
                           max_tokens=p["max_tokens"])

    def generate_one(
        self,
        grade: str,
        standard: str,
        difficulty: str,
        question_type: str = "mcq",
        subject: str = "ELA",
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """生成单道题目，返回标准化的内部格式 dict。

        失败时返回 {"error": "...", "standard": ..., "difficulty": ..., "type": ...}。
        """
        model = model or self.default_model
        provider, api_key, model = self._setup_model(model)
        examples = self._get_examples(grade, subject)

        # 设置该年级的 prompt_rules 环境变量
        model_tag = model.replace("/", "_").replace("-", "_")
        rules_path = _PROJECT_ROOT / "processed_training_data" / f"{grade}_{subject}_prompt_rules_{model_tag}_matrix.json"
        if rules_path.exists():
            os.environ["PROMPT_RULES_PATH"] = str(rules_path)

        filtered = _filter_examples_for_standard_difficulty(
            examples, standard, difficulty, question_type=question_type,
        )
        system, user = build_full_prompt(
            grade=grade, standard=standard, difficulty=difficulty,
            examples=filtered, subject=subject, question_type=question_type,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        for attempt in range(max(max_retries, 1)):
            try:
                raw, usage = self._call_model(provider, api_key, model, messages)
                mcq = parse_mcq(raw, expected_type=question_type)
                if mcq:
                    out = normalize_for_inceptbench(mcq)
                    out["grade"] = grade
                    out["standard"] = standard
                    out["subject"] = subject
                    out["difficulty"] = difficulty
                    out["type"] = question_type
                    return out
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed for {standard}|{difficulty}|{question_type}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(min(5 * (attempt + 1), 30))

        return {
            "error": f"Failed after {max_retries} attempts",
            "standard": standard, "difficulty": difficulty, "type": question_type,
        }

    def generate_batch(
        self,
        grade: str,
        subject: str = "ELA",
        question_type: str = "all",
        model: Optional[str] = None,
        workers: int = 10,
        max_retries: int = 3,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """生成指定年级的全部 (standard, difficulty, type) 组合。

        返回 (results_list, failed_count)。
        """
        dims = analyze_dimensions_from_curriculum(grade, subject)
        plan_sd = build_diverse_plan(dims, n=9999, all_combinations=True)
        if question_type == "all":
            types = list(QUESTION_TYPES)
        else:
            types = [question_type]

        plan: list[tuple[str, str, str]] = []
        for s, d in plan_sd:
            for t in types:
                plan.append((s, d, t))

        results: list[Optional[Dict[str, Any]]] = [None] * len(plan)
        failed = 0

        model = model or self.default_model
        t0 = time.time()
        logger.info(f"Generating {len(plan)} questions for grade {grade} ({len(plan_sd)} combos x {len(types)} types) with {workers} workers")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, (s, d, t) in enumerate(plan):
                fut = executor.submit(
                    self.generate_one,
                    grade=grade, standard=s, difficulty=d,
                    question_type=t, subject=subject,
                    model=model, max_retries=max_retries,
                )
                futures[fut] = i
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    result = fut.result()
                    if result.get("error"):
                        failed += 1
                        logger.warning(f"[{idx+1}/{len(plan)}] Failed: {result.get('error')}")
                    results[idx] = result
                except Exception as e:
                    failed += 1
                    s, d, t = plan[idx]
                    results[idx] = {"error": str(e), "standard": s, "difficulty": d, "type": t}

        elapsed = time.time() - t0
        logger.info(f"Batch done: {len(plan)-failed}/{len(plan)} succeeded in {elapsed:.1f}s")
        return [r for r in results if r and not r.get("error")], failed

    def get_combinations(self, grade: str, subject: str = "ELA",
                         question_type: str = "all") -> List[Tuple[str, str, str]]:
        """返回指定年级的所有 (standard, difficulty, type) 组合。"""
        dims = analyze_dimensions_from_curriculum(grade, subject)
        plan_sd = build_diverse_plan(dims, n=9999, all_combinations=True)
        types = list(QUESTION_TYPES) if question_type == "all" else [question_type]
        return [(s, d, t) for s, d in plan_sd for t in types]
