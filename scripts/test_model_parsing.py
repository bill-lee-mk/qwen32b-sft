#!/usr/bin/env python3
"""批量测试 OpenRouter 各模型生成题目 + 解析是否正常。

用法:
  python scripts/test_model_parsing.py                   # 测试所有 OR 模型
  python scripts/test_model_parsing.py gpt-5.2 o3 grok-4 # 仅测试指定模型
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.generate_questions import (
    OPENROUTER_API_BASE,
    OPENROUTER_MODEL_MAP,
    call_openai,
    parse_mcq,
    _get_generation_params,
    _resolve_openrouter_model,
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    print("ERROR: OPENROUTER_API_KEY 环境变量未设置")
    sys.exit(1)

TEST_PROMPT_SYSTEM = (
    "You are an expert K-12 educational content creator. "
    "Generate exactly ONE multiple-choice question (MCQ) in JSON format. "
    "The JSON MUST have ALL these fields: id, type (must be 'mcq'), question, "
    "answer_options (object with keys A, B, C, D), answer (one of A/B/C/D), "
    "answer_explanation, standard, difficulty.\n"
    "Output ONLY the JSON object, no markdown, no extra text."
)

TEST_PROMPT_USER = (
    "Generate one MCQ for standard CCSS.MATH.CONTENT.4.OA.A.1 "
    "(Interpret a multiplication equation as a comparison), difficulty: easy.\n"
    "Output JSON only."
)


def test_model(short_name: str) -> dict:
    """测试单个模型，返回结果字典。"""
    or_model = _resolve_openrouter_model(f"or/{short_name}")
    params = _get_generation_params("openrouter", short_name)

    result = {
        "model": short_name,
        "or_model": or_model,
        "params": params,
        "status": "unknown",
        "raw_preview": "",
        "parsed": None,
        "error": "",
        "time_s": 0,
    }

    messages = [
        {"role": "system", "content": TEST_PROMPT_SYSTEM},
        {"role": "user", "content": TEST_PROMPT_USER},
    ]

    t0 = time.time()
    try:
        raw, usage = call_openai(
            messages,
            OPENROUTER_API_KEY,
            or_model,
            base_url=OPENROUTER_API_BASE,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
        )
        result["time_s"] = round(time.time() - t0, 1)
        result["raw_preview"] = (raw or "")[:800]

        if not raw or not raw.strip():
            result["status"] = "EMPTY_RESPONSE"
            result["error"] = "模型返回为空"
            return result

        mcq = parse_mcq(raw, expected_type="mcq")
        if mcq:
            result["status"] = "OK"
            result["parsed"] = mcq
        else:
            from scripts.generate_questions import _try_get_reject_reason
            reason = _try_get_reject_reason(raw, "mcq")
            result["status"] = "PARSE_FAIL"
            result["error"] = reason or "parse_mcq returned None"

    except Exception as e:
        result["time_s"] = round(time.time() - t0, 1)
        result["status"] = "API_ERROR"
        result["error"] = str(e)[:500]

    return result


def main():
    models_to_test = sys.argv[1:] if len(sys.argv) > 1 else list(OPENROUTER_MODEL_MAP.keys())

    print(f"{'='*70}")
    print(f"  OpenRouter 模型批量解析验证 — 共 {len(models_to_test)} 个模型")
    print(f"{'='*70}\n")

    results = []
    ok_count = 0
    fail_count = 0

    for i, name in enumerate(models_to_test, 1):
        print(f"[{i}/{len(models_to_test)}] 测试 {name} ({OPENROUTER_MODEL_MAP.get(name, name)}) ...")
        r = test_model(name)
        results.append(r)

        if r["status"] == "OK":
            ok_count += 1
            q = r["parsed"]
            print(f"  ✓ OK  ({r['time_s']}s)  Q: {q.get('question','')[:60]}...")
        else:
            fail_count += 1
            print(f"  ✗ {r['status']}  ({r['time_s']}s)  {r['error'][:120]}")
            if r["raw_preview"]:
                print(f"    raw前200: {r['raw_preview'][:200]}")
        print()
        time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"  汇总: {ok_count} OK / {fail_count} FAIL / {len(results)} 总计")
    print(f"{'='*70}")

    if fail_count > 0:
        print("\n失败模型详情:")
        for r in results:
            if r["status"] != "OK":
                print(f"\n  模型: {r['model']} ({r['or_model']})")
                print(f"  状态: {r['status']}")
                print(f"  错误: {r['error'][:300]}")
                if r["raw_preview"]:
                    print(f"  raw前300: {r['raw_preview'][:300]}")

    out_path = Path("evaluation_output/model_parse_test_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
