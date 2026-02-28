# -*- coding: utf-8 -*-
"""
InceptBench 评估客户端

API 说明：
- URL: POST https://inceptbench.api.inceptlabs.ai/2.3.0/evaluate 或 api.inceptbench.com/evaluate
- 认证: Authorization: Bearer <token>（需 Bearer token，非 query 的 api_key）
- 主配置: INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN；第二套: EVALUATOR_TOKEN（URL 固定）
- 多 key 降级: 主配置为 INCEPTBENCH_*；EVALUATOR_TOKEN 作第二套（URL 固定 api.inceptbench.com），主配置失败时自动切换
"""
import os
import json
from typing import Dict, Any, Optional, List, Tuple

_PRIMARY_URL = "https://inceptbench.api.inceptlabs.ai/2.3.3/evaluate"
_EVALUATOR_FALLBACK_URL = "https://api.inceptbench.com/evaluate"


def _get_evaluator_endpoints() -> List[Tuple[str, str]]:
    """
    解析多套 (url, token) 配置，全部从环境变量读取。
    主配置: INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN
    第二套: EVALUATOR_TOKEN（URL 固定为 api.inceptbench.com，主配置失败时自动切换）
    """
    def _pair(url: str, token: Optional[str]) -> Optional[Tuple[str, str]]:
        if not token or not token.strip():
            return None
        return (url.strip().rstrip("/"), token.strip())

    pairs: List[Tuple[str, str]] = []

    # 主配置
    tok1 = os.environ.get("INCEPTBENCH_API_KEY") or os.environ.get("INCEPTBENCH_TOKEN")
    p1 = _pair(_PRIMARY_URL, tok1)
    if p1:
        pairs.append(p1)

    # 第二套（主配置失败时降级，如 429 配额超限等）
    tok2 = os.environ.get("EVALUATOR_TOKEN")
    p2 = _pair(_EVALUATOR_FALLBACK_URL, tok2)
    if p2 and p2 not in pairs:
        pairs.append(p2)

    return pairs


def _convert_answer_options(opts: Any) -> List[Dict[str, str]]:
    """将 answer_options 转为 InceptBench 要求的 [{key, text}, ...] 格式"""
    if isinstance(opts, list):
        return [{"key": str(o.get("key", "")), "text": str(o.get("text", ""))} for o in opts]
    if isinstance(opts, dict):
        return [{"key": k, "text": str(v)} for k, v in opts.items()]
    return []


def _mcq_to_inceptbench_item(
    q: Dict[str, Any],
    req_ctx: Dict[str, Any],
    index: int = 0,
) -> Dict[str, Any]:
    """将单个题目转为 generated_content 的一项（支持 mcq/msq/fill-in）"""
    grade = q.get("grade") or req_ctx.get("grade", "3")
    standard = q.get("standard") or req_ctx.get("standard", "CCSS.ELA-LITERACY.L.3.1.E")
    subject = q.get("subject") or req_ctx.get("subject", "ELA")
    qtype = str(q.get("type", "mcq")).lower().strip()

    question = str(q.get("question", ""))

    if qtype == "fill-in":
        content = {
            "question": question,
            "answer": str(q.get("answer", "")),
            "answer_explanation": str(q.get("answer_explanation", "")),
        }
        if q.get("acceptable_answers"):
            content["acceptable_answers"] = q["acceptable_answers"]
    else:
        opts = q.get("answer_options", {})
        if qtype == "msq":
            ans_value = str(q.get("answer", "A,B")).upper().strip()
        else:
            ans_key = str(q.get("answer", "A")).upper().strip()[:1]
            correct_text = ""
            if isinstance(opts, dict):
                correct_text = opts.get(ans_key, opts.get(ans_key.lower(), ""))
            elif isinstance(opts, list):
                for o in opts:
                    if str(o.get("key", "")).upper() == ans_key:
                        correct_text = str(o.get("text", ""))
                        break
            if correct_text and " " in str(correct_text).strip() and "which word" in question.lower():
                question = question.replace("Which word", "Which choice").replace("which word", "which choice")
            ans_value = ans_key
        content = {
            "question": question,
            "answer": ans_value,
            "answer_options": _convert_answer_options(opts),
            "answer_explanation": str(q.get("answer_explanation", "")),
        }

    default_lesson = f"K-12 {subject}" if subject else "K-12 ELA"
    request = {
        "grade": str(grade),
        "subject": subject,
        "type": qtype,
        "difficulty": str(q.get("difficulty", "medium")),
        "locale": "en-US",
        "skills": {
            "lesson_title": req_ctx.get("lesson_title", default_lesson),
            "substandard_id": standard,
            "substandard_description": req_ctx.get("substandard_description", standard),
        },
    }

    item_id = str(q.get("id", index))
    metadata = dict(q.get("metadata", {}))
    metadata["generated_question_id"] = item_id

    return {
        "id": item_id,
        "request": request,
        "content": content,
        "image_url": q.get("image_url", []),
        "metadata": metadata,
        "verbose": False,
    }


def to_inceptbench_payload(
    question_data: Dict[str, Any] | List[Dict[str, Any]],
    request_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    将我们的 MCQ 格式转为 InceptBench API 请求体。

    输入（我们的格式）:
      单个: {id, question, answer, answer_options: {A:..., B:...}, answer_explanation, difficulty}
      批量: [{...}, {...}]
      可选: grade, standard

    输出（InceptBench 格式）:
      {generated_content: [{id, request, content, image_url, metadata, verbose}, ...]}
    """
    req_ctx = request_context or {}
    if isinstance(question_data, list):
        items = [_mcq_to_inceptbench_item(q, req_ctx, i) for i, q in enumerate(question_data)]
    else:
        items = [_mcq_to_inceptbench_item(question_data, req_ctx)]
    return {"generated_content": items}


def _extract_overall_score(result: Dict[str, Any]) -> Optional[float]:
    """
    从 API 返回的 nested evaluations 中提取 overall_score。
    自适应解析：先尝试主配置格式，无则尝试第二套，单条结果独立解析，混用兼容。
    - 主配置: evaluations.<id>.inceptbench_new_evaluation.overall.score
    - 第二套: evaluations.<id>.overall.score
    """
    evals = result.get("evaluations") or {}
    scores = []
    for ev in evals.values():
        s = None
        # 主配置格式
        inc = ev.get("inceptbench_new_evaluation") or {}
        overall = inc.get("overall") or {}
        s = overall.get("score")
        # 第二套格式：直接在 overall 下
        if s is None:
            overall = ev.get("overall") or {}
            s = overall.get("score")
        if s is not None:
            scores.append(float(s))
    if not scores:
        return None
    return sum(scores) / len(scores) if scores else None


def normalize_for_inceptbench(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将题目归一化为内部标准格式（供 generate_questions 输出）。
    支持 mcq/msq/fill-in：
    - mcq/msq: 保证 answer_options 为 dict
    - fill-in: 无 answer_options
    """
    qtype = str(question_data.get("type", "mcq")).lower().strip()
    out = {}

    if qtype == "fill-in":
        for k in ["id", "type", "question", "answer", "answer_explanation", "difficulty"]:
            v = question_data.get(k)
            if v is None:
                if k == "type":
                    out[k] = "fill-in"
                elif k == "difficulty":
                    out[k] = "medium"
                else:
                    out[k] = ""
            else:
                out[k] = v
        if "acceptable_answers" in question_data:
            out["acceptable_answers"] = question_data["acceptable_answers"]
        return out

    # mcq / msq
    for k in ["id", "type", "question", "answer", "answer_options", "answer_explanation", "difficulty"]:
        v = question_data.get(k)
        if v is None:
            if k == "type":
                out[k] = qtype
            elif k == "difficulty":
                out[k] = "medium"
            else:
                out[k] = ""
        else:
            out[k] = v
    opts = out.get("answer_options")
    if isinstance(opts, dict):
        normalized_opts = {}
        for letter in "ABCD":
            normalized_opts[letter] = opts.get(letter, opts.get(letter.lower(), ""))
        out["answer_options"] = normalized_opts
    elif isinstance(opts, list):
        normalized_opts = {}
        for o in opts:
            k = str(o.get("key", "")).upper()
            if k in "ABCD":
                normalized_opts[k] = str(o.get("text", ""))
        out["answer_options"] = normalized_opts

    if qtype == "msq":
        ans = str(out.get("answer", "")).upper().strip()
        ans_letters = sorted(set(l.strip() for l in ans.replace(" ", "").split(",") if l.strip() and l.strip() in "ABCD"))
        out["answer"] = ",".join(ans_letters) if len(ans_letters) >= 2 else "A,B"
    else:
        ans = str(out.get("answer", "")).upper().strip()
        out["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
        correct_text = (out.get("answer_options") or {}).get(out["answer"], "")
        if correct_text and " " in str(correct_text).strip():
            q = out.get("question", "")
            if q and "which word" in q.lower():
                out["question"] = q.replace("Which word", "Which choice").replace("which word", "which choice")
    return out


def _do_evaluate(
    url: str,
    token: str,
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    """单次请求评估 API，返回结果或 error 字典。"""
    import urllib.request
    import urllib.error

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode())
    if isinstance(result, dict):
        score = _extract_overall_score(result)
        if score is not None and "overall_score" not in result:
            result["overall_score"] = round(score, 2)
        # 200 但存在 failed_items（如 429 配额超限）：视为错误，触发切换备用端点
        failed_items = result.get("failed_items") or []
        if failed_items:
            first_err = failed_items[0]
            msg = first_err.get("error", str(first_err))[:500] if isinstance(first_err, dict) else str(first_err)[:500]
            return {
                "overall_score": 0.0,
                "status": "error",
                "message": msg,
                "response_body": json.dumps(result, ensure_ascii=False)[:1000],
            }
        # 200 但无分数：可能是 API 返回了 detail/error 等错误格式
        if score is None and (result.get("detail") or result.get("error")):
            msg = result.get("detail") or result.get("error")
            if isinstance(msg, dict):
                msg = msg.get("message", str(msg))[:500]
            else:
                msg = str(msg)[:500]
            return {
                "overall_score": 0.0,
                "status": "error",
                "message": msg,
                "response_body": json.dumps(result, ensure_ascii=False)[:1000],
            }
        return result
    return {"raw": result}


def _do_evaluate_catch(
    url: str,
    token: str,
    payload: Dict[str, Any],
    timeout: int,
) -> Dict[str, Any]:
    """单次请求并捕获异常，返回 error 字典或正常结果。"""
    import urllib.request
    import urllib.error

    try:
        return _do_evaluate(url, token, payload, timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            if hasattr(e, "read"):
                body = e.read().decode("utf-8", errors="replace")
            elif getattr(e, "fp", None):
                body = e.fp.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = f"HTTP {e.code}: {e.reason}"
        if body.strip():
            msg += f" | 服务器返回: {body[:500]}"
        return {
            "overall_score": 0.0,
            "status": "error",
            "message": msg,
            "response_body": body,
        }
    except Exception as e:
        return {
            "overall_score": 0.0,
            "status": "error",
            "message": str(e),
        }


class InceptBenchEvaluator:
    """
    InceptBench MCQ 评估器

    认证方式: Authorization: Bearer <token>
    主配置需设置 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN。
    第二套设置 EVALUATOR_TOKEN（主配置失败时自动切换）。
    """

    def __init__(self, timeout: int = 300):
        self.endpoints = _get_evaluator_endpoints()
        self.timeout = timeout
        if not self.endpoints:
            import warnings
            warnings.warn(
                "未设置 InceptBench 认证 token。"
                "请设置环境变量 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN；"
                "第二套降级用 EVALUATOR_TOKEN。"
                "认证方式: Authorization: Bearer <token>"
            )

    def evaluate_mcq(
        self,
        question_data: Dict[str, Any] | List[Dict[str, Any]],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        评估 MCQ（支持单个或批量）。
        超时/服务端错误自动重试 max_retries 次（每次切换端点）。
        """
        if not self.endpoints:
            return {
                "overall_score": 0.0,
                "status": "error",
                "message": "未设置 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN（主配置）",
            }

        payload = to_inceptbench_payload(question_data)
        last_err: Dict[str, Any] = {}

        for attempt in range(max_retries + 1):
            for idx, (url, token) in enumerate(self.endpoints):
                result = _do_evaluate_catch(url, token, payload, self.timeout)
                if result.get("status") != "error":
                    return result
                last_err = result
                msg = str(result.get("message", "")).lower()
                is_retryable = any(k in msg for k in (
                    "timeout", "timed out", "connection", "reset",
                    "500", "502", "503", "504", "429",
                    "could not save", "save evaluation", "internal",
                ))
                if is_retryable:
                    if attempt < max_retries:
                        import time as _time
                        wait = min(10 * (attempt + 1), 60)
                        _time.sleep(wait)
                        break
                    return last_err
                if idx < len(self.endpoints) - 1:
                    continue
                return last_err
            else:
                return last_err

        return last_err
