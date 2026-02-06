# -*- coding: utf-8 -*-
"""
InceptBench 评估客户端

API 说明：
- URL: POST https://inceptbench.api.inceptlabs.ai/2.3.0/evaluate
- 认证: Authorization: Bearer <token>（需 Bearer token，非 query 的 api_key）
- 环境变量: INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN
"""
import os
import json
from typing import Dict, Any, Optional, List

INCEPTBENCH_URL = "https://inceptbench.api.inceptlabs.ai/2.3.0/evaluate"


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
    """将单个 MCQ 转为 generated_content 的一项"""
    grade = q.get("grade") or req_ctx.get("grade", "3")
    standard = q.get("standard") or req_ctx.get("standard", "CCSS.ELA-LITERACY.L.3.1.E")
    subject = q.get("subject") or req_ctx.get("subject", "ELA")

    content = {
        "question": str(q.get("question", "")),
        "answer": str(q.get("answer", "A")).upper().strip()[:1],
        "answer_options": _convert_answer_options(q.get("answer_options", {})),
        "answer_explanation": str(q.get("answer_explanation", "")),
    }

    request = {
        "grade": str(grade),
        "subject": subject,
        "type": "mcq",
        "difficulty": str(q.get("difficulty", "medium")),
        "locale": "en-US",
        "skills": {
            "lesson_title": req_ctx.get("lesson_title", "K-12 ELA"),
            "substandard_id": standard,
            "substandard_description": req_ctx.get("substandard_description", standard),
        },
    }

    # InceptBench API/DB 要求 generated_question_id 为整数；若原 id 非整数则用 index
    raw_id = q.get("id", index)
    try:
        int_id = int(raw_id) if raw_id is not None else index
    except (TypeError, ValueError):
        int_id = index
    metadata = dict(q.get("metadata", {}))
    metadata["generated_question_id"] = int_id

    return {
        "id": int_id,
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
    """从 API 返回的 nested evaluations 中提取 overall_score（用于兼容显示）"""
    evals = result.get("evaluations") or {}
    scores = []
    for ev in evals.values():
        inc = ev.get("inceptbench_new_evaluation") or {}
        overall = inc.get("overall") or {}
        s = overall.get("score")
        if s is not None:
            scores.append(float(s))
    if not scores:
        return None
    return sum(scores) / len(scores) if scores else None


def normalize_for_inceptbench(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 MCQ 归一化为内部标准格式（供 generate_mcq 输出）。
    保证 answer_options 为 dict，answer 为 A/B/C/D。
    """
    out = {}
    for k in ["id", "type", "question", "answer", "answer_options", "answer_explanation", "difficulty"]:
        v = question_data.get(k)
        if v is None:
            if k == "type":
                out[k] = "mcq"
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
    ans = str(out.get("answer", "")).upper().strip()
    out["answer"] = ans if ans in ("A", "B", "C", "D") else "A"
    return out


class InceptBenchEvaluator:
    """
    InceptBench MCQ 评估器

    认证方式: Authorization: Bearer <token>
    需设置环境变量 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN，或传入 api_key 参数。
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 300):
        self.token = (
            api_key
            or os.environ.get("INCEPTBENCH_API_KEY")
            or os.environ.get("INCEPTBENCH_TOKEN")
            )
        self.timeout = timeout
        if not self.token:
            import warnings
            warnings.warn(
                "未设置 InceptBench 认证 token。"
                "请通过 --api-key 或环境变量 INCEPTBENCH_API_KEY / INCEPTBENCH_TOKEN 提供。"
                "认证方式: Authorization: Bearer <token>"
            )

    def evaluate_mcq(
        self,
        question_data: Dict[str, Any] | List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        评估 MCQ（支持单个或批量）。

        输入: 我们的 MCQ 格式，单个 dict 或 dict 列表
        输出: API 返回的评估结果
        """
        if not self.token:
            return {
                "overall_score": 0.0,
                "status": "error",
                "message": "未设置 INCEPTBENCH_API_KEY 或 INCEPTBENCH_TOKEN",
            }

        payload = to_inceptbench_payload(question_data)
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        try:
            import urllib.request
            import urllib.error
            req = urllib.request.Request(
                INCEPTBENCH_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
            if isinstance(result, dict):
                # 补充 overall_score 便于显示（API 返回在 evaluations.<id>.inceptbench_new_evaluation.overall.score）
                score = _extract_overall_score(result)
                if score is not None and "overall_score" not in result:
                    result["overall_score"] = round(score, 2)
                return result
            return {"raw": result}
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
