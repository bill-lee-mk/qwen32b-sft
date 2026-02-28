#!/usr/bin/env python3
"""从 PostgreSQL 数据库中提取高分 ELA 题目，增强 few-shot 示例文件。

对每个年级的 inceptbench_highscore_grade{N}_ela.json：
1. 找出所有 (standard, difficulty, type) 组合的空缺
2. 找出已有但分数较低的组合
3. 从数据库中查询同组合的最高分题目
4. 替换低分题目 + 填补空缺（仅保留 score >= 0.85 的题目）
"""
import json
import os
import sys
import psycopg2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"

DB_CONFIG = {
    "host": "incept-rds.c7esqey6q6bf.us-west-2.rds.amazonaws.com",
    "user": "postgres",
    "password": ":tJm1A3wVEZHpqIJj(47_?.NBPE5",
    "dbname": "edubench",
    "sslmode": "require",
}

MIN_SCORE = 0.85

# 数据库中 ELA 内容分布在这几个 subject 下
ELA_SUBJECTS = ("ela", "language", "reading")

QUERY_BEST_PER_COMBO = """
SELECT
    r.standard_id_l1 AS standard,
    g.question_type,
    g.model_parsed_response->>'difficulty' AS difficulty,
    a.evaluator_score AS score,
    g.model_parsed_response AS parsed,
    r.standard_desc_l1 AS standard_desc
FROM generated_questions g
JOIN question_recipes r ON g.recipe_id = r.recipe_id
JOIN ai_evaluation_results a ON a.question_id = g.id
WHERE r.subject IN %s
  AND r.grade_level = %s
  AND r.standard_id_l1 LIKE 'CCSS.ELA-LITERACY%%'
  AND g.question_type IN ('mcq', 'msq', 'fill-in')
  AND g.model_parsed_response->>'difficulty' IN ('easy', 'medium', 'hard')
  AND a.evaluator_score >= %s
ORDER BY r.standard_id_l1, g.question_type, g.model_parsed_response->>'difficulty', a.evaluator_score DESC
"""


def _normalize_answer_options(opts):
    """将 answer_options 从 dict 转为 [{key, text}, ...] 格式"""
    if isinstance(opts, dict):
        return [{"key": k, "text": v} for k, v in sorted(opts.items())]
    if isinstance(opts, list):
        return opts
    return []


def _db_item_to_example(row, grade):
    """将数据库行转换为 few-shot 示例格式"""
    parsed = row["parsed"] if isinstance(row["parsed"], dict) else json.loads(row["parsed"])
    standard = row["standard"]
    difficulty = row["difficulty"]
    qtype = row["question_type"]
    score = row["score"]
    standard_desc = row.get("standard_desc") or ""

    item = {
        "id": parsed.get("id", f"db-{grade}-{qtype}-{difficulty}"),
        "type": qtype,
        "question": parsed.get("question", ""),
        "answer": parsed.get("answer", ""),
        "answer_explanation": parsed.get("answer_explanation", ""),
        "difficulty": difficulty,
        "standard": standard,
        "substandard_description": standard_desc,
        "score": score,
    }

    opts = parsed.get("answer_options")
    if opts:
        item["answer_options"] = _normalize_answer_options(opts)

    if parsed.get("context"):
        item["context"] = parsed["context"]

    return item


def load_existing_examples(grade):
    path = RAW_DATA_DIR / f"inceptbench_highscore_grade{grade}_ela.json"
    if not path.exists():
        return []
    return json.load(open(path, encoding="utf-8"))


def build_existing_index(examples):
    """建立 (standard, difficulty, type) -> (score, index) 的索引"""
    idx = {}
    for i, item in enumerate(examples):
        key = (item.get("standard", ""), item.get("difficulty", ""), item.get("type", "mcq"))
        score = item.get("score", 0) or 0
        if key not in idx or score > idx[key][0]:
            idx[key] = (score, i)
    return idx


def get_all_combos(grade):
    """获取该年级所有 (standard, difficulty) 组合"""
    sys.path.insert(0, str(PROJECT_ROOT))
    from data_processing.analyze_dimensions import analyze_dimensions_from_curriculum
    from scripts.generate_questions import build_diverse_plan
    dims = analyze_dimensions_from_curriculum(str(grade), "ELA")
    return build_diverse_plan(dims, n=9999, all_combinations=True)


def query_db_best(conn, grade):
    """从数据库查询该年级每个 (standard, difficulty, type) 的最高分题目"""
    cur = conn.cursor()
    cur.execute(QUERY_BEST_PER_COMBO, (ELA_SUBJECTS, str(grade), MIN_SCORE))
    columns = [desc[0] for desc in cur.description]

    best = {}
    for row_tuple in cur.fetchall():
        row = dict(zip(columns, row_tuple))
        key = (row["standard"], row["difficulty"], row["question_type"])
        if key not in best or row["score"] > best[key]["score"]:
            best[key] = row

    cur.close()
    return best


def process_grade(conn, grade):
    """处理单个年级"""
    examples = load_existing_examples(grade)
    existing_idx = build_existing_index(examples)
    all_combos = get_all_combos(grade)
    types = ["mcq", "msq", "fill-in"]

    db_best = query_db_best(conn, grade)

    replaced = 0
    filled = 0
    skipped_low = 0

    for s, d in all_combos:
        for t in types:
            key = (s, d, t)
            db_row = db_best.get(key)

            if key in existing_idx:
                cur_score = existing_idx[key][0]
                cur_idx = existing_idx[key][1]
                if db_row and db_row["score"] > cur_score:
                    new_item = _db_item_to_example(db_row, grade)
                    examples[cur_idx] = new_item
                    replaced += 1
            else:
                if db_row and db_row["score"] >= MIN_SCORE:
                    new_item = _db_item_to_example(db_row, grade)
                    examples.append(new_item)
                    filled += 1
                else:
                    skipped_low += 1

    out_path = RAW_DATA_DIR / f"inceptbench_highscore_grade{grade}_ela.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    return len(examples), replaced, filled, skipped_low


def main():
    print("连接数据库...", flush=True)
    conn = psycopg2.connect(**DB_CONFIG)
    print("连接成功\n", flush=True)

    print(f"{'Grade':>5} | {'原有':>5} | {'替换':>4} | {'填补':>4} | {'跳过(<0.85)':>11} | {'最终':>5}")
    print("-" * 55)

    total_replaced = 0
    total_filled = 0

    for grade in range(1, 13):
        orig_count = len(load_existing_examples(grade))
        final_count, replaced, filled, skipped = process_grade(conn, grade)
        total_replaced += replaced
        total_filled += filled
        print(f"{grade:>5} | {orig_count:>5} | {replaced:>4} | {filled:>4} | {skipped:>11} | {final_count:>5}")

    print("-" * 55)
    print(f"总计: 替换 {total_replaced} 条, 填补 {total_filled} 条")

    conn.close()
    print("\n完成!")


if __name__ == "__main__":
    main()
