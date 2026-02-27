# -*- coding: utf-8 -*-
"""
解析 raw_data/ccss_curriculum.md → data/curriculum_standards.json

输出格式: { standard_id: { subject, course, grade, unit_name, cluster_name,
           standard_description, learning_objectives, assessment_boundaries,
           common_misconceptions } }

同时生成 subject 缩写映射表和 (grade, subject_abbr) 有效组合表，
供运行时校验和提示。
"""
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "raw_data" / "ccss_curriculum.md"
OUT_PATH = PROJECT_ROOT / "data" / "curriculum_standards.json"
META_PATH = PROJECT_ROOT / "data" / "curriculum_meta.json"

# Subject 全名 → 缩写
SUBJECT_ABBR = {
    "Language": "ELA",
    "Reading": "READ",
    "Math": "MATH",
    "Science": "SCI",
    "American History": "USHIST",
    "World History": "WHIST",
    "Social Studies": "SS",
    "Music": "MUS",
    "Visual Arts": "ART",
    "Biology": "BIO",
    "Chemistry": "CHEM",
    "Computer Science": "CS",
    "Economics": "ECON",
    "English": "ENG",
    "Government": "GOV",
    "Human Geography": "HGEO",
    "SAT Prep": "SAT",
}

# 缩写 → 全名（反向映射）
ABBR_TO_SUBJECT = {v: k for k, v in SUBJECT_ABBR.items()}

SKIP_GRADES = {"K", "AP", "HS", "SAT", "unknown"}


def _infer_grade(course: str) -> str:
    """从 course 名推断 grade。"""
    if course.startswith("AP ") or course.startswith("FRQ Skills - AP"):
        return "AP"
    if "Kindergarten" in course:
        return "K"
    m = re.search(r"(\d+)(?:st|nd|rd|th)\s+Grade", course)
    if m:
        return m.group(1)
    m = re.search(r"Grade\s+(\d+)", course)
    if m:
        return m.group(1)
    m = re.search(r"CK Grade\s+(\d+)", course)
    if m:
        return m.group(1)
    m = re.search(r"CK Grade\s+K\b", course)
    if m:
        return "K"
    if "High School" in course:
        return "HS"
    if "SAT" in course:
        return "SAT"
    return "unknown"


def _parse_bullets(text: str) -> list:
    """提取 '* ...' 项目符号列表，忽略 *None specified*。"""
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("* "):
            content = line[2:].strip()
            if content and content != "*None specified*" and content != "None specified":
                items.append(content)
    return items


def _parse_block(block: str) -> dict | None:
    """解析单个 --- 分隔块。"""
    lines = block.strip().splitlines()
    if not lines:
        return None

    kv = {}
    for line in lines:
        m = re.match(r"^(Subject|Course|Unit ID|Unit Name|Cluster ID|Cluster Name|"
                     r"Lesson Name|Lesson Order|Standard ID|Standard Description):\s*(.*)", line)
        if m:
            kv[m.group(1)] = m.group(2).strip()

    std_id = kv.get("Standard ID", "").strip()
    if not std_id:
        return None

    subject_raw = kv.get("Subject", "")
    course = kv.get("Course", "")
    grade = _infer_grade(course)
    subject_abbr = SUBJECT_ABBR.get(subject_raw, subject_raw)

    # 提取多行字段
    lo_match = re.search(r"Learning Objectives:\n(.*?)(?=\n(?:Assessment Boundaries|Common Misconceptions|Difficulty Definitions|$))",
                         block, re.DOTALL)
    ab_match = re.search(r"Assessment Boundaries:\n(.*?)(?=\n(?:Common Misconceptions|Difficulty Definitions|$))",
                         block, re.DOTALL)
    cm_match = re.search(r"Common Misconceptions:\n(.*?)(?=\n(?:Difficulty Definitions|$))",
                         block, re.DOTALL)

    learning_objectives = _parse_bullets(lo_match.group(1)) if lo_match else []
    assessment_boundaries = _parse_bullets(ab_match.group(1)) if ab_match else []
    common_misconceptions = _parse_bullets(cm_match.group(1)) if cm_match else []

    std_desc = kv.get("Standard Description", "")
    if std_desc in ("*None specified*", "None specified"):
        std_desc = ""

    return {
        "standard_id": std_id,
        "subject": subject_raw,
        "subject_abbr": subject_abbr,
        "course": course,
        "grade": grade,
        "unit_name": kv.get("Unit Name", ""),
        "cluster_name": kv.get("Cluster Name", ""),
        "standard_description": std_desc,
        "learning_objectives": learning_objectives,
        "assessment_boundaries": assessment_boundaries,
        "common_misconceptions": common_misconceptions,
    }


def parse_curriculum(raw_path: Path = RAW_PATH) -> dict:
    """解析整个 curriculum 文件，返回 {standard_id: {...}} 字典。"""
    text = raw_path.read_text(encoding="utf-8")
    # 跳过文件头（##... 注释块）
    header_end = text.find("Subject:")
    if header_end > 0:
        text = text[header_end:]

    blocks = text.split("\n---\n")
    standards = {}
    for block in blocks:
        parsed = _parse_block(block)
        if parsed:
            if parsed["grade"] in SKIP_GRADES:
                continue
            sid = parsed.pop("standard_id")
            if sid not in standards:
                standards[sid] = parsed
            else:
                existing = standards[sid]
                if not existing["learning_objectives"] and parsed["learning_objectives"]:
                    existing["learning_objectives"] = parsed["learning_objectives"]
                if not existing["assessment_boundaries"] and parsed["assessment_boundaries"]:
                    existing["assessment_boundaries"] = parsed["assessment_boundaries"]
                if not existing["common_misconceptions"] and parsed["common_misconceptions"]:
                    existing["common_misconceptions"] = parsed["common_misconceptions"]
    return standards


def build_meta(standards: dict) -> dict:
    """构建元数据：有效的 (grade, subject_abbr) 组合 + 缩写映射。"""
    valid_combos = set()
    for info in standards.values():
        valid_combos.add((info["grade"], info["subject_abbr"]))
    combos_by_grade: dict[str, list] = {}
    combos_by_subject: dict[str, list] = {}
    for g, s in sorted(valid_combos):
        combos_by_grade.setdefault(g, []).append(s)
        combos_by_subject.setdefault(s, []).append(g)
    for k in combos_by_grade:
        combos_by_grade[k] = sorted(set(combos_by_grade[k]))
    for k in combos_by_subject:
        combos_by_subject[k] = sorted(set(combos_by_subject[k]),
                                       key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    return {
        "subject_abbr": SUBJECT_ABBR,
        "abbr_to_subject": ABBR_TO_SUBJECT,
        "valid_grades": sorted(set(g for g, _ in valid_combos),
                               key=lambda x: (0, int(x)) if x.isdigit() else (1, x)),
        "valid_subjects": sorted(set(s for _, s in valid_combos)),
        "combos_by_grade": combos_by_grade,
        "combos_by_subject": combos_by_subject,
        "total_standards": len(standards),
    }


def main():
    if not RAW_PATH.exists():
        print(f"错误: {RAW_PATH} 不存在", file=sys.stderr)
        sys.exit(1)

    print(f"解析 {RAW_PATH} ...")
    standards = parse_curriculum()
    print(f"  解析到 {len(standards)} 个标准")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(standards, f, ensure_ascii=False, indent=2)
    print(f"  已写入 {OUT_PATH}")

    meta = build_meta(standards)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  已写入 {META_PATH}")

    rich = sum(1 for v in standards.values() if v["learning_objectives"])
    print(f"\n  有 Learning Objectives 的标准: {rich}/{len(standards)}")
    print(f"  有效 grade 值: {meta['valid_grades']}")
    print(f"  有效 subject 缩写: {meta['valid_subjects']}")
    print(f"\n  Grade → Subject 组合:")
    for g in meta["valid_grades"]:
        subjects = meta["combos_by_grade"].get(g, [])
        print(f"    {g:>3}: {', '.join(subjects)}")


if __name__ == "__main__":
    main()
