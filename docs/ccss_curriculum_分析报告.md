# ccss_curriculum.md 分析报告

## 一、数据概览

| 项目 | 数值 |
|------|------|
| 总块数 | ~15,967 |
| 唯一标准 ID | ~14,852 |
| 文件大小 | ~15.8 MB |
| 版本 | CCSS 1.2，快照 2026-01-14 |

---

## 二、维度分析

### 2.1 学科 (Subject) — 17 个

| 学科 | 条数 | 说明 |
|-----|------|------|
| Science | 3,776 | 科学 |
| World History | 2,971 | 世界史 |
| American History | 2,333 | 美国史 |
| Math | 1,739 | 数学 |
| Language | 1,128 | 语言（含 CCSS.ELA-LITERACY） |
| Social Studies | 782 | 社会研究 |
| AP United States History | 1,315 | AP 美国史 |
| AP World History: Modern | 778 | AP 世界史现代 |
| Biology | 615 | 生物 |
| Music | 552 | 音乐 |
| Reading | 480 | 阅读 |
| Visual Arts | 405 | 视觉艺术 |
| Computer Science | 249 | 计算机科学 |
| Chemistry | 232 | 化学 |
| Human Geography | 150 | 人文地理 |
| Government | 139 | 政府/政治 |
| Economics | 169 | 经济学 |
| English | 86 | 英语 |
| SAT Prep | 114 | SAT 备考 |

### 2.2 难度 (Difficulty Definitions)

- **结构**：每块含 Easy / Medium / Hard 三档
- **现状**：**全部为 `<unspecified>`**，未给出具体难度定义
- **影响**：需自行定义或从 Learning Objectives / Assessment Boundaries 推断

### 2.3 课程 (Course) — 80+ 种

- **年级**：K–12（Kindergarten, 1st–12th Grade）
- **AP 课程**：AP US History, AP Biology, AP Chemistry, AP Calculus, AP Psychology 等
- **CK 课程**：CK Grade K–8 American/World History, Music, Visual Arts
- **其他**：SAT Prep, High School 系列

### 2.4 类别 / 层级结构

| 字段 | 说明 | 示例 |
|------|------|------|
| Unit ID / Unit Name | 单元 | AP USH Unit 1, Period 1: 1491-1607 |
| Cluster ID / Cluster Name | 主题簇 | AP USH Topic 1.1, Contextualizing Period 1 |
| Lesson Name / Lesson Order | 课时 | 可选 |
| Standard ID | 标准编号 | KC-1.1.I, CCSS.ELA-LITERACY.L.3.1.A |
| Standard Description | 标准描述 | 核心考察内容 |

### 2.5 提示类字段（元数据）

| 字段 | 有内容块数 | 用途 |
|------|------------|------|
| **Learning Objectives** | 2,617 | 学习目标，可作出题方向 |
| **Assessment Boundaries** | 2,505 | 考查边界，避免超纲/偏题 |
| **Common Misconceptions** | 2,526 | 常见误区，可作干扰项设计 |
| **Key Concepts** | 少量 | 多为 *None specified* |

**注意**：Grade 3 ELA（Language 学科下的 L.3.x）共 155 条，**上述元数据均为空**。  
元数据主要集中在 AP 课程（如 American History、Biology 等）。

---

## 三、标准 ID 格式

### 3.1 多种体系并存

| 格式 | 示例 | 学科 |
|------|------|------|
| CCSS.ELA-LITERACY.L.3.x | L.3.1.A, L.3.2.B | Language（Grade 3 ELA） |
| CCSS.ELA-LITERACY.L.2.x+1 | L.2.1.A+1 | Reading |
| CCSS.ELA-LITERACY.L.9-10.x | L.9-10.1.A | Language（高中） |
| KC-x.x.x | KC-1.1.I, KC-1.2.II+1+A | AP US History |
| CKHG-G3-x | CKHG-G3-1A-1 | CK Grade 3 American History |
| 其他 | Unit 1: Learning Objective A | 各 AP 课程 |

### 3.2 与当前项目的对应关系

- **当前项目**：`data/standard_descriptions.json` 使用 **CCSS.ELA-LITERACY.L.3.x, RL.3.x, RI.3.x, RF.3.x, SL.3.x, W.3.x**（79 个 Grade 3 ELA 标准）
- **ccss_curriculum**：Language 学科下有 **155 个** L.3.x 相关标准（含子项如 +1, +2）
- **结论**：标准体系部分重叠，但 curriculum 中 L.3.x 的元数据为空，需结合 `standard_descriptions.json` 使用

---

## 四、作为出题引导的有效性与可行性

### 4.1 有效性

| 维度 | 评估 | 说明 |
|------|------|------|
| **Learning Objectives** | 高 | 明确考察点，适合作为题干和选项设计依据 |
| **Assessment Boundaries** | 高 | 限定考查范围，减少超纲、偏题 |
| **Common Misconceptions** | 高 | 可直接转化为干扰项，提升区分度 |
| **Standard Description** | 高 | 与 standard_descriptions 互补 |
| **Difficulty Definitions** | 低 | 全部未指定，需自建难度定义 |

### 4.2 可行性

| 场景 | 可行性 | 建议 |
|------|--------|------|
| **Grade 3 ELA（当前项目）** | 中 | curriculum 中 L.3.x 元数据为空，可继续用 `standard_descriptions.json`；若有 Reading/Language 中与 L.3 相近标准，可尝试映射 |
| **AP 课程（American History, Biology 等）** | 高 | 元数据完整，可直接作为 prompt 输入 |
| **多学科扩展** | 高 | 可解析 curriculum，按 Subject/Course 筛选，构建新 (standard, difficulty) 组合 |

### 4.3 实施建议

1. **解析器**：按 `---` 分块，用正则提取 Subject, Course, Standard ID, Standard Description, Learning Objectives, Assessment Boundaries, Common Misconceptions。
2. **Grade 3 ELA**：优先用 `standard_descriptions.json`；curriculum 中 L.3.x 仅作标准列表扩展，元数据需从其他来源补充。
3. **AP / 其他学科**：将 Learning Objectives、Assessment Boundaries、Common Misconceptions 注入 user prompt，作为「出题引导」。
4. **难度**：因 Difficulty 全为 unspecified，可先固定为 easy/medium/hard，或根据 Learning Objectives 复杂度做简单启发式分级。

---

## 五、与现有流程的衔接

```
现有流程: standard_descriptions.json → build_prompt → generate_mcq
                ↓
扩展方案: ccss_curriculum.md (解析) → 按 Subject/Standard 筛选
                ↓
          Learning Objectives + Assessment Boundaries + Common Misconceptions
                ↓
          注入 user prompt（作为 Reminders 或额外引导）
                ↓
          generate_mcq / run_with_bundle
```

---

## 六、小结

- **学科**：17 个，覆盖 K–12 与 AP。
- **难度**：结构存在，但全部未指定，需自建。
- **提示类字段**：Learning Objectives、Assessment Boundaries、Common Misconceptions 在约 2,500+ 块中有内容，适合作为出题引导。
- **Grade 3 ELA**：与当前项目标准部分重叠，但 curriculum 中 L.3.x 元数据为空，直接价值有限。
- **AP 等课程**：元数据完整，适合作为新学科、新标准的出题数据源。
