# -*- coding: utf-8 -*-

"""
MCQ生成模型训练数据处理器
处理两种格式的数据：指令跟随格式和DPO格式
"""
import json
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import random
from pathlib import Path

@dataclass
class DataConfig:
    """数据处理配置"""
    sft_output_path: str = "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl"
    dpo_output_path: str = "/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl"
    keep_think_chain: bool = True  # 是否保留思考链
    max_length: int = 2048
    train_test_split: float = 0.9
    seed: int = 42


class MCQDataProcessor:
    """MCQ训练数据处理器"""
    
    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()
        random.seed(self.config.seed)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(self.config.sft_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.dpo_output_path), exist_ok=True)
    
    def load_raw_file(self, file_path: str) -> List[Dict]:
        """加载原始数据文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f][0]
        return data
    
    def extract_json_from_response(self, text: str) -> Optional[Dict]:
        """从响应文本中提取JSON"""
        try:
            # 查找JSON开始和结束位置
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                return None
            
            json_str = text[start_idx:end_idx]
            
            # 处理可能的转义字符
            json_str = json_str.replace('\\"', '"')
            json_str = json_str.replace('\\\\', '\\')
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            # 尝试修复常见的JSON格式问题
            try:
                # 尝试添加缺失的引号
                json_str = re.sub(r'(\w+):', r'"\1":', json_str)
                return json.loads(json_str)
            except:
                return None
    
    def process_instruction_following_sample(self, sample: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        处理指令跟随格式的样本
        返回：(sft_samples, dpo_samples)
        """
        sft_samples = []
        dpo_samples = []
        
        # 检查样本格式
        if "messages" in sample:
            messages = sample["messages"]
            
            # 确保是有效的对话格式
            if len(messages) >= 3 and messages[-1]["role"] == "assistant":
                
                # 1. 创建SFT样本
                formatted_text = self._format_conversation(messages)
                sft_samples.append({
                    "text": formatted_text,
                    "metadata": {
                        "source": "instruction_following",
                        "question_id": sample.get("question_id"),
                        "recipe_id": sample.get("recipe_id"),
                        "has_think_chain": "<think>" in messages[-1]["content"]
                    }
                })
                
                # 2. 如果没有rejected响应，无法创建DPO样本
                # 这种格式通常只用于SFT
        
        return sft_samples, dpo_samples
    
    def process_dpo_sample(self, sample: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        处理DPO格式的样本
        返回：(sft_samples, dpo_samples)
        """
        sft_samples = []
        dpo_samples = []
        
        # 检查样本格式
        if "prompt" in sample and "chosen" in sample and "rejected" in sample:
            prompt = sample["prompt"]  # 列表，包含system和user
            chosen = sample["chosen"]
            rejected = sample["rejected"]
            
            # 1. 从chosen创建SFT样本（高质量）
            formatted_prompt = self._format_conversation(prompt)
            chosen_response = chosen["content"]
            full_conversation = formatted_prompt + self._format_assistant_response(chosen_response)
            
            sft_samples.append({
                "text": full_conversation,
                "metadata": {
                    "source": "dpo_chosen",
                    "quality_score": sample.get("metadata", {}).get("chosen_score", 0.9),
                    "has_think_chain": "<think>" in chosen_response
                }
            })
            
            # 2. 创建DPO样本
            dpo_samples.append({
                "prompt": formatted_prompt,
                "chosen": chosen_response,
                "rejected": rejected["content"],
                "metadata": {
                    "chosen_score": sample.get("metadata", {}).get("chosen_score", 0.9),
                    "rejected_score": sample.get("metadata", {}).get("rejected_score", 0.5),
                    "difficulty": sample.get("metadata", {}).get("difficulty", "medium")
                }
            })
        
        return sft_samples, dpo_samples
    
    def process_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """处理单个文件"""
        print(f"处理文件: {file_path}")
        
        all_sft_samples = []
        all_dpo_samples = []
        
        try:
            data = self.load_raw_file(file_path)
            
            # 根据文件内容判断格式
            if isinstance(data, list):
                # 可能是包含多个样本的列表
                for item in data:
                    sft_samples, dpo_samples = self.process_sample(item)
                    all_sft_samples.extend(sft_samples)
                    all_dpo_samples.extend(dpo_samples)
            else:
                # 单个样本
                sft_samples, dpo_samples = self.process_sample(data)
                all_sft_samples.extend(sft_samples)
                all_dpo_samples.extend(dpo_samples)
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return all_sft_samples, all_dpo_samples
    
    def process_sample(self, sample: Dict) -> Tuple[List[Dict], List[Dict]]:
        """处理单个样本，自动检测格式"""
        # 检测样本格式
        if "messages" in sample:
            return self.process_instruction_following_sample(sample)
        elif "prompt" in sample and "chosen" in sample:
            return self.process_dpo_sample(sample)
        else:
            print(f"未知的样本格式: {sample.keys()}")
            return [], []
    
    def augment_data(self, sft_samples: List[Dict], target_count: int = 100) -> List[Dict]:
        """数据增强"""
        if len(sft_samples) >= target_count:
            return sft_samples
        
        print(f"当前SFT样本数: {len(sft_samples)}，目标: {target_count}")
        
        augmented_samples = []
        
        # ELA标准列表用于数据增强
        ela_standards = [
            ("CCSS.ELA-LITERACY.L.3.1.A", "Explain the function of nouns, pronouns, verbs, adjectives, and adverbs"),
            ("CCSS.ELA-LITERACY.L.3.1.B", "Form and use regular and irregular plural nouns"),
            ("CCSS.ELA-LITERACY.L.3.1.C", "Use abstract nouns"),
            ("CCSS.ELA-LITERACY.L.3.1.D", "Form and use regular and irregular verbs"),
            ("CCSS.ELA-LITERACY.L.3.1.E", "Form and use the simple verb tenses"),
            ("CCSS.ELA-LITERACY.L.3.1.F", "Ensure subject-verb and pronoun-antecedent agreement"),
            ("CCSS.ELA-LITERACY.L.3.1.G", "Form and use comparative and superlative adjectives and adverbs"),
            ("CCSS.ELA-LITERACY.L.3.1.H", "Use coordinating and subordinating conjunctions"),
            ("CCSS.ELA-LITERACY.L.3.1.I", "Produce simple, compound, and complex sentences"),
        ]
        
        difficulties = ["easy", "medium", "hard"]
        grades = ["3", "4", "5"]
        subjects = ["Reading", "Writing", "Language", "Vocabulary"]
        
        # 模板
        system_template = """SYSTEM:
You are an expert grade {grade} ELA MCQ designer.

Rules:
1. Generate a clear and concise MCQ that directly assesses the provided standard.
2. The generated question must be '{difficulty}' difficulty.
3. Generate a unique `id` for the question.
4. Set the `type` field to 'mcq'.
5. Provide a brief `answer_explanation`.
6. Include 4 answer options (A, B, C, D).
7. Ensure only one correct answer.
8. Make distractors plausible but incorrect.
9. Return a single, valid JSON object with correct delimiters and quotations.
10. Output one JSON object.

Schema:
{{
  "id": "unique_id",
  "type": "mcq",
  "question": "The text of the question",
  "answer": "A/B/C/D",
  "answer_explanation": "Brief explanation",
  "answer_options": {{ "A": "...", "B": "...", "C": "...", "D": "..." }},
  "difficulty": "{difficulty}"
}}"""
        
        # 基于现有样本创建变体
        base_samples = sft_samples.copy()
        
        for i in range(target_count - len(sft_samples)):
            # 选择基础样本
            base_sample = random.choice(base_samples)
            text = base_sample["text"]
            
            # 随机修改参数
            grade = random.choice(grades)
            difficulty = random.choice(difficulties)
            standard_id, standard_desc = random.choice(ela_standards)
            subject = random.choice(subjects)
            
            # 替换系统提示中的参数
            new_system = system_template.format(
                grade=grade,
                difficulty=difficulty
            )
            
            # 构建用户提示
            user_prompt = f"""Generate one MCQ at '{difficulty}' difficulty that obeys the binding guidance below.

--- BLUEPRINT CONTRACT (binding) ---
Grade Level: {grade}
Subject: {subject}
Standard ID: {standard_id}
Standard Description: {standard_desc}
Difficulty: {difficulty}

Instructions:
- Create a question that accurately assesses understanding of this specific standard.
- The question's difficulty must be '{difficulty}'.
- Return only the JSON object as specified in the schema."""
            
            # 创建助手响应模板
            assistant_template = """<think>
I need to create a grade {grade} {subject} MCQ assessing {standard_id}: {standard_desc}. 
The question should be {difficulty} difficulty.

Let me think through the design:
1. First, I'll identify the key concept from the standard...
2. Then, I'll create a clear, concise question that tests this concept...
3. I'll design 4 answer options with exactly one correct answer...
4. The distractors should be plausible but incorrect, reflecting common student errors...
5. Finally, I'll provide a brief explanation of why the correct answer is right.

Here's my designed question:
</think>
{{
  "id": "{standard_id}_mcq_{difficulty}_{i:03d}",
  "type": "mcq",
  "question": "Sample question assessing {standard_desc}",
  "answer": "B",
  "answer_options": {{
    "A": "Incorrect option 1",
    "B": "Correct option",
    "C": "Incorrect option 2",
    "D": "Incorrect option 3"
  }},
  "answer_explanation": "Option B is correct because it accurately demonstrates the concept. The other options are incorrect because they represent common misunderstandings.",
  "difficulty": "{difficulty}"
}}"""
            
            assistant_response = assistant_template.format(
                grade=grade,
                subject=subject,
                standard_id=standard_id,
                standard_desc=standard_desc,
                difficulty=difficulty,
                i=i
            )
            
            # 格式化完整对话
            formatted_text = f"<|im_start|>system\n{new_system}<|im_end|>\n"
            formatted_text += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            formatted_text += f"<|im_start|>assistant\n{assistant_response}<|im_end|>"
            
            augmented_samples.append({
                "text": formatted_text,
                "metadata": {
                    "source": "augmented",
                    "grade": grade,
                    "standard": standard_id,
                    "difficulty": difficulty,
                    "augmentation_id": i
                }
            })
        
        return sft_samples + augmented_samples
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """将对话格式转换为Qwen2.5格式"""
        formatted = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    def _format_assistant_response(self, response: str) -> str:
        """格式化助手响应"""
        return f"<|im_start|>assistant\n{response}<|im_end|>"
    
    def save_data(self, sft_samples: List[Dict], dpo_samples: List[Dict]):
        """保存处理后的数据"""
        
        # 保存SFT数据
        print(f"保存SFT数据到: {self.config.sft_output_path}")
        with open(self.config.sft_output_path, 'w', encoding='utf-8') as f:
            for sample in sft_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # 保存DPO数据
        print(f"保存DPO数据到: {self.config.dpo_output_path}")
        with open(self.config.dpo_output_path, 'w', encoding='utf-8') as f:
            for sample in dpo_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print("数据处理完成!")
        print(f"  SFT样本数: {len(sft_samples)}")
        print(f"  DPO样本数: {len(dpo_samples)}")


def main():
    """主函数：处理数据"""
    config = DataConfig(
        sft_output_path="/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/sft_data.jsonl",
        dpo_output_path="/home/ubuntu/lilei/projects/qwen32b-sft/processed_training_data/dpo_data.jsonl",
        keep_think_chain=True,
        train_test_split=0.9
    )
    
    processor = MCQDataProcessor(config)
    
    # 处理原始数据文件
    raw_data_dir = "/home/ubuntu/lilei/projects/qwen32b-sft/raw_data"
    
    all_sft_samples = []
    all_dpo_samples = []
    
    # 处理所有原始数据文件
    for file_name in os.listdir(raw_data_dir):
        if file_name.endswith('.jsonl') or file_name.endswith('.json'):
            file_path = os.path.join(raw_data_dir, file_name)
            sft_samples, dpo_samples = processor.process_file(file_path)
            all_sft_samples.extend(sft_samples)
            all_dpo_samples.extend(dpo_samples)
    
    # 数据增强
    print("进行数据增强...")
    all_sft_samples = processor.augment_data(all_sft_samples, target_count=200)
    
    # 保存数据
    processor.save_data(all_sft_samples, all_dpo_samples)


if __name__ == "__main__":
    main()

