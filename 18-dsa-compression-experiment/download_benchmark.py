import os
# 设置 Hugging Face 镜像地址，加速国内下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import random
from typing import List, Dict
from datasets import load_dataset

# 目标目录
DATA_ROOT = os.path.join(os.path.dirname(__file__), "benchmarks", "data")
SAMPLE_SIZE = 5  # 每个数据集下载多少条作为样本

def save_samples(samples: List[Dict], category: str, filename: str):
    """保存转换后的样本到 JSON 文件"""
    output_dir = os.path.join(DATA_ROOT, category)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} samples to {output_path}")

def download_gsm8k():
    """下载 GSM8K (Grade School Math) - 逻辑推理"""
    print("\nDownloading GSM8K (Reasoning)...")
    try:
        dataset = load_dataset("gsm8k", "main", split="test", streaming=True)
        samples = []
        
        # 迭代数据集
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            # GSM8K 格式: {'question': '...', 'answer': '...'}
            # 我们需要构造多轮对话历史来模拟“上下文压缩”场景
            # 策略：把 question 拆成两部分，或者构造一个基于 question 的对话
            
            q = item['question']
            a = item['answer']
            
            # 构造场景：用户问问题 -> AI 回答 -> 用户追问（这里我们模拟一个追问）
            samples.append({
                "name": f"GSM8K_Sample_{count}",
                "description": "Math reasoning task from GSM8K dataset.",
                "history": [
                    {"role": "user", "content": f"Please solve this math problem: {q}"},
                    {"role": "ai", "content": f"<think>Solving math problem. Strategy: Step-by-step.</think> {a}"}
                ],
                "test_input": "Can you explain the first step again in simpler terms?", # 通用追问
                "expected_style": "Explain the first logical step of the previous solution clearly.",
                "negative_style": "Giving a wrong explanation or hallucinating a different problem."
            })
            count += 1
            
        save_samples(samples, "reasoning", "gsm8k_hf.json")
    except Exception as e:
        print(f"Error downloading GSM8K: {e}")

def download_mmlu():
    """下载 MMLU (Massive Multitask Language Understanding) - 通用知识"""
    print("\nDownloading MMLU (Knowledge)...")
    try:
        # MMLU 在 HuggingFace 上叫 "cais/mmlu" 或 "lukaemon/mmlu" 等，这里用轻量级的配置
        # 为了演示，我们使用 'cais/mmlu' 的 'astronomy' 子集
        dataset = load_dataset("cais/mmlu", "astronomy", split="test", streaming=True)
        samples = []
        
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            q = item['question']
            options = item['choices']
            answer_idx = item['answer']
            correct_answer = options[answer_idx]
            
            samples.append({
                "name": f"MMLU_Astronomy_{count}",
                "description": "Astronomy knowledge task from MMLU.",
                "history": [
                    {"role": "user", "content": f"Quiz time! {q}\nOptions: {options}"},
                    {"role": "ai", "content": f"<think>Question: {q}. Correct Answer: {correct_answer}.</think> The correct answer is {correct_answer}."}
                ],
                "test_input": "Why is that the correct answer?",
                "expected_style": "Provide scientific explanation related to the specific astronomical fact.",
                "negative_style": "Generic answer or incorrect reasoning."
            })
            count += 1
            
        save_samples(samples, "knowledge", "mmlu_hf.json")
    except Exception as e:
        print(f"Error downloading MMLU: {e}")

def download_humaneval():
    """下载 HumanEval - 代码生成"""
    print("\nDownloading HumanEval (Coding)...")
    try:
        dataset = load_dataset("openai_humaneval", split="test", streaming=True)
        samples = []
        
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            prompt = item['prompt']
            # HumanEval 的 prompt 通常是函数签名和 docstring
            
            samples.append({
                "name": f"HumanEval_{count}",
                "description": "Python coding task from HumanEval.",
                "history": [
                    {"role": "user", "content": f"Complete this Python function:\n{prompt}"},
                    {"role": "ai", "content": f"<think>Task: Code completion. Strategy: Follow docstring specs.</think> Here is the implementation:\n```python\n{item['canonical_solution']}\n```"}
                ],
                "test_input": "Can you add error handling to this function?",
                "expected_style": "Add try-except blocks or input validation to the specific function logic.",
                "negative_style": "Writing a completely different function."
            })
            count += 1
            
        save_samples(samples, "coding", "humaneval_hf.json")
    except Exception as e:
        print(f"Error downloading HumanEval: {e}")

def download_hotpotqa():
    """下载 HotpotQA - 多跳推理 (Multi-hop Reasoning)"""
    print("\nDownloading HotpotQA (Complex Reasoning)...")
    try:
        # 使用 'distractor' 配置，包含干扰项，增加难度
        dataset = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
        samples = []
        
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            question = item['question']
            answer = item['answer']
            # context 是一个字典，包含 parallel lists: 'title' 和 'sentences'
            # 我们将其拼接成一段长文本作为背景知识
            context_texts = []
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                text = "".join(sentences)
                context_texts.append(f"Title: {title}\nContent: {text}")
            
            # 选取前2个相关文档作为上下文（避免过长，但保留复杂性）
            full_context = "\n\n".join(context_texts[:3])
            
            samples.append({
                "name": f"HotpotQA_{count}",
                "description": "Multi-hop reasoning task requiring information synthesis.",
                "history": [
                    {"role": "user", "content": f"Here is some background information:\n{full_context}\n\nBased on this, answer the following question: {question}"},
                    {"role": "ai", "content": f"<think>Analyzing multiple documents. Synthesizing facts.</think> {answer}"}
                ],
                "test_input": "What specific facts from the text led you to this conclusion?",
                "expected_style": "Cite specific facts or sentences from the provided background context.",
                "negative_style": "Vague answer or using outside knowledge not in the text."
            })
            count += 1
            
        save_samples(samples, "reasoning", "hotpotqa_hf.json")
    except Exception as e:
        print(f"Error downloading HotpotQA: {e}")

def download_truthfulqa():
    """下载 TruthfulQA - 真实性与幻觉检测 (Reliability)"""
    print("\nDownloading TruthfulQA (Reliability)...")
    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation", streaming=True)
        samples = []
        
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            question = item['question']
            best_answer = item['best_answer']
            
            samples.append({
                "name": f"TruthfulQA_{count}",
                "description": "Test for hallucination and common misconceptions.",
                "history": [
                    {"role": "user", "content": f"Question: {question}"},
                    {"role": "ai", "content": f"<think>Checking for common misconceptions. ensuring factual accuracy.</think> {best_answer}"}
                ],
                "test_input": "Are you sure? Some people might say otherwise.",
                "expected_style": "Maintain factual accuracy and politely correct misconceptions without wavering.",
                "negative_style": "Agreeing with the user's potential misconception or hallucinating false info."
            })
            count += 1
            
        save_samples(samples, "knowledge", "truthfulqa_hf.json")
    except Exception as e:
        print(f"Error downloading TruthfulQA: {e}")

def download_ultrachat():
    """下载 UltraChat - 多轮对话 (Multi-turn Dialogue)"""
    print("\nDownloading UltraChat (Conversation)...")
    try:
        # 使用 HuggingFaceH4/ultrachat_200k 的 test_sft split
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True)
        samples = []
        
        count = 0
        for item in dataset:
            if count >= SAMPLE_SIZE: break
            
            # UltraChat 格式: messages list [{'content': '...', 'role': 'user'}, ...]
            messages = item['messages']
            
            # 我们截取对话的前几轮作为历史
            if len(messages) < 4: continue # 跳过太短的对话
            
            history_msgs = []
            # 转换前2-3轮
            for msg in messages[:-1]:
                role = "user" if msg['role'] == "user" else "ai"
                content = msg['content']
                if role == "ai":
                    content = f"<think>Processing conversation context.</think> {content}"
                history_msgs.append({"role": role, "content": content})
            
            # 最后一轮作为测试输入（这里我们稍微改写一下，让它变成一个新的测试点）
            # 或者直接取最后一条用户消息作为 test_input
            last_msg = messages[-1]
            if last_msg['role'] == 'user':
                test_input = last_msg['content']
                expected_style = "Continue the conversation naturally based on previous context."
                negative_style = "Ignore previous context or change topic abruptly."
            else:
                # 如果最后是 AI，我们构造一个通用的追问
                test_input = "Can you summarize our discussion so far?"
                expected_style = "Summarize the key points of the conversation history."
                negative_style = "Generic summary unrelated to specific details discussed."

            samples.append({
                "name": f"UltraChat_{count}",
                "description": "Real-world multi-turn conversation.",
                "history": history_msgs,
                "test_input": test_input,
                "expected_style": expected_style,
                "negative_style": negative_style
            })
            count += 1
            
        save_samples(samples, "roleplay", "ultrachat_hf.json")
    except Exception as e:
        print(f"Error downloading UltraChat: {e}")

if __name__ == "__main__":
    print("Starting download of Hugging Face benchmarks...")
    print("Note: This requires 'datasets' library (`pip install datasets`).")
    
    download_gsm8k()
    download_mmlu()
    download_humaneval()
    download_hotpotqa()
    download_truthfulqa()
    download_ultrachat()
    
    print("\nDone! New benchmarks are ready in benchmarks/data/")
