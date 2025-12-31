#!/usr/bin/env python3
"""
V2.5 vs Baseline 对比测试 - 20题完整评估

测试集构成:
- 10题 GSM8K (5 简单 + 5 中等)
- 10题 手工设计 (3 逻辑 + 4 易混淆 + 3 常识)

目标: 量化证明 V2.5 在复杂推理和逻辑一致性上显著优于 Baseline
"""

import os
import json
import re
import time
import signal
from typing import List, Dict
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

# 超时异常
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("执行超时")

# 加载环境配置
load_dotenv(override=True)

class BenchmarkRunner20Tasks:
    def __init__(self):
        use_backend = os.getenv("USE_BACKEND", "ollama")

        if use_backend == "deepseek_api":
            # DeepSeek API 配置
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")

            print(f"Test Model: {self.model_name} (DeepSeek API)")
            print(f"Judge Model: {self.model_name} (DeepSeek API)")

            # 1. Baseline LLM
            self.baseline_llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0
            )

            # 2. V2.5 Agent
            self.agent_v2_5 = DeepSeekR1AgentV2(
                model=self.model_name,
                enable_tools=True,
                enable_loop_detection=True,
                enable_hallucination_detection=False
            )

            # 3. Judge LLM
            self.judge_llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=0
            )
        else:
            # Ollama 本地配置
            self.model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
            self.judge_model_name = os.getenv("JUDGE_MODEL", self.model_name)
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            print(f"Test Model: {self.model_name} (Ollama)")
            print(f"Judge Model: {self.judge_model_name} (Ollama)")

            # 1. Baseline LLM (普通直接回答)
            self.baseline_llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=0
            )

            # 2. V2.5 Agent (结构化4阶段推理)
            self.agent_v2_5 = DeepSeekR1AgentV2(
                model=self.model_name,
                enable_tools=True,
                enable_loop_detection=True,
                enable_hallucination_detection=False
            )

            # 3. Judge LLM
            self.judge_llm = ChatOllama(
                model=self.judge_model_name,
                base_url=self.base_url,
                temperature=0
            )

    def load_gsm8k_data(self, limit=10) -> List[Dict]:
        """加载 GSM8K 测试数据"""
        file_path = os.path.join(os.path.dirname(__file__),
                               "../18-dsa-compression-experiment/benchmarks/data/reasoning/gsm8k_hf.json")

        if not os.path.exists(file_path):
            print(f"Warning: GSM8K file not found at {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tasks = []
        for item in data[:limit]:
            history = item.get('history', [])
            if not history:
                continue

            question = history[0]['content']
            question = question.replace("Please solve this math problem: ", "")
            gold_answer = history[1]['content']

            tasks.append({
                "id": item['name'],
                "question": question,
                "gold_answer": gold_answer,
                "category": "gsm8k_simple" if tasks.__len__() < 5 else "gsm8k_medium"
            })

        print(f"Loaded {len(tasks)} tasks from GSM8K.")
        return tasks

    def get_hand_crafted_tasks(self) -> List[Dict]:
        """手工设计的10道测试题"""
        tasks = []

        # === 1. 逻辑谜题 (3题) ===
        tasks.append({
            "id": "Logic_Knights_Knaves",
            "category": "logic_puzzle",
            "question": "Three people (A, B, C) are either Knights (always tell truth) or Knaves (always lie). A says: 'B is a knave'. B says: 'A and C are the same type'. C says: 'I am a Knight'. Determine who is who.",
            "gold_answer": "A is a Knight, B is a Knave, C is a Knave."
        })

        tasks.append({
            "id": "Logic_Truth_Teller",
            "category": "logic_puzzle",
            "question": "In a room, there are two people. One always tells the truth, one always lies. The first person says: 'We are both liars.' What is each person?",
            "gold_answer": "The first person is a liar, the second person is a truth-teller. The first person's statement ('We are both liars') must be false because if they were both liars, the statement would be true (a contradiction). So the first is a liar, and thus the second must be a truth-teller."
        })

        tasks.append({
            "id": "Logic_Hats",
            "category": "logic_puzzle",
            "question": "Three people wearing hats (red or blue). Each can see others' hats but not their own. First person says 'I don't know my color'. Second says 'I don't know either'. Third says 'I know my color is blue'. What are the hat colors?",
            "gold_answer": "First: Red, Second: Red, Third: Blue. If the third person can deduce their color after hearing the first two don't know, it must be because the first two are wearing the same color (red), so the third knows they must be wearing the other color (blue)."
        })

        # === 2. 易混淆/易误导题 (4题) ===
        tasks.append({
            "id": "Confusing_Feathers",
            "category": "misleading",
            "question": "Which is heavier: a pound of bricks or a pound of feathers?",
            "gold_answer": "They weigh the same. Both are one pound."
        })

        tasks.append({
            "id": "Confusing_Robe_Fiber",
            "category": "misleading",
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "gold_answer": "3 bolts total. Blue: 2 bolts, White: 1 bolt (half of 2), Total: 2 + 1 = 3."
        })

        tasks.append({
            "id": "Confusing_House_Flip",
            "category": "misleading",
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "gold_answer": "$70,000 profit. Cost: $80k + $50k = $130k. Increased value BY 150% means: new value = $80k × 2.5 = $200k. Profit: $200k - $130k = $70k."
        })

        tasks.append({
            "id": "Confusing_Percentage",
            "category": "misleading",
            "question": "A store has a shirt that costs $40. They increase the price by 50%, then decrease it by 50%. What is the final price?",
            "gold_answer": "$30. First increase: $40 × 1.5 = $60. Then decrease: $60 × 0.5 = $30. (Not back to $40 because percentages are based on different bases!)"
        })

        # === 3. 常识推理题 (3题) ===
        tasks.append({
            "id": "Common_Sense_Age",
            "category": "common_sense",
            "question": "If Sally is 15 years old and her brother is half her age, how old will her brother be when Sally is 30?",
            "gold_answer": "27.5 years old. When Sally is 15, her brother is 7.5. Age difference: 7.5 years. When Sally is 30, her brother is 30 - 7.5 = 22.5. Actually, if 'half her age' means 15/2 = 7.5, difference is 7.5. So when Sally is 30, brother is 22.5. Or if it means brother is currently 7.5, then when Sally is 30 (15 years later), brother is 7.5 + 15 = 22.5."
        })

        tasks.append({
            "id": "Common_Sense_Clock",
            "category": "common_sense",
            "question": "A clock shows 3:15. What is the angle between the hour hand and the minute hand?",
            "gold_answer": "7.5 degrees. At 3:15, minute hand is at 3 (90° from 12). Hour hand is 1/4 of the way between 3 and 4. Each hour = 30°, 15 min = 30°/4 = 7.5°. Hour hand is at 90° + 7.5° = 97.5°. Angle = 97.5° - 90° = 7.5°."
        })

        tasks.append({
            "id": "Common_Sense_Speed",
            "category": "common_sense",
            "question": "A car travels 60 miles in 1 hour on the highway, then 30 miles in 1 hour in the city. What is the average speed for the entire trip?",
            "gold_answer": "45 mph. Total distance: 60 + 30 = 90 miles. Total time: 1 + 1 = 2 hours. Average speed: 90 / 2 = 45 mph."
        })

        return tasks

    def clean_think_tags(self, text: str) -> str:
        """清理 <think> 标签"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    def evaluate_correctness(self, question: str, gold_answer: str, candidate_answer: str, category: str) -> bool:
        """使用 LLM 裁判判断答案是否正确"""
        candidate_answer = self.clean_think_tags(candidate_answer)
        gold_answer = self.clean_think_tags(gold_answer)

        prompt = f"""
你是一个数学和逻辑阅卷老师。请判断考生的答案是否正确。

【题目】
{question}

【标准答案】
{gold_answer}

【考生答案】
{candidate_answer}

请注意：
1. 只要最终结论或数值结果正确，即使过程略有不同也算对。
2. 如果标准答案包含 "#### X"，则 X 是最终数值。
3. 对于逻辑题（如Knights/Knaves），请检查最终的角色分配是否一致。
4. 请忽略格式差异。
5. 对于中英文混合，"骑士"=Knight, "小人"=Knave

请只输出 "CORRECT" 或 "INCORRECT"。
        """

        try:
            response = self.judge_llm.invoke(prompt).content.strip()

            if "CORRECT" in response.upper() and "INCORRECT" not in response.upper():
                return True

            # Fallback: 简单的数值匹配
            cand_nums = re.findall(r'\d+\.?\d*', candidate_answer)
            gold_nums = re.findall(r'\d+\.?\d*', gold_answer)

            if "####" in gold_answer:
                gold_val = gold_answer.split("####")[-1].strip()
                gold_nums = re.findall(r'\d+\.?\d*', gold_val)

            if gold_nums and cand_nums:
                if abs(float(cand_nums[-1]) - float(gold_nums[-1])) < 1e-6:
                    return True

            return False
        except Exception as e:
            print(f"Judge Error: {e}")
            return False

    def run_benchmark(self):
        """运行完整的 20 题 benchmark"""
        # 1. 加载测试数据
        gsm8k_tasks = self.load_gsm8k_data(limit=10)
        hand_tasks = self.get_hand_crafted_tasks()
        all_tasks = gsm8k_tasks + hand_tasks

        if len(all_tasks) < 20:
            print(f"Warning: Only loaded {len(all_tasks)} tasks (expected 20)")

        print(f"\n{'='*60}")
        print(f"开始 V2.5 vs Baseline 对比测试 (20题)")
        print(f"{'='*60}\n")

        results = {
            "baseline": {
                "correct": 0,
                "total": 0,
                "times": [],
                "by_category": {}
            },
            "v2_5": {
                "correct": 0,
                "total": 0,
                "times": [],
                "by_category": {}
            }
        }

        # 2. 运行每个测试
        for idx, task in enumerate(all_tasks, 1):
            category = task.get('category', 'unknown')
            print(f"\n{'='*60}")
            print(f"Task {idx}/20: {task['id']} [{category}]")
            print(f"问题: {task['question'][:80]}...")
            print(f"{'='*60}")

            # ========== Baseline ==========
            print(f"\n  [1/2] Running Baseline (Direct Answer)...", end=" ", flush=True)
            start_time = time.time()
            baseline_correct = False
            baseline_time = 0

            # 设置5分钟超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5分钟 = 300秒

            try:
                baseline_response = self.baseline_llm.invoke([HumanMessage(content=task['question'])])
                baseline_answer = baseline_response.content
                baseline_time = time.time() - start_time
                signal.alarm(0)  # 取消超时

                baseline_correct = self.evaluate_correctness(
                    task['question'],
                    task['gold_answer'],
                    baseline_answer,
                    category
                )
                results["baseline"]["total"] += 1
                results["baseline"]["times"].append(baseline_time)
                if baseline_correct:
                    results["baseline"]["correct"] += 1

                # 按类别统计
                if category not in results["baseline"]["by_category"]:
                    results["baseline"]["by_category"][category] = {"correct": 0, "total": 0}
                results["baseline"]["by_category"][category]["total"] += 1
                if baseline_correct:
                    results["baseline"]["by_category"][category]["correct"] += 1

                print(f"Time: {baseline_time:.2f}s | Result: {'✅' if baseline_correct else '❌'}")
            except TimeoutException:
                signal.alarm(0)
                baseline_time = time.time() - start_time
                print(f"⏱️ TIMEOUT ({baseline_time:.0f}s) | Result: ❌")
                results["baseline"]["total"] += 1
                results["baseline"]["times"].append(baseline_time)
                if category not in results["baseline"]["by_category"]:
                    results["baseline"]["by_category"][category] = {"correct": 0, "total": 0}
                results["baseline"]["by_category"][category]["total"] += 1
            except Exception as e:
                signal.alarm(0)
                print(f"Error: {e}")
                results["baseline"]["total"] += 1

            # ========== V2.5 Structured 4-Stage ==========
            print(f"  [2/2] Running V2.5 (Structured 4-Stage)...", end=" ", flush=True)
            start_time = time.time()
            v2_5_correct = False
            v2_5_time = 0

            # 设置5分钟超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5分钟 = 300秒

            try:
                # 强制使用 structured_4stage 模式
                v2_5_answer = self.agent_v2_5.run(
                    task['question'],
                    mode="structured_4stage",
                    verbose=False
                )
                v2_5_time = time.time() - start_time
                signal.alarm(0)  # 取消超时

                v2_5_correct = self.evaluate_correctness(
                    task['question'],
                    task['gold_answer'],
                    v2_5_answer,
                    category
                )
                results["v2_5"]["total"] += 1
                results["v2_5"]["times"].append(v2_5_time)
                if v2_5_correct:
                    results["v2_5"]["correct"] += 1

                # 按类别统计
                if category not in results["v2_5"]["by_category"]:
                    results["v2_5"]["by_category"][category] = {"correct": 0, "total": 0}
                results["v2_5"]["by_category"][category]["total"] += 1
                if v2_5_correct:
                    results["v2_5"]["by_category"][category]["correct"] += 1

                print(f"Time: {v2_5_time:.2f}s | Result: {'✅' if v2_5_correct else '❌'}")
            except TimeoutException:
                signal.alarm(0)
                v2_5_time = time.time() - start_time
                print(f"⏱️ TIMEOUT ({v2_5_time:.0f}s) | Result: ❌")
                results["v2_5"]["total"] += 1
                results["v2_5"]["times"].append(v2_5_time)
                if category not in results["v2_5"]["by_category"]:
                    results["v2_5"]["by_category"][category] = {"correct": 0, "total": 0}
                results["v2_5"]["by_category"][category]["total"] += 1
            except Exception as e:
                signal.alarm(0)
                print(f"Error: {e}")
                results["v2_5"]["total"] += 1

        # 3. 输出最终结果
        self.generate_report(results)

    def generate_report(self, results: Dict):
        """生成详细对比报告"""
        print(f"\n\n{'='*60}")
        print("V2.5 vs Baseline 对比测试结果（20题）")
        print(f"{'='*60}\n")

        # 1. 总体准确率对比
        baseline_acc = (results["baseline"]["correct"] / results["baseline"]["total"] * 100) if results["baseline"]["total"] > 0 else 0
        v2_5_acc = (results["v2_5"]["correct"] / results["v2_5"]["total"] * 100) if results["v2_5"]["total"] > 0 else 0

        print("【准确率对比】")
        print(f"Baseline:    {baseline_acc:.1f}% ({results['baseline']['correct']}/{results['baseline']['total']})")
        print(f"V2.5:        {v2_5_acc:.1f}% ({results['v2_5']['correct']}/{results['v2_5']['total']})")
        print(f"提升幅度:    {v2_5_acc - baseline_acc:+.1f}%\n")

        # 2. 分类别表现
        print("【分类别表现】")
        print(f"{'类别':<20} {'Baseline':<15} {'V2.5':<15} {'说明'}")
        print("-" * 70)

        all_categories = set(list(results["baseline"]["by_category"].keys()) + list(results["v2_5"]["by_category"].keys()))
        for cat in sorted(all_categories):
            baseline_cat = results["baseline"]["by_category"].get(cat, {"correct": 0, "total": 0})
            v2_5_cat = results["v2_5"]["by_category"].get(cat, {"correct": 0, "total": 0})

            baseline_pct = (baseline_cat["correct"] / baseline_cat["total"] * 100) if baseline_cat["total"] > 0 else 0
            v2_5_pct = (v2_5_cat["correct"] / v2_5_cat["total"] * 100) if v2_5_cat["total"] > 0 else 0

            desc = ""
            if cat == "gsm8k_simple":
                desc = "(对照组)"
            elif cat == "logic_puzzle":
                desc = "(核心验证)"
            elif cat == "misleading":
                desc = "(幻觉防护)"

            print(f"{cat:<20} {baseline_pct:>5.0f}% ({baseline_cat['correct']}/{baseline_cat['total']})      {v2_5_pct:>5.0f}% ({v2_5_cat['correct']}/{v2_5_cat['total']})      {desc}")

        # 3. 推理时间
        baseline_avg_time = sum(results["baseline"]["times"]) / len(results["baseline"]["times"]) if results["baseline"]["times"] else 0
        v2_5_avg_time = sum(results["v2_5"]["times"]) / len(results["v2_5"]["times"]) if results["v2_5"]["times"] else 0

        print(f"\n【推理时间】")
        print(f"Baseline: 平均 {baseline_avg_time:.1f}s/题")
        print(f"V2.5:     平均 {v2_5_avg_time:.1f}s/题")
        if baseline_avg_time > 0:
            print(f"时间比:    {v2_5_avg_time/baseline_avg_time:.1f}x")

        # 4. 结论
        print(f"\n【结论】")
        if v2_5_acc > baseline_acc:
            print(f"✅ V2.5 在准确率上显著优于 Baseline (+{v2_5_acc - baseline_acc:.1f}%)")
        elif v2_5_acc == baseline_acc:
            print(f"⚖️ V2.5 与 Baseline 表现相当")
        else:
            print(f"⚠️ V2.5 表现不如 Baseline (可能需要进一步优化)")

        if v2_5_avg_time > baseline_avg_time * 2:
            print(f"⚠️ 代价是推理时间增加约 {v2_5_avg_time/baseline_avg_time:.1f} 倍")

        print(f"💡 建议: 简单问题用 single_think，复杂问题用 structured_4stage")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    runner = BenchmarkRunner20Tasks()
    runner.run_benchmark()
