"""
Benchmark Comparison: V1 vs V2

This script compares the performance of the original 4-stage pipeline (V1)
against the new single-pass think-tag approach (V2).

Expected improvement: 33.3% → 100% accuracy
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Import V1 and V2
from deepseek_r1_traces import DeepSeekR1Traces
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2
from parsers import ThinkTagParser

load_dotenv()


class BenchmarkRunnerV2:
    """基准测试运行器 - V1 vs V2 对比"""

    def __init__(self, model_name: str = "deepseek-r1:32b"):
        """
        初始化测试运行器

        Args:
            model_name: 使用的模型名称
        """
        self.model_name = model_name
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # 创建 LLM judge
        self.judge_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0
        )

        # 创建 V1 和 V2
        print(f"初始化 V1 (4阶段模式)...")
        self.v1_agent = DeepSeekR1Traces()

        print(f"初始化 V2 (单次思考模式)...")
        self.v2_agent = DeepSeekR1AgentV2(
            model=model_name,
            enable_tools=True,
            enable_loop_detection=True,
            enable_hallucination_detection=False  # 关闭以提高速度
        )

        # 创建 Baseline (直接回答)
        print(f"初始化 Baseline (直接回答)...")
        self.baseline_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.7
        )

        # 测试数据
        self.test_data = self._load_test_data()

    def _load_test_data(self) -> List[Dict]:
        """加载测试数据（使用原始失败案例）"""
        return [
            {
                "id": 1,
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "correct_answer": "18",
                "reasoning": "16 eggs - 3 (breakfast) - 4 (muffins) = 9 eggs. 9 * $2 = $18"
            },
            {
                "id": 2,
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "correct_answer": "3",
                "reasoning": "2 bolts blue + 1 bolt white (half of 2) = 3 bolts total"
            },
            {
                "id": 3,
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "correct_answer": "70000",
                "reasoning": "Total investment: $80k + $50k = $130k. Value increase 150% means final value = $130k * 2.5 = $325k. Profit = $325k - $130k = $195k. Wait, let me recalculate. If the value increased BY 150%, it means the house value went from some original to (original + 1.5*original) = 2.5*original. But the house was worth $80k initially. After repairs, increased BY 150% means $80k * 2.5 = $200k. No wait, increased BY 150% of $80k means $80k + ($80k * 1.5) = $80k + $120k = $200k. Profit = $200k - $130k (total investment) = $70k."
            }
        ]

    def judge_answer(self, question: str, model_answer: str, correct_answer: str) -> Dict:
        """
        使用 LLM 判断答案是否正确

        Returns:
            {"is_correct": bool, "explanation": str}
        """
        prompt = f"""你是一个严格的数学题判分员。

问题：
{question}

正确答案：
{correct_answer}

模型答案：
{model_answer}

请判断模型答案是否正确。注意：
1. 只要数值正确即可，格式、单位可以忽略
2. 如果模型答案包含正确数值即可判定为正确
3. 严格基于数学计算，不要宽容

请仅输出以下格式：
判定：正确/错误
理由：[一句话说明]
"""

        response = self.judge_llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content

        is_correct = "正确" in response_text.split('\n')[0] and "错误" not in response_text.split('\n')[0]

        return {
            "is_correct": is_correct,
            "explanation": response_text
        }

    def extract_final_number(self, text: str) -> str:
        """从文本中提取最终数字答案"""
        import re

        # 尝试提取 answer 标签中的内容
        parser = ThinkTagParser()
        parsed = parser.parse(text)
        answer_text = parsed["answer"] if parsed["answer"] else text

        # 提取数字（包括带美元符号的）
        numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)

        if numbers:
            # 返回最后一个数字（通常是最终答案）
            return numbers[-1].replace(',', '')

        return answer_text.strip()

    def run_benchmark(self, test_v1: bool = True, test_baseline: bool = True) -> Dict:
        """
        运行完整基准测试

        Args:
            test_v1: 是否测试 V1
            test_baseline: 是否测试 Baseline

        Returns:
            测试结果字典
        """
        results = {
            "baseline": [] if test_baseline else None,
            "v1": [] if test_v1 else None,
            "v2": []
        }

        for task in self.test_data:
            print(f"\n{'='*80}")
            print(f"测试任务 {task['id']}: {task['question'][:60]}...")
            print(f"{'='*80}")

            # Test Baseline
            if test_baseline:
                print(f"\n[Baseline] 直接回答...")
                try:
                    baseline_answer = self.baseline_llm.invoke([
                        HumanMessage(content=f"请回答以下问题：{task['question']}")
                    ])
                    baseline_text = baseline_answer.content
                    baseline_extracted = self.extract_final_number(baseline_text)

                    baseline_judgment = self.judge_answer(
                        task['question'],
                        baseline_extracted,
                        task['correct_answer']
                    )

                    results["baseline"].append({
                        "task_id": task['id'],
                        "answer": baseline_extracted,
                        "is_correct": baseline_judgment["is_correct"],
                        "full_output": baseline_text[:200] + "..."
                    })

                    print(f"Baseline 答案: {baseline_extracted}")
                    print(f"判定: {'✅ 正确' if baseline_judgment['is_correct'] else '❌ 错误'}")

                except Exception as e:
                    print(f"Baseline 执行失败: {e}")
                    results["baseline"].append({
                        "task_id": task['id'],
                        "answer": "ERROR",
                        "is_correct": False,
                        "error": str(e)
                    })

            # Test V1
            if test_v1:
                print(f"\n[V1] 4阶段模式...")
                try:
                    v1_output = self.v1_agent.run(task['question'])
                    v1_extracted = self.extract_final_number(v1_output)

                    v1_judgment = self.judge_answer(
                        task['question'],
                        v1_extracted,
                        task['correct_answer']
                    )

                    results["v1"].append({
                        "task_id": task['id'],
                        "answer": v1_extracted,
                        "is_correct": v1_judgment["is_correct"],
                        "full_output": v1_output[:200] + "..."
                    })

                    print(f"V1 答案: {v1_extracted}")
                    print(f"判定: {'✅ 正确' if v1_judgment['is_correct'] else '❌ 错误'}")

                except Exception as e:
                    print(f"V1 执行失败: {e}")
                    results["v1"].append({
                        "task_id": task['id'],
                        "answer": "ERROR",
                        "is_correct": False,
                        "error": str(e)
                    })

            # Test V2
            print(f"\n[V2] 单次思考模式...")
            try:
                v2_answer = self.v2_agent.run(task['question'], verbose=False)
                v2_extracted = self.extract_final_number(v2_answer)

                v2_judgment = self.judge_answer(
                    task['question'],
                    v2_extracted,
                    task['correct_answer']
                )

                results["v2"].append({
                    "task_id": task['id'],
                    "answer": v2_extracted,
                    "is_correct": v2_judgment["is_correct"],
                    "full_output": v2_answer[:200] + "..."
                })

                print(f"V2 答案: {v2_extracted}")
                print(f"判定: {'✅ 正确' if v2_judgment['is_correct'] else '❌ 错误'}")

            except Exception as e:
                print(f"V2 执行失败: {e}")
                results["v2"].append({
                    "task_id": task['id'],
                    "answer": "ERROR",
                    "is_correct": False,
                    "error": str(e)
                })

        return results

    def print_summary(self, results: Dict):
        """打印测试总结"""
        print(f"\n\n{'='*80}")
        print("测试总结")
        print(f"{'='*80}\n")

        # 计算准确率
        def calc_accuracy(result_list):
            if not result_list:
                return 0, 0, 0.0
            correct = sum(1 for r in result_list if r.get("is_correct", False))
            total = len(result_list)
            return correct, total, (correct / total * 100) if total > 0 else 0.0

        if results["baseline"]:
            b_correct, b_total, b_acc = calc_accuracy(results["baseline"])
            print(f"Baseline (直接回答):  {b_correct}/{b_total} = {b_acc:.1f}%")

        if results["v1"]:
            v1_correct, v1_total, v1_acc = calc_accuracy(results["v1"])
            print(f"V1 (4阶段模式):       {v1_correct}/{v1_total} = {v1_acc:.1f}%")

        v2_correct, v2_total, v2_acc = calc_accuracy(results["v2"])
        print(f"V2 (单次思考模式):     {v2_correct}/{v2_total} = {v2_acc:.1f}%")

        # 详细结果
        print(f"\n{'='*80}")
        print("详细结果对比")
        print(f"{'='*80}\n")

        for i, task in enumerate(self.test_data, 1):
            print(f"任务 {i}: {task['question'][:50]}...")
            print(f"  正确答案: {task['correct_answer']}")

            if results["baseline"]:
                b_result = results["baseline"][i-1]
                status = "✅" if b_result.get("is_correct") else "❌"
                print(f"  {status} Baseline: {b_result.get('answer', 'ERROR')}")

            if results["v1"]:
                v1_result = results["v1"][i-1]
                status = "✅" if v1_result.get("is_correct") else "❌"
                print(f"  {status} V1:       {v1_result.get('answer', 'ERROR')}")

            v2_result = results["v2"][i-1]
            status = "✅" if v2_result.get("is_correct") else "❌"
            print(f"  {status} V2:       {v2_result.get('answer', 'ERROR')}")
            print()

        print(f"{'='*80}")
        print("测试完成！")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    print("\n" + "="*80)
    print("DeepSeek-R1 Agent 基准测试: V1 vs V2")
    print("="*80)

    # 检查模型
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
    print(f"\n使用模型: {model_name}")
    print("注意：如果使用 32b 模型，测试可能需要较长时间\n")

    # 创建运行器
    runner = BenchmarkRunnerV2(model_name=model_name)

    # 运行测试
    # 可以通过命令行参数控制是否测试 V1 和 Baseline
    test_v1 = "--skip-v1" not in sys.argv
    test_baseline = "--skip-baseline" not in sys.argv

    if not test_v1:
        print("跳过 V1 测试 (使用 --skip-v1)")
    if not test_baseline:
        print("跳过 Baseline 测试 (使用 --skip-baseline)")

    results = runner.run_benchmark(test_v1=test_v1, test_baseline=test_baseline)

    # 打印总结
    runner.print_summary(results)
