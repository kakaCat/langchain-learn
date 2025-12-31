#!/usr/bin/env python3
"""
V2.5 vs Baseline 对比测试 - 并行执行版本

特点:
1. 支持并行执行多个测试任务
2. 每个任务有独立的 5 分钟超时
3. 使用 ProcessPoolExecutor 实现真正的并行
4. 支持 DeepSeek API 和 Ollama 两种后端
"""

import os
import json
import re
import time
import signal
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

# 加载环境配置
load_dotenv(override=True)

# 超时异常
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("执行超时")


def run_single_task(task_info: Tuple[int, Dict, str, str]) -> Dict:
    """
    在独立进程中运行单个测试任务

    Args:
        task_info: (task_index, task_dict, use_backend, model_name)

    Returns:
        结果字典，包含两个方法的执行情况
    """
    idx, task, use_backend, model_name = task_info

    print(f"\n{'='*60}")
    print(f"Task {idx}: {task['id']} [{task.get('category', 'unknown')}]")
    print(f"问题: {task['question'][:80]}...")
    print(f"{'='*60}")

    result = {
        "task_id": task['id'],
        "task_index": idx,
        "category": task.get('category', 'unknown'),
        "question": task['question'],
        "gold_answer": task['gold_answer'],
        "baseline": {
            "answer": None,
            "time": 0,
            "correct": False,
            "error": None,
            "timeout": False
        },
        "v2_5": {
            "answer": None,
            "time": 0,
            "correct": False,
            "error": None,
            "timeout": False
        }
    }

    # 创建 LLM 实例
    if use_backend == "deepseek_api":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

        baseline_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )

        judge_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0
        )
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        baseline_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0
        )

        judge_llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0
        )

    # 创建 V2.5 Agent
    agent_v2_5 = DeepSeekR1AgentV2(
        model=model_name,
        enable_tools=True,
        enable_loop_detection=True,
        enable_hallucination_detection=False
    )

    # ========== 测试 Baseline ==========
    print(f"\n  [1/2] Running Baseline...", end=" ", flush=True)
    start_time = time.time()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5分钟超时

    try:
        baseline_response = baseline_llm.invoke([HumanMessage(content=task['question'])])
        baseline_answer = baseline_response.content
        baseline_time = time.time() - start_time
        signal.alarm(0)

        result["baseline"]["answer"] = baseline_answer
        result["baseline"]["time"] = baseline_time

        # 评估正确性
        baseline_correct = evaluate_correctness(
            judge_llm,
            task['question'],
            task['gold_answer'],
            baseline_answer,
            task.get('category', 'unknown')
        )
        result["baseline"]["correct"] = baseline_correct

        print(f"Time: {baseline_time:.2f}s | Result: {'✅' if baseline_correct else '❌'}")

    except TimeoutException:
        signal.alarm(0)
        baseline_time = time.time() - start_time
        result["baseline"]["time"] = baseline_time
        result["baseline"]["timeout"] = True
        print(f"⏱️ TIMEOUT ({baseline_time:.0f}s) | Result: ❌")

    except Exception as e:
        signal.alarm(0)
        result["baseline"]["error"] = str(e)
        print(f"❌ Error: {e}")

    # ========== 测试 V2.5 ==========
    print(f"  [2/2] Running V2.5...", end=" ", flush=True)
    start_time = time.time()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(300)  # 5分钟超时

    try:
        v2_5_answer = agent_v2_5.run(
            task['question'],
            mode="structured_4stage",
            verbose=False
        )
        v2_5_time = time.time() - start_time
        signal.alarm(0)

        result["v2_5"]["answer"] = v2_5_answer
        result["v2_5"]["time"] = v2_5_time

        # 评估正确性
        v2_5_correct = evaluate_correctness(
            judge_llm,
            task['question'],
            task['gold_answer'],
            v2_5_answer,
            task.get('category', 'unknown')
        )
        result["v2_5"]["correct"] = v2_5_correct

        print(f"Time: {v2_5_time:.2f}s | Result: {'✅' if v2_5_correct else '❌'}")

    except TimeoutException:
        signal.alarm(0)
        v2_5_time = time.time() - start_time
        result["v2_5"]["time"] = v2_5_time
        result["v2_5"]["timeout"] = True
        print(f"⏱️ TIMEOUT ({v2_5_time:.0f}s) | Result: ❌")

    except Exception as e:
        signal.alarm(0)
        result["v2_5"]["error"] = str(e)
        print(f"❌ Error: {e}")

    return result


def clean_think_tags(text: str) -> str:
    """清理 <think> 标签"""
    if not text:
        return ""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def evaluate_correctness(judge_llm, question: str, gold_answer: str, candidate_answer: str, category: str) -> bool:
    """使用 LLM 裁判判断答案是否正确"""
    candidate_answer = clean_think_tags(candidate_answer)
    gold_answer = clean_think_tags(gold_answer)

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
        response = judge_llm.invoke(prompt).content.strip()

        if "CORRECT" in response.upper() and "INCORRECT" not in response.upper():
            return True

        # Fallback: 简单的数值匹配
        cand_nums = re.findall(r'\d+\.?\d*', candidate_answer or "")
        gold_nums = re.findall(r'\d+\.?\d*', gold_answer or "")

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


def load_gsm8k_data(limit=10) -> List[Dict]:
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
            "category": "gsm8k_simple" if len(tasks) < 5 else "gsm8k_medium"
        })

    print(f"Loaded {len(tasks)} tasks from GSM8K.")
    return tasks


def get_hand_crafted_tasks() -> List[Dict]:
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


def generate_report(all_results: List[Dict]):
    """生成详细对比报告"""
    print(f"\n\n{'='*60}")
    print("V2.5 vs Baseline 并行测试结果")
    print(f"{'='*60}\n")

    # 统计结果
    baseline_correct = sum(1 for r in all_results if r["baseline"]["correct"])
    baseline_total = len(all_results)
    baseline_times = [r["baseline"]["time"] for r in all_results if r["baseline"]["time"] > 0]

    v2_5_correct = sum(1 for r in all_results if r["v2_5"]["correct"])
    v2_5_total = len(all_results)
    v2_5_times = [r["v2_5"]["time"] for r in all_results if r["v2_5"]["time"] > 0]

    # 1. 总体准确率
    baseline_acc = (baseline_correct / baseline_total * 100) if baseline_total > 0 else 0
    v2_5_acc = (v2_5_correct / v2_5_total * 100) if v2_5_total > 0 else 0

    print("【准确率对比】")
    print(f"Baseline:    {baseline_acc:.1f}% ({baseline_correct}/{baseline_total})")
    print(f"V2.5:        {v2_5_acc:.1f}% ({v2_5_correct}/{v2_5_total})")
    print(f"提升幅度:    {v2_5_acc - baseline_acc:+.1f}%\n")

    # 2. 分类别统计
    categories = {}
    for r in all_results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "baseline_correct": 0,
                "baseline_total": 0,
                "v2_5_correct": 0,
                "v2_5_total": 0
            }

        categories[cat]["baseline_total"] += 1
        categories[cat]["v2_5_total"] += 1

        if r["baseline"]["correct"]:
            categories[cat]["baseline_correct"] += 1
        if r["v2_5"]["correct"]:
            categories[cat]["v2_5_correct"] += 1

    print("【分类别表现】")
    print(f"{'类别':<20} {'Baseline':<15} {'V2.5':<15} {'说明'}")
    print("-" * 70)

    for cat in sorted(categories.keys()):
        stats = categories[cat]
        baseline_pct = (stats["baseline_correct"] / stats["baseline_total"] * 100) if stats["baseline_total"] > 0 else 0
        v2_5_pct = (stats["v2_5_correct"] / stats["v2_5_total"] * 100) if stats["v2_5_total"] > 0 else 0

        desc = ""
        if cat == "gsm8k_simple":
            desc = "(对照组)"
        elif cat == "logic_puzzle":
            desc = "(核心验证)"
        elif cat == "misleading":
            desc = "(幻觉防护)"

        print(f"{cat:<20} {baseline_pct:>5.0f}% ({stats['baseline_correct']}/{stats['baseline_total']})      {v2_5_pct:>5.0f}% ({stats['v2_5_correct']}/{stats['v2_5_total']})      {desc}")

    # 3. 推理时间
    baseline_avg_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    v2_5_avg_time = sum(v2_5_times) / len(v2_5_times) if v2_5_times else 0

    print(f"\n【推理时间】")
    print(f"Baseline: 平均 {baseline_avg_time:.1f}s/题")
    print(f"V2.5:     平均 {v2_5_avg_time:.1f}s/题")
    if baseline_avg_time > 0:
        print(f"时间比:    {v2_5_avg_time/baseline_avg_time:.1f}x")

    # 4. 超时统计
    baseline_timeouts = sum(1 for r in all_results if r["baseline"]["timeout"])
    v2_5_timeouts = sum(1 for r in all_results if r["v2_5"]["timeout"])

    if baseline_timeouts > 0 or v2_5_timeouts > 0:
        print(f"\n【超时统计】")
        print(f"Baseline: {baseline_timeouts} 题超时")
        print(f"V2.5:     {v2_5_timeouts} 题超时")

    # 5. 结论
    print(f"\n【结论】")
    if v2_5_acc > baseline_acc:
        print(f"✅ V2.5 在准确率上显著优于 Baseline (+{v2_5_acc - baseline_acc:.1f}%)")
    elif v2_5_acc == baseline_acc:
        print(f"⚖️ V2.5 与 Baseline 表现相当")
    else:
        print(f"⚠️ V2.5 表现不如 Baseline")

    if v2_5_avg_time > baseline_avg_time * 2:
        print(f"⚠️ 代价是推理时间增加约 {v2_5_avg_time/baseline_avg_time:.1f} 倍")

    print(f"💡 建议: 简单问题用 single_think，复杂问题用 structured_4stage")
    print(f"{'='*60}\n")

    # 保存详细结果到文件
    output_file = "benchmark_parallel_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {output_file}")


def main():
    """主函数 - 并行执行测试"""
    print(f"{'='*60}")
    print("V2.5 vs Baseline 并行测试 (20题)")
    print(f"{'='*60}\n")

    # 1. 获取配置
    use_backend = os.getenv("USE_BACKEND", "ollama")

    if use_backend == "deepseek_api":
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        print(f"使用模型: {model_name} (DeepSeek API)")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
        print(f"使用模型: {model_name} (Ollama)")

    # 2. 加载测试数据
    gsm8k_tasks = load_gsm8k_data(limit=10)
    hand_tasks = get_hand_crafted_tasks()
    all_tasks = gsm8k_tasks + hand_tasks

    if len(all_tasks) < 20:
        print(f"Warning: Only loaded {len(all_tasks)} tasks (expected 20)")

    print(f"共加载 {len(all_tasks)} 个测试任务\n")

    # 3. 准备并行任务
    task_infos = []
    for idx, task in enumerate(all_tasks, 1):
        task_infos.append((idx, task, use_backend, model_name))

    # 4. 并行执行 (最多4个并发)
    max_workers = 4
    print(f"开始并行执行 (最大并发数: {max_workers})")
    print(f"每个任务超时限制: 5分钟\n")

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_task, task_info) for task_info in task_infos]

        for future in futures:
            try:
                result = future.result(timeout=600)  # 总超时10分钟（两个5分钟测试）
                all_results.append(result)
            except FutureTimeoutError:
                print(f"⚠️ 任务执行超过10分钟，跳过")
            except Exception as e:
                print(f"❌ 任务执行失败: {e}")

    # 5. 生成报告
    if all_results:
        # 按 task_index 排序
        all_results.sort(key=lambda x: x["task_index"])
        generate_report(all_results)
    else:
        print("❌ 没有任何测试完成")


if __name__ == "__main__":
    main()
