#!/usr/bin/env python3
"""
V2.5 vs Baseline 高难度题目测试

专门选择复杂的逻辑推理、数学和常识题目来测试 V2.5 的优势
"""

import os
import json
import re
import time
import signal
from typing import List, Dict, Tuple
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
    """在独立进程中运行单个测试任务"""
    idx, task, use_backend, model_name = task_info

    print(f"\n{'='*60}")
    print(f"Task {idx}: {task['id']}")
    print(f"难度: {task.get('difficulty', 'unknown')}")
    print(f"问题: {task['question'][:100]}...")
    print(f"{'='*60}")

    result = {
        "task_id": task['id'],
        "task_index": idx,
        "difficulty": task.get('difficulty', 'unknown'),
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
2. 对于逻辑题，请检查最终的结论是否逻辑一致。
3. 对于数学题，请检查数值计算是否正确。
4. 请忽略格式差异。
5. 如果标准答案有多个可能的正确答案，考生答对其中一个即算对。

请只输出 "CORRECT" 或 "INCORRECT"。
    """

    try:
        response = judge_llm.invoke(prompt).content.strip()

        if "CORRECT" in response.upper() and "INCORRECT" not in response.upper():
            return True

        return False
    except Exception as e:
        print(f"Judge Error: {e}")
        return False


def get_hard_questions() -> List[Dict]:
    """返回高难度测试题目"""
    tasks = []

    # === 1. 复杂逻辑推理题 ===

    # 1.1 经典逻辑谜题 - The Hardest Logic Puzzle Ever
    tasks.append({
        "id": "Hard_Logic_Three_Gods",
        "difficulty": "very_hard",
        "category": "logic",
        "question": """Three gods A, B, and C are called Truth, False, and Random. Truth always speaks truly, False always speaks falsely, but Random speaks truly or falsely randomly. You can ask three yes-no questions, each to one god. The gods understand English, but will answer in their own language where 'da' means yes or no, and 'ja' means yes or no (you don't know which is which). What questions do you ask to determine which god is which?""",
        "gold_answer": "This is one of the hardest logic puzzles. A valid solution involves asking meta-questions about what gods would say, accounting for the language ambiguity and Random's unpredictability. One solution: 1) Ask A if B would say 'da' means yes (eliminates Random from A or B), 2) Ask identified non-Random if C is Random, 3) Ask identified Truth/False about their identity."
    })

    # 1.2 四色逻辑推理
    tasks.append({
        "id": "Hard_Logic_Four_Color_Hats",
        "difficulty": "hard",
        "category": "logic",
        "question": """Four prisoners are buried in the ground up to their necks in a line, all facing forward. The fourth prisoner (at the back) can see the three in front of him. The third can see the two in front of him. The second can see the one in front. The first (at the front) can see no one. Between the third and fourth prisoner is a brick wall that blocks their view. Each prisoner has a hat on his head, either red or blue. There are 2 red hats and 2 blue hats. No prisoner can see their own hat. To be freed, one prisoner must correctly call out the color of their own hat. If they are wrong, all will be executed. They cannot communicate with each other in any way once the hats are placed. However, before the hats are placed, they can discuss a strategy. Which prisoner can guarantee to save them all, and what is their strategy?""",
        "gold_answer": "The third prisoner can guarantee to save them all. Strategy: The fourth prisoner (behind the wall) can see no useful information. The third prisoner can see prisoners 1 and 2. If they are wearing the same color, the third prisoner calls out the opposite color (since there are only 2 of each color). If prisoners 1 and 2 are wearing different colors, the third prisoner can deduce their own hat color based on what they see and the constraint that there are 2 of each color."
    })

    # 1.3 日期推理问题 - Cheryl's Birthday
    tasks.append({
        "id": "Hard_Logic_Cheryls_Birthday",
        "difficulty": "hard",
        "category": "logic",
        "question": """Albert and Bernard just met Cheryl. "When is your birthday?" Albert asked Cheryl. Cheryl thought for a moment and said, "I won't tell you, but I'll give you some clues." She wrote down a list of 10 dates:
May 15, May 16, May 19
June 17, June 18
July 14, July 16
August 14, August 15, August 17

"My birthday is one of these," she said. Then Cheryl whispered in Albert's ear the month, and only the month. To Bernard, she whispered the day, and only the day.

Albert: "I don't know when your birthday is, but I know Bernard doesn't know either."
Bernard: "I didn't know originally, but now I do."
Albert: "Well, now I know too!"

When is Cheryl's birthday?""",
        "gold_answer": "July 16. Albert knows Bernard doesn't know, so the month cannot be May or June (which have unique days 18 and 19). Bernard then knows, so the day must be unique among the remaining dates (July 14, July 16, Aug 14, Aug 15, Aug 17). Days 14, 15, 17 appear multiple times, so it must be 16. Albert then knows the month is July."
    })

    # === 2. 复杂数学推理题 ===

    # 2.1 概率问题 - Monty Hall 变种
    tasks.append({
        "id": "Hard_Math_Monty_Hall_Extended",
        "difficulty": "hard",
        "category": "math",
        "question": """You're on a game show with 5 doors. Behind one door is a car, behind the other four are goats. You pick door #1. The host, who knows where the car is, opens 3 doors (not #1, not the car) revealing 3 goats. You now have a choice: stick with door #1, or switch to the other unopened door. What should you do, and what is your probability of winning if you follow the optimal strategy?""",
        "gold_answer": "You should switch. Initial probability of picking the car: 1/5. Probability the car is behind one of the other 4 doors: 4/5. When the host opens 3 doors with goats, the 4/5 probability concentrates on the remaining unopened door (the host cannot open the door with the car). So switching gives you 4/5 probability of winning, while staying gives you 1/5."
    })

    # 2.2 数论问题
    tasks.append({
        "id": "Hard_Math_Number_Theory",
        "difficulty": "hard",
        "category": "math",
        "question": """Find the smallest positive integer n such that n/2 is a perfect square, n/3 is a perfect cube, and n/5 is a perfect fifth power.""",
        "gold_answer": "n = 2^15 × 3^10 × 5^6 = 2,592,000,000,000,000. For n/2 to be a perfect square, n must have the form 2^(2a+1) × other. For n/3 to be a perfect cube, n must have the form 3^(3b+1) × other. For n/5 to be a perfect fifth power, n must have the form 5^(5c+1) × other. Taking minimum exponents: 2^15 × 3^10 × 5^6."
    })

    # 2.3 组合数学
    tasks.append({
        "id": "Hard_Math_Combinatorics",
        "difficulty": "hard",
        "category": "math",
        "question": """In how many ways can you tile a 2×n rectangle with 1×2 dominoes?""",
        "gold_answer": "F(n+1), where F is the Fibonacci sequence. Let a(n) be the number of ways to tile a 2×n rectangle. Either the first column has a vertical domino (leaving a 2×(n-1) rectangle, giving a(n-1) ways), or it has two horizontal dominoes (leaving a 2×(n-2) rectangle, giving a(n-2) ways). So a(n) = a(n-1) + a(n-2), with a(0)=1, a(1)=1. This is the Fibonacci sequence."
    })

    # === 3. 需要深度推理的问题 ===

    # 3.1 递归问题
    tasks.append({
        "id": "Hard_Reasoning_Towers_Of_Hanoi",
        "difficulty": "medium",
        "category": "reasoning",
        "question": """You have the Tower of Hanoi puzzle with 10 disks. What is the minimum number of moves required to solve it? And if you make one move per second, how long would it take?""",
        "gold_answer": "2^10 - 1 = 1,023 moves. At one move per second, it would take 1,023 seconds = 17 minutes and 3 seconds. The formula for n disks is 2^n - 1 moves."
    })

    # 3.2 策略博弈问题
    tasks.append({
        "id": "Hard_Reasoning_Game_Strategy",
        "difficulty": "hard",
        "category": "reasoning",
        "question": """Two players play a game with a pile of 100 stones. On each turn, a player must remove 1, 2, or 3 stones. The player who takes the last stone wins. You go first. What is your winning strategy?""",
        "gold_answer": "Take 3 stones on your first move (leaving 97). Then, whatever your opponent takes (1, 2, or 3), you take enough to make the total taken in that round equal to 4. This way you always leave a multiple of 4 stones for your opponent. Eventually you'll leave them with 4 stones, and whatever they take (1, 2, or 3), you can take the rest and win. Initial move: 100 mod 4 = 0, so take (4 - 0) mod 4 = 0... actually, 100 = 4×25, so you want to leave 96 stones (take 4). Then always respond to keep the remaining stones as a multiple of 4."
    })

    # 3.3 几何推理
    tasks.append({
        "id": "Hard_Reasoning_Geometry_Area",
        "difficulty": "medium",
        "category": "reasoning",
        "question": """A circle is inscribed in a square. Then a smaller square is inscribed in the circle. What is the ratio of the area of the larger square to the area of the smaller square?""",
        "gold_answer": "2:1. Let the large square have side length a. The inscribed circle has diameter a, so radius a/2. The smaller square inscribed in this circle has diagonal equal to the circle's diameter (a). If the small square has side length b, then b√2 = a, so b = a/√2. Area ratio = a² / (a/√2)² = a² / (a²/2) = 2."
    })

    # === 4. 反直觉问题 ===

    # 4.1 生日悖论变种
    tasks.append({
        "id": "Hard_Paradox_Birthday_Extended",
        "difficulty": "medium",
        "category": "probability",
        "question": """In a room of 30 people, what is the approximate probability that at least two people share the same birthday (ignoring leap years)? Is it closer to 30%, 50%, 70%, or 90%?""",
        "gold_answer": "Approximately 70% (70.6% to be exact). This is the famous birthday paradox. P(at least 2 share) = 1 - P(all different) = 1 - (365/365 × 364/365 × 363/365 × ... × 336/365) ≈ 0.706. The answer is closer to 70%."
    })

    # 4.2 Simpson's Paradox
    tasks.append({
        "id": "Hard_Paradox_Simpsons",
        "difficulty": "hard",
        "category": "statistics",
        "question": """Hospital A has a success rate of 90% for small surgeries (900 successes out of 1000) and 80% for large surgeries (800 successes out of 1000). Hospital B has a success rate of 85% for small surgeries (85 successes out of 100) and 95% for large surgeries (95 successes out of 100). Which hospital has the better overall success rate?""",
        "gold_answer": "Hospital A has a better overall success rate (1700/2000 = 85%) compared to Hospital B (180/200 = 90%). Wait, B is higher! This is Simpson's Paradox. Hospital B actually has a higher overall rate (90%) than Hospital A (85%), even though A has a better rate for both small and large surgeries individually. This is because B performs more of the easier small surgeries proportionally."
    })

    # 4.3 概率直觉陷阱
    tasks.append({
        "id": "Hard_Paradox_Two_Children",
        "difficulty": "hard",
        "category": "probability",
        "question": """Mr. Smith has two children. At least one of them is a boy. What is the probability that both children are boys?""",
        "gold_answer": "1/3. The possible combinations for two children are: BB, BG, GB, GG (where order matters: first/second child). We know 'at least one boy', so GG is eliminated. That leaves BB, BG, GB. Only 1 out of these 3 has both boys, so probability is 1/3. (Note: This assumes the information 'at least one is a boy' was obtained without reference to birth order. If we knew 'the eldest is a boy', the answer would be 1/2.)"
    })

    return tasks


def generate_report(all_results: List[Dict]):
    """生成详细对比报告"""
    print(f"\n\n{'='*60}")
    print("V2.5 vs Baseline 高难度题目测试结果")
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

    # 2. 按难度统计
    difficulties = {}
    for r in all_results:
        diff = r["difficulty"]
        if diff not in difficulties:
            difficulties[diff] = {
                "baseline_correct": 0,
                "baseline_total": 0,
                "v2_5_correct": 0,
                "v2_5_total": 0
            }

        difficulties[diff]["baseline_total"] += 1
        difficulties[diff]["v2_5_total"] += 1

        if r["baseline"]["correct"]:
            difficulties[diff]["baseline_correct"] += 1
        if r["v2_5"]["correct"]:
            difficulties[diff]["v2_5_correct"] += 1

    print("【按难度统计】")
    print(f"{'难度':<15} {'Baseline':<15} {'V2.5':<15} {'差距'}")
    print("-" * 60)

    for diff in ["medium", "hard", "very_hard"]:
        if diff in difficulties:
            stats = difficulties[diff]
            baseline_pct = (stats["baseline_correct"] / stats["baseline_total"] * 100) if stats["baseline_total"] > 0 else 0
            v2_5_pct = (stats["v2_5_correct"] / stats["v2_5_total"] * 100) if stats["v2_5_total"] > 0 else 0
            diff_pct = v2_5_pct - baseline_pct

            print(f"{diff:<15} {baseline_pct:>5.0f}% ({stats['baseline_correct']}/{stats['baseline_total']})      {v2_5_pct:>5.0f}% ({stats['v2_5_correct']}/{stats['v2_5_total']})      {diff_pct:+.0f}%")

    # 3. 详细结果
    print(f"\n【逐题结果】")
    print(f"{'题目ID':<35} {'难度':<12} {'Baseline':<10} {'V2.5':<10}")
    print("-" * 70)

    for r in all_results:
        baseline_result = "✅" if r["baseline"]["correct"] else ("⏱️" if r["baseline"]["timeout"] else "❌")
        v2_5_result = "✅" if r["v2_5"]["correct"] else ("⏱️" if r["v2_5"]["timeout"] else "❌")

        print(f"{r['task_id']:<35} {r['difficulty']:<12} {baseline_result:<10} {v2_5_result:<10}")

    # 4. 推理时间
    baseline_avg_time = sum(baseline_times) / len(baseline_times) if baseline_times else 0
    v2_5_avg_time = sum(v2_5_times) / len(v2_5_times) if v2_5_times else 0

    print(f"\n【推理时间】")
    print(f"Baseline: 平均 {baseline_avg_time:.1f}s/题")
    print(f"V2.5:     平均 {v2_5_avg_time:.1f}s/题")
    if baseline_avg_time > 0:
        print(f"时间比:    {v2_5_avg_time/baseline_avg_time:.1f}x")

    # 5. 超时统计
    baseline_timeouts = sum(1 for r in all_results if r["baseline"]["timeout"])
    v2_5_timeouts = sum(1 for r in all_results if r["v2_5"]["timeout"])

    if baseline_timeouts > 0 or v2_5_timeouts > 0:
        print(f"\n【超时统计】")
        print(f"Baseline: {baseline_timeouts} 题超时")
        print(f"V2.5:     {v2_5_timeouts} 题超时")

    # 6. 结论
    print(f"\n【结论】")
    if v2_5_acc > baseline_acc + 5:
        print(f"✅ V2.5 在高难度题目上显著优于 Baseline (+{v2_5_acc - baseline_acc:.1f}%)")
        print(f"💡 V2.5 的结构化推理在复杂问题上展现出明显优势")
    elif v2_5_acc > baseline_acc:
        print(f"✅ V2.5 略优于 Baseline (+{v2_5_acc - baseline_acc:.1f}%)")
    elif v2_5_acc == baseline_acc:
        print(f"⚖️ V2.5 与 Baseline 表现相当")
    else:
        print(f"⚠️ V2.5 表现不如 Baseline")

    if v2_5_avg_time > baseline_avg_time * 2:
        print(f"⚠️ 代价是推理时间增加约 {v2_5_avg_time/baseline_avg_time:.1f} 倍")

    print(f"{'='*60}\n")

    # 保存详细结果到文件
    output_file = "benchmark_hard_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"详细结果已保存到: {output_file}")


def main():
    """主函数 - 并行执行高难度测试"""
    print(f"{'='*60}")
    print("V2.5 vs Baseline 高难度题目测试")
    print(f"{'='*60}\n")

    # 1. 获取配置
    use_backend = os.getenv("USE_BACKEND", "ollama")

    if use_backend == "deepseek_api":
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        print(f"使用模型: {model_name} (DeepSeek API)")
    else:
        model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
        print(f"使用模型: {model_name} (Ollama)")

    # 2. 加载高难度测试题
    all_tasks = get_hard_questions()
    print(f"共加载 {len(all_tasks)} 个高难度测试任务")
    print(f"难度分布: {len([t for t in all_tasks if t['difficulty']=='medium'])} medium, "
          f"{len([t for t in all_tasks if t['difficulty']=='hard'])} hard, "
          f"{len([t for t in all_tasks if t['difficulty']=='very_hard'])} very_hard\n")

    # 3. 准备并行任务
    task_infos = []
    for idx, task in enumerate(all_tasks, 1):
        task_infos.append((idx, task, use_backend, model_name))

    # 4. 串行执行 (改为顺序执行，避免并发输出混乱)
    print(f"开始串行执行 (共 {len(task_infos)} 个任务)")
    print(f"每个任务超时限制: 5分钟\n")

    all_results = []

    # 直接循环执行，不使用并发
    for task_info in task_infos:
        try:
            result = run_single_task(task_info)
            all_results.append(result)
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
