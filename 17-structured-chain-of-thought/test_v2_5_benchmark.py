#!/usr/bin/env python3
"""
V2.5 快速基准测试（10条数据）

测试分布：
- 简单数学题：3条（验证 single_think）
- 中等推理题：4条（验证自动模式选择）
- 复杂推理题：3条（验证 structured_4stage）
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("DeepSeek-R1 Agent V2.5 - 快速基准测试")
print("="*80)

# 导入 V2 Agent
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

# 测试数据集（10条）
TEST_CASES = [
    # ===== 简单数学题（3条）=====
    {
        "id": 1,
        "difficulty": "simple",
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "expected_answer": "18",
        "category": "simple_math"
    },
    {
        "id": 2,
        "difficulty": "simple",
        "question": "If I have 10 apples and eat 3, then buy 5 more, how many apples do I have?",
        "expected_answer": "12",
        "category": "simple_math"
    },
    {
        "id": 3,
        "difficulty": "simple",
        "question": "A store has 48 bottles of water. They sell half of them in the morning. How many bottles are left?",
        "expected_answer": "24",
        "category": "simple_math"
    },

    # ===== 中等推理题（4条）=====
    {
        "id": 4,
        "difficulty": "medium",
        "question": "A parking lot has 12 spaces. 8 cars are parked. Then 3 cars leave and 5 new cars arrive. How many empty spaces are there now?",
        "expected_answer": "2",
        "category": "medium_logic"
    },
    {
        "id": 5,
        "difficulty": "medium",
        "question": "Tom has twice as many books as Jerry. Jerry has 15 books. If Tom gives Jerry 6 books, how many books does Tom have now?",
        "expected_answer": "24",
        "category": "medium_logic"
    },
    {
        "id": 6,
        "difficulty": "medium",
        "question": "A recipe needs 2 cups of flour for 12 cookies. How many cups of flour are needed for 30 cookies?",
        "expected_answer": "5",
        "category": "medium_math"
    },
    {
        "id": 7,
        "difficulty": "medium",
        "question": "Sarah saves $50 per month. After 6 months, she spends $180 on a gift. How much money does she have left?",
        "expected_answer": "120",
        "category": "medium_math"
    },

    # ===== 复杂推理题（3条）=====
    {
        "id": 8,
        "difficulty": "complex",
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "expected_answer": "3",
        "category": "complex_reasoning",
        "note": "V1 failed with hallucination (red fiber)"
    },
    {
        "id": 9,
        "difficulty": "complex",
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "expected_answer": "70000",
        "category": "complex_reasoning",
        "note": "V1 failed with loop and wrong answer ($26,000)"
    },
    {
        "id": 10,
        "difficulty": "complex",
        "question": "A company has 100 employees. First, they hire 20% more employees. Then, they reduce the workforce by 15%. How many employees are there now?",
        "expected_answer": "102",
        "category": "complex_percentage"
    }
]

# 创建 Agent
model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
print(f"\n初始化 Agent (模型: {model_name})...")
print("-"*80)

agent = DeepSeekR1AgentV2(
    model=model_name,
    enable_tools=True,
    enable_loop_detection=True,
    enable_hallucination_detection=False  # 关闭以加快速度
)

print("✅ Agent 初始化完成\n")

# 运行测试
results = []
total_time = 0
correct_count = 0

print("="*80)
print("开始测试")
print("="*80)

for i, test_case in enumerate(TEST_CASES, 1):
    print(f"\n{'='*80}")
    print(f"测试 {i}/10 [难度: {test_case['difficulty']}] [类别: {test_case['category']}]")
    print(f"{'='*80}")
    print(f"\n问题: {test_case['question']}")

    if 'note' in test_case:
        print(f"⚠️  注意: {test_case['note']}")

    print(f"\n预期答案: {test_case['expected_answer']}")
    print("-"*80)

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行 Agent（让它自动选择模式）
        answer = agent.run(test_case['question'], verbose=False)

        # 记录结束时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        # 检查答案是否正确（简单的字符串包含检查）
        is_correct = test_case['expected_answer'] in answer or test_case['expected_answer'].replace('.', '') in answer

        if is_correct:
            correct_count += 1
            result_icon = "✅"
        else:
            result_icon = "❌"

        print(f"\n{result_icon} Agent 答案: {answer[:200]}...")
        print(f"⏱️  耗时: {elapsed_time:.1f}秒")
        print(f"📊 正确性: {'正确' if is_correct else '错误'}")

        # 保存结果
        results.append({
            "id": test_case['id'],
            "difficulty": test_case['difficulty'],
            "category": test_case['category'],
            "question": test_case['question'],
            "expected_answer": test_case['expected_answer'],
            "agent_answer": answer,
            "is_correct": is_correct,
            "time_seconds": elapsed_time,
            "note": test_case.get('note', '')
        })

    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        results.append({
            "id": test_case['id'],
            "difficulty": test_case['difficulty'],
            "category": test_case['category'],
            "question": test_case['question'],
            "expected_answer": test_case['expected_answer'],
            "agent_answer": f"ERROR: {str(e)}",
            "is_correct": False,
            "time_seconds": 0,
            "note": test_case.get('note', '')
        })

# 输出统计结果
print("\n" + "="*80)
print("测试完成 - 统计结果")
print("="*80)

accuracy = (correct_count / len(TEST_CASES)) * 100
avg_time = total_time / len(TEST_CASES)

print(f"\n📊 总体表现:")
print(f"  - 准确率: {correct_count}/{len(TEST_CASES)} ({accuracy:.1f}%)")
print(f"  - 平均耗时: {avg_time:.1f}秒")
print(f"  - 总耗时: {total_time:.1f}秒")

# 按难度统计
print(f"\n📈 按难度分组:")
for difficulty in ["simple", "medium", "complex"]:
    diff_results = [r for r in results if r['difficulty'] == difficulty]
    if diff_results:
        diff_correct = sum(1 for r in diff_results if r['is_correct'])
        diff_total = len(diff_results)
        diff_accuracy = (diff_correct / diff_total) * 100
        diff_avg_time = sum(r['time_seconds'] for r in diff_results) / diff_total

        print(f"  {difficulty.capitalize()}: {diff_correct}/{diff_total} ({diff_accuracy:.1f}%) - 平均 {diff_avg_time:.1f}秒")

# 显示失败案例
failed_cases = [r for r in results if not r['is_correct']]
if failed_cases:
    print(f"\n❌ 失败案例 ({len(failed_cases)}个):")
    for case in failed_cases:
        print(f"\n  ID {case['id']} [{case['difficulty']}] {case['category']}")
        print(f"  问题: {case['question'][:80]}...")
        print(f"  预期: {case['expected_answer']}")
        print(f"  实际: {case['agent_answer'][:100]}...")

# 保存结果到文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = f"benchmark_v2_5_result_{timestamp}.json"

with open(result_file, 'w', encoding='utf-8') as f:
    json.dump({
        "timestamp": timestamp,
        "model": model_name,
        "total_cases": len(TEST_CASES),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "avg_time_seconds": avg_time,
        "total_time_seconds": total_time,
        "results": results
    }, f, ensure_ascii=False, indent=2)

print(f"\n💾 结果已保存到: {result_file}")

print("\n" + "="*80)
print("测试结束")
print("="*80)
