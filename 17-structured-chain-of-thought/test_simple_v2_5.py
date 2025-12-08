#!/usr/bin/env python3
"""
V2.5 简单测试 - 直接测试 3 个案例
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("DeepSeek-R1 Agent V2.5 - 简单快速测试")
print("="*80)

from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

# 创建 Agent
model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
print(f"\n初始化 Agent (模型: {model_name})...")

agent = DeepSeekR1AgentV2(
    model=model_name,
    enable_tools=False,  # 暂时禁用工具以简化测试
    enable_loop_detection=True,
    enable_hallucination_detection=False
)

print("✅ Agent 初始化完成\n")

# 测试 1: 简单数学题
print("\n" + "="*80)
print("测试 1: 简单数学题 (Janet's eggs)")
print("="*80)

question1 = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

print(f"\n问题: {question1}")
print("\n预期答案: 18")
print("-"*80)

start = time.time()
answer1 = agent.run(question1, verbose=False)
elapsed1 = time.time() - start

print(f"\nAgent 答案: {answer1}")
print(f"耗时: {elapsed1:.1f}秒")
print(f"正确性: {'✅ 正确' if '18' in answer1 else '❌ 错误'}")

# 测试 2: 复杂推理题 (Robe fibers - V1 失败案例)
print("\n" + "="*80)
print("测试 2: 复杂推理题 (Robe fibers - V1 失败)")
print("="*80)

question2 = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"

print(f"\n问题: {question2}")
print("\n预期答案: 3")
print("⚠️  V1 错误: 2.5 (引入了不存在的红色纤维)")
print("-"*80)

# 强制使用 structured_4stage 模式
start = time.time()
answer2 = agent.run(question2, mode="structured_4stage", verbose=True)
elapsed2 = time.time() - start

print(f"\nAgent 答案: {answer2}")
print(f"耗时: {elapsed2:.1f}秒")
print(f"正确性: {'✅ 正确' if '3' in answer2 and '2.5' not in answer2 else '❌ 错误'}")

# 测试 3: House Flip (V1 循环失败案例)
print("\n" + "="*80)
print("测试 3: House Flip (V1 循环失败)")
print("="*80)

question3 = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"

print(f"\n问题: {question3}")
print("\n预期答案: $70,000")
print("⚠️  V1 错误: $26,000 (误解题意 + 循环10+次)")
print("-"*80)

# 强制使用 structured_4stage 模式
start = time.time()
answer3 = agent.run(question3, mode="structured_4stage", verbose=True)
elapsed3 = time.time() - start

print(f"\nAgent 答案: {answer3}")
print(f"耗时: {elapsed3:.1f}秒")
print(f"正确性: {'✅ 正确' if '70' in answer3 or '70000' in answer3 else '❌ 错误'}")

# 总结
print("\n" + "="*80)
print("测试总结")
print("="*80)

total_time = elapsed1 + elapsed2 + elapsed3
avg_time = total_time / 3

print(f"\n总耗时: {total_time:.1f}秒")
print(f"平均耗时: {avg_time:.1f}秒/题")

print("\n测试完成！")
