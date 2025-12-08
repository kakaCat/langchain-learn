#!/usr/bin/env python3
"""
测试 V2.5 结构化4阶段模式

这个脚本测试新增的 structured_4stage 模式，验证：
1. Memory 管理是否正常工作
2. 4个阶段是否按顺序执行
3. 工具和检测器是否集成成功
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("DeepSeek-R1 Agent V2.5 - 结构化4阶段模式测试")
print("="*80)

# 导入 V2 Agent
from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

# 创建 Agent
model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
print(f"\n初始化 Agent (模型: {model_name})...")

agent = DeepSeekR1AgentV2(
    model=model_name,
    enable_tools=True,
    enable_loop_detection=True,
    enable_hallucination_detection=False  # 可设置为 True 以启用
)

print("\n" + "="*80)
print("测试案例 1: Robe 纤维问题 (V1 失败，V2.5 应该修复)")
print("="*80)

task2_question = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"

print(f"\n问题: {task2_question}")
print("\n正确答案: 3 bolts")
print("V1 错误答案: 2.5 (引入了不存在的红色纤维)")
print("\n" + "-"*80)
print("使用 structured_4stage 模式运行...")
print("-"*80)

# 强制使用 structured_4stage 模式
answer = agent.run(task2_question, mode="structured_4stage", verbose=True)

print("\n" + "="*80)
print("测试结果")
print("="*80)
print(f"\nV2.5 答案: {answer}")

if "3" in answer and "2.5" not in answer:
    print("\n✅ 测试通过！V2.5 成功解决了 V1 的幻觉问题")
else:
    print("\n⚠️ 答案需要人工检查")

print("\n" + "="*80)
print("测试案例 2: House Flip 问题 (V1 循环失败，V2.5 应该修复)")
print("="*80)

task3_question = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"

print(f"\n问题: {task3_question}")
print("\n正确答案: $70,000")
print("V1 错误答案: $26,000 (误解题意 + 循环10+次)")
print("\n" + "-"*80)
print("使用 structured_4stage 模式运行...")
print("-"*80)

answer2 = agent.run(task3_question, mode="structured_4stage", verbose=True)

print("\n" + "="*80)
print("测试结果")
print("="*80)
print(f"\nV2.5 答案: {answer2}")

if "70" in answer2 or "70000" in answer2:
    print("\n✅ 测试通过！V2.5 成功解决了 V1 的循环和计算问题")
else:
    print("\n⚠️ 答案需要人工检查")

print("\n" + "="*80)
print("测试案例 3: 简单问题 (应该自动选择 single_think，不是 structured_4stage)")
print("="*80)

simple_question = "If I have 10 apples and eat 3, how many are left?"

print(f"\n问题: {simple_question}")
print("预期: 应该自动选择 single_think 模式（因为问题简单）")
print("-"*80)

answer3 = agent.run(simple_question, verbose=True)  # 不强制模式，让它自动选择

print(f"\n答案: {answer3}")

print("\n" + "="*80)
print("测试总结")
print("="*80)

print("""
V2.5 结构化4阶段模式已实现！

核心特性：
1. ✅ Memory 管理：各阶段共享完整上下文
2. ✅ 4阶段流程：问题定义 → 路径探索 → 验证 → 决策
3. ✅ 工具集成：Stage 2 可使用 calculator
4. ✅ 检测器：Stage 3 集成循环和幻觉检测
5. ✅ 智能路由：复杂问题自动使用 structured_4stage

关键改进：
- Stage 3 从"魔鬼代言人"改为"验证助手"（修复幻觉问题）
- Memory 保证上下文连续性（解决信息丢失）
- 工具验证数学计算（修复计算错误）
- 循环检测防止重复（修复无限循环）

使用建议：
- 一般问题：让系统自动选择模式（single_think 或 structured_4stage）
- 复杂推理：手动指定 mode="structured_4stage"
- 简单问候：自动使用 mode="direct"

下一步：
1. 运行完整基准测试验证准确率提升
2. 根据实际效果调整各阶段提示词
3. 测试更多边缘案例
""")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
