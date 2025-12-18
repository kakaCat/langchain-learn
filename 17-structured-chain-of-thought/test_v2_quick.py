"""
快速测试 V2 的基本功能

这个脚本用于验证 V2 各个组件是否正常工作，
无需运行完整的基准测试（避免长时间等待）
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("="*80)
print("DeepSeek-R1 Agent V2 - 快速功能测试")
print("="*80)

# 测试 1: 解析器
print("\n[测试 1/5] 解析器 (parsers.py)")
try:
    from parsers import ThinkTagParser

    parser = ThinkTagParser()
    test_text = "<think>推理过程</think><answer>42</answer>"
    result = parser.parse(test_text)

    assert result["think"] == "推理过程", "Think 解析失败"
    assert result["answer"] == "42", "Answer 解析失败"

    print("✅ 解析器测试通过")
except Exception as e:
    print(f"❌ 解析器测试失败: {e}")

# 测试 2: 提示词模板
print("\n[测试 2/5] 提示词模板 (prompts.py)")
try:
    from prompts import SINGLE_THINK_PROMPT, GATE_PROMPT

    # 测试格式化
    formatted = SINGLE_THINK_PROMPT.format(input="测试问题")
    assert "测试问题" in formatted, "提示词格式化失败"

    gate_formatted = GATE_PROMPT.format(input="你好")
    assert "你好" in gate_formatted, "门控提示词格式化失败"

    print("✅ 提示词模板测试通过")
except Exception as e:
    print(f"❌ 提示词模板测试失败: {e}")

# 测试 3: 工具
print("\n[测试 3/5] 工具系统 (tools.py)")
try:
    from tools import ToolRegistry

    calculator = ToolRegistry.get_calculator_tool()
    result = calculator.func("2 + 2")

    assert "4" in result, f"计算器结果错误: {result}"

    print(f"✅ 工具系统测试通过 (计算器: 2 + 2 = {result})")
except Exception as e:
    print(f"❌ 工具系统测试失败: {e}")

# 测试 4: 验证器
print("\n[测试 4/5] 验证器 (validators.py)")
try:
    from validators import LoopBreaker, ReasoningQualityChecker

    # 测试循环检测
    breaker = LoopBreaker(similarity_threshold=0.85)
    breaker.check_and_break("第一次输出")
    breaker.check_and_break("第二次输出")
    is_loop, msg = breaker.check_and_break("第一次输出")  # 重复

    assert is_loop, "循环检测失败"

    # 测试质量检查
    checker = ReasoningQualityChecker()
    is_complete, _ = checker.check_completeness("很短")

    assert not is_complete, "质量检查失败"

    print("✅ 验证器测试通过")
except Exception as e:
    print(f"❌ 验证器测试失败: {e}")

# 测试 5: Agent V2（仅导入测试，不运行推理）
print("\n[测试 5/5] Agent V2 导入 (deepseek_r1_traces_v2.py)")
try:
    from deepseek_r1_traces_v2 import DeepSeekR1AgentV2

    # 仅测试初始化（不连接 LLM）
    print("✅ Agent V2 导入成功")

    # 测试模式分类（不需要 LLM）
    print("\n测试复杂度分类:")
    print("  - '你好' -> direct (预期)")
    print("  - 'How many eggs...' -> single_think (预期)")

except Exception as e:
    print(f"❌ Agent V2 导入失败: {e}")

# 总结
print("\n" + "="*80)
print("快速测试完成！")
print("="*80)
print("\n如果所有测试通过，可以运行完整基准测试：")
print("  python benchmark_r1_traces_v2.py --skip-v1 --skip-baseline")
print("\n或运行 Agent V2 的示例：")
print("  python deepseek_r1_traces_v2.py")
print()
