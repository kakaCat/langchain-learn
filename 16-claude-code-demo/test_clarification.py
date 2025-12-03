#!/usr/bin/env python3
"""
测试反问机制的脚本
"""

import sys
import os

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

# 加载环境变量
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 导入主程序（使用 importlib 处理数字开头的模块名）
import importlib.util

spec = importlib.util.spec_from_file_location(
    "enhanced_module",
    os.path.join(os.path.dirname(__file__), "11_claude_code_style_enhanced.py"),
)
enhanced_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enhanced_module)

# 导入需要的类和函数
ClaudeCodeState = enhanced_module.ClaudeCodeState
detect_clarification_need_node = enhanced_module.detect_clarification_need_node
ClarificationQuestion = enhanced_module.ClarificationQuestion
ClarificationNeed = enhanced_module.ClarificationNeed


def test_clarification_detection():
    """测试澄清检测节点"""
    print("=" * 80)
    print("测试 1：澄清检测节点")
    print("=" * 80)

    # 测试模糊需求
    test_cases = [
        "研究 AI",  # 非常模糊
        "研究 2025 年学习 Rust 的最佳路径，重点关注 Web 开发方向，我是初学者",  # 非常明确
        "帮我分析一下这个技术",  # 模糊
    ]

    for i, request in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}：{request}")
        print("-" * 80)

        state = ClaudeCodeState(
            user_request=request,
            enable_clarification=True,
        )

        try:
            result_state = detect_clarification_need_node(state)

            if result_state.clarification_need:
                need = result_state.clarification_need
                print(f"✓ 检测结果：需要澄清 = {need.need_clarification}")
                print(f"  理由：{need.reasoning}")
                print(f"  紧迫性：{need.urgency}")

                if need.questions:
                    print(f"  问题数：{len(need.questions)}")
                    for j, q in enumerate(need.questions, 1):
                        print(f"\n  问题 {j}:")
                        print(f"    - {q.question}")
                        print(f"    - 类型：{q.question_type}")
                        print(f"    - 原因：{q.reason}")
                        if q.options:
                            print(f"    - 选项：{q.options}")
            else:
                print("✗ 未生成澄清检测结果")

        except Exception as e:
            print(f"✗ 测试失败：{e}")
            import traceback

            traceback.print_exc()


def test_data_models():
    """测试数据模型"""
    print("\n" + "=" * 80)
    print("测试 2：数据模型验证")
    print("=" * 80)

    try:
        # 测试创建澄清问题
        question = ClarificationQuestion(
            question="您希望重点关注哪个方面？",
            reason="需求过于宽泛",
            question_type="scope",
            options=["基础入门", "进阶实战", "最佳实践"],
        )
        print("✓ ClarificationQuestion 创建成功")
        print(f"  {question.model_dump_json(indent=2)}")

        # 测试创建澄清需求
        need = ClarificationNeed(
            need_clarification=True,
            questions=[question],
            reasoning="测试理由",
            urgency="medium",
        )
        print("\n✓ ClarificationNeed 创建成功")
        print(f"  需要澄清：{need.need_clarification}")
        print(f"  问题数：{len(need.questions)}")

    except Exception as e:
        print(f"✗ 数据模型测试失败：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("\n开始测试反问机制...\n")

    # 测试数据模型
    test_data_models()

    # 测试澄清检测（需要 LLM）
    if os.getenv("OPENAI_API_KEY") or os.getenv("OLLAMA_BASE_URL"):
        print("\n检测到 LLM 配置，运行实际检测测试...")
        test_clarification_detection()
    else:
        print(
            "\n⚠ 未配置 LLM（OPENAI_API_KEY 或 OLLAMA_BASE_URL），跳过实际检测测试"
        )

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
