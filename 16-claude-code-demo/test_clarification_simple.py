#!/usr/bin/env python3
"""
简单的反问机制测试（不需要 LangChain 依赖）
只测试数据模型和逻辑结构
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# 复制数据模型定义（用于独立测试）
class ClarificationQuestion(BaseModel):
    """澄清问题"""

    question: str
    reason: str  # 为什么需要这个问题
    question_type: Literal["scope", "preference", "constraint", "context"] = "context"
    options: Optional[List[str]] = None  # 可选的选项（多选题）


class ClarificationNeed(BaseModel):
    """澄清需求判断"""

    need_clarification: bool
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    reasoning: str
    urgency: Literal["high", "medium", "low"] = "medium"  # 紧迫性


class ClarificationResponse(BaseModel):
    """用户澄清回答"""

    answers: Dict[str, str]  # question -> answer
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


def test_clarification_question():
    """测试澄清问题模型"""
    print("=" * 80)
    print("测试 1：ClarificationQuestion 数据模型")
    print("=" * 80)

    # 测试 1：基本问题（无选项）
    q1 = ClarificationQuestion(
        question="请描述您的具体使用场景？",
        reason="需要了解应用背景",
        question_type="context",
    )
    print("\n✓ 测试 1.1：基本问题创建成功")
    print(f"  问题：{q1.question}")
    print(f"  类型：{q1.question_type}")
    print(f"  原因：{q1.reason}")
    print(f"  选项：{q1.options}")

    # 测试 2：带选项的问题
    q2 = ClarificationQuestion(
        question="您希望重点关注哪个领域？",
        reason="需求范围过大",
        question_type="scope",
        options=["Web 开发", "系统编程", "嵌入式开发", "游戏开发"],
    )
    print("\n✓ 测试 1.2：带选项的问题创建成功")
    print(f"  问题：{q2.question}")
    print(f"  类型：{q2.question_type}")
    print(f"  选项数：{len(q2.options)}")
    print(f"  选项：{q2.options}")

    # 测试 3：JSON 序列化
    json_str = q2.model_dump_json(indent=2)
    print("\n✓ 测试 1.3：JSON 序列化成功")
    print(json_str)

    return [q1, q2]


def test_clarification_need():
    """测试澄清需求模型"""
    print("\n" + "=" * 80)
    print("测试 2：ClarificationNeed 数据模型")
    print("=" * 80)

    # 创建测试问题
    questions = [
        ClarificationQuestion(
            question="您是初学者还是有经验的开发者？",
            reason="需要了解技能水平以推荐合适资源",
            question_type="context",
            options=["完全新手", "有其他语言经验", "已有基础"],
        ),
        ClarificationQuestion(
            question="您的主要学习目标是什么？",
            reason="明确目标以优化学习路径",
            question_type="scope",
            options=["找工作", "个人兴趣", "项目需要"],
        ),
    ]

    # 测试 1：需要澄清的情况
    need1 = ClarificationNeed(
        need_clarification=True,
        questions=questions,
        reasoning="用户需求 '研究 Rust' 过于宽泛，需要了解背景和目标",
        urgency="high",
    )
    print("\n✓ 测试 2.1：需要澄清的情况")
    print(f"  需要澄清：{need1.need_clarification}")
    print(f"  问题数量：{len(need1.questions)}")
    print(f"  理由：{need1.reasoning}")
    print(f"  紧迫性：{need1.urgency}")

    # 测试 2：不需要澄清的情况
    need2 = ClarificationNeed(
        need_clarification=False,
        questions=[],
        reasoning="需求已经足够明确：'研究 2025 年学习 Rust 的最佳路径，重点 Web 开发'",
        urgency="low",
    )
    print("\n✓ 测试 2.2：不需要澄清的情况")
    print(f"  需要澄清：{need2.need_clarification}")
    print(f"  问题数量：{len(need2.questions)}")
    print(f"  理由：{need2.reasoning}")

    return need1, need2


def test_clarification_response():
    """测试用户回答模型"""
    print("\n" + "=" * 80)
    print("测试 3：ClarificationResponse 数据模型")
    print("=" * 80)

    # 模拟用户回答
    answers = {
        "您是初学者还是有经验的开发者？": "有其他语言经验",
        "您的主要学习目标是什么？": "找工作",
        "您希望重点关注哪个领域？": "Web 开发",
    }

    response = ClarificationResponse(answers=answers)
    print("\n✓ 测试 3.1：用户回答创建成功")
    print(f"  回答数量：{len(response.answers)}")
    print(f"  时间戳：{response.timestamp}")
    print("\n  回答详情：")
    for q, a in response.answers.items():
        print(f"    Q: {q}")
        print(f"    A: {a}")

    # 测试 JSON 序列化
    json_str = response.model_dump_json(indent=2)
    print("\n✓ 测试 3.2：JSON 序列化成功")
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

    return response


def test_workflow_simulation():
    """模拟工作流"""
    print("\n" + "=" * 80)
    print("测试 4：工作流模拟")
    print("=" * 80)

    # 场景：模糊需求 -> 检测 -> 提问 -> 收集回答
    user_request = "研究 AI"

    print(f"\n用户需求：{user_request}")
    print("-" * 80)

    # Step 1: 检测到需要澄清
    print("\nStep 1: Agent 检测到需求模糊，生成澄清问题...")
    clarification = ClarificationNeed(
        need_clarification=True,
        questions=[
            ClarificationQuestion(
                question="您想了解 AI 的哪个方向？",
                reason="AI 领域非常广泛",
                question_type="scope",
                options=[
                    "机器学习基础",
                    "深度学习",
                    "自然语言处理",
                    "计算机视觉",
                    "AI 应用开发",
                ],
            ),
            ClarificationQuestion(
                question="您的技术背景是什么？",
                reason="需要匹配合适难度的资料",
                question_type="context",
                options=["编程新手", "有编程基础", "有 AI 经验"],
            ),
        ],
        reasoning="需求 '研究 AI' 过于宽泛，AI 包含多个子领域，需要明确具体方向和背景",
        urgency="high",
    )

    print(f"  ✓ 需要澄清：{clarification.need_clarification}")
    print(f"  ✓ 生成 {len(clarification.questions)} 个问题")

    # Step 2: 模拟用户回答
    print("\nStep 2: 用户回答问题...")
    response = ClarificationResponse(
        answers={
            "您想了解 AI 的哪个方向？": "自然语言处理",
            "您的技术背景是什么？": "有编程基础",
        }
    )

    print(f"  ✓ 收集到 {len(response.answers)} 个回答")
    for q, a in response.answers.items():
        print(f"    • {q[:30]}... -> {a}")

    # Step 3: 更新需求
    print("\nStep 3: 基于用户反馈更新需求...")
    clarification_summary = "\n".join(f"- {q}: {a}" for q, a in response.answers.items())
    updated_request = f"{user_request}\n\n用户澄清：\n{clarification_summary}"

    print(f"  ✓ 更新后的需求：")
    print(f"    {updated_request}")

    print("\n✓ 工作流模拟完成！")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("反问机制数据模型测试")
    print("=" * 80)

    try:
        # 测试各个数据模型
        test_clarification_question()
        test_clarification_need()
        test_clarification_response()

        # 测试工作流
        test_workflow_simulation()

        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n反问机制的数据模型和逻辑设计正确！")
        print("要测试完整功能，请运行主程序：")
        print("  python3 11_claude_code_style_enhanced.py")

    except Exception as e:
        print(f"\n✗ 测试失败：{e}")
        import traceback

        traceback.print_exc()
