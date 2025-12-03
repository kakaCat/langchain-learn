#!/usr/bin/env python3
"""
13 - 多轮澄清对话 (Multi-Round Clarification)

演示如何实现多轮对话式澄清：
1. 支持多轮迭代澄清
2. 基于上一轮回答生成下一轮问题
3. 动态调整澄清策略
4. 自动判断何时停止澄清

技术特点：
- Iterative Clarification（迭代式澄清）
- Context-Aware Question Generation（上下文感知问题生成）
- Adaptive Questioning Strategy（自适应提问策略）
- Automatic Stopping Criterion（自动停止条件）

适用场景：
- 复杂需求分析
- 个性化咨询服务
- 智能客服对话
- 医疗诊断助手
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False
    )


def get_llm(model: Optional[str] = None, temperature: float = 0.3) -> object:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=1500,
        )


# ============================================================================
# 数据模型
# ============================================================================


class Question(BaseModel):
    """单个问题"""

    question: str
    reason: str
    question_type: Literal["scope", "preference", "constraint", "context"] = "context"
    options: Optional[List[str]] = None


class ClarificationRound(BaseModel):
    """单轮澄清"""

    round_number: int
    questions: List[Question]
    answers: Dict[str, str] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ContinueAssessment(BaseModel):
    """继续澄清的评估"""

    should_continue: bool
    reason: str
    completeness: float = Field(ge=0.0, le=1.0)  # 需求完整度 0-1


class MultiRoundState(BaseModel):
    """多轮对话状态"""

    user_request: str
    clarification_rounds: List[ClarificationRound] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = Field(default=3)
    should_continue_clarification: bool = True
    requirement_summary: Optional[str] = None
    execution_plan: List[str] = Field(default_factory=list)
    final_output: Optional[str] = None


# ============================================================================
# 辅助函数
# ============================================================================


def parse_json_safely(
    llm, prompt: str, target_model: type[BaseModel]
) -> Optional[BaseModel]:
    """安全解析 JSON"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        return target_model(**data)
    except Exception as e:
        print(f"[警告] JSON 解析失败：{e}")
        return None


# ============================================================================
# 节点实现
# ============================================================================


def generate_questions_node(state: MultiRoundState) -> MultiRoundState:
    """生成本轮问题"""
    if state.current_round >= state.max_rounds:
        state.should_continue_clarification = False
        return state

    llm = get_llm()

    # 收集之前的对话历史
    history = []
    for r in state.clarification_rounds:
        for q, a in r.answers.items():
            if a != "（跳过）":
                history.append(f"Q{r.round_number}: {q}\nA{r.round_number}: {a}")

    history_text = "\n\n".join(history) if history else "（无历史对话）"

    prompt = f"""
用户原始需求：{state.user_request}

已有对话历史：
{history_text}

当前是第 {state.current_round + 1} 轮澄清（共最多 {state.max_rounds} 轮）。

请基于当前信息生成 1-2 个问题。要求：
1. 基于已有回答深入挖掘
2. 避免重复已问过的问题
3. 聚焦最关键的信息缺口

输出 JSON：

{{
  "questions": [
    {{
      "question": "具体问题",
      "reason": "为什么问这个",
      "question_type": "scope|preference|constraint|context",
      "options": ["选项1", "选项2"]
    }}
  ]
}}

如果信息已经足够，返回空数组 {{"questions": []}}
"""

    result = parse_json_safely(llm, prompt, type("TempModel", (BaseModel,), {
        "__annotations__": {"questions": List[Question]},
        "questions": Field(default_factory=list)
    }))

    questions = result.questions if result else []

    if not questions:
        print(f"\n[第{state.current_round + 1}轮] 无需进一步澄清")
        state.should_continue_clarification = False
        return state

    # 创建新一轮
    new_round = ClarificationRound(
        round_number=state.current_round + 1, questions=questions
    )
    state.clarification_rounds.append(new_round)

    print(f"\n[第{state.current_round + 1}轮] 生成 {len(questions)} 个问题")

    return state


def ask_questions_node(state: MultiRoundState) -> MultiRoundState:
    """向用户提问"""
    if not state.clarification_rounds:
        return state

    current_round = state.clarification_rounds[-1]

    print("\n" + "=" * 80)
    print(f"💬 第 {current_round.round_number} 轮澄清")
    print("=" * 80)

    answers = {}
    for i, q in enumerate(current_round.questions, 1):
        print(f"\n问题 {i} [{q.question_type}]:")
        print(f"  {q.question}")
        print(f"  💡 {q.reason}")

        if q.options:
            print(f"  选项：")
            for j, opt in enumerate(q.options, 1):
                print(f"    {j}. {opt}")
            user_input = input(f"\n  您的选择（1-{len(q.options)} 或自定义）[回车跳过]: ").strip()

            if user_input.isdigit() and 1 <= int(user_input) <= len(q.options):
                answers[q.question] = q.options[int(user_input) - 1]
            elif user_input:
                answers[q.question] = user_input
            else:
                answers[q.question] = "（跳过）"
        else:
            user_input = input("  您的回答 [回车跳过]: ").strip()
            answers[q.question] = user_input if user_input else "（跳过）"

    current_round.answers = answers
    state.current_round += 1

    print(f"\n✓ 第 {current_round.round_number} 轮完成")
    return state


def assess_continuation_node(state: MultiRoundState) -> MultiRoundState:
    """评估是否继续澄清"""
    if state.current_round >= state.max_rounds:
        print(f"\n[评估] 达到最大轮数 ({state.max_rounds})，停止澄清")
        state.should_continue_clarification = False
        return state

    llm = get_llm()

    # 收集所有对话
    all_qa = []
    for r in state.clarification_rounds:
        for q, a in r.answers.items():
            if a != "（跳过）":
                all_qa.append(f"- {q} → {a}")

    qa_text = "\n".join(all_qa) if all_qa else "（无有效回答）"

    prompt = f"""
原始需求：{state.user_request}

已澄清信息：
{qa_text}

当前轮次：{state.current_round}/{state.max_rounds}

请评估是否需要继续澄清。输出 JSON：

{{
  "should_continue": false,
  "reason": "评估理由",
  "completeness": 0.85
}}

评估标准：
- completeness >= 0.80 时建议停止
- 关键信息已收集完毕建议停止
- 用户频繁跳过问题建议停止
"""

    result = parse_json_safely(llm, prompt, ContinueAssessment)

    if not result:
        # 回退
        result = ContinueAssessment(
            should_continue=state.current_round < state.max_rounds,
            reason="默认策略",
            completeness=0.5,
        )

    state.should_continue_clarification = result.should_continue

    print(f"\n[评估] 完整度：{result.completeness:.0%}")
    print(f"  {result.reason}")
    print(f"  继续澄清：{result.should_continue}")

    return state


def summarize_requirements_node(state: MultiRoundState) -> MultiRoundState:
    """总结需求"""
    llm = get_llm()

    # 收集所有澄清信息
    all_info = []
    for r in state.clarification_rounds:
        for q, a in r.answers.items():
            if a != "（跳过）":
                all_info.append(f"- {q} → {a}")

    clarifications = "\n".join(all_info) if all_info else "（无澄清信息）"

    prompt = f"""
原始需求：{state.user_request}

澄清信息：
{clarifications}

请整合所有信息，生成完整的需求描述（2-3 段）。
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state.requirement_summary = response.content.strip()

    print("\n" + "=" * 80)
    print("📋 需求总结")
    print("=" * 80)
    print(state.requirement_summary)

    return state


def plan_and_execute_node(state: MultiRoundState) -> MultiRoundState:
    """制定计划并执行"""
    llm = get_llm()

    requirement = state.requirement_summary or state.user_request

    # 制定计划
    plan_prompt = f"""
需求：{requirement}

请制定执行计划（3-5 步）。
只输出 JSON 数组：["步骤1", "步骤2", ...]
"""

    response = llm.invoke([HumanMessage(content=plan_prompt)])
    content = response.content.strip()

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        steps = json.loads(content)
        state.execution_plan = [str(s) for s in steps]
    except:
        state.execution_plan = [requirement]

    print(f"\n[计划] {len(state.execution_plan)} 个步骤")
    for i, step in enumerate(state.execution_plan, 1):
        print(f"  {i}. {step}")

    # 执行计划（简化版）
    exec_prompt = f"""
需求：{requirement}

计划：{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(state.execution_plan))}

请总结执行结果（3-4 段）。
"""

    final_response = llm.invoke([HumanMessage(content=exec_prompt)])
    state.final_output = final_response.content.strip()

    print("\n" + "=" * 80)
    print("✅ 执行结果")
    print("=" * 80)
    print(state.final_output)

    return state


# ============================================================================
# 工作流构建
# ============================================================================


def create_multi_round_workflow():
    """创建多轮澄清工作流"""
    graph = StateGraph(MultiRoundState)

    # 添加节点
    graph.add_node("generate", generate_questions_node)
    graph.add_node("ask", ask_questions_node)
    graph.add_node("assess", assess_continuation_node)
    graph.add_node("summarize", summarize_requirements_node)
    graph.add_node("execute", plan_and_execute_node)

    # 定义流程
    graph.set_entry_point("generate")

    # 条件分支：如果生成了问题，则询问用户
    graph.add_conditional_edges(
        "generate",
        lambda s: "ask" if s.clarification_rounds and s.clarification_rounds[-1].questions else "summarize",
        {"ask": "ask", "summarize": "summarize"},
    )

    graph.add_edge("ask", "assess")

    # 条件分支：是否继续澄清
    graph.add_conditional_edges(
        "assess",
        lambda s: "generate" if s.should_continue_clarification else "summarize",
        {"generate": "generate", "summarize": "summarize"},
    )

    graph.add_edge("summarize", "execute")
    graph.add_edge("execute", END)

    return graph.compile()


# ============================================================================
# 主函数
# ============================================================================


def run_demo():
    """运行演示"""
    load_environment()
    workflow = create_multi_round_workflow()

    print("\n" + "=" * 80)
    print("多轮澄清对话 Agent 演示")
    print("=" * 80)
    print("\n提示：Agent 会进行最多 3 轮澄清，基于您的回答逐步细化需求。\n")

    user_input = input("请输入您的需求（回车使用默认）: ").strip()
    if not user_input:
        user_input = "帮我设计一个学习计划"
        print(f"使用默认需求：{user_input}")

    state = MultiRoundState(user_request=user_input, max_rounds=3)

    try:
        final_state = workflow.invoke(state)

        print("\n" + "=" * 80)
        print("📊 对话统计")
        print("=" * 80)
        print(f"总轮数：{len(final_state.clarification_rounds)}")

        for r in final_state.clarification_rounds:
            valid_answers = sum(1 for a in r.answers.values() if a != "（跳过）")
            print(
                f"\n第 {r.round_number} 轮："
                f"\n  问题数：{len(r.questions)}"
                f"\n  有效回答：{valid_answers}/{len(r.answers)}"
            )

    except Exception as e:
        print(f"\n执行出错：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
