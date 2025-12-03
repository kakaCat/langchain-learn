#!/usr/bin/env python3
"""
12 - 智能澄清 Agent (Intelligent Clarification Agent)

演示如何构建主动提问的智能 Agent：
1. 自动检测需求是否需要澄清
2. 生成结构化的澄清问题
3. 基于用户反馈调整执行策略
4. 支持问题类型分类和紧迫性评估

技术特点：
- Proactive Questioning（主动提问）
- Structured Question Generation（结构化问题生成）
- Context-Aware Clarification（上下文感知澄清）
- Adaptive Execution（自适应执行）

适用场景：
- 需求模糊的任务
- 多领域交叉主题
- 个性化推荐系统
- 智能客服和助手
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False
    )


def get_llm(model: Optional[str] = None, temperature: float = 0.2) -> object:
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


class ClarificationQuestion(BaseModel):
    """澄清问题"""

    question: str
    reason: str
    question_type: Literal["scope", "preference", "constraint", "context"] = "context"
    options: Optional[List[str]] = None


class ClarificationNeed(BaseModel):
    """澄清需求"""

    need_clarification: bool
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    reasoning: str
    urgency: Literal["high", "medium", "low"] = "medium"


class ClarificationResponse(BaseModel):
    """用户回答"""

    answers: Dict[str, str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AgentState(BaseModel):
    """Agent 状态"""

    user_request: str
    clarification_need: Optional[ClarificationNeed] = None
    clarification_responses: List[ClarificationResponse] = Field(default_factory=list)
    refined_request: Optional[str] = None
    execution_plan: List[str] = Field(default_factory=list)
    results: List[str] = Field(default_factory=list)
    final_output: Optional[str] = None


# ============================================================================
# 辅助函数
# ============================================================================


def parse_json_safely(
    llm, prompt: str, target_model: type[BaseModel], max_retries: int = 2
) -> Optional[BaseModel]:
    """安全解析 JSON"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # 提取 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return target_model(**data)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                prompt += f"\n\n解析失败：{e}\n请严格按照 JSON 格式输出，不要添加任何其他文字。"
            else:
                print(f"[警告] JSON 解析失败：{e}")
                return None

    return None


# ============================================================================
# 节点实现
# ============================================================================


def detect_clarification_node(state: AgentState) -> AgentState:
    """检测是否需要澄清"""
    if state.clarification_responses:
        # 已经澄清过，跳过
        return state

    llm = get_llm()

    prompt = f"""
你是一个智能助手，正在分析用户需求。

用户需求：{state.user_request}

请判断这个需求是否需要澄清。考虑：
1. 需求是否模糊或有多种理解？
2. 是否缺少关键信息？
3. 是否有技术选型等决策点？

输出 JSON：

{{
  "need_clarification": true,
  "questions": [
    {{
      "question": "您希望重点关注哪个方面？",
      "reason": "需求过于宽泛",
      "question_type": "scope",
      "options": ["选项1", "选项2"]
    }}
  ],
  "reasoning": "为什么需要澄清",
  "urgency": "high"
}}

question_type: scope | preference | constraint | context
urgency: high | medium | low

注意：
- 只在真正需要时设置 need_clarification=true
- 问题数量 1-3 个
- 如果需求已经清晰，返回 need_clarification=false
"""

    result = parse_json_safely(llm, prompt, ClarificationNeed)

    if not result:
        # 回退
        result = ClarificationNeed(
            need_clarification=False, questions=[], reasoning="解析失败，默认不需要澄清"
        )

    state.clarification_need = result

    if result.need_clarification:
        print(f"\n[检测] 需要澄清：{result.reasoning}")
        print(f"  问题数：{len(result.questions)}, 紧迫性：{result.urgency}")
    else:
        print(f"\n[检测] 需求明确：{result.reasoning}")

    return state


def ask_user_node(state: AgentState) -> AgentState:
    """向用户提问"""
    if not (state.clarification_need and state.clarification_need.need_clarification):
        return state

    if state.clarification_responses:
        # 已经问过了
        return state

    clarification = state.clarification_need

    print("\n" + "=" * 80)
    print("🤔 Agent 需要您的帮助")
    print("=" * 80)
    print(f"\n原因：{clarification.reasoning}")
    print(f"紧迫性：{clarification.urgency.upper()}\n")

    answers = {}
    for i, q in enumerate(clarification.questions, 1):
        print(f"\n问题 {i}/{len(clarification.questions)} [{q.question_type}]:")
        print(f"  {q.question}")
        print(f"  → {q.reason}")

        if q.options:
            print(f"  可选项：")
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

    response = ClarificationResponse(answers=answers)
    state.clarification_responses.append(response)

    print("\n✓ 感谢您的反馈！")
    return state


def refine_request_node(state: AgentState) -> AgentState:
    """基于澄清结果精炼需求"""
    if not state.clarification_responses:
        state.refined_request = state.user_request
        return state

    # 整合澄清信息
    clarifications = []
    for resp in state.clarification_responses:
        for q, a in resp.answers.items():
            if a != "（跳过）":
                clarifications.append(f"- {q} -> {a}")

    if clarifications:
        clarification_text = "\n".join(clarifications)
        state.refined_request = (
            f"{state.user_request}\n\n用户澄清：\n{clarification_text}"
        )
        print(f"\n[精炼] 更新后的需求：")
        print(state.refined_request)
    else:
        state.refined_request = state.user_request

    return state


def plan_execution_node(state: AgentState) -> AgentState:
    """制定执行计划"""
    llm = get_llm()

    request = state.refined_request or state.user_request

    prompt = f"""
需求：{request}

请制定执行计划，分解为 3-5 个步骤。
只输出 JSON 数组：

["步骤1描述", "步骤2描述", ...]

注意：基于用户的澄清信息来优化计划。
"""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    content = response.content.strip()

    # 提取 JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        steps = json.loads(content)
        state.execution_plan = [str(s) for s in steps]
    except:
        # 回退：按行分割
        state.execution_plan = [
            line.strip() for line in content.split("\n") if line.strip()
        ][:5]

    print(f"\n[规划] 执行计划：")
    for i, step in enumerate(state.execution_plan, 1):
        print(f"  {i}. {step}")

    return state


def execute_plan_node(state: AgentState) -> AgentState:
    """执行计划"""
    llm = get_llm()

    request = state.refined_request or state.user_request

    for i, step in enumerate(state.execution_plan, 1):
        prompt = f"""
总体需求：{request}
当前步骤：{step}

请执行这个步骤，给出结果（2-3 句话）。
"""

        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()

        state.results.append(f"步骤{i}: {result}")
        print(f"\n[执行] 步骤 {i}: {result[:100]}...")

    return state


def finalize_node(state: AgentState) -> AgentState:
    """生成最终输出"""
    llm = get_llm()

    results_text = "\n".join(state.results)
    request = state.refined_request or state.user_request

    prompt = f"""
用户需求：{request}

执行结果：
{results_text}

请生成最终报告（2-3 段）。
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state.final_output = response.content.strip()

    print("\n" + "=" * 80)
    print("✓ 最终报告")
    print("=" * 80)
    print(state.final_output)

    return state


# ============================================================================
# 工作流构建
# ============================================================================


def create_clarification_workflow():
    """创建澄清工作流"""
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("detect", detect_clarification_node)
    graph.add_node("ask", ask_user_node)
    graph.add_node("refine", refine_request_node)
    graph.add_node("plan", plan_execution_node)
    graph.add_node("execute", execute_plan_node)
    graph.add_node("finalize", finalize_node)

    # 定义流程
    graph.set_entry_point("detect")

    # 条件分支：是否需要澄清
    graph.add_conditional_edges(
        "detect",
        lambda s: "ask" if (s.clarification_need and s.clarification_need.need_clarification) else "refine",
        {"ask": "ask", "refine": "refine"},
    )

    graph.add_edge("ask", "refine")
    graph.add_edge("refine", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================================
# 主函数
# ============================================================================


def run_demo():
    """运行演示"""
    load_environment()
    workflow = create_clarification_workflow()

    print("\n" + "=" * 80)
    print("智能澄清 Agent 演示")
    print("=" * 80)

    user_input = input("\n请输入您的需求（回车使用默认）: ").strip()
    if not user_input:
        user_input = "研究 AI"
        print(f"使用默认需求：{user_input}")

    state = AgentState(user_request=user_input)

    try:
        final_state = workflow.invoke(state)

        print("\n" + "=" * 80)
        print("执行总结")
        print("=" * 80)

        if final_state.clarification_responses:
            print("\n澄清过程：")
            for resp in final_state.clarification_responses:
                for q, a in resp.answers.items():
                    if a != "（跳过）":
                        print(f"  Q: {q}")
                        print(f"  A: {a}")

        print(f"\n执行步骤：{len(final_state.execution_plan)}")
        print(f"完成结果：{len(final_state.results)}")

    except Exception as e:
        print(f"\n执行出错：{e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
