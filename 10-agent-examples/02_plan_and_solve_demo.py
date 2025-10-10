#!/usr/bin/env python3
"""
Module 7: Plan-and-Solve 模式代理实现（无记忆版本）
演示如何在 LangChain 和 LangGraph 中实现 Plan-and-Solve（先规划再执行）代理模式。

流程概览：
1. Plan（规划）：根据用户问题生成可执行的分步方案
2. Solve（执行）：按步骤执行，必要时调用工具，获得最终答案
3. Observation（观察）：工具返回结果后继续执行直至完成
注意：此版本不包含会话记忆功能，每次对话都是独立的。
"""
import os
import sys
from typing import List, Optional
from datetime import datetime
from dataclasses import dataclass

# 确保中文显示正常
sys.stdout.reconfigure(encoding='utf-8')

# 第三方库
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 从当前模块目录加载 .env

def load_environment():
    """加载环境变量"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 获取配置的语言模型

def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)

# 定义工具

@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    基本数学计算工具。
    operation: 'add' | 'subtract' | 'multiply' | 'divide'
    """
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    else:
        raise ValueError(f"不支持的操作: {operation}")

@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    """获取当前日期字符串。"""
    return datetime.now().strftime(format)

@tool
def search_information(query: str) -> str:
    """
    模拟信息检索工具。
    """
    mock_results = {
        "langgraph": "LangGraph是一个用于构建代理工作流的框架，支持状态管理、条件路由和工具调用。",
        "plan-and-solve": "Plan-and-Solve是一种先规划步骤再逐步执行的模式，适合复杂任务分解。",
        "langchain": "LangChain是一个用于构建LLM应用的框架，提供模型、提示、工具和内存等组件。",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"未找到与 '{query}' 相关的信息。"

# 定义状态类

@dataclass
class AgentState:
    """Plan-and-Solve 代理状态（无记忆版本）"""
    messages: List[BaseMessage]
    plan: Optional[str] = None
    is_terminated: bool = False

# 规划阶段：生成分步方案

def plan_node(state: AgentState) -> AgentState:
    llm = get_llm()
    # 仅基于当前用户问题生成计划
    user_msg = next((m for m in state.messages if isinstance(m, HumanMessage)), None)
    question = user_msg.content if user_msg else ""
    plan_prompt = (
        "你是一个规划专家。请为以下问题生成一个清晰的分步解决计划，"
        "每一步应简洁可执行，必要时标注需要使用的工具（计算器/日期/信息检索）。\n"
        f"用户问题：{question}"
    )
    plan_ai = llm.invoke([HumanMessage(content=plan_prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"规划方案：\n{plan_ai.content}"))

    return AgentState(messages=new_messages, plan=plan_ai.content, is_terminated=False)

# 执行阶段：按计划解决问题，必要时调用工具

def solve_node(state: AgentState) -> AgentState:
    llm = get_llm()
    tools = [calculator, get_current_date, search_information]
    llm_with_tools = llm.bind_tools(tools)

    solve_instruction = (
        "现在基于上述规划逐步执行。遇到需要计算、日期或检索的步骤时，请调用相应工具。"
        "完成后给出最终答案。如果已经得到最终答案，则不要再调用工具。"
    )
    messages = state.messages + [HumanMessage(content=solve_instruction)]

    # 规范化消息序列，确保每条 ToolMessage 前都有对应的 assistant(tool_calls)
    last_ai_with_tools_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", []):
            last_ai_with_tools_idx = idx
            break
    if any(isinstance(m, ToolMessage) for m in messages):
        if last_ai_with_tools_idx is not None:
            context_messages = messages[last_ai_with_tools_idx:]
        else:
            context_messages = [m for m in messages if not isinstance(m, ToolMessage)]
    else:
        context_messages = messages

    response = llm_with_tools.invoke(context_messages)

    new_messages = state.messages.copy()
    new_messages.append(response)

    # 是否结束：当没有工具调用时认为可能已生成最终答案
    is_terminated = not getattr(response, "tool_calls", [])
    return AgentState(messages=new_messages, plan=state.plan, is_terminated=is_terminated)

# 条件路由：决定下一步

def check_next(state: AgentState) -> str:
    if state.plan is None:
        return "plan"
    last = state.messages[-1] if state.messages else None
    if last and getattr(last, "tool_calls", []):
        return "tools"
    if state.is_terminated:
        return "END"
    return "solve"

# 创建工作流

def create_plan_and_solve_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("solve", solve_node)
    graph.add_node("tools", ToolNode([calculator, get_current_date, search_information]))

    graph.set_entry_point("plan")

    graph.add_conditional_edges(
        "plan",
        check_next,
        {"solve": "solve", "plan": "plan", "tools": "tools", "END": END},
    )

    graph.add_conditional_edges(
        "solve",
        check_next,
        {"tools": "tools", "solve": "solve", "END": END},
    )

    graph.add_edge("tools", "solve")
    app = graph.compile()
    return app

# 运行演示 CLI

def main():
    load_environment()
    app = create_plan_and_solve_workflow()

    print("===== Plan-and-Solve 模式代理演示 ======")
    print("该代理会先为你的问题制定计划，再按计划执行并在必要时调用工具。")
    print("注意：此版本不保存对话历史，每次对话都是独立的。输入 '退出' 结束对话。\n")
    print("示例问题（可用工具：计算器/日期/检索）：")
    print("1. 今天是几号？")
    print("2. 15乘以23等于多少？")
    print("3. 检索一下 LangGraph 是什么并给出简要说明")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            initial_state = AgentState(messages=[HumanMessage(content=user_input)], plan=None, is_terminated=False)
            final_state = app.invoke(initial_state)

            # 输出最后一条纯 AI 回复（不含工具调用）
            for msg in reversed(final_state.messages):
                if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
                    print(f"代理: {msg.content}")
                    break
        except KeyboardInterrupt:
            print("\n代理: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"代理: 发生错误: {str(e)}")

if __name__ == "__main__":
    main()