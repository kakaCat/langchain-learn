#!/usr/bin/env python3
"""
Module 7: Basic Reflection 模式代理实现（无记忆版本）
演示如何在 LangChain 与 LangGraph 中实现“先生成初稿 → 自我反思 → 修订答案”的基本反思模式。

流程概览：
1. Draft（初稿）：根据用户问题生成第一版答案
2. Reflect（反思）：对初稿进行结构化自我评估与改进建议
3. Revise（修订）：依据反思建议输出改善后的最终答案
注意：此版本不包含会话记忆功能，每次对话都是独立的；不调用任何外部工具。
"""
import os
import sys
from typing import List, Optional
from dataclasses import dataclass

# 确保中文显示正常
sys.stdout.reconfigure(encoding='utf-8')

# 第三方库
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

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

# 定义状态类

@dataclass
class AgentState:
    """Basic Reflection 代理状态（无记忆版本）"""
    messages: List[BaseMessage]
    draft: Optional[str] = None
    critique: Optional[str] = None
    is_terminated: bool = False

# 初稿阶段：生成第一版答案

def draft_node(state: AgentState) -> AgentState:
    llm = get_llm()
    user_msg = next((m for m in state.messages if isinstance(m, HumanMessage)), None)
    question = user_msg.content if user_msg else ""
    draft_prompt = (
        "请基于下面的问题生成一个清晰、结构化的初稿答案。\n"
        "要求：语言简洁、条理清晰，若为步骤型任务请给出可执行步骤。\n"
        f"用户问题：{question}"
    )
    draft_ai = llm.invoke([HumanMessage(content=draft_prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"初稿：\n{draft_ai.content}"))

    return AgentState(messages=new_messages, draft=draft_ai.content, critique=None, is_terminated=False)

# 反思阶段：自我评估并提出改进建议

def reflect_node(state: AgentState) -> AgentState:
    llm = get_llm()
    reflect_prompt = (
        "你是一个审阅与自我反思专家。请对上述初稿进行结构化评审，输出如下要点：\n"
        "1) 正确性/一致性检查（如有计算或事实，指出可能的错误）\n"
        "2) 清晰度与结构性建议\n"
        "3) 可执行性或实用性建议\n"
        "4) 是否需要修订（是/否）及具体修订方向\n"
        f"初稿内容：\n{state.draft or ''}"
    )
    critique_ai = llm.invoke(state.messages + [HumanMessage(content=reflect_prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"反思：\n{critique_ai.content}"))

    return AgentState(messages=new_messages, draft=state.draft, critique=critique_ai.content, is_terminated=False)

# 修订阶段：依据反思建议输出最终答案

def revise_node(state: AgentState) -> AgentState:
    llm = get_llm()
    revise_prompt = (
        "请依据上述反思建议，对初稿进行一次整体修订，输出最终答案。\n"
        "要求：\n"
        "- 若反思认为无需修订，则简要说明理由并维持原答案；\n"
        "- 若需要修订：修正错误、优化结构与表述、提升可执行性；\n"
        f"初稿：\n{state.draft or ''}\n"
        f"反思建议：\n{state.critique or ''}"
    )
    final_ai = llm.invoke(state.messages + [HumanMessage(content=revise_prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"修订答案：\n{final_ai.content}"))

    return AgentState(messages=new_messages, draft=state.draft, critique=state.critique, is_terminated=True)

# 条件路由：决定下一步

def check_next(state: AgentState) -> str:
    if state.draft is None:
        return "draft"
    if state.critique is None:
        return "reflect"
    if state.is_terminated:
        return "END"
    return "revise"

# 创建工作流

def create_basic_reflection_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("draft", draft_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("revise", revise_node)

    graph.set_entry_point("draft")

    graph.add_conditional_edges(
        "draft",
        check_next,
        {"reflect": "reflect", "draft": "draft", "END": END},
    )

    graph.add_conditional_edges(
        "reflect",
        check_next,
        {"revise": "revise", "reflect": "reflect", "END": END},
    )

    graph.add_conditional_edges(
        "revise",
        check_next,
        {"END": END, "revise": "revise"},
    )

    app = graph.compile()
    return app

# 运行演示 CLI

def main():
    load_environment()
    app = create_basic_reflection_workflow()

    print("===== Basic Reflection 模式代理演示 ======")
    print("该代理会先生成初稿，再自我反思，最后输出修订后的答案（不调用外部工具）。")
    print("注意：此版本不保存对话历史，每次对话都是独立的。输入 '退出' 结束对话。\n")
    print("示例问题（先生成初稿 → 反思 → 修订）：")
    print("1. 请给出三条提升睡眠质量的建议，并自我检查是否具体可执行")
    print("2. 计算 15×23 的结果，并在反思中检查计算过程是否有误")
    print("3. 解释 LangGraph 的作用，并反思答案是否清晰简洁")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            initial_state = AgentState(messages=[HumanMessage(content=user_input)], draft=None, critique=None, is_terminated=False)
            final_state = app.invoke(initial_state)

            # 兼容字典或数据类返回，统一获取消息列表
            messages = []
            if isinstance(final_state, dict):
                messages = final_state.get("messages", [])
            else:
                messages = getattr(final_state, "messages", [])

            # 输出最后一个 AI 回复（修订答案）
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print(f"代理: {msg.content}")
                    break
            else:
                print("代理: 对不起，未能生成答案。")
        except KeyboardInterrupt:
            print("\n代理: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"代理: 发生错误: {str(e)}")

if __name__ == "__main__":
    main()