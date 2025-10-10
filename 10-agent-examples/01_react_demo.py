#!/usr/bin/env python3
"""
Module 7: REACT 模式代理实现（无记忆版本）
演示如何在LangChain和LangGraph中实现无记忆的REACT（推理-行动-观察-思考）代理模式

REACT模式是一种代理架构，包含以下核心组件：
1. Reasoning（推理）：代理思考如何解决问题
2. Action（行动）：代理执行动作（通常是调用工具）
3. Observation（观察）：代理观察动作的结果
4. 注意：此版本不包含会话记忆功能，每次对话都是独立的
"""
import os
import sys
from typing import List, Any, Optional
from datetime import datetime

# 确保中文显示正常
sys.stdout.reconfigure(encoding='utf-8')

# 导入必要的库
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dataclasses import dataclass

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
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

# 定义工具
@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    用于执行基本数学计算的工具。
    
    参数:
    a: 第一个数字
    b: 第二个数字
    operation: 操作类型，可以是 'add'(加), 'subtract'(减), 'multiply'(乘), 'divide'(除)
    
    返回:
    计算结果
    """
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    else:
        raise ValueError(f"不支持的操作: {operation}")

@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    """
    获取当前日期。
    
    参数:
    format: 日期格式字符串，默认为 '%Y-%m-%d'(年-月-日)
    
    返回:
    当前日期的字符串表示
    """
    return datetime.now().strftime(format)

@tool
def search_information(query: str) -> str:
    """
    用于搜索相关信息的工具。
    
    参数:
    query: 搜索查询字符串
    
    返回:
    搜索结果的文本描述
    """
    # 模拟搜索结果
    mock_results = {
        "langgraph": "LangGraph是一个用于构建代理工作流的框架，支持状态管理、条件路由和工具调用。",
        "react模式": "REACT模式是一种代理架构，包含推理(Reasoning)、行动(Action)、观察(Observation)和思考(Think)四个步骤。",
        "langchain": "LangChain是一个用于构建LLM应用的框架，提供了模型、提示、工具和内存等组件。"
    }
    
    # 查找最匹配的结果
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    
    return f"未找到与 '{query}' 相关的信息。"

# 定义状态类
@dataclass
class AgentState:
    """定义REACT代理的状态类（无记忆版本）"""
    messages: List[BaseMessage]
    is_terminated: bool = False
    reasoning: Optional[str] = None

# REACT代理节点 - 推理阶段
def reasoning_node(state: AgentState) -> AgentState:
    """
    REACT模式的推理阶段：分析问题，决定下一步行动
    """
    llm = get_llm()
    tools = [calculator, get_current_date, search_information]
    llm_with_tools = llm.bind_tools(tools)
    
    # 构造合法的上下文消息：
    # 若存在 tool 消息，则必须确保其前面紧邻的是包含 tool_calls 的 assistant 消息；
    # 若不存在这样的 assistant 消息，则移除悬空的 tool 消息，避免 OpenAI 400 错误。
    messages = state.messages
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
            # 移除所有悬空的 tool 消息
            context_messages = [m for m in messages if not isinstance(m, ToolMessage)]
    else:
        context_messages = messages
    
    # 调用LLM进行推理
    response = llm_with_tools.invoke(context_messages)
    
    # 更新状态
    new_messages = state.messages.copy()
    new_messages.append(response)
    
    # 检查是否需要终止
    is_terminated = not hasattr(response, "tool_calls") or not response.tool_calls
    
    return AgentState(
        messages=new_messages,
        is_terminated=is_terminated,
        reasoning="完成推理，决定下一步行动"
    )

# 检查是否需要调用工具
def check_tool_call(state: AgentState) -> str:
    """检查是否需要调用工具"""
    messages = state.messages
    if not messages:
        return "END"
    
    last_message = messages[-1]
    
    # 检查是否有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # 检查是否需要结束对话
    if state.is_terminated:
        return "END"
    
    # 无记忆版本，直接结束
    return "END"

# 创建REACT工作流
def create_react_workflow():
    """创建无记忆的REACT模式LangGraph工作流"""
    # 初始化状态图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", ToolNode([calculator, get_current_date, search_information]))
    
    # 设置入口点
    graph.set_entry_point("reasoning")
    
    # 添加条件边 - 检查是否需要调用工具
    graph.add_conditional_edges(
        "reasoning",
        check_tool_call,
        {
            "tools": "tools",
            "END": END
        }
    )
    
    # 添加边 - 工具调用后回到推理节点，形成REACT循环
    graph.add_edge("tools", "reasoning")
    
    # 编译图（无记忆版本，不需要检查点）
    app = graph.compile()

    return app

# 运行REACT代理
def main():
    """运行无记忆的REACT代理演示"""
    # 加载环境变量
    load_environment()
    
    # 创建工作流
    app = create_react_workflow()
    
    print("===== 无记忆REACT模式代理演示 ======")
    print("这是一个基于REACT（推理-行动-观察-思考）模式的代理演示。")
    print("注意：此版本不保存对话历史，每次对话都是独立的。")
    print("你可以问问题，代理会通过推理决定是否需要使用工具来回答。")
    print("示例问题：")
    print("1. 今天是几号？")
    print("2. 15乘以23等于多少？")
    print("3. 什么是LangGraph？")
    print("输入'退出'结束对话\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ")
            
            # 检查是否退出
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break
            
            # 创建初始状态（无记忆版本，不需要session_id）
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                is_terminated=False
            )
            
            # 执行工作流
            final_state = app.invoke(initial_state)
            
            # 显示最终回复
            if hasattr(final_state, 'messages') and final_state.messages:
                # 查找最后一条AI消息作为回复
                for message in reversed(final_state.messages):
                    if isinstance(message, AIMessage) and not hasattr(message, 'tool_calls'):
                        print(f"代理: {message.content}")
                        break
        except KeyboardInterrupt:
            print("\n代理: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"代理: 发生错误: {str(e)}")

# 主函数
if __name__ == "__main__":
    main()