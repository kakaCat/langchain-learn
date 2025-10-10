#!/usr/bin/env python3
"""
Module 7: LangGraph Workflow Chatbot Demo
演示如何在LangChain中使用LangGraph构建高级工作流

本示例实现了一个具有多节点工作流的聊天机器人，包含以下功能：
1. 工作流状态管理
2. 条件分支和路由
3. 工具调用能力
4. 会话记忆功能
5. 错误处理和重试机制
"""
import os
import sys
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

# 确保中文显示正常
sys.stdout.reconfigure(encoding='utf-8')

# 导入必要的库
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# 全局存储会话历史
memory_store: Dict[str, BaseChatMessageHistory] = {}

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

from dataclasses import dataclass

# 定义状态类
@dataclass
class AgentState:
    """定义LangGraph工作流的状态类"""
    messages: List[BaseMessage]
    session_id: str
    is_terminated: bool = False

# 节点函数定义

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """获取或创建会话历史"""
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]


def chatbot_node(state: AgentState) -> AgentState:
    """主聊天机器人节点，处理用户输入并生成回复"""
    llm = get_llm()
    tools = [calculator, get_current_date]
    
    # 将工具绑定到LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # 获取会话历史
    chat_history = get_session_history(state.session_id).messages
    
    # 构建完整的消息列表（历史消息 + 当前消息）
    all_messages = chat_history + state.messages
    
    try:
        # 调用LLM生成回复
        response = llm_with_tools.invoke(all_messages)
        
        # 更新状态中的消息
        return AgentState(
            messages=state.messages + [response],
            session_id=state.session_id,
            is_terminated=False
        )
    except Exception as e:
        # 错误处理
        error_message = AIMessage(content=f"处理请求时发生错误: {str(e)}")
        return AgentState(
            messages=state.messages + [error_message],
            session_id=state.session_id,
            is_terminated=True
        )


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
    
    # 否则继续对话
    return "END"


def update_memory_node(state: AgentState) -> AgentState:
    """更新会话记忆"""
    chat_history = get_session_history(state.session_id)
    
    # 添加新消息到会话历史
    for message in state.messages:
        chat_history.add_message(message)
    
    # 限制历史消息数量，防止token过多
    if len(chat_history.messages) > 10:
        chat_history.messages = chat_history.messages[-10:]
    
    return state

# 创建LangGraph工作流
def create_workflow():
    """创建LangGraph工作流"""
    # 初始化状态图
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("chatbot", chatbot_node)
    # 为ToolNode指定如何处理AgentState实例
    graph.add_node("tools", ToolNode([calculator, get_current_date], input_messages_key="messages", output_messages_key="messages"))
    graph.add_node("update_memory", update_memory_node)
    
    # 设置入口点
    graph.set_entry_point("chatbot")
    
    # 添加条件边 - 检查是否需要调用工具
    graph.add_conditional_edges(
        "chatbot",
        check_tool_call,
        {
            "tools": "tools",
            "END": "update_memory"
        }
    )
    
    # 添加边 - 工具调用后回到聊天机器人节点
    graph.add_edge("tools", "chatbot")
    
    # 添加边 - 更新记忆后结束
    graph.add_edge("update_memory", END)
    
    # 添加检查点，启用状态持久化
    checkpointer = MemorySaver()
    
    # 编译图
    app = graph.compile(checkpointer=checkpointer)
    
    return app

# 运行聊天机器人
def run_chatbot():
    """运行交互式聊天机器人"""
    # 创建工作流
    app = create_workflow()
    
    # 使用固定的会话ID
    session_id = "user_123"
    
    print("===== LangGraph工作流聊天机器人演示 =====")
    print("我是一个基于LangGraph构建的高级聊天机器人，可以执行数学计算和日期查询。")
    print("输入 'exit' 或 'quit' 退出程序。")
    print("\n示例问题:")
    print("1. 计算 123 加 456 等于多少？")
    print("2. 今天是几号？")
    print("3. 我想知道5乘以10的结果。")
    print("\n请输入你的问题：")
    
    # 交互式对话循环
    while True:
        user_input = input("用户: ")
        
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("再见！")
            break
        
        try:
            # 创建初始状态
            initial_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                session_id=session_id,
                is_terminated=False
            )
            
            # 执行工作流
            final_state = app.invoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}}
            )
            
            # 输出最终回复
            # 注意：final_state可能是字典或AgentState实例，需要兼容处理
            try:
                if hasattr(final_state, 'messages'):
                    messages = final_state.messages
                else:
                    messages = final_state.get("messages", [])
                    
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        print(f"AI: {last_message.content}")
                    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        # 如果最后一条消息是工具调用，说明工具调用后没有生成最终回复
                        print("AI: 正在处理您的请求...")
                    else:
                        print(f"AI: {last_message.content}")
            except Exception as e:
                print(f"处理回复时发生错误: {e}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            print("请检查OPENAI_API_KEY等配置是否正确")

def main() -> None:
    """主函数"""
    try:
        # 加载环境变量
        load_environment()
        
        # 运行聊天机器人
        run_chatbot()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()