"""
LangChain 1.0+ Modern Approach
==============================

此脚本演示了 LangChain 1.0+ (使用 LangGraph) 构建 Chatbot 的推荐方式。

核心特点：
1. 架构：使用 Graph (有向有环图)
2. 状态管理：使用 StateSchema (TypedDict)
3. 工具调用：使用 prebuilt.create_react_agent
4. 统一模型初始化：使用 init_chat_model

本示例展示：
- 使用 LangGraph create_react_agent 构建对话 Agent
- 使用 MemorySaver 进行状态持久化
- 使用 init_chat_model 初始化模型
"""

import os
import sys
from typing import Annotated, Literal, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, trim_messages
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# 加载环境变量 (强制覆盖)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

def get_llm():
    # 使用统一的模型初始化入口 init_chat_model
    # 1.0+ 推荐方式：解耦具体提供商的类依赖
    return init_chat_model(
        model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
        model_provider="openai",  # 显式指定提供商 (DeepSeek 使用 OpenAI 协议)
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7
    )

# ==========================================
# Modern Approach (LangChain 1.0+ / LangGraph)
# 使用 StateGraph 显式构建 Graph (Best Practice)
# 
# LangChain 1.0 架构分层：
# - LangChain 层：提供 create_agent 等快速入口，适合 PoC。
# - LangGraph 层：提供底层图式编排（如下所示），适合生产级精细控制。
#   具备状态管理、检查点、可中断/恢复、并发控制能力。
# ==========================================

# 1. 定义状态 (State)
# 使用 TypedDict 定义图中的共享状态，这里我们只存储消息列表
# add_messages 是一个 reducer，它会将新消息追加到现有列表中，而不是覆盖
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Annotated[list, add_messages] 告诉 LangGraph 当有新消息时自动 append
    messages: Annotated[list[BaseMessage], add_messages]

# 2. 定义节点 (Nodes)
# 节点是图中的执行单元，接收当前 State，返回更新后的 State
def chatbot(state: State):
    """
    Chatbot 节点：调用 LLM 并返回响应
    """
    llm = get_llm()
    # 调用模型
    response = llm.invoke(state["messages"])
    # 返回更新的状态（这里只返回新消息，add_messages 会处理追加）
    return {"messages": [response]}

# ==========================================
# LangChain 1.0 三大里程碑能力演示
# 1. 统一入口 (create_agent)
# 2. 中间件 (Middleware - 模型调用限制)
# 3. 标准化内容块 (Standard Content Blocks)
# ==========================================

@tool
def get_weather(city: str):
    """查询某城市的天气"""
    # 模拟工具调用
    return f"{city} 的天气是晴天，25度。"

def demonstrate_milestones():
    print("\n=== LangChain 1.0 三大里程碑能力演示 ===")
    
    llm = get_llm()
    memory = MemorySaver()
    
    # --- 能力 2: 中间件 (Middleware) ---
    # 使用 1.0 原生 Middleware 机制
    # 示例：ModelCallLimitMiddleware (限制模型调用次数，防止死循环或超额消耗)
    # 这比手动修改状态（如 trim_messages）更符合 AOP 切面编程思想
    middleware = [
        ModelCallLimitMiddleware(
            thread_limit=5,  # 每个线程最多调用 5 次模型
            run_limit=3,     # 每次运行最多调用 3 次模型
            exit_behavior="end" # 达到限制后停止
        )
    ]
    
    # --- 能力 1: 统一入口 (create_agent) ---
    # 使用统一 API create_agent 创建 Agent
    # 替代了旧版的 create_react_agent
    agent = create_agent(
        model=llm, 
        tools=[get_weather], 
        checkpointer=memory,
        middleware=middleware,  # 注入中间件
        # 使用 system_prompt 参数注入系统指令
        system_prompt="你是一个有用的助手。" 
    )
    
    print("\n[Milestone 1] 统一入口 create_agent 已就绪")
    print("[Milestone 2] 中间件 ModelCallLimitMiddleware 已注入")
    
    config = {"configurable": {"thread_id": "milestone_demo_1"}}
    
    # 执行对话
    print("\nUser: 北京天气怎么样？")
    input_msg = HumanMessage(content="北京天气怎么样？")
    
    # --- 能力 3: 标准化内容块 (Standard Content Blocks) ---
    # 观察输出，无论是文本还是工具调用，都遵循统一格式
    for chunk in agent.stream({"messages": [input_msg]}, config=config):
        # chunk 格式标准化：{'agent': {'messages': [...]}} 或 {'tools': {'messages': [...]}}
        for node_name, node_output in chunk.items():
            if node_output is None:
                continue
            if "messages" in node_output:
                last_msg = node_output["messages"][-1]
                print(f"\n[Node: {node_name}] 消息类型: {type(last_msg).__name__}")
                
                # 检查标准内容块
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                     print(f"  -> 触发工具调用 (Standard Tool Call Block): {last_msg.tool_calls}")
                elif hasattr(last_msg, "content") and last_msg.content:
                     print(f"  -> 文本内容: {last_msg.content}")

    print("\n[Milestone 3] 验证完成：所有模型输出均被标准化为 Message 对象和 Block")

def run_modern_demo():
    # 先运行里程碑演示
    demonstrate_milestones()
    
    print("\n" + "="*50)
    print("以下是底层 LangGraph 手动构建演示 (用于精细控制)")
    print("="*50)
    
    print("\n--- [Modern] LangGraph Agent Demo (Manual Graph) ---")
    
    # 3. 构建图 (Graph Construction)
    # 这是 LangGraph 的核心：显式定义工作流
    workflow = StateGraph(State)
    
    # 添加节点
    workflow.add_node("chatbot", chatbot)
    
    # 添加边 (Edges)
    # 定义执行流程：开始 -> chatbot -> 结束
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)
    
    # 4. 编译图 (Compile)
    # 编译后生成可执行的 Runnable
    # checkpointer 替代了旧版的 Memory 类，用于持久化状态
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    # 配置线程 ID (类似 Session ID)
    # 不同的 thread_id 会隔离不同的对话历史
    config = {"configurable": {"thread_id": "demo_thread_manual_1"}}
    
    # --- 第一轮对话 ---
    input_message = HumanMessage(content="你好，我是通过 LangGraph 手动构建的图来的。")
    print(f"User: {input_message.content}")
    
    # stream 是 1.0+ 的一等公民，支持流式输出中间步骤
    # event 格式: {'chatbot': {'messages': [AIMessage(...)]}}
    for event in app.stream({"messages": [input_message]}, config=config):
        for key, value in event.items():
            if key == "chatbot" and "messages" in value:
                # 获取最后一条消息的内容
                last_msg = value["messages"][-1]
                print(f"Bot (Node: {key}): {last_msg.content}")

    # --- 第二轮对话 (测试记忆) ---
    input_message2 = HumanMessage(content="我刚才说了什么？")
    print(f"\nUser: {input_message2.content}")
    
    for event in app.stream({"messages": [input_message2]}, config=config):
        for key, value in event.items():
            if key == "chatbot" and "messages" in value:
                last_msg = value["messages"][-1]
                print(f"Bot (Node: {key}): {last_msg.content}")

    print("\n[Info] 图结构已构建完成，这种方式比 create_react_agent 更灵活，")
    print("       允许你自定义任意的节点、条件分支和循环。")

if __name__ == "__main__":
    run_modern_demo()
