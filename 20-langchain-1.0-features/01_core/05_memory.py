"""
LangChain 1.0 - 短期记忆 (Short-term Memory)

官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory

核心概念:
- 短期记忆让应用在单个线程/对话中记住之前的交互
- 使用 thread_id 组织多轮对话
- 自动管理上下文窗口（避免超出限制）
- 使用 InMemorySaver / SqliteSaver 持久化对话

本文件包含 6 个示例:
1. 基础短期记忆 - InMemorySaver
2. 上下文窗口管理 - trim_messages
3. @before_model 中间件
4. 多线程对话隔离
5. SqliteSaver 持久化
6. 最佳实践总结

参考资源:
- 官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
"""

import os
from typing import TypedDict, Annotated, Sequence, Any
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import MemorySaver

# 注意: 以下导入在官方 LangChain 1.0 文档中提到，但可能需要最新版本
# from langchain.agents import create_agent, AgentState
# from langchain.agents.middleware import before_model
# from langgraph.runtime import Runtime

load_dotenv()


# ========== 辅助函数 ==========
def get_llm(model_name: str = None, **kwargs):
    """创建并配置语言模型实例"""
    api_key = os.getenv("DEEPSEEK_KEY")
    model = model_name or os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise ValueError("DEEPSEEK_KEY 未设置")
    if not base_url:
        raise ValueError("DEEPSEEK_BASE_URL 未设置")

    params = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0.7,
        **kwargs
    }

    return init_chat_model(**params)


# ========== 示例 1: 基础短期记忆 ==========
def demo_basic_short_term_memory():
    """
    基础短期记忆 - 使用 InMemorySaver

    官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory

    核心概念:
    - thread_id: 标识一个对话线程
    - InMemorySaver: 在内存中保存对话历史
    - messages: 自动累积对话历史
    """
    print("=" * 60)
    print("示例 1: 基础短期记忆 (InMemorySaver)")
    print("=" * 60)

    # 定义状态
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # 定义 agent 节点
    def chatbot(state: AgentState) -> AgentState:
        """简单的聊天机器人"""
        llm = get_llm()
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 创建图
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    # ⭐ 添加 checkpointer (短期记忆的核心)
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("\n✅ Agent 已创建，包含短期记忆 (InMemorySaver)")
    print("=" * 60)

    # 配置 thread_id
    config: RunnableConfig = {"configurable": {"thread_id": "conversation_1"}}

    # 第一轮对话
    print("\n📝 第一轮对话:")
    print("-" * 60)
    print("用户: 我叫张三")
    result1 = app.invoke(
        {"messages": [HumanMessage("我叫张三")]},
        config=config
    )
    print(f"助手: {result1['messages'][-1].content}")

    # 第二轮对话 (测试记忆)
    print("\n📝 第二轮对话 (测试记忆):")
    print("-" * 60)
    print("用户: 我叫什么名字？")
    result2 = app.invoke(
        {"messages": [HumanMessage("我叫什么名字？")]},
        config=config
    )
    print(f"助手: {result2['messages'][-1].content}")
    print("\n💡 Agent 记住了之前的对话内容！")


# ========== 示例 2: 上下文窗口管理 ==========
def demo_trim_messages():
    """
    上下文窗口管理 - 使用 trim_messages

    问题: 长对话会超出模型的上下文窗口
    解决: 使用 trim_messages 保留最近的消息

    官方文档示例中使用的方法
    """
    print("\n\n" + "=" * 60)
    print("示例 2: 上下文窗口管理 (trim_messages)")
    print("=" * 60)

    print("""
🎯 问题: 长对话超出上下文窗口

解决方案: trim_messages
- 保留最近的 N 条消息
- 或者根据 token 数量裁剪
- 确保始终保留系统消息

官方示例:
```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

trimmed_messages = trim_messages(
    state["messages"],
    token_counter=count_tokens_approximately,  # Token 计数器
    strategy="last",                           # 策略: 保留最后 N 条
    max_tokens=500,                            # 最大 token 数
    start_on="human",                          # 从人类消息开始
    allow_partial=True                         # 允许部分消息
)
```

常用参数:
- strategy: "last" (最后N条) 或 "first" (最前N条)
- max_tokens: 最大 token 数量
- start_on: "human" 或 "ai"
- end_on: ("human", "tool") - 结束位置
- allow_partial: 是否允许裁剪单条消息

实战示例:
    """)

    # 定义状态
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # 定义 agent 节点 (带 trim_messages)
    def chatbot_with_trim(state: AgentState) -> AgentState:
        """使用 trim_messages 的聊天机器人"""
        llm = get_llm()

        # ⭐ 裁剪消息，保留最近的 500 tokens
        trimmed_messages = trim_messages(
            state["messages"],
            token_counter=count_tokens_approximately,
            strategy="last",
            max_tokens=500,
            start_on="human",
            allow_partial=True
        )

        print(f"\n  📊 原始消息数: {len(state['messages'])}")
        print(f"  ✂️ 裁剪后消息数: {len(trimmed_messages)}")

        response = llm.invoke(trimmed_messages)
        return {"messages": [response]}

    # 创建图
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot_with_trim)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    config: RunnableConfig = {"configurable": {"thread_id": "conversation_2"}}

    print("\n📝 模拟长对话:")
    print("-" * 60)

    # 模拟 5 轮对话
    questions = [
        "你好",
        "我喜欢编程",
        "Python 是我最喜欢的语言",
        "我在学习 LangChain",
        "你能总结一下我说了什么吗？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{i}. 用户: {question}")
        result = app.invoke(
            {"messages": [HumanMessage(question)]},
            config=config
        )
        print(f"   助手: {result['messages'][-1].content[:100]}...")

    print("\n\n💡 trim_messages 确保对话始终在上下文窗口内！")


# ========== 示例 3: @before_model 中间件 ==========
def demo_before_model_middleware():
    """
    @before_model 中间件 - 在调用模型前处理消息

    官方文档展示的高级用法:
    使用 @before_model 装饰器 + RemoveMessage

    注意: 这需要 LangChain 1.0 的 create_agent API
    """
    print("\n\n" + "=" * 60)
    print("示例 3: @before_model 中间件")
    print("=" * 60)

    print("""
🎯 @before_model 中间件

官方文档示例:
```python
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from typing import Any

@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    \"\"\"Keep only the last few messages to fit context window.\"\"\"
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # 不需要修改

    # 保留第一条消息（通常是系统消息）
    first_msg = messages[0]

    # 保留最近的 3-4 条消息
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]

    # 构建新的消息列表
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 删除所有旧消息
            *new_messages                            # 添加新消息
        ]
    }

# 创建 agent
agent = create_agent(
    "gpt-4o-mini",
    tools=[],
    middleware=[trim_messages],  # ⭐ 添加中间件
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}
```

工作流程:
1. 用户发送消息
2. @before_model 中间件被触发
3. 检查消息数量，决定是否裁剪
4. 使用 RemoveMessage 删除旧消息
5. 保留必要的消息
6. 调用模型

优势:
- ✅ 自动管理上下文窗口
- ✅ 始终保留系统消息
- ✅ 保留最近的对话
- ✅ 对用户透明

💡 这是官方推荐的上下文管理方法！
    """)


# ========== 示例 4: 多线程对话隔离 ==========
def demo_multiple_threads():
    """
    多线程对话隔离

    演示如何使用不同的 thread_id 隔离不同用户/会话的对话
    """
    print("\n\n" + "=" * 60)
    print("示例 4: 多线程对话隔离")
    print("=" * 60)

    # 定义状态
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # 定义 agent 节点
    def chatbot(state: AgentState) -> AgentState:
        llm = get_llm()
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # 创建图
    workflow = StateGraph(AgentState)
    workflow.add_node("chatbot", chatbot)
    workflow.add_edge(START, "chatbot")
    workflow.add_edge("chatbot", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    print("\n✅ Agent 已创建")
    print("=" * 60)

    # 用户 A 的对话
    config_a: RunnableConfig = {"configurable": {"thread_id": "user_alice"}}
    print("\n👤 用户 A (Alice):")
    print("-" * 60)
    print("用户: 我是 Alice，我喜欢猫")
    result_a1 = app.invoke(
        {"messages": [HumanMessage("我是 Alice，我喜欢猫")]},
        config=config_a
    )
    print(f"助手: {result_a1['messages'][-1].content}")

    # 用户 B 的对话
    config_b: RunnableConfig = {"configurable": {"thread_id": "user_bob"}}
    print("\n👤 用户 B (Bob):")
    print("-" * 60)
    print("用户: 我是 Bob，我喜欢狗")
    result_b1 = app.invoke(
        {"messages": [HumanMessage("我是 Bob，我喜欢狗")]},
        config=config_b
    )
    print(f"助手: {result_b1['messages'][-1].content}")

    # 用户 A 继续对话
    print("\n👤 用户 A 继续对话:")
    print("-" * 60)
    print("用户: 我喜欢什么动物？")
    result_a2 = app.invoke(
        {"messages": [HumanMessage("我喜欢什么动物？")]},
        config=config_a
    )
    print(f"助手: {result_a2['messages'][-1].content}")

    # 用户 B 继续对话
    print("\n👤 用户 B 继续对话:")
    print("-" * 60)
    print("用户: 我喜欢什么动物？")
    result_b2 = app.invoke(
        {"messages": [HumanMessage("我喜欢什么动物？")]},
        config=config_b
    )
    print(f"助手: {result_b2['messages'][-1].content}")

    print("\n\n💡 不同 thread_id 的对话完全隔离！")
    print("   每个用户都有独立的对话历史")


# ========== 示例 5: SqliteSaver 持久化 ==========
def demo_sqlite_saver():
    """
    SqliteSaver 持久化存储

    InMemorySaver: 内存存储，重启后丢失
    SqliteSaver: 数据库存储，持久化
    """
    print("\n\n" + "=" * 60)
    print("示例 5: SqliteSaver 持久化存储")
    print("=" * 60)

    print("""
🎯 持久化选项对比:

1. MemorySaver (内存)
   from langgraph.checkpoint.memory import MemorySaver
   checkpointer = MemorySaver()

   ✅ 优点: 快速、简单
   ❌ 缺点: 重启后丢失
   📍 适用: 开发、测试

2. SqliteSaver (SQLite 数据库) ⭐推荐生产使用
   from langgraph.checkpoint.sqlite import SqliteSaver
   checkpointer = SqliteSaver("conversations.db")

   ✅ 优点: 持久化、可靠
   ✅ 优点: 不需要额外服务
   ❌ 缺点: 单机部署
   📍 适用: 中小型生产环境

3. PostgresSaver (PostgreSQL 数据库)
   from langgraph.checkpoint.postgres import PostgresSaver
   checkpointer = PostgresSaver(connection_string)

   ✅ 优点: 高性能、可扩展
   ✅ 优点: 支持分布式
   ❌ 缺点: 需要 PostgreSQL 服务
   📍 适用: 大规模生产环境

使用示例:
```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建持久化 checkpointer
checkpointer = SqliteSaver("chat_history.db")

app = workflow.compile(checkpointer=checkpointer)

# 使用方式完全相同
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"messages": [...]}, config=config)
```

💡 生产环境推荐使用 SqliteSaver 或 PostgresSaver！
    """)


# ========== 示例 6: 最佳实践 ==========
def demo_best_practices():
    """
    短期记忆最佳实践总结
    """
    print("\n\n" + "=" * 60)
    print("示例 6: 短期记忆最佳实践")
    print("=" * 60)

    print("""
🎯 LangChain 1.0 短期记忆最佳实践

1️⃣ 选择合适的 Checkpointer
   - 开发/测试: MemorySaver
   - 生产环境: SqliteSaver 或 PostgresSaver
   - 分布式: PostgresSaver

2️⃣ 上下文窗口管理
   ✅ 使用 trim_messages 裁剪消息
   ✅ 使用 @before_model 中间件自动管理
   ✅ 始终保留系统消息
   ❌ 不要让对话超出模型限制

3️⃣ thread_id 设计
   - 用户级别: thread_id = f"user_{user_id}"
   - 会话级别: thread_id = f"session_{session_id}"
   - 确保唯一性
   - 不要泄露敏感信息

4️⃣ 消息裁剪策略
   strategy="last": 保留最近的消息 ⭐推荐
   strategy="first": 保留最早的消息

   start_on="human": 从人类消息开始 ⭐推荐
   end_on=("human", "tool"): 在人类或工具消息结束

5️⃣ Token 管理
   - 使用 count_tokens_approximately 估算
   - 设置合理的 max_tokens (建议模型限制的 80%)
   - 允许 allow_partial=True 裁剪单条消息

6️⃣ 性能优化
   ✅ 批量查询历史消息
   ✅ 使用数据库索引 (thread_id)
   ✅ 定期清理过期对话
   ❌ 避免频繁的小查询

7️⃣ 安全性
   ✅ 加密存储敏感对话
   ✅ 实现访问控制
   ✅ 定期备份数据库
   ❌ 不要在 thread_id 中包含敏感信息

8️⃣ 监控和调试
   ✅ 记录对话长度
   ✅ 监控 token 使用量
   ✅ 追踪裁剪次数
   ✅ 记录错误和异常

9️⃣ 错误处理
   ```python
   try:
       result = app.invoke({"messages": [...]}, config=config)
   except Exception as e:
       print(f"错误: {e}")
       # 回退到无记忆模式
   ```

🔟 测试策略
   ✅ 测试长对话（100+ 轮）
   ✅ 测试多线程隔离
   ✅ 测试裁剪逻辑
   ✅ 测试持久化恢复

---

📚 完整生产配置示例:
```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def chatbot_with_trim(state: AgentState) -> AgentState:
    # 裁剪消息
    trimmed_messages = trim_messages(
        state["messages"],
        token_counter=count_tokens_approximately,
        strategy="last",
        max_tokens=4000,  # GPT-4 的 80%
        start_on="human",
        allow_partial=True
    )

    # 调用模型
    llm = get_llm()
    response = llm.invoke(trimmed_messages)

    return {"messages": [response]}

# 创建图
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_with_trim)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# ⭐ 持久化存储
checkpointer = SqliteSaver("production_chat.db")
app = workflow.compile(checkpointer=checkpointer)

# 使用
config = {"configurable": {"thread_id": f"user_{user_id}"}}
result = app.invoke({"messages": [HumanMessage("...")]}, config=config)
```

这是一个安全、高效、可扩展的生产级配置！
    """)


# ========== 主函数 ==========
def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LangChain 1.0 短期记忆完整示例")
    print("=" * 60)
    print("\n官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory")
    print("\n本文件包含 6 个示例:")
    print("  1. 基础短期记忆 - InMemorySaver")
    print("  2. 上下文窗口管理 - trim_messages ⭐")
    print("  3. @before_model 中间件 ⭐")
    print("  4. 多线程对话隔离")
    print("  5. SqliteSaver 持久化 ⭐")
    print("  6. 最佳实践总结")
    print("=" * 60)

    # 运行所有示例
    demo_basic_short_term_memory()
    demo_trim_messages()
    demo_before_model_middleware()
    demo_multiple_threads()
    demo_sqlite_saver()
    demo_best_practices()

    print("\n\n" + "=" * 60)
    print("✅ 所有短期记忆示例演示完成")
    print("=" * 60)
    print("\n📚 参考资源:")
    print("  - 官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory")
    print("  - LangGraph 文档: https://langchain-ai.github.io/langgraph/")
    print("  - trim_messages 示例: https://cleancodestack.com/add-short-term-memory-in-langgraph/")
    print("=" * 60)


if __name__ == "__main__":
    main()
