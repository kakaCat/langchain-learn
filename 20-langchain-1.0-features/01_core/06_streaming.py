"""
LangChain 1.0 - 流式处理 (Streaming) - 官方文档完整实现

严格按照 https://docs.langchain.com/oss/python/langchain/streaming 实现

核心功能:
1. Token-by-Token 流式输出
2. 流式工具调用
3. 异步流式处理
4. 事件流 (astream_events)
5. 流式与非流式对比
6. 实际应用场景

参考文档:
- https://python.langchain.com/docs/how_to/streaming/
- https://python.langchain.com/docs/concepts/streaming/
"""

import os
import asyncio
from typing import AsyncIterator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
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


# ========== 1. 基础流式输出 ==========
def demo_basic_streaming():
    """基础的 Token-by-Token 流式输出"""
    print("=" * 70)
    print("1. 基础流式输出 (Token-by-Token)")
    print("=" * 70)

    def get_weather(city: str) -> str:
        """Get weather for a given city."""

        return f"It's always sunny in {city}!"

    agent_model  = init_chat_model()
    agent = create_agent(
        model = agent_model,
        tools=[get_weather],
    )
    for chunk in agent.stream(  
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")

# ========== 2. 基础流式token输出 ==========
def demo_token_streaming():
    def get_weather(city: str) -> str:
        """Get weather for a given city."""

        return f"It's always sunny in {city}!"

    agent_model  = init_chat_model()
    agent = create_agent(
        model=agent_model,
        tools=[get_weather],
    )
    for token, metadata in agent.stream(  
        {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
        stream_mode="messages",
    ):
        print(f"node: {metadata['langgraph_node']}")
        print(f"content: {token.content_blocks}")
        print("\n")

# ========== 2. 流式 vs 非流式对比 ==========
def demo_streaming_comparison():
    """对比流式和非流式输出"""
    print("\n\n" + "=" * 70)
    print("2. 流式 vs 非流式对比")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = "解释什么是量子计算"

    # 非流式 - 等待完整响应
    print("\n❌ 非流式 (invoke):")
    print("  [等待...]")
    import time
    start = time.time()
    response = llm.invoke(prompt)
    end = time.time()
    print(f"  完整响应: {response.content[:50]}...")
    print(f"  耗时: {end-start:.2f}秒\n")

    # 流式 - 逐步显示
    print("✅ 流式 (stream):")
    print("  ", end="", flush=True)
    start = time.time()
    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)
    end = time.time()
    print(f"\n  耗时: {end-start:.2f}秒")

    print("\n💡 优势: 流式输出提供即时反馈，改善用户体验")




# ========== 4. 流式工具调用 ==========
def demo_streaming_with_tools():
    """流式输出 + 工具调用"""
    print("\n\n" + "=" * 70)
    print("4. 流式工具调用")
    print("=" * 70)

    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气"""
        return f"{city}: 晴天, 25°C"

    @tool
    def search_flights(origin: str, destination: str) -> str:
        """搜索航班"""
        return f"{origin} → {destination}: 找到3个航班"

    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools([get_weather, search_flights])

    print("\n用户: 查询北京天气，然后搜索北京到上海的航班")
    print("\nAI思考过程 (流式):")

    for chunk in llm_with_tools.stream("查询北京天气，然后搜索北京到上海的航班"):
        # 可能是文本内容
        if chunk.content:
            print(f"  [思考] {chunk.content}")

        # 可能是工具调用
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                print(f"  [调用工具] {tool_call['name']}({tool_call['args']})")


# ========== 5. 批量流式处理 ==========
def demo_batch_streaming():
    """批量请求的流式处理"""
    print("\n\n" + "=" * 70)
    print("5. 批量流式处理")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    # 批量请求
    prompts = [
        "Python的优点是什么？",
        "JavaScript的优点是什么？",
        "Go语言的优点是什么？"
    ]

    print("\n批量流式处理 (逐个请求):")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. {prompt}")
        print("   回答: ", end="", flush=True)

        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)


# ========== 6. 异步流式输出 ==========
async def demo_async_streaming():
    """异步流式输出"""
    print("\n\n" + "=" * 70)
    print("6. 异步流式输出 (astream)")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    print("\n异步流式输出:")
    print("  ", end="", flush=True)

    # astream() 返回异步生成器
    async for chunk in llm.astream("写一首关于春天的诗"):
        print(chunk.content, end="", flush=True)

    print("\n")


# ========== 7. 异步并发流式处理 ==========
async def demo_concurrent_streaming():
    """并发处理多个流式请求"""
    print("\n\n" + "=" * 70)
    print("7. 异步并发流式处理")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    async def stream_response(query: str, index: int):
        """单个流式响应"""
        print(f"\n[请求 {index}] {query}")
        print(f"[响应 {index}] ", end="", flush=True)

        async for chunk in llm.astream(query):
            print(chunk.content, end="", flush=True)

        print()  # 换行

    # 并发执行多个流式请求
    queries = [
        "什么是AI？",
        "什么是机器学习？",
        "什么是深度学习？"
    ]

    print("\n🚀 并发执行3个流式请求:")
    tasks = [stream_response(q, i) for i, q in enumerate(queries, 1)]
    await asyncio.gather(*tasks)


# ========== 8. 事件流 (astream_events) ==========
async def demo_astream_events():
    """使用 astream_events 获取详细事件"""
    print("\n\n" + "=" * 70)
    print("8. 事件流 (astream_events)")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    @tool
    def calculator(a: float, b: float, operation: str) -> float:
        """执行数学计算"""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        return 0

    llm_with_tools = llm.bind_tools([calculator])

    print("\n监听所有事件:")

    # astream_events 返回详细的事件流
    async for event in llm_with_tools.astream_events(
        "计算 5 + 3 的结果",
        version="v2"
    ):
        kind = event["event"]
        data = event.get("data", {})

        if kind == "on_chat_model_start":
            print(f"  [事件] 模型开始")

        elif kind == "on_chat_model_stream":
            content = data.get("chunk", {}).content
            if content:
                print(f"  [Token] {content}", end="", flush=True)

        elif kind == "on_tool_start":
            print(f"\n  [事件] 工具开始: {event['name']}")

        elif kind == "on_tool_end":
            print(f"  [事件] 工具结束")

        elif kind == "on_chat_model_end":
            print(f"\n  [事件] 模型结束")


# ========== 9. 流式聊天历史 ==========
def demo_streaming_with_history():
    """带对话历史的流式输出"""
    print("\n\n" + "=" * 70)
    print("9. 带对话历史的流式输出")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    messages = [
        SystemMessage(content="你是一个简洁的助手"),
        HumanMessage(content="我叫Alice"),
    ]

    print("\n第一轮:")
    print("  用户: 我叫Alice")
    print("  助手: ", end="", flush=True)

    full_response = ""
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content

    print()

    # 添加到历史
    from langchain_core.messages import AIMessage
    messages.append(AIMessage(content=full_response))
    messages.append(HumanMessage(content="我叫什么名字？"))

    print("\n第二轮:")
    print("  用户: 我叫什么名字？")
    print("  助手: ", end="", flush=True)

    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)

    print()


# ========== 10. 实际应用：实时代码生成 ==========
def demo_real_world_code_generation():
    """实际应用：实时代码生成"""
    print("\n\n" + "=" * 70)
    print("10. 实际应用：实时代码生成")
    print("=" * 70)

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = """
    编写一个Python函数，实现二分查找算法。
    要求：包含详细注释和类型提示。
    """

    print("\n📝 代码生成 (实时显示):")
    print("-" * 70)

    for chunk in llm.stream(prompt):
        print(chunk.content, end="", flush=True)

    print()
    print("-" * 70)
    print("\n💡 流式输出让用户看到代码生成过程，提供更好的体验")


# ========== 11. 最佳实践 ==========
def best_practices():
    """流式处理最佳实践"""
    print("\n\n" + "=" * 70)
    print("11. 最佳实践")
    print("=" * 70)

    print("""
🎯 何时使用流式输出:

✅ 推荐使用场景:
1. 长文本生成 (文章、代码、报告)
2. 聊天应用 (即时反馈)
3. 用户界面响应 (改善体验)
4. 实时数据处理

❌ 不推荐场景:
1. 结构化数据提取 (需要完整响应)
2. 批量处理 (性能不是瓶颈)
3. 需要完整响应才能处理的场景

💡 实现方式对比:

┌──────────────┬─────────────────┬──────────────────┐
│ 方法         │ 使用场景        │ 返回类型         │
├──────────────┼─────────────────┼──────────────────┤
│ invoke()     │ 等待完整响应    │ Message          │
│ stream()     │ 同步流式输出    │ Generator[Chunk] │
│ ainvoke()    │ 异步完整响应    │ Message          │
│ astream()    │ 异步流式输出    │ AsyncGen[Chunk]  │
│ astream_events() │ 详细事件流 │ AsyncGen[Event]  │
└──────────────┴─────────────────┴──────────────────┘

📝 代码模式:

# 1. 基础流式
for chunk in llm.stream(prompt):
    print(chunk.content, end="")

# 2. 异步流式
async for chunk in llm.astream(prompt):
    print(chunk.content, end="")

# 3. 流式 + 解析
for chunk in (llm | parser).stream(prompt):
    process(chunk)

# 4. 事件流 (调试用)
async for event in llm.astream_events(prompt, version="v2"):
    handle_event(event)

⚠️ 常见问题:

Q1: stream() 和 invoke() 结果一样吗?
A: 一样。stream() 只是逐步返回，最终内容相同。

Q2: 如何获取流式输出的完整文本?
A:
   full_text = ""
   for chunk in llm.stream(prompt):
       full_text += chunk.content

Q3: 流式输出会更快吗?
A: 总时间相同，但首个 token 更快 (TTFT↓)

Q4: 如何在Web应用中使用流式输出?
A: 使用 Server-Sent Events (SSE) 或 WebSocket

Q5: 异步流式比同步流式快吗?
A: 单个请求速度相同，但异步支持并发

🔍 性能指标:

- TTFT (Time To First Token): 流式输出优势
- 总延迟: 流式和非流式相同
- 并发性能: 异步流式最佳

🔗 参考资源:
- Streaming Guide: https://python.langchain.com/docs/how_to/streaming/
- Streaming Concepts: https://python.langchain.com/docs/concepts/streaming/
- astream_events: https://python.langchain.com/api_reference/core/runnables/
    """)


def main():
    """运行所有示例"""
    demo_basic_streaming()
    demo_streaming_comparison()
    demo_streaming_with_parser()
    demo_streaming_with_tools()
    demo_batch_streaming()

    # 异步示例需要在异步环境中运行
    print("\n\n🔄 运行异步示例...")
    asyncio.run(demo_async_streaming())
    asyncio.run(demo_concurrent_streaming())
    asyncio.run(demo_astream_events())

    demo_streaming_with_history()
    demo_real_world_code_generation()
    best_practices()

    print("\n\n" + "=" * 70)
    print("✅ Streaming 所有示例演示完成")
    print("=" * 70)
    print("\n核心要点:")
    print("  ✓ stream() - 同步流式输出")
    print("  ✓ astream() - 异步流式输出")
    print("  ✓ 改善用户体验 (即时反馈)")
    print("  ✓ 适用于长文本生成")
    print("  ✓ 支持工具调用流式输出")


if __name__ == "__main__":
    main()
