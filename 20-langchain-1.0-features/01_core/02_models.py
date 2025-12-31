"""
LangChain 1.0 - 模型 (Models)

内容概述:
- 基础模型调用
- 流式输出 (Streaming)
- 多模型支持
- 模型参数配置
- 批量处理

参考文档:
- https://docs.langchain.com/oss/python/langchain/models
- https://python.langchain.com/api_reference/openai/chat_models/
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool
load_dotenv()


# ========== 辅助函数 ==========
def get_llm(model_name: str = None, **kwargs):
    """
    创建并配置语言模型实例 (统一初始化方法)

    Args:
        model_name: 模型名称，默认使用环境变量
        **kwargs: 其他模型参数 (temperature, max_tokens 等)

    Returns:
        配置好的模型实例
    """
    api_key = os.getenv("DEEPSEEK_KEY")
    model = model_name or os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise ValueError("DEEPSEEK_KEY 未设置")
    if not base_url:
        raise ValueError("DEEPSEEK_BASE_URL 未设置")

    # 合并参数
    
    # model 字符串必填 您希望与提供商一起使用的特定模型的名称或标识符。
    # api_key 字符串 用于与模型提供商进行身份验证所需的密钥。通常在您注册访问模型时颁发。通常通过设置环境变量来访问。环境变量.
    # temperature 数字 控制模型输出的随机性。数字越高，响应越有创意；数字越低，响应越确定。
    # timeout 数字 在取消请求之前，等待模型响应的最长时间（以秒为单位）。
    # max_tokens 数字 限制响应中token的总数，有效控制输出的长度。token在响应中，有效控制输出的长度。
    # max_retries 数字 如果请求因网络超时或速率限制等问题而失败，系统将尝试重新发送请求的最大次数。

    # 限流
    # rate_limiter = InMemoryRateLimiter(
    # requests_per_second=0.1,  # 1 request every 10s
    # check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    # max_bucket_size=10,  # Controls the maximum burst size.
    # )
    params = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs  # 允许覆盖默认参数
    }

    return init_chat_model(**params)


# ========== 基础示例: 模型调用 ==========
def demo_basic_model():
    """最简单的模型调用"""
    print("=" * 60)
    print("基础示例: 模型调用")
    print("=" * 60)

    # 创建模型实例
    model = get_llm(temperature=0.7, max_tokens=100)

    # 调用模型 支持数组模式
    # 消息字典
    # conversation = [
    # {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    # {"role": "user", "content": "Translate: I love programming."},
    # {"role": "assistant", "content": "J'adore la programmation."},
    # {"role": "user", "content": "Translate: I love building applications."}
    # ]
    # 消息对象
    # conversation = [
    # SystemMessage("You are a helpful assistant that translates English to French."),
    # HumanMessage("Translate: I love programming."),
    # AIMessage("J'adore la programmation."),
    # HumanMessage("Translate: I love building applications.")
    # ]


    response = model.invoke(conversation)
    print(f"\n回复: {response.content}")
    print(f"模型: {response.response_metadata.get('model_name', 'unknown')}")
    print(f"Token使用: {response.response_metadata.get('token_usage', {})}")


# ========== 流式输出 ==========
def demo_streaming():
    """实时流式输出"""
    print("\n\n" + "=" * 60)
    print("流式输出示例")
    print("=" * 60)

    model = get_llm(
        temperature=0.7,
        streaming=True  # 启用流式输出
    )

    print("\n输出: ", end="", flush=True)

    # stream() 方法返回生成器
    for chunk in model.stream("写一首关于AI的五言绝句"):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n\n✅ 流式输出完成")


# ========== 批量处理 ==========
def demo_batch_processing():
    """批量处理多个请求"""
    print("\n\n" + "=" * 60)
    print("批量处理示例")
    print("=" * 60)

    model = get_llm(temperature=0)

    # 批量请求
    queries = [
        "Python的优点是什么？",
        "JavaScript的优点是什么？",
        "Go语言的优点是什么？"
    ]

    print("\n批量请求:")
    for i, query in enumerate(queries, 1):
        print(f"  {i}. {query}")

    # 使用 batch() 方法
    responses = model.batch(queries)

    print("\n批量响应:")
    for i, response in enumerate(responses, 1):
        print(f"  {i}. {response.content[:50]}...")


# ========== 带消息历史的调用 ==========
def demo_with_message_history():
    """使用消息历史"""
    print("\n\n" + "=" * 60)
    print("消息历史示例")
    print("=" * 60)

    model = get_llm()

    # 构建对话历史
    messages = [
        SystemMessage(content="你是一个专业的Python教师"),
        HumanMessage(content="什么是列表推导式？"),
    ]

    # 第一次调用
    response1 = model.invoke(messages)
    print(f"\n🤖 助手: {response1.content}")

    # 添加到历史
    messages.append(AIMessage(content=response1.content))
    messages.append(HumanMessage(content="能给个例子吗？"))

    # 第二次调用（带历史）
    response2 = model.invoke(messages)
    print(f"\n🤖 助手: {response2.content}")

# ========== 工具调用 ==========

@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."

def demo_multiple_models():
    model = get_llm(temperature=0.7, max_tokens=100)

    model_with_tools = model.bind_tools([get_weather])  

    response = model_with_tools.invoke("What's the weather like in Boston?")
    for tool_call in response.tool_calls:
        # View tool calls made by the model
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")


# ========== 多模型支持 ==========
def demo_multiple_models():
    """使用不同的模型"""
    print("\n\n" + "=" * 60)
    print("多模型支持")
    print("=" * 60)

    # DeepSeek Chat (快速、便宜)
    model_mini = get_llm()

    # DeepSeek Reasoner (更强大) - 示例，实际使用相同模型
    # 如需使用不同模型，传入 model_name 参数
    model_4o = get_llm()

    prompt = "用一句话解释量子计算"

    print("\n📍 DeepSeek Chat:")
    response_mini = model_mini.invoke(prompt)
    print(f"   {response_mini.content}")

    print("\n📍 模型对比:")
    response_4o = model_4o.invoke(prompt)
    print(f"   {response_4o.content}")

    print("\n💡 不同模型提供不同的速度/质量权衡")
    print("   可通过 get_llm(model_name='模型名') 使用不同模型")


# ========== 成本计算 ==========
def demo_cost_tracking():
    """追踪API调用成本"""
    print("\n\n" + "=" * 60)
    print("成本追踪")
    print("=" * 60)

    from langchain_core.callbacks import UsageMetadataCallbackHandler

    model = get_llm()
    callback = UsageMetadataCallbackHandler()
    response = model.invoke("Hello", config={"callbacks": [callback]})
    callback.usage_metadata

# ========== 配置模型 ==========
def demo_config_model():
    """模型切换"""
    print("\n\n" + "=" * 60)
    print("成本追踪")
    print("=" * 60)
    model = get_llm()
    model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
    )
    model.invoke(
        "what's your name",
        config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
    )
        

def main():
    """运行所有示例"""
    demo_basic_model()
    demo_streaming()
    demo_batch_processing()
    demo_with_message_history()
    demo_multiple_models()
    demo_cost_tracking()
    demo_config_model()

    print("\n\n" + "=" * 60)
    print("✅ 所有模型示例演示完成")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio

    # 运行同步示例
    main()

