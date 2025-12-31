"""
LangChain 1.0 - 消息 (Messages)

内容概述:
- HumanMessage - 用户消息
- AIMessage - AI 回复
- SystemMessage - 系统提示
- FunctionMessage - 函数调用结果
- 消息历史管理

参考文档:
- https://docs.langchain.com/oss/python/langchain/messages
- https://python.langchain.com/api_reference/core/messages/
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage
)

load_dotenv()


# ========== 基础示例: 消息类型 ==========
def demo_message_types():
    """演示不同的消息类型"""
    print("=" * 60)
    print("消息类型示例")
    print("=" * 60)

    # 1. SystemMessage - 设定AI行为
    system_msg = SystemMessage(content="你是一个专业的Python教师")
    print(f"\n1️⃣  SystemMessage: {system_msg.content}")

    # 2. HumanMessage - 用户输入
    human_msg = HumanMessage(content="什么是装饰器？")
    print(f"2️⃣  HumanMessage: {human_msg.content}")
    human_metadata_msg = HumanMessage(
        content="Hello!",
        name="alice",  # Optional: identify different users
        id="msg_123",  # Optional: unique identifier for tracing
    )
    print(f"2️⃣  HumanMessage: {human_metadata_msg.content}")
    print(f"2️⃣  HumanMessage: {human_metadata_msg.name}")
    print(f"2️⃣  HumanMessage: {human_metadata_msg.id}")

    # 3. AIMessage - AI回复
    ai_msg = AIMessage(content="装饰器是Python中的一种设计模式...")
    print(f"3️⃣  AIMessage: {ai_msg.content}")

    print("\n✅ 三种基本消息类型")


# ========== 构建对话历史 ==========
def demo_conversation_history():
    """构建和管理对话历史"""
    print("\n\n" + "=" * 60)
    print("对话历史管理")
    print("=" * 60)

    model = ChatOpenAI(model="gpt-4o-mini")

    # 初始化对话历史
    messages = [
        SystemMessage(content="你是一个友好的助手，用简洁的语言回答问题"),
    ]

    # 第一轮对话
    messages.append(HumanMessage(content="你好，我叫Alice"))
    response1 = model.invoke(messages)
    messages.append(AIMessage(content=response1.content))

    print("\n第一轮:")
    print(f"👤 用户: {messages[-2].content}")
    print(f"🤖 助手: {messages[-1].content}")

    # 第二轮对话 - 测试记忆
    messages.append(HumanMessage(content="我刚才说我叫什么名字？"))
    response2 = model.invoke(messages)
    messages.append(AIMessage(content=response2.content))

    print("\n第二轮:")
    print(f"👤 用户: {messages[-2].content}")
    print(f"🤖 助手: {messages[-1].content}")

    print(f"\n📊 对话历史长度: {len(messages)} 条消息")


# ========== 消息属性 ==========
def demo_message_properties():
    """消息的属性和元数据"""
    print("\n\n" + "=" * 60)
    print("消息属性")
    print("=" * 60)

    # 创建带元数据的消息
    message = HumanMessage(
        content="这是消息内容",
        additional_kwargs={
            "user_id": "user_123",
            "timestamp": "2024-12-19",
            "priority": "high"
        }
    )

    print(f"\n内容: {message.content}")
    print(f"类型: {message.type}")  # 'human'
    print(f"额外信息: {message.additional_kwargs}")

    # AI消息可能包含函数调用
    ai_message = AIMessage(
        content="我需要调用工具",
        additional_kwargs={
            "function_call": {
                "name": "search_database",
                "arguments": '{"query": "用户数据"}'
            }
        }
    )

    print(f"\n\nAI消息内容: {ai_message.content}")
    print(f"函数调用: {ai_message.additional_kwargs.get('function_call')}")


# ========== 多模态消息 ==========
def demo_multimodal_messages():
    """多模态消息（文本+图片）"""
    print("\n\n" + "=" * 60)
    print("多模态消息")
    print("=" * 60)

    # 文本+图片消息（OpenAI GPT-4o 支持）
    message = HumanMessage(
        content=[
            {"type": "text", "text": "这张图片里有什么？"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.jpg"}
            }
        ]
    )

    print(f"\n消息类型: 多模态")
    print(f"包含: 文本 + 图片URL")
    print("\n💡 GPT-4o 和 GPT-4-vision 支持图片输入")


# ========== 消息格式化 ==========
def demo_message_formatting():
    """格式化和显示消息"""
    print("\n\n" + "=" * 60)
    print("消息格式化")
    print("=" * 60)

    messages = [
        SystemMessage(content="你是助手"),
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮助你的？"),
        HumanMessage(content="介绍一下Python"),
        AIMessage(content="Python是一种高级编程语言...")
    ]

    print("\n对话历史:")
    print("-" * 60)
    for i, msg in enumerate(messages, 1):
        role = {
            'system': '⚙️  系统',
            'human': '👤 用户',
            'ai': '🤖 助手'
        }.get(msg.type, msg.type)

        print(f"{i}. {role}: {msg.content[:50]}...")

    print(f"\n总计: {len(messages)} 条消息")


# ========== 消息过滤和处理 ==========
def demo_message_filtering():
    """过滤和处理消息列表"""
    print("\n\n" + "=" * 60)
    print("消息过滤")
    print("=" * 60)

    messages = [
        SystemMessage(content="系统设置"),
        HumanMessage(content="问题1"),
        AIMessage(content="回答1"),
        HumanMessage(content="问题2"),
        AIMessage(content="回答2"),
        HumanMessage(content="问题3"),
    ]

    # 只保留用户消息
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    print(f"\n用户消息数量: {len(human_messages)}")
    for msg in human_messages:
        print(f"  - {msg.content}")

    # 只保留AI消息
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    print(f"\nAI消息数量: {len(ai_messages)}")

    # 获取最近N条消息
    recent_messages = messages[-3:]
    print(f"\n最近3条消息:")
    for msg in recent_messages:
        print(f"  - {msg.type}: {msg.content}")


# ========== 消息转换 ==========
def demo_message_conversion():
    """消息格式转换"""
    print("\n\n" + "=" * 60)
    print("消息转换")
    print("=" * 60)

    messages = [
        HumanMessage(content="你好"),
        AIMessage(content="你好！")
    ]

    # 转换为字典格式
    print("\n字典格式:")
    for msg in messages:
        msg_dict = {
            "role": msg.type,
            "content": msg.content
        }
        print(f"  {msg_dict}")

    # 转换为 OpenAI 格式
    print("\nOpenAI API 格式:")
    openai_messages = []
    for msg in messages:
        role_map = {
            'human': 'user',
            'ai': 'assistant',
            'system': 'system'
        }
        openai_messages.append({
            "role": role_map.get(msg.type, msg.type),
            "content": msg.content
        })
        print(f"  {openai_messages[-1]}")


# ========== 消息链 ==========
def demo_message_chain():
    """消息链和批量处理"""
    print("\n\n" + "=" * 60)
    print("消息链")
    print("=" * 60)

    model = ChatOpenAI(model="gpt-4o-mini")

    # 准备多个对话
    conversations = [
        [HumanMessage(content="1+1等于几？")],
        [HumanMessage(content="Python之父是谁？")],
        [HumanMessage(content="什么是AI？")]
    ]

    print("\n批量处理多个对话:")
    responses = model.batch(conversations)

    for i, response in enumerate(responses, 1):
        print(f"\n对话 {i}:")
        print(f"  问题: {conversations[i-1][0].content}")
        print(f"  回答: {response.content[:50]}...")


# ========== 最佳实践 ==========
def best_practices():
    """最佳实践"""
    print("\n\n" + "=" * 60)
    print("最佳实践")
    print("=" * 60)

    print("""
🎯 消息使用建议:

1. **SystemMessage**
   - 放在对话开始
   - 设定AI的角色和行为
   - 定义输出格式

2. **HumanMessage**
   - 用户的每次输入
   - 可以包含多模态内容
   - 保持对话上下文

3. **AIMessage**
   - AI的回复
   - 保存到对话历史
   - 可能包含函数调用

💡 对话历史管理:
- 限制历史长度（最近N条）
- 使用 SummarizationMiddleware 自动摘要
- 重要信息提取到 SystemMessage

⚠️ 常见问题:

Q1: 对话历史太长怎么办？
A: 1) 只保留最近N条消息
   2) 使用摘要 middleware
   3) 提取关键信息到 system prompt

Q2: 如何处理多模态输入？
A: 使用 content 数组格式:
   [{"type": "text", ...}, {"type": "image_url", ...}]

Q3: 如何保存对话历史？
A: 1) 使用 Checkpointer (推荐)
   2) 保存到数据库
   3) 序列化为 JSON

Q4: 消息格式如何转换？
A: LangChain 消息 ↔ OpenAI API 格式
   使用 role 映射: human→user, ai→assistant

🔗 参考资源:
- Messages API: https://docs.langchain.com/oss/python/langchain/messages
- OpenAI Messages: https://platform.openai.com/docs/api-reference/chat
    """)


def main():
    """运行所有示例"""
    demo_message_types()
    demo_conversation_history()
    demo_message_properties()
    demo_multimodal_messages()
    demo_message_formatting()
    demo_message_filtering()
    demo_message_conversion()
    demo_message_chain()
    best_practices()

    print("\n\n" + "=" * 60)
    print("✅ 所有消息示例演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
