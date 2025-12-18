import os
import tiktoken
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug



# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

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

# 会话存储
store: Dict[str, BaseChatMessageHistory] = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 自定义令牌计数函数
def count_tokens(messages: List[BaseMessage], model: str = "gpt-3.5-turbo") -> int:
    """计算消息的令牌数量"""
    try:
        # 使用tiktoken库进行令牌计数
        encoding = tiktoken.encoding_for_model(model)
        buffer_string = get_buffer_string(messages)
        return len(encoding.encode(buffer_string))
    except KeyError:
        # 如果模型不支持，使用近似计数
        return sum(len(str(msg.content)) // 4 for msg in messages)

# 实现简单的令牌压缩逻辑
def compress_messages_if_needed(session_id: str, max_tokens: int = 1000):
    """如果消息令牌数超过限制，保留最新消息并生成摘要"""
    history = get_session_history(session_id)
    current_tokens = count_tokens(history.messages)
    
    if current_tokens > max_tokens:
        # 保留最新的几条消息
        keep_recent = 2
        if len(history.messages) > keep_recent:
            # 提取要总结的旧消息
            messages_to_summarize = history.messages[:-keep_recent]
            
            # 生成摘要
            llm = get_llm()
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", "请为以下对话生成简洁的摘要："),
                ("human", "{conversation}")
            ])
            
            conversation_str = get_buffer_string(messages_to_summarize)
            summary_chain = summary_prompt | llm
            summary = summary_chain.invoke({"conversation": conversation_str})
            
            # 重置历史，保留摘要和最新消息
            history.clear()
            history.add_user_message(f"过往对话摘要：{summary.content}")
            history.add_messages(history.messages[-keep_recent:])

            print(f"已压缩历史消息，当前令牌数: {count_tokens(history.messages)}")


def create_conversation_chain():
    """创建带历史记忆和令牌压缩功能的会话链"""
    # 创建模型
    llm = get_llm()

    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 创建基本链
    chain = prompt | llm

    # 使用RunnableWithMessageHistory集成记忆
    conversation = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return conversation

def main() -> None:
    try:
        set_debug(True)
        # 加载环境变量
        load_environment()
        
        # 创建会话链
        conversation = create_conversation_chain()
        session_id = "user_123"

        # 第一轮对话
        response = conversation.invoke(
            {"input": "你好，我叫张三。"},
            config={"configurable": {"session_id": session_id}},
            verbose=True
        ) 
        print(f"AI响应: {response.content}")
        
        # 检查并压缩消息
        compress_messages_if_needed(session_id, max_tokens=1000)

        # 第二轮对话，检查记忆功能
        response = conversation.invoke(
            {"input": "我刚才告诉你我叫什么名字？"},
            config={"configurable": {"session_id": session_id}},
            verbose=True
        )
        print(f"AI响应: {response.content}")

        # 打印当前记忆内容
        print(f"\n当前记忆内容:")
        print(get_session_history(session_id).messages)

    except Exception as e:
        print(f"错误: {e}")
        print("请检查OPENAI_API_KEY等配置是否正确")

if __name__ == "__main__":
    main()