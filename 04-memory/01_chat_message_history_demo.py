from __future__ import annotations

import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    ChatMessageHistory, 
    RedisChatMessageHistory,
    SQLChatMessageHistory,
    FileChatMessageHistory
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


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
# 内存存储
store = {}

# 1. 不同类型的历史记录存储实现

def get_in_memory_history(session_id: str):
    """使用内存存储聊天历史记录"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    """使用Redis存储聊天历史记录"""
    return RedisChatMessageHistory(
        session_id,
        url="redis://localhost:6379/0",
        ttl=600  # 10分钟过期
    )

def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    """使用SQL数据库存储聊天历史记录"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )

def get_file_history(session_id: str) -> FileChatMessageHistory:
    """使用文件存储聊天历史记录"""
    return FileChatMessageHistory(f"chat_history_{session_id}.json")





def create_conversation_chain(store_type: str = "memory"):

    load_environment()
    # 2. 初始化语言模型
    llm = get_llm()

    """创建带不同记忆存储类型的会话链"""
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，能够记住用户之前说过的话。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # 3. 创建带记忆的会话链
    chain = prompt | llm
    
    # 根据存储类型选择历史记录获取函数
    history_getter = {
        "memory": get_in_memory_history,
        "redis": get_redis_history,
        "sql": get_sql_history,
        "file": get_file_history
    }.get(store_type, get_in_memory_history)
    
    # 包装为带历史记录的会话链
    conversation = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=history_getter,
        input_messages_key="input",
        history_messages_key="history",
        verbose=True
    )
    
    return conversation

# 4. 运行对话示例

def run_conversation_example(store_type: str = "memory", session_id: str = "user_123"):
    """运行对话示例"""
    try:
        # 启用调试模式
        langchain.debug = True
        
        # 创建会话链
        conversation = create_conversation_chain(store_type)
        
        print(f"\n=== 开始对话示例 (存储类型: {store_type}, 会话ID: {session_id}) ===")
        
        # 第一回合对话
        response = conversation.invoke(
            {"input": "你好，我叫张三。"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        # 第二回合对话（测试记忆）
        response = conversation.invoke(
            {"input": "我刚才告诉你我叫什么名字？"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        # 第三回合对话（测试上下文理解）
        response = conversation.invoke(
            {"input": "能帮我生成一个简短的自我介绍吗？"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        print(f"\n=== 对话示例结束 ===")
        
        # 关闭调试模式
        langchain.debug = False
        
    except Exception as e:
        print(f"发生错误：{e}")
        print("请检查环境配置和依赖是否正确。")

# 5. 测试不同的存储类型

def test_all_storage_types():
    """测试所有的存储类型"""
    # 测试内存存储
    run_conversation_example("memory", "memory_test")
    
    # 注意：以下存储类型需要相应的服务或会创建文件
    # 测试Redis存储 (需要本地运行Redis服务)
    # run_conversation_example("redis", "redis_test")
    
    # 测试SQL存储 (会创建sqlite数据库文件)
    run_conversation_example("sql", "sql_test")
    
    # 测试文件存储 (会创建JSON文件)
    run_conversation_example("file", "file_test")

def main() -> None:
    """主函数"""
    # 运行默认的对话示例
    run_conversation_example()

if __name__ == "__main__":
    main()