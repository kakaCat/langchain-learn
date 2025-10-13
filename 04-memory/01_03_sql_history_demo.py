from __future__ import annotations

import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment() -> None:
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
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)


def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    """使用 SQL 数据库存储聊天历史记录（SQLite）"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db",
    )


def create_conversation_chain() -> RunnableWithMessageHistory:
    """创建带 SQL 历史记录的会话链"""
    load_environment()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手，能够记住用户之前说过的话。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    conversation = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_sql_history,
        input_messages_key="input",
        history_messages_key="history",
        verbose=True,
    )
    return conversation


def run_conversation_example(session_id: str = "sql_demo") -> None:
    """运行 SQL 存储的对话示例（会创建本地 sqlite 数据库文件）"""
    try:
        langchain.debug = True
        conversation = create_conversation_chain()

        print(f"\n=== 开始对话示例 (存储类型: sql, 会话ID: {session_id}) ===")

        response = conversation.invoke({"input": "你好，我叫张三。"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        response = conversation.invoke({"input": "我刚才告诉你我叫什么名字？"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        response = conversation.invoke({"input": "能帮我生成一个简短的自我介绍吗？"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        print("\n=== 对话示例结束 ===")
        langchain.debug = False
    except Exception as e:
        print(f"发生错误：{e}")
        print("请确认运行目录写入权限正常，或检查环境配置与依赖。")


def main() -> None:
    """主函数：运行 SQL 存储示例"""
    run_conversation_example()


if __name__ == "__main__":
    main()