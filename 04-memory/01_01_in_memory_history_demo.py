from __future__ import annotations

import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def load_environment() -> None:
    """
    从当前模块目录加载环境变量配置文件
    
    Raises:
        FileNotFoundError: 当.env文件不存在时抛出
        Exception: 其他加载环境变量时发生的异常
    """
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        print(f"正在加载环境变量文件: {env_path}")
        load_dotenv(dotenv_path=env_path, override=False)
        print("环境变量加载成功")
    except FileNotFoundError:
        print("警告: .env文件未找到，将使用系统环境变量")
    except Exception as e:
        print(f"加载环境变量时发生错误: {e}")
        raise


def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例
    
    Returns:
        ChatOpenAI: 配置好的语言模型实例
        
    Raises:
        ValueError: 当OPENAI_API_KEY未设置时抛出
        Exception: 其他配置模型时发生的异常
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY环境变量未设置，请检查.env文件或系统环境变量")
            
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

        print(f"正在配置语言模型: {model}")
        print(f"温度参数: {temperature}, 最大令牌数: {max_tokens}")
        
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
        
        llm = ChatOpenAI(**kwargs)
        print("语言模型配置成功")
        return llm
        
    except ValueError as e:
        print(f"配置验证错误: {e}")
        raise
    except Exception as e:
        print(f"配置语言模型时发生错误: {e}")
        raise


# 内存存储（仅进程内，不持久化）
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_in_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    使用内存存储获取或创建聊天历史记录
    
    Args:
        session_id: 会话标识符
        
    Returns:
        InMemoryChatMessageHistory: 对应会话的聊天历史记录
        
    Raises:
        ValueError: 当session_id为空或无效时抛出
        Exception: 其他创建历史记录时发生的异常
    """
    try:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id不能为空且必须为字符串类型")
            
        if session_id not in _store:
            print(f"为会话 {session_id} 创建新的内存历史记录")
            _store[session_id] = InMemoryChatMessageHistory()
        else:
            print(f"获取会话 {session_id} 的现有内存历史记录")
            
        return _store[session_id]
    except ValueError as e:
        print(f"参数验证错误: {e}")
        raise
    except Exception as e:
        print(f"获取内存历史记录时发生错误: {e}")
        raise


def create_conversation_chain() -> RunnableWithMessageHistory:
    """
    创建带内存历史记录的会话链
    
    Returns:
        RunnableWithMessageHistory: 配置好的带历史记录的会话链
        
    Raises:
        Exception: 创建会话链过程中发生的任何异常
    """
    try:
        print("正在创建带内存历史记录的会话链...")
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
            get_session_history=get_in_memory_history,
            input_messages_key="input",
            history_messages_key="history",
            verbose=True,
        )
        print("会话链创建成功")
        return conversation
    except Exception as e:
        print(f"创建会话链时发生错误: {e}")
        raise


def run_conversation_example(session_id: str = "memory_demo") -> None:
    """
    运行内存存储的对话示例
    
    Args:
        session_id: 会话标识符，默认为"memory_demo"
        
    Raises:
        Exception: 运行对话示例过程中发生的任何异常
    """
    try:
        print("正在启动内存存储对话示例...")
        langchain.debug = True
        conversation = create_conversation_chain()

        print(f"\n=== 开始对话示例 (存储类型: memory, 会话ID: {session_id}) ===")

        # 第一轮对话
        response = conversation.invoke({"input": "你好，我叫张三。"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        # 第二轮对话 - 测试记忆功能
        response = conversation.invoke({"input": "我刚才告诉你我叫什么名字？"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        # 第三轮对话 - 测试生成能力
        response = conversation.invoke({"input": "能帮我生成一个简短的自我介绍吗？"}, config={"configurable": {"session_id": session_id}})
        print(f"AI: {response.content}")

        print("\n=== 对话示例结束 ===")
        langchain.debug = False
        print("对话示例运行成功")
        
    except Exception as e:
        print(f"运行对话示例时发生错误: {e}")
        print("请检查以下可能的问题:")
        print("1. 环境变量配置是否正确")
        print("2. API密钥是否有效")
        print("3. 网络连接是否正常")
        raise


def main() -> None:
    """主函数：运行内存存储示例"""
    run_conversation_example()


if __name__ == "__main__":
    main()