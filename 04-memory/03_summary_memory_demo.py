import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug


# 从当前模块目录加载 .env
def load_environment():
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

# 会话存储
store = {}

def get_session_history(session_id: str):
    """根据session_id获取或创建对应的聊天历史记录"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()  # 使用内存存储
    return store[session_id]

def create_summary_chain():
    """创建带摘要功能的会话链"""
    # 创建模型
    llm = get_llm()
    
    # 构建提示模板，包含系统指令、历史消息和用户输入
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手，能够理解并总结对话内容。"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息将动态注入于此
        ("human", "{input}"),
    ])
    
    # 创建基本链
    chain = prompt | llm
    
    # 使用RunnableWithMessageHistory包装链，实现对话历史管理
    conversation = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=get_session_history,
        input_messages_key="input",  # 指定输入信息在链中的key
        history_messages_key="history",  # 指定历史信息在提示模板中的key
        verbose=True
    )
    
    return conversation

def main() -> None:
    try:
        set_debug(True)

        load_environment()
        
        # 创建带历史记忆的会话链
        conversation = create_summary_chain()
        
        # 模拟多轮对话
        session_id = "test_session"
        
        # 第一轮对话
        response = conversation.invoke(
            {"input": "你好，我叫张三。"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        # 第二轮对话
        response = conversation.invoke(
            {"input": "我是一名软件工程师。"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        # 第三轮对话
        response = conversation.invoke(
            {"input": "我喜欢编程和机器学习。"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI: {response.content}")
        
        # 第四轮对话 - 测试记忆和摘要能力
        response = conversation.invoke(
            {"input": "请总结一下我们的对话内容。"},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"AI摘要: {response.content}")
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查 OPENAI_API_KEY 与 OPENAI_MODEL 是否已配置。")


if __name__ == "__main__":
    main()