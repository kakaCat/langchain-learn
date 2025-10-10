import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


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

# 自定义键值记忆类
class KeyValueChatMessageHistory(BaseChatMessageHistory):
    """结合键值存储和消息历史的记忆类"""
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages
        # 添加键值存储字典
        self.key_value_store = {}
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息并保持消息数量限制"""
        self.messages.append(message)
        # 超过最大消息数时，删除最旧的消息
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def clear(self) -> None:
        """清除所有消息和键值存储"""
        self.messages = []
        self.key_value_store = {}
        
    def set_key_value(self, key: str, value: str) -> None:
        """存储键值对"""
        self.key_value_store[key] = value
    
    def get_key_value(self, key: str, default: str = None) -> str:
        """获取键对应的值"""
        return self.key_value_store.get(key, default)
    
    def get_all_key_values(self) -> dict:
        """获取所有键值对"""
        return self.key_value_store.copy()

# 会话存储
store: dict[str, BaseChatMessageHistory] = {}

# 获取会话历史
def get_session_history(session_id: str) -> KeyValueChatMessageHistory:
    """获取特定会话ID的聊天历史"""
    if session_id not in store:
        # 创建带消息数量限制和键值存储的历史记录
        store[session_id] = KeyValueChatMessageHistory(max_messages=3)
    # 类型断言，确保返回的是KeyValueChatMessageHistory类型
    return store[session_id]

def extract_and_store_key_values(conversation: list[BaseMessage], session_id: str) -> None:
    """从对话中提取并存储键值信息"""
    history = get_session_history(session_id)
    
    # 简单的规则来提取姓名和公司等信息
    for message in conversation:
        content = message.content.lower()
        if message.type == "human":  # 只从用户消息中提取
            # 提取姓名
            if "我叫" in content or "我的名字是" in content:
                name = content.split("我叫")[-1].split("。")[0].strip() if "我叫" in content else \
                       content.split("我的名字是")[-1].split("。")[0].strip()
                history.set_key_value("name", name)
            # 提取公司
            if "我在" in content and "工作" in content:
                company = content.split("我在")[-1].split("工作")[0].strip()
                history.set_key_value("company", company)
            # 提取其他可能的键值对
            if "我是" in content and not "我的名字是" in content:
                info = content.split("我是")[-1].split("。")[0].strip()
                history.set_key_value("occupation", info)

def main() -> None:
    try:
        # 加载环境变量
        load_environment()
        
        # 初始化语言模型
        llm = get_llm()
        
        # 会话ID
        session_id = "user_123"
        
        # 获取键值记忆实例
        history = get_session_history(session_id)
        
        # 创建包含键值信息的聊天提示模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手。以下是关于用户的重要信息：\n{user_info}\n请根据这些信息回答问题。"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])
        
        # 创建链
        chain = prompt | llm
        
        # 使用RunnableWithMessageHistory包装链
        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        # 测试对话
        print("===== 测试键值记忆功能 =====")
        
        # 第一轮对话
        user_message = "你好，我叫张三。"
        print(f"用户: {user_message}")
        
        # 获取当前键值信息
        key_values = history.get_all_key_values()
        user_info = "\n".join([f"{k}: {v}" for k, v in key_values.items()]) if key_values else "暂无信息"
        
        # 调用链
        config = {"configurable": {"session_id": session_id}}
        response = with_message_history.invoke({
            "input": user_message,
            "user_info": user_info  # 传递键值信息给提示模板
        }, config)
        
        print(f"AI响应: {response.content}")
        
        # 提取并存储键值对
        extract_and_store_key_values(get_session_history(session_id).messages, session_id)
        print(f"当前存储的键值对: {history.get_all_key_values()}")
        
        # 第二轮对话
        user_message = "我在微软工作。"
        print(f"用户: {user_message}")
        
        # 更新键值信息
        key_values = history.get_all_key_values()
        user_info = "\n".join([f"{k}: {v}" for k, v in key_values.items()]) if key_values else "暂无信息"
        
        response = with_message_history.invoke({
            "input": user_message,
            "user_info": user_info
        }, config)
        
        print(f"AI响应: {response.content}")
        
        # 提取并存储键值对
        extract_and_store_key_values(get_session_history(session_id).messages, session_id)
        print(f"更新后的键值对: {history.get_all_key_values()}")
        
        # 测试键值记忆 - 询问已存储的信息
        user_message = "我叫什么名字？我在哪里工作？"
        print(f"用户: {user_message}")
        
        key_values = history.get_all_key_values()
        user_info = "\n".join([f"{k}: {v}" for k, v in key_values.items()]) if key_values else "暂无信息"
        
        response = with_message_history.invoke({
            "input": user_message,
            "user_info": user_info
        }, config)
        
        print(f"AI响应: {response.content}")
        
        # 打印当前的历史记录，验证窗口限制
        print(f"\n当前历史记录（最多{history.max_messages}条消息）:")
        for msg in get_session_history(session_id).messages:
            print(f"{msg.type}: {msg.content}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查OPENAI_API_KEY等配置是否正确")


if __name__ == "__main__":
    main()