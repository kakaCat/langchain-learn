
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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

# 自定义有限历史记录类
class LimitedChatMessageHistory:
    """限制消息数量的聊天历史记录类"""
    def __init__(self, max_messages: int = 10):
        self.messages = []
        self.max_messages = max_messages  # 明确设置max_messages属性
        print(f"初始化LimitedChatMessageHistory，最大消息数: {self.max_messages}")
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息并保持消息数量限制"""
        self.messages.append(message)
        # 超过最大消息数时，删除最旧的消息
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def clear(self) -> None:
        """清除所有消息"""
        self.messages = []
    
    @property
    def messages(self):
        """获取消息列表"""
        return self._messages
    
    @messages.setter
    def messages(self, value):
        """设置消息列表"""
        self._messages = value

# 使用带限制的历史记录
store = {}

def get_limited_history(session_id: str):
    if session_id not in store:
        print(f"为session_id {session_id} 创建新的LimitedChatMessageHistory实例")
        # 直接使用我们自定义的类，不继承ChatMessageHistory
        store[session_id] = LimitedChatMessageHistory(max_messages=4)  # 只保留4条消息
    return store[session_id]

def main() -> None:
    try:
        # 加载环境变量
        load_environment()
        
        session_id = "user_123"
        history = get_limited_history(session_id)
        
        # 测试添加消息
        print("添加测试消息...")
        history.add_message(HumanMessage(content="你好，我是张三"))
        history.add_message(AIMessage(content="你好张三，我是AI助手"))
        
        # 验证消息数量
        print(f"当前消息数量: {len(history.messages)}")
        print(f"消息内容: {[msg.content for msg in history.messages]}")
        
        # 测试消息数量限制
        print("测试消息数量限制...")
        for i in range(5):
            history.add_message(HumanMessage(content=f"测试消息 {i}"))
            history.add_message(AIMessage(content=f"测试响应 {i}"))
            print(f"添加消息后数量: {len(history.messages)}")
            print(f"保留的消息: {[msg.content for msg in history.messages]}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查OPENAI_API_KEY等配置是否正确")


if __name__ == "__main__":
    main()