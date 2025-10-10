import os
from typing import Dict, List, Set, Tuple, Any

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

# 会话存储
store: Dict[str, BaseChatMessageHistory] = {}
# 实体存储
entity_store: Dict[str, Dict[str, str]] = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 提取实体的提示模板
ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个实体提取助手。请从以下对话中提取实体及其相关信息。请以JSON格式输出，其中键是实体名称，值是关于该实体的信息。如果没有实体，请返回空对象。"),
    ("human", "对话: {conversation}")
])

# 提取实体
def extract_entities(conversation: List[BaseMessage], session_id: str) -> Dict[str, str]:
    """从对话中提取实体信息"""
    llm = get_llm()
    
    # 构建对话字符串
    conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation])
    
    # 提取实体
    entity_chain = ENTITY_EXTRACTION_PROMPT | llm
    response = entity_chain.invoke({"conversation": conversation_str})
    
    try:
        # 尝试解析JSON响应
        import json
        entities = json.loads(response.content)
    except json.JSONDecodeError:
        # 如果解析失败，使用空字典
        entities = {}
    
    # 更新实体存储
    if session_id not in entity_store:
        entity_store[session_id] = {}
    entity_store[session_id].update(entities)
    
    return entities

# 创建带实体记忆的对话链
# 修改create_entity_aware_chain函数
def create_entity_aware_chain(session_id: str):
    """创建能够识别和利用实体信息的对话链"""
    llm = get_llm()
    
    # 构建包含实体信息的提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的AI助手。以下是对话中提到的重要实体信息：\n{entity_info}\n请根据这些信息回答问题。"),
        ("human", "{input}"),
    ])
    
    # 获取实体信息
    entity_info = "\n".join([f"{entity}: {info}" for entity, info in 
                           entity_store.get(session_id, {}).items()]) or "暂无实体信息"
    
    # 创建链 - 不使用bind方法，而是在调用时传递参数
    chain = prompt | llm
    
    return chain, entity_info

# 修改main函数中的调用部分
def main() -> None:
    try:
        # 加载环境变量
        load_environment()
        session_id = "user_123"
        
        # 创建会话历史
        history = get_session_history(session_id)
        
        # 添加初始对话
        user_message = "我叫张三，我在微软工作。"
        history.add_user_message(user_message)
        
        # 提取实体
        entities = extract_entities(history.messages, session_id)
        print(f"识别的实体：{entities}")
        
        # 创建实体感知的对话链 - 获取chain和entity_info
        conversation_chain, entity_info = create_entity_aware_chain(session_id)
        
        # 获取AI响应 - 直接传递entity_info参数
        response = conversation_chain.invoke({"input": user_message, "entity_info": entity_info})
        ai_response = response.content
        history.add_ai_message(ai_response)
        
        print(f"AI响应: {ai_response}")
        
        # 第二轮对话测试实体记忆
        second_user_message = "我在哪里工作？"
        history.add_user_message(second_user_message)
        
        # 更新实体存储
        extract_entities(history.messages, session_id)
        
        # 更新对话链以包含最新实体信息
        conversation_chain, updated_entity_info = create_entity_aware_chain(session_id)
        
        # 获取第二轮AI响应 - 直接传递更新的entity_info参数
        second_response = conversation_chain.invoke({"input": second_user_message, "entity_info": updated_entity_info})
        second_ai_response = second_response.content
        history.add_ai_message(second_ai_response)
        
        print(f"AI响应: {second_ai_response}")
        
        # 打印最终实体存储
        print(f"最终实体存储：{entity_store.get(session_id, {})}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查OPENAI_API_KEY等配置是否正确")

if __name__ == "__main__":
    main()