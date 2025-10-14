#!/usr/bin/env python3
"""
Module 5: Tools Chatbot Demo
演示如何在LangChain中使用工具功能
"""
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Union

sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 从当前模块目录加载 .env
def load_environment():
    """加载环境变量"""
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

# 定义一个简单的计算器工具
@tool
def calculator(a: float, b: float, operation: str) -> float:
    """
    用于执行基本数学计算的工具。
    
    参数:
    a: 第一个数字
    b: 第二个数字
    operation: 操作类型，可以是 'add'(加), 'subtract'(减), 'multiply'(乘), 'divide'(除)
    
    返回:
    计算结果
    """
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            raise ValueError("除数不能为零")
        return a / b
    else:
        raise ValueError(f"不支持的操作: {operation}")

# 定义一个日期查询工具
@tool
def get_current_date(format: str = "%Y-%m-%d") -> str:
    """
    获取当前日期。
    
    参数:
    format: 日期格式字符串，默认为 '%Y-%m-%d'(年-月-日)
    
    返回:
    当前日期的字符串表示
    """
    return datetime.now().strftime(format)

# 定义一个天气查询工具（模拟）
@tool
def get_weather(city: str, date: str = None) -> Dict[str, Union[str, float]]:
    """
    获取指定城市的天气信息。
    
    参数:
    city: 城市名称（中文）
    date: 日期，格式为 YYYY-MM-DD，如果为None则获取当前日期天气
    
    返回:
    包含天气信息的字典，包括温度、天气状况等
    """
    # 这是一个模拟工具，实际应用中可以连接到真实的天气API
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 模拟一些城市的天气数据
    weather_data = {
        "北京": {"temperature": 25, "condition": "晴", "wind": "3-4级"},
        "上海": {"temperature": 28, "condition": "多云", "wind": "2-3级"},
        "广州": {"temperature": 32, "condition": "雷阵雨", "wind": "3-4级"},
        "深圳": {"temperature": 31, "condition": "多云转晴", "wind": "2-3级"},
        "杭州": {"temperature": 27, "condition": "晴", "wind": "1-2级"}
    }
    
    # 如果城市不在模拟数据中，返回默认数据
    city_data = weather_data.get(city, {"temperature": 22, "condition": "未知", "wind": "未知"})
    
    return {
        "city": city,
        "date": date,
        "temperature": city_data["temperature"],
        "condition": city_data["condition"],
        "wind": city_data["wind"]
    }

# 定义一个文本翻译工具（模拟）
@tool
def translate_text(text: str, target_language: str = "en") -> str:
    """
    将文本翻译成指定语言。
    
    参数:
    text: 要翻译的文本
    target_language: 目标语言代码，默认为 'en'(英语)，可选 'zh'(中文), 'ja'(日语), 'ko'(韩语), 'fr'(法语), 'de'(德语)
    
    返回:
    翻译后的文本
    """
    # 这是一个模拟工具，实际应用中可以使用真实的翻译API
    translations = {
        "en": {"你好": "Hello", "谢谢": "Thank you", "再见": "Goodbye"},
        "ja": {"你好": "こんにちは", "谢谢": "ありがとう", "再见": "さようなら"},
        "ko": {"你好": "안녕하세요", "谢谢": "감사합니다", "再见": "안녕히 가세요"},
        "fr": {"你好": "Bonjour", "谢谢": "Merci", "再见": "Au revoir"},
        "de": {"你好": "Hallo", "谢谢": "Danke", "再见": "Auf Wiedersehen"}
    }
    
    # 检查是否有直接匹配的翻译
    if text in translations.get(target_language, {}):
        return translations[target_language][text]
    
    # 否则返回一个模拟的翻译结果
    return f"[翻译到{target_language}] {text}"

# 创建工具Agent（不需要记忆功能）
def create_tool_agent():
    """创建能够使用工具的Agent（不包含记忆功能）"""
    # 获取LLM
    llm = get_llm()
    
    # 定义工具列表
    tools = [calculator, get_current_date, get_weather, translate_text]
    
    # 创建提示模板（确保包含agent_scratchpad）
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的助手，能够使用工具来回答问题。请根据用户的问题选择合适的工具。"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # 验证提示模板包含所有必需的变量
    if "agent_scratchpad" not in prompt.input_variables:
        print("警告: 提示模板缺少agent_scratchpad变量")
        print(f"当前变量: {prompt.input_variables}")
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建Agent执行器
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # 直接返回agent_executor，不添加会话历史功能
    return agent_executor

def main() -> None:
    try:
        # 加载环境变量
        load_environment()
        
        print("===== LangChain 工具聊天机器人演示 ======")
        print("我可以使用多种工具来帮助你，包括计算器、日期查询、天气查询和文本翻译。")
        print("输入 'exit' 或 'quit' 退出程序。")
        print("注意：此版本不包含会话记忆功能，每次对话都是独立的。")
        print("\n示例问题:")
        print("1. 计算 123 加 456 等于多少？")
        print("2. 今天是几号？")
        print("3. 北京今天的天气怎么样？")
        print("4. 把'你好'翻译成英语。")
        print("\n请输入你的问题：")
        
        # 创建Agent
        agent = create_tool_agent()
        
        # 交互式对话循环
        while True:
            user_input = input("用户: ")
            
            if user_input.lower() in ["exit", "quit", "退出"]:
                print("再见！")
                break
            
            # 直接调用Agent处理用户输入
            response = agent.invoke({
                "input": user_input,
                "agent_scratchpad": []  # 提供空的agent_scratchpad列表
            })
            
            print(f"AI: {response['output']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查OPENAI_API_KEY等配置是否正确")


if __name__ == "__main__":
    main()