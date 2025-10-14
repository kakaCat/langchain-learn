---
title: "LangChain 入门教程：函数调用与工具 (Function Calling / Tools)"
description: "基于模块 07-function_calling 的完整实战教程：工具定义、Agent 构建、提示词设计与 CLI 交互，含环境配置与常见错误排查。"
keywords:
  - LangChain
  - FunctionCalling
  - Tools
  - Agent
  - AgentExecutor
  - ChatPromptTemplate
  - MessagesPlaceholder
  - @tool
  - ToolNode
tags:
  - Tutorial
  - Tools
  - Agent
author: "langchain-learn"
date: "2025-10-14"
lang: "zh-CN"
canonical: "/blog/langchain-function-calling-tutorial"
audience: "初学者 / 具备Python基础的LLM工程师"
difficulty: "beginner-intermediate"
estimated_read_time: "14-20min"
topics:
  - LangChain Core
  - Tools / FunctionCalling
  - AgentExecutor
  - ChatPromptTemplate
  - 调用链与提示词结构
entities:
  - LangChain
  - OpenAI
  - dotenv
  - Python
qa_intents:
  - "什么是函数调用（工具调用）？与 Agent 有何关系？"
  - "如何用 @tool 定义工具并自动被模型选择？"
  - "为什么需要 MessagesPlaceholder('agent_scratchpad')？"
  - "如何创建支持工具调用的 AgentExecutor？"
  - "如何组织提示词让模型正确路由到工具？"
  - "常见错误如何排查（缺少变量、参数类型错误、API配置）？"
  - "如何扩展到真实 API 与多工具协作？"
---

# LangChain 入门教程：函数调用与工具 (Function Calling / Tools)

## 本页快捷跳转
- 目录：
  - [引言](#intro)
  - [函数调用与工具的核心概念](#concepts)
  - [AI 使用工具完成任务](#define-tools)
  - [常见错误与快速排查 (Q/A)](#qa)
  - [官方链接与源码](#links)
  - [总结](#summary)
  - [术语与别名（便于检索与问法对齐）](#glossary)

---

<a id="intro" data-alt="introduction 引言 概述 目标 受众"></a>
## 引言

此前我们主要讲如何与 AI 对话，AI 只是单纯进行“文本生成”。但这与我们期望的智能助手相距甚远——我们希望它像人一样完成具体任务。要做到这一点，AI 必须“会用工具”。例如，用户想查询当前时间或天气，模型需要调用相应的工具获取真实数据。LangChain 的“函数调用/工具调用”能力，让模型能依据指令自动选择并调用开发者提供的工具，从“能说”升级为“能做”。


<a id="concepts" data-alt="概念 工具 函数调用 Agent 调度 路由"></a>
## 函数调用与工具的核心概念

- 工具（Tool）：开发者暴露给模型的“可调用函数”，具备明确的输入参数与输出形态，用于执行外部操作或计算。
- 函数调用（Function Calling）：模型在生成过程中提出“调用某函数”的意图，框架据此路由并执行对应工具后将结果反馈给模型继续生成或直接返回给用户。

### 功能起源与生态
- 由主流大模型厂商在 2023 年率先提出并普及，其中以 OpenAI 的 Function Calling 最为知名，随后 Anthropic（Claude 的 Tool Use）、Google（Gemini 的工具调用）等生态均提供了类似能力。
- 不同厂商的接口名称与消息格式略有差异，但核心思想一致：在对话生成过程中由模型发起“调用某工具/函数”的结构化请求，开发者执行工具并将结果以规范消息返回，模型据此继续生成或直接回答用户。

### API 层实现机制（通用）
- 开发者在请求中声明“工具列表”（包含名称、描述、参数 schema）。
- 模型在回答阶段产出结构化的“工具调用”对象（例如 `tool_calls`：`{"name": ..., "arguments": {...}}`）。
- 应用层拿到该对象后实际执行对应函数/外部 API，并将结果以“tool 消息”附带到后续对话中（通常需要携带 `tool_call_id` 以便模型对齐调用关系）。
- 模型在拿到工具结果后继续生成最终答案，或提出下一次工具调用，从而形成“思考→调用→观察→继续”的循环。

示例（抽象化的消息片段）：

```json
{
  "assistant": {
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"杭州\", \"unit\": \"celsius\"}"
        }
      }
    ]
  }
}

{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "杭州 23℃，晴"
}
```


### LangChain 的封装方式
- 通过 `@tool` 将 Python 函数声明为可调用工具（自动生成 schema 与描述）。
- 使用 `ChatPromptTemplate + MessagesPlaceholder('agent_scratchpad')` 设计提示词，承载工具调用轨迹与中间结果。
- 通过 `create_tool_calling_agent` 与 `AgentExecutor` 组合 LLM 与工具集合，实现“自动选择并调用工具”的 Agent。



<a id="define-tools" data-alt="@tool 装饰器 定义工具 参数校验 错误处理"></a>
## AI 使用工具完成任务


```python
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
            response = agent.invoke({"input": user_input})
            
            print(f"AI: {response['output']}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查OPENAI_API_KEY等配置是否正确")


if __name__ == "__main__":
    main()
```

要点：
- 使用 `@tool` 装饰器暴露函数作为工具；参数类型注解与文档字符串用于自动生成“可调用 schema”。
- 工具内部应处理错误与边界（例如除零），并返回明确的结构。
- 当你替换为真实 API（天气/翻译）时，务必在工具内进行参数校验、异常捕获与安全限制。


<a id="qa" data-alt="常见错误 排查 缺少变量 参数错误 API 环境"></a>
## 常见错误与快速排查 (Q/A)

- 提示词缺少 `agent_scratchpad`
  - 现象：Agent 报错或无法正确路由到工具
  - 修复：在 `ChatPromptTemplate.from_messages([...])` 中添加 `MessagesPlaceholder(variable_name="agent_scratchpad")`

- 工具参数类型/取值错误
  - 现象：如 `calculator` 的 `operation` 非法、`divide` 除数为 0
  - 修复：在工具里做参数校验并抛出明确错误；模型侧提示词补充调用示例与限制说明

- 环境变量未配置或端点错误
  - 现象：401/403/404/429/超时
  - 修复：检查 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`；降低并发与 `max_tokens`，增加重试与超时

- 返回结构不一致
  - 现象：工具返回字段名不统一导致后续处理失败
  - 修复：统一工具返回结构（使用字典固定键），在 Agent 层做轻量验证与容错

- 调试困难
  - 现象：难以定位模型为何选择某工具
  - 修复：打开 `verbose=True` 观察调用轨迹；在 system 提示词加入路由规则与使用示例

<a id="links" data-alt="官方链接 源码 路径 文档"></a>
## 官方链接与源码

- 完整代码：查看 [GitHub 仓库](https://github.com/kakaCat/langchain-learn/tree/04-memory/01_01_in_memory_history_demo.py)
- 项目结构：参考仓库中的 `README.md`
- LangChain 文档：https://python.langchain.com/
- LangChain OpenAI 集成（Python API 索引）：https://api.python.langchain.com/
- LangGraph 文档：https://langchain-ai.github.io/langgraph/

<a id="summary" data-alt="总结 回顾 下一步"></a>
## 总结

通过本教程，你已掌握：
- 如何用 `@tool` 暴露函数为模型可调用的工具
- 如何用 `create_tool_calling_agent`+`AgentExecutor` 构建具备工具调用能力的 Agent
- 如何设计提示词结构并通过 `agent_scratchpad` 支持多轮工具调用
- 如何在 CLI 中完成交互式工具调用与结果展示

下一步建议：接入真实 API（天气、翻译等），引入会话记忆/状态管理，设计多工具协作链路，面向生产环境完善缓存、限流与权限控制。

<a id="glossary" data-alt="术语 别名 同义词 检索"></a>
## 术语与别名（便于检索与问法对齐）

- 函数调用 / 工具调用：Function Calling / Tool Calling / Tools
- 工具：Tool / Function / 可调用模块
- Agent：代理 / 决策器 / 调度器
- `agent_scratchpad`：调用轨迹占位 / 中间记录区 / scratchpad
- 路由：工具选择 / 决策 / 调用计划