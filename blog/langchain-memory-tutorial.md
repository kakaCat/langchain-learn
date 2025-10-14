---
title: "LangChain 入门教程：学习记忆模块"
description: "基于模块 04-memory 的实战介绍，含环境准备、依赖、基础用法与示例索引。"
keywords:
  - LangChain
  - Memory
  - ChatMessageHistory
  - RunnableWithMessageHistory
  - MessagesPlaceholder
  - InMemoryChatMessageHistory
  - 内存存储
  - 文件存储
  - 实体记忆
  - 摘要记忆
  - 令牌压缩记忆
  - 限制历史窗口记忆
tags:
  - Tutorial
  - Memory
  - LLM
author: "langchain-learn"
date: "2025-10-12"
lang: "zh-CN"
canonical: "/blog/langchain-memory-tutorial"
audience: "初学者 / 具备Python基础的LLM工程师"
difficulty: "beginner-intermediate"
estimated_read_time: "12-18min"
topics:
  - LangChain Core
  - Memory
  - ChatMessageHistory
  - InMemoryChatMessageHistory
  - RunnableWithMessageHistory
entities:
  - LangChain
  - OpenAI
  - dotenv
  - DeepSeek
  - Ollama
---

# LangChain 入门教程：学习记忆模块

## 本页快捷跳转
- 目录：
  - [引言](#intro)
  - [记忆的基础用法：将历史注入提示词](#memory-basics)
  - [聊天消息记忆存储](#stores)
    - [AI记忆内存存储(短期记忆)](#store-memory)
    - [AI记忆文本存储(长期记忆)](#store-text-memory)
  - [聊天消息记忆压缩](#compression)
    - [AI记忆信息实体压缩](#entity)
    - [AI记忆信息总结（摘要）压缩](#summary)
    - [聊天消息记忆滑窗保留](#limited)
    - [聊天消息记忆令牌滑窗压缩（Token）](#token-compression-window)
  - [常见错误与快速排查 (Q/A)](#qa)
    - [存储类型选择指南](#qa-storage-choice)
    - [滑窗 vs 令牌压缩如何取舍](#qa-window-vs-compression)
    - [不同模型/供应商下启用记忆](#qa-providers)
    - [避免上下文爆炸的实用建议](#qa-context-bloat)
    - [压缩损失的评估与监控](#qa-eval)
  - [参考资料](#references)
  - [更新记录](#changelog)
  - [总结](#summary-final)

---

<a id="intro" data-alt="引言 概述 目标 受众"></a>
## 引言
本教程围绕 LangChain 的记忆模块，帮助你在工程实践中构建可维护的对话记忆：让模型“记住”上下文、理解长期目标，并在多轮交互中保持一致性。

<a id="what-is-ai-memory" data-alt="什么是 AI 记忆 定义 概念"></a>
## 什么是AI记忆

AI 默认只会根据当前输入生成回答，无法记住之前的对话。为了让 AI 在多轮对话中保持上下文连贯，我们需要将对话历史记录注入到提示词中，这种机制就是**记忆**。

<a id="why-memory" data-alt="为什么 需要 记忆 作用"></a>
### 01、为什么需要记忆

由于模型的上下文窗口（输入给AI的内容）有限，必须控制注入内容的体量与质量。如何记录与使用记忆，直接影响对话的一致性与成本。下文将基于 LangChain 记忆模块，按场景给出可复用的使用方法与实践。

<a id="memory-basics" data-alt="基础 用法 ChatPromptTemplate MessagesPlaceholder RunnableWithMessageHistory"></a>
### 02、记忆的基础用法：将历史注入提示词

在 LangChain 中有 2 种方式可以将历史注入提示词：
1. 记忆通过 ChatPromptTemplate 与 MessagesPlaceholder 注入到提示词中，并用 RunnableWithMessageHistory 统一管理。
2. 手动拼接历史到 系统提示词 / 用户提示词 消息（适合极简或一次性脚本）；需自行控制格式与令牌（Token）预算，可能出现重复注入或越权内容。

接下来我将介绍不同记忆使用方法。
项目配置与依赖安装请参见下文的，请参考另一篇教程：[LangChain 入门教程：构建你的第一个聊天机器人](https://juejin.cn/post/7559428036514709554)。


<a id="stores" data-alt="不同存储类型 内存 文件"></a>
## 聊天消息记忆存储

记忆可以分为短期记忆和长期记忆:
- 短期记忆：存储在内存中的对话历史记录，仅在当前会话中有效。
- 长期记忆：存储在外部数据库或文件中的对话历史记录，可跨会话访问。

<a id="store-memory" data-alt="内存存储 进程内 无配置"></a>
### AI记忆内存存储(短期记忆)

```python

import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment() -> None:
    """加载环境变量配置"""
    try:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
        print("✅ 环境变量加载成功")
    except Exception as e:
        print(f"❌ 环境变量加载失败: {e}")
        raise


def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请在.env文件中配置")
        
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # 默认0.7以获得更好的对话效果
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
        
        print(f"✅ LLM配置成功 - 模型: {model}, 温度: {temperature}")
        return ChatOpenAI(**kwargs)
    except Exception as e:
        print(f"❌ LLM配置失败: {e}")
        raise


# 内存存储（仅进程内，不持久化）
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_in_memory_history(session_id: str) -> InMemoryChatMessageHistory:
    """使用内存存储聊天历史记录"""
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
        print(f"✅ 创建新的内存会话历史 - session_id: {session_id}")
    else:
        print(f"✅ 获取现有内存会话历史 - session_id: {session_id}, 消息数: {len(_store[session_id].messages)}")
    return _store[session_id]


def create_conversation_chain() -> RunnableWithMessageHistory:
    """创建带内存历史记录的会话链"""
    try:
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
        
        print("✅ 对话链创建成功")
        return conversation
    except Exception as e:
        print(f"❌ 对话链创建失败: {e}")
        raise


def run_conversation_example(session_id: str = "memory_demo") -> None:
    """
    运行内存存储的对话示例
    
    Args:
        session_id: 会话标识符，默认为 "memory_demo"
    
    Raises:
        Exception: 对话过程中发生的异常
    """
    try:
        langchain.debug = True
        conversation = create_conversation_chain()

        print(f"\n🚀 开始对话示例 (存储类型: memory, 会话ID: {session_id})")

        print("\n📝 第一轮对话...")
        response = conversation.invoke({"input": "你好，我叫张三。"}, config={"configurable": {"session_id": session_id}})
        print(f"🤖 AI: {response.content}")

        print("\n📝 第二轮对话...")
        response = conversation.invoke({"input": "我刚才告诉你我叫什么名字？"}, config={"configurable": {"session_id": session_id}})
        print(f"🤖 AI: {response.content}")

        print("\n📝 第三轮对话...")
        response = conversation.invoke({"input": "能帮我生成一个简短的自我介绍吗？"}, config={"configurable": {"session_id": session_id}})
        print(f"🤖 AI: {response.content}")

        print("\n✅ 对话示例运行完成")
        langchain.debug = False
    except Exception as e:
        print(f"❌ 对话示例运行失败: {e}")
        print("💡 请检查环境配置和依赖是否正确。")
        raise


def main() -> None:
    """主函数：运行内存存储示例"""
    run_conversation_example()


if __name__ == "__main__":
    main()
```

要点：
- 使用 MessagesPlaceholder("history") 将历史消息嵌入提示词。
- 用 RunnableWithMessageHistory 管理会话 id 与历史读写，免去手工拼接历史。
- 用 InMemoryChatMessageHistory 存储到内存中，会话结束后清除历史记录。
- 提示词: 系统提示词+用户提示词1+AI回答1+用户提示词2+AI回答2+最新的用户提示词的逻辑

<a id="store-text-memory" data-alt="文本内存存储 文件存储 JSON"></a>
### AI记忆文本存储(长期记忆)

```python
import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment() -> None:
    """加载环境变量配置"""
    try:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
        print("✅ 环境变量加载成功")
    except Exception as e:
        print(f"❌ 环境变量加载失败: {e}")
        raise


def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请在.env文件中配置")
        
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # 默认0.7以获得更好的对话效果
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
        
        print(f"✅ LLM配置成功 - 模型: {model}, 温度: {temperature}")
        return ChatOpenAI(**kwargs)
    except Exception as e:
        print(f"❌ LLM配置失败: {e}")
        raise


def get_file_history(session_id: str) -> FileChatMessageHistory:
    """
    使用文件存储聊天历史记录（JSON）
    
    Args:
        session_id: 会话标识符，用于区分不同用户的对话历史
        
    Returns:
        FileChatMessageHistory: 文件存储的聊天历史记录实例
        
    Raises:
        Exception: 当文件创建或读取失败时抛出异常
    """
    try:
        file_path = f"chat_history_{session_id}.json"
        history = FileChatMessageHistory(file_path, encoding="utf-8", ensure_ascii=False)
        print(f"✅ 文件历史记录加载成功 - session_id: {session_id}, 文件: {file_path}")
        return history
    except Exception as e:
        print(f"❌ 文件历史记录加载失败: {e}")
        raise


def create_conversation_chain() -> RunnableWithMessageHistory:
    """
    创建带文件历史记录的会话链
    
    Returns:
        RunnableWithMessageHistory: 配置好的会话链实例
        
    Raises:
        Exception: 会话链创建过程中发生的异常
    """
    try:
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
            get_session_history=get_file_history,
            input_messages_key="input",
            history_messages_key="history",
            verbose=True,
        )
        print("✅ 文件存储对话链创建成功")
        return conversation
    except Exception as e:
        print(f"❌ 文件存储对话链创建失败: {e}")
        raise


def run_conversation_example(session_id: str = "file_demo") -> None:
    """运行文件存储的对话示例（会在当前目录创建 JSON 文件）"""
    try:
        langchain.debug = True
        conversation = create_conversation_chain()

        print(f"\n=== 开始对话示例 (存储类型: file, 会话ID: {session_id}) ===")

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
        print("请检查运行目录写入权限，或检查环境配置与依赖。")


def main() -> None:
    """主函数：运行文件存储示例"""
    run_conversation_example()


if __name__ == "__main__":
    main()
```

要点：
- 使用 MessagesPlaceholder("history") 将历史消息嵌入提示词。
- 用 RunnableWithMessageHistory 管理会话 id 与历史读写，免去手工拼接历史。
- 用 FileChatMessageHistory 存储到本地文件中，需要注意编码问题，避免中文乱码。
- 提示词: 系统提示词+用户提示词1+AI回答1+用户提示词2+AI回答2+最新的用户提示词的逻辑

<a id="compression" data-alt="聊天 消息 记忆 压缩 概览"></a>
## AI聊天消息记忆压缩

经过多轮对话后，会话记忆会迅速膨胀并逼近模型的上下文窗口限制。为避免超限与成本飙升，需要对记忆进行压缩与治理。压缩的目标是用最小信息损失维持上下文连贯性与可用性。下面给出常见策略与取舍，再详细介绍 4 种压缩方法。
- 结合实体记忆维护关键事实（如姓名、公司、偏好），减少摘要遗漏的重要信息。
- 对话压缩：将较早的历史对话压缩为摘要，保留关键信息与上下文，降低总令牌（Token）占用。
- 会话内短期：优先使用滑窗保留，实现简单、开销低，但可能丢失早期上下文。
- 成本敏感：根据令牌（Token）预算调整摘要触发阈值与保留窗口，常见做法是在上下文窗口的 70%–80% 时启动压缩。


<a id="entity" data-alt="实体 记忆 EntityMemory"></a>
### AI记忆信息实体压缩
- 通过从对话中抽取“人物、地点、组织、关键事实”等实体信息，形成结构化记忆，便于长期引用与更新。
- 常见做法：在每轮对话后抽取并合并实体档案（如“姓名、公司、兴趣”等），在提示词中以结构化字段注入上下文。
  
```python

import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain.globals import set_verbose
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import StdOutCallbackHandler
from langchain.globals import set_verbose, set_debug
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment():
    """加载环境变量配置"""
    try:
        load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
        print("✅ 环境变量加载成功")
    except Exception as e:
        print(f"❌ 环境变量加载失败: {e}")
        raise

# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例
    
    Returns:
        ChatOpenAI: 配置好的语言模型实例
        
    Raises:
        ValueError: 当API密钥未设置时抛出
        Exception: 其他配置错误时抛出
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请在.env文件中配置")
        
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # 默认0.7以获得更好的对话效果
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

        kwargs = {
            "model": model,
            "api_key": api_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 120,
            "max_retries": 3,
            "request_timeout": 120,
            "base_url": base_url,
            "verbose": True
        }
        
        print(f"✅ LLM配置成功 - 模型: {model}, 温度: {temperature}")
        return ChatOpenAI(**kwargs)
    except ValueError as e:
        print(f"❌ 配置验证失败: {e}")
        raise
    except Exception as e:
        print(f"❌ LLM配置失败: {e}")
        raise

# 会话存储 - 存储不同会话的历史记录
store: Dict[str, BaseChatMessageHistory] = {}
# 实体存储 - 存储不同会话的实体信息
entity_store: Dict[str, Dict[str, str]] = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取或创建指定会话的历史记录
    
    Args:
        session_id: 会话标识符
        
    Returns:
        BaseChatMessageHistory: 会话历史记录实例
        
    Raises:
        Exception: 会话历史获取过程中发生的异常
    """
    try:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
            print(f"✅ 创建新的会话历史 - 会话ID: {session_id}")
        else:
            print(f"✅ 加载现有会话历史 - 会话ID: {session_id}")
        return store[session_id]
    except Exception as e:
        print(f"❌ 获取会话历史失败 - 会话ID: {session_id}, 错误: {e}")
        raise

# 提取实体的提示模板
ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个实体提取助手。请从以下对话中提取实体及其相关信息。请以JSON格式输出，其中键是实体名称，值是关于该实体的信息。如果没有实体，请返回空对象。"),
    ("human", "对话: {conversation}")
])

# 提取实体
def extract_entities(conversation: List[BaseMessage], session_id: str) -> Dict[str, str]:
    """
    从对话内容中提取关键实体信息
    
    该方法使用LLM模型分析对话文本，识别并提取其中的人物、地点、时间等实体信息。
    提取的实体将用于增强对话记忆和上下文理解。
    
    Args:
        conversation: 需要分析的对话消息列表
        session_id: 会话标识符，用于区分不同用户的实体存储
        
    Returns:
        Dict[str, str]: 提取的实体字典，键为实体名称，值为实体相关信息
        
    Raises:
        json.JSONDecodeError: 当LLM响应无法解析为有效JSON时抛出
        Exception: 其他处理过程中发生的异常
    """
    try:
        llm = get_llm()
        
        # 构建对话字符串
        conversation_str = "\n".join([f"{msg.type}: {msg.content}" for msg in conversation])
        
        print(f"🔍 实体提取 - 对话内容长度: {len(conversation_str)} 字符")
        print(f"🔍 实体提取 - 会话ID: {session_id}")
        
        # 提取实体
        entity_chain = ENTITY_EXTRACTION_PROMPT | llm
        print(f"🔍 实体提取 - 开始调用实体提取链")
        handler = StdOutCallbackHandler()
        response = entity_chain.invoke({"conversation": conversation_str}, config={"callbacks": [handler]})
        print(f"🔍 实体提取 - 原始响应: {response.content}")
        
        try:
            # 尝试解析JSON响应
            import json
            entities = json.loads(response.content)
            print(f"✅ 实体提取 - 解析成功，实体数量: {len(entities)}")
            print(f"🔍 实体提取 - 解析后的实体: {entities}")
        except json.JSONDecodeError as e:
            # 如果解析失败，使用空字典
            entities = {}
            print(f"⚠️ 实体提取 - JSON解析失败: {e}，使用空字典")
            print("⚠️ 建议检查LLM响应格式是否符合JSON规范")
        
        # 更新实体存储
        if session_id not in entity_store:
            entity_store[session_id] = {}
            print(f"✅ 创建新的实体存储 - 会话ID: {session_id}")
        
        old_entities = entity_store[session_id].copy()
        entity_store[session_id].update(entities)
        
        print(f"✅ 实体存储更新 - 会话ID: {session_id}")
        print(f"🔍 实体存储更新详情 - 从 {old_entities} 更新为 {entity_store[session_id]}")
        
        return entities
    except Exception as e:
        print(f"❌ 实体提取过程发生异常 - 会话ID: {session_id}, 错误: {e}")
        print("⚠️ 请检查网络连接、API密钥和模型配置")
        return {}

# 创建带实体记忆的对话链
# 修改create_entity_aware_chain函数
def  create_entity_aware_chain(session_id: str):
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
    
    print(f"[VERBOSE] 创建对话链 - 会话ID: {session_id}")
    print(f"[VERBOSE] 创建对话链 - 实体信息: {entity_info}")
    
    # 创建链 - 不使用bind方法，而是在调用时传递参数
    chain = prompt | llm
    
    return chain, entity_info

# 修改main函数中的调用部分
def main() -> None:
    try:
        set_debug(True)
        # 加载环境变量
        load_environment()
        # 开启LangChain全局verbose日志
    
        # 如需更详细调试日志，可开启下面这行（输出更为冗长）
        # set_debug(True)
        session_id = "user_123"
        
        # 创建会话历史
        history = get_session_history(session_id)
        
        # 添加初始对话
        user_message = "我叫张三，我在微软工作。"
        history.add_user_message(user_message)
        
        # 大模型提取实体
        entities = extract_entities(history.messages, session_id)
        print(f"识别的实体：{entities}")
        
        # 创建实体感知的对话链 - 获取chain和entity_info
        conversation_chain, entity_info = create_entity_aware_chain(session_id)
        
        # 获取AI响应 - 直接传递entity_info参数 我叫张三，我在微软工作。
        print(f"[VERBOSE] 第一轮对话 - 用户输入: {user_message}")
        print(f"[VERBOSE] 第一轮对话 - 传递的实体信息: {entity_info}")
        handler = StdOutCallbackHandler()
        response = conversation_chain.invoke({"input": user_message, "entity_info": entity_info}, config={"callbacks": [handler]})
        ai_response = response.content
        history.add_ai_message(ai_response)
        
        print(f"[VERBOSE] 第一轮对话 - AI响应: {ai_response}")
        print(f"AI响应: {ai_response}")
        
        # 第二轮对话测试实体记忆
        second_user_message = "我在哪里工作？"
        history.add_user_message(second_user_message)
        
        # 更新实体存储
        extract_entities(history.messages, session_id)
        
        # 更新对话链以包含最新实体信息
        conversation_chain, updated_entity_info = create_entity_aware_chain(session_id)
        
        # 获取第二轮AI响应 - 直接传递更新的entity_info参数
        print(f"[VERBOSE] 第二轮对话 - 用户输入: {second_user_message}")
        print(f"[VERBOSE] 第二轮对话 - 更新的实体信息: {updated_entity_info}")
        handler = StdOutCallbackHandler()
        second_response = conversation_chain.invoke({"input": second_user_message, "entity_info": updated_entity_info}, config={"callbacks": [handler]})
        second_ai_response = second_response.content
        history.add_ai_message(second_ai_response)
        
        print(f"[VERBOSE] 第二轮对话 - AI响应: {second_ai_response}")
        print(f"AI响应: {second_ai_response}")
        
        # 打印最终实体存储
        print(f"最终实体存储：{entity_store.get(session_id, {})}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请检查OPENAI_API_KEY等配置是否正确")

if __name__ == "__main__":
    main()
```

要点：
- 使用 extract_entities 通过LLM提炼成实体，键是实体名称，值是关于该实体的信息。
- 用 InMemoryChatMessageHistory 存储实体的记忆。
- 通过把实体json化，嵌入到系统提示词中。

<a id="summary" data-alt="摘要 记忆 ConversationSummary"></a>
### AI记忆信息总结（摘要）压缩

- 将较早的历史对话压缩为摘要，保留关键信息与上下文，降低总令牌（Token）占用。
- 下文“令牌压缩记忆”示例演示了在超过阈值时自动生成摘要并重置历史，以实现稳定的长对话。

```python
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI



# 从当前模块目录加载 .env
def load_environment():
    """
    加载环境变量配置文件
    
    从当前模块目录加载.env文件，用于配置API密钥等环境变量。
    
    Raises:
        FileNotFoundError: 当.env文件不存在时抛出
        Exception: 其他加载环境变量时发生的异常
    """
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        print(f"🔧 加载环境变量文件: {env_path}")
        load_dotenv(dotenv_path=env_path, override=False)
        print("✅ 环境变量加载成功")
    except FileNotFoundError:
        print("⚠️ 警告: .env文件未找到，将使用系统环境变量")
    except Exception as e:
        print(f"❌ 加载环境变量失败: {e}")
        raise



# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 未设置，请在.env文件中配置")
        
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))  # 默认0.7以获得更好的对话效果
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

        print(f"✅ LLM配置成功 - 模型: {model}, 温度: {temperature}")
        return ChatOpenAI(**kwargs)
    except Exception as e:
        print(f"❌ LLM配置失败: {e}")
        raise

# 会话存储
store = {}

def get_session_history(session_id: str):
    """根据session_id获取或创建对应的聊天历史记录"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()  # 使用内存存储
        print(f"✅ 创建新的会话历史 - session_id: {session_id}")
    else:
        print(f"✅ 获取现有会话历史 - session_id: {session_id}, 消息数: {len(store[session_id].messages)}")
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
```

要点：
- 用 InMemoryChatMessageHistory 存储聊天的记忆。
- 经过几轮的对话，最后告诉模型“请总结一下我们的对话内容。”，实现对聊天过程的数据进行压缩。


## LangChain 记忆模块总结

### 核心记忆类型与结构

```
BaseChatMessageHistory（会话消息历史）
├── InMemoryChatMessageHistory（内存）
├── FileChatMessageHistory（文件）
├── RedisChatMessageHistory（Redis）
└── SQLChatMessageHistory（SQL/DB）

Memory Strategies（记忆策略）
├── Limited History（滑窗保留）
├── Token Compression（令牌滑窗/摘要）
├── Summary Memory（摘要记忆）
├── Entity Memory（实体记忆）
├── VectorStore Memory（向量存储记忆）
└── Key-Value Memory（键值记忆）
```

### 记忆类型功能对比

| 记忆类型 | 主要用途 | 持久性 | 上下文保留 | 复杂度 | 适用场景 |
|----------|----------|--------|------------|--------|----------|
| 内存存储（InMemory） | 单会话快速读写 | 否 | 最近上下文 | 低 | 临时对话、原型验证 |
| 文件存储（File） | 本地持久化 | 是 | 全量可控 | 低 | 个人项目、小型应用 |
| Redis/SQL 存储 | 服务端持久化 | 是 | 全量可控 | 中 | 生产环境、并发与多用户 |
| 向量存储记忆 | 语义检索长期知识 | 是 | 主题语义保留 | 中 | FAQs、知识库、RAG 结合 |
| 实体记忆 | 关键人物/偏好/事实 | 可选 | 关键实体强化 | 中 | 个性化助手、客户画像 |
| 摘要记忆 | 长对话压缩 | 可选 | 主题与要点 | 中 | 长会话成本控制 |
| 滑窗保留 | 保留最近 N 条 | 否/是 | 近期上下文 | 低 | 简洁高效、无需摘要 |
| 令牌滑窗压缩 | 令牌预算内保留 | 可选 | 近期+旧摘要 | 中 | 成本敏感、上下文平衡 |
| 键值记忆 | 参数/配置/状态 | 可选 | 精准字段 | 低 | 工具调用、流程状态 |

### 记忆选择指南

1. 临时对话/原型 → `InMemoryChatMessageHistory`
   - 单用户或单会话、速度优先、无需持久化。
2. 本地轻量级持久化 → `FileChatMessageHistory`
   - 低运维、易备份、开发机或个人项目。
3. 生产持久化与并发 → `Redis/SQLChatMessageHistory`
   - 多用户、可靠存储、可做审计与统计。
4. 长期知识与检索 → 向量存储记忆（结合 RAG）
   - 文档/FAQ/手册，语义召回提升上下文质量。
5. 个性化与画像 → 实体记忆（偏好/身份/约束）
   - 提取并更新关键实体，减少重复问答。
6. 长会话成本控制 → 摘要记忆或令牌滑窗压缩
   - 定期摘要旧对话，保留近期消息与关键事实。
7. 工具与流程参数 → 键值记忆
   - 保存会话上下文中的指令、配置、临时变量。

### 最佳实践

- 历史键名一致：`MessagesPlaceholder("chat_history")` 与链输入键保持一致。
- 会话隔离：为不同用户/会话设置独立 `session_id` 并正确传递。
- 混合策略：滑窗 + 摘要 + 实体记忆联合使用，兼顾近期与长期。
- 成本控制：按令牌预算触发压缩，避免超上下文窗口与费用激增。
- 冗余治理：去重系统提示与规则、合并冗长回复与无效噪声。
- 持久化选择：本地用文件，服务端用 Redis/SQL，注意并发与一致性。

### 策略与工具补充

- 令牌计数：使用 `tiktoken` 估算上下文长度，按预算触发压缩。
- 会话封装：用 `RunnableWithMessageHistory` 集成链与历史，简化多轮对话。
- 摘要生成：系统提示约束“保留关键事实与任务上下文”，避免信息丢失。
- 实体抽取：定期从对话抽取人物/偏好/任务状态并更新实体存储。
- 检索增强：向量存储结合记忆，优先召回相关知识后再注入上下文。



<a id="limited" data-alt="限制 历史 滑窗 Limited History"></a>
### 聊天消息记忆滑窗保留

```python

import os
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI


# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """
    创建并配置语言模型实例
    
    Returns:
        ChatOpenAI: 配置好的语言模型实例
        
    Raises:
        ValueError: 当OPENAI_API_KEY未设置时抛出
        Exception: 创建模型实例时发生的其他异常
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY环境变量未设置，请检查.env文件或系统环境变量")
        
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        base_url = os.getenv("OPENAI_BASE_URL")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))

        print(f"🤖 创建语言模型 - 模型: {model}, 温度: {temperature}, 最大令牌: {max_tokens}")
        
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
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        raise
    except Exception as e:
        print(f"❌ 创建语言模型失败: {e}")
        raise

# 自定义有限历史记录类
class LimitedChatMessageHistory:
    """
    限制消息数量的聊天历史记录类
    
    Attributes:
        max_messages (int): 最大允许的消息数量
        _messages (list): 内部存储的消息列表
    """
    def __init__(self, max_messages: int = 10):
        """
        初始化有限历史记录类
        
        Args:
            max_messages (int): 最大允许的消息数量，默认为10
            
        Raises:
            ValueError: 当max_messages小于等于0时抛出
        """
        try:
            if max_messages <= 0:
                raise ValueError("max_messages必须大于0")
            
            self._messages = []
            self.max_messages = max_messages
            print(f"✅ 初始化有限历史记录 - 最大消息数: {max_messages}")
        except ValueError as e:
            print(f"❌ 初始化失败: {e}")
            raise
        except Exception as e:
            print(f"❌ LimitedChatMessageHistory初始化失败: {e}")
            raise
    
    def add_message(self, message: BaseMessage) -> None:
        """
        添加消息并保持消息数量限制
        
        Args:
            message (BaseMessage): 要添加的消息对象
            
        Raises:
            TypeError: 当message不是BaseMessage类型时抛出
        """
        try:
            if not isinstance(message, BaseMessage):
                raise TypeError("message必须是BaseMessage类型")
            
            self._messages.append(message)
            # 超过最大消息数时，删除最旧的消息
            if len(self._messages) > self.max_messages:
                self._messages = self._messages[-self.max_messages:]
                print(f"⚠️ 消息数量超过限制，已删除旧消息，当前保留: {len(self._messages)}条")
        except Exception as e:
            print(f"❌ 添加消息失败: {e}")
            raise
    
    def clear(self) -> None:
        """清除所有消息"""
        try:
            self._messages = []
            print("✅ 已清除所有消息")
        except Exception as e:
            print(f"❌ 清除消息失败: {e}")
            raise
    
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
    """
    获取或创建有限历史记录实例
    
    Args:
        session_id (str): 会话标识符
        
    Returns:
        LimitedChatMessageHistory: 有限历史记录实例
        
    Raises:
        ValueError: 当session_id为空或无效时抛出
    """
    try:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id不能为空且必须是字符串类型")
        
        if session_id not in store:
            print(f"✅ 为session_id {session_id} 创建新的LimitedChatMessageHistory实例")
            # 直接使用我们自定义的类，不继承ChatMessageHistory
            store[session_id] = LimitedChatMessageHistory(max_messages=4)  # 只保留4条消息
        else:
            print(f"✅ 获取现有有限历史记录 - session_id: {session_id}")
        return store[session_id]
    except Exception as e:
        print(f"❌ 获取有限历史记录失败 - session_id: {session_id}, 错误: {e}")
        raise

def main() -> None:
    """
    演示有限历史记录功能的主函数
    
    该函数展示了如何使用LimitedChatMessageHistory类来限制聊天历史消息数量，
    包括添加消息、验证消息数量限制等功能。
    
    Raises:
        Exception: 执行过程中发生的任何异常
    """
    try:
        # 加载环境变量
        print("🔧 加载环境变量...")
        load_environment()
        
        session_id = "user_123"
        print(f"📝 获取历史记录 - session_id: {session_id}")
        history = get_limited_history(session_id)
        
        # 测试添加消息
        print("\n📤 添加测试消息...")
        history.add_message(HumanMessage(content="你好，我是张三"))
        history.add_message(AIMessage(content="你好张三，我是AI助手"))
        
        # 验证消息数量
        print(f"\n📊 当前消息数量: {len(history.messages)}")
        print(f"📝 消息内容: {[msg.content for msg in history.messages]}")
        
        # 测试消息数量限制
        print("\n🧪 测试消息数量限制...")
        for i in range(5):
            history.add_message(HumanMessage(content=f"测试消息 {i}"))
            history.add_message(AIMessage(content=f"测试响应 {i}"))
            print(f"📊 添加消息后数量: {len(history.messages)}")
            print(f"📝 保留的消息: {[msg.content for msg in history.messages]}")
            
        print("\n✅ 有限历史记录测试完成")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️ 请检查OPENAI_API_KEY等配置是否正确")


if __name__ == "__main__":
    main()
```

要点：
- 这是一种会遗漏部分对话的压缩方法。
- 优点是保留了最近的对话内容，避免了信息丢失。
- 通过设置`max_messages`参数可以控制保留最近几次的消息数量。

<a id="token-compression-window" data-alt="令牌 压缩 滑窗 Window"></a>
### 聊天消息记忆令牌滑窗压缩（Token）

```python
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

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    获取或创建会话历史记录
    
    Args:
        session_id (str): 会话标识符
        
    Returns:
        BaseChatMessageHistory: 会话历史记录实例
        
    Raises:
        ValueError: 当session_id为空或无效时抛出
        Exception: 获取会话历史时发生的其他异常
    """
    try:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id不能为空且必须是字符串类型")
        
        if session_id not in store:
            print(f"✅ 创建新的会话历史记录 - session_id: {session_id}")
            store[session_id] = InMemoryChatMessageHistory()
        else:
            print(f"✅ 获取现有会话历史记录 - session_id: {session_id}")
        return store[session_id]
    except Exception as e:
        print(f"❌ 获取会话历史失败 - session_id: {session_id}, 错误: {e}")
        raise

# 自定义令牌计数函数
def count_tokens(messages: List[BaseMessage], model: str = "gpt-3.5-turbo") -> int:
    """
    计算消息的令牌数量
    
    Args:
        messages (List[BaseMessage]): 消息列表
        model (str): 模型名称，默认"gpt-3.5-turbo"
        
    Returns:
        int: 消息的令牌数量
        
    Raises:
        Exception: 令牌计数过程中发生的异常
    """
    try:
        if not messages:
            print("ℹ️ 消息列表为空，令牌数为0")
            return 0
            
        # 使用tiktoken库进行令牌计数
        encoding = tiktoken.encoding_for_model(model)
        buffer_string = get_buffer_string(messages)
        token_count = len(encoding.encode(buffer_string))
        print(f"🔢 令牌计数 - 模型: {model}, 消息数量: {len(messages)}, 令牌数: {token_count}")
        return token_count
    except KeyError:
        # 如果模型不支持，使用近似计数
        print(f"⚠️ 模型{model}不支持，使用近似令牌计数")
        approximate_count = sum(len(str(msg.content)) // 4 for msg in messages)
        print(f"🔢 近似令牌计数: {approximate_count}")
        return approximate_count
    except Exception as e:
        print(f"❌ 令牌计数失败: {e}")
        # 返回一个安全的默认值
        return sum(len(str(msg.content)) // 4 for msg in messages)

# 实现简单的令牌压缩逻辑
def compress_messages_if_needed(session_id: str, max_tokens: int = 1000):
    """
    检查并压缩会话历史消息，确保令牌数不超过限制
    
    当会话历史消息的令牌数超过最大限制时，该方法会保留最新的几条消息，
    并对旧消息生成摘要，从而实现令牌压缩的目的。
    
    Args:
        session_id (str): 会话标识符
        max_tokens (int): 最大允许的令牌数，默认1000
        
    Returns:
        bool: 是否执行了压缩操作
        
    Raises:
        Exception: 压缩过程中发生的异常
    """
    try:
        history = get_session_history(session_id)
        current_tokens = count_tokens(history.messages)
        
        print(f"🔍 令牌压缩检查 - 会话ID: {session_id}, 当前令牌数: {current_tokens}, 最大限制: {max_tokens}")
        
        if current_tokens > max_tokens:
            print(f"⚠️ 令牌数超过限制，开始压缩历史消息")
            
            # 保留最新的几条消息
            keep_recent = 2
            if len(history.messages) > keep_recent:
                # 提取要总结的旧消息
                messages_to_summarize = history.messages[:-keep_recent]
                print(f"📝 需要压缩的消息数量: {len(messages_to_summarize)}")
                
                # 生成摘要
                llm = get_llm()
                summary_prompt = ChatPromptTemplate.from_messages([
                    ("system", "请为以下对话生成简洁的摘要，保留重要信息和上下文："),
                    ("human", "{conversation}")
                ])
                
                conversation_str = get_buffer_string(messages_to_summarize)
                summary_chain = summary_prompt | llm
                summary = summary_chain.invoke({"conversation": conversation_str})
                
                # 重置历史，保留摘要和最新消息
                history.clear()
                history.add_user_message(f"过往对话摘要：{summary.content}")
                history.add_messages(history.messages[-keep_recent:])

                compressed_tokens = count_tokens(history.messages)
                print(f"✅ 成功压缩历史消息，压缩后令牌数: {compressed_tokens}")
                return True
            else:
                print(f"ℹ️ 消息数量不足，无需压缩")
                return False
        else:
            print(f"ℹ️ 令牌数在限制范围内，无需压缩")
            return False
            
    except Exception as e:
        print(f"❌ 令牌压缩失败 - 会话ID: {session_id}, 错误: {e}")
        print("⚠️ 压缩过程中发生异常，历史消息保持不变")
        return False


def create_conversation_chain():
    """
    创建带历史记忆和令牌压缩功能的会话链
    
    Returns:
        RunnableWithMessageHistory: 配置好的会话链实例
        
    Raises:
        Exception: 创建会话链过程中发生的异常
    """
    try:
        print("🔗 创建会话链...")
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
        
        print("✅ 会话链创建成功")
        return conversation
    except Exception as e:
        print(f"❌ 创建会话链失败: {e}")
        raise

def main() -> None:
    """
    主函数：演示令牌压缩记忆功能
    
    演示如何使用令牌压缩来管理对话历史，包括：
    1. 初始化环境
    2. 创建会话链
    3. 进行多轮对话
    4. 检查并压缩消息
    5. 验证记忆功能
    
    Raises:
        Exception: 演示过程中发生的异常
    """
    try:
        print("🚀 启动令牌压缩记忆演示...")
        set_debug(True)
        
        # 加载环境变量
        load_environment()
        
        # 创建会话链
        conversation = create_conversation_chain()
        session_id = "user_123"
        print(f"📝 使用会话ID: {session_id}")

        # 第一轮对话
        print("\n💬 第一轮对话...")
        response = conversation.invoke(
            {"input": "你好，我叫张三。"},
            config={"configurable": {"session_id": session_id}},
            verbose=True
        ) 
        print(f"🤖 AI响应: {response.content}")
        
        # 检查并压缩消息
        print("\n🔍 检查令牌压缩...")
        compress_messages_if_needed(session_id, max_tokens=1000)

        # 第二轮对话，检查记忆功能
        print("\n💬 第二轮对话...")
        response = conversation.invoke(
            {"input": "我刚才告诉你我叫什么名字？"},
            config={"configurable": {"session_id": session_id}},
            verbose=True
        )
        print(f"🤖 AI响应: {response.content}")

        # 打印当前记忆内容
        print(f"\n📋 当前记忆内容:")
        history_messages = get_session_history(session_id).messages
        for i, message in enumerate(history_messages, 1):
            print(f"  {i}. {message.type}: {message.content}")
            
        print("\n✅ 令牌压缩记忆演示完成")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        print("⚠️ 请检查以下配置:")
        print("  - OPENAI_API_KEY环境变量是否正确设置")
        print("  - 网络连接是否正常")
        print("  - 依赖包是否正确安装")
        raise

if __name__ == "__main__":
    main()
```

要点：
- 这是一种通过每次输入大模型前计算令牌（Token）数量来记忆总结（摘要）方式压缩记忆。
- 优点是一定不会超出最大令牌（Token）数量，仅最大可能压缩记忆和保留最近的消息。
- 通过`count_tokens`方法计算当前记忆的令牌（Token）数量，超过最大令牌（Token）数量时进行压缩。
- 压缩逻辑：保留最新的几条消息，生成对旧消息的摘要，替换旧消息。


<a id="qa" data-alt="常见 错误 排查 Q/A"></a>
## 常见错误与快速排查 (Q/A)

- 记忆未生效：核对 <code>MessagesPlaceholder("history")</code> 与 <code>input/history</code> 键名一致。
- 会话串线：确保为不同用户设置不同 <code>session_id</code>，并正确传递到配置中。

<a id="qa-storage-choice"></a>
### 存储类型选择指南

- 内存存储：会话内、速度快、不持久；适合临时对话与小型应用。
- 文件/数据库：跨会话、可持久；文件适合本地与轻量级，Redis/SQL 适合生产环境。
- 向量存储：用于语义检索与长期知识库，结合记忆提升召回质量。

<a id="qa-window-vs-compression"></a>
### 滑窗 vs 令牌压缩如何取舍

- 滑窗保留：固定保留最近 N 条消息，简单高效但可能丢失长期上下文。
- 令牌滑窗压缩：按令牌预算保留最近内容并摘要旧消息，更平衡上下文与成本。

<a id="qa-providers"></a>
### 不同模型/供应商下启用记忆

- OpenAI/DeepSeek：按官方 SDK 配置，确保 `max_tokens` 与超时设置合理。
- Ollama/本地模型：关注上下文窗口大小与生成速度，适当降低温度与长度。

<a id="qa-context-bloat"></a>
### 避免上下文爆炸的实用建议

- 控制消息冗余、合并冗长回复；设置历史上限与定期摘要策略。
- 对系统提示与规则做去重与压缩，避免重复注入。

<a id="qa-eval"></a>
### 压缩损失的评估与监控

- 对摘要召回率、关键实体完整性、对话一致性做最小评测。
- 监控上下文长度与响应质量，必要时切换/混合压缩策略。


<a id="links"></a>
### 详细代码和文档

- 完整代码：查看 [GitHub 仓库](https://github.com/kakaCat/langchain-learn/tree/04-memory/01_01_in_memory_history_demo.py)
- 项目结构：参考仓库中的 `README.md`
- LangChain 文档：https://python.langchain.com/
- LangChain OpenAI 集成（Python API 索引）：https://api.python.langchain.com/
- DeepSeek API 文档：https://api-docs.deepseek.com/
- OpenAI API 文档：https://platform.openai.com/docs/api-reference
- Azure OpenAI 文档：https://learn.microsoft.com/azure/ai-services/openai/
- Ollama 文档：https://ollama.com/docs


<a id="references" data-alt="引用 参考 文献 链接"></a>
## 参考资料

- LangChain Memory 指南：https://python.langchain.com/docs/modules/memory/
- RunnableWithMessageHistory（表达式语言记忆）：https://python.langchain.com/docs/expression_language/memory/
- ChatMessageHistory（聊天消息记忆）：https://python.langchain.com/docs/modules/memory/chat_messages/
- 项目示例目录：`./04-memory/`、`./11-token-compression/`
- 相关教程：[`langchain-chatbot-tutorial.md`](./langchain-chatbot-tutorial.md)、[`langchain-prompt-templates-tutorial.md`](./langchain-prompt-templates-tutorial.md)


<a id="summary-final" data-alt="总结 收尾 最佳实践"></a>
## 总结

