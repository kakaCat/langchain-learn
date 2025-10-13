---
title: "LangChain 入门教程：学习记忆模块"
description: "基于模块 04-memory 的实战介绍，含环境准备、依赖、基础用法与示例索引。"
keywords:
  - LangChain
  - Memory
  - ChatMessageHistory
  - RunnableWithMessageHistory
  - MessagesPlaceholder
  - Redis
  - SQLite
  - 文件存储
  - 实体记忆
  - 摘要记忆
  - 键值记忆
  - 向量存储
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
  - RunnableWithMessageHistory
entities:
  - LangChain
  - OpenAI
  - dotenv
---

# LangChain 入门教程：学习记忆模块

## 本页快捷跳转
- 目录：
  - [引言](#intro)
  - [记忆的基础用法：将历史注入提示词](#memory-basics)
  - [不同存储类型的快速示例](#stores)
    - [内存存储（进程内）](#store-memory)
    - [文件存储（JSON）](#store-file)
  - [实体记忆（Entity Memory）](#entity)
  - [摘要记忆（Summary Memory）](#summary)
  - [令牌压缩记忆（Token Compression）](#token-compression)
  - [限制历史窗口（Limited History）](#limited)
  - [键值记忆（Key-Value Memory）](#kv)
  - [常见错误与快速排查 (Q/A)](#qa)
  - [总结](#summary-final)

---

<a id="intro" data-alt="引言 概述 目标 受众"></a>
## 引言
本教程围绕 LangChain 的记忆模块，帮助你在工程实践中构建可维护的对话记忆：让模型“记住”上下文、理解长期目标，并在多轮交互中保持一致性。示例脚本均来自本仓库的 <a href="../04-memory/README.md">04-memory</a> 模块，可直接运行。环境与依赖见 

### 什么是AI记忆

AI 默认只会根据当前输入生成回答，无法记住之前的对话。为了让 AI 在多轮对话中保持上下文连贯，我们需要将对话历史记录注入到提示词中，这种机制就是**记忆**。由于输入AI的内容长度是有限的，所以我们要考虑那些内容输入给AI。因此如何记录记忆和使用记录也是一个问题。接下来我们开始使用langchain的记忆模块和不同场景的记忆使用办法。接下来我们将详细介绍 LangChain 记忆模块的使用方法和实践。

<a id="memory-basics" data-alt="基础 用法 ChatPromptTemplate MessagesPlaceholder RunnableWithMessageHistory"></a>
## 记忆的基础用法：将历史注入提示词
在 LangChain 中，记忆通常通过 ChatPromptTemplate 与 MessagesPlaceholder 注入到提示词中，并用 RunnableWithMessageHistory 统一管理。接下来我将介绍2种不借助其他存储工具的记忆使用办法。

### AI记忆内存存储

```python

import os
import langchain
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
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


# 内存存储（仅进程内，不持久化）
_store: dict[str, ChatMessageHistory] = {}


def get_in_memory_history(session_id: str) -> ChatMessageHistory:
    """使用内存存储聊天历史记录"""
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def create_conversation_chain() -> RunnableWithMessageHistory:
    """创建带内存历史记录的会话链"""
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
    return conversation


def run_conversation_example(session_id: str = "memory_demo") -> None:
    """运行内存存储的对话示例"""
    try:
        langchain.debug = True
        conversation = create_conversation_chain()

        print(f"\n=== 开始对话示例 (存储类型: memory, 会话ID: {session_id}) ===")

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
        print("请检查环境配置和依赖是否正确。")


def main() -> None:
    """主函数：运行内存存储示例"""
    run_conversation_example()


if __name__ == "__main__":
    main()
```
要点：
- 使用 MessagesPlaceholder("history") 将历史消息嵌入提示词。
- 用 RunnableWithMessageHistory 管理会话 id 与历史读写，免去手工拼接历史。
- 提示词: 系统提示词+用户提示词1+AI回答1+用户提示词2+AI回答2+最新的用户提示词的逻辑


<a id="entity" data-alt="实体 记忆 EntityMemory"></a>
## 实体记忆（Entity Memory）
- 示例脚本：<a href="../04-memory/02_entity_memory_demo.py">02_entity_memory_demo.py</a>
- 作用：识别并记录用户或对象的关键属性（如姓名、公司、偏好），在后续对话中进行引用。
- 提示：将实体抽取与结构化输出结合，提升一致性与可用性。

<a id="summary" data-alt="摘要 记忆 ConversationSummary"></a>
## 摘要记忆（Summary Memory）
- 示例脚本：<a href="../04-memory/03_summary_memory_demo.py">03_summary_memory_demo.py</a>
- 作用：对长对话进行滚动摘要，保留关键要点，控制上下文长度。
- 提示：与 <a href="../11-token-compression/README.md">令牌压缩</a> 思路互补，综合使用更稳健。

<a id="token-compression" data-alt="令牌 压缩 Token Compression"></a>
## 令牌压缩记忆（Token Compression）
- 示例脚本：<a href="../04-memory/03_token_compression_memory_demo.py">03_token_compression_memory_demo.py</a>
- 作用：在不丢失关键信息的前提下，压缩历史消息，提升上下文填充效率。

<a id="limited" data-alt="限制 历史 窗口"></a>
## 限制历史窗口（Limited History）
- 示例脚本：<a href="../04-memory/04_limited_history_memory_demo.py">04_limited_history_memory_demo.py</a>
- 作用：仅保留最近 N 条消息，控制成本与延迟。

<a id="kv" data-alt="键值 记忆 Key-Value"></a>
## 键值记忆（Key-Value Memory）
- 示例脚本：<a href="../04-memory/05_key_value_memory_demo.py">05_key_value_memory_demo.py</a>
- 作用：通过键值对保存配置信息或临时变量，便于跨轮引用。

<a id="qa" data-alt="常见 错误 排查 Q/A"></a>
## 常见错误与快速排查 (Q/A)
- 无法调用模型：检查 <code>OPENAI_API_KEY</code>、<code>OPENAI_BASE_URL</code>、<code>OPENAI_MODEL</code>；参见 <a href="./运行指导文档.md">运行指导文档</a>。
- Redis 连接失败：确认本地/云端 Redis 已启动，URL 与权限正确。
- SQLite 写入失败：确认运行目录写入权限；文件路径无只读限制。
- 记忆未生效：核对 <code>MessagesPlaceholder("history")</code> 与 <code>input/history</code> 键名一致。
- 会话串线：确保为不同用户设置不同 <code>session_id</code>，并正确传递到配置中。

<a id="summary-final" data-alt="总结 收尾 最佳实践"></a>
## 总结
- 通过统一的提示词接口与历史包装，记忆变得可组合、可测试、可落地。
- 依据场景选择存储方案：内存/文件适合本地与轻量场景，Redis/SQL 适合生产；向量存储用于语义检索。
- 建议与结构化输出、RAG、压缩技术组合，形成稳定的长对话工程方案。