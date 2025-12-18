#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_react_demo.py — LangChain v1 create_agent 版本（ReAct 风格）

本示例使用 LangChain v1 的标准代理接口 `create_agent`，并展示经典的
ReAct（先思考再行动）思路：模型在需要时调用工具、观察结果、继续推理，直到得到最终答案。

依赖：
- langchain>=1.0.0
- langchain-core>=1.0.0
- langchain-openai>=1.0.0
- langgraph>=1.0.0（create_agent 基于 LangGraph，但你无需直接使用它）
- openai>=1.0.0

环境变量：
- OPENAI_API_KEY 或者兼容的推理服务 API Key
- OPENAI_BASE_URL（可选，若使用自托管或第三方兼容服务）
- LC_AGENT_MODEL（优先使用），例如："openai:gpt-4o-mini"、"openai:gpt-4o"、"openai:gpt-4.1"
- OPENAI_MODEL（作为兜底），例如同上

运行：
python 01_react_demo.py
"""

import os
import sys
from typing import Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


from langchain_tavily import TavilySearch



def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

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



def get_tools():
    tavily_key = os.getenv("TAVILY_API_KEY")
    try:
        if tavily_key:
            return [TavilySearch(max_results=3, tavily_api_key=tavily_key)]
        else:
            print("[提示] 未检测到 TAVILY_API_KEY，搜索工具将禁用。")
            return []
    except Exception as e:
        print(f"[提示] Tavily 工具不可用：{e}")
        return []


def get_react_agent(system_prompt:str,tools:list) :

    llm = get_llm()

    return  create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )




def main():
    load_environment()
    agent = get_react_agent(
        system_prompt="You are a helpful research assistant.",
        tools=get_tools()
    )

    print("\n=== LangChain v1 create_agent · ReAct 示例 ===")

    user_input = "2024年诺贝尔物理获奖者"
    # 使用 v1 接口：传入 messages（role/content 对）
    try:
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })
    except Exception as e:
        print(f"助手发生错误: {e}")

    messages = result.get("messages", [])
    final_text = _extract_final_text_from_messages(messages)
    if final_text:
        print("助理:", final_text)
    else:
        print("[提示] 未从返回消息中提取到最终文本。完整结构如下：")
        print(result)






# 从返回的对话消息中提取最后一条助理文本
def _extract_final_text_from_messages(messages) -> Optional[str]:
    if not messages:
        return None
    last_assistant = None
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            last_assistant = m
            break
    msg = last_assistant or messages[-1]
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
        try:
            return "\n".join(str(b) for b in content)
        except Exception:
            return None
    return None


if __name__ == "__main__":
    main()
