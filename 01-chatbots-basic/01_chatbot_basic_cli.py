#!/usr/bin/env python3
"""
Module 1: Chatbots Basic CLI Demo
此脚本实现：
- LLMChain(llm=OpenAI(...), prompt=PromptTemplate, verbose=True, memory=ConversationBufferMemory)
- 从 .env 或环境变量读取 OPENAI_API_KEY
- 简单 CLI 循环与错误处理
"""

from __future__ import annotations

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


def get_llm() -> ChatOpenAI:

    """创建并配置语言模型实例"""
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
    
    # 前置校验：快速失败 + 清晰错误信息
    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置，请在 .env 中配置有效密钥")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL 未设置，DeepSeek 示例需配置为 https://api.deepseek.com")
    if not model:
        raise ValueError("OPENAI_MODEL 未设置，请指定可用模型（示例：deepseek-chat）")
    
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

def main() -> None:
    try:
        load_environment()
        llm = get_llm()
        question = "AI是什么？"
        print(f"问题：{question}")
        response = llm.invoke(question)
        print(f"答案：{response.content}")
    except Exception as e:
        print("❌ 运行失败，请检查 .env 配置与网络。")
        print(f"错误详情：{e}")


if __name__ == "__main__":
    main()