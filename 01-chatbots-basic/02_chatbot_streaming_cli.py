#!/usr/bin/env python3
"""
Module 1: Chatbots Basic Streaming CLI Demo
此脚本演示如何使用 LangChain 的 ChatOpenAI 进行“流式输出”，实现边生成边打印的体验。
- 从 .env 或环境变量读取 OPENAI_API_KEY 等配置
- 提供交互式 CLI，逐段打印模型回复
- 简单错误处理
"""

from __future__ import annotations

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 确保中文输出正常
sys.stdout.reconfigure(encoding="utf-8")

# 从当前模块目录加载 .env

def load_environment():
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


def stream_answer(llm: ChatOpenAI, question: str) -> None:
    """使用流式输出打印模型答案"""
    print(f"问题：{question}\n答案：", end="", flush=True)
    for chunk in llm.stream(question):
        # chunk 通常是 AIMessageChunk，包含增量 content
        content = getattr(chunk, "content", None)
        if content:
            print(content, end="", flush=True)
    print("\n—— 完成 ——")


def run_streaming_cli() -> None:
    """交互式流式输出 CLI"""
    load_environment()
    llm = get_llm()

    print("===== LangChain 流式输出演示 =====")
    print("输入你的问题后，模型会边生成边输出；输入 'exit' 或 'quit' 退出程序。\n")

    while True:
        user_input = input("用户: ")
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        stream_answer(llm, user_input)


def main() -> None:
    run_streaming_cli()


if __name__ == "__main__":
    main()