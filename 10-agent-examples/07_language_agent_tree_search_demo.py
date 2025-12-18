#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_language_agent_tree_search_demo.py — LangChain v1 create_agent 版本（语言代理树搜索，LATS 思路）

统一工具与 CLI。该示例强调：提出多个候选方案，进行简短评估与对比，选择最佳方案并给出最终答案。
"""

import os
from typing import Optional
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# 工具定义
@tool
def calculator(expression: str) -> str:
    """计算一个简单的数学表达式，例如 "2 + 3 * 4"。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"计算失败: {e}"

@tool
def get_current_date(_: str = "") -> str:
    """返回当前日期（YYYY-MM-DD）。参数忽略。"""
    return datetime.now().strftime("%Y-%m-%d")

@tool
def search_information(query: str) -> str:
    """占位的信息检索工具（不访问外网）。"""
    return (
        "[模拟搜索结果] 这是关于 '" + query + "' 的简要信息。"
        " 在真实环境中，请接入向量数据库或 Web 检索 API。"
    )

TOOLS = [calculator, get_current_date, search_information]


def load_environment():
    if load_dotenv is not None:
        try:
            load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
        except Exception:
            pass


def get_agent(system_prompt: str, tools: list):
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    if not api_key:
        raise ValueError("OPENAI_API_KEY 未设置，请在 .env 中配置有效密钥")
    if not base_url and model.startswith("deepseek"):
        raise ValueError("OPENAI_BASE_URL 未设置，DeepSeek 示例需配置为 https://api.deepseek.com")
    if not model:
        raise ValueError("OPENAI_MODEL 未设置，请指定可用模型（示例：deepseek-chat）")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


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


def main():
    load_environment()
    agent = get_agent(
        system_prompt=(
            "你是一个能进行语言代理树搜索（LATS）的助理。"
            "请先生成多个候选方案（2-4个），对每个方案进行简短评估（优缺点/风险/可行性），"
            "若需要调用工具进行验证，请在评估阶段执行。最后选择最优方案并给出清晰的最终答案。"
        ),
        tools=TOOLS,
    )

    print("\n=== LangChain v1 create_agent · LATS 示例 ===")
    print("可用工具:", ", ".join(t.name for t in TOOLS))
    print("提示: 直接输入问题，或输入 'exit' 退出。")

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("[退出]")
            break

        try:
            result = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            })
        except Exception as e:
            print(f"助手发生错误: {e}")
            continue

        messages = result.get("messages", [])
        final_text = _extract_final_text_from_messages(messages)
        if final_text:
            print("助理:", final_text)
        else:
            print("[提示] 未从返回消息中提取到最终文本。完整结构如下：")
            print(result)


if __name__ == "__main__":
    main()