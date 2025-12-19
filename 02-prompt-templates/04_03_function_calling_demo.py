#!/usr/bin/env python3
"""
函数调用模板示例：广告文案生成
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": False,
        "base_url": base_url
    }
    return ChatOpenAI(**kwargs)

def main() -> None:
    load_environment()
    """函数调用模板示例：广告文案生成"""
    print("=== 函数调用模板示例：广告文案生成 ===")

    # 定义函数调用模板
    template = """
    根据用户需求，调用相应的广告文案生成函数：

    需求：{query}

    可用的函数：
    - generate_ad_copy(product: str, style: str): 生成指定产品和风格的广告文案
    - analyze_market_trend(keyword: str): 分析市场趋势
    - optimize_ad_performance(copy: str): 优化广告文案效果

    请选择最合适的函数并返回函数调用信息。
    """

    prompt = PromptTemplate.from_template(template)

    # 示例查询
    queries = [
        "请为智能手机生成科幻风格的广告文案",
        "分析当前智能手表的市场趋势",
        "优化这段电动汽车广告文案的效果",
    ]

    for query in queries:
        formatted_prompt = prompt.format(query=query)
        print(f"需求：{query}")
        print("提示词：")
        print(formatted_prompt)
        print("-" * 30)
        response = get_llm().invoke(formatted_prompt)
        print("AI回复：")
        print(response.content)
        print("-" * 30)

if __name__ == "__main__":
    main()