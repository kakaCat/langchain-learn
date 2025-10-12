#!/usr/bin/env python3
"""
JSON 模式输出示例：广告文案生成
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env
def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


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
    """JSON 模式输出示例：广告文案生成"""
    print("=== JSON 模式输出示例：广告文案生成 ===")

    # 创建 JSON 输出解析器
    parser = JsonOutputParser()

    # 创建提示词模板
    template = """
    请为指定产品生成广告文案，并以 JSON 格式输出：

    产品：{product}
    风格：{style}

    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["product", "style"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # 示例产品
    product = "智能手机"
    style = "科幻"

    # 格式化提示词
    formatted_prompt = prompt.format(product=product, style=style)
    print("提示词：")
    print(formatted_prompt)
    print("\n" + "=" * 50 + "\n")
    response = get_llm().invoke(formatted_prompt)
    print("模型回复：")
    print(response.content)
    print("\n" + "=" * 50 + "\n")
    # 解析 JSON 输出
    try:
        ad_copy = parser.parse(response.content)
        print("解析后的广告文案：")
        print(ad_copy)
    except Exception as e:
        print("JSON 解析错误:", e)
if __name__ == "__main__":
    main()