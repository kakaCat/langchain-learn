#!/usr/bin/env python3
"""
Pydantic 模型约束输出示例：广告文案生成
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
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
        "verbose": False,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)

class AdCopy(BaseModel):
    """广告文案模型"""
    product: str = Field(description="产品名称")
    style: str = Field(description="文案风格")
    headline: str = Field(description="广告标题")
    description: str = Field(description="广告描述")
    call_to_action: str = Field(description="行动号召")

    # Pydantic v2 配置
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product": "智能手机",
                "style": "科幻",
                "headline": "穿越时空的智能体验",
                "description": "量子芯片带来前所未有的运算速度，全息投影让沟通触手可及。这不是手机，这是通往未来的钥匙。",
                "call_to_action": "立即预订，开启未来之旅",
            }
        }
    )


def main() -> None:
    load_environment()
    """Pydantic 模型约束输出示例：广告文案生成"""
    print("=== Pydantic 模型约束输出示例：广告文案生成 ===")

    # 创建 Pydantic 输出解析器
    parser = PydanticOutputParser(pydantic_object=AdCopy)

    # 创建提示词模板
    template = """
    请为指定产品生成广告文案，并按照指定格式输出：

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
    print("-" * 30)

    # 调用模型
    response =  get_llm().invoke(formatted_prompt)
    print("模型回复：")
    print(response.content)
    # 解析输出
    ad_copy = parser.parse(response.content)
    print("\n解析后的广告文案：")
    print(ad_copy)

if __name__ == "__main__":
    main()
