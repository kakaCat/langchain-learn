#!/usr/bin/env python3
"""
国际化模板示例：根据目标语言生成广告文案

功能要点：
- 使用 PromptTemplate 构建多语言提示词
- 通过语言代码控制输出语言（zh-CN、en-US、ja-JP 等）
- 统一的环境加载与 LLM 初始化
- 基本异常处理，确保示例稳定运行
"""

from __future__ import annotations

import os
from typing import Dict
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 从当前模块目录加载 .env

def load_environment() -> None:
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
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)


LANG_MAP: Dict[str, str] = {
    "zh-CN": "请用中文为{product}创作{style}风格的广告文案，并保持地道表达。",
    "en-US": "Please write an {style} style ad copy for {product} in natural American English.",
    "ja-JP": "{product}のために{style}スタイルの広告コピーを、日本語で自然に作成してください。",
}


def build_prompt(language: str) -> PromptTemplate:
    template = LANG_MAP.get(
        language,
        "Please write an {style} style ad copy for {product} in the specified language.",
    )
    return PromptTemplate.from_template(template)


def main() -> None:
    print("🔄 正在加载环境配置...")
    load_environment()
    print("=== 国际化模板示例：广告文案生成 ===")

    # 示例输入
    tests = [
        {"product": "智能手机", "style": "科幻", "language": "zh-CN"},
        {"product": "smartwatch", "style": "tech", "language": "en-US"},
        {"product": "電気自動車", "style": "エコ", "language": "ja-JP"},
        {"product": "智能家居系统", "style": "温馨", "language": "unknown"},
    ]

    for t in tests:
        product = t["product"]
        style = t["style"]
        language = t["language"]
        prompt = build_prompt(language)
        final_prompt = prompt.format(product=product, style=style)
        print(f"\n语言：{language} | 产品：{product} | 风格：{style}")
        print("提示词：")
        print(final_prompt)
        print("-" * 30)
        try:
            llm = get_llm()
            response = llm.invoke(final_prompt)
            print("模型回复：")
            print(response.content)
        except Exception as e:
            print("❌ 调用失败，请检查 .env 配置（OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL）或网络。")
            print(f"错误详情：{e}")


if __name__ == "__main__":
    main()