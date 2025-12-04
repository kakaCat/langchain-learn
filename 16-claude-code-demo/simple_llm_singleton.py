#!/usr/bin/env python3
"""
简化版 LLM 单例模式 - 适用于单线程顺序执行场景

核心理念:
1. 只创建一个 ChatOpenAI 实例（全局单例）
2. httpx 自动处理 HTTP 连接池复用
3. 无需手动池化管理

适用场景:
- ✅ 单线程顺序执行（LangGraph 默认模式）
- ✅ 同步调用
- ❌ 多线程并发（需要使用连接池版本）
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def load_environment() -> None:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


# ============== 全局单例实例 ==============

# 在模块加载时创建一次，后续复用
_llm_instances = {}


def get_llm(model: Optional[str] = None, temperature: float = 0.2) -> object:
    """
    获取 LLM 单例实例

    特性:
    - 每个配置（model + temperature）只创建一个实例
    - httpx 自动处理 HTTP 连接池
    - 线程不安全（单线程场景使用）

    Returns:
        ChatOpenAI 或 ChatOllama 实例
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = (provider in {"ollama", "local"}) and not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        cache_key = ("ollama", model_name, temperature)
    else:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        cache_key = ("openai", model_name, temperature)

    # 单例模式: 如果已创建则直接返回
    if cache_key not in _llm_instances:
        if use_ollama:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"[LLM] 创建 Ollama 单例 model={model_name} temp={temperature}")
            _llm_instances[cache_key] = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                verbose=True,
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "800"))
            print(f"[LLM] 创建 OpenAI 单例 model={model_name} temp={temperature}")
            _llm_instances[cache_key] = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120,
                request_timeout=120,
                max_retries=3,
                verbose=True,
            )
    else:
        print(f"[LLM] 复用单例 {cache_key}")

    return _llm_instances[cache_key]


# ============== 使用示例 ==============

if __name__ == "__main__":
    load_environment()

    # 示例1: 默认配置
    llm1 = get_llm()
    llm2 = get_llm()
    print(f"是否同一个对象: {llm1 is llm2}")  # True

    # 示例2: 不同配置
    llm_creative = get_llm(temperature=0.8)
    llm_precise = get_llm(temperature=0.0)
    print(f"是否同一个对象: {llm_creative is llm_precise}")  # False

    # 示例3: 在节点中使用
    from langchain_core.messages import HumanMessage

    def some_node(state):
        """直接调用 get_llm() 即可"""
        llm = get_llm()  # 复用单例
        response = llm.invoke([HumanMessage(content="Hello")])
        return response.content

    result = some_node({})
    print(f"结果: {result}")
