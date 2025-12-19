#!/usr/bin/env python3
"""
Tavily 检索工具调用示例

两种用法：
1) 直接调用 Tavily 进行文本搜索（无需 LLM，但需 TAVILY_API_KEY）。
2) 作为 LangChain 工具（@tool）集成到 Agent 中，由模型自动选择调用。

运行：
- 直接搜索模式（不依赖 OPENAI）：
  python 03_tavily_search_tool_demo.py  # 若未设置 OPENAI_API_KEY，将执行直接模式

- Agent 工具调用模式（需 .env 提供 OPENAI_API_KEY 与 TAVILY_API_KEY）：
  python 03_tavily_search_tool_demo.py  # 检测到 OPENAI_API_KEY 将执行 Agent 示例
"""
import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor


# ============================
# 环境加载与 LLM 配置
# ============================
def load_environment() -> None:
    """从当前模块目录加载 .env"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


def get_llm() -> ChatOpenAI:
    """创建并配置语言模型实例（用于 Agent 模式）"""
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
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)


# ============================
# Tavily 搜索实现（直接调用）
# ============================
def tavily_text_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    使用 Tavily API 进行文本搜索。

    返回：[{"title", "url", "snippet"}]
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")
    if max_results <= 0:
        raise ValueError("max_results 必须为正整数")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 TAVILY_API_KEY，请在 .env 中配置后重试。")

    try:
        from tavily import TavilyClient
    except Exception as e:
        raise RuntimeError("缺少 tavily-python 依赖，请先安装：pip install tavily-python") from e

    client = TavilyClient(api_key=api_key)

    # TavilyClient.search 返回包含 results（title/url/content 等）
    resp = client.search(query=query, max_results=max_results)
    results = resp.get("results", [])

    normalized: List[Dict[str, str]] = []
    for r in results[:max_results]:
        normalized.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "") or r.get("text", ""),
            }
        )
    return normalized


# ============================
# LangChain 工具封装（@tool）
# ============================
@tool
def web_search_tavily(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    使用 Tavily 搜索的 LangChain 工具。

    返回：统一的列表结构，包含 title/url/snippet。
    """
    # 使用 LangChain 社区工具封装，自动读取 TAVILY_API_KEY
    from langchain_community.tools.tavily_search import TavilySearchResults

    tool_impl = TavilySearchResults(max_results=max_results)
    raw = tool_impl.invoke(query)

    results: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for r in raw:
            # TavilySearchResults 通常返回 {"url", "content"}
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", ""),
                }
            )
    else:
        # 兼容返回为字符串的情况
        results.append({"title": "", "url": "", "snippet": str(raw)})
    return results


def create_search_agent_tavily() -> AgentExecutor:
    """创建具备 Tavily 搜索工具的 AgentExecutor"""
    llm = get_llm()
    tools = [web_search_tavily]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是信息检索助手。遇到需要实时/事实查询的问题时，优先调用 web_search_tavily 工具。返回时总结关键信息并附上来源URL。",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ============================
# 入口逻辑（按环境变量自动选择模式）
# ============================
def main() -> None:
    load_environment()

    example_query = "LangChain 函数调用（Function Calling）教程"
    fixed_input = "帮我检索 LangChain 的函数调用（Function Calling）核心要点，并附上来源链接。"

    tavily_key = os.getenv("TAVILY_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not tavily_key:
        print("未检测到 TAVILY_API_KEY，请在 .env 中配置后重试。")
        print("示例未执行。")
        return

    if openai_key:
        print("===== LangChain Agent 固定示例（Tavily 工具）=====")
        print(f"固定输入: {fixed_input}\n")
        try:
            agent = create_search_agent_tavily()
            response = agent.invoke({"input": fixed_input, "agent_scratchpad": []})
            print(f"AI: {response.get('output', '')}\n")
        except Exception as e:
            print(f"错误: {e}\n")
    else:
        print("===== Tavily 直接搜索模式 =====")
        print(f"固定关键词: {example_query}\n")
        try:
            results = tavily_text_search(example_query, max_results=5)
            if not results:
                print("未找到结果。\n")
            else:
                print(f"共返回 {len(results)} 条：")
                for i, r in enumerate(results, 1):
                    print(f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}\n")
        except Exception as e:
            print(f"错误: {e}\n")


if __name__ == "__main__":
    main()