#!/usr/bin/env python3
"""
DDGS（DuckDuckGo Search）工具调用示例

两种用法：
1) 直接调用 DDGS 进行文本搜索（无需 LLM）。
2) 作为 LangChain 工具（@tool）集成到 Agent 中，由模型自动选择调用。

运行：
- 直接搜索模式（不依赖 OPENAI）：
  python 02_ddgs_search_tool_demo.py --mode direct

- Agent 工具调用模式（需 .env 提供 OPENAI_API_KEY 等）：
  python 02_ddgs_search_tool_demo.py --mode agent
"""
import os
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv
from duckduckgo_search import DDGS

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor


# ============================
# 环境加载与 LLM 配置
# ============================
def load_environment() -> None:
    """从当前模块目录加载 .env"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


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
# DDGS 搜索实现
# ============================
def ddgs_text_search(
    query: str,
    max_results: int = 5,
    region: str = "cn-zh",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    使用 DuckDuckGo Search 文本搜索。

    参数：
    - query: 搜索关键词
    - max_results: 最大返回条数
    - region: 地区/语言（例如："cn-zh"、"wt-wt"）
    - safesearch: 安全级别（"off"/"moderate"/"strict"）
    - timelimit: 时间范围（例如："d"/"w"/"m"/"y"），None 表示不限制

    返回：[{"title", "url", "snippet"}]
    """
    if not query or not query.strip():
        raise ValueError("query 不能为空")
    if max_results <= 0:
        raise ValueError("max_results 必须为正整数")

    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results,
            )
        )

    normalized: List[Dict[str, str]] = []
    for r in results:
        normalized.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
        )
    return normalized


# ============================
# LangChain 工具封装（@tool）
# ============================
@tool
def web_search(
    query: str,
    max_results: int = 5,
    region: str = "cn-zh",
    safesearch: str = "moderate",
    timelimit: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    使用 DuckDuckGo 文本搜索的工具。

    参数：
    - query: 搜索关键词
    - max_results: 最大返回条数（默认 5）
    - region: 地区/语言（默认 "cn-zh"）
    - safesearch: 安全级别（默认 "moderate"）
    - timelimit: 时间范围（可选："d"/"w"/"m"/"y"）

    返回：搜索结果列表，包含 title/url/snippet 字段。
    """
    return ddgs_text_search(query, max_results, region, safesearch, timelimit)


def create_search_agent() -> AgentExecutor:
    """创建具备 DDGS 搜索工具的 AgentExecutor"""
    llm = get_llm()
    tools = [web_search]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是信息检索助手。遇到需要实时/事实查询的问题时，优先调用 web_search 工具。返回时总结关键信息并附上来源URL。",
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# ============================
# CLI 入口
# ============================
def run_direct_cli() -> None:
    print("===== DDGS 直接搜索模式 =====")
    print("示例：输入关键词进行搜索，输入 'exit' 退出。")
    while True:
        q = input("搜索关键词: ")
        if q.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        try:
            results = ddgs_text_search(q, max_results=5)
            if not results:
                print("未找到结果。\n")
                continue
            print(f"共返回 {len(results)} 条：")
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}\n")
        except Exception as e:
            print(f"错误: {e}\n")


def run_agent_cli() -> None:
    print("===== LangChain Agent（DDGS 工具）模式 =====")
    print("我可以使用 web_search 工具来检索信息。输入 'exit' 退出。\n")

    agent = create_search_agent()
    while True:
        user_input = input("用户: ")
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("再见！")
            break
        try:
            response = agent.invoke({"input": user_input, "agent_scratchpad": []})
            print(f"AI: {response.get('output', '')}\n")
        except Exception as e:
            print(f"错误: {e}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDGS 搜索示例：direct 或 agent 模式")
    parser.add_argument(
        "--mode",
        choices=["direct", "agent"],
        default="direct",
        help="选择运行模式：direct（直接 DDGS 搜索）或 agent（LangChain 工具调用）",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
  
    print("===== LangChain Agent 固定示例（DDGS 工具）=====")
    fixed_input = "帮我检索 LangChain 的函数调用（Function Calling）核心要点，并附上来源链接。"
    print(f"固定输入: {fixed_input}\n")
    try:
        agent = create_search_agent()
        response = agent.invoke({"input": fixed_input, "agent_scratchpad": []})
        print(f"AI: {response.get('output', '')}\n")
    except Exception as e:
        print(f"错误: {e}\n")


if __name__ == "__main__":
    main()