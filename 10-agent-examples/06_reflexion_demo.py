#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_reflexion_demo.py — LangChain v1 create_agent 版本（Reflexion 技法）

统一工具与 CLI。该示例强调：先生成候选答案，随后进行自我反思与误差分析，最后给出改进版答案。
"""

import os
from datetime import datetime

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError
from langchain_core.tools import StructuredTool

from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper



def get_tools():
    """返回 Tavily 搜索工具列表，优先官方包，其次社区版；失败则空工具继续。"""
    try:
        from langchain_tavily import TavilySearch
        return [TavilySearch(max_results=3)]
    except Exception:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            return [TavilySearchResults(k=3)]
        except Exception:
            print("[warn] 无法加载 TavilySearch 工具，继续无工具模式。请检查 TAVILY_API_KEY 与依赖。")
            return []
TOOLS = get_tools()

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

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)




class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response}




class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question. Provide an answer, reflection,

    cite your reflection with references, and finally
    add search queries to improve the answer."""

    references: list[str] = Field(
        description="Citations motivating your updated answer."
    )

    revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


def get_revisor():
    actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "\n\n<system>Reflect on the user's original question and the"
            " actions taken thus far. Respond using the {function_name} function.</reminder>",
        ),
    ]
).partial(
    time=lambda: datetime.now().isoformat(),
)
    revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
    revision_chain = actor_prompt_template.partial(
        first_instruction=revise_instructions,
        function_name=ReviseAnswer.__name__,
    ) | get_llm().bind_tools(tools=[ReviseAnswer])
    revision_validator = PydanticToolsParser(tools=[ReviseAnswer])

    revisor = ResponderWithRetries(runnable=revision_chain, validator=revision_validator)
    return revisor

def get_first_responder():
    actor_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert assistant.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "\n\n<system>Reflect on the user's original question and the actions taken thus far. Respond using the {function_name} function.</reminder>",
            ),
        ]
    ).partial(
        time=lambda: datetime.now().isoformat(),
    )
    first_instruction = (
        "Answer the question. Provide a ~250-word answer, then a reflection, "
        "and 1-3 search queries to improve the answer."
    )
    chain = actor_prompt_template.partial(
        first_instruction=first_instruction,
        function_name=AnswerQuestion.__name__,
    ) | get_llm().bind_tools(tools=[AnswerQuestion])
    validator = PydanticToolsParser(tools=[AnswerQuestion])
    return ResponderWithRetries(runnable=chain, validator=validator)


def _get_num_iterations(state: list):
    i = 0
    for m in state[::-1]:
        if m.type not in {"tool", "ai"}:
            break
        i += 1
    return i


def event_loop(state: list):
    MAX_ITERATIONS = 5
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state["messages"])
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


tool_node = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)



class Reflexion(TypedDict):
    messages: Annotated[list, add_messages]


def create_workflow():
    first_responder = get_first_responder()
    revisor = get_revisor()
    workflow = StateGraph(Reflexion)
    workflow.add_node("draft", first_responder.respond)
    workflow.add_node("execute_tools", tool_node)
    workflow.add_node("revise", revisor.respond)

    workflow.add_edge(START, "draft")
    # draft -> execute_tools
    workflow.add_edge("draft", "execute_tools")
    # execute_tools -> revise
    workflow.add_edge("execute_tools", "revise")
    workflow.add_conditional_edges("revise", event_loop, ["execute_tools", END])


    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path='blog/flow_01.png')
    return app

def main() -> None:
    load_environment()
    """运行工作流并输出最终结果"""
    app = create_workflow()



if __name__ == "__main__":
    main()
