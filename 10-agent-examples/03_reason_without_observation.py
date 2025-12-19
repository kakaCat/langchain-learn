#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_reason_without_observation.py — LangChain v1 create_agent 版本（Reason Without Observation）

与 01/02 示例一致，统一使用 `create_agent`、共享工具与 CLI。
本示例强调：优先进行逻辑推理（不观察工具结果），仅在确有需要时调用工具。
"""

import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from typing import List
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults

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


try:
    search = TavilySearchResults(max_results=3)
except Exception:
    search = None

class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


def get_plan(state: ReWOO):
    task = state["task"]

    regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

    prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
    which external tool together with tool input to retrieve evidence. You can store the evidence into a \
    variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

    Tools can be one of the following:
    (1) Google[input]: Worker that searches results from Google. Useful when you need to find short
    and succinct answers about a specific topic. The input should be a search query.
    (2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
    world knowledge and common sense. Prioritize it when you are confident in solving the problem
    yourself. Input can be any instruction.

    For example,
    Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
    hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
    less than Toby. How many hours did Rebecca work?
    Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
    with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
    Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
    Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

    Begin!
    Describe your plans with rich details. Each Plan should be followed by only one #E.

    Task: {task}"""

    prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
    planner = prompt_template | get_llm()
    result = planner.invoke({"task": task})

    # Find all matches in the sample text
    matches = re.findall(regex_pattern, result.content)
    # 对齐 02 示例：写入 state 并记录 past_steps
    state["steps"] = matches
    state["plan_string"] = result.content
    past_steps = state.get("past_steps") or []
    past_steps.append(("planner", result.content))
    state["past_steps"] = past_steps
    return state


def _get_current_task(state: ReWOO):
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1


def tool_execution(state: ReWOO):
    """Worker node that executes the tools of a given plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state["results"] or {}) if "results" in state else {}
    for k, v in _results.items():
        tool_input = tool_input.replace(k, v)
    if tool == "Google":
        if search is None:
            result = "Search unavailable (missing TAVILY_API_KEY)."
        else:
            result = search.invoke(tool_input)
    elif tool == "LLM":
        result = get_llm().invoke(tool_input)
    else:
        raise ValueError
    _results[step_name] = str(result)
    # 对齐 02 示例：写入 state 并记录 past_steps
    state["results"] = _results
    past_steps = state.get("past_steps") or []
    past_steps.append((step_name, str(result)))
    state["past_steps"] = past_steps
    return state


def solve(state: ReWOO):

    solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""

    plan = ""
    for _plan, step_name, tool, tool_input in state["steps"]:
        _results = (state["results"] or {}) if "results" in state else {}
        for k, v in _results.items():
            tool_input = tool_input.replace(k, v)
            step_name = step_name.replace(k, v)
        plan += f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]"
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = get_llm().invoke(prompt)
    # 对齐 02 示例：写入 state 并记录 past_steps
    state["result"] = result.content
    past_steps = state.get("past_steps") or []
    past_steps.append(("solve", result.content))
    state["past_steps"] = past_steps
    return state

def _route(state):
    _step = _get_current_task(state)
    if _step is None:
        # We have executed all tasks
        return "solve"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "tool"


def create_workflow():

    workflow = StateGraph(ReWOO)
    workflow.add_node("plan", get_plan)
    workflow.add_node("tool", tool_execution)
    workflow.add_node("solve", solve)

    workflow.add_edge(START, "plan")

    workflow.add_edge("plan", "tool")
    workflow.add_conditional_edges("tool", _route)
    workflow.add_edge("solve", END)

    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path='blog/rewoo.png')
    return app

def main() -> None:
    load_environment()
    app = create_workflow()
    # 初始化状态：仅需提供用户输入，其余字段由工作流填充
    initial_state = {"task": ""}
    final_state = app.invoke(initial_state)
    print("Result:", final_state.get("result"))

if __name__ == "__main__":
    main()
