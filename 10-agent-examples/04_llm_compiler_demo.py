#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_llm_compiler_demo.py — LangChain v1 create_agent 版本（LLM-Compiler 思路）

与 01/02/03 保持一致：统一工具、统一 CLI 与消息输出提取。
本示例强调：先将问题“编译”为可执行子任务（要点式），再调用工具执行，最后合并答案。
"""
import itertools
import os
import re
import time
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union, Sequence
class Task(TypedDict):
    idx: int
    tool: Union[BaseTool, str]
    args: Any
    dependencies: List[int]

class LLMCompilerPlanParser:
    def __init__(self, tools: Sequence[BaseTool]):
        self.tools = tools
    def invoke(self, llm_output):
        # Fallback: return no tasks to keep demo running without custom parser
        return []
    def stream(self, llm_output):
        # Provide an empty iterator so upstream StopIteration is handled
        return iter([])

from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    AIMessage
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph.message import add_messages
from typing import Annotated
from pydantic import BaseModel, Field



class CalculatorTool(BaseTool):
    name = "calculator"
    description = "calculator(expression) - evaluate simple arithmetic expressions."

    def _run(self, expression: str, run_manager=None):
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"ERROR(Calculator failed: {e})"

    async def _arun(self, expression: str, run_manager=None):
        return self._run(expression)


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
    search = TavilySearchResults(
        max_results=1,
        description='tavily_search_results_json(query="the search query") - a search engine.',
    )
except Exception:
    class DummySearch(BaseTool):
        name = "tavily_search_results_json"
        description = 'tavily_search_results_json(query="the search query") - a search engine.'
        def _run(self, query: str, run_manager=None):
            return f"Search unavailable (missing TAVILY_API_KEY). Query: {query}"
        async def _arun(self, query: str, run_manager=None):
            return self._run(query)
    search = DummySearch()

calculate = CalculatorTool()

# (removed example invocation)

tools = [search, calculate]




def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # $1 or ${1} -> 1
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback

        observation = traceback.format_exception()  # repr(e) +
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke, dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
def plan_and_schedule(state):
    messages = state["messages"]
    tasks = get_planner().stream(messages)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    return {"messages": scheduled_tasks}




class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # You can optionally add examples
llm = ChatOpenAI(model="gpt-4o")

runnable = joiner_prompt | llm.with_structured_output(
    JoinOutputs, method="function_calling"
)

# (duplicate block removed)

def get_planner():
    base_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a planning assistant. {replan}\nAvailable tools ({num_tools}):\n{tool_descriptions}"
        ),
        ("placeholder", "{messages}")
    ])
    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n" for i, tool in enumerate(tools)
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=(' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
                "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
                'You MUST use these information to create the next plan under "Current Plan".\n'
                ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
                " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
                " - You must continue the task index from the end of the previous one. Do not repeat task indices."),
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )
    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | get_llm()
        | LLMCompilerPlanParser(tools=tools)
    )

def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

def wrap_messages(state: list):
    return {"messages": state}

def wrap_and_get_last_index(state: list):
    next_task = 0
    for message in state[::-1]:
        if isinstance(message, FunctionMessage):
            next_task = message.additional_kwargs["idx"] + 1
            break
    state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
    return {"messages": state}

def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}"
                )
            ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state) -> dict:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


def get_joiner():
    return select_recent_messages | runnable | _parse_joiner_output

def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"

class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_workflow():

    workflow = StateGraph(State)

    # 1.  Define vertices
    # We defined plan_and_schedule above already
    # Assign each node to a state variable to update
    workflow.add_node("plan_and_schedule", plan_and_schedule)
    workflow.add_node("join", get_joiner())


    ## Define edges
    workflow.add_edge("plan_and_schedule", "join")

    ### This condition determines looping logic

    workflow.add_conditional_edges(
        "join",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
    )
    workflow.add_edge(START, "plan_and_schedule")
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path='blog/flow_01.png')
    return app

def main() -> None:
    load_environment()
    """运行工作流并输出最终结果"""
    app = create_workflow()

    for step in app.stream(
        {
            "messages": [
                HumanMessage(
                    content="Find the current temperature in Tokyo, then, respond with a flashcard summarizing this information"
                )
            ]
        }
    ):
        print(step)
    # # 初始化状态：仅需提供用户输入，其余字段由工作流填充
    # initial_state = {"input": "用一句话解释 Plan-and-Solve 是什么？"}
    # final_state = app.invoke(initial_state)
    # print("Response:", step)

if __name__ == "__main__":
    main()