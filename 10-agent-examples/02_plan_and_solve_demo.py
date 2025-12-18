#!/usr/bin/env python3
"""Plan-and-Solve 代理（无记忆版）

- 规划：根据用户输入生成分步计划（Plan）。
- 执行：按步骤执行，必要时调用 Tavily 搜索工具。
- 重规划：根据执行结果决定返回最终答案（Response）或继续计划。

在 `blog/agent_02.png` 生成工作流图，并在终端输出最终答案。
"""
import operator
import os
from pathlib import Path
from typing import Union, Annotated, List, Tuple

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


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
    """加载当前目录下的 .env 配置。"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

# 获取配置的语言模型
def get_llm() -> ChatOpenAI:
    """创建并返回语言模型实例。"""
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
        "verbose": True,
        "base_url": base_url
    }

    return ChatOpenAI(**kwargs)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


class Plan (BaseModel):
    """具体执行的分步计划。"""
    steps: list[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

def get_plan_agent() :
    """构造规划代理，返回生成 `Plan` 的链条。"""

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """For the given objective, come up with a simple step by step plan. \
The result of the final step should be the final answer.

If the plan is already 'one step', leave as is.
Otherwise, your role is to break down the plan into granular steps, making it easier to check.
Return a pure JSON object with field `steps` as an array of strings. Do not include any extra text.
                """,
            ),
            ("placeholder", "{messages}"),
        ]
    )

    llm = get_llm()
    return planner_prompt | llm.with_structured_output(Plan, method="function_calling")



def get_react_agent(system_prompt: str, tools: list = TOOLS) :
    """构造执行代理，支持调用工具完成具体任务。"""
    llm = get_llm()

    return  create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt
    )

class Response(BaseModel):
    """返回给用户的最终答案。"""

    response: str

class Act(BaseModel):
    """重规划阶段的动作，可能是 `Response` 或 `Plan`。"""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


def get_replanner_agent():
    """构造重规划代理，返回生成 `Act` 的链条。"""


    replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Return a pure JSON object with field `action` containing either:
- {{"response": "..."}} (if you can provide the final answer)
- {{"steps": ["...", "..."]}} (if more steps are needed)
Do not include any extra text.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the following steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)
    llm = get_llm()

    return replanner_prompt | llm.with_structured_output(Act, method="function_calling")


def plan_step(state: PlanExecute):
    """规划节点：生成计划、记录步骤并更新状态。"""
    planner = get_plan_agent()
    result = planner.invoke({"messages": [("user", state["input"]) ]})
    print(f"Planner 原始返回: {result}")
    plan_obj = result
    print(f"解析后的计划步骤: {plan_obj.steps}")
    state["plan"] = plan_obj.steps
    past_steps = state.get("past_steps") or []
    past_steps.append(("planner", "\n".join(plan_obj.steps)))
    state["past_steps"] = past_steps
    return state




def execute_step(state: PlanExecute):
    """执行节点：执行当前任务，记录工具调用与输出，并维护计划进度。"""
    plan = state.get("plan") or []
    if not plan:
        print("Execute Step - 计划为空，跳过执行")
        return state
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:\
    {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    print("Execute Step - 当前计划:\n", plan_str)
    print("Execute Step - 执行任务:", task)
    agent_response = get_react_agent("You are a helpful assistant.").invoke(
        {"messages": [("user", task_formatted)]}
    )
    used_tools = []
    for msg in agent_response.get("messages", []):
        try:
            if isinstance(msg, ToolMessage):
                if getattr(msg, "name", None):
                    used_tools.append(msg.name)
                else:
                    used_tools.append("<unknown_tool>")
            elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = (
                        getattr(tc, "name", None)
                        or (tc.get("name") if isinstance(tc, dict) else None)
                        or getattr(getattr(tc, "tool", None), "name", None)
                    )
                    if name:
                        used_tools.append(name)
        except Exception:
            pass
    if used_tools:
        print("Execute Step - 调用了工具:", ", ".join(used_tools))
    else:
        print("Execute Step - 未调用任何工具")
    # 提取最后一条模型消息内容，增强稳健性
    last_content = None
    try:
        msgs = agent_response.get("messages", [])
        if msgs:
            last_content = msgs[-1].content
    except Exception:
        pass
    if last_content is None:
        last_content = str(agent_response)
    print("Execute Step - Agent 返回:", last_content)
    past_steps = state.get("past_steps") or []
    past_steps.append((task, last_content))
    state["past_steps"] = past_steps
    # 移除已执行的第一步，保持计划前进
    if state.get("plan"):
        state["plan"] = state["plan"][1:]
    return state




def replan_step(state: PlanExecute):
    """重规划节点：决定返回最终答案或更新计划，并记录步骤。"""
    act_obj = get_replanner_agent().invoke(state)
    if isinstance(act_obj.action, Response):
        print("Replan Step - 决策: 返回最终答案")
        print("Replan Step - 答案:", act_obj.action.response)
        state["response"] = act_obj.action.response
        past_steps = state.get("past_steps") or []
        past_steps.append(("replanner", act_obj.action.response))
        state["past_steps"] = past_steps
        return state
    else:
        print("Replan Step - 决策: 更新计划")
        print("Replan Step - 新计划步骤:", act_obj.action.steps)
        state["plan"] = act_obj.action.steps
        past_steps = state.get("past_steps") or []
        past_steps.append(("replanner", "\n".join(act_obj.action.steps)))
        state["past_steps"] = past_steps
        return state


def should_end(state: PlanExecute):
    """若 `response` 非空则结束，否则继续到执行节点。"""
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"




def create_workflow():
    """构建并编译工作流，生成可执行 `app` 与工作流图。"""

    workflow = StateGraph(PlanExecute)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        should_end,
        ["agent", END],
    )
    app = workflow.compile()
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "blog" / "plan.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app.get_graph().draw_mermaid_png(output_file_path=str(output_path))
    return app

def main() -> None:
    """运行工作流：初始化环境、执行并打印最终结果。"""
    load_environment()
    app = create_workflow()
    initial_state = {"input": "2024 年奥运会乒乓球混合双打冠军的家乡在哪里？"}
    final_state = app.invoke(initial_state)
    print("最终结果:", final_state.get("response"))

if __name__ == "__main__":
    main()



