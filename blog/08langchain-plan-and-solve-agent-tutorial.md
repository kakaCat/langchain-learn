---
title: "LangChain 入门教程：Plan-and-Solve 代理实现详解"
description: "深入解析 Plan-and-Solve（先规划再执行）代理模式，从原理到 LangChain/LangGraph 实现，掌握规划与行动结合的智能代理构建。"
keywords:
  - LangChain
  - Plan-and-Solve
  - Agent
  - Planning
  - LangGraph
  - Tool Usage
  - Prompt Engineering
  - AI Agent
tags:
  - Tutorial
  - Agent
  - Planning
  - LLM
author: "langchain-learn"
date: "2025-10-25"
lang: "zh-CN"
canonical: "/blog/langchain-plan-and-solve-agent-tutorial"
audience: "中级 / 具备Python基础和LLM应用经验的AI工程师"
difficulty: "intermediate"
estimated_read_time: "12-18min"
topics:
  - LangChain Agents
  - Planning Pattern
  - LangGraph Workflow
  - Prompt Engineering
entities:
  - LangChain
  - LangGraph
  - Plan-and-Solve
  - Agent
  - Tools
qa_intents:
  - "什么是 Plan-and-Solve？为什么需要它？"
  - "Plan-and-Solve 如何把规划与执行结合？"
  - "如何用 LangChain/LangGraph 实现该模式？"
  - "提示词设计在 Plan/Act/Replan 中的作用是什么？"
  - "该模式有哪些局限和适用场景？"
---

# LangChain 入门教程：Plan-and-Solve Agent 实现详解

## 本页快捷跳转
- 目录：
  - [引言：为什么需要 Plan-and-Solve？](#intro)
  - [代码示例：LangChain/LangGraph 实现](#code-example)
  - [论文与理念：规划-执行的理论基础](#paper-content)
  - [提示词作用：Plan / Act / Replan](#prompt-role)
  - [缺点与局限性](#limitations)
  - [总结](#summary)

---

<a id="intro" data-alt="introduction 引言 概述 plan_and_solve"></a>
## 引言

不同的 AI Agent 模式是在模拟人的思考与行为方式。Plan-and-Solve 面向复杂问题：先规划，再执行。以“拍黄瓜”为例：

同一需求在复杂度上可能截然不同。用“拍黄瓜”对比：
- 复杂问题：要“做一道拍黄瓜菜”，需要先拟定可执行的步骤，再逐步完成：
  1) 准备食材（黄瓜、蒜、醋、酱油、香油等）
  2) 清洗与处理（拍、切）
  3) 调味拌匀（按口味调整）
  4) 品尝观察，必要时微调口味
- 简单指令：只做“拍一下黄瓜”，属于单步、无依赖，直接执行即可。

因此：简单指令适合 ReAct 或一次工具调用；复杂问题更适合 Plan‑and‑Solve——先规划、再执行，并在每步依据观察必要时重规划。

这个过程体现了：先制定计划→按步执行→基于观察调整。复杂问题的求解，本质是将目标分解成子任务，并在执行中迭代。Plan-and-Solve 负责“规划与重规划”，而每个子任务的具体动作可以由 ReAct 代理去完成。

### Agent的实现步骤

**Plan-and-Solve（先规划再执行）** 通过“先生成可执行的分步计划，再按步骤执行并根据观察结果动态重规划”的闭环，兼顾全局性与可操作性：
1. 规划（Plan）：将目标拆解为可执行的步骤序列。
2. 执行（Solve/Act）：逐步调用工具或回答，产出中间结果。
3. 观察与重规划（Observe & Replan）：基于结果调整后续步骤，直到完成。

该模式的核心价值：
- 全局视角：避免盲目工具调用，先设定策略与顺序。
- 动态适应：执行中根据观察结果迭代计划。
- 可解释性：每步都有“计划→行动→观察”的清晰链条。

---

<a id="code-example" data-alt="代码示例 实现 langchain langgraph"></a>
## 代码示例：LangChain/LangGraph 实现

下面是一个简单的 LangChain/LangGraph 实现示例，展示了如何用 Plan-and-Solve 模式解决问题。

### 环境准备

```bash
pip install langchain langgraph langchain-openai python-dotenv langchain-community

# .env 示例（DeepSeek 模型示例；你也可替换为其他兼容模型）
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0
```

requirements.txt
```text
# 核心
langgraph>=0.6.10
langchain>=0.3.27
langchain-core>=0.3.76

# OpenAI 提供商（使用 OpenAI API 时需要）
langchain-openai>=0.3.28

# 本地模型（使用 Ollama 时需要）
langchain-ollama>=0.3.10

# 社区集成
langchain-community>=0.3.31

# 常用辅助
tiktoken>=0.9.0
python-dotenv>=1.1.1
```

.env
```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=512

TAVILY_API_KEY= your_tavily_api_key_here
```

```python

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
    """返回 Tavily 搜索工具列表，优先官方包，其次社区版。"""
    try:
        from langchain_tavily import TavilySearch
        return [TavilySearch(max_results=3)]
    except Exception as e:
        raise RuntimeError(
                "无法加载 TavilySearch，请安装相关依赖并配置 TAVILY_API_KEY"
            ) from e


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
TOOLS = get_tools()

def load_environment():
    """加载当前目录下的 .env 配置。"""
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

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

You have currently done the follow steps:
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
    """执行节点：执行当前任务，记录工具调用与输出。"""
    plan = state["plan"]
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
    print("Execute Step - Agent 返回:", agent_response["messages"][-1].content)
    past_steps = state.get("past_steps") or []
    past_steps.append((task, agent_response["messages"][-1].content))
    state["past_steps"] = past_steps
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
    output_path = project_root / "blog" / "agent_02.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    app.get_graph().draw_mermaid_png(output_file_path=str(output_path))
    return app

def main() -> None:
    """运行工作流：初始化环境、执行并打印最终结果。"""
    load_environment()
    app = create_workflow()
    initial_state = {"input": "2024 年奥运会乒乓球混合双打冠军的家乡在哪里？"}
    final_state = app.invoke(initial_state)
    print("Response:", final_state.get("response"))

if __name__ == "__main__":
    main()

```

** Plan-and-Solve 构建要点 **
- plan_step：根据输入生成执行计划列表。
- execute_step（React模型）： 按照列表顺序执行每个任务，记录工具调用与输出。
- replan_step： 思考执行的命令是否符合期望，不符合则更新计划，符合则继续执行计划。
- should_end：判断是否完成任务，若完成则结束，否则继续执行。

** Plan-and-Solve 提示词特点 **
- plan_step：要求模型输出有序、完整、可执行的步骤清单，例如：制定一份简单的分步计划。
- execute_step（React模型）：在执行任务时，要求模型根据上下文与预期，选择合适的工具并执行。
- replan_step： 思考执行的命令是否符合期望，不符合则更新计划，符合则继续执行计划。
---

<a id="paper-content" data-alt="论文 理论基础 规划 执行"></a>
## 论文与理念：规划-执行的理论基础

Plan-and-Solve 属于“先规划后执行”的通用工程范式，灵感来源于经典 AI 的规划理论与现代 LLM 的链式推理：
- 先规划能提升全局最优性与步骤完整性，避免漏项与循环依赖。
- 执行中的观察反馈支撑动态调整，使得代理在复杂环境中更稳健。
- 与 ReAct 的差异：ReAct更强调“思考-行动-观察”的交替；Plan-and-Solve先拿到全局计划再执行，二者可互补。

该范式在检索问答、工具组合、任务编排等场景表现出良好可解释性与稳定性，利于调试和审计。

---

<a id="prompt-role" data-alt="提示词 设计 规划 重规划"></a>
## 提示词作用：Plan / Act / Replan

在示例代码中，提示词承担三类职责：
- 规划提示（Planner Prompt）：要求模型输出“有序、完整、可执行”的步骤清单，并用结构化模式约束格式。
- 执行提示（Agent System Prompt）：在执行某一步时，明确上下文与预期，驱动工具选择与行动。
- 重规划提示（Replanner Prompt）：结合已完成步骤与目标，决定“继续补充计划”或“直接向用户回复”。

提示词的关键要点：
- 明确目标与输出格式，减少歧义。
- 将“信息充分、无多余步骤”写入约束，提升可执行性。
- 在重规划阶段，仅保留“仍需执行”的步骤，避免重复。

---

<a id="limitations" data-alt="缺点 局限 适用场景"></a>
## 缺点与局限性

- 计划质量依赖模型：若初始计划不充分，后续需要频繁重规划。
- 较长交互成本：规划、执行、观察的闭环可能增加时延与调用次数。
- 工具选择不当：若缺乏高质量工具或检索结果，执行效果受限。
- 状态管理复杂：需要妥善维护“已完成步骤、上下文与历史”。
- 适用边界：对一次性、简单问题可能“过度工程”，直接回答更高效。

---

<a id="summary" data-alt="总结 收尾 建议"></a>
## 总结

Plan-and-Solve 以“先规划、再执行、必要时重规划”为核心，将全局策略与逐步行动结合，特别适合多步、具外部工具依赖的任务。结合 LangChain 的结构化输出与 LangGraph 的有向工作流，你可以：
- 让模型生成清晰的步骤计划并受格式约束。
- 在每步中调用合适工具，保留观察结果以便迭代。
- 通过重规划节点实现动态调整，最终稳定地收敛到答案。

实践建议：从简单目标入手，逐步丰富工具与状态管理；为关键节点编写可测试的提示与评估指标，持续提升计划质量与执行稳健性。