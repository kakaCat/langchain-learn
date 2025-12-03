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
  - Plan-Execute
  - Planning Agent
  - Replanning
  - StateGraph
  - Structured Output
  - Tool Use
  - FAQ
  - Troubleshooting
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
  - "如何切换模型并兼容版本？"
  - "工具不可用或检索失败时怎么办？"
  - "生成工作流图失败如何处理？"
  - "为什么要在执行后移除已完成的步骤？"
---

# LangChain 入门教程：Plan-and-Solve Agent 实现详解

## 本页快捷跳转

*   目录：
    *   [引言：为什么需要 Plan-and-Solve？](#intro)
    *   [论文与理念：规划-执行的理论基础](#paper-content)
    *   [代码示例：LangChain/LangGraph 实现](#code-example)
    *   [运行指南](#run-guide)
    *   [模型兼容性与切换](#model-compat)
    *   [缺点与局限性](#limitations)
    *   [官方链接的内容](#links)
    *   [FAQ 常见问题](#faq)
    *   [总结](#summary)

***

<a id="tldr" data-alt="TLDR 要点 速览 总结"></a>

### 本页摘要

*   模式定义：先生成多步计划，再按步执行并观察重规划，适合复杂、多步骤与工具依赖的任务。
*   与 ReAct 的差异：ReAct强调“思考-行动-观察”交替；Plan‑and‑Solve先拿到全局计划，二者互补。
*   三节点闭环：Planner（JSON步骤）→ Agent（执行与工具调用）→ Replanner（返回答案或续步）。
*   环境与依赖：统一使用 `pip install -r requirements.txt`；`.env` 包含 `OPENAI_*` 与 `TAVILY_API_KEY`。
*   工具降级策略：优先 `langchain-tavily`，其次 `langchain_community`，失败时进入无工具模式继续执行。
*   模型切换：OpenAI/DeepSeek 云端或 Ollama 本地（切换至 `ChatOllama`）。
*   输出产物：终端日志与 `blog/agent_02.png` 工作流图，最后打印“最终结果”。

<a id="intro" data-alt="introduction 引言 概述 plan_and_solve"></a>

## 引言

不同的 AI Agent 模式是在模拟人的思考与行为方式。Plan-and-Solve 面向复杂问题：先规划，再执行。以“拍黄瓜”为例：

同一需求在复杂度上可能截然不同。用“拍黄瓜”对比：

*   复杂问题：要“做一道拍黄瓜菜”，需要先拟定可执行的步骤，再逐步完成：
    1.  准备食材（黄瓜、蒜、醋、酱油、香油等）
    2.  清洗与处理（拍、切）
    3.  调味拌匀（按口味调整）
    4.  品尝观察，必要时微调口味
*   简单指令：只做“拍一下黄瓜”，属于单步、无依赖，直接执行即可。

因此：简单指令适合 ReAct 或一次工具调用；复杂问题更适合 Plan‑and‑Solve——先规划、再执行，并在每步依据观察必要时重规划。

这个过程体现了：先制定计划→按步执行→基于观察调整。复杂问题的求解，本质是将目标分解成子任务，并在执行中迭代。Plan-and-Solve 负责“规划与重规划”，而每个子任务的具体动作可以由 ReAct 代理去完成。

### Agent的实现步骤

**Plan-and-Solve（先规划再执行）** 通过“先生成可执行的分步计划，再按步骤执行并根据观察结果动态重规划”的闭环，兼顾全局性与可操作性：

1.  规划（Plan）：将目标拆解为可执行的步骤序列。
2.  执行（Solve/Act）：逐步调用工具或回答，产出中间结果。
3.  观察与重规划（Observe & Replan）：基于结果调整后续步骤，直到完成。

该模式的核心价值：

*   全局视角：避免盲目工具调用，先设定策略与顺序。
*   动态适应：执行中根据观察结果迭代计划。
*   可解释性：每步都有“计划→行动→观察”的清晰链条。

***

<a id="paper-content" data-alt="论文 理论基础 规划 执行"></a>

## 论文与理念：规划-执行的理论基础 [Plan\_And\_Solve论文](https://arxiv.org/abs/2305.04091)

Plan-and-Solve 属于“先规划后执行”的通用工程范式，灵感来源于经典 AI 的规划理论与现代 LLM 的链式推理：

*   先规划能提升全局最优性与步骤完整性，避免漏项与循环依赖。
*   执行中的观察反馈支撑动态调整，使得代理在复杂环境中更稳健。
*   与 ReAct 的差异：ReAct更强调“思考-行动-观察”的交替；Plan-and-Solve先拿到全局计划再执行，二者可互补。

该范式在检索问答、工具组合、任务编排等场景表现出良好可解释性与稳定性，利于调试和审计。

***

<a id="code-example" data-alt="代码示例 实现 langchain langgraph"></a>

## 代码示例：LangChain/LangGraph 实现 [Plan\_And\_Solve代码](https://github.com/kakaCat/langchain-learn/blob/main/10-agent-examples/02_plan_and_solve_demo.py)

架构上是如图所示：
![graph_plan_and_solve.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/8d40c45c920441a8a053b76151edb478~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5Lik5LiH5LqU5Y2D5Liq5bCP5pe2:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiMzk2NjY5MzY4Mjk3MTg3MCJ9&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1761755968&x-orig-sign=7C%2BZCTPblfhDmu9QrZVlDN4itCk%3D)
每个大脑就是调用一次大模型，其中通过计划列表可以减少大模型的幻觉，了解大模型要执行的计划更好的控制模型的执行。通过React模型执行任务，最后通过观察结果，动态调整计划，直到完成目标。

下面是一个简单的 LangChain/LangGraph 实现示例，展示了如何用 Plan-and-Solve 模式解决问题。

### 环境准备

```bash
pip install -r requirements.txt
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

# 搜索工具（Tavily 官方包）
langchain-tavily>=0.3.10

# 常用辅助
tiktoken>=0.9.0
python-dotenv>=1.1.1
```

.env

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
OPENAI_TEMPERATURE=0
OPENAI_MAX_TOKENS=512

TAVILY_API_KEY=your_tavily_api_key_here
```
Langgraph的流程节点图



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
    """返回 Tavily 搜索工具列表，优先官方包，其次社区版；失败则空工具继续。"""
    # 优先使用官方 `langchain-tavily`
    try:
        from langchain_tavily import TavilySearch
        return [TavilySearch(max_results=3)]
    except Exception:
        # 回退到社区版工具 `langchain_community`
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            return [TavilySearchResults(k=3)]
        except Exception:
            print("[warn] 无法加载 TavilySearch 工具，继续无工具模式。请检查 TAVILY_API_KEY 与依赖。")
            return []


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
    print("最终结果:", final_state.get("response"))

if __name__ == "__main__":
    main()

```

<a id="run-guide" data-alt="运行 安装 指南 依赖 env"></a>

## 运行指南

*   安装依赖：`pip install -r requirements.txt`
*   配置环境：在项目根或脚本同目录创建 `.env`，正确设置 `OPENAI_*` 与 `TAVILY_API_KEY`（注意不要在键值前后留空格）。
*   运行示例：`python 10-agent-examples/02_plan_and_solve_demo.py`
*   预期输出：终端打印规划、执行与重规划过程日志，并在 `blog/agent_02.png` 生成工作流图，最后输出“最终结果”。
*   图形依赖与排错：若工作流图生成失败，请根据系统提示安装图形渲染依赖（如 Graphviz/Mermaid 渲染链），或在无图模式下仅查看终端日志与最终结果。

<a id="model-compat" data-alt="模型 兼容 切换 openai deepseek ollama"></a>

## 模型兼容性与切换

*   OpenAI 示例（官方接口）：
    ```text
    OPENAI_API_KEY=your_openai_api_key
    OPENAI_BASE_URL=https://api.openai.com
    OPENAI_MODEL=gpt-3.5-turbo
    OPENAI_TEMPERATURE=0
    ```
*   DeepSeek 示例（已在上文给出）：
    ```text
    OPENAI_API_KEY=your_api_key_here
    OPENAI_BASE_URL=https://api.deepseek.com
    OPENAI_MODEL=deepseek-chat
    OPENAI_TEMPERATURE=0
    ```
*   Ollama 示例（需改用 `ChatOllama`；本示例默认用 `ChatOpenAI`）：
    ```python
    from langchain_ollama import ChatOllama
    def get_llm():
        return ChatOllama(model="qwen2.5:7b", temperature=0)
    ```
*   使用提示：
    *   选择云端提供商时，确保 `OPENAI_BASE_URL` 指向对应的服务地址，`OPENAI_MODEL` 为该服务支持的模型名。
    *   若切换到本地 Ollama，请在代码中将 `ChatOpenAI` 替换为 `ChatOllama`，并安装/拉取相应本地模型。

\*\* Plan-and-Solve 构建要点 \*\*

*   plan\_step：根据输入生成执行计划列表。
*   execute\_step（React模型）： 按照列表顺序执行每个任务，记录工具调用与输出；执行后显式移除已完成步骤，避免重复执行、减轻与重规划的交叠。
*   replan\_step： 思考执行的命令是否符合期望，不符合则更新计划，符合则继续执行计划。
*   should\_end：判断是否完成任务，若完成则结束，否则继续执行。

\*\* Plan-and-Solve 提示词特点 \*\*

*   规划提示（Planner Prompt）：要求模型输出有序、完整、可执行的步骤清单，例如：制定一份简单的分步计划。
*   执行提示（Agent System Prompt）：在执行任务时，要求模型根据上下文与预期，选择合适的工具并执行。
*   重规划提示（Replanner Prompt）：把执行任务的内容、计划列表、执行结果，让模型思考判断是否符合期望，不符合则更新计划，符合则继续执行计划。

***

<a id="limitations" data-alt="缺点 局限 适用场景"></a>

## 缺点与局限性

*   计划质量依赖模型：若初始计划不充分，后续需要频繁重规划。
*   较长交互成本：规划、执行、观察的闭环可能增加时延与调用次数。
*   工具选择不当：若缺乏高质量工具或检索结果，执行效果受限。
*   状态管理复杂：需要妥善维护“已完成步骤、上下文与历史”。
*   适用边界：对一次性、简单问题可能“过度工程”，直接回答更高效。

***

<a id="faq" data-alt="FAQ 常见 问题 排错"></a>

## FAQ 常见问题

### Plan‑and‑Solve 与 ReAct 的核心区别是什么？

Plan‑and‑Solve 先拿到全局计划再执行，强调分步计划与重规划；ReAct 更强调思考-行动-观察的交替。复杂多步任务优先 Plan‑and‑Solve，单步或无需全局规划的任务可用 ReAct。

### 什么时候应该重规划，什么时候直接返回答案？

当执行结果显示目标已达成或可以直接回答时返回答案；否则，在信息不足、工具结果不理想或计划不完整时进行重规划并仅保留未完成步骤。

### 工具不可用或检索失败怎么办？

本示例提供了降级策略：优先 `langchain-tavily`，次选 `langchain_community`，最终进入无工具模式继续执行；同时在日志中给出提示以便排查 API Key 与依赖问题。

### 如何让 Planner 输出严格 JSON？

使用结构化输出（如 `with_structured_output` + Pydantic），并在提示词中要求“返回纯 JSON，不包含多余文本”，确保解析稳定性与对齐性。

### 为什么在 `execute_step` 要移除已完成步骤？

避免重复执行与和重规划的交叠，保持计划按序推进；这也减少状态管理复杂度，提升闭环收敛速度。

### 如何切换到本地 Ollama 并选模型？

将 `ChatOpenAI` 替换为 `ChatOllama`，并在本地拉取所需模型（如 `qwen2.5:7b`）；无需配置 `OPENAI_BASE_URL`，但需确保 Ollama 服务运行正常。

### 生成的工作流图失败怎么处理？

优先按系统提示安装图形渲染依赖；如仍失败，可暂时跳过图生成，仅使用终端日志与最终结果进行验证。

### 常见报错如何排查？

检查 API Key 与模型名是否正确、网络是否可用；若调用超时，适当增加 `timeout/request_timeout` 或减少 `max_tokens`；工具不可用时检查依赖与环境变量。
\*\* 成功检查清单 \*\*

*   终端出现规划日志：包含“Planner 原始返回”和计划步骤。
*   执行节点日志：显示“当前计划”“执行任务”“Agent 返回”。
*   重规划日志：出现“决策: 更新计划”或“返回最终答案”。
*   工作流图生成：`agent_02.png` 存在并能打开。
*   最终输出：打印“最终结果: …”。

<a id="links"></a>

## 官方链接的内容

*   LangChain 文档：<https://python.langchain.com/docs>
*   LangChain Agents 指南：<https://python.langchain.com/docs/use_cases/agents>
*   LangChain 结构化输出（Structured Output）：<https://python.langchain.com/docs/guides/structured_output>
*   LangChain 工具（Tools）集成：<https://python.langchain.com/docs/integrations/tools>
*   LangGraph 文档主页：<https://langchain-ai.github.io/langgraph/>
*   LangGraph 状态图（StateGraph）与概念：<https://langchain-ai.github.io/langgraph/concepts/>
*   LangGraph 可视化与图导出：<https://langchain-ai.github.io/langgraph/concepts/visualization/>
*   Plan-and-Solve 论文（arXiv）：<https://arxiv.org/abs/2305.04091>
*   Tavily 官方文档：<https://docs.tavily.com/>
*   OpenAI API 文档：<https://platform.openai.com/docs/>
*   DeepSeek API 文档：<https://api-docs.deepseek.com/>
*   Ollama 本地模型文档：<https://ollama.com/docs>
*   Pydantic 文档：<https://docs.pydantic.dev/latest/>
*   python-dotenv 文档：<https://saurabh-kumar.com/python-dotenv/>

<a id="summary" data-alt="总结 收尾 建议"></a>

## 总结

Plan-and-Solve 以“先规划、再执行、必要时重规划”为核心，将全局策略与逐步行动结合，特别适合多步、具外部工具依赖的任务。结合 LangChain 的结构化输出与 LangGraph 的有向工作流，你可以：

*   让模型生成清晰的步骤计划并受格式约束。
*   在每步中调用合适工具，保留观察结果以便迭代。
*   通过重规划节点实现动态调整，最终稳定地收敛到答案。

实践建议：从简单目标入手，逐步丰富工具与状态管理；为关键节点编写可测试的提示与评估指标，持续提升计划质量与执行稳健性。
