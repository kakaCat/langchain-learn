#!/usr/bin/env python3
"""
11 - Claude Code Style Hierarchical Agent Demo

目标：
模拟 Claude Code 文档中展示的“Lead Researcher + 动态子 Agent”流程，
包含：任务规划、按需派生子 Agent、重复的研究循环、记忆/引用存储以及最终引用总结。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field


def load_environment() -> None:
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def get_llm(model: Optional[str] = None, temperature: float = 0.2) -> object:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            verbose=True,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "800"))

        return ChatOpenAI(
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


class ResearchMemory(BaseModel):
    aspect: str
    summary: str
    citations: str
    agent: str


class ClaudeCodeState(BaseModel):
    user_request: str
    plan: List[str] = Field(default_factory=list)
    backlog: List[str] = Field(default_factory=list)
    current_aspect: Optional[str] = None
    current_agent_brief: Optional[Dict[str, str]] = None
    memory: List[ResearchMemory] = Field(default_factory=list)
    research_logs: List[str] = Field(default_factory=list)
    loop_count: int = 0
    continue_research: bool = True
    final_report: Optional[str] = None


# Lead Researcher Nodes -----------------------------------------------------

def plan_research_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: create plan of aspects."""
    if state.plan:
        return state

    llm = get_llm()
    prompt = f"""
你是 Lead Researcher。请将用户需求拆解为 3-5 个关键研究方面，按优先级排序。
用户需求：{state.user_request}
用 JSON 数组输出，例如：
["方面A：...", "方面B：..."]
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        plan = json.loads(response.content)
        if not isinstance(plan, list):
            raise ValueError
    except Exception:
        plan = [state.user_request]

    state.plan = [str(item) for item in plan]
    state.backlog = state.plan.copy()
    state.research_logs.append(f"[Lead] 生成研究规划：{state.plan}")
    return state


def pick_aspect_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: select next aspect from backlog."""
    if state.backlog:
        state.current_aspect = state.backlog.pop(0)
    else:
        state.current_aspect = None
    return state


def spawn_subagent_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: dynamically create a subagent brief for the aspect."""
    if not state.current_aspect:
        return state

    llm = get_llm()
    prompt = f"""
用户总体请求：{state.user_request}
当前研究方面：{state.current_aspect}
现有记忆：{[m.summary for m in state.memory] or "（无）"}

请以 JSON 输出一个子 Agent 设计，包含字段：
{{
  "name": "...",
  "focus": "...",
  "instructions": "...",  # 说明如何执行（包含 ReAct/自我反思要求）
  "expected_tools": "..."  # 说明将使用的工具或方法
}}
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        brief = json.loads(response.content)
    except json.JSONDecodeError:
        brief = {
            "name": "Generalist",
            "focus": state.current_aspect,
            "instructions": "使用 ReAct 风格，思考->行动->观察，并在结束前反思。",
            "expected_tools": "internal reasoning",
        }

    state.current_agent_brief = brief
    state.research_logs.append(f"[Lead] 为 {state.current_aspect} 派生子 Agent：{brief.get('name')}")
    return state


# SubAgent execution -------------------------------------------------------

def subagent_execution_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Spawned SubAgent executes using ReAct + reflection pattern."""
    if not (state.current_aspect and state.current_agent_brief):
        return state

    llm = get_llm(temperature=0)
    memory_snippets = "\n".join(f"- {m.aspect}: {m.summary}" for m in state.memory[-5:])

    system_prompt = f"""
你是子 Agent {state.current_agent_brief.get('name')}，职责：{state.current_agent_brief.get('focus')}
执行要求：{state.current_agent_brief.get('instructions')}
你必须使用 ReAct 风格记录思考、行动、观察，并在结束时反思。
"""
    human_prompt = f"""
用户总请求：{state.user_request}
当前子任务：{state.current_aspect}
最近的研究笔记：
{memory_snippets or "（无）"}

请输出 JSON：
{{
  "thoughts": ["..."],
  "actions": ["..."],
  "observations": ["..."],
  "result_summary": "...",
  "citations": "...",
  "next_suggestions": "..."
}}
"""

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    )
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {
            "thoughts": [response.content],
            "actions": [],
            "observations": [],
            "result_summary": response.content,
            "citations": "N/A",
            "next_suggestions": "请 Lead Researcher 复核。",
        }

    note = ResearchMemory(
        aspect=state.current_aspect,
        summary=parsed.get("result_summary", ""),
        citations=parsed.get("citations", ""),
        agent=state.current_agent_brief.get("name", "Unknown"),
    )
    state.memory.append(note)

    log = (
        f"[SubAgent:{note.agent}] 完成 {note.aspect}\n"
        f"思考: {parsed.get('thoughts')}\n"
        f"行动: {parsed.get('actions')}\n"
        f"观察: {parsed.get('observations')}\n"
        f"总结: {note.summary}"
    )
    state.research_logs.append(log)
    return state


# Lead reflection + loop control ------------------------------------------

def lead_reflection_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher reflects on subagent output, decides on follow-ups."""
    if not state.memory:
        return state

    llm = get_llm()
    latest = state.memory[-1]
    prompt = f"""
你是 Lead Researcher。刚刚收到子 Agent {latest.agent} 对“{latest.aspect}”的结果：
总结：{latest.summary}
引用：{latest.citations}

请评估：
1. 该结果是否可信并可纳入最终报告
2. 是否需要追加研究（True/False）
3. 如果需要，列出新的研究方面（最多2个）

输出 JSON：
{{
  "accepted": true,
  "need_more_research": false,
  "new_aspects": ["..."],
  "comment": "..."
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        verdict = json.loads(response.content)
    except json.JSONDecodeError:
        verdict = {
            "accepted": True,
            "need_more_research": False,
            "new_aspects": [],
            "comment": "无法解析，默认接受。",
        }

    state.research_logs.append(
        f"[Lead] 审核 {latest.aspect}: accepted={verdict.get('accepted')} note={verdict.get('comment')}"
    )

    if verdict.get("new_aspects"):
        for aspect in verdict["new_aspects"]:
            if aspect not in state.backlog:
                state.backlog.append(aspect)
        state.research_logs.append(f"[Lead] 新增研究方面：{verdict['new_aspects']}")

    state.continue_research = bool(verdict.get("need_more_research")) or bool(state.backlog)
    state.loop_count += 1
    return state


def final_report_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Citation Agent: synthesize final report with citations."""
    llm = get_llm()
    notes = "\n".join(
        f"- {m.aspect}（{m.agent}）: {m.summary} 引用: {m.citations}" for m in state.memory
    )
    prompt = f"""
你是 Citation Agent，需要基于研究记忆编写最终报告，引用记忆中的信息。

用户请求：{state.user_request}
研究记忆：
{notes or "（无）"}

输出需要包含：
1. 结构化摘要
2. 关键发现
3. 引用注释（可引用原始记忆中的 citations 字段）
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state.final_report = response.content
    state.research_logs.append("[CitationAgent] 完成最终报告。")
    return state


# Graph --------------------------------------------------------------------

MAX_LOOPS = 6


def create_claude_code_workflow():
    graph = StateGraph(ClaudeCodeState)
    graph.add_node("plan", plan_research_node)
    graph.add_node("pick", pick_aspect_node)
    graph.add_node("spawn", spawn_subagent_node)
    graph.add_node("execute", subagent_execution_node)
    graph.add_node("reflect", lead_reflection_node)
    graph.add_node("final", final_report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "pick")
    graph.add_conditional_edges(
        "pick",
        lambda state: "spawn" if state.current_aspect else "final",
    )
    graph.add_edge("spawn", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "pick"
        if state.continue_research and state.loop_count < MAX_LOOPS
        else "final",
    )
    graph.add_edge("final", END)
    return graph.compile()


def run_claude_code_demo() -> None:
    load_environment()
    workflow = create_claude_code_workflow()

    tasks = [
        "研究 2025 年学习 Rust 的最佳路径，并给出推荐资料",
        "制定一份生成式 AI 产品发布会的准备清单",
    ]

    for idx, task in enumerate(tasks, 1):
        print(f"\n=== Claude Code Style Demo #{idx} ===")
        print(f"用户任务：{task}")

        state = ClaudeCodeState(user_request=task)
        try:
            final_state = workflow.invoke(state)
            print("--- Research Memory ---")
            for note in final_state["memory"]:
                print(f"{note.aspect} ({note.agent}): {note.summary} | {note.citations}")
            print("--- Final Report ---")
            print(final_state["final_report"])
        except Exception as exc:
            print(f"执行出错：{exc}")


if __name__ == "__main__":
    run_claude_code_demo()
