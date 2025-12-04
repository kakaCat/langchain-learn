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
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


# ============== LLM 单例模式实现 ==============
# 适用于 LangGraph 单线程顺序执行场景
# httpx 会自动处理 HTTP 连接池复用

_llm_instances: Dict[tuple, object] = {}


def get_llm(model: Optional[str] = None, temperature: float = 0.2, json_mode: bool = False) -> object:
    """
    获取 LLM 单例实例

    特性:
    - 每个配置（model + temperature + json_mode）只创建一个实例
    - httpx 自动处理 HTTP 连接池复用
    - 适用于单线程顺序执行（LangGraph 默认模式）

    Args:
        model: 模型名称（None 则使用环境变量）
        temperature: 温度参数
        json_mode: 是否启用 JSON 模式（仅 OpenAI 支持，强制返回合法 JSON）

    Returns:
        ChatOpenAI 或 ChatOllama 实例
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = (provider in {"ollama", "local"}) and not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        cache_key = ("ollama", model_name, temperature, False)  # Ollama 不支持 JSON mode
    else:
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        cache_key = ("openai", model_name, temperature, json_mode)

    # 单例模式：如果已创建则直接返回
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

            # ✅ 配置 JSON Mode（OpenAI 原生支持，从根本上解决 JSON 解析问题）
            model_kwargs = {}
            if json_mode:
                model_kwargs["response_format"] = {"type": "json_object"}

            print(f"[LLM] 创建 OpenAI 单例 model={model_name} temp={temperature} json_mode={json_mode}")
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
                model_kwargs=model_kwargs,  # ✅ 传递 JSON Mode 配置
            )
    else:
        print(f"[LLM] 复用单例 {cache_key}")

    return _llm_instances[cache_key]


class ResearchMemory(BaseModel):
    aspect: str
    summary: str
    citations: str
    agent: str


# ============== Structured Output Models ==============

class ReflectionVerdict(BaseModel):
    """Lead Researcher 的评估结果（用于强制 JSON 输出）"""
    accepted: bool = Field(description="是否接受该研究结果")
    need_more_research: bool = Field(description="是否需要更多研究")
    new_aspects: List[str] = Field(default_factory=list, description="新的研究方面列表")
    comment: str = Field(description="评估意见")


class ClaudeCodeState(BaseModel):
    user_request: str
    plan: list[str] = Field(default_factory=list)
    backlog: list[str] = Field(default_factory=list)
    current_aspect: str | None = None
    current_agent_brief: dict[str, str] | None = None
    memory: list[ResearchMemory] = Field(default_factory=list)
    research_logs: list[str] = Field(default_factory=list)
    loop_count: int = 0
    continue_research: bool = True
    final_report: str | None = None


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
    print(f"[Lead] 规划研究：{state.plan}")
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
    print(f"[Lead] 派生子 Agent：{brief}")
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
    print(f"[SubAgent:{note.agent}] 完成 {parsed}")
    return state


# Lead reflection + loop control ------------------------------------------

def lead_reflection_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher reflects on subagent output, decides on follow-ups."""
    if not state.memory:
        return state

    # ✅ 使用 JSON Mode（OpenAI 原生支持，从根本上解决 JSON 解析问题）
    llm = get_llm(json_mode=True)
    latest = state.memory[-1]

    # ✅ 改进的 Prompt：明确说明 JSON 结构
    prompt = f"""
请评估子 Agent {latest.agent} 对"{latest.aspect}"的研究结果。

研究总结：{latest.summary}
引用来源：{latest.citations}

请以 JSON 格式输出评估结果，包含以下字段：
- accepted: 布尔值，表示是否接受该研究结果并纳入最终报告
- need_more_research: 布尔值，表示是否需要追加研究
- new_aspects: 字符串数组，列出新的研究方面（最多2个，如不需要则为空数组）
- comment: 字符串，你的评估意见

示例格式：
{{"accepted": true, "need_more_research": false, "new_aspects": [], "comment": "结果详实可信"}}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        # ✅ JSON Mode 保证返回合法 JSON，可以直接解析
        verdict_dict = json.loads(response.content)
        print(f"✅ [Lead] JSON 解析成功: {verdict_dict}")

    except json.JSONDecodeError as e:
        # 理论上不应该进入这里（JSON Mode 保证合法 JSON）
        print(f"⚠️ [Lead] JSON 解析失败 ({e})，使用默认值")
        print(f"原始响应:\n{response.content}")
        verdict_dict = {
            "accepted": True,
            "need_more_research": False,
            "new_aspects": [],
            "comment": "JSON 解析失败，默认接受。",
        }
    except Exception as e:
        print(f"⚠️ [Lead] 调用 LLM 失败 ({e})，使用默认值")
        verdict_dict = {
            "accepted": True,
            "need_more_research": False,
            "new_aspects": [],
            "comment": f"LLM 调用失败: {str(e)}",
        }

    # 记录评估结果
    state.research_logs.append(
        f"[Lead] 审核 {latest.aspect}: {verdict_dict.get('comment')}"
    )

    # 添加新的研究方面
    if verdict_dict.get("new_aspects"):
        new_added = []
        for aspect in verdict_dict["new_aspects"]:
            if aspect and aspect not in state.backlog:
                state.backlog.append(aspect)
                new_added.append(aspect)
        if new_added:
            state.research_logs.append(f"[Lead] 新增研究方面：{new_added}")

    # 决定是否继续研究
    state.continue_research = bool(verdict_dict.get("need_more_research")) or bool(state.backlog)
    state.loop_count += 1
    print(f"[Lead] 评估完成: accepted={verdict_dict.get('accepted')}, continue={state.continue_research}")
    return state


def final_report_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Citation Agent: synthesize final report with citations."""
    llm = get_llm()
    notes = "\n".join(
        f"- {m.aspect}（{m.agent}）: {m.summary} 引用: {m.citations}" for m in state.memory
    )
    print(f"[CitationAgent] 引用记忆：{notes}")
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
    print(f"[CitationAgent] 完成最终报告：{state.final_report}")
    return state


# Graph --------------------------------------------------------------------

MAX_LOOPS = 6


def create_claude_code_workflow():
    graph = StateGraph(ClaudeCodeState)
    graph.add_node("plan", plan_research_node)  # 添加“计划”节点：生成研究规划
    graph.add_node("pick", pick_aspect_node)  # 添加“选择”节点：从待办中选择下一个方面
    graph.add_node("spawn", spawn_subagent_node)  # 添加“孵化”节点：为当前方面派生子 Agent
    graph.add_node("execute", subagent_execution_node)  # 添加“执行”节点：子 Agent 执行并记录
    graph.add_node("reflect", lead_reflection_node)  # 添加“复盘”节点：Lead 评审并决定是否继续
    graph.add_node("final", final_report_node)  # 添加“终稿”节点：生成最终报告与引用

    graph.set_entry_point("plan")  # 设置工作流入口为“计划”
    graph.add_edge("plan", "pick")  # 计划完成后进入选择阶段
    graph.add_conditional_edges(
        "pick",
        lambda state: "spawn" if state.current_aspect else "final",  # 有当前方面则孵化子 Agent，否则直接终稿
    )
    graph.add_edge("spawn", "execute")  # 孵化后进入执行
    graph.add_edge("execute", "reflect")  # 执行后进入复盘
    graph.add_conditional_edges(
        "reflect",
        lambda state: "pick"
        if state.continue_research and state.loop_count < MAX_LOOPS
        else "final",  # 若需继续且未超循环上限则回到选择，否则终稿
    )
    graph.add_edge("final", END)
    export_workflow_graph_png()  # 终稿后结束工作流
    return graph.compile()


def export_workflow_graph_png(png_path: Optional[str] = None) -> str:
    app = create_claude_code_workflow()
    default_png = os.path.join(os.path.dirname(__file__), "claude_code_workflow.png")
    out_path = png_path or default_png
    try:
        app.get_graph().draw_mermaid_png(output_file_path=out_path)
        return out_path
    except Exception:
        mermaid_code = app.get_graph().draw_mermaid()
        fallback_path = (png_path or default_png).rsplit(".", 1)[0] + ".mmd"
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        return fallback_path


def run_claude_code_demo() -> None:
    load_environment()
    workflow = create_claude_code_workflow()

    tasks = [
        "研究 2025 年学习 Rust 的最佳路径，并给出推荐资料"
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
