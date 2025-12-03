#!/usr/bin/env python3
"""
11 - Claude Code Style with Parallel SubAgent Execution

在增强版基础上，新增：
1. 并行 SubAgent 执行能力
2. 依赖关系分析（识别独立任务）
3. 批量处理节点
4. 性能监控和对比
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Set, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError


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
        max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1500"))

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


# ============================================================================
# Tool Registry (复用增强版)
# ============================================================================


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        try:
            search = DuckDuckGoSearchRun()
            self.tools["web_search"] = Tool(
                name="web_search",
                description="搜索互联网获取最新信息。输入搜索关键词，返回相关结果摘要。",
                func=search.run,
            )
        except Exception as e:
            print(f"[警告] Web 搜索工具注册失败: {e}")

        try:
            python_repl = PythonREPL()
            self.tools["python_repl"] = Tool(
                name="python_repl",
                description="执行 Python 代码进行计算、数据处理等。输入 Python 代码，返回执行结果。",
                func=python_repl.run,
            )
        except Exception as e:
            print(f"[警告] Python REPL 工具注册失败: {e}")

        def read_file(file_path: str) -> str:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                return f"读取失败: {str(e)}"

        self.tools["file_read"] = Tool(
            name="file_read",
            description="读取本地文件内容。输入文件路径，返回文件内容。",
            func=read_file,
        )

    def get_tools_for_agent(self, agent_type: str) -> List[Tool]:
        tool_mapping = {
            "researcher": ["web_search"],
            "analyst": ["python_repl", "file_read"],
            "generalist": ["web_search", "python_repl", "file_read"],
        }
        tool_names = tool_mapping.get(agent_type, ["web_search"])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_all_tools(self) -> List[Tool]:
        return list(self.tools.values())


TOOL_REGISTRY = ToolRegistry()


# ============================================================================
# Data Models (复用增强版)
# ============================================================================


class Citation(BaseModel):
    source_type: Literal["web", "file", "calculation", "reasoning"]
    source: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    snippet: Optional[str] = None


class ResearchMemory(BaseModel):
    aspect: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    agent: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    execution_time: float = Field(default=0.0)  # 新增：执行时间（秒）


class SubAgentBrief(BaseModel):
    name: str
    agent_type: Literal["researcher", "analyst", "generalist"] = "generalist"
    focus: str
    instructions: str
    expected_tools: List[str] = Field(default_factory=list)
    aspect: str  # 新增：关联的研究方面


class SubAgentResult(BaseModel):
    thoughts: List[str]
    actions_taken: List[str]
    observations: List[str]
    result_summary: str
    citations: List[Citation] = Field(default_factory=list)
    next_suggestions: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LeadReflection(BaseModel):
    accepted: bool
    need_more_research: bool
    new_aspects: List[str] = Field(default_factory=list)
    comment: str
    quality_score: float = Field(default=0.7, ge=0.0, le=1.0)


class GoalAssessment(BaseModel):
    goal_achieved: bool
    completeness: float = Field(ge=0.0, le=1.0)
    missing_aspects: List[str] = Field(default_factory=list)
    reasoning: str


# 新增：依赖关系模型
class AspectDependency(BaseModel):
    """研究方面之间的依赖关系"""
    aspect: str
    depends_on: List[str] = Field(default_factory=list)  # 依赖的其他方面
    can_parallel: bool = True  # 是否可以并行执行


class ClaudeCodeState(BaseModel):
    user_request: str
    plan: List[str] = Field(default_factory=list)
    backlog: List[str] = Field(default_factory=list)
    current_aspect: Optional[str] = None
    current_agent_brief: Optional[SubAgentBrief] = None
    memory: List[ResearchMemory] = Field(default_factory=list)
    research_logs: List[str] = Field(default_factory=list)
    loop_count: int = 0
    continue_research: bool = True
    final_report: Optional[str] = None

    # 预算监控
    total_tokens_used: int = 0
    max_tokens_budget: int = Field(default=50000)
    max_loops: int = Field(default=8)

    # 目标达成追踪
    goal_assessment: Optional[GoalAssessment] = None

    # 新增：并行执行相关
    aspect_dependencies: Dict[str, AspectDependency] = Field(default_factory=dict)
    parallel_briefs: List[SubAgentBrief] = Field(default_factory=list)  # 待并行执行的 briefs
    parallel_enabled: bool = Field(default=True)  # 是否启用并行
    max_parallel: int = Field(default=3)  # 最大并行数

    # 性能统计
    total_execution_time: float = Field(default=0.0)
    parallel_speedup: float = Field(default=1.0)  # 并行加速比


# ============================================================================
# 辅助函数 (复用增强版)
# ============================================================================


def parse_json_with_retry(
    llm: Any, messages: List, target_model: type[BaseModel], max_retries: int = 3
) -> Optional[BaseModel]:
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            content = response.content.strip()

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_data = json.loads(content)
            return target_model(**parsed_data)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                messages.append(
                    HumanMessage(
                        content=f"解析失败：{str(e)}\n请严格按照 JSON 格式重新输出，不要添加任何解释文字。"
                    )
                )
            else:
                print(f"[错误] 解析失败（尝试 {max_retries} 次）: {e}")
                return None

    return None


# ============================================================================
# 新增：依赖分析节点
# ============================================================================


def analyze_dependencies_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """分析研究方面之间的依赖关系，识别可并行执行的任务"""
    if not state.plan or not state.parallel_enabled:
        return state

    llm = get_llm()

    prompt = f"""
你是 Lead Researcher，需要分析以下研究方面之间的依赖关系。

研究计划：
{json.dumps(state.plan, ensure_ascii=False, indent=2)}

用户总请求：{state.user_request}

请分析每个研究方面，并判断它们之间的依赖关系。输出 JSON 数组：

[
  {{
    "aspect": "方面1：...",
    "depends_on": [],  // 依赖的其他方面（如果需要前置研究结果）
    "can_parallel": true  // 是否可以并行执行
  }},
  {{
    "aspect": "方面2：...",
    "depends_on": ["方面1：..."],  // 需要等方面1完成
    "can_parallel": false
  }}
]

依赖关系判断标准：
- 如果方面 A 的研究结果会影响方面 B 的研究方向，则 B depends_on A
- 如果两个方面完全独立，可以同时研究，则都标记 can_parallel=true
- 如果不确定，默认标记为可并行

示例：
计划 = ["Rust 基础资源", "Rust 实战项目", "学习路径对比"]
分析：
- "Rust 基础资源" 和 "Rust 实战项目" 可并行（互不依赖）
- "学习路径对比" 可能依赖前两者的结果（需要知道有哪些资源和项目）
"""

    messages = [HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        dependencies = json.loads(content)

        for dep_data in dependencies:
            dep = AspectDependency(**dep_data)
            state.aspect_dependencies[dep.aspect] = dep

        state.research_logs.append(
            f"[Lead] 依赖分析完成：{len([d for d in state.aspect_dependencies.values() if d.can_parallel])} 个方面可并行执行"
        )

    except Exception as e:
        # 回退：假设所有方面都可以并行
        state.research_logs.append(f"[Lead] 依赖分析失败（{str(e)}），默认所有方面可并行")
        for aspect in state.plan:
            state.aspect_dependencies[aspect] = AspectDependency(
                aspect=aspect,
                depends_on=[],
                can_parallel=True
            )

    return state


# ============================================================================
# 新增：批量派生节点
# ============================================================================


def batch_spawn_subagents_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """批量派生可并行执行的 SubAgent"""
    if not state.backlog or not state.parallel_enabled:
        return state

    # 获取可并行执行的方面
    parallel_aspects = []
    for aspect in state.backlog[:state.max_parallel]:
        dep = state.aspect_dependencies.get(aspect)
        if dep and dep.can_parallel:
            # 检查依赖是否已完成
            if all(dep_aspect in [m.aspect for m in state.memory] for dep_aspect in dep.depends_on):
                parallel_aspects.append(aspect)
        else:
            # 如果没有依赖信息，默认可并行
            parallel_aspects.append(aspect)

    if not parallel_aspects:
        # 没有可并行的方面，回退到串行
        if state.backlog:
            state.current_aspect = state.backlog.pop(0)
        return state

    # 为每个并行方面创建 SubAgent brief
    llm = get_llm()
    state.parallel_briefs = []

    for aspect in parallel_aspects:
        state.backlog.remove(aspect)

        memory_context = "\n".join(f"- {m.aspect}: {m.summary}" for m in state.memory[-3:])

        prompt = f"""
用户总体请求：{state.user_request}
当前研究方面：{aspect}
现有记忆：{memory_context or "（无）"}

请设计一个子 Agent 来完成这个研究方面。输出 JSON 格式：

{{
  "name": "Agent名称",
  "agent_type": "researcher | analyst | generalist",
  "focus": "具体研究焦点",
  "instructions": "执行指南（要求使用 ReAct）",
  "expected_tools": ["web_search", "python_repl", "file_read"],
  "aspect": "{aspect}"
}}
"""

        messages = [HumanMessage(content=prompt)]
        brief = parse_json_with_retry(llm, messages, SubAgentBrief)

        if not brief:
            brief = SubAgentBrief(
                name=f"Agent_{len(state.parallel_briefs)}",
                agent_type="generalist",
                focus=aspect,
                instructions="使用 ReAct 风格执行研究",
                expected_tools=["web_search"],
                aspect=aspect
            )

        state.parallel_briefs.append(brief)

    state.research_logs.append(
        f"[Lead] 批量派生 {len(state.parallel_briefs)} 个并行 SubAgent：{[b.name for b in state.parallel_briefs]}"
    )

    return state


# ============================================================================
# 新增：并行执行节点（异步）
# ============================================================================


async def execute_single_subagent_async(
    brief: SubAgentBrief,
    state: ClaudeCodeState
) -> ResearchMemory:
    """异步执行单个 SubAgent"""
    start_time = time.time()

    llm = get_llm(temperature=0)
    tools = TOOL_REGISTRY.get_tools_for_agent(brief.agent_type)
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)

    memory_snippets = "\n".join(
        f"- {m.aspect}: {m.summary} (置信度: {m.confidence})"
        for m in state.memory[-5:]
    )

    system_prompt = f"""
你是子 Agent {brief.name}，类型：{brief.agent_type}
职责：{brief.focus}
执行要求：{brief.instructions}

可用工具：
{tool_descriptions or "（无可用工具，仅使用内部推理）"}

你必须使用 ReAct 模式：
1. Thought（思考）：分析当前需要做什么
2. Action（行动）：选择工具并说明如何使用
3. Observation（观察）：记录行动结果
4. Reflection（反思）：评估结果质量
"""

    human_prompt = f"""
用户总请求：{state.user_request}
当前子任务：{brief.aspect}
最近的研究笔记：
{memory_snippets or "（无）"}

请按照以下 JSON 格式输出结果：

{{
  "thoughts": ["思考1", "思考2"],
  "actions_taken": ["行动1", "行动2"],
  "observations": ["观察1", "观察2"],
  "result_summary": "简洁总结（2-3句话）",
  "citations": [
    {{
      "source_type": "web|file|calculation|reasoning",
      "source": "具体来源",
      "snippet": "关键信息摘录（可选）"
    }}
  ],
  "next_suggestions": "建议 Lead Researcher 后续关注的方向（可选）",
  "confidence": 0.8
}}
"""

    # 简化版工具调用（实际生产应使用 create_react_agent）
    tool_results = []
    if "web_search" in brief.expected_tools and "web_search" in TOOL_REGISTRY.tools:
        try:
            # 运行在事件循环中
            loop = asyncio.get_event_loop()
            search_result = await loop.run_in_executor(
                None,
                TOOL_REGISTRY.tools["web_search"].func,
                brief.aspect
            )
            tool_results.append(f"[WebSearch] {search_result[:500]}")
        except Exception as e:
            tool_results.append(f"[WebSearch Error] {str(e)}")

    if tool_results:
        human_prompt += f"\n\n工具执行结果：\n" + "\n".join(tool_results)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    # LLM 调用也需要异步化
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        parse_json_with_retry,
        llm,
        messages,
        SubAgentResult,
        2
    )

    if not result:
        result = SubAgentResult(
            thoughts=["执行出错"],
            actions_taken=[],
            observations=[],
            result_summary=f"针对 '{brief.aspect}' 的研究未能完成",
            citations=[],
            confidence=0.3,
        )

    execution_time = time.time() - start_time

    note = ResearchMemory(
        aspect=brief.aspect,
        summary=result.result_summary,
        citations=result.citations,
        agent=brief.name,
        confidence=result.confidence,
        execution_time=execution_time
    )

    return note


def parallel_execute_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """并行执行所有待处理的 SubAgent（同步包装）"""
    if not state.parallel_briefs:
        return state

    async def run_parallel():
        tasks = [
            execute_single_subagent_async(brief, state)
            for brief in state.parallel_briefs
        ]
        return await asyncio.gather(*tasks)

    # 运行异步任务
    start_time = time.time()
    results = asyncio.run(run_parallel())
    total_time = time.time() - start_time

    # 添加到记忆
    for note in results:
        state.memory.append(note)
        log = (
            f"[SubAgent:{note.agent}] 完成 {note.aspect}\n"
            f"总结: {note.summary}\n"
            f"置信度: {note.confidence}\n"
            f"执行时间: {note.execution_time:.2f}s"
        )
        state.research_logs.append(log)

    # 计算加速比（理论串行时间 / 实际并行时间）
    serial_time = sum(note.execution_time for note in results)
    speedup = serial_time / total_time if total_time > 0 else 1.0
    state.parallel_speedup = speedup

    state.research_logs.append(
        f"[Parallel] 并行执行完成：{len(results)} 个任务，"
        f"总耗时 {total_time:.2f}s，"
        f"加速比 {speedup:.2f}x"
    )

    state.parallel_briefs = []  # 清空
    return state


# ============================================================================
# 复用增强版的其他节点
# ============================================================================


def plan_research_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """从增强版复制"""
    if state.plan:
        return state

    llm = get_llm()

    prompt = f"""
你是 Lead Researcher，负责将用户需求拆解为具体可执行的研究方面。

用户需求：{state.user_request}

要求：
1. 拆解为 3-5 个独立的研究方面
2. 每个方面需具体、可操作
3. 按优先级排序
4. 使用 JSON 数组格式输出

示例输出：
["方面1：Rust 基础学习资源", "方面2：实战项目推荐", "方面3：学习路径对比"]
"""

    messages = [HumanMessage(content=prompt)]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        plan = json.loads(content)
        state.plan = [str(item) for item in plan]
    except:
        state.plan = [state.user_request]

    state.backlog = state.plan.copy()
    state.research_logs.append(f"[Lead] 生成研究规划：{state.plan}")
    return state


def lead_reflection_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """简化版审核（批量）"""
    if not state.memory:
        return state

    # 批量审核最近添加的所有结果
    recent_count = len(state.parallel_briefs) if state.parallel_briefs else 1
    recent_memories = state.memory[-recent_count:] if recent_count else []

    for latest in recent_memories:
        llm = get_llm()

        prompt = f"""
你是 Lead Researcher，审核子 Agent 的研究结果。

研究方面：{latest.aspect}
结果总结：{latest.summary}
置信度：{latest.confidence}

请评估并输出 JSON：
{{
  "accepted": true,
  "need_more_research": false,
  "new_aspects": [],
  "comment": "评价",
  "quality_score": 0.85
}}
"""

        messages = [HumanMessage(content=prompt)]
        verdict = parse_json_with_retry(llm, messages, LeadReflection)

        if not verdict:
            verdict = LeadReflection(
                accepted=True,
                need_more_research=False,
                new_aspects=[],
                comment="默认接受",
                quality_score=0.7,
            )

        state.research_logs.append(
            f"[Lead] 审核 '{latest.aspect}': accepted={verdict.accepted}, quality={verdict.quality_score:.2f}"
        )

        if verdict.new_aspects:
            for aspect in verdict.new_aspects:
                if aspect not in state.backlog:
                    state.backlog.append(aspect)

    state.continue_research = bool(state.backlog)
    state.loop_count += 1

    return state


def assess_goal_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """从增强版复制（简化）"""
    llm = get_llm()

    memory_summary = "\n".join(f"- {m.aspect}: {m.summary}" for m in state.memory)

    prompt = f"""
评估研究任务是否已完成。

用户需求：{state.user_request}
已完成研究：
{memory_summary}
剩余待研究：{state.backlog}

输出 JSON：
{{
  "goal_achieved": true,
  "completeness": 0.9,
  "missing_aspects": [],
  "reasoning": "评估理由"
}}
"""

    messages = [HumanMessage(content=prompt)]
    assessment = parse_json_with_retry(llm, messages, GoalAssessment)

    if not assessment:
        assessment = GoalAssessment(
            goal_achieved=len(state.memory) >= 3 and not state.backlog,
            completeness=min(len(state.memory) / 5, 1.0),
            missing_aspects=state.backlog.copy(),
            reasoning="基于规则判断",
        )

    state.goal_assessment = assessment

    if assessment.goal_achieved:
        state.continue_research = False
    elif assessment.missing_aspects:
        for aspect in assessment.missing_aspects:
            if aspect not in state.backlog:
                state.backlog.append(aspect)

    state.research_logs.append(
        f"[GoalAssessment] 达成={assessment.goal_achieved}, 完整度={assessment.completeness:.2f}"
    )

    return state


def budget_monitor_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """从增强版复制"""
    estimated_tokens = len(state.memory) * 500 + state.loop_count * 2000
    state.total_tokens_used = estimated_tokens

    if state.total_tokens_used > state.max_tokens_budget:
        state.continue_research = False
        state.research_logs.append(f"[BudgetMonitor] Token 预算耗尽，强制终止")
    elif state.loop_count >= state.max_loops:
        state.continue_research = False
        state.research_logs.append(f"[BudgetMonitor] 达到最大循环次数，终止")

    return state


def final_report_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """从增强版复制（简化）"""
    llm = get_llm()

    notes_str = "\n\n".join(
        f"- {m.aspect}（由 {m.agent} 研究，置信度: {m.confidence}，耗时: {m.execution_time:.2f}s）\n"
        f"  总结: {m.summary}"
        for m in state.memory
    )

    prompt = f"""
基于研究记忆编写最终报告。

用户请求：{state.user_request}

研究记忆：
{notes_str}

并行执行统计：
- 总加速比：{state.parallel_speedup:.2f}x

请输出 Markdown 格式报告。
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state.final_report = response.content
    state.research_logs.append("[CitationAgent] 完成最终报告。")
    return state


# ============================================================================
# Graph Construction
# ============================================================================


def create_parallel_workflow():
    """创建支持并行执行的工作流"""
    graph = StateGraph(ClaudeCodeState)

    # 添加节点
    graph.add_node("plan", plan_research_node)
    graph.add_node("analyze_deps", analyze_dependencies_node)  # 新增
    graph.add_node("batch_spawn", batch_spawn_subagents_node)  # 新增
    graph.add_node("parallel_execute", parallel_execute_node)  # 新增
    graph.add_node("reflect", lead_reflection_node)
    graph.add_node("assess_goal", assess_goal_node)
    graph.add_node("budget_monitor", budget_monitor_node)
    graph.add_node("final", final_report_node)

    # 定义流程
    graph.set_entry_point("plan")
    graph.add_edge("plan", "analyze_deps")  # 先分析依赖
    graph.add_edge("analyze_deps", "batch_spawn")

    # 条件分支：如果有并行任务，则执行；否则进入最终报告
    graph.add_conditional_edges(
        "batch_spawn",
        lambda state: "parallel_execute" if state.parallel_briefs else "final",
    )

    graph.add_edge("parallel_execute", "reflect")
    graph.add_edge("reflect", "assess_goal")
    graph.add_edge("assess_goal", "budget_monitor")

    # 条件分支：根据 continue_research 决定是否继续循环
    graph.add_conditional_edges(
        "budget_monitor",
        lambda state: "batch_spawn" if state.continue_research else "final",
    )

    graph.add_edge("final", END)

    return graph.compile()


# ============================================================================
# Main Execution
# ============================================================================


def run_parallel_demo() -> None:
    """运行并行执行演示"""
    load_environment()
    workflow = create_parallel_workflow()

    tasks = [
        {
            "request": "研究 Python、Rust、Go 在 2025 年的 Web 开发生态系统对比",
            "budget": 40000,
            "max_loops": 6,
            "max_parallel": 3,  # 最多同时执行 3 个 SubAgent
        },
    ]

    for idx, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Claude Code Parallel Execution Demo #{idx}")
        print(f"{'='*80}")
        print(f"用户任务：{task['request']}")
        print(f"并行配置：最多同时执行 {task['max_parallel']} 个 SubAgent")

        state = ClaudeCodeState(
            user_request=task["request"],
            max_tokens_budget=task["budget"],
            max_loops=task["max_loops"],
            max_parallel=task["max_parallel"],
            parallel_enabled=True,
        )

        try:
            start_time = time.time()
            final_state = workflow.invoke(state)
            total_time = time.time() - start_time

            print("\n--- 研究记忆 ---")
            for note in final_state["memory"]:
                print(f"\n[{note.agent}] {note.aspect}")
                print(f"  总结: {note.summary}")
                print(f"  置信度: {note.confidence}")
                print(f"  执行时间: {note.execution_time:.2f}s")

            print("\n--- 性能统计 ---")
            print(f"总执行时间: {total_time:.2f}s")
            print(f"并行加速比: {final_state.get('parallel_speedup', 1.0):.2f}x")
            print(f"Token 使用: ~{final_state['total_tokens_used']} / {final_state['max_tokens_budget']}")
            print(f"循环次数: {final_state['loop_count']} / {final_state['max_loops']}")

            print("\n--- 最终报告 ---")
            print(final_state["final_report"])

        except Exception as exc:
            print(f"\n执行出错：{exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run_parallel_demo()
