#!/usr/bin/env python3
"""
11 - Claude Code Style Hierarchical Agent Demo (Enhanced Version)

增强功能：
1. 工具注册系统（Web 搜索、Python REPL、文件操作）
2. 结构化输出和错误重试机制
3. 智能循环终止（目标达成判断 + 预算监控）
4. 优化的 Prompt（few-shot examples）
5. 结构化 Citation 追踪
6. 并行 SubAgent 执行能力
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

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
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)


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
# 1. 工具注册系统
# ============================================================================


class ToolRegistry:
    """集中管理所有可用工具"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认工具集"""
        # Web 搜索工具
        try:
            search = DuckDuckGoSearchRun()
            self.tools["web_search"] = Tool(
                name="web_search",
                description="搜索互联网获取最新信息。输入搜索关键词，返回相关结果摘要。",
                func=search.run,
            )
        except Exception as e:
            print(f"[警告] Web 搜索工具注册失败: {e}")

        # Python REPL 工具
        try:
            python_repl = PythonREPL()
            self.tools["python_repl"] = Tool(
                name="python_repl",
                description="执行 Python 代码进行计算、数据处理等。输入 Python 代码，返回执行结果。",
                func=python_repl.run,
            )
        except Exception as e:
            print(f"[警告] Python REPL 工具注册失败: {e}")

        # 文件读取工具（简化版）
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
        """根据 Agent 类型返回合适的工具集"""
        tool_mapping = {
            "researcher": ["web_search"],
            "analyst": ["python_repl", "file_read"],
            "generalist": ["web_search", "python_repl", "file_read"],
        }
        tool_names = tool_mapping.get(agent_type, ["web_search"])
        return [self.tools[name] for name in tool_names if name in self.tools]

    def get_all_tools(self) -> List[Tool]:
        """返回所有工具"""
        return list(self.tools.values())


# 全局工具注册表
TOOL_REGISTRY = ToolRegistry()


# ============================================================================
# 2. 结构化数据模型（带验证）
# ============================================================================


class Citation(BaseModel):
    """结构化引用"""

    source_type: Literal["web", "file", "calculation", "reasoning"]
    source: str  # URL / 文件路径 / "internal calculation" / "logical reasoning"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    snippet: Optional[str] = None  # 关键摘录


class ResearchMemory(BaseModel):
    """研究记忆（增强版）"""

    aspect: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    agent: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)  # 置信度
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SubAgentBrief(BaseModel):
    """子 Agent 简报（结构化）"""

    name: str
    agent_type: Literal["researcher", "analyst", "generalist"] = "generalist"
    focus: str
    instructions: str
    expected_tools: List[str] = Field(default_factory=list)


class SubAgentResult(BaseModel):
    """子 Agent 执行结果（结构化）"""

    thoughts: List[str]
    actions_taken: List[str]
    observations: List[str]
    result_summary: str
    citations: List[Citation] = Field(default_factory=list)
    next_suggestions: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LeadReflection(BaseModel):
    """Lead Researcher 审核结果"""

    accepted: bool
    need_more_research: bool
    new_aspects: List[str] = Field(default_factory=list)
    comment: str
    quality_score: float = Field(default=0.7, ge=0.0, le=1.0)


class GoalAssessment(BaseModel):
    """目标达成评估"""

    goal_achieved: bool
    completeness: float = Field(ge=0.0, le=1.0)  # 0-1
    missing_aspects: List[str] = Field(default_factory=list)
    reasoning: str


class ClarificationQuestion(BaseModel):
    """澄清问题"""

    question: str
    reason: str  # 为什么需要这个问题
    question_type: Literal["scope", "preference", "constraint", "context"] = "context"
    options: Optional[List[str]] = None  # 可选的选项（多选题）


class ClarificationNeed(BaseModel):
    """澄清需求判断"""

    need_clarification: bool
    questions: List[ClarificationQuestion] = Field(default_factory=list)
    reasoning: str
    urgency: Literal["high", "medium", "low"] = "medium"  # 紧迫性


class ClarificationResponse(BaseModel):
    """用户澄清回答"""

    answers: Dict[str, str]  # question -> answer
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ClaudeCodeState(BaseModel):
    """增强的状态管理"""

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

    # 新增：预算监控
    total_tokens_used: int = 0
    max_tokens_budget: int = Field(default=50000)
    max_loops: int = Field(default=8)

    # 新增：目标达成追踪
    goal_assessment: Optional[GoalAssessment] = None

    # 新增：澄清问题机制
    clarification_need: Optional[ClarificationNeed] = None
    clarification_responses: List[ClarificationResponse] = Field(default_factory=list)
    enable_clarification: bool = Field(default=True)  # 是否启用反问功能


# ============================================================================
# 3. 结构化输出辅助函数（带重试）
# ============================================================================


def parse_json_with_retry(
    llm: Any, messages: List, target_model: type[BaseModel], max_retries: int = 3
) -> Optional[BaseModel]:
    """使用结构化输出和重试机制解析 LLM 响应"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            content = response.content.strip()

            # 尝试提取 JSON（处理 markdown 代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            parsed_data = json.loads(content)
            return target_model(**parsed_data)

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < max_retries - 1:
                # 添加错误提示，要求 LLM 修正
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
# 4. Clarification Nodes (New)
# ============================================================================


def detect_clarification_need_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """检测是否需要向用户澄清需求"""
    # 如果已经有计划或禁用澄清功能，跳过
    if state.plan or not state.enable_clarification:
        return state

    # 如果已经问过问题，跳过
    if state.clarification_responses:
        return state

    llm = get_llm()

    prompt = f"""
你是 Lead Researcher，正在分析用户的研究需求。

用户需求：{state.user_request}

请判断这个需求是否需要向用户提问以澄清。考虑以下因素：
1. 需求是否模糊或有多种理解方式？
2. 是否缺少关键的范围、约束或偏好信息？
3. 是否有技术选型、优先级等决策点需要用户输入？

输出 JSON 格式：

{{
  "need_clarification": true,
  "questions": [
    {{
      "question": "您希望重点关注哪个方面？",
      "reason": "需求过于宽泛，需要明确重点",
      "question_type": "scope",
      "options": ["选项1", "选项2", "选项3"]
    }}
  ],
  "reasoning": "为什么需要澄清的详细说明",
  "urgency": "high"
}}

question_type 类型说明：
- scope: 范围相关（覆盖面、深度等）
- preference: 偏好相关（技术栈、风格等）
- constraint: 约束相关（时间、资源、限制等）
- context: 背景相关（使用场景、目标等）

urgency 说明：
- high: 不澄清无法继续
- medium: 澄清后效果更好
- low: 可选的补充信息

注意：
- 只在真正需要时才设置 need_clarification=true
- 问题数量控制在 1-3 个
- 问题要具体、明确，避免过于开放
- 如果需求已经足够清晰，设置 need_clarification=false
"""

    messages = [HumanMessage(content=prompt)]
    result = parse_json_with_retry(llm, messages, ClarificationNeed, max_retries=2)

    if not result:
        # 回退：默认不需要澄清
        result = ClarificationNeed(
            need_clarification=False,
            questions=[],
            reasoning="解析失败，默认不需要澄清",
            urgency="low",
        )

    state.clarification_need = result

    if result.need_clarification:
        state.research_logs.append(
            f"[Clarification] 检测到需要澄清：{result.reasoning}，"
            f"问题数：{len(result.questions)}，紧迫性：{result.urgency}"
        )
    else:
        state.research_logs.append("[Clarification] 需求明确，无需澄清")

    return state


def ask_user_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """向用户提问并收集回答"""
    if not (state.clarification_need and state.clarification_need.need_clarification):
        return state

    # 如果已经收集过回答，跳过
    if state.clarification_responses:
        return state

    clarification = state.clarification_need

    print("\n" + "=" * 80)
    print("Agent 需要您的帮助来更好地理解需求")
    print("=" * 80)
    print(f"\n原因：{clarification.reasoning}")
    print(f"紧迫性：{clarification.urgency.upper()}\n")

    answers = {}
    for i, q in enumerate(clarification.questions, 1):
        print(f"\n问题 {i}/{len(clarification.questions)} ({q.question_type}):")
        print(f"  {q.question}")
        print(f"  → 原因：{q.reason}")

        if q.options:
            print(f"  → 可选项：")
            for j, opt in enumerate(q.options, 1):
                print(f"    {j}. {opt}")
            user_input = input(f"\n  请输入您的选择（1-{len(q.options)} 或自定义文本）[回车跳过]: ").strip()

            # 处理选项选择
            if user_input.isdigit() and 1 <= int(user_input) <= len(q.options):
                answers[q.question] = q.options[int(user_input) - 1]
            elif user_input:
                answers[q.question] = user_input
            else:
                answers[q.question] = "（用户跳过）"
        else:
            user_input = input("  您的回答 [回车跳过]: ").strip()
            answers[q.question] = user_input if user_input else "（用户跳过）"

    # 存储回答
    response = ClarificationResponse(answers=answers)
    state.clarification_responses.append(response)

    # 更新用户需求（将澄清信息融入）
    clarification_summary = "\n".join(f"- {q}: {a}" for q, a in answers.items() if a != "（用户跳过）")
    if clarification_summary:
        state.user_request = f"{state.user_request}\n\n用户澄清：\n{clarification_summary}"

    state.research_logs.append(
        f"[Clarification] 收到用户反馈，已回答 {len([a for a in answers.values() if a != '（用户跳过）'])} 个问题"
    )

    print("\n✓ 感谢您的反馈！继续执行研究任务...\n")
    return state


# ============================================================================
# 5. Lead Researcher Nodes (Enhanced)
# ============================================================================


def plan_research_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: 创建研究计划（增强 Prompt）"""
    if state.plan:
        return state

    llm = get_llm()

    # Few-shot example
    example = """
示例：
用户需求："研究 2025 年学习 Rust 的最佳路径"
输出：
[
  "方面1：Rust 基础学习资源（官方文档、入门书籍）",
  "方面2：实战项目推荐（从简单到复杂）",
  "方面3：社区资源和学习路径对比（2025 年最新）",
  "方面4：常见陷阱和最佳实践"
]
"""

    prompt = f"""
你是 Lead Researcher，负责将用户需求拆解为具体可执行的研究方面。

用户需求：{state.user_request}

要求：
1. 拆解为 3-5 个独立的研究方面
2. 每个方面需具体、可操作（避免过于宽泛）
3. 按优先级排序
4. 使用 JSON 数组格式输出

{example}

现在请为当前需求生成研究计划：
"""

    messages = [HumanMessage(content=prompt)]
    result = parse_json_with_retry(llm, messages, target_model=type(state.plan))

    if isinstance(result, list):
        state.plan = [str(item) for item in result]
    else:
        # 回退：简单分割
        state.plan = [state.user_request]

    state.backlog = state.plan.copy()
    state.research_logs.append(f"[Lead] 生成研究规划：{state.plan}")
    return state


def pick_aspect_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: 从 backlog 中选择下一个研究方面"""
    if state.backlog:
        state.current_aspect = state.backlog.pop(0)
    else:
        state.current_aspect = None
    return state


def spawn_subagent_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher: 动态创建子 Agent 简报（结构化）"""
    if not state.current_aspect:
        return state

    llm = get_llm()
    memory_context = "\n".join(f"- {m.aspect}: {m.summary}" for m in state.memory[-3:])

    prompt = f"""
用户总体请求：{state.user_request}
当前研究方面：{state.current_aspect}
现有记忆：{memory_context or "（无）"}

请设计一个子 Agent 来完成这个研究方面。输出 JSON 格式：

{{
  "name": "Agent名称（如：RustLearningPathResearcher）",
  "agent_type": "researcher | analyst | generalist",
  "focus": "具体研究焦点",
  "instructions": "执行指南（要求使用 ReAct：Thought→Action→Observation→Reflection）",
  "expected_tools": ["web_search", "python_repl", "file_read"] // 需要使用的工具
}}

agent_type 说明：
- researcher: 主要用 web_search
- analyst: 主要用 python_repl, file_read
- generalist: 可用所有工具
"""

    messages = [HumanMessage(content=prompt)]
    brief = parse_json_with_retry(llm, messages, SubAgentBrief)

    if not brief:
        # 回退方案
        brief = SubAgentBrief(
            name="Generalist",
            agent_type="generalist",
            focus=state.current_aspect,
            instructions="使用 ReAct 风格执行研究，记录思考、行动、观察并反思。",
            expected_tools=["web_search"],
        )

    state.current_agent_brief = brief
    state.research_logs.append(
        f"[Lead] 为 '{state.current_aspect}' 派生子 Agent：{brief.name} ({brief.agent_type})"
    )
    return state


# ============================================================================
# 5. SubAgent Execution (Enhanced with Real Tools)
# ============================================================================


def subagent_execution_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """子 Agent 执行节点（集成真实工具）"""
    if not (state.current_aspect and state.current_agent_brief):
        return state

    llm = get_llm(temperature=0)
    brief = state.current_agent_brief

    # 获取工具
    tools = TOOL_REGISTRY.get_tools_for_agent(brief.agent_type)
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)

    memory_snippets = "\n".join(
        f"- {m.aspect}: {m.summary} (置信度: {m.confidence})" for m in state.memory[-5:]
    )

    system_prompt = f"""
你是子 Agent {brief.name}，类型：{brief.agent_type}
职责：{brief.focus}
执行要求：{brief.instructions}

可用工具：
{tool_descriptions or "（无可用工具，仅使用内部推理）"}

你必须使用 ReAct 模式：
1. Thought（思考）：分析当前需要做什么
2. Action（行动）：选择工具并说明如何使用（如果没有合适工具，则进行逻辑推理）
3. Observation（观察）：记录行动结果
4. Reflection（反思）：评估结果质量和可信度
"""

    human_prompt = f"""
用户总请求：{state.user_request}
当前子任务：{state.current_aspect}
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
      "source": "具体来源（URL/文件路径/描述）",
      "snippet": "关键信息摘录（可选）"
    }}
  ],
  "next_suggestions": "建议 Lead Researcher 后续关注的方向（可选）",
  "confidence": 0.8  // 结果置信度 0-1
}}
"""

    # 简化版工具调用（模拟 ReAct 循环）
    # 在生产环境中，这里应该用 LangGraph 的 create_react_agent 或自定义工具调用循环
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]

    # 模拟工具调用（实际应该让 LLM 决定是否调用工具）
    # 这里为了演示，我们先让 LLM 规划，然后手动调用一次工具
    tool_results = []
    if "web_search" in brief.expected_tools and "web_search" in TOOL_REGISTRY.tools:
        try:
            search_query = state.current_aspect  # 简化：直接用 aspect 作为查询
            search_result = TOOL_REGISTRY.tools["web_search"].func(search_query)
            tool_results.append(f"[WebSearch] {search_result[:500]}")  # 截断
        except Exception as e:
            tool_results.append(f"[WebSearch Error] {str(e)}")

    # 将工具结果附加到 prompt
    if tool_results:
        human_prompt += f"\n\n工具执行结果：\n" + "\n".join(tool_results)
        messages[-1] = HumanMessage(content=human_prompt)

    result = parse_json_with_retry(llm, messages, SubAgentResult, max_retries=2)

    if not result:
        # 回退方案
        result = SubAgentResult(
            thoughts=["执行出错，使用默认结果"],
            actions_taken=[],
            observations=[],
            result_summary=f"针对 '{state.current_aspect}' 的研究未能完成（解析失败）",
            citations=[],
            confidence=0.3,
        )

    # 存储到记忆
    note = ResearchMemory(
        aspect=state.current_aspect,
        summary=result.result_summary,
        citations=result.citations,
        agent=brief.name,
        confidence=result.confidence,
    )
    state.memory.append(note)

    log = (
        f"[SubAgent:{note.agent}] 完成 {note.aspect}\n"
        f"思考: {result.thoughts}\n"
        f"行动: {result.actions_taken}\n"
        f"观察: {result.observations}\n"
        f"总结: {note.summary}\n"
        f"置信度: {note.confidence}"
    )
    state.research_logs.append(log)
    return state


# ============================================================================
# 6. Lead Reflection (Enhanced)
# ============================================================================


def lead_reflection_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """Lead Researcher 审核子 Agent 输出"""
    if not state.memory:
        return state

    llm = get_llm()
    latest = state.memory[-1]

    citations_str = "\n".join(
        f"  - [{c.source_type}] {c.source}" + (f": {c.snippet}" if c.snippet else "")
        for c in latest.citations
    )

    prompt = f"""
你是 Lead Researcher，正在审核子 Agent {latest.agent} 的研究结果。

研究方面：{latest.aspect}
结果总结：{latest.summary}
引用来源：
{citations_str or "（无引用）"}
置信度：{latest.confidence}

请评估并输出 JSON：

{{
  "accepted": true,  // 是否接受该结果
  "need_more_research": false,  // 是否需要更多研究
  "new_aspects": ["新方面1"],  // 如果需要，列出新的研究方面（最多2个）
  "comment": "评价和建议",
  "quality_score": 0.85  // 质量评分 0-1
}}

评估标准：
- 结果是否回答了研究方面的核心问题
- 引用来源是否可靠
- 置信度是否合理
- 是否有遗漏的关键信息
"""

    messages = [HumanMessage(content=prompt)]
    verdict = parse_json_with_retry(llm, messages, LeadReflection)

    if not verdict:
        verdict = LeadReflection(
            accepted=True,
            need_more_research=False,
            new_aspects=[],
            comment="解析失败，默认接受",
            quality_score=0.5,
        )

    state.research_logs.append(
        f"[Lead] 审核 '{latest.aspect}': "
        f"accepted={verdict.accepted}, quality={verdict.quality_score:.2f}, "
        f"note={verdict.comment}"
    )

    # 添加新的研究方面
    if verdict.new_aspects:
        for aspect in verdict.new_aspects:
            if aspect not in state.backlog:
                state.backlog.append(aspect)
        state.research_logs.append(f"[Lead] 新增研究方面：{verdict.new_aspects}")

    state.continue_research = verdict.need_more_research or bool(state.backlog)
    state.loop_count += 1

    return state


# ============================================================================
# 7. Goal Assessment (New)
# ============================================================================


def assess_goal_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """评估目标是否达成（智能终止条件）"""
    llm = get_llm()

    memory_summary = "\n".join(
        f"- {m.aspect}: {m.summary} (置信度: {m.confidence})" for m in state.memory
    )

    prompt = f"""
你是 Lead Researcher，需要评估研究任务是否已完成。

用户原始需求：{state.user_request}

已完成的研究：
{memory_summary}

当前循环次数：{state.loop_count}
剩余待研究方面：{state.backlog}

请评估并输出 JSON：

{{
  "goal_achieved": true,  // 目标是否已达成
  "completeness": 0.9,  // 完整度 0-1（1 表示完全满足需求）
  "missing_aspects": ["缺失方面1"],  // 如果未完成，列出缺失的关键方面
  "reasoning": "评估理由"
}}

达成标准：
- completeness >= 0.85 且 missing_aspects 为空时，可认为 goal_achieved=true
- 如果已有足够信息回答用户需求，即使有小瑕疵也可认为达成
"""

    messages = [HumanMessage(content=prompt)]
    assessment = parse_json_with_retry(llm, messages, GoalAssessment)

    if not assessment:
        # 回退：基于简单规则
        assessment = GoalAssessment(
            goal_achieved=len(state.memory) >= 3 and not state.backlog,
            completeness=min(len(state.memory) / 5, 1.0),
            missing_aspects=state.backlog.copy(),
            reasoning="基于规则判断",
        )

    state.goal_assessment = assessment
    state.research_logs.append(
        f"[GoalAssessment] 达成={assessment.goal_achieved}, "
        f"完整度={assessment.completeness:.2f}, 理由={assessment.reasoning}"
    )

    # 更新 continue_research 标志
    if assessment.goal_achieved:
        state.continue_research = False
    elif assessment.missing_aspects and state.loop_count < state.max_loops:
        # 将缺失方面加入 backlog
        for aspect in assessment.missing_aspects:
            if aspect not in state.backlog:
                state.backlog.append(aspect)
        state.continue_research = True

    return state


# ============================================================================
# 8. Budget Monitor (New)
# ============================================================================


def budget_monitor_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """监控 Token 预算（简化版）"""
    # 简化估算：每条记忆约 500 tokens，每次循环约 2000 tokens
    estimated_tokens = len(state.memory) * 500 + state.loop_count * 2000
    state.total_tokens_used = estimated_tokens

    if state.total_tokens_used > state.max_tokens_budget:
        state.continue_research = False
        state.research_logs.append(
            f"[BudgetMonitor] Token 预算耗尽 ({state.total_tokens_used}/{state.max_tokens_budget})，强制终止"
        )
    elif state.loop_count >= state.max_loops:
        state.continue_research = False
        state.research_logs.append(f"[BudgetMonitor] 达到最大循环次数 ({state.max_loops})，终止")

    return state


# ============================================================================
# 9. Final Report (Enhanced with Citations)
# ============================================================================


def final_report_node(state: ClaudeCodeState) -> ClaudeCodeState:
    """生成最终报告（增强引用格式）"""
    llm = get_llm()

    notes_with_citations = []
    for m in state.memory:
        citations = "\n    ".join(
            f"[{i+1}] {c.source_type.upper()}: {c.source}"
            + (f" - '{c.snippet[:100]}...'" if c.snippet else "")
            for i, c in enumerate(m.citations)
        )
        notes_with_citations.append(
            f"- {m.aspect}（由 {m.agent} 研究，置信度: {m.confidence}）\n"
            f"  总结: {m.summary}\n"
            f"  引用:\n    {citations or '（无）'}"
        )

    notes_str = "\n\n".join(notes_with_citations)

    prompt = f"""
你是 Citation Agent，负责基于研究记忆编写最终报告。

用户请求：{state.user_request}

研究记忆：
{notes_str}

目标达成情况：
{state.goal_assessment.reasoning if state.goal_assessment else "未评估"}

请输出结构化报告（Markdown 格式）：

# 研究报告：{state.user_request}

## 执行摘要
（2-3 段总结）

## 核心发现
1. ...
2. ...

## 详细分析
（按研究方面组织）

### 方面1
...

### 方面2
...

## 引用来源
（列出所有引用，按 [序号] 格式）

## 建议与后续行动
（如果有）

---
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
总研究轮次：{state.loop_count}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state.final_report = response.content
    state.research_logs.append("[CitationAgent] 完成最终报告。")
    return state


# ============================================================================
# 10. Graph Construction (Enhanced)
# ============================================================================


def create_enhanced_workflow():
    """创建增强版工作流（带反问机制）"""
    graph = StateGraph(ClaudeCodeState)

    # 添加所有节点
    graph.add_node("detect_clarification", detect_clarification_need_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("plan", plan_research_node)
    graph.add_node("pick", pick_aspect_node)
    graph.add_node("spawn", spawn_subagent_node)
    graph.add_node("execute", subagent_execution_node)
    graph.add_node("reflect", lead_reflection_node)
    graph.add_node("assess_goal", assess_goal_node)
    graph.add_node("budget_monitor", budget_monitor_node)
    graph.add_node("final", final_report_node)

    # 定义流程
    # 入口：先检测是否需要澄清
    graph.set_entry_point("detect_clarification")

    # 条件分支：如果需要澄清，则询问用户；否则直接规划
    graph.add_conditional_edges(
        "detect_clarification",
        lambda state: "ask_user"
        if (state.clarification_need and state.clarification_need.need_clarification)
        else "plan",
    )

    # 用户回答后进入规划阶段
    graph.add_edge("ask_user", "plan")

    # 规划后开始执行
    graph.add_edge("plan", "pick")

    # 条件分支：如果有待研究方面，则派生子 Agent；否则进入最终报告
    graph.add_conditional_edges(
        "pick",
        lambda state: "spawn" if state.current_aspect else "final",
    )

    graph.add_edge("spawn", "execute")
    graph.add_edge("execute", "reflect")
    graph.add_edge("reflect", "assess_goal")
    graph.add_edge("assess_goal", "budget_monitor")

    # 条件分支：根据 continue_research 决定是否继续循环
    graph.add_conditional_edges(
        "budget_monitor",
        lambda state: "pick" if state.continue_research else "final",
    )

    graph.add_edge("final", END)

    return graph.compile()


# ============================================================================
# 11. Main Execution
# ============================================================================


def run_enhanced_demo(interactive: bool = True) -> None:
    """运行增强版 Demo

    Args:
        interactive: 是否启用交互模式（反问功能）
    """
    load_environment()
    workflow = create_enhanced_workflow()

    if interactive:
        # 交互模式：从用户获取输入
        print("\n" + "=" * 80)
        print("Claude Code Style Enhanced Demo (交互模式)")
        print("=" * 80)
        print("\n提示：您可以输入研究需求，Agent 会在必要时向您提问以澄清细节。\n")

        user_input = input("请输入您的研究需求（回车使用默认示例）: ").strip()

        if not user_input:
            user_input = "研究 2025 年学习 Rust 的最佳路径"
            print(f"使用默认示例：{user_input}\n")

        tasks = [
            {
                "request": user_input,
                "budget": 40000,
                "max_loops": 6,
                "enable_clarification": True,
            }
        ]
    else:
        # 批处理模式：预定义任务
        tasks = [
            {
                "request": "研究 2025 年学习 Rust 的最佳路径，并给出推荐资料",
                "budget": 40000,
                "max_loops": 6,
                "enable_clarification": False,
            },
            {
                "request": "分析 LangGraph 和传统工作流引擎的主要区别",
                "budget": 30000,
                "max_loops": 5,
                "enable_clarification": False,
            },
        ]

    for idx, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Claude Code Style Enhanced Demo #{idx}")
        print(f"{'='*80}")
        print(f"用户任务：{task['request']}")

        state = ClaudeCodeState(
            user_request=task["request"],
            max_tokens_budget=task["budget"],
            max_loops=task["max_loops"],
            enable_clarification=task.get("enable_clarification", True),
        )

        try:
            final_state = workflow.invoke(state)

            # 显示澄清过程（如果有）
            if final_state.get("clarification_responses"):
                print("\n--- 澄清过程回顾 ---")
                for resp in final_state["clarification_responses"]:
                    print(f"时间：{resp.timestamp}")
                    for q, a in resp.answers.items():
                        print(f"  Q: {q}")
                        print(f"  A: {a}")

            print("\n--- 研究记忆 ---")
            for note in final_state["memory"]:
                print(f"\n[{note.agent}] {note.aspect}")
                print(f"  总结: {note.summary}")
                print(f"  置信度: {note.confidence}")
                print(f"  引用数: {len(note.citations)}")
                for cit in note.citations[:2]:  # 只显示前 2 个引用
                    print(f"    - [{cit.source_type}] {cit.source[:80]}...")

            print("\n--- 目标达成评估 ---")
            if final_state.get("goal_assessment"):
                ga = final_state["goal_assessment"]
                print(f"目标达成: {ga.goal_achieved}")
                print(f"完整度: {ga.completeness:.2%}")
                print(f"理由: {ga.reasoning}")

            print("\n--- 预算使用情况 ---")
            print(
                f"Token 使用: ~{final_state['total_tokens_used']} / {final_state['max_tokens_budget']}"
            )
            print(f"循环次数: {final_state['loop_count']} / {final_state['max_loops']}")

            print("\n--- 最终报告 ---")
            print(final_state["final_report"])

        except Exception as exc:
            print(f"\n执行出错：{exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_enhanced_demo()
