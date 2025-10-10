#!/usr/bin/env python3
"""
Module 9: Self-Discover 示例（自发现技能与成功标准的多阶段工作流）

简化版 Self-Discover：
- Discover：LLM 自主“发现”解决问题所需的技能清单、成功标准（Rubric）以及高层计划步骤
- Execute：按计划逐步执行，每个步骤结合技能与成功标准生成内容
- Evaluate：对汇总结果进行评估，判定是否满足成功标准，并提出改进建议
- Revise：若未满足，则根据建议与技能进行一次聚合修订
- Finalize：输出最终答案

特性：
- 无外部工具调用（纯内部推理），每次对话独立
- 使用 LangGraph 管理工作流与状态
- CLI 交互，兼容 dict/数据类两种返回形态
"""
import os
import sys
from typing import List, Optional
from dataclasses import dataclass, field

# 终端中文输出
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 环境

def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 模型

def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "768"))
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=3,
        request_timeout=120,
        verbose=True,
    )

# 状态

@dataclass
class AgentState:
    messages: List[BaseMessage]
    question: str
    skills: List[str] = field(default_factory=list)  # 自发现技能
    rubric: List[str] = field(default_factory=list)  # 成功标准
    plan_steps: List[str] = field(default_factory=list)  # 高层计划
    drafts: List[str] = field(default_factory=list)   # 每步输出草稿
    current_index: int = 0
    final_answer: Optional[str] = None
    suggestions: Optional[str] = None
    success: bool = False

# Discover：自发现技能、成功标准和计划

def discover_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = (
        "请针对用户问题，自主‘发现’解决所需的技能清单、成功标准（Rubric）以及高层计划步骤。\n"
        "严格输出如下结构：\n"
        "技能:\n- <技能1>\n- <技能2>\n- ...\n"
        "成功标准:\n- <标准1>\n- <标准2>\n- ...\n"
        "计划:\n1. <步骤1>\n2. <步骤2>\n3. <步骤3>（如需要可更多）\n\n"
        f"用户问题：{state.question}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    content = ai.content

    skills, rubric, plans = parse_discover_content(content)

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=(
        "Discover 阶段：\n"
        "技能：\n" + "\n".join(f"- {s}" for s in skills) + "\n\n" +
        "成功标准：\n" + "\n".join(f"- {r}" for r in rubric) + "\n\n" +
        "计划：\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(plans))
    )))

    return AgentState(
        messages=new_messages,
        question=state.question,
        skills=skills,
        rubric=rubric,
        plan_steps=plans,
        drafts=[],
        current_index=0,
        final_answer=None,
        suggestions=None,
        success=False,
    )

# Execute：执行当前计划步骤

def execute_step_node(state: AgentState) -> AgentState:
    llm = get_llm()
    idx = state.current_index
    step = state.plan_steps[idx] if idx < len(state.plan_steps) else ""
    skills_text = "\n".join(f"- {s}" for s in state.skills) if state.skills else "(无)"
    rubric_text = "\n".join(f"- {r}" for r in state.rubric) if state.rubric else "(无)"

    prompt = (
        "你将执行计划中的一个步骤，结合已发现技能与成功标准产出该步骤的具体内容。\n"
        "要求：清晰、结构化、可执行；必要时给出要点或步骤，避免泛泛而谈。\n"
        f"当前步骤：{step}\n"
        f"已发现技能：\n{skills_text}\n"
        f"成功标准：\n{rubric_text}\n"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    draft = ai.content

    new_drafts = state.drafts + [draft]
    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Execute 阶段（步骤 {idx+1}/{len(state.plan_steps)}）：\n{draft}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        skills=state.skills,
        rubric=state.rubric,
        plan_steps=state.plan_steps,
        drafts=new_drafts,
        current_index=state.current_index + 1,
        final_answer=None,
        suggestions=None,
        success=False,
    )

# 进度检查：继续执行或进入评估

def progress_check_node(state: AgentState) -> AgentState:
    new_messages = state.messages.copy()
    if state.current_index < len(state.plan_steps):
        new_messages.append(AIMessage(content="Progress Check：继续执行下一步骤。"))
        return AgentState(
            messages=new_messages,
            question=state.question,
            skills=state.skills,
            rubric=state.rubric,
            plan_steps=state.plan_steps,
            drafts=state.drafts,
            current_index=state.current_index,
            final_answer=None,
            suggestions=None,
            success=False,
        )
    else:
        # 汇总最终答案
        final_text = assemble_final_answer(state)
        new_messages.append(AIMessage(content="Progress Check：所有步骤完成，准备评估最终答案。"))
        return AgentState(
            messages=new_messages,
            question=state.question,
            skills=state.skills,
            rubric=state.rubric,
            plan_steps=state.plan_steps,
            drafts=state.drafts,
            current_index=state.current_index,
            final_answer=final_text,
            suggestions=None,
            success=False,
        )

# Evaluate：评估最终答案是否满足成功标准

def evaluate_node(state: AgentState) -> AgentState:
    llm = get_llm()
    rubric_text = "\n".join(f"- {r}" for r in state.rubric) if state.rubric else "(无)"
    prompt = (
        "你是评估器。请针对最终答案与成功标准进行评估。\n"
        "严格输出以下结构：\n"
        "结论: 成功 或 失败\n"
        "理由: <一句话或几句话>\n"
        "改进建议: <若失败，给出具体可执行建议；若成功，可填'无'>\n\n"
        f"成功标准：\n{rubric_text}\n\n"
        f"最终答案：\n{state.final_answer or ''}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    feedback = ai.content

    success = parse_success(feedback)
    suggestions = parse_suggestions(feedback)

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Evaluate 阶段：\n{feedback}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        skills=state.skills,
        rubric=state.rubric,
        plan_steps=state.plan_steps,
        drafts=state.drafts,
        current_index=state.current_index,
        final_answer=state.final_answer,
        suggestions=suggestions,
        success=success,
    )

# Revise：若未满足标准，进行聚合修订

def revise_node(state: AgentState) -> AgentState:
    llm = get_llm()
    skills_text = "\n".join(f"- {s}" for s in state.skills) if state.skills else "(无)"
    prompt = (
        "请根据评估建议与已发现技能，对最终答案进行一次聚合修订，使其更符合成功标准。\n"
        "输出要求：保持清晰结构与可执行性，不要冗长解释。\n"
        f"评估建议：\n{state.suggestions or ''}\n\n"
        f"已发现技能：\n{skills_text}\n\n"
        f"原最终答案：\n{state.final_answer or ''}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    revised = ai.content

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"Revise 阶段：\n{revised}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        skills=state.skills,
        rubric=state.rubric,
        plan_steps=state.plan_steps,
        drafts=state.drafts,
        current_index=state.current_index,
        final_answer=revised,
        suggestions=state.suggestions,
        success=True,  # 单次修订后结束
    )

# Finalize：输出最终答案

def finalize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = (
        "请直接输出最终答案，不要添加与过程相关的说明。保持结构化与可执行性。\n\n"
        f"最终答案：\n{state.final_answer or assemble_final_answer(state)}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"最终答案：\n{ai.content}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        skills=state.skills,
        rubric=state.rubric,
        plan_steps=state.plan_steps,
        drafts=state.drafts,
        current_index=state.current_index,
        final_answer=ai.content,
        suggestions=state.suggestions,
        success=True,
    )

# 解析 Discover 输出

def parse_discover_content(text: str) -> (List[str], List[str], List[str]):
    lines = [l.strip() for l in text.split("\n")]
    section = None
    skills: List[str] = []
    rubric: List[str] = []
    plans: List[str] = []
    for l in lines:
        low = l.lower()
        if low.startswith("技能"):
            section = "skills"; continue
        if low.startswith("成功标准"):
            section = "rubric"; continue
        if low.startswith("计划"):
            section = "plan"; continue
        if l.startswith("-"):
            item = l.lstrip("- ")
            if section == "skills":
                skills.append(item)
            elif section == "rubric":
                rubric.append(item)
        elif l[:1].isdigit():
            # 形如 "1. xxx"
            dot_pos = l.find(".")
            item = l[dot_pos+1:].strip() if dot_pos != -1 else l
            if section == "plan":
                plans.append(item)
    # 兜底：若解析失败，简单分段
    if not skills:
        skills = [s.strip() for s in text.split("技能")[-1].split("\n") if s.strip().startswith("-")]
        skills = [s.lstrip("- ") for s in skills]
    if not rubric:
        rubric = [r.strip() for r in text.split("成功标准")[-1].split("\n") if r.strip().startswith("-")]
        rubric = [r.lstrip("- ") for r in rubric]
    if not plans:
        plans = [p.strip() for p in text.split("计划")[-1].split("\n") if p.strip() and (p.strip()[0].isdigit() or p.startswith("-"))]
        plans = [p.lstrip("- ") for p in plans]
    # 去重与裁剪
    skills = [s for s in skills if s][:8]
    rubric = [r for r in rubric if r][:8]
    plans = [p for p in plans if p][:8]
    if not plans:
        plans = ["分解问题", "列出可执行清单", "给出注意事项与结论"]
    return skills, rubric, plans

# 解析评估结果成功与建议

def parse_success(text: str) -> bool:
    t = text.lower()
    positive = any(k in t for k in ["成功", "通过", "满足", "yes", "pass"])
    negative = any(k in t for k in ["失败", "不符合", "no", "not pass"])
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    return "结论" in text and ("成功" in text)


def parse_suggestions(text: str) -> str:
    lines = [l.strip() for l in text.split("\n")]
    started = False
    suggestions: List[str] = []
    for l in lines:
        if l.startswith("改进建议"):
            started = True
            continue
        if started:
            if l:
                suggestions.append(l)
    return "\n".join(suggestions).strip()

# 汇总最终答案

def assemble_final_answer(state: AgentState) -> str:
    parts = []
    for i, d in enumerate(state.drafts, start=1):
        parts.append(f"步骤 {i}:\n{d}")
    parts.append("注意事项：\n- 遵循成功标准\n- 保持清晰与可执行性")
    return "\n\n".join(parts)

# 路由

def after_discover(state: AgentState) -> str:
    return "execute"

def after_execute(state: AgentState) -> str:
    return "progress_check"

def after_progress(state: AgentState) -> str:
    return "execute" if state.current_index < len(state.plan_steps) else "evaluate"

def after_evaluate(state: AgentState) -> str:
    return "finalize" if state.success else "revise"

def after_revise(state: AgentState) -> str:
    return "finalize"

# 构建工作流

def create_self_discover_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("discover", discover_node)
    graph.add_node("execute", execute_step_node)
    graph.add_node("progress_check", progress_check_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("discover")

    graph.add_conditional_edges("discover", after_discover, {"execute": "execute"})
    graph.add_conditional_edges("execute", after_execute, {"progress_check": "progress_check"})
    graph.add_conditional_edges("progress_check", after_progress, {
        "execute": "execute",
        "evaluate": "evaluate",
    })
    graph.add_conditional_edges("evaluate", after_evaluate, {
        "finalize": "finalize",
        "revise": "revise",
    })
    graph.add_conditional_edges("revise", after_revise, {"finalize": "finalize"})

    graph.add_edge("finalize", END)

    app = graph.compile()
    return app

# CLI

def main():
    load_environment()
    app = create_self_discover_workflow()

    print("===== Self-Discover 演示 ======")
    print("该代理会：Discover（自发现技能/标准/计划）→ Execute（逐步执行）→ Evaluate（评估）→ Revise（修订）→ Finalize（输出最终答案）。")
    print("每次对话独立，不调用外部工具。输入 '退出' 结束对话。\n")
    print("示例问题（适合 Self-Discover 的任务）：")
    print("1. 为职场新人设计一周的时间管理方案，并自发现成功标准与执行计划")
    print("2. 规划一个三步骤的健康早餐准备流程，包含标准与注意事项")
    print("3. 讲解如何入门数据结构（数组、链表、栈/队列），并给出学习计划")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            init_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                question=user_input,
            )
            final_state = app.invoke(init_state)

            # 获取消息列表（兼容 dict / 数据类）
            messages = []
            if isinstance(final_state, dict):
                messages = final_state.get("messages", [])
            else:
                messages = getattr(final_state, "messages", [])

            # 输出最后一个 AI 回复（最终答案）
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print(f"代理: {msg.content}")
                    break
            else:
                print("代理: 对不起，未能生成答案。")
        except KeyboardInterrupt:
            print("\n代理: 对话被中断，再见！")
            break
        except Exception as e:
            print(f"代理: 发生错误: {str(e)}")

if __name__ == "__main__":
    main()