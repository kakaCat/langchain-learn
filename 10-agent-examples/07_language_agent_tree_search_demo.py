#!/usr/bin/env python3
"""
Module 8: Language Agent Tree Search (LATS) 示例（基于 LangGraph 的多分支搜索）

该示例实现一个简化版的 LATS：
- 针对用户问题进行“树状多分支规划与评估”，以 Beam Search 形式迭代：
  1) Propose：为每条候选路径提出若干下一步行动
  2) Evaluate：对所有扩展后的候选路径进行打分
  3) Prune：保留评分最高的若干条路径（Beam Width）
  4) Goal Check：判断是否已有足够完整的方案或达到最大深度
  5) Finalize：基于最佳路径生成最终答案

设计目标：
- 无外部工具调用（纯内部推理），每次对话独立
- 控制分支因子（branch_factor）、束宽（beam_width）、最大深度（max_depth）
- 使用 LangGraph 构建与编排工作流
"""
import os
import sys
from typing import List, Optional
from dataclasses import dataclass, field

# 保证中文输出
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 环境加载

def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 获取 LLM

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

# 数据结构

@dataclass
class StepCandidate:
    steps: List[str]
    score: float = 0.0
    rationale: str = ""

@dataclass
class AgentState:
    messages: List[BaseMessage]
    question: str
    beam: List[StepCandidate] = field(default_factory=list)
    best: Optional[StepCandidate] = None
    depth: int = 0
    max_depth: int = 4
    branch_factor: int = 3
    beam_width: int = 2
    done: bool = False

# 节点：提出下一步行动（Propose）

def propose_node(state: AgentState) -> AgentState:
    llm = get_llm()
    new_candidates: List[StepCandidate] = []

    for cand_idx, cand in enumerate(state.beam or [StepCandidate(steps=[])]):
        plan_text = "\n".join(f"- {s}" for s in cand.steps) if cand.steps else "(空)"
        prompt = (
            "你是一个计划生成器。给定用户问题和当前部分计划，提出若干可能的下一步行动。\n"
            "要求：每条行动用一句话表达，尽量具体可执行，不要写理由。\n"
            f"用户问题：{state.question}\n"
            f"当前计划：\n{plan_text}\n"
            f"请提出不超过 {state.branch_factor} 个下一步行动，每个独立一行。"
        )
        ai = llm.invoke([HumanMessage(content=prompt)])
        proposals = [p.strip("- ") for p in ai.content.split("\n") if p.strip()]
        proposals = proposals[: state.branch_factor]
        for step in proposals:
            new_candidates.append(StepCandidate(steps=cand.steps + [step]))

    # 记录到消息
    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=(
        "Propose 阶段：已为每条候选路径提出下一步行动，候选总数 = "
        f"{len(new_candidates)}\n示例候选：\n" + "\n".join(
            f"- {' -> '.join(c.steps)}" for c in new_candidates[:min(5, len(new_candidates))]
        )
    )))

    return AgentState(
        messages=new_messages,
        question=state.question,
        beam=new_candidates,
        best=state.best,
        depth=state.depth + 1,
        max_depth=state.max_depth,
        branch_factor=state.branch_factor,
        beam_width=state.beam_width,
        done=False,
    )

# 节点：评估候选路径（Evaluate）

def evaluate_node(state: AgentState) -> AgentState:
    llm = get_llm()
    if not state.beam:
        return state

    listing = "\n".join(
        f"[{i}] {' -> '.join(c.steps)}" for i, c in enumerate(state.beam)
    )
    prompt = (
        "你是评估器。请针对下列候选方案（为解决用户问题的部分或完整计划），按“完成度/清晰度/可执行性/风险控制”综合评分0-10，并给出简短理由。\n"
        "严格输出：每行格式为 [index] score=<0-10> rationale=<一句话>。\n"
        f"用户问题：{state.question}\n候选列表：\n{listing}\n"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    lines = [l.strip() for l in ai.content.split("\n") if l.strip()]

    index_to_score = {}
    index_to_rat = {}
    for line in lines:
        # 解析形如：[0] score=7.5 rationale=...
        try:
            if line.startswith("[") and "]" in line:
                idx_str = line[1: line.index("]")]
                idx = int(idx_str)
                s_pos = line.find("score=")
                r_pos = line.find("rationale=")
                if s_pos != -1:
                    s_end = line.find(" ", s_pos)
                    score_str = line[s_pos + 6: s_end if s_end != -1 else None]
                    score = float(score_str)
                else:
                    score = 5.0
                rationale = line[r_pos + 10:].strip() if r_pos != -1 else ""
                index_to_score[idx] = score
                index_to_rat[idx] = rationale
        except Exception:
            continue

    # 赋分
    new_beam: List[StepCandidate] = []
    for idx, cand in enumerate(state.beam):
        score = index_to_score.get(idx, 5.0)
        rat = index_to_rat.get(idx, "")
        new_beam.append(StepCandidate(steps=cand.steps, score=score, rationale=rat))

    # 记录消息
    new_messages = state.messages.copy()
    beam_report = "\n".join(
        f"- ({c.score:.2f}) {' -> '.join(c.steps)}" for c in new_beam[:min(5, len(new_beam))]
    )
    new_messages.append(AIMessage(content=f"Evaluate 阶段：评分完成（示例）\n{beam_report}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        beam=new_beam,
        best=state.best,
        depth=state.depth,
        max_depth=state.max_depth,
        branch_factor=state.branch_factor,
        beam_width=state.beam_width,
        done=False,
    )

# 节点：剪枝（Prune）

def prune_node(state: AgentState) -> AgentState:
    if not state.beam:
        return state
    sorted_beam = sorted(state.beam, key=lambda c: c.score, reverse=True)
    pruned = sorted_beam[: state.beam_width]
    best = pruned[0] if pruned else None

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=(
        "Prune 阶段：已保留评分最高的候选路径（Beam Width = "
        f"{state.beam_width}）。当前最佳：\n- " + (" -> ".join(best.steps) if best else "(无)")
    )))

    return AgentState(
        messages=new_messages,
        question=state.question,
        beam=pruned,
        best=best,
        depth=state.depth,
        max_depth=state.max_depth,
        branch_factor=state.branch_factor,
        beam_width=state.beam_width,
        done=False,
    )

# 节点：目标检查（Goal Check）

def goal_check_node(state: AgentState) -> AgentState:
    llm = get_llm()
    if not state.beam:
        return state

    # 若达到最大深度则直接结束
    if state.depth >= state.max_depth:
        new_messages = state.messages.copy()
        new_messages.append(AIMessage(content=f"Goal Check：达到最大深度 {state.max_depth}，准备生成最终答案。"))
        return AgentState(
            messages=new_messages,
            question=state.question,
            beam=state.beam,
            best=state.best or state.beam[0],
            depth=state.depth,
            max_depth=state.max_depth,
            branch_factor=state.branch_factor,
            beam_width=state.beam_width,
            done=True,
        )

    # 询问评估器是否已有足够完整的方案
    listing = "\n".join(
        f"[{i}] {' -> '.join(c.steps)}" for i, c in enumerate(state.beam)
    )
    prompt = (
        "你是方案完整性检查器。针对候选方案列表，判断是否有任一项已足够完整，可用于生成最终答案。\n"
        "严格输出：若存在，请输出 `index=<数字>`；若不可用，请输出 `continue`。不输出其他内容。\n"
        f"用户问题：{state.question}\n候选列表：\n{listing}\n"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])
    content = ai.content.strip().lower()

    chosen_idx: Optional[int] = None
    if content.startswith("index="):
        try:
            chosen_idx = int(content.split("=", 1)[1].strip())
        except Exception:
            chosen_idx = None

    new_messages = state.messages.copy()
    if chosen_idx is not None and 0 <= chosen_idx < len(state.beam):
        best = state.beam[chosen_idx]
        new_messages.append(AIMessage(content=f"Goal Check：发现可用完整方案，选用索引 [{chosen_idx}]。"))
        return AgentState(
            messages=new_messages,
            question=state.question,
            beam=state.beam,
            best=best,
            depth=state.depth,
            max_depth=state.max_depth,
            branch_factor=state.branch_factor,
            beam_width=state.beam_width,
            done=True,
        )
    else:
        new_messages.append(AIMessage(content="Goal Check：暂无完整方案，继续搜索下一深度。"))
        return AgentState(
            messages=new_messages,
            question=state.question,
            beam=state.beam,
            best=state.best,
            depth=state.depth,
            max_depth=state.max_depth,
            branch_factor=state.branch_factor,
            beam_width=state.beam_width,
            done=False,
        )

# 节点：最终生成（Finalize）

def finalize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    best = state.best or (state.beam[0] if state.beam else StepCandidate(steps=[]))
    plan_text = "\n".join(f"- {s}" for s in best.steps) if best.steps else "(无计划)"
    prompt = (
        "你是方案整理器。请基于所选最佳步骤，输出最终答案。\n"
        "要求：清晰、结构化、可执行；必要时给出步骤与注意事项。\n"
        f"用户问题：{state.question}\n所选步骤：\n{plan_text}\n"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"最终答案：\n{ai.content}"))

    return AgentState(
        messages=new_messages,
        question=state.question,
        beam=state.beam,
        best=best,
        depth=state.depth,
        max_depth=state.max_depth,
        branch_factor=state.branch_factor,
        beam_width=state.beam_width,
        done=True,
    )

# 路由判断

def check_next(state: AgentState) -> str:
    # 初始或继续搜索时的顺序：propose -> evaluate -> prune -> goal_check -> (continue or finalize)
    if not state.done:
        # 若刚完成 propose 阶段（depth 递增），下一步是 evaluate
        # 为简化，使用状态字段与图的结构控制具体流转
        return "evaluate"
    # 如果 done=True，进入 finalize
    return "finalize"

# 构建工作流

def create_lats_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("propose", propose_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("prune", prune_node)
    graph.add_node("goal_check", goal_check_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("propose")

    # propose -> evaluate
    graph.add_edge("propose", "evaluate")
    # evaluate -> prune
    graph.add_edge("evaluate", "prune")
    # prune -> goal_check
    graph.add_edge("prune", "goal_check")

    # goal_check：若继续则回到 propose，否则 finalize
    def next_after_goal(state: AgentState) -> str:
        return "finalize" if state.done else "propose"

    graph.add_conditional_edges("goal_check", next_after_goal, {
        "propose": "propose",
        "finalize": "finalize",
    })

    # finalize -> END
    graph.add_edge("finalize", END)

    app = graph.compile()
    return app

# CLI 演示

def main():
    load_environment()
    app = create_lats_workflow()

    print("===== Language Agent Tree Search (LATS) 演示 ======")
    print("该代理会进行：Propose → Evaluate → Prune → Goal Check →（继续/终结）→ Finalize 的多分支搜索。")
    print("每次对话独立，不调用外部工具。输入 '退出' 结束对话。\n")
    print("示例问题（适合树搜索的多步规划）：")
    print("1. 设计一个三天的北京城市探索行程，强调效率与交通方便")
    print("2. 为新手制定一个四周的跑步入门计划，包含逐周目标与注意事项")
    print("3. 解释如何用分治法解决归并排序，并给出步骤与示例")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            init_state = AgentState(
                messages=[HumanMessage(content=user_input)],
                question=user_input,
                beam=[],
                best=None,
                depth=0,
                max_depth=4,
                branch_factor=3,
                beam_width=2,
                done=False,
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