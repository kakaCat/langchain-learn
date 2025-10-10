#!/usr/bin/env python3
"""
Module 7: Reflexion 模式代理示例（多轮反思记忆，无外部工具）

Reflexion 是一种“带记忆的反思改进”模式：
- 先尝试生成答案（Attempt）
- 再进行结构化评估（Evaluate），给出成败判定与改进建议
- 将改进建议作为“反思记忆”（Reflection Memory）
- 基于记忆再次尝试（Attempt），循环多轮，直到成功或达到迭代上限

本示例：
- 不调用外部工具或检索，每次对话独立（无会话历史记忆）
- 使用 LangGraph 管理工作流与条件路由
- 提供 CLI 交互与适配“dict/数据类”两种返回形态
"""
import os
import sys
from typing import List, Optional
from dataclasses import dataclass, field

# 确保中文显示正常
sys.stdout.reconfigure(encoding='utf-8')

# 第三方库
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 加载环境变量

def load_environment():
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

# 获取模型

def get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "768"))
    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 120,
        "max_retries": 3,
        "request_timeout": 120,
        "verbose": True,
        "base_url": base_url,
    }
    return ChatOpenAI(**kwargs)

# 状态定义

@dataclass
class AgentState:
    messages: List[BaseMessage]
    memory: List[str] = field(default_factory=list)  # 反思记忆（跨轮次聚合）
    attempt: Optional[str] = None                   # 当前尝试的答案
    feedback: Optional[str] = None                  # 评估反馈（理由+建议）
    success: bool = False                           # 是否达成成功标准
    iterations: int = 0                             # 已迭代次数
    max_iterations: int = 3                         # 最大迭代次数

# 尝试节点：生成答案，若存在记忆则按记忆改进

def attempt_node(state: AgentState) -> AgentState:
    llm = get_llm()
    user_msg = next((m for m in state.messages if isinstance(m, HumanMessage)), None)
    question = user_msg.content if user_msg else ""

    guidelines = "\n".join(f"- {m}" for m in state.memory) if state.memory else "(无)"
    prompt = (
        "你将尝试回答用户问题。若存在反思记忆，请严格遵循其中的改进建议。\n"
        "输出要求：清晰、结构化、可执行；必要时给出分点与步骤。\n"
        f"反思记忆：\n{guidelines}\n\n"
        f"用户问题：{question}"
    )
    ai = llm.invoke([HumanMessage(content=prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"尝试答案（第{state.iterations + 1}轮）：\n{ai.content}"))
    return AgentState(
        messages=new_messages,
        memory=state.memory,
        attempt=ai.content,
        feedback=None,
        success=False,
        iterations=state.iterations,
        max_iterations=state.max_iterations,
    )

# 评估节点：判定成功/失败，并给出改进建议

def evaluate_node(state: AgentState) -> AgentState:
    llm = get_llm()
    user_msg = next((m for m in state.messages if isinstance(m, HumanMessage)), None)
    question = user_msg.content if user_msg else ""

    eval_prompt = (
        "你是评估器，请对上述尝试答案进行判定并给出改进建议。\n"
        "请严格输出以下结构：\n"
        "结论: 成功 或 失败\n"
        "理由: （简明说明是否满足用户需求、是否清晰、是否可执行）\n"
        "改进建议: （如失败，给出具体、可执行的改进要点；如成功，可给轻微优化建议或写'无'）\n\n"
        f"用户问题：\n{question}\n\n"
        f"尝试答案：\n{state.attempt or ''}"
    )
    ai = llm.invoke(state.messages + [HumanMessage(content=eval_prompt)])
    feedback_text = ai.content

    success = parse_success(feedback_text)

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"评估：\n{feedback_text}"))

    return AgentState(
        messages=new_messages,
        memory=state.memory,
        attempt=state.attempt,
        feedback=feedback_text,
        success=success,
        iterations=state.iterations,
        max_iterations=state.max_iterations,
    )

# 反思节点：若失败，将改进建议写入记忆，准备下一轮尝试

def reflect_node(state: AgentState) -> AgentState:
    llm = get_llm()
    reflect_prompt = (
        "基于上述评估中的改进建议，提炼为可复用的反思记忆要点（3-6条）。\n"
        "要求：简洁、具体、可执行；用于指导下一轮答案生成。\n"
        f"评估内容：\n{state.feedback or ''}"
    )
    ai = llm.invoke(state.messages + [HumanMessage(content=reflect_prompt)])
    reflection_points = [p.strip("- ") for p in ai.content.split("\n") if p.strip()]

    new_memory = state.memory + reflection_points
    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"反思记忆更新：\n" + "\n".join(f"- {p}" for p in reflection_points)))

    return AgentState(
        messages=new_messages,
        memory=new_memory,
        attempt=state.attempt,
        feedback=state.feedback,
        success=False,
        iterations=state.iterations + 1,
        max_iterations=state.max_iterations,
    )

# 终结节点：输出最终答案（成功或迭代上限）

def finalize_node(state: AgentState) -> AgentState:
    llm = get_llm()
    status_text = "成功" if state.success else f"未在 {state.max_iterations} 轮内达成成功（给出最佳答案）"
    guidelines = "\n".join(f"- {m}" for m in state.memory) if state.memory else "(无)"
    final_prompt = (
        f"请基于当前最佳尝试与反思记忆，输出最终答案（状态：{status_text}）。\n"
        "要求：清晰、结构化、可执行；必要时给出步骤。\n"
        f"反思记忆：\n{guidelines}\n\n"
        f"当前最佳尝试：\n{state.attempt or ''}"
    )
    ai = llm.invoke(state.messages + [HumanMessage(content=final_prompt)])

    new_messages = state.messages.copy()
    new_messages.append(AIMessage(content=f"最终答案：\n{ai.content}"))

    return AgentState(
        messages=new_messages,
        memory=state.memory,
        attempt=state.attempt,
        feedback=state.feedback,
        success=True,
        iterations=state.iterations,
        max_iterations=state.max_iterations,
    )

# 成功判定解析（简单规则）：包含“成功/通过/满足/yes”且不包含“失败/no”

def parse_success(text: str) -> bool:
    t = text.lower()
    positive = any(k in t for k in ["成功", "通过", "满足", "yes", "pass"])
    negative = any(k in t for k in ["失败", "不符合", "no", "not pass"])
    if positive and not negative:
        return True
    if negative and not positive:
        return False
    # 模糊情况：若出现“结论: 成功”优先判定成功
    return "结论" in text and ("成功" in text)

# 条件路由

def check_next(state: AgentState) -> str:
    # 初始：无尝试答案 → 进入 attempt
    if state.attempt is None:
        return "attempt"
    # 未评估 → 进入 evaluate
    if state.feedback is None:
        return "evaluate"
    # 若成功 → finalize
    if state.success:
        return "finalize"
    # 未成功：是否可继续迭代
    if state.iterations < state.max_iterations:
        return "reflect"
    # 达到上限 → finalize
    return "finalize"

# 构建工作流

def create_reflexion_workflow():
    graph = StateGraph(AgentState)
    graph.add_node("attempt", attempt_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("attempt")

    graph.add_conditional_edges("attempt", check_next, {
        "evaluate": "evaluate",
        "attempt": "attempt",
        "finalize": "finalize",
    })

    graph.add_conditional_edges("evaluate", check_next, {
        "finalize": "finalize",
        "reflect": "reflect",
        "evaluate": "evaluate",
    })

    graph.add_conditional_edges("reflect", check_next, {
        "attempt": "attempt",
        "finalize": "finalize",
    })

    graph.add_conditional_edges("finalize", check_next, {"finalize": "finalize", "END": END})

    app = graph.compile()
    return app

# CLI 演示

def main():
    load_environment()
    app = create_reflexion_workflow()

    print("===== Reflexion 模式代理演示 ======")
    print("该代理会进行多轮：尝试 → 评估 → 反思记忆 → 再尝试，直到成功或达到迭代上限（不调用外部工具）。")
    print("注意：此版本不保存跨问题的会话历史，每次对话都是独立的。输入 '退出' 结束对话。\n")
    print("示例问题（适合 Reflexion 多轮改进）：")
    print("1. 给出提升专注力的 5 条建议，若不够具体可执行则自动改进")
    print("2. 解释二分查找算法并给出示例，若不清晰则改进结构和示例")
    print("3. 设计一个三步泡茶流程，若缺少安全注意事项则自动补充并改进")

    while True:
        try:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit", "q"]:
                print("代理: 再见！")
                break

            initial_state = AgentState(messages=[HumanMessage(content=user_input)], memory=[], attempt=None, feedback=None, success=False, iterations=0, max_iterations=3)
            final_state = app.invoke(initial_state)

            # 兼容字典或数据类返回，统一获取消息列表
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