#!/usr/bin/env python3
"""
11 - Human-in-the-Loop (人机协同) 基础示例

演示如何在 LangGraph 工作流中实现人机协同：
1. Agent 执行过程中暂停并请求人类输入
2. 基于人类反馈调整执行路径
3. 使用 interrupt_before/after 机制实现断点
4. 状态持久化和恢复

适用场景：
- 需要人工审核的决策点
- 不确定性高的任务
- 需要用户授权的操作
- 教学和演示场景
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True
    )


def get_llm(model: Optional[str] = None, temperature: float = 0.7) -> object:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=1000,
        )


# ============================================================================
# 状态定义
# ============================================================================


class WorkflowState(TypedDict):
    """工作流状态"""

    user_request: str
    plan: list[str]
    current_step: int
    step_results: list[str]
    human_feedback: Optional[str]
    should_continue: bool
    final_output: Optional[str]


# ============================================================================
# 节点实现
# ============================================================================


def create_plan_node(state: WorkflowState) -> WorkflowState:
    """创建执行计划"""
    llm = get_llm()

    prompt = f"""
用户需求：{state['user_request']}

请将这个需求分解为 3-5 个可执行的步骤。
只输出步骤列表，每行一个步骤，格式如下：
1. 步骤描述
2. 步骤描述
...
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    # 解析步骤
    steps = []
    for line in content.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # 去除序号
            step = line.split(".", 1)[-1].strip() if "." in line else line[1:].strip()
            if step:
                steps.append(step)

    state["plan"] = steps if steps else [state["user_request"]]
    state["current_step"] = 0
    state["step_results"] = []

    print(f"\n✓ 生成执行计划：")
    for i, step in enumerate(state["plan"], 1):
        print(f"  {i}. {step}")

    return state


def execute_step_node(state: WorkflowState) -> WorkflowState:
    """执行当前步骤"""
    if state["current_step"] >= len(state["plan"]):
        state["should_continue"] = False
        return state

    llm = get_llm()
    current_step = state["plan"][state["current_step"]]

    prompt = f"""
总体任务：{state['user_request']}
当前步骤：{current_step}

{'之前的步骤结果：' + chr(10).join(f'- {r}' for r in state['step_results']) if state['step_results'] else ''}

{'用户反馈：' + state['human_feedback'] if state.get('human_feedback') else ''}

请执行当前步骤并给出结果（1-2 句话）。
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip()

    state["step_results"].append(f"步骤 {state['current_step'] + 1}: {result}")
    state["current_step"] += 1

    print(f"\n✓ 执行步骤 {state['current_step']}/{len(state['plan'])}")
    print(f"  结果：{result}")

    # 清除已使用的反馈
    state["human_feedback"] = None

    return state


def request_human_input_node(state: WorkflowState) -> WorkflowState:
    """请求人类输入（这个节点会触发中断）"""
    print("\n" + "=" * 80)
    print("🤚 Agent 请求人类反馈")
    print("=" * 80)
    print(f"\n已完成步骤：{state['current_step']}/{len(state['plan'])}")
    print(f"当前进展：")
    for result in state["step_results"]:
        print(f"  - {result}")

    print(
        "\n提示：Agent 将暂停等待您的输入。\n"
        "您可以：\n"
        "  1. 提供反馈建议\n"
        "  2. 要求修改某个步骤\n"
        "  3. 批准继续执行（输入 'continue' 或直接回车）\n"
        "  4. 终止执行（输入 'stop'）"
    )

    # 注意：实际的输入会在 workflow.stream() 的循环中处理
    # 这里只是展示提示信息
    return state


def finalize_node(state: WorkflowState) -> WorkflowState:
    """生成最终输出"""
    llm = get_llm()

    results_str = "\n".join(state["step_results"])

    prompt = f"""
任务：{state['user_request']}

执行过程：
{results_str}

请总结执行结果，给出简洁的最终报告（2-3 段）。
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    state["final_output"] = response.content.strip()

    print("\n" + "=" * 80)
    print("✓ 任务完成")
    print("=" * 80)
    print(state["final_output"])

    return state


# ============================================================================
# 条件判断
# ============================================================================


def should_continue(state: WorkflowState) -> Literal["execute", "request_input", "finalize"]:
    """判断下一步操作"""
    # 如果所有步骤完成
    if state["current_step"] >= len(state["plan"]):
        return "finalize"

    # 每 2 个步骤请求一次人类输入
    if state["current_step"] > 0 and state["current_step"] % 2 == 0:
        return "request_input"

    return "execute"


# ============================================================================
# 工作流构建
# ============================================================================


def create_hitl_workflow():
    """创建 Human-in-the-Loop 工作流"""
    graph = StateGraph(WorkflowState)

    # 添加节点
    graph.add_node("plan", create_plan_node)
    graph.add_node("execute", execute_step_node)
    graph.add_node("request_input", request_human_input_node)
    graph.add_node("finalize", finalize_node)

    # 定义流程
    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")

    # 条件分支
    graph.add_conditional_edges(
        "execute",
        should_continue,
        {
            "execute": "execute",  # 继续执行
            "request_input": "request_input",  # 请求输入
            "finalize": "finalize",  # 完成
        },
    )

    # 人类输入后继续执行
    graph.add_edge("request_input", "execute")
    graph.add_edge("finalize", END)

    # 使用 MemorySaver 支持状态持久化
    # interrupt_before 指定在哪个节点前暂停
    return graph.compile(
        checkpointer=MemorySaver(), interrupt_before=["request_input"]
    )


# ============================================================================
# 交互式执行
# ============================================================================


def run_interactive_demo():
    """运行交互式演示"""
    load_environment()
    workflow = create_hitl_workflow()

    print("\n" + "=" * 80)
    print("Human-in-the-Loop 交互式演示")
    print("=" * 80)

    user_request = input("\n请输入任务（回车使用默认）: ").strip()
    if not user_request:
        user_request = "制定一个学习 Python 的计划"
        print(f"使用默认任务：{user_request}")

    # 初始状态
    initial_state = {
        "user_request": user_request,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "human_feedback": None,
        "should_continue": True,
        "final_output": None,
    }

    # 配置（用于状态持久化）
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    # 开始执行
    print("\n开始执行...")

    try:
        # 第一次执行（直到遇到 interrupt）
        for event in workflow.stream(initial_state, config):
            if "__interrupt__" in event:
                # 遇到中断点
                print("\n⏸ 工作流已暂停")

                # 获取当前状态
                current_state = workflow.get_state(config)
                print(f"\n当前节点：{current_state.next}")
                print(f"待执行节点：{list(current_state.next)}")

                # 请求用户输入
                user_input = input("\n您的反馈（回车继续）: ").strip()

                if user_input.lower() == "stop":
                    print("\n用户终止执行")
                    break

                # 更新状态（添加人类反馈）
                if user_input and user_input.lower() != "continue":
                    # 获取当前状态值
                    state_values = current_state.values
                    state_values["human_feedback"] = user_input
                    # 更新状态
                    workflow.update_state(config, state_values)
                    print(f"\n✓ 已记录您的反馈：{user_input}")

                # 继续执行
                print("\n继续执行...")
                for event2 in workflow.stream(None, config):
                    if "__interrupt__" in event2:
                        # 再次遇到中断
                        break

        # 获取最终状态
        final_state = workflow.get_state(config)
        if final_state.values.get("final_output"):
            print("\n任务已完成！")

    except Exception as e:
        print(f"\n执行出错：{e}")
        import traceback

        traceback.print_exc()


# ============================================================================
# 自动演示（用于测试）
# ============================================================================


def run_automated_demo():
    """运行自动演示（模拟用户输入）"""
    load_environment()
    workflow = create_hitl_workflow()

    print("\n" + "=" * 80)
    print("Human-in-the-Loop 自动演示")
    print("=" * 80)

    initial_state = {
        "user_request": "制定一个周末学习计划",
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "human_feedback": None,
        "should_continue": True,
        "final_output": None,
    }

    config = {"configurable": {"thread_id": "auto-demo-1"}}

    # 模拟的用户反馈
    simulated_feedbacks = [
        "请增加更多实践项目",
        "时间安排要更灵活",
    ]
    feedback_index = 0

    try:
        for event in workflow.stream(initial_state, config):
            if "__interrupt__" in event:
                print("\n⏸ 工作流暂停（自动模式）")

                # 模拟用户输入
                if feedback_index < len(simulated_feedbacks):
                    feedback = simulated_feedbacks[feedback_index]
                    print(f"[模拟用户输入]: {feedback}")

                    current_state = workflow.get_state(config)
                    state_values = current_state.values
                    state_values["human_feedback"] = feedback
                    workflow.update_state(config, state_values)

                    feedback_index += 1
                else:
                    print("[模拟用户输入]: continue")

                # 继续执行
                print("继续执行...")
                for event2 in workflow.stream(None, config):
                    if "__interrupt__" in event2:
                        break

    except Exception as e:
        print(f"\n执行出错：{e}")
        import traceback

        traceback.print_exc()


# ============================================================================
# 主入口
# ============================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--auto":
        run_automated_demo()
    else:
        run_interactive_demo()
