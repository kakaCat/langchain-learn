#!/usr/bin/env python3
"""
简化版 HITL (Human-in-the-Loop) 人机交互演示
真实的命令行交互审批流程

功能特点：
1. 用户输入任务名称
2. 工作流暂停等待人工审批
3. 用户决定批准或拒绝（y/n）
4. 可选输入审批意见
5. 根据审批结果执行或终止任务

运行方式：
- 交互模式：python 04_simple_hitl_demo.py
- 演示模式：python 04_simple_hitl_demo.py --demo

这是一个简化版本，适合理解 HITL 的基本原理和人机交互流程。
如需完整的自动化演示场景，请参考 04_langgraph_hitl.py
"""

import sys

from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


@dataclass
class TaskState:
    """任务状态"""
    task_name: str = ""
    approved: bool = False
    comment: str = ""
    reviewed: bool = False  # 是否已审批（避免重复暂停）


def submit_task(state: TaskState) -> TaskState:
    """提交任务"""
    print(f"\n📝 提交任务: {state.task_name}")
    return state


def wait_approval(state: TaskState) -> TaskState:
    """等待审批 - 暂停点"""
    if not state.reviewed:
        print(f"⏸️  等待审批中...")
        interrupt("请通过 update_state() 提供审批结果")
    return state


def execute_task(state: TaskState) -> TaskState:
    """执行任务"""
    if state.approved:
        print(f"\n✅ 任务已批准！开始执行...")
        print(f"   执行中: {state.task_name}")
        print(f"   批注: {state.comment}")
    else:
        print(f"\n❌ 任务被拒绝，不执行")
        print(f"   拒绝理由: {state.comment}")
    return state



def run_interactive_mode():
    """交互模式 - 真实人机交互"""
    # 1. 创建工作流
    workflow = StateGraph(TaskState)
    workflow.add_node("submit", submit_task)
    workflow.add_node("wait", wait_approval)
    workflow.add_node("execute", execute_task)

    workflow.set_entry_point("submit")
    workflow.add_edge("submit", "wait")
    workflow.add_edge("wait", "execute")
    workflow.add_edge("execute", END)

    # 编译时需要 checkpointer
    app = workflow.compile(checkpointer=MemorySaver())

    # 2. 获取用户输入
    print("="*50)
    print("🚀 HITL 人机交互审批流程")
    print("="*50)
    print("\n💡 提示：如果无法输入，请使用演示模式: python 04_simple_hitl_demo.py --demo\n")

    try:
        task_name = input("📝 请输入任务名称（直接回车使用默认'购买新服务器'）: ").strip()
        if not task_name:
            task_name = "购买新服务器"

        initial_state = TaskState(task_name=task_name)
        config = {"configurable": {"thread_id": "interactive_demo"}}

        # 3. 启动工作流（会在 wait_approval 节点暂停）
        print(f"\n>>> 启动工作流...")
        app.invoke(initial_state, config)

        # 4. 查看当前状态
        current = app.get_state(config)
        print(f"\n📊 当前状态:")
        print(f"   任务: {current.values['task_name']}")
        print(f"   已审批: {current.values['approved']}")

        # 5. 等待人工审批决策
        print("\n" + "-"*50)
        print("👤 请进行人工审批:")
        print("-"*50)

        while True:
            decision = input("是否批准此任务？(y/n): ").strip().lower()
            if decision in ['y', 'yes', 'n', 'no']:
                break
            print("❌ 无效输入，请输入 y 或 n")

        if decision in ['y', 'yes']:
            comment = input("请输入审批意见（可选，直接回车跳过）: ").strip()
            if not comment:
                comment = "同意执行"

            print(f"\n✅ 审批通过！")
            app.update_state(
                config,
                {"approved": True, "comment": comment, "reviewed": True}
            )
        else:
            comment = input("请输入拒绝理由（可选，直接回车跳过）: ").strip()
            if not comment:
                comment = "不同意执行"

            print(f"\n❌ 审批拒绝！")
            app.update_state(
                config,
                {"approved": False, "comment": comment, "reviewed": True}
            )

        # 6. 恢复执行
        print("\n🔄 恢复工作流执行...")
        result = app.invoke(None, config)

        # 7. 显示最终结果
        print("\n" + "="*50)
        print("📋 最终结果:")
        print("="*50)
        print(f"   任务名称: {result['task_name']}")
        print(f"   审批状态: {'✅ 已批准' if result['approved'] else '❌ 已拒绝'}")
        print(f"   审批意见: {result['comment']}")
        print("\n✨ 流程完成！")
        print("="*50)

    except EOFError:
        print("\n\n❌ 检测到无法接收输入（可能在IDE中运行）")
        print("💡 请尝试以下方式之一：")
        print("   1. 在真实终端中运行: python 04_simple_hitl_demo.py")
        print("   2. 使用演示模式: python 04_simple_hitl_demo.py --demo")
        print("   3. 使用管道输入: echo -e '任务名\\ny\\n意见\\n' | python 04_simple_hitl_demo.py")
        sys.exit(1)


def main():
    """主函数 - 根据参数选择模式"""
    # 检查命令行参数
    run_interactive_mode()
        


if __name__ == "__main__":
    main()
