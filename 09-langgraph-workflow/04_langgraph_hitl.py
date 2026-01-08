#!/usr/bin/env python3
"""
LangGraph HITL (Human-in-the-Loop) Demo
演示如何在 LangGraph 工作流中实现人机协作

本示例实现了以下功能：
1. 使用 interrupt 暂停工作流等待人工输入
2. 实现审批流程（单级和多级审批）
3. 使用 checkpoint 保存暂停状态
4. 支持审批通过和拒绝的条件分支
5. 展示 stream 模式观察暂停过程

学习要点：
- NodeInterrupt：在节点中主动暂停工作流
- update_state()：提供人工输入后恢复执行
- MemorySaver：保存暂停状态以便恢复
- 实现真实的人工审批场景
"""

from dataclasses import dataclass
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt


# 定义状态类
@dataclass
class ApprovalState:
    """审批工作流状态"""
    request_content: str = ""  # 审批请求内容
    request_amount: float = 0.0  # 请求金额
    approval_status: str = "pending"  # 审批状态: pending, approved, rejected
    approval_comment: str = ""  # 审批意见
    approver: str = ""  # 当前审批人
    approval_level: int = 0  # 审批级别（用于多级审批）


# ========== 示例1: 基本单级审批 ==========

def submit_request_node(state: ApprovalState) -> ApprovalState:
    """节点1: 提交审批请求"""
    print(f"\n📝 提交审批请求:")
    print(f"   内容: {state.request_content}")
    print(f"   金额: ¥{state.request_amount:,.2f}")
    return state


def wait_for_approval_node(state: ApprovalState) -> ApprovalState:
    """节点2: 等待人工审批（暂停点）"""
    # 如果还没有审批结果，则暂停等待
    if state.approval_status == "pending":
        print(f"\n⏸️  工作流已暂停，等待审批...")
        print(f"   审批人: {state.approver}")

        # 使用 interrupt 暂停工作流，等待人工输入
        # 外部需要调用 update_state() 提供审批结果才能继续
        interrupt(f"等待 {state.approver} 审批。请使用 update_state() 更新审批状态。")

    # 审批状态已更新，继续执行
    return state


def approval_decision_router(state: ApprovalState) -> str:
    """路由函数: 根据审批结果决定下一步"""
    if state.approval_status == "approved":
        return "process_approval"
    else:
        return "process_rejection"


def process_approval_node(state: ApprovalState) -> ApprovalState:
    """节点3a: 处理审批通过"""
    print(f"\n✅ 审批通过!")
    print(f"   审批意见: {state.approval_comment}")
    print(f"   开始执行请求...")
    return state


def process_rejection_node(state: ApprovalState) -> ApprovalState:
    """节点3b: 处理审批拒绝"""
    print(f"\n❌ 审批被拒绝!")
    print(f"   拒绝理由: {state.approval_comment}")
    print(f"   请求已终止。")
    return state


def create_approval_workflow():
    """创建单级审批工作流

    流程: submit -> wait_approval -> (approved/rejected) -> process -> END
    """
    workflow = StateGraph(ApprovalState)

    # 添加节点
    workflow.add_node("submit", submit_request_node)
    workflow.add_node("wait_approval", wait_for_approval_node)
    workflow.add_node("process_approval", process_approval_node)
    workflow.add_node("process_rejection", process_rejection_node)

    # 设置流程
    workflow.set_entry_point("submit")
    workflow.add_edge("submit", "wait_approval")

    # 条件分支：根据审批结果路由
    workflow.add_conditional_edges(
        "wait_approval",
        approval_decision_router,
        {
            "process_approval": "process_approval",
            "process_rejection": "process_rejection"
        }
    )

    workflow.add_edge("process_approval", END)
    workflow.add_edge("process_rejection", END)

    # 使用 MemorySaver 保存暂停状态
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


# ========== 示例2: 多级审批（经理->总监->CEO）==========

@dataclass
class MultiLevelApprovalState:
    """多级审批状态"""
    request_content: str = ""
    request_amount: float = 0.0
    manager_approved: bool = False
    manager_comment: str = ""
    director_approved: bool = False
    director_comment: str = ""
    ceo_approved: bool = False
    ceo_comment: str = ""
    current_level: str = "manager"  # manager, director, ceo, completed, rejected
    final_status: str = "pending"


def submit_multi_request_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """提交多级审批请求"""
    print(f"\n📝 提交审批请求:")
    print(f"   内容: {state.request_content}")
    print(f"   金额: ¥{state.request_amount:,.2f}")
    print(f"   需要经过: 经理 → 总监 → CEO 三级审批")
    return state


def wait_manager_approval_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """等待经理审批"""
    if not state.manager_approved and state.manager_comment == "":
        print(f"\n⏸️  [第1级] 等待经理审批...")
        interrupt("等待经理审批。请更新 manager_approved 和 manager_comment。")
    return state


def wait_director_approval_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """等待总监审批"""
    if not state.director_approved and state.director_comment == "":
        print(f"\n⏸️  [第2级] 等待总监审批...")
        interrupt("等待总监审批。请更新 director_approved 和 director_comment。")
    return state


def wait_ceo_approval_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """等待CEO审批"""
    if not state.ceo_approved and state.ceo_comment == "":
        print(f"\n⏸️  [第3级] 等待CEO审批...")
        interrupt("等待CEO审批。请更新 ceo_approved 和 ceo_comment。")
    return state


def manager_decision_router(state: MultiLevelApprovalState) -> str:
    """经理审批决策"""
    if state.manager_approved:
        print(f"✅ 经理批准: {state.manager_comment}")
        return "wait_director"
    else:
        print(f"❌ 经理拒绝: {state.manager_comment}")
        state.final_status = "rejected"
        return "handle_rejection"


def director_decision_router(state: MultiLevelApprovalState) -> str:
    """总监审批决策"""
    if state.director_approved:
        print(f"✅ 总监批准: {state.director_comment}")
        return "wait_ceo"
    else:
        print(f"❌ 总监拒绝: {state.director_comment}")
        state.final_status = "rejected"
        return "handle_rejection"


def ceo_decision_router(state: MultiLevelApprovalState) -> str:
    """CEO审批决策"""
    if state.ceo_approved:
        print(f"✅ CEO批准: {state.ceo_comment}")
        state.final_status = "approved"
        return "handle_approval"
    else:
        print(f"❌ CEO拒绝: {state.ceo_comment}")
        state.final_status = "rejected"
        return "handle_rejection"


def handle_multi_approval_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """处理三级审批全部通过"""
    print(f"\n🎉 三级审批全部通过!")
    print(f"   经理意见: {state.manager_comment}")
    print(f"   总监意见: {state.director_comment}")
    print(f"   CEO意见: {state.ceo_comment}")
    print(f"   开始执行请求...")
    return state


def handle_multi_rejection_node(state: MultiLevelApprovalState) -> MultiLevelApprovalState:
    """处理审批拒绝"""
    print(f"\n❌ 审批流程终止（在某一级被拒绝）")
    return state


def create_multi_level_approval_workflow():
    """创建三级审批工作流

    流程: submit -> manager -> director -> ceo -> process -> END
          任何一级拒绝都会终止流程
    """
    workflow = StateGraph(MultiLevelApprovalState)

    # 添加节点
    workflow.add_node("submit", submit_multi_request_node)
    workflow.add_node("wait_manager", wait_manager_approval_node)
    workflow.add_node("wait_director", wait_director_approval_node)
    workflow.add_node("wait_ceo", wait_ceo_approval_node)
    workflow.add_node("handle_approval", handle_multi_approval_node)
    workflow.add_node("handle_rejection", handle_multi_rejection_node)

    # 设置流程
    workflow.set_entry_point("submit")
    workflow.add_edge("submit", "wait_manager")

    # 经理审批分支
    workflow.add_conditional_edges(
        "wait_manager",
        manager_decision_router,
        {
            "wait_director": "wait_director",
            "handle_rejection": "handle_rejection"
        }
    )

    # 总监审批分支
    workflow.add_conditional_edges(
        "wait_director",
        director_decision_router,
        {
            "wait_ceo": "wait_ceo",
            "handle_rejection": "handle_rejection"
        }
    )

    # CEO审批分支
    workflow.add_conditional_edges(
        "wait_ceo",
        ceo_decision_router,
        {
            "handle_approval": "handle_approval",
            "handle_rejection": "handle_rejection"
        }
    )

    workflow.add_edge("handle_approval", END)
    workflow.add_edge("handle_rejection", END)

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)

    return app


# ========== 演示场景 ==========

def demo_basic_approval():
    """演示1: 基本单级审批流程"""
    print("\n" + "="*60)
    print("演示1: 基本单级审批流程（审批通过）")
    print("="*60)

    app = create_approval_workflow()

    # 初始化请求
    initial_state = ApprovalState(
        request_content="购买新服务器",
        request_amount=50000.0,
        approver="张经理"
    )

    config = {"configurable": {"thread_id": "approval_1"}}

    # 第1步: 启动工作流，会在审批节点暂停
    print("\n>>> 启动审批流程...")
    result = app.invoke(initial_state, config)

    # 第2步: 获取当前状态
    current_state = app.get_state(config)
    print(f"\n📊 当前状态: {current_state.values}")

    # 第3步: 模拟人工审批（通过）
    print("\n>>> 张经理审批中...")
    print("   审批结果: 通过")
    app.update_state(
        config,
        {
            "approval_status": "approved",
            "approval_comment": "采购合理，同意购买"
        }
    )

    # 第4步: 恢复执行
    print("\n>>> 恢复工作流执行...")
    final_result = app.invoke(None, config)
    print(f"\n✨ 最终状态: {final_result}")


def demo_rejection_flow():
    """演示2: 审批拒绝流程"""
    print("\n" + "="*60)
    print("演示2: 审批拒绝流程")
    print("="*60)

    app = create_approval_workflow()

    initial_state = ApprovalState(
        request_content="团建活动经费",
        request_amount=100000.0,
        approver="李总监"
    )

    config = {"configurable": {"thread_id": "approval_2"}}

    # 启动并暂停
    app.invoke(initial_state, config)

    # 模拟拒绝
    print("\n>>> 李总监审批中...")
    print("   审批结果: 拒绝")
    app.update_state(
        config,
        {
            "approval_status": "rejected",
            "approval_comment": "预算超支，建议削减30%后重新申请"
        }
    )

    # 恢复执行
    final_result = app.invoke(None, config)


def demo_multi_level_approval():
    """演示3: 三级审批流程（全部通过）"""
    print("\n" + "="*60)
    print("演示3: 三级审批流程（经理→总监→CEO）")
    print("="*60)

    app = create_multi_level_approval_workflow()

    initial_state = MultiLevelApprovalState(
        request_content="开设新分公司",
        request_amount=5000000.0
    )

    config = {"configurable": {"thread_id": "multi_approval_1"}}

    # 启动流程，在经理审批处暂停
    print("\n>>> 启动三级审批流程...")
    app.invoke(initial_state, config)

    # 经理审批
    print("\n>>> [第1级] 经理审批...")
    app.update_state(
        config,
        {
            "manager_approved": True,
            "manager_comment": "市场调研充分，支持开设分公司"
        }
    )

    # 继续到总监审批，会再次暂停
    app.invoke(None, config)

    # 总监审批
    print("\n>>> [第2级] 总监审批...")
    app.update_state(
        config,
        {
            "director_approved": True,
            "director_comment": "财务预算合理，同意推进"
        }
    )

    # 继续到CEO审批
    app.invoke(None, config)

    # CEO审批
    print("\n>>> [第3级] CEO审批...")
    app.update_state(
        config,
        {
            "ceo_approved": True,
            "ceo_comment": "战略方向正确，批准执行"
        }
    )

    # 最终执行
    final_result = app.invoke(None, config)


def demo_multi_level_rejection():
    """演示4: 三级审批在第二级被拒绝"""
    print("\n" + "="*60)
    print("演示4: 三级审批（在总监级别被拒绝）")
    print("="*60)

    app = create_multi_level_approval_workflow()

    initial_state = MultiLevelApprovalState(
        request_content="收购竞争对手公司",
        request_amount=50000000.0
    )

    config = {"configurable": {"thread_id": "multi_approval_2"}}

    # 经理审批（通过）
    app.invoke(initial_state, config)

    print("\n>>> [第1级] 经理审批...")
    app.update_state(
        config,
        {
            "manager_approved": True,
            "manager_comment": "可以考虑收购"
        }
    )

    # 总监审批（拒绝）
    app.invoke(None, config)

    print("\n>>> [第2级] 总监审批...")
    app.update_state(
        config,
        {
            "director_approved": False,
            "director_comment": "估值过高，风险太大，不建议收购"
        }
    )

    # 流程终止
    final_result = app.invoke(None, config)


def demo_stream_mode():
    """演示5: 使用 stream 模式观察暂停过程"""
    print("\n" + "="*60)
    print("演示5: Stream模式观察工作流暂停")
    print("="*60)

    app = create_approval_workflow()

    initial_state = ApprovalState(
        request_content="升级办公设备",
        request_amount=30000.0,
        approver="王主管"
    )

    config = {"configurable": {"thread_id": "approval_stream"}}

    print("\n>>> 使用 stream 模式启动工作流...")

    # 使用 stream 观察每个节点的执行
    for event in app.stream(initial_state, config):
        print(f"\n📡 Stream事件: {event}")

        # 检查是否遇到暂停
        if "__interrupt__" in str(event):
            print("   ⏸️  检测到工作流暂停")
            break

    # 提供审批意见
    print("\n>>> 王主管审批通过...")
    app.update_state(
        config,
        {
            "approval_status": "approved",
            "approval_comment": "设备老化严重，同意升级"
        }
    )

    # 继续观察后续执行
    print("\n>>> 继续执行...")
    for event in app.stream(None, config):
        print(f"\n📡 Stream事件: {event}")


def main():
    """运行所有演示场景"""
    print("\n🚀 LangGraph HITL (Human-in-the-Loop) 演示")
    print("本演示展示如何在工作流中实现人机协作和审批流程\n")

    # 运行所有演示
    demo_basic_approval()
    demo_rejection_flow()
    demo_multi_level_approval()
    demo_multi_level_rejection()
    demo_stream_mode()

    print("\n" + "="*60)
    print("✅ 所有演示完成!")
    print("="*60)
    print("\n💡 关键要点:")
    print("   1. NodeInterrupt 可以在任何节点暂停工作流")
    print("   2. update_state() 用于提供人工输入并恢复执行")
    print("   3. MemorySaver 保存暂停状态，支持跨会话恢复")
    print("   4. 条件分支可以处理审批通过/拒绝等不同场景")
    print("   5. stream() 方法可以实时观察工作流执行过程")
    print("   6. 多级审批通过多次暂停实现，灵活控制审批流程")


if __name__ == "__main__":
    main()
