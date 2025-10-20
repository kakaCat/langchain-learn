#!/usr/bin/env python3
"""
LangGraph 嵌套示例：父图中包含一个子图（子工作流）
- 父图节点：node_1 -> subflow -> node_4 -> END
- 子图节点：node_2 -> (条件) -> node_3_1 / node_3_2 -> END
- 状态：BaseState(count)

本示例展示如何将已编译的子图作为父图的一个节点使用。
父图与子图共享同一个状态类型，以便子图执行后将更新的状态传回父图继续执行。
"""
from dataclasses import dataclass
from langgraph.graph import StateGraph, END

@dataclass
class BaseState:
    """工作流共享状态"""
    count: int = 0

# ===== 子图节点 =====
def node_2(state: BaseState) -> BaseState:
    print("子图-节点2")
    state.count += 1
    return state

def node_3_1(state: BaseState) -> BaseState:
    print("子图-节点3_1")
    state.count += 1
    return state

def node_3_2(state: BaseState) -> BaseState:
    print("子图-节点3_2")
    state.count += 1
    return state

# 条件路由：根据计数奇偶选择不同分支
# 返回值需与 add_conditional_edges 显式映射的键一致

def route_node(state: BaseState) -> str:
    """根据计数奇偶路由到不同子图节点"""
    return "node_3_1" if (state.count % 2 == 0) else "node_3_2"


def create_subworkflow():
    """创建并编译子图：node_2 -> 条件 -> node_3_1 / node_3_2 -> END"""
    sub = StateGraph(BaseState)
    sub.add_node("node_2", node_2)
    sub.add_node("node_3_1", node_3_1)
    sub.add_node("node_3_2", node_3_2)

    sub.set_entry_point("node_2")
    sub.add_conditional_edges(
        "node_2",
        route_node,
        {"node_3_1": "node_3_1", "node_3_2": "node_3_2"}
    )
    sub.add_edge("node_3_1", END)
    sub.add_edge("node_3_2", END)

    return sub.compile()

# ===== 父图节点 =====
def node_1(state: BaseState) -> BaseState:
    print("父图-节点1")
    state.count += 1
    return state

def node_4(state: BaseState) -> BaseState:
    print("父图-节点4")
    state.count += 1
    return state


def create_workflow():
    """创建父图，并将已编译的子图作为一个节点加入"""
    workflow = StateGraph(BaseState)

    workflow.add_node("node_1", node_1)
    workflow.add_node("subflow", create_subworkflow())  # 嵌套子图
    workflow.add_node("node_4", node_4)

    workflow.set_entry_point("node_1")
    workflow.add_edge("node_1", "subflow")
    workflow.add_edge("subflow", "node_4")
    workflow.add_edge("node_4", END)

    app = workflow.compile()

    # 生成工作流可视化图（如无法联网，可改用 mermaid-cli 离线渲染）
    app.get_graph().draw_mermaid_png(output_file_path = 'blog/flow_03.png')
    return app


def main() -> None:
    app = create_workflow()
    initial_state = BaseState()
    final_state = app.invoke(initial_state)

    # 兼容状态对象或字典形式
    count_val = final_state.get("count") if isinstance(final_state, dict) else getattr(final_state, "count", None)
    print("最终计数:" + str(count_val))


if __name__ == "__main__":
    main()