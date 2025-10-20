#!/usr/bin/env python3
"""
Module 7: LangGraph Workflow Chatbot Demo
演示如何在LangChain中使用LangGraph构建高级工作流

本示例实现了一个具有多节点工作流的聊天机器人，包含以下功能：
1. 工作流状态管理
2. 条件分支和路由
3. 工具调用能力
4. 会话记忆功能
5. 错误处理和重试机制
"""
from dataclasses import dataclass
from langgraph.graph import StateGraph, END

# 定义状态类

@dataclass
class BaseState:
    """定义LangGraph工作流的状态类"""
    count: int = 0
    finished: bool = False

def node_1(state: BaseState) -> BaseState:
    print(f"节点1")
    state.count += 1
    return state

def node_2(state: BaseState) -> BaseState:
    print(f"节点2")
    state.count += 1
    return state

def node_3(state: BaseState) -> BaseState:
    print(f"节点3分叉")
    state.count += 1
    if (state.count>3):
        state.finished = True
    return state

def node_4(state: BaseState) -> BaseState:
    print(f"节点4")
    state.count += 1
    return state

MAX_LOOPS = 10

def route_node(state: BaseState) -> str:
    """根据状态路由到下一个节点，并添加退出条件避免无限循环"""
    if state.finished:
        return "node_4"
    return "node_1"

# 创建LangGraph工作流
def create_workflow():
    """创建LangGraph工作流"""
    # 初始化状态图
    workflow = StateGraph(BaseState)

    # 添加节点
    workflow.add_node("node_1",node_1)
    workflow.add_node("node_2",node_2)
    workflow.add_node("node_3",node_3)
    workflow.add_node("node_4",node_4)

    # 设置入口点和边连接
    workflow.set_entry_point("node_1")

    # 工作流顺序
    workflow.add_edge("node_1","node_2")
    workflow.add_edge("node_2","node_3")
    # 条件分支：node_3 根据路由选择进入 node_1 或最终 node_4
    workflow.add_conditional_edges(
        "node_3",
        route_node,
        {"node_1": "node_1", "node_4": "node_4"}
    )
    # 注意：不再添加直接边 node_3 -> node_4，避免与条件分支重复导致并发写入
    # 最终收尾后结束
    workflow.add_edge("node_4", END)

    # 编译图
    app = workflow.compile()

    # 生成工作流可视化图
    app.get_graph().draw_mermaid_png(output_file_path = 'blog/flow_02.png')
    return app


def main() -> None:
    # 创建工作流
    app = create_workflow()

    initial_state = BaseState()

    final_state = app.invoke(initial_state)

    # LangGraph 返回值可能是字典或状态对象，做兼容处理
    count_val = final_state.get("count") if isinstance(final_state, dict) else getattr(final_state, "count", None)
    
    print("最终计数:" + str(count_val))

if __name__ == "__main__":
    main()
