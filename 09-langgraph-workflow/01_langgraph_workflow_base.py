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
    count: int = 0  # 记录节点执行次数

def node_1(state: BaseState) -> BaseState:
    """第1节点：打印日志并递增计数"""
    print(f"节点1")
    state.count += 1
    return state

def node_2(state: BaseState) -> BaseState:
    """第2节点：进入分叉并递增计数"""
    print(f"节点2分叉")
    state.count += 1
    return state

def node_3_1(state: BaseState) -> BaseState:
    """分支 3_1：递增计数"""
    print(f"节点3_1")
    state.count += 1
    return state

def node_3_2(state: BaseState) -> BaseState:
    """分支 3_2：递增计数"""
    print(f"节点3_2")
    state.count += 1
    return state

def node_4(state: BaseState) -> BaseState:
    """第4节点：结束前递增计数"""
    print(f"节点4")
    state.count += 1
    return state

def route_node(state: BaseState) -> str:
    """根据状态路由到下一个节点（奇偶分支）
    - 偶数：返回 'node_3_1'
    - 奇数：返回 'node_3_2'
    返回值必须与 add_conditional_edges 的映射键保持一致
    """
    # 与 add_conditional_edges 映射键保持一致
    # 简单路由逻辑：根据计数奇偶路由不同分支
    return "node_3_1" if (state.count % 2 == 0) else "node_3_2"

# 创建LangGraph工作流
def create_workflow():
    """创建 LangGraph 基础工作流：
    node_1 -> node_2 -> (node_3_1 | node_3_2) -> node_4 -> END
    """
    # 初始化状态图并绑定状态类型
    workflow = StateGraph(BaseState)

    # 注册所有节点名称与处理函数
    workflow.add_node("node_1",node_1)
    workflow.add_node("node_2",node_2)
    workflow.add_node("node_3_1",node_3_1)
    workflow.add_node("node_3_2",node_3_2)
    workflow.add_node("node_4",node_4)

    # 设置入口点与边连接（顺序 + 条件分支）
    workflow.set_entry_point("node_1")

    # 工作流顺序：1 -> 2
    workflow.add_edge("node_1","node_2")

    # 条件分支：
    # - route_node 返回 'node_3_1' 或 'node_3_2'
    # - 显式映射确保导出图中清晰展示两个分支，并在返回非法值时提前报错
    workflow.add_conditional_edges(
        "node_2",
        route_node,
        {"node_3_1": "node_3_1", "node_3_2": "node_3_2"}
    )

    # 分支收敛到节点4并结束
    workflow.add_edge("node_3_1","node_4")
    workflow.add_edge("node_3_2","node_4")
    workflow.add_edge("node_4",END)

    # 编译图为可执行应用
    app = workflow.compile()

    # 导出工作流可视化图（Mermaid PNG）。该方法默认请求在线服务：
    # - 如本地网络受限，可改为保存 Mermaid 源并离线渲染
    # - 如需保存到脚本同目录，可改为：os.path.join(os.path.dirname(__file__), 'flow_01.png')
    app.get_graph().draw_mermaid_png(output_file_path = 'blog/flow_01.png')
    return app


def main() -> None:
    """运行工作流并输出最终计数"""
    # 1) 创建工作流应用
    app = create_workflow()

    # 2) 初始化状态（count=0）
    initial_state = BaseState()

    # 3) 执行工作流
    final_state = app.invoke(initial_state)

    # 4) 兼容返回类型（dict 或状态对象）
    count_val = final_state.get("count") if isinstance(final_state, dict) else getattr(final_state, "count", None)
    
    # 输出最终计数（即经过的节点数）
    print("最终计数:" + str(count_val))

if __name__ == "__main__":
    main()
