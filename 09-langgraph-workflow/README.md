# 09 - LangGraph 工作流

本目录包含 3 个可运行示例，聚焦 LangGraph 的核心用法：状态管理、条件分支、循环与退出、嵌套子图以及工作流可视化。示例均不依赖 LLM 或工具调用，便于本地快速上手。

## 目录概览

- `01_langgraph_workflow_base.py`：基础工作流 + 条件分支路由；生成 `blog/flow_01.png`
- `02_langgraph_workflow_ cycle.py`：循环与退出条件（避免无限循环）；生成 `blog/flow_02.png`
- `03_langgraph_workflow_nested.py`：父图嵌套子图（子工作流）；生成 `blog/flow_03.png`
- `requirements.txt`：依赖说明

## 已实现的学习点

### 1. 基础工作流构建（01_langgraph_workflow_base.py）
- 使用 `StateGraph` 构建工作流，注册节点并设置入口：`set_entry_point`
- 定义共享状态 `BaseState(count)`，在节点中更新计数
- 使用 `add_edge` 串联节点；使用 `add_conditional_edges` 实现条件分支
- 通过 `route_node` 将分支路由到 `node_3_1` 或 `node_3_2`
- 编译并运行：`workflow.compile().invoke(initial_state)`

### 2. 循环与退出条件（02_langgraph_workflow_ cycle.py）
- 节点序列：`node_1 -> node_2 -> node_3 -> (条件) -> node_1 / node_4 -> END`
- 在 `node_3` 中根据计数设置 `finished=True`，通过路由函数返回 `node_1` 或收尾的 `node_4`
- 使用条件分支替代直接边，避免并发写入与重复路径导致的错误
- 展示如何在 LangGraph 中设计循环并保证退出条件，避免无限循环

### 3. 嵌套工作流（03_langgraph_workflow_nested.py）
- 将已编译的子图作为父图的一个节点加入：`workflow.add_node("subflow", create_subworkflow())`
- 父图与子图共享同一状态类型，子图执行后返回更新状态继续在父图中运行
- 展示复杂工作流的分层组织方式（父图-子图）

## 快速开始

### 环境配置
1. 安装依赖：`pip install -r requirements.txt`
2. 这三个示例不需要配置任何 API Key，即可本地运行

### 运行示例
```bash
# 基础工作流 + 条件分支
python 01_langgraph_workflow_base.py

# 循环与退出条件（文件名包含空格，建议加引号）
python "02_langgraph_workflow_ cycle.py"

# 父图嵌套子图
python 03_langgraph_workflow_nested.py
```

### 工作流可视化（生成图片）
- 运行各脚本后，会在 `blog/` 目录生成对应的 Mermaid PNG：
  - 基础示例：`blog/flow_01.png`
  - 循环示例：`blog/flow_02.png`
  - 嵌套示例：`blog/flow_03.png`
- 生成方法基于 `get_graph().draw_mermaid_png(...)`。该方法通常依赖在线服务渲染，如本地网络受限，可改为导出 Mermaid 源后使用 `mmdc`（mermaid-cli）离线渲染。

### 输出示例
- 运行结束后会在控制台打印形如：`最终计数: <数字>` 的结果，表示经过的节点数或状态计数。

## 学习目标

通过本目录的学习，您将能够：
- 使用 `StateGraph` 构建和编译工作流
- 在节点中读写共享状态并进行条件分支
- 设计含循环的流程并加入退出条件以避免死循环
- 将子工作流嵌入到父工作流实现分层结构
- 导出工作流的可视化图以便理解与分享

## 进阶思考

1. 如何基于实际业务设计更复杂的分支与合流？
2. 循环中的退出条件应该如何参数化与配置化？
3. 子图与父图之间共享状态的边界如何更好地设计？
4. 如何在持续集成环境中自动导出并上传工作流图？
5. 如需引入 LLM 或工具调用，工作流应如何扩展其节点处理与异常控制？
