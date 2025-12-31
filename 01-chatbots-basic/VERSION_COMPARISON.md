# LangChain 1.0 版本对比演示

本目录包含演示 LangChain 1.0 (Modern) 与旧版 (Legacy) 差异的示例代码。

## 核心进化：从“拼装库”到“Agent Runtime”

LangChain 1.0 带来了从理念到工程体系的全面重构，核心是将 Agent 运行时全面迁移至 **LangGraph**。

### 1. 架构定位变化
*   **Legacy (0.x)**: 链式拼装 (Chain)，基于 `AgentExecutor`，状态隐式管理，难以精细控制。
*   **Modern (1.0+)**: Agent 生态，基于 **LangGraph** 运行时。
    *   **LangChain**: 提供统一抽象 + 快速搭建 (`create_agent`)，适合快速 PoC。
    *   **LangGraph**: 提供底层图式编排、状态管理、检查点、并发控制，适合生产级精细控制。

### 2. 三大里程碑能力
1.  **统一入口 (`create_agent`)**: 基于 LangGraph 的开箱即用 API，具备循环调用、工具执行等能力。
2.  **标准化内容块 (Standard Content Blocks)**: 统一不同模型厂商的输出格式，解决兼容性问题。
3.  **中间件 (Middleware)**: 围绕模型调用的拦截器，支持 Human-in-the-loop、消息修剪/汇总、动态路由等。

---

## 演示文件说明

### `06_version_comparison_demo.py`
这是一个对比演示脚本，展示了 LangChain 1.0+ (Modern) 的推荐实现方式——**使用 LangGraph 进行精细控制**。

1.  **Modern Approach (1.0+ / LangGraph)**
    *   使用 `StateGraph` 手动构建图结构 (Best Practice for Production)。
    *   使用 `MemorySaver` (Checkpointer) 进行状态持久化。
    *   特点：
        *   **显式控制流 (Explicit Control Flow)**：通过 `add_node` 和 `add_edge` 清晰定义执行逻辑。
        *   **显式状态 (State)**：所有上下文都保存在 `messages` 列表中，透明可见。
        *   **线程管理 (Thread)**：通过 `thread_id` 原生支持多用户/多会话。
        *   **流式优先 (Streaming First)**：原生支持事件流。

## 如何运行

确保已安装依赖：
```bash
pip install -r requirements.txt
```

运行脚本：
```bash
python 06_version_comparison_demo.py
```

## 关键代码差异

| 特性 | Legacy (旧版) | Modern (1.0+) |
| :--- | :--- | :--- |
| **核心类** | `ConversationChain` | `create_agent` / `StateGraph` |
| **模型初始化** | `ChatOpenAI` (具体类) | `init_chat_model` (统一入口) |
| **内存** | `ConversationBufferMemory` | `MemorySaver` (Checkpointer) |
| **状态传递** | 隐式 (在 Memory 对象中) | 显式 (通过 `State` 字典传递) |
| **会话ID** | 需要手动 hack Memory | 原生支持 `config={"configurable": {"thread_id": "..."}}` |
| **流式** | 回调函数 (`StreamingStdOutCallbackHandler`) | `.stream()` 方法返回生成器 |
