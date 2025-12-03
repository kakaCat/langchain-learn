# Claude Code Style Enhanced - 使用说明

## 概述

这是一个增强版的 Claude Code 风格分层 Agent 系统，相比原版增加了以下核心功能：

### 主要增强点

1. **工具注册系统** - 真实工具集成（Web 搜索、Python REPL、文件操作）
2. **结构化输出** - 使用 Pydantic 模型 + 重试机制保证数据质量
3. **智能终止条件** - 目标达成评估 + Token 预算监控
4. **优化 Prompt** - 添加 few-shot examples 提升输出质量
5. **结构化 Citation** - 可追溯的引用系统（来源类型、时间戳、摘录）
6. **增强审核机制** - Lead Researcher 审核包含质量评分

## 架构对比

### 原版 vs 增强版

| 特性 | 原版 | 增强版 |
|------|------|--------|
| 工具调用 | ❌ 无，仅模拟 | ✅ 真实工具（DuckDuckGo、Python REPL） |
| 输出解析 | 简单 try-catch | ✅ Pydantic 验证 + 3次重试 |
| 循环终止 | 固定次数 (MAX_LOOPS=6) | ✅ 目标达成评估 + 预算监控 |
| Prompt 质量 | 基础提示 | ✅ Few-shot examples |
| Citation | 自由文本 | ✅ 结构化（类型、来源、时间戳） |
| 置信度追踪 | ❌ 无 | ✅ 每个结果包含 0-1 置信度 |
| 并行执行 | ❌ 不支持 | ⚠️ 框架支持（需进一步实现） |

## 安装依赖

```bash
cd 10-agent-examples

# 安装核心依赖
pip install -r requirements.txt

# 安装增强版额外依赖
pip install duckduckgo-search langchain-experimental
```

## 配置环境变量

创建 `.env` 文件：

```bash
# LLM 提供商选择
LLM_PROVIDER=openai  # 或 ollama

# OpenAI 配置（使用 OpenAI 时）
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=1500

# Ollama 配置（使用本地模型时）
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

## 快速开始

```bash
python 11_claude_code_style_enhanced.py
```

### 自定义任务

编辑 `run_enhanced_demo()` 函数中的 `tasks` 列表：

```python
tasks = [
    {
        "request": "你的研究问题",
        "budget": 40000,      # Token 预算
        "max_loops": 6,       # 最大循环次数
    },
]
```

## 核心组件详解

### 1. 工具注册系统 (ToolRegistry)

```python
TOOL_REGISTRY = ToolRegistry()

# 获取特定类型 Agent 的工具
tools = TOOL_REGISTRY.get_tools_for_agent("researcher")  # 返回 [web_search]
tools = TOOL_REGISTRY.get_tools_for_agent("analyst")     # 返回 [python_repl, file_read]
```

**支持的工具：**
- `web_search`: DuckDuckGo 搜索（无需 API Key）
- `python_repl`: 安全的 Python 代码执行环境
- `file_read`: 本地文件读取

### 2. 结构化数据模型

#### Citation（引用）

```python
Citation(
    source_type="web",  # web | file | calculation | reasoning
    source="https://example.com",
    timestamp="2025-01-15T10:30:00",
    snippet="关键信息摘录（可选）"
)
```

#### ResearchMemory（研究记忆）

```python
ResearchMemory(
    aspect="研究方面",
    summary="结果总结",
    citations=[Citation(...)],
    agent="AgentName",
    confidence=0.85,  # 0-1，表示结果可信度
    timestamp="2025-01-15T10:30:00"
)
```

### 3. 智能终止机制

系统通过两个节点控制循环：

#### assess_goal_node（目标评估）

```python
GoalAssessment(
    goal_achieved=True,        # 是否达成目标
    completeness=0.92,         # 完整度 0-1
    missing_aspects=[],        # 缺失的研究方面
    reasoning="所有核心问题已解答，引用充分"
)
```

**达成标准：**
- `completeness >= 0.85` 且 `missing_aspects` 为空
- 或者 LLM 判断已有足够信息回答用户需求

#### budget_monitor_node（预算监控）

```python
# 触发终止条件：
if state.total_tokens_used > state.max_tokens_budget:
    state.continue_research = False
if state.loop_count >= state.max_loops:
    state.continue_research = False
```

### 4. 错误处理和重试

所有 LLM 输出解析都使用 `parse_json_with_retry`：

```python
result = parse_json_with_retry(
    llm=llm,
    messages=[HumanMessage(content=prompt)],
    target_model=SubAgentResult,  # Pydantic 模型
    max_retries=3
)
```

**重试流程：**
1. 首次解析失败 → 向 LLM 反馈错误信息
2. LLM 重新生成符合格式的输出
3. 最多重试 3 次
4. 失败后使用回退方案（保证系统不崩溃）

### 5. 工作流程图

```
plan (创建研究计划)
  ↓
pick (选择下一个研究方面)
  ↓
spawn (动态派生子 Agent)
  ↓
execute (子 Agent 使用工具执行 ReAct 循环)
  ↓
reflect (Lead 审核结果，添加新方面)
  ↓
assess_goal (评估目标是否达成)
  ↓
budget_monitor (检查预算和循环次数)
  ↓
继续循环或进入 final
  ↓
final (生成最终报告)
```

## 输出示例

### 研究记忆输出

```
[RustLearningPathResearcher] Rust 基础学习资源
  总结: Rust 官方文档提供了最权威的学习路径...
  置信度: 0.88
  引用数: 3
    - [web] https://doc.rust-lang.org/book/
    - [web] https://www.rust-lang.org/learn
```

### 目标达成评估

```
目标达成: True
完整度: 92.00%
理由: 已覆盖学习路径、资源推荐、实战项目和常见陷阱，足以回答用户需求
```

### 预算使用情况

```
Token 使用: ~18500 / 40000
循环次数: 4 / 6
```

## 进阶定制

### 1. 添加自定义工具

```python
def custom_tool_func(input_str: str) -> str:
    # 你的工具逻辑
    return "结果"

TOOL_REGISTRY.tools["custom_tool"] = Tool(
    name="custom_tool",
    description="工具描述",
    func=custom_tool_func
)
```

### 2. 调整 Agent 类型映射

编辑 `ToolRegistry.get_tools_for_agent()`：

```python
tool_mapping = {
    "researcher": ["web_search", "custom_tool"],
    "analyst": ["python_repl", "file_read"],
    "generalist": ["web_search", "python_repl", "file_read", "custom_tool"],
}
```

### 3. 修改目标达成阈值

在 `assess_goal_node` 的 prompt 中修改：

```python
达成标准：
- completeness >= 0.90 且 missing_aspects 为空  # 从 0.85 改为 0.90
```

### 4. 调整预算

```python
state = ClaudeCodeState(
    user_request="...",
    max_tokens_budget=100000,  # 增加预算
    max_loops=10,              # 允许更多循环
)
```

## 已知限制和未来改进

### 当前限制

1. **工具调用简化**
   - 当前是"预执行一次工具 → LLM 分析结果"
   - 未实现真正的 LLM 驱动的多轮工具调用循环

2. **并行执行未实现**
   - 框架支持并行，但当前是串行执行
   - 需要改造 `execute` 节点为批处理模式

3. **工具安全性**
   - `python_repl` 使用 `langchain_experimental`，有安全风险
   - 生产环境应使用沙箱（如 Docker、E2B）

### 建议改进方向

1. **集成 LangGraph 的 create_react_agent**
   ```python
   from langgraph.prebuilt import create_react_agent

   agent = create_react_agent(llm, tools)
   result = agent.invoke({"messages": [...]})
   ```

2. **实现并行 SubAgent**
   ```python
   # 识别独立的研究方面
   independent_aspects = identify_parallel_aspects(state.backlog)
   # 并发执行
   results = await asyncio.gather(*[
       execute_subagent(aspect) for aspect in independent_aspects
   ])
   ```

3. **添加人工审核节点**
   ```python
   def human_review_node(state):
       print(f"当前结果: {state.memory[-1].summary}")
       feedback = input("是否接受？(y/n/修改建议): ")
       # 处理反馈...
   ```

4. **持久化存储**
   ```python
   # 将 state.memory 存储到数据库
   # 支持跨会话的记忆检索
   ```

## 故障排除

### Q: DuckDuckGo 搜索失败

```
[警告] Web 搜索工具注册失败: ...
```

**解决方案：**
```bash
pip install --upgrade duckduckgo-search
# 如果仍失败，检查网络连接或使用代理
```

### Q: JSON 解析总是失败

**可能原因：**
- 模型能力不足（如小于 7B 的本地模型）
- Prompt 过于复杂

**解决方案：**
1. 使用更强的模型（如 GPT-4、Claude）
2. 简化 Pydantic 模型结构
3. 增加 few-shot examples

### Q: Token 预算快速耗尽

**解决方案：**
1. 减少 `max_loops`
2. 在 `spawn_subagent_node` 中限制 `memory_context` 数量（已设为最近 3 条）
3. 使用更小的模型（如 `gpt-4o-mini`）

## 许可证

MIT

## 贡献

欢迎提交 Issue 和 Pull Request！

主要改进方向：
- 更多工具集成（数据库查询、API 调用等）
- 真正的并行执行
- 可视化界面（展示 Agent 执行流程）
- 成本优化（智能缓存、结果复用）
