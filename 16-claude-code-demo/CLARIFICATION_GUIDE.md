# 反问机制使用指南

## 概述

本指南介绍如何使用 `11_claude_code_style_enhanced.py` 中新增的**反问机制**（Clarification Mechanism）。这个功能让 Agent 能够在需求不明确时主动向用户提问，从而提供更精准的研究结果。

## 核心功能

### 1. 智能需求检测
Agent 会自动分析用户需求，判断是否需要澄清：
- ✅ 需求模糊或有多种理解方式
- ✅ 缺少关键的范围、约束或偏好信息
- ✅ 有技术选型、优先级等决策点

### 2. 结构化提问
生成的问题包含：
- **问题内容**：具体、明确的问题
- **问题类型**：scope（范围）/ preference（偏好）/ constraint（约束）/ context（背景）
- **提问原因**：为什么需要这个问题
- **可选选项**：预设的选择（可选）
- **紧迫性**：high / medium / low

### 3. 交互式回答
支持两种回答方式：
- 从预设选项中选择（输入数字）
- 自定义文本回答
- 跳过问题（按回车）

## 数据模型

### ClarificationQuestion（澄清问题）
```python
{
    "question": "您希望重点关注哪个方面？",
    "reason": "需求过于宽泛，需要明确重点",
    "question_type": "scope",  # scope | preference | constraint | context
    "options": ["基础入门", "进阶实战", "最佳实践"]  # 可选
}
```

### ClarificationNeed（澄清需求）
```python
{
    "need_clarification": true,
    "questions": [ClarificationQuestion, ...],
    "reasoning": "为什么需要澄清的详细说明",
    "urgency": "high"  # high | medium | low
}
```

### ClarificationResponse（用户回答）
```python
{
    "answers": {
        "问题1": "回答1",
        "问题2": "回答2"
    },
    "timestamp": "2025-11-30T19:57:41.748147"
}
```

## 使用方法

### 方式 1：交互模式（推荐）

```bash
python3 11_claude_code_style_enhanced.py
```

程序会：
1. 提示您输入研究需求
2. Agent 自动检测是否需要澄清
3. 如果需要，显示问题并等待您回答
4. 基于您的反馈执行研究任务

**示例交互流程：**

```
请输入您的研究需求: 研究 AI

================================================================================
Agent 需要您的帮助来更好地理解需求
================================================================================

原因：需求 '研究 AI' 过于宽泛，AI 包含多个子领域
紧迫性：HIGH

问题 1/2 (scope):
  您想了解 AI 的哪个方向？
  → 原因：AI 领域非常广泛
  → 可选项：
    1. 机器学习基础
    2. 深度学习
    3. 自然语言处理
    4. 计算机视觉
    5. AI 应用开发

  请输入您的选择（1-5 或自定义文本）[回车跳过]: 3

问题 2/2 (context):
  您的技术背景是什么？
  → 原因：需要匹配合适难度的资料
  → 可选项：
    1. 编程新手
    2. 有编程基础
    3. 有 AI 经验

  请输入您的选择（1-3 或自定义文本）[回车跳过]: 2

✓ 感谢您的反馈！继续执行研究任务...
```

### 方式 2：批处理模式（禁用反问）

修改 `run_enhanced_demo()` 调用：

```python
if __name__ == "__main__":
    run_enhanced_demo(interactive=False)  # 禁用交互模式
```

或在代码中设置 `enable_clarification=False`：

```python
state = ClaudeCodeState(
    user_request="研究 Rust",
    enable_clarification=False,  # 禁用反问
)
```

### 方式 3：编程调用

```python
from langchain_learn.enhanced_module import (
    ClaudeCodeState,
    create_enhanced_workflow,
)

# 创建工作流
workflow = create_enhanced_workflow()

# 启用反问功能
state = ClaudeCodeState(
    user_request="研究区块链技术",
    enable_clarification=True,
    max_tokens_budget=40000,
    max_loops=6,
)

# 执行
final_state = workflow.invoke(state)

# 查看澄清过程
if final_state.get("clarification_responses"):
    for resp in final_state["clarification_responses"]:
        print(f"用户在 {resp.timestamp} 提供了反馈：")
        for q, a in resp.answers.items():
            print(f"  {q} -> {a}")
```

## 工作流说明

### 增强后的工作流

```
[用户输入]
    ↓
[detect_clarification] ← 检测是否需要澄清
    ↓ (需要澄清)
[ask_user] ← 向用户提问
    ↓
[plan] ← 基于澄清后的需求制定计划
    ↓
[pick] → [spawn] → [execute] → [reflect] → [assess_goal] → [budget_monitor]
    ↓ (循环或结束)
[final] ← 生成最终报告
```

### 关键节点

1. **detect_clarification_need_node**
   - 分析用户需求
   - 使用 LLM 判断是否需要澄清
   - 生成结构化的问题列表

2. **ask_user_node**
   - 向用户展示问题
   - 收集用户回答
   - 将澄清信息融入原始需求

3. **plan_research_node**
   - 基于（可能已澄清的）需求制定研究计划
   - 后续流程与原版相同

## 配置选项

### ClaudeCodeState 新增字段

```python
ClaudeCodeState(
    # ... 原有字段 ...

    # 澄清相关字段
    enable_clarification=True,  # 是否启用反问功能
    clarification_need=None,    # 澄清需求（由系统填充）
    clarification_responses=[],  # 用户回答历史（由系统填充）
)
```

### 自定义澄清检测 Prompt

如需调整澄清检测的策略，修改 [detect_clarification_need_node](11_claude_code_style_enhanced.py#L300-L378) 中的 prompt：

```python
prompt = f"""
你是 Lead Researcher，正在分析用户的研究需求。

用户需求：{state.user_request}

# 在这里调整检测策略
请判断这个需求是否需要向用户提问以澄清...
"""
```

## 最佳实践

### 1. 何时使用反问功能？

✅ **适合场景：**
- 研究型任务（需求开放）
- 多领域交叉主题
- 新用户不熟悉领域
- 需要个性化推荐

❌ **不适合场景：**
- 需求已经非常具体
- 批处理任务
- 快速原型验证
- 自动化流水线

### 2. 问题设计原则

- 每次提问 **1-3 个问题**（避免用户疲劳）
- 问题要 **具体、明确**
- 提供 **合理的选项**（但允许自定义）
- 说明 **为什么需要这个问题**

### 3. 紧迫性设置

- **high**：不澄清无法继续（如完全模糊的需求）
- **medium**：澄清后效果更好（如可以猜测但不确定）
- **low**：可选的补充信息（如锦上添花的优化）

## 测试

### 运行单元测试

```bash
# 测试数据模型和逻辑（无需 LangChain）
python3 test_clarification_simple.py

# 完整功能测试（需要 LLM 配置）
python3 test_clarification.py
```

### 测试用例

```python
# 测试用例 1：极度模糊
"研究 AI"  # 应触发多个问题

# 测试用例 2：适度模糊
"研究 Rust 学习路径"  # 可能触发 1-2 个问题

# 测试用例 3：非常明确
"研究 2025 年学习 Rust 的最佳路径，重点 Web 开发，我是初学者"
# 应跳过澄清
```

## 故障排查

### 问题 1：澄清功能未触发

**可能原因：**
- `enable_clarification=False`
- 需求已经足够明确
- LLM 判断不需要澄清

**解决方法：**
- 检查 `state.enable_clarification`
- 查看日志：`state.research_logs`
- 使用更模糊的测试需求

### 问题 2：JSON 解析失败

**可能原因：**
- LLM 输出格式不正确
- 网络超时

**解决方法：**
- 检查 `parse_json_with_retry` 日志
- 增加重试次数
- 切换更稳定的模型

### 问题 3：交互无响应

**可能原因：**
- 输入流被阻塞
- 在非交互环境运行

**解决方法：**
- 确保在终端环境运行
- 使用 `interactive=False` 批处理模式

## 扩展开发

### 添加新的问题类型

在 `ClarificationQuestion` 中扩展 `question_type`：

```python
question_type: Literal["scope", "preference", "constraint", "context", "custom"]
```

### 实现智能选项生成

修改 `detect_clarification_need_node` 让 LLM 动态生成选项：

```python
prompt = f"""
...
请基于以下候选项生成选项（最多 5 个）：
{get_domain_specific_options(state.user_request)}
...
"""
```

### 集成外部知识库

在澄清检测前先查询知识库：

```python
def detect_clarification_need_node(state):
    # 查询历史类似需求
    similar_requests = query_knowledge_base(state.user_request)

    if similar_requests:
        # 基于历史优化检测策略
        ...
```

## 性能考虑

- **Token 消耗**：澄清检测会额外消耗约 500-1000 tokens
- **延迟**：用户回答时间不计入系统延迟
- **重试机制**：JSON 解析失败会自动重试（最多 3 次）

## 版本历史

- **v1.0** (2025-11-30): 初始版本
  - 基础反问机制
  - 结构化问题和回答
  - 交互式命令行界面

## 相关资源

- 主程序：[11_claude_code_style_enhanced.py](11_claude_code_style_enhanced.py)
- 数据模型测试：[test_clarification_simple.py](test_clarification_simple.py)
- LangGraph 文档：https://langchain-ai.github.io/langgraph/

## 贡献

欢迎提交 Issue 或 Pull Request 来改进反问机制！

重点改进方向：
- [ ] 支持多轮澄清
- [ ] 澄清历史持久化
- [ ] 智能选项推荐
- [ ] 图形界面支持
