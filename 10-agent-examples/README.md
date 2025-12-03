# 08 - Agent 示例集合

本目录包含基于 LangGraph 的各种 Agent 模式实现示例，涵盖从基础推理到复杂多步推理的完整 Agent 技术栈。

## ✅ 已实现示例

### 1. 基础推理模式
- **文件**: `01_react_demo.py`
- **内容**: 推理-行动-观察的经典循环，无记忆的轻量级 Agent 实现
- **示例**: REACT 模式，支持工具调用与状态管理

### 2. Plan-and-Solve 模式
- **文件**: `02_plan_and_solve_demo.py`
- **内容**: 先规划后执行的决策流程，结构化问题分解
- **示例**: 演示规划、执行、观察的工作流

### 3. Reason without Observation 模式
- **文件**: `03_reason_without_observation.py`
- **内容**: 纯推理步骤，无观察步骤，内部推理优化
- **示例**: 适用于不需要外部工具调用的场景

### 4. 编译与执行模式
- **文件**: `04_llm_compiler_demo.py`
- **内容**: 将用户任务编译为可执行指令序列，指令调度与执行管理
- **示例**: LLMCompiler 模式，支持结果验证与反馈机制

### 5. 反思与优化模式
- **文件**: `05_basic_reflection_demo.py`
- **内容**: 简单的反思和重试机制，Agent 的自我改进能力
- **示例**: 基础反思模式，支持错误检测与纠正

### 6. Reflexion 模式
- **文件**: `06_reflexion_demo.py`
- **内容**: 复杂的反思和记忆机制，基于历史经验的决策优化
- **示例**: Reflexion 模式，支持长期学习与适应

### 7. 搜索与发现模式
- **文件**: `07_language_agent_tree_search_demo.py`
- **内容**: 多分支搜索和评估，Beam Search 算法实现
- **示例**: 语言代理树搜索 (LATS)，包含 Propose、Evaluate、Prune、Goal Check、Finalize 步骤

### 8. Self-Discover 模式
- **文件**: `08_self_discover_demo.py`
- **内容**: 自发现技能与成功标准，自主问题解决能力
- **示例**: Self-Discover 模式，包含 Discover、Execute、Evaluate、Revise、Finalize 多阶段工作流

### 9. 结构化推理模式
- **文件**: `09_storm.py`
- **内容**: 结构化写作与大纲生成，思考-观察-反思的多步推理
- **示例**: STORM 模式，支持多阶段写作工作流

### 10. 任务管理模式
- **文件**: `10_todolist_agent_demo.py`
- **内容**: 任务分解与状态管理，动态任务调度
- **示例**: ToDoList Agent 模式，支持 ToDoList 的创建、查询、更新和统计操作

### 11. Claude Code 式层级协作
- **文件**: `11_claude_code_style_demo.py`
- **内容**: Lead Researcher 动态派生子 Agent，带记忆与引用的研究循环
- **示例**: 贴合 Claude Code 流程图的主从协作演示，包含子 Agent ReAct 与反思、Citation Agent 汇总

### 12. Human-in-the-Loop（人机协同）
- **文件**: `11_human_in_the_loop_demo.py`
- **内容**: 工作流中断机制，Agent 执行过程中请求人类输入和反馈
- **示例**: 使用 LangGraph 的 interrupt_before 和 MemorySaver 实现断点交互
- **适用场景**: 人工审核、不确定性决策、需要授权的操作

### 13. 智能澄清 Agent
- **文件**: `12_clarification_agent_demo.py`
- **内容**: 主动检测需求模糊度，生成结构化澄清问题，基于反馈调整策略
- **示例**: Proactive Questioning + Structured Question Generation
- **技术特点**:
  - 自动需求模糊度检测
  - 问题类型分类（scope/preference/constraint/context）
  - 紧迫性评估（high/medium/low）
  - 自适应执行策略

### 14. 多轮澄清对话
- **文件**: `13_multi_round_clarification_demo.py`
- **内容**: 迭代式多轮对话澄清，上下文感知问题生成，动态停止条件
- **示例**: Multi-Round Clarification + Adaptive Questioning
- **技术特点**:
  - 基于历史对话生成问题
  - 自动评估需求完整度
  - 智能停止机制（最多 3 轮）
  - 需求整合和总结

## 🔮 进阶学习点

### 架构设计模式
- **状态管理与持久化**
- **工作流编排与调度**
- **错误处理与重试机制**
- **性能优化与缓存策略**

### 算法与优化
- **搜索算法实现**
- **推理路径优化**
- **记忆压缩与检索**
- **多模态推理集成**

### 生产环境考虑
- **可扩展性与并发**
- **监控与日志记录**
- **安全与权限控制**
- **部署与运维最佳实践**

## 🚀 快速开始

### 环境配置
1. 复制环境配置：`cp ../07-langgraph-workflow/.env .`
2. 安装依赖：`pip install -r ../07-langgraph-workflow/requirements.txt`
3. 配置 OpenAI API 密钥和其他环境变量

### 运行示例
```bash
# 基础推理模式
python 01_react_demo.py

# 复杂搜索模式
python 07_language_agent_tree_search_demo.py

# 自发现模式
python 08_self_discover_demo.py

# 人机协同模式（交互式）
python 11_human_in_the_loop_demo.py

# 人机协同模式（自动演示）
python 11_human_in_the_loop_demo.py --auto

# 智能澄清 Agent
python 12_clarification_agent_demo.py

# 多轮澄清对话
python 13_multi_round_clarification_demo.py
```

## 🆕 反问机制（Clarification Mechanism）

### 核心概念

**反问机制**（也称为 Human-in-the-Loop 或 Interactive Clarification）是指 Agent 在执行过程中主动向用户提问以澄清需求的能力。

### 三种实现模式

#### 1. **Human-in-the-Loop（基础人机协同）**
- **实现**: 使用 LangGraph 的 `interrupt_before` 机制
- **特点**: 在预定节点暂停，等待人类输入
- **适用**: 需要人工审核、授权的场景

#### 2. **Intelligent Clarification（智能澄清）**
- **实现**: LLM 自动检测需求模糊度 + 结构化问题生成
- **特点**: 主动提问，问题分类（scope/preference/constraint/context）
- **适用**: 需求分析、个性化推荐

#### 3. **Multi-Round Dialogue（多轮对话）**
- **实现**: 迭代式澄清 + 上下文感知 + 智能停止
- **特点**: 基于历史对话深入挖掘，自动评估完整度
- **适用**: 复杂需求分析、咨询服务

### 技术对比

| 特性 | Human-in-the-Loop | Intelligent Clarification | Multi-Round Dialogue |
|------|-------------------|---------------------------|----------------------|
| 文件 | 11_*.py | 12_*.py | 13_*.py |
| 自动检测 | ❌ | ✅ | ✅ |
| 结构化问题 | ❌ | ✅ | ✅ |
| 多轮迭代 | ❌ | ❌ | ✅ |
| 智能停止 | ❌ | ❌ | ✅ |
| 实现复杂度 | 简单 | 中等 | 复杂 |

### 使用建议

- **初学者**: 从 `11_human_in_the_loop_demo.py` 开始
- **实战应用**: 使用 `12_clarification_agent_demo.py`
- **高级场景**: 探索 `13_multi_round_clarification_demo.py`

## 🎯 学习目标

通过本模块的学习，您将能够：
- 理解不同 Agent 架构的设计原理和适用场景
- 掌握基于 LangGraph 的复杂工作流构建方法
- 实现从简单推理到复杂多步推理的完整 Agent 系统
- 应用搜索、反思、编译等高级 Agent 技术
- 设计可扩展、可维护的 Agent 架构
- **新增**: 实现智能反问和人机协同机制

## 💡 技术特点

- **统一架构**：所有示例都基于 LangGraph 构建，保持一致的架构模式
- **多样化模式**：涵盖推理、搜索、反思、编译等多种 Agent 决策模式
- **状态管理**：完整的 StateGraph 状态管理和工作流控制
- **生态系统集成**：与 LangChain 生态系统深度集成
- **生产就绪**：包含错误处理、重试机制等生产环境考虑

## 🔗 与模块 7 的关系

这些 Agent 示例是模块 7 (LangGraph 工作流) 的具体应用，展示了如何在实际场景中使用 LangGraph 构建复杂的 Agent 系统。每个示例都体现了不同的工作流设计模式和状态管理策略。

## 🤔 进阶思考

1. **架构演进**：如何从简单 REACT 模式演进到复杂的多步推理系统？
2. **性能优化**：在大规模应用中如何优化 Agent 的响应时间和资源消耗？
3. **错误恢复**：如何设计健壮的错误处理和恢复机制？
4. **可观测性**：如何监控和调试复杂的 Agent 工作流？
5. **安全考虑**：在 Agent 系统中如何确保数据安全和权限控制？
6. **用户体验**：如何设计直观的交互界面和反馈机制？
