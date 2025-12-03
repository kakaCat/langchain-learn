# 更新日志

## [v1.1.0] - 2025-11-30

### 新增功能：反问机制 (Clarification Mechanism)

#### 概述
为 Claude Code Style Enhanced Demo 添加了智能反问功能，使 Agent 能够在需求不明确时主动向用户提问。

#### 核心改进

1. **新增数据模型**
   - `ClarificationQuestion`: 结构化的澄清问题
   - `ClarificationNeed`: 澄清需求判断
   - `ClarificationResponse`: 用户回答记录

2. **新增工作流节点**
   - `detect_clarification_need_node`: 智能检测是否需要澄清
   - `ask_user_node`: 交互式向用户提问并收集回答

3. **增强状态管理**
   - `ClaudeCodeState` 新增字段：
     - `clarification_need`: 存储澄清需求
     - `clarification_responses`: 存储用户回答历史
     - `enable_clarification`: 控制是否启用反问功能

4. **优化工作流**
   - 新的入口点：先检测澄清需求
   - 条件分支：根据检测结果决定是否询问用户
   - 需求更新：将用户澄清融入原始需求

5. **支持两种模式**
   - **交互模式** (`interactive=True`): 启用反问功能
   - **批处理模式** (`interactive=False`): 传统自动执行

#### 技术亮点

- ✅ 结构化的问题类型（scope/preference/constraint/context）
- ✅ 支持预设选项和自定义回答
- ✅ 紧迫性级别（high/medium/low）
- ✅ 完整的错误处理和重试机制
- ✅ JSON 序列化和反序列化
- ✅ 时间戳追踪

#### 文件变更

**修改的文件：**
- `11_claude_code_style_enhanced.py`: 核心实现（+150 行）

**新增的文件：**
- `CLARIFICATION_GUIDE.md`: 完整使用指南
- `test_clarification_simple.py`: 数据模型单元测试
- `test_clarification.py`: 完整功能测试
- `CHANGELOG.md`: 本文件

#### 使用示例

**交互模式：**
```bash
python3 11_claude_code_style_enhanced.py
# 程序会提示输入需求，并在必要时提问
```

**编程调用：**
```python
state = ClaudeCodeState(
    user_request="研究 AI",
    enable_clarification=True,
)
final_state = workflow.invoke(state)
```

#### 测试覆盖

- ✅ 数据模型验证
- ✅ JSON 序列化/反序列化
- ✅ 工作流模拟
- ✅ 边界条件测试

运行测试：
```bash
python3 test_clarification_simple.py
```

#### 性能影响

- Token 额外消耗：约 500-1000 tokens/次（仅在需要澄清时）
- 延迟：用户输入时间不计入系统延迟
- 兼容性：完全向后兼容，可选启用

#### 已知限制

- 当前仅支持单轮澄清（未来可扩展多轮）
- 澄清历史不持久化（仅在内存中）
- 命令行界面（未来可支持 Web UI）

#### 升级指南

无需特殊操作，现有代码完全兼容：

```python
# 方式 1：保持原行为（禁用反问）
run_enhanced_demo(interactive=False)

# 方式 2：启用新功能
run_enhanced_demo(interactive=True)  # 默认
```

#### 贡献者

- 实现：Claude Code AI Assistant
- 设计：基于 Claude Code 官方最佳实践

#### 下一步计划

- [ ] 支持多轮澄清对话
- [ ] 澄清历史持久化（SQLite/JSON）
- [ ] Web UI 支持
- [ ] 智能选项推荐（基于历史数据）
- [ ] 澄清质量评估

---

## [v1.0.0] - 2025-10-17

### 初始版本

- ✅ 工具注册系统
- ✅ 结构化输出和错误重试
- ✅ 智能循环终止
- ✅ 优化的 Prompt
- ✅ 结构化 Citation 追踪
- ✅ 并行 SubAgent 执行
