# Claude Code 功能示例 - 缺失列表

## 📋 已有示例

### ✅ 核心架构
1. **11_claude_code_style_demo.py** - 基础层级协作
2. **11_claude_code_style_enhanced.py** - 增强版（带反问机制）
3. **11_claude_code_parallel.py** - 并行版

### ✅ 新增功能
4. **14_tool_usage_demo.py** - 工具使用示例 ✨ NEW

## 🎯 还需要补充的示例

### 1. **文件操作示例** (15_file_operations_demo.py)
**优先级：高**

模拟 Claude Code 的文件操作能力：
- Read（读取文件，支持偏移和限制）
- Write（写入文件，覆盖检查）
- Edit（精确编辑，基于字符串替换）
- Glob（模式匹配查找文件）

**核心功能：**
```python
# Read with offset/limit
def read_file_advanced(file_path, offset=0, limit=None):
    # 读取大文件时使用偏移
    pass

# Edit with exact replacement
def edit_file(file_path, old_string, new_string, replace_all=False):
    # 必须精确匹配，避免误修改
    pass

# Glob with recursive search
def glob_files(pattern, path="."):
    # 支持 **/*.py 等复杂模式
    pass
```

**示例任务：**
1. 读取长文件的特定部分
2. 精确替换代码片段
3. 批量查找和修改文件

---

### 2. **代码分析示例** (16_code_analysis_demo.py)
**优先级：高**

模拟 Claude Code 的代码理解能力：
- 代码审查（发现 bug、性能问题）
- 重构建议
- 依赖分析
- 测试生成

**核心功能：**
```python
class CodeAnalyzer:
    def analyze_quality(self, code: str) -> Dict:
        # 分析代码质量
        return {
            "bugs": [...],
            "smells": [...],
            "suggestions": [...]
        }

    def suggest_refactor(self, code: str) -> List[str]:
        # 重构建议
        pass

    def generate_tests(self, code: str) -> str:
        # 生成测试用例
        pass
```

**示例任务：**
1. 分析 Python 代码找出潜在 bug
2. 建议重构方案
3. 自动生成单元测试

---

### 3. **错误处理和重试示例** (17_error_handling_demo.py)
**优先级：中**

演示健壮的错误处理：
- 工具调用失败重试
- 优雅降级
- 错误恢复策略
- 断点续传

**核心功能：**
```python
class RobustAgent:
    def execute_with_retry(self, tool, args, max_retries=3):
        for attempt in range(max_retries):
            try:
                return tool(args)
            except Exception as e:
                if attempt < max_retries - 1:
                    # 重试策略
                    pass
                else:
                    # 优雅降级
                    return fallback_result

    def checkpoint_state(self, state):
        # 保存检查点
        pass

    def resume_from_checkpoint(self):
        # 从检查点恢复
        pass
```

**示例任务：**
1. 网络请求失败自动重试
2. LLM 超时后降级到缓存结果
3. 长任务中断后恢复

---

### 4. **流式输出示例** (18_streaming_demo.py)
**优先级：中**

演示实时反馈：
- LLM 流式输出
- 进度条显示
- 渐进式结果展示
- 用户中断处理

**核心功能：**
```python
async def stream_response(llm, prompt):
    async for chunk in llm.astream(prompt):
        yield chunk
        # 实时显示

def show_progress(current, total):
    # 进度条
    print(f"[{'=' * current}{' ' * (total - current)}] {current}/{total}")
```

**示例任务：**
1. 流式生成代码并实时显示
2. 长任务进度实时反馈
3. 支持用户随时中断

---

### 5. **上下文管理示例** (19_context_management_demo.py)
**优先级：低**

演示上下文优化：
- 长文本摘要
- 上下文窗口优化
- 记忆压缩
- 相关性过滤

**核心功能：**
```python
class ContextManager:
    def summarize_long_text(self, text, max_length):
        # 压缩长文本
        pass

    def filter_relevant_context(self, query, contexts):
        # 过滤相关上下文
        pass

    def manage_memory(self, conversation_history, max_tokens):
        # 记忆管理
        pass
```

**示例任务：**
1. 总结超长文档
2. 从大量上下文中提取关键信息
3. 优化对话历史

---

### 6. **Git 操作示例** (20_git_operations_demo.py)
**优先级：中**

模拟 Claude Code 的 Git 集成：
- 创建 commit（带格式化消息）
- 创建 PR
- 代码审查
- 分支管理

**核心功能：**
```python
class GitHelper:
    def create_commit(self, message, files):
        # 格式化 commit 消息
        formatted_message = f"""
{message}

🤖 Generated with Claude Code

Co-Authored-By: Claude <noreply@anthropic.com>
"""
        # 执行 git commit
        pass

    def create_pr(self, title, body, base_branch):
        # 使用 gh CLI 创建 PR
        pass
```

**示例任务：**
1. 自动生成符合规范的 commit 消息
2. 创建 PR 并填写描述
3. 代码审查建议

---

## 📊 优先级排序

### 高优先级（立即实现）
1. ✅ **14_tool_usage_demo.py** - 已完成
2. **15_file_operations_demo.py** - 核心功能
3. **16_code_analysis_demo.py** - 差异化能力

### 中优先级（本周完成）
4. **17_error_handling_demo.py** - 生产就绪必需
5. **18_streaming_demo.py** - 用户体验优化
6. **20_git_operations_demo.py** - 实际工作流

### 低优先级（可选）
7. **19_context_management_demo.py** - 高级优化

---

## 🎯 实现建议

### 文件命名规范
```
14_tool_usage_demo.py         ✅ 已完成
15_file_operations_demo.py    待实现
16_code_analysis_demo.py      待实现
17_error_handling_demo.py     待实现
18_streaming_demo.py          待实现
19_context_management_demo.py 待实现
20_git_operations_demo.py     待实现
```

### 代码结构模板
```python
#!/usr/bin/env python3
"""
XX - Claude Code Style XXX Demo

演示 Claude Code 的 XXX 能力：
1. 功能1
2. 功能2
3. 功能3
"""

# 标准导入
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

# 第三方库
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
# ...

# 环境加载
def load_environment() -> None:
    pass

# 工具/类定义
class XXXTool:
    pass

# 示例演示
def demo_feature_1():
    print("\n" + "=" * 80)
    print("示例 1: XXX")
    print("=" * 80)
    # ...

# 主入口
def main():
    demos = [
        ("功能1", demo_feature_1),
        # ...
    ]
    # 交互式菜单
    pass

if __name__ == "__main__":
    main()
```

### 文档要求
每个示例需要：
1. 清晰的文档字符串
2. 实际可运行的代码
3. 错误处理
4. 交互式演示菜单
5. 示例输出说明

---

## 📚 参考资源

### Claude Code 官方文档
- 工具使用: https://claude.com/claude-code/tools
- 最佳实践: https://claude.com/claude-code/best-practices

### LangChain 文档
- Tools: https://python.langchain.com/docs/modules/tools/
- Agents: https://python.langchain.com/docs/modules/agents/

### 相关示例
- LangChain Tools Gallery
- Claude Code 官方示例

---

## ✅ 完成检查清单

### 实现阶段
- [x] 14_tool_usage_demo.py
- [ ] 15_file_operations_demo.py
- [ ] 16_code_analysis_demo.py
- [ ] 17_error_handling_demo.py
- [ ] 18_streaming_demo.py
- [ ] 19_context_management_demo.py
- [ ] 20_git_operations_demo.py

### 测试阶段
- [ ] 语法检查（py_compile）
- [ ] 实际运行测试
- [ ] 输出结果验证

### 文档阶段
- [ ] 更新 README.md
- [ ] 添加使用说明
- [ ] 创建对比表格

---

## 🚀 快速开始（已完成示例）

```bash
# 工具使用示例
python 14_tool_usage_demo.py

# 选择要运行的示例：
#   1. 文件操作
#   2. Bash 命令
#   3. Web 搜索
#   4. 工具组合
#   5. 运行所有示例
#   0. 退出
```

---

## 💡 实现技巧

### 1. 使用 ReAct Agent 模式
```python
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

### 2. 工具定义最佳实践
```python
Tool(
    name="tool_name",
    description="清晰的工具描述，包括输入格式",
    func=tool_function,
)
```

### 3. 错误处理模式
```python
try:
    result = tool_function(input)
except SpecificError as e:
    # 针对性处理
    result = fallback
except Exception as e:
    # 通用处理
    result = error_message
```

---

## 🔗 相关文件

- [11_claude_code_style_enhanced.py](11_claude_code_style_enhanced.py) - 增强版主程序
- [CLARIFICATION_GUIDE.md](CLARIFICATION_GUIDE.md) - 反问机制指南
- [README.md](README.md) - 项目主文档

---

**更新日期**: 2025-11-30
**状态**: 进行中（1/7 完成）
