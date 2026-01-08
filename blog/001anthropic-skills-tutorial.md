---
title: "Anthropic Claude Code Skills 完全指南：从入门到精通"
description: "如何使用 Anthropic Skills 扩展 Claude Code 能力？本文提供从 Hello World 到复杂脚本开发的完整教程，涵盖代码审查、Git 自动化及 LangChain 集成实战。"
image: "/images/blog/anthropic-skills-cover.jpg"
keywords:
  - Anthropic
  - Claude Code
  - Skills
  - AI Agent
  - Workflow Automation
  - 斜杠命令
  - 自动化
  - LangChain
  - 脚本开发
  - config.json
  - 代码审查
  - Git 辅助
tags:
  - Tutorial
  - Skills
  - Automation
  - LLM
  - Agentic AI
author: "langchain-learn"
date: "2025-12-31"
last_modified_at: "2026-01-01"
lang: "zh-CN"
canonical: "https://your-domain.com/blog/anthropic-skills-tutorial"
audience: "开发者 / 具备 Python/Bash 基础的工程师"
difficulty: "beginner-intermediate"
estimated_read_time: "25-35min"
topics:
  - Anthropic Skills
  - Claude Code
  - Automation
  - LangChain Integration
entities:
  - Anthropic
  - Claude
  - LangChain
  - Python
  - Bash
---

# Anthropic Claude Code Skills 完全指南：从入门到精通

## 📍 导航指南

根据你的需求，选择合适的阅读路径：

- 🎓 **新手？从这里开始** → [SKILL - 快速开始](#tutorial) - 创建你的第一个 Skill
- 🛠️ **有具体问题？** → [SKILL - 实用指南](#howto) - 快速找到解决方案
- 📚 **想深入理解？** → [SKILL - 深入理解](#explanation) - 了解工作原理
- 📖 **需要查 API？** → [SKILL - 参考文档](#reference) - 字段和参数参考

---

## 目录

### 第一部分：SKILL - 快速开始 🎓
- [快速开始](#tutorial)
  - [创建你的第一个 Skill](#first-skill)
  - [测试运行](#test-skill)
  - [理解 Skills 的三种类型](#three-types)

### 第二部分：SKILL - 实用指南 🛠️
- [实用指南](#howto)
  - [如何创建脚本型 Skill](#howto-script-skill)
  - [如何添加参数处理](#howto-parameters)
  - [如何调试 Skill 问题](#howto-debug)
  - [如何集成 LangChain](#howto-langchain)

### 第三部分：SKILL - 深入理解 📚
- [深入理解](#explanation)
  - [Skills 工作机制详解](#deep-dive-mechanisms)
  - [渐进式加载机制](#progressive-loading)
  - [设计哲学](#design-philosophy)
  - [Skills vs MCP Servers](#skills-vs-mcp)

### 第四部分：SKILL - 参考文档 📖
- [完整参考](#reference)
  - [SKILL.md 字段完整列表](#config-reference)
  - [allowed-tools 工具参考](#tools-reference)
  - [脚本 API 规范](#script-api)
  - [目录结构规范](#directory-structure-ref)

### 附录
- [常见问题 FAQ](#appendix-faq)

---

## 什么是 Anthropic Skills？

**Anthropic Skills** 是 Claude Code 的扩展机制，让你通过简单的 **Markdown 文件**自定义斜杠命令（如 `/code-review`、`/git-helper`）。

**一句话总结**：Skills = 给 Claude 提供可复用的"技能包"，让它按照你定义的方式工作。

### 核心概念

```
创建 .claude/skills/my-skill/SKILL.md
    ↓
定义指令和工作流程
    ↓
调用 /my-skill
    ↓
Claude 按照 SKILL.md 执行任务
```

**最简单的例子**：
```markdown
---
name: hello-world
description: 打印问候语
---

# Hello World Skill

当用户调用 /hello-world 时，打印 "✨ Hello, World!"
```

就这么简单！只需一个 Markdown 文件。

### 为什么需要 Skills？

| 场景 | 没有 Skills ❌ | 使用 Skills ✅ |
|------|--------------|--------------|
| **代码审查** | 每次都要说明检查标准 | `/code-review file.py` 一键完成 |
| **Git 操作** | 手动执行多个命令 | `/git-commit "message"` 自动化 |
| **项目初始化** | 手动创建目录和文件 | `/init-project python my-app` 秒创建 |

**核心价值**：
- 📦 **可复用** - 一次定义，随处使用
- 🤝 **可分享** - 团队共享最佳实践
- 🎯 **标准化** - 避免重复解释
- ⚡ **高效** - 一键完成复杂任务

---

<a id="tutorial" data-alt="Tutorial 快速开始 入门"></a>
# 第一部分：SKILL - 快速开始 🎓

> **目标**：创建并运行你的第一个 Skill，完成基础的 "Hello World" 功能。

<a id="first-skill" data-alt="第一个 Skill 创建"></a>
## Step 1: 创建你的第一个 Skill

让我们创建一个简单的 "Hello World" Skill。

### 1.1 创建目录

```bash
mkdir -p .claude/skills/hello-world
cd .claude/skills/hello-world
```

### 1.2 创建 SKILL.md 文件

创建 `SKILL.md`（这是唯一必需的文件）:

```markdown
---
name: hello-world
description: 我的第一个 Skill - 打印问候语。用于演示 Agent Skills 标准的基本结构。
version: "1.0.0"
metadata:
  author: Your Name
  difficulty: beginner
---

# Hello World Skill

这是一个简单的示例 Skill，用于演示 Agent Skills 的基本结构。

## 使用方法

调用此 Skill 向用户打印问候语:


/hello-world


自定义问候对象:

/hello-world --name Alice

## 功能说明

当用户调用 `/hello-world` 时，Claude 会:
1. 读取用户提供的 name 参数（默认为 "World"）
2. 生成友好的问候消息
3. 添加欢迎信息和表情符号

## 示例

**示例 1**: 基本用法

用户: /hello-world
Claude: ✨ Hello, World!
        🎉 Welcome to Anthropic Skills!


**示例 2**: 自定义名字

用户: /hello-world --name Alice
Claude: ✨ Hello, Alice!
        🎉 Welcome to Anthropic Skills!

## 参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| name | string | 否 | World | 要问候的名字 |

## 最佳实践

- 保持问候语简洁友好
- 使用适当的表情符号增强视觉效果
- 确保输出格式一致
```

**就这样！** 一个简单的 Skill 只需要一个 SKILL.md 文件。

---

<a id="test-skill" data-alt="测试 运行 Skill"></a>
## Step 2: 测试你的 Skill

保存文件后，在 Claude Code 中运行：

```
执行 hello-world
```

你应该看到：

```
Claude: ✨ Hello, World!
        🎉 Welcome to Anthropic Skills!
```

### ✅ 成功！

恭喜！你已经创建了第一个 Skill。

现在试试自定义名字：

```
你: /hello-world --name Alice
```

输出：

```
Claude: ✨ Hello, Alice!
        🎉 Welcome to Anthropic Skills!
```


你已经掌握了 Skills 的基础。接下来了解三种 Skill 类型。

> 💡 **遇到问题？** 如果 Skill 没有加载，查看[附录 B：常见问题](#appendix-faq)

---

<a id="three-types" data-alt="三种 类型 Skills"></a>
## Step 3: 理解 Skills 的三种类型

Skills 有三种工作方式，从简单到复杂：

### ⭐ 类型 1：纯知识型（最简单）

只需要 SKILL.md 文件，为 Claude 提供知识和指导。

**适用场景**：代码风格指南、最佳实践

**最小示例**：
```markdown
---
name: python-style
description: Python 代码风格指南
---

# Python Style Guide

审查代码时检查：
- 函数名使用 snake_case
- 类名使用 PascalCase
```

### ⭐⭐ 类型 2：工具调用型（中等）

使用 `allowed-tools` 让 Claude 调用文件工具。

**适用场景**：文件分析、代码搜索

**最小示例**：
```markdown
---
name: find-todos
description: 查找 TODO 标记
allowed-tools: "Grep"
---

# Find TODOs

使用 Grep 搜索 "TODO"，列出所有待办事项。
```

### ⭐⭐⭐ 类型 3：脚本执行型（高级）

编写 Python/Bash 脚本执行复杂任务。

**适用场景**：复杂计算、API 调用

<a id="howto" data-alt="How-to 实用指南"></a>
# 第二部分：SKILL - 实用指南 

> **目标**：SKILL的规范

<a id="directory-structure" data-alt="目录 结构 规范"></a>
### 目录结构规范

**Agent Skills 标准**目录结构:

```
.claude/skills/<skill-name>/
├── SKILL.md           # 核心文件（必需）
├── scripts/           # 辅助脚本（可选）
│   ├── helper.py
│   ├── process.sh
│   └── utils.js
├── references/        # 参考文档（可选）
│   ├── api-docs.md
│   └── examples.md
└── assets/            # 资源文件（可选）
    ├── logo.png
    └── template.json
```

**要点**：
- ✅ 目录名使用小写字母和连字符（kebab-case）
- ✅ **SKILL.md 是唯一必需的文件**
- ✅ SKILL.md 必须以 YAML frontmatter 开头
- ✅ 其他目录（scripts、references、assets）都是可选的
- ✅ 支持渐进式加载：Claude 只在需要时加载额外资源

<a id="howto-script-skill" data-alt="脚本 Skill 开发"></a>
### 如何创建脚本型 Skill

对于需要复杂计算、外部 API 调用或特殊逻辑的场景，可以编写自定义脚本。

**适用场景**：
- 复杂数据处理
- API 集成
- 自动化测试
- 代码生成
- 性能分析

**目录结构**：

```
.claude/skills/code-analyzer/
├── SKILL.md           # 核心文件
└── scripts/           # 脚本目录
    ├── analyze.py     # Python 分析脚本
    ├── format.sh      # Bash 格式化脚本
    └── utils.js       # Node.js 工具脚本
```

**Step 1: 创建 SKILL.md**

```markdown
---
name: code-analyzer
description: 深度代码分析工具，支持复杂度分析、依赖检测和安全审查
allowed-tools: "Bash Read"
version: "1.0.0"
---

# Code Analyzer

深度分析代码质量、复杂度和安全性。

## 使用方法

分析 Python 代码：

/code-analyzer analyze src/main.py

## 执行流程

1. Claude 使用 Read 工具读取源文件
2. 调用 Bash 工具执行 `scripts/analyze.py <file>`
3. 脚本执行深度分析并输出 JSON 结果
4. Claude 解析结果并生成用户友好的报告
```

**Step 2: 创建 Python 脚本 (`scripts/analyze.py`)**

```python
#!/usr/bin/env python3
"""代码分析脚本"""
import sys
import ast
import json
from pathlib import Path

def analyze_python_file(file_path):
    """分析 Python 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    tree = ast.parse(content)

    # 统计信息
    stats = {
        'file': file_path,
        'lines': len(content.split('\n')),
        'functions': 0,
        'classes': 0,
        'imports': [],
        'complexity': 0
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats['functions'] += 1
        elif isinstance(node, ast.ClassDef):
            stats['classes'] += 1
        elif isinstance(node, ast.Import):
            stats['imports'].extend([n.name for n in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                stats['imports'].append(node.module)

    return stats

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing file path"}))
        sys.exit(1)

    file_path = sys.argv[1]

    if not Path(file_path).exists():
        print(json.dumps({"error": f"File not found: {file_path}"}))
        sys.exit(1)

    try:
        result = analyze_python_file(file_path)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Step 3: 测试运行**

```bash
chmod +x scripts/analyze.py
```

在 Claude 中运行：
```
/code-analyzer analyze src/main.py
```

---

<a id="howto-parameters" data-alt="参数处理指南"></a>
### 如何添加参数处理

Skills 支持灵活的参数传递。你可以通过位置参数或命名参数（flags）来控制 Skill 的行为。

**1. 在 SKILL.md 中定义**

虽然不是强制的，但最好在文档中说明支持的参数：

```markdown
## 参数
| 参数 | 类型 | 说明 |
|------|------|------|
| file | string | 目标文件路径 |
| --verbose | boolean | 启用详细输出 |
```

**2. 在 Python 脚本中解析**

使用 `argparse` 处理参数是最标准的方式：

```python
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Target file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")
    
    # 解析参数（Claude 会将所有后续内容传给脚本）
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing {args.file} in verbose mode...")
```

**3. 在 Bash 脚本中解析**

```bash
#!/bin/bash
FILE=$1
VERBOSE=false

shift # 移除第一个参数

while getopts ":v" opt; do
  case ${opt} in
    v ) VERBOSE=true ;;
    \? ) echo "Invalid option: -$OPTARG" 1>&2; exit 1 ;;
  esac
done
``` 

---

<a id="howto-debug" data-alt="调试技巧"></a>
### 如何调试 Skill 问题

当 Skill 不按预期工作时，使用以下技巧进行排查：

**1. 检查 stderr 输出**
Claude 会捕获脚本的标准错误输出。确保你的脚本在出错时打印到 stderr：
```python
print("Error: File not found", file=sys.stderr)
sys.exit(1)
```

**2. 使用 `/tool` 命令手动调用**
你可以尝试直接调用底层工具来验证路径和权限：
```
/tool Bash
command: ls -l .claude/skills/my-skill/scripts/
```

**3. 验证 JSON 格式**
如果你的脚本输出 JSON 给 Claude 解析，确保它是合法的。
*   ❌ 错误：`print(f"{'key': 'value'}")` (Python 默认单引号不是标准 JSON)
*   ✅ 正确：`print(json.dumps({'key': 'value'}))`


<a id="explanation" data-alt="Explanation 深入理解"></a>
# 第三部分：SKILL - 深入理解 📚

> **目标**：深入理解 Skills 的工作机制、设计哲学和安全模型，知其然更知其所以然。

<a id="deep-dive-mechanisms" data-alt="Skills 工作机制详解"></a>
### Skills 工作机制详解

当你在 Claude Code 中输入 `/my-skill` 时，实际上发生了一系列复杂的交互：

1.  **意图识别 (Intent Recognition)**：
    Claude 首先解析你的输入，识别出斜杠命令 `/my-skill`。它会在 `.claude/skills` 目录下查找对应的 `SKILL.md` 文件。

2.  **上下文加载 (Context Loading)**：
    Claude 读取 `SKILL.md` 的内容，包括 frontmatter 配置和正文描述。这些内容被动态注入到当前的对话上下文中，作为 System Prompt 的一部分。
    *   *关键点*：这就是为什么 Skill 的描述越清晰，效果越好。你实际上是在编写 Prompt。

3.  **规划与执行 (Planning & Execution)**：
    基于 Skill 的描述和用户的输入参数，Claude 生成一个执行计划。
    *   如果是 **知识型 Skill**，它直接根据知识库回答。
    *   如果是 **工具型 Skill**，它会调用 `allowed-tools` 中定义的工具（如 `Grep`, `Read`）。
    *   如果是 **脚本型 Skill**，它会构建命令行调用，执行 `scripts/` 目录下的脚本。

4.  **结果处理 (Result Processing)**：
    工具或脚本的输出（stdout/stderr）被捕获并返回给 Claude。Claude 分析这些输出，根据 Skill 的目标生成最终的用户响应。

<a id="progressive-loading" data-alt="渐进式加载机制"></a>
### 渐进式加载机制 (Progressive Loading)

Skills 采用了一种高效的**渐进式加载**策略，这对于维护大规模 Skill 库至关重要：

1.  **惰性注册 (Lazy Registration)**：
    Claude 启动时，仅扫描 `.claude/skills/*/SKILL.md` 的 Frontmatter 部分（获取 `name` 和 `description`），用于构建自动补全索引。此时**不会**读取正文内容，几乎不占用 Context Window。

2.  **按需激活 (On-Demand Activation)**：
    只有当你真正输入 `/my-skill` 并回车时，Claude 才会读取该 Skill 的完整 `SKILL.md` 内容并注入上下文。

3.  **资源隔离 (Resource Isolation)**：
    `scripts/` 目录下的代码文件、`references/` 下的文档，除非被明确调用（如 `Read` 工具读取或 `Bash` 工具执行），否则永远不会进入 LLM 的上下文。

**优势**：这意味着你可以拥有几百个 Skills，而不会拖慢 Claude 的响应速度，也不会因为无关的 Prompt 干扰 Claude 的推理。

<a id="design-philosophy" data-alt="设计 哲学"></a>
### 设计哲学

Anthropic Skills 的设计遵循了几个核心原则：

*   **声明式定义 (Declarative)**：你告诉 Claude "做什么" (What)，而不是详细的 "怎么做" (How)。Markdown 格式本身就是为了让人类和 AI 都能轻松阅读。
*   **极简主义 (Minimalism)**：最简的 Skill 只需要一个文件。没有复杂的样板代码，没有编译过程。
*   **安全沙箱 (Sandboxing)**：通过 `allowed-tools` 显式授权。如果一个 Skill 只需要读取文件，它就不应该有写入权限或网络访问权限。这遵循了"最小权限原则"。
*   **语言无关 (Language Agnostic)**：脚本可以用 Python, Bash, Node.js, Go 等任何语言编写，只要系统环境支持即可。Skills 协议只关心标准输入/输出。

<a id="skills-vs-mcp" data-alt="Skills vs MCP"></a>
### Skills vs MCP Servers

你可能听说过 **Model Context Protocol (MCP)**。Skills 和 MCP 有什么区别？

| 特性 | Skills | MCP Servers |
| :--- | :--- | :--- |
| **定位** | 轻量级、任务导向的自动化脚本 | 重量级、通用的上下文/工具提供者 |
| **部署** | 本地文件 (`.claude/skills`) | 独立服务 (Local or Remote Server) |
| **交互** | 斜杠命令触发 (`/cmd`) | 被动工具调用 (Function Calling) |
| **编写难度** | 低 (Markdown + Simple Scripts) | 中/高 (TypeScript/Python SDK) |
| **典型场景** | 代码审查、Git 操作、项目脚手架 | 数据库连接、API 集成、复杂系统交互 |

**结论**：
*   **MCP Servers** 是底层的**能力提供者 (Capability Provider)**，它们像 API 一样提供原子化的工具（如"查询数据库"、"读取 Slack"）。
*   **Skills** 是上层的**工作流编排者 (Workflow Orchestrator)**，它们将这些原子工具（包括 MCP 工具和内置工具）串联起来，形成一个特定的、可执行的业务流程（如"审查代码并提交"）。

简而言之：**MCP 提供积木，Skills 搭建房子。**

---

<a id="reference" data-alt="Reference 完整参考"></a>
# 第四部分：SKILL - 完整参考 📖

> **目标**：提供查阅手册，包含所有配置字段、工具列表和 API 规范。

<a id="config-reference" data-alt="SKILL.md 字段参考"></a>
### SKILL.md Frontmatter 配置

`SKILL.md` 文件的头部 YAML 配置支持以下字段：

| 字段 | 类型 | 必需 | 说明 | 示例 |
| :--- | :--- | :--- | :--- | :--- |
| `name` | string | ✅ | Skill 的唯一标识符，对应斜杠命令。只能包含小写字母、数字和连字符。 | `code-review` |
| `description` | string | ✅ | 简短描述，显示在自动补全菜单中。 | `Automated code review tool` |
| `version` | string | ❌ | 版本号，遵循 SemVer 规范。 | `1.0.0` |
| `allowed-tools` | string | ❌ | 允许使用的工具列表，空格分隔。 | `Read Grep Bash` |
| `metadata` | object | ❌ | 额外的元数据字典。 | `author: "Alice"` |

<a id="tools-reference" data-alt="allowed-tools 工具列表"></a>
### 内置工具列表 (allowed-tools)

在 `allowed-tools` 字段中可以指定以下工具：

*   **`Read`**: 读取文件内容。
    *   *权限*：只读。
*   **`Grep`**: 在代码库中搜索字符串或正则。
    *   *权限*：只读。
*   **`LS`**: 列出目录内容。
    *   *权限*：只读。
*   **`Bash`**: 执行 Shell 命令。
    *   *权限*：**高风险**。允许执行任意系统命令和脚本。通常用于脚本型 Skill。
*   **`Edit`** (慎用): 编辑文件。
    *   *权限*：读写。通常不建议在 Skill 中直接使用，除非是专门的重构工具。

<a id="script-api" data-alt="脚本 API 规范"></a>
### 脚本交互规范

如果你编写自定义脚本（如 Python/Bash），请遵循以下 I/O 规范：

1.  **输入 (Input)**：
    *   **命令行参数**：通过 `sys.argv` 或 `$1`, `$2` 接收用户参数。
    *   **标准输入 (stdin)**：虽然支持，但通常通过参数传递文件路径更为常见。

2.  **输出 (Output)**：
    *   **标准输出 (stdout)**：打印主要结果。Claude 会读取这些内容并呈现给用户。建议输出结构化数据（如 JSON）以便 Claude 解析，或者输出清晰的文本。
    *   **标准错误 (stderr)**：打印日志、调试信息或错误消息。

3.  **退出码 (Exit Codes)**：
    *   `0`: 成功。
    *   `非0`: 失败。Claude 会根据 stderr 的内容告知用户发生了错误。

<a id="directory-structure-ref" data-alt="目录结构参考"></a>
### 目录结构参考

```text
.claude/skills/
└── <skill-name>/          # Skill 根目录
    ├── SKILL.md           # [必需] 配置文件
    ├── scripts/           # [可选] 存放可执行脚本
    │   ├── run.py
    │   └── utils.sh
    └── assets/            # [可选] 静态资源
        └── template.txt
```

---

<a id="appendix-faq" data-alt="FAQ 常见问题"></a>
### 附录：常见问题 FAQ

**Q: 我修改了 SKILL.md，为什么没有生效？**
A: Claude Code 通常会缓存 Skills。尝试重启 Claude Code 或运行 `/refresh` (如果支持) 来重新加载配置。

**Q: 脚本没有执行权限怎么办？**
A: 确保你的脚本文件具有可执行权限。在 Unix 系统上运行 `chmod +x .claude/skills/<name>/scripts/<script>`。

**Q: 可以在 Skill 中调用其他 Skill 吗？**
A: 目前官方标准不支持直接嵌套调用，但你可以通过 Bash 脚本间接调用其他 CLI 工具。

---

<a id="conclusion" data-alt="结语"></a>
## 结语

你现在已经掌握了构建 Claude Skills 的完整技能树。从简单的自动化脚本到复杂的 Agent 工作流，Skills 是连接人类意图与 AI 执行力的关键桥梁。

不要让这些知识停留在文档里——现在就审视你的日常工作流，找出那些重复、繁琐的环节，用一个精心设计的 Skill 来替代它。这不仅是效率的提升，更是向 AI Native 开发模式迈出的重要一步。

**未来已来，动手创造吧！** 🚀



