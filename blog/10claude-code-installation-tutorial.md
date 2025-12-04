---
title: "Claude Code 安装教程：快速搭建AI编程助手"
description: "10分钟完成Claude Code安装配置，包含环境准备、依赖安装、API配置和基础使用示例。"
keywords:
  - Claude Code
  - Anthropic
  - AI编程助手
  - API Key
  - .env配置
  - 安装教程
  - 快速上手
tags:
  - Tutorial
  - Claude
  - Installation
author: "langchain-learn"
date: "2025-10-10"
lang: "zh-CN"
canonical: "/blog/claude-code-installation-tutorial"
---

# Claude Code 安装教程：快速搭建AI编程助手

## 本页快捷跳转

- 直达： [引言](#intro) | [环境准备](#setup) | [Claude Code安装配置](#installation) | [基础使用示例](#basic-usage) | [常见错误与快速排查 (Q/A)](#qa) | [官方链接](#links) | [总结](#summary)

---

<a id="intro"></a>
## 引言

Claude Code是Anthropic推出的AI编程助手，能够帮助开发者提高编程效率。通过本教程，你将学会如何快速安装和配置Claude Code，并将其集成到你的开发环境中。

Claude Code的主要优势：
- **智能代码补全**：基于上下文提供准确的代码建议
- **代码解释**：帮助理解复杂的代码逻辑
- **错误调试**：快速定位和修复代码问题
- **多语言支持**：支持Python、JavaScript、Java等多种编程语言

<a id="setup"></a>

## 环境准备

### 软件准备

#### 1. node.js 安装

确保Node.js 版本需大于 18（推荐使用当前 LTS 版本，如 20+）。

##### Windows 安装与校验

###### 前置条件
- 操作系统：Windows 11 或 Windows 10（版本 1809 及以上）
- 具备安装权限：可安装 App Installer 或 winget
- 网络可访问 Microsoft Store 或 GitHub Releases

###### 安装 winget（Windows 包管理器）
- 方式一：Microsoft Store 安装 App Installer（系统会随之提供 `winget`）。企业环境若禁用商店，请联系管理员通过离线源分发 App Installer。
- 方式二：网站下载并手动安装：
  - 官方文档：https://learn.microsoft.com/windows/package-manager/winget/
  - GitHub Releases：https://github.com/microsoft/winget-cli/releases
  - 下载 `Microsoft.DesktopAppInstaller*.msixbundle`（App Installer 安装包），双击安装；安装完毕重新打开终端运行版本校验。
- 校验与维护源：
```powershell
winget --version     # 显示版本号表示安装成功
winget source list   # 查看源
winget source update # 更新源
winget source reset --force # 源异常时重置
```

###### 安装 Node（确保版本 > 18）
```powershell
# 安装 LTS（通常 >= 18，推荐 20+）
winget install OpenJS.NodeJS.LTS
# 或安装当前版
winget install OpenJS.NodeJS
# 安装前查看包信息：
winget show OpenJS.NodeJS.LTS
```

###### 版本校验
```powershell
node -v
npm -v
# 期望输出示例：v20.x.x（必须 > 18）
```

###### 版本升级与多版本管理
```powershell
# 若检测到版本 ≤ 18，升级到 LTS：
winget upgrade OpenJS.NodeJS.LTS

# 可选：使用 nvm-windows 进行多版本管理
# 安装：
# https://github.com/coreybutler/nvm-windows/releases
# 切换到 20 版本：
nvm install 20
nvm use 20
node -v  # 确认 > 18
```

##### macOS 安装与校验

###### 前置条件
- 系统版本：建议 macOS 12+（兼容性更佳）
- 已安装 Homebrew（https://brew.sh），或准备使用 nvm 多版本管理
- 终端为 `zsh` 或 `bash`，具备编辑 `~/.zshrc` / `~/.bashrc` 权限

###### 安装 Homebrew（纯终端）
```bash
# 1) 运行官方安装脚本
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 若系统未安装 Command Line Tools（CLT），安装脚本会自动提示并安装必要组件，无需手动操作

# 2) 配置 PATH（按芯片架构）
# Apple Silicon（arm64）：
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
# Intel（x86_64）：
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/usr/local/bin/brew shellenv)"

# 3) 校验与维护
brew --version && which brew
brew update && brew doctor
```

###### 使用 Homebrew 安装 Node（确保版本 > 18）
```bash
brew update
# 安装最新稳定版（通常 >= 18，推荐 20+）
brew install node

# 明确安装 20 LTS 并强制链接（如系统仍指向旧版本）
brew install node@20
brew link --overwrite --force node@20
```

###### 使用 nvm 进行多版本管理（推荐）
```bash
# 安装步骤参考官方文档：https://github.com/nvm-sh/nvm
# 典型安装（示例）：
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 使 nvm 生效（根据安装提示添加到 shell 配置文件）
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# 安装并切换到 20 版本，设为默认
nvm install 20
nvm use 20
nvm alias default 20
```

###### 版本校验
```bash
node -v
npm -v
# 期望输出示例：v20.x.x（必须 > 18）
```

###### 升级与路径冲突处理
```bash
# 若存在旧版本优先于新版本（PATH 冲突），优先强制链接 brew 的 node@20：
brew link --overwrite --force node@20

# 确认 PATH 中优先出现 Homebrew 或 nvm 的 node：
which node
echo $PATH

# 若通过 nvm 管理，确保当前会话已加载 nvm 并指向 20：
nvm use 20
node -v  # 确认 > 18
```

若 `node -v` 显示版本 ≤ 18，请按上述升级流程切换到 20+ LTS，以保证兼容性与更佳性能。


<a id="installation"></a>

## Claude Code 安装配置
### 前置条件
- 已安装 Node.js（`> 18`，推荐 `20+`），并已校验：`node -v`、`npm -v`

#### 全局安装 Claude Code
```bash powershell
npm install -g @anthropic-ai/claude-code --registry=https://registry.npmmirror.com
```
#### 版本校验
```bash powershell
claude --version
```
#### 更新 Claude Code
```bash powershell
claude update
```

#### 配置 Claude Code

##### 配置文件路径
- Windows：`C:/Users/<你的用户名>/.claude/settings.json`
- Linux/macOS：`~/.claude/settings.json`
- 若不存在该文件，可自行创建；不需要时可删除，不影响 Claude 的正常使用。

##### 推荐配置示例
```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "你的API密钥",
    "ANTHROPIC_BASE_URL": "地址",
    "ANTHROPIC_MODEL": "模型",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  },
  "permissions": {
    "allow": [],
    "deny": []
  }
}
```

###### 说明与建议
- `ANTHROPIC_AUTH_TOKEN`：填写你的 Anthropic API 密钥（与 `ANTHROPIC_API_KEY` 等效场景下二选一即可）。
- `ANTHROPIC_BASE_URL`：如使用代理或网关服务，填写你的 API 入口地址；不需要代理时可省略该字段（官方云常见为 `https://api.anthropic.com/`）。
- `ANTHROPIC_MODEL`：设置调用的模型名称，例如 `claude-3-5-sonnet-20241022`、`claude-3-opus-20240229`、`claude-3-haiku-20240307`。确保与服务商和 `ANTHROPIC_BASE_URL` 相匹配。
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`：设为 `"1"` 可减少非必要的网络请求，便于企业内网或受限环境使用。
- JSON 格式需使用双引号与合法键值；编辑完可用 `jq . ~/.claude/settings.json` 校验格式（macOS/Linux）。

##### 初始化示例
###### 启动 Claude Code

```bash powershell
claude
```
首次运行会提示设置 style，可以根据自己的喜好设置


首次第一次提问

## Claude Code 基础使用

### Claude Code 命令

#### 常用斜杠命令速览
- 基础与帮助：`/help`、`/config`、`/status`、`/doctor`
- 模型与隐私：`/model`、`/privacy-settings`
- 上下文与提示：`/clear`、`/compact [instructions]`、`/context`、`/output-style [style]`、`/statusline`、`/cost`、`/usage`、`/memory`
- 项目与工具：`/init`、`/add-dir`、`/permissions`、`/sandbox`、`/todos`、`/review`、`/pr_comments`、`/hooks`、`/bashes`
- 账号与导出：`/login`、`/logout`、`/export [filename]`、`/terminal-setup`、`/vim`
- MCP 管理：`/mcp`（管理 MCP 服务器连接与 OAuth）

#### 命令用法示例
- 切换模型：`/model claude-3-5-sonnet-20241022`
- 压缩对话：`/compact 仅保留最近 20 条交互的关键信息`
- 可视化上下文：`/context`
- 查看状态：`/status`（版本、模型、账户与连接性）
- 权限查看：`/permissions`（查看或更新允许/拒绝的权限）
- 初始化项目：`/init`（基于 `CLAUDE.md` 初始化工程说明与约定）
- MCP 管理：`/mcp list` 查看已连接、`/mcp connect <server>` 连接服务器

#### 自定义斜杠命令
- 项目级命令目录：`.claude/commands/`（随仓库共享，`/help` 列出时显示 `(project)`）
- 用户级命令目录：`~/.claude/commands/`（个人全局可用，`/help` 列出时显示 `(user)`）

示例：创建项目命令 `/optimize`
```bash
mkdir -p .claude/commands
echo "Analyze this code for performance issues and suggest optimizations:" > .claude/commands/optimize.md
```

示例：创建用户命令 `/security-review`
```bash
mkdir -p ~/.claude/commands
echo "Review this code for security vulnerabilities:" > ~/.claude/commands/security-review.md
```

在命令中使用 Bash 前缀与文件引用（需允许 Bash 工具）：
```md
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit
---

## Context
- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -10`

## Your task
Based on the above changes, create a single git commit.
```

更多命令与细节请参考官方文档：https://code.claude.com/docs/zh-CN/slash-commands

### Claude Code 安装MCP

#### 安装命令格式
- 标准格式：
```bash
claude mcp add --transport http <MCP名称> "<MCP地址>"
```
- 适用平台：Windows 与 macOS 使用同一命令；URL 建议使用引号括起。

#### 从 Smithery 安装步骤
1) 打开 Smithery（MCP 目录），选择目标 MCP。
2) 选择 Claude Code 客户端，复制安装命令（包含带密钥的 URL）。
3) 在终端执行复制的命令：
```bash
claude mcp add --transport http my-mcp "https://smithery.example.com/your-mcp?token=..."
```
4) 连接校验（REPL 内）：使用 `/mcp list` 查看已连接 MCP，或 `/mcp connect <name>` 主动连接。

#### 示例：安装“即梦”MCP（示意）
```bash
claude mcp add --transport http hans-m-yin-jimeng-mcp "<你的Smithery中带密钥的URL>"
```
- 说明：需要先在即梦服务获取 `sessionid`，再在 Smithery 对应页面生成带密钥的 URL；若连接异常，可能是会话过期，重新获取并更新。

#### 常用管理命令
- 查看：`/mcp list`
- 连接：`/mcp connect <name>`
- 断开：`/mcp disconnect <name>`
- 权限：`/permissions` 查看或更新权限策略
- 健康检查：`/doctor`

参考：Claude Code 使用进阶与 MCP 安装说明（示例）https://blog.csdn.net/baixiaobai2025/article/details/153931261

### Claude Code 安装plugin

#### 插件安装与管理
- 在 REPL 内打开插件管理：`/hooks` 或 `/permissions`
- 启用安全沙箱执行：`/sandbox`（隔离文件系统与网络，更安全）

#### 安装 Skills 插件（插件市场）
- 添加官方 Skills 插件市场：
```bash
/plugin marketplace add anthropics/skills
```
- 也可运行 `/plugin`，按提示添加市场后输入官方仓库地址：
```text
https://github.com/anthropics/skills
```
- 安装示例技能包：
```bash
/plugin install document-skills@anthropic-agent-skills
/plugin install example-skills@anthropic-agent-skills
```
- 管理插件：在 `/plugin` 菜单中可更新、删除、切换市场；安装成功后技能会出现在 `~/.claude/skills/` 或项目 `.claude/skills/` 目录中。

#### Skills 目录与使用
- 目录位置：
  - 个人技能：`~/.claude/skills/`（所有项目可用）
  - 项目技能：`.claude/skills/`（仅当前项目，便于团队共享）
- 每个技能目录包含 `SKILL.md` 与相关资源；也可手动将自定义技能放入上述目录。
- 运行机制：Claude 会按需加载与当前任务相关的技能，减少上下文膨胀与冗余提示；适用于文档处理（Excel/Word/PPT/PDF）、视觉设计、测试流水线等。
- 使用建议：结合 MCP（连接外部工具/数据）与 Skills（流程与规则封装）形成协同工作流。

参考：Skills 安装与使用实践（示例）https://javastack.blog.csdn.net/article/details/154370363

#### 终端辅助插件（示例）
- 安装 Shift+Enter 插件（iTerm2 / VSCode 换行支持）：
```bash
/terminal-setup
```
- Vim 模式辅助：
```bash
/vim
```

#### 注意事项
- 插件涉及的工具需要在权限中显式允许（如 Bash 的特定命令模式）。
- 企业环境中推荐优先启用沙箱化工具，减少外部访问。

### Claude Code 设置agent

#### 创建与管理 Sub Agents
- 打开代理管理：`/agents` → Create New Agent
- 设置内容：
  - 名称与用途：为代理设定唯一名称与职责范围
  - 系统提示：明确角色、目标、流程与必须遵守的规则
  - 工具权限：仅授予必要工具（例如审查代理只读文件，不赋予写或执行权限）
  - 触发指令：教授调用方式与使用场景（唤醒词/触发命令）

#### 使用方式
- 显式调用：在复杂场景中指定代理执行特定任务，确保可控性。
- 链式调用：针对多阶段任务依次使用多个代理协作，提升准确性与质量。

#### 启动选项（谨慎）
- 跳过交互式权限确认：
```bash
claude --dangerously-skip-permissions
```
- 提示：该选项会默认允许代理执行操作，存在安全风险，仅在隔离环境或充分评估后使用。

参考：Sub Agents 创建与实践（示例）https://blog.csdn.net/baixiaobai2025/article/details/153931261


## Claude Code VScode 插件安装

#### 安装要求
- VS Code 版本：`1.98.0` 或更高。

#### 安装步骤
- 在 VS Code 扩展市场搜索并安装 “Claude Code”。
- 安装后点击侧边栏的 Spark 图标打开 Claude Code 面板，即可在 IDE 内使用（支持计划与编辑预览、自动接受编辑模式、文件管理、对话历史、多个会话、快捷键与斜杠命令）。
- 提示：将侧边栏拖宽可查看内联差异，点击可展开完整详情。

#### 支持与工作原理
- 扩展会使用通过 CLI 配置的 MCP 服务器；请先在 CLI 中完成 MCP 的添加与连接。
- 扩展支持大多数 CLI 斜杠命令与键盘快捷键，提供原生 IDE 体验与多会话管理。

#### 使用第三方提供商（Vertex / Bedrock）
- 在 VS Code 设置中搜索 “Claude Code: Environment Variables”，添加以下环境变量以启用第三方提供商：
  - `CLAUDE_CODE_USE_BEDROCK`：启用 Amazon Bedrock（值：`"1"/"true"`）
  - `CLAUDE_CODE_USE_VERTEX`：启用 Google Vertex AI（值：`"1"/"true"`）
  - `ANTHROPIC_API_KEY`：第三方访问的 API 密钥（必需）
  - `AWS_REGION` / `AWS_PROFILE`：Bedrock 所需区域与配置文件
  - `CLOUD_ML_REGION` / `ANTHROPIC_VERTEX_PROJECT_ID`：Vertex AI 所需区域与项目
  - `ANTHROPIC_MODEL`：覆盖主模型的 ID（例如供应商提供的完整模型 ID）
  - `ANTHROPIC_SMALL_FAST_MODEL`：可选，小型/快速模型覆盖
  - `CLAUDE_CODE_SKIP_AUTH_LOGIN`：禁用登录提示（`"1"/"true"`）

#### MCP 与子代理
- 扩展暂不提供完整 MCP 服务器与子代理配置入口；请在 CLI 中先行配置（扩展会自动使用 CLI 配置）。

#### 旧版 CLI 集成
- 从 VS Code 集成终端运行 `claude` 可自动启用旧版 IDE 交互（共享选区/选项卡、在 IDE 查看差异、文件引用快捷键等）。
- 外部终端使用 `/ide` 连接到 VS Code 实例；运行 `claude` 后输入 `/config`，将差异工具设置为 `auto` 以自动 IDE 检测。
- 支持 VS Code、Cursor、Windsurf、VSCodium；确保对应 `code/cursor/windsurf/codium` 命令可用。

#### 安全考虑
- 在 VS Code 中启用自动编辑权限时，Claude Code 可能修改 IDE 配置文件，存在被 IDE 自动执行的风险。
- 建议：为不受信任的工作区启用 VS Code 受限模式；对编辑使用手动批准模式；谨慎确保仅在受信任提示下使用。

#### 故障排除
- 扩展未安装：确认 VS Code 版本兼容；检查权限；尝试从市场网站直接安装。
- 旧版集成不工作：从 VS Code 集成终端运行 `claude`；确保安装并可用 `code/cursor/windsurf/codium` 命令；在 `claude` 中 `/config` 设置差异工具为 `auto`。
- MCP 不可用：先通过 CLI 添加并连接 MCP（参考前文 “Claude Code 安装MCP”）。

参考文档：https://code.claude.com/docs/zh-CN/vs-code


<a id="qa"></a>
## 常见错误与快速排查 (Q/A)

- Node/WSL 路径与版本冲突
  - 现象：`exec: node: not found`，`which node` 指向 `/mnt/c/...` 或 Node 版本 ≤ 18。
  - 修复：在 WSL/Unix 使用发行版包管理器或 `nvm` 安装并确保 `which node` 指向 `/usr/...`；macOS 用 Homebrew 或 `nvm` 并执行 `brew link --overwrite --force node@20`；必要时关闭 WSL 继承 Windows PATH。
  - 参考：故障排除（官方）`https://docs.claude.com/zh-CN/docs/claude-code/troubleshooting`

- 全局安装后 `claude` 不在 PATH
  - 现象：`claude --version` 报“命令不存在”。
  - 修复：确认 `npm prefix -g` 的 `bin` 目录已加入 PATH（例如 `~/.npm-global/bin`、`/opt/homebrew/bin`）；用 `which claude` 验证；必要时重新安装或修复 shell 初始化文件。
  - 参考：NPM 包页面 `https://www.npmjs.com/package/@anthropic-ai/claude-code`

- Windows 原生环境问题与 WSL 建议
  - 现象：在纯 Windows 环境下安装/运行体验不佳。
  - 修复：优先在 WSL(Ubuntu) 中按 Linux 指南安装并使用；VS Code 插件也建议配合 WSL。
  - 参考：IDE 集成（官方概览）`https://docs.claude.com/zh-CN/docs/claude-code/ide-integrations`

- 配置文件 `~/.claude/settings.json` 语法或变量不生效
  - 现象：启动失败或环境变量未被识别；仍提示 `/login`。
  - 修复：用 `jq . ~/.claude/settings.json` 校验 JSON；环境变量值使用字符串（如 `"1"`）；根据部署选择 `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY` 并保持键名一致。
  - 参考：设置与权限（官方）`https://docs.claude.com/zh-CN/docs/claude-code/settings`

- `ANTHROPIC_BASE_URL` 与 `ANTHROPIC_MODEL` 不匹配
  - 现象：调用报 4xx/5xx 或提示模型不存在。
  - 修复：确保模型 ID 与服务商/网关一致；必要时查阅第三方提供商/网关文档，使用其要求的 base URL 与模型全名。
  - 参考：设置与权限（官方）`https://docs.claude.com/zh-CN/docs/claude-code/settings`

- `/mcp list` 显示 “No MCP servers configured”
  - 现象：MCP 列表为空、无法连接服务器。
  - 修复：使用 `claude mcp add --transport http <name> "<url>"` 或 `--transport stdio` 添加；注意 URL 需加引号；添加后重启 CLI 并用 `/mcp connect <name>` 连接。
  - 参考：故障排除（官方）`https://docs.claude.com/zh-CN/docs/claude-code/troubleshooting`、命令（官方）`https://code.claude.com/docs/zh-CN/slash-commands`

- 插件与 Skills 未生效或缺权限
  - 现象：`/plugin` 安装后技能未加载，或技能内工具调用失败。
  - 修复：在 `/permissions` 显式允许相关工具（如 Bash/WebFetch）；按规范将技能放到 `~/.claude/skills/` 或项目 `.claude/skills/` 并重启。
  - 参考：斜杠命令（官方）`https://code.claude.com/docs/zh-CN/slash-commands`、设置与权限（官方）`https://docs.claude.com/zh-CN/docs/claude-code/settings`

- VS Code 插件不可用或第三方提供商配置不全
  - 现象：扩展面板不可用；Vertex/Bedrock 无法调用。
  - 修复：确保 VS Code 版本 `1.98.0+`；在设置中添加 `CLAUDE_CODE_USE_BEDROCK`/`CLAUDE_CODE_USE_VERTEX`、`ANTHROPIC_API_KEY`、区域/项目与 `ANTHROPIC_MODEL` 等变量；重启 VS Code。
  - 参考：VS Code 插件（官方）`https://code.claude.com/docs/zh-CN/vs-code`

- 网络/代理/证书导致 API 不通
  - 现象：`/doctor` 报网络问题或 TLS 证书错误。
  - 修复：在系统/CLI 正确配置代理与可信 CA；企业网关需在 `settings.json` 同时配置 `ANTHROPIC_BASE_URL` 与令牌并使用支持的模型 ID。
  - 参考：故障排除（官方）`https://docs.claude.com/zh-CN/docs/claude-code/troubleshooting`



<a id="links"></a>
## 官方链接

- Claude Code 文档首页（中文）：https://code.claude.com/docs/zh-CN/
- 斜杠命令（中文）：https://code.claude.com/docs/zh-CN/slash-commands
- VS Code 插件（中文）：https://code.claude.com/docs/zh-CN/vs-code
- 故障排除（中文）：https://docs.claude.com/zh-CN/docs/claude-code/troubleshooting
- 设置与权限（中文）：https://docs.claude.com/zh-CN/docs/claude-code/settings
- IDE 集成（概览）：https://docs.claude.com/zh-CN/docs/claude-code/ide-integrations
- NPM 包页面：https://www.npmjs.com/package/@anthropic-ai/claude-code


<a id="summary"></a>
## 总结

🎉 **恭喜你成功安装并配置了Claude Code！**

通过本教程，你已经：
- 完成了Claude Code的环境准备和依赖安装
- 配置了Anthropic API密钥和模型参数
- 学习了基础的使用方法和常见场景示例
- 掌握了常见问题的排查方法

现在你可以开始使用Claude Code来提高你的编程效率了！记得在实际使用中根据具体需求调整参数配置。