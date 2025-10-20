---
title: "LangChain 入门教程：学习 MCP（Model Context Protocol）"
description: "基于模块 08-mcp-server 的实战教程，涵盖协议原理、最小服务端/客户端实现、运行与排错、进阶扩展与最佳实践。"
keywords:
  - LangChain
  - MCP
  - Model Context Protocol
  - JSON-RPC
  - stdin/stdout
  - 工具调用
  - 子进程
  - 安全
tags:
  - Tutorial
  - MCP
  - Protocol
author: "langchain-learn"
date: "2025-10-15"
lang: "zh-CN"
canonical: "/blog/langchain-mcp-server-tutorial"
audience: "初学者 / 具备Python基础的LLM工程师"
difficulty: "beginner-intermediate"
estimated_read_time: "12-18min"
topics:
  - Model Context Protocol
  - JSON-RPC
  - 工具注册与调用
  - 子进程通信
  - 安全与权限
entities:
  - LangChain
  - MCP
  - JSON
  - subprocess
qa_intents:
  - "MCP 是什么？为什么值得在本地先学最小实现？"
  - "如何快速运行服务端与客户端并完成工具调用？"
  - "JSON-RPC 行分隔通信具体长什么样？"
  - "如何设计工具元数据与参数校验？"
  - "子进程通信的常见坑与排错思路？"
  - "如何扩展到文件系统/HTTP/数据库等高级工具？"
  - "如何做安全与权限控制（白名单、审计）？"
  - "与 LangChain 工具如何对齐与集成？"
---

# LangChain 入门教程：学习 MCP（Model Context Protocol）

## 本页快捷跳转
- 目录：
  - [引言](#intro)
  - [环境准备](#setup)
  - [MCP 是什么？原理速览](#what-is-mcp)
  - [最小服务端实现](#server)
  - [最小客户端实现](#client)
  - [运行与演示](#run-demo)
  - [进阶扩展与最佳实践](#advanced)
  - [常见错误与快速排查 (Q/A)](#qa)
  - [总结](#summary)
  - [术语与别名](#glossary)

---

<a id="intro" data-alt="introduction 引言 概述 动机"></a>
## 引言
此前《[函数调用与工具（Function Calling / Tools）](/blog/langchain-function-calling-tutorial)》一文介绍了函数调用能力，并指出不同大模型厂商的实现存在差异。为统一工具调用的接口与语义，Anthropic 于 2024 年开源 MCP（Model Context Protocol）。


<a id="mcp-basics" data-alt="什么是 MCP 模型上下文协议 定义 基本概念"></a>
## 什么是MCP（Model Context Protocol）

MCP（Model Context Protocol）是一套为“模型可调用工具”设计的开放标准，统一“工具发现、工具描述、参数传入、结果返回”的流程，降低因厂商差异带来的集成与维护成本。MCP分为MCP 客户端、MCP 服务器、传输层（）。

- MCP 客户端（Client）：运行在 LLM 应用或 IDE/助手侧，负责发现工具、发起调用、处理响应。
- MCP 服务器（Server）：将某一数据源或能力封装为标准化工具，提供工具列表与调用接口。
- 传输层：本地常用行分隔 JSON（每行一条消息）经 `stdin/stdout`；也可使用 HTTP/WebSocket 等。
  - stdin/stdout ：本机进程间通信，轻量、简单，但不适合跨网络或多客户端并发。
  - HTTP/WebSocket ：网络协议，支持远程访问、长连接、并发，但实现稍复杂、需要端口和服务治理。

---

<a id="setup" data-alt="环境准备 依赖 安装 运行"></a>
## 环境准备
- 代码位置：`/Users/mac/Documents/ai/langchain-learn/langchain-learn/08-mcp-server/`
- Python ≥ 3.8 即可运行；当前示例为最小实现，无需额外依赖。
- 可选依赖（未来扩展）：在 `requirements.txt` 中按需添加，如 `langchain`, `mcp`。

运行方式：
- 服务端：`python 08-mcp-server/mcp_server_demo.py`
- 客户端：`python 08-mcp-server/mcp_client_demo.py`

---

<a id="what-is-mcp" data-alt="MCP 原理 JSON-RPC 行分隔"></a>
## MCP 是什么？原理速览
在最小实现中，通信协议采用“行分隔 JSON-RPC”：
- 每一行是一条完整的 JSON 请求/响应
- 请求包含 `id`、`method`、`params` 字段
- 服务端支持两类方法：`list_tools`（列出可用工具）、`call_tool`（调用具体工具）

这一设计的好处：
- 简单可用、跨平台稳定（不依赖 socket，纯 `stdin/stdout`）
- 易于调试与扩展（任何语言均可按此约定实现客户端/服务端）

---

<a id="server" data-alt="服务端 实现 list_tools call_tool TOOLS"></a>
## 最小服务端实现
文件：`mcp_server_demo.py`

核心结构：
```python
TOOLS = [
    {"name": "get_time", "description": "Return current date and time in ISO format", "args_schema": []},
    {"name": "echo", "description": "Echo back the provided message", "args_schema": [{"name": "message", "type": "string", "required": True}]},
    {"name": "add",  "description": "Add two numbers (a + b)", "args_schema": [{"name": "a", "type": "number", "required": True}, {"name": "b", "type": "number", "required": True}]},
]

def handle_list_tools():
    return {"tools": TOOLS}

def handle_call_tool(params):
    name = params.get("name")
    args = params.get("args", {})
    # 根据 name 分派并返回 {"output": ...} 或 {"error": ...}
```

请求处理：
```python
def process_request(line: str):
    req = json.loads(line)
    method = req.get("method")
    if method == "list_tools":
        return {"id": req.get("id"), "result": handle_list_tools()}
    if method == "call_tool":
        return {"id": req.get("id"), "result": handle_call_tool(req.get("params", {}))}
    return {"id": req.get("id"), "error": f"Unknown method: {method}"}
```

主循环：
```python
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    resp = process_request(line.strip())
    if resp is not None:
        print(json.dumps(resp), flush=True)
```

实现要点：
- 工具元数据清晰定义，便于客户端自动发现与 UI 展示
- 参数类型做基础校验，避免 `call_tool` 误用
- 以 `{"ready": true}` 通知客户端服务就绪，提升健壮性
- 捕获 `KeyboardInterrupt` 优雅退出，避免回溯污染输出

---

<a id="client" data-alt="客户端 子进程 启动 list_tools call_tool"></a>
## 最小客户端实现
文件：`mcp_client_demo.py`

子进程启动与握手：
```python
self.proc = subprocess.Popen([sys.executable, SERVER_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
ready_line = self.proc.stdout.readline().strip()
data = json.loads(ready_line)
if not data.get("ready"):
    raise RuntimeError("Server not ready")
```

发送与接收：
```python
def send(self, payload):
    self.proc.stdin.write(json.dumps(payload) + "\n")
    self.proc.stdin.flush()
    line = self.proc.stdout.readline().strip()
    return json.loads(line)
```

封装调用：
```python
def list_tools(self):
    return self.send({"id": int(time.time()), "method": "list_tools", "params": {}})

def call_tool(self, name, args):
    return self.send({"id": int(time.time()), "method": "call_tool", "params": {"name": name, "args": args}})
```

---

<a id="run-demo" data-alt="运行 演示 命令 输出"></a>
## 运行与演示
1) 启动客户端（其会自动启动服务端子进程）：
```bash
python 08-mcp-server/mcp_client_demo.py
```

示例输出（节选）：
```text
===== Minimal MCP-like Client Demo =====
可用工具:
- get_time: Return current date and time in ISO format
- echo: Echo back the provided message
- add: Add two numbers (a + b)

调用 get_time:
{"id": 1699999999, "result": {"output": "2025-10-15T12:34:56.789012"}}

调用 echo(message='Hello MCP'):
{"id": 1699999999, "result": {"output": "Hello MCP"}}

调用 add(a=15, b=23):
{"id": 1699999999, "result": {"output": 38}}
```

2) 若需单独运行服务端并手工交互：
```bash
python 08-mcp-server/mcp_server_demo.py
# 然后从另一个终端向其 stdin 写入一行 JSON 请求
```

---

<a id="advanced" data-alt="进阶 扩展 安全 性能 最佳实践"></a>
## 进阶扩展与最佳实践
在 `README.md` 中，我们建议的扩展方向包括：
- 工具丰富化：文件系统（读/写/列表）、HTTP API、数据库查询、搜索/翻译/天气等
- 协议增强：批量调用、异步执行、结果缓存、超时控制
- 安全与权限：白名单、参数过滤、敏感操作确认、审计日志
- 性能与可维护性：连接池、工具热加载、负载均衡、监控与指标

实践建议：
- 为每个工具提供明确的 `args_schema` 并做强校验（类型、必填）
- 对外部副作用工具（文件/网络/DB）务必加入权限控制与审计
- 用 `id` 做请求-响应配对；在客户端统一封装错误处理与重试
- 将“最小实现”逐步替换为规范库（如官方 MCP 客户端/服务端），但先掌握原理再引入框架更有助于稳定落地

---

<a id="qa" data-alt="常见错误 排查 调试"></a>
## 常见错误与快速排查 (Q/A)
- 服务端未就绪：客户端读取的第一行不是 `{"ready": true}`。检查服务端是否正常启动、编码与缓冲设置（`text=True`, `bufsize=1`）。
- JSON 解析失败：确保每次写入是一行完整 JSON。若出现半行、空行，服务端会返回 `Invalid JSON`。
- 未知方法/工具：`method` 或 `name` 错误。使用 `list_tools` 获取工具列表，严格按名称调用。
- 类型错误：`add` 期望数字；传入非数字会返回 `error`。在客户端做输入校验更友好。
- 路径问题：请按仓库实际路径运行（本模块为 `08-mcp-server/`），避免示例注释中的其他目录名混淆。
- 子进程卡死：忘记 `flush` 或 stdout/stderr 阻塞。建议最小化 stderr 输出或异步读取，必要时加超时与心跳。

---

<a id="summary" data-alt="总结 收获 要点"></a>
## 总结
通过本最小 MCP 风格实现，你已掌握：
- 用行分隔 JSON-RPC 进行跨进程工具调用的基本套路
- 服务端的工具注册、参数校验与请求分派
- 客户端的子进程管理、握手与封装方法
这些经验可无缝迁移到更完整的 MCP 生态或 LangChain 工具系统中。建议在本地先练好原理与数据面结构，再逐步引入更强大的库与框架。

---

<a id="glossary" data-alt="术语 别名 检索"></a>
## 术语与别名（便于检索与问法对齐）
- MCP：Model Context Protocol；模型上下文协议；工具协议
- JSON-RPC（行分隔）：Line-delimited JSON；一行一条消息；stdin/stdout RPC
- 工具（Tool）：可调用的函数/服务；具备 `name/description/args_schema` 的能力单元
- 子进程：`subprocess.Popen`；管道通信；握手 `ready`
- 白名单/审计：权限控制；安全日志；敏感操作确认