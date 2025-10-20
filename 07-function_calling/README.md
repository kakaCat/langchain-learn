# 05 - Tools 能力 Demo

本目录包含 LangChain 工具调用的完整实现，展示了如何创建和使用自定义工具来增强 AI 助手的能力。

## 已实现的学习点

### 1. 基础工具定义与使用
- **文件**: `chatbot_tools_demo.py`
- **内容**: 使用 `@tool` 装饰器定义自定义工具，包括参数类型注解、文档字符串和错误处理
- **示例**: 计算器工具、日期查询工具、天气查询工具、文本翻译工具
- **具体实现**: 
  - `calculator()`: 基本数学运算（加减乘除）
  - `get_current_date()`: 获取当前日期
  - `get_weather()`: 模拟天气查询功能
  - `translate_text()`: 模拟多语言翻译功能

### 2. 工具调用 Agent 构建
- **文件**: `chatbot_tools_demo.py`
- **内容**: 创建能够自动选择和使用工具的 AI Agent
- **示例**: 使用 `create_tool_calling_agent()` 和 `AgentExecutor`
- **具体实现**: 
  - 工具列表管理
  - 提示模板设计
  - Agent 执行器配置

### 3. 交互式工具对话系统
- **文件**: `chatbot_tools_demo.py`
- **内容**: 构建完整的交互式对话系统，支持多轮工具调用
- **示例**: 用户输入解析、工具选择、结果返回
- **具体实现**: 
  - 命令行交互界面
  - 工具调用结果展示
- 错误处理和异常管理

### 4. DDGS（DuckDuckGo Search）外部搜索工具
- **文件**: `02_ddgs_search_tool_demo.py`
- **内容**: 使用 `duckduckgo_search` 的 `DDGS` 实现文本搜索；以 `@tool` 封装为 `web_search` 并集成到 Agent。
- **示例**:
  - 直接搜索模式（不依赖 LLM）：`python 02_ddgs_search_tool_demo.py --mode direct`
  - Agent 工具模式（需 `.env` 配置 OPENAI_API_KEY）：`python 02_ddgs_search_tool_demo.py --mode agent`
- **要点**:
  - 统一返回结构：`title/url/snippet`
  - 参数校验与错误处理（空查询、条数限制等）
  - 提示词引导模型“需要事实检索时优先调用 web_search 工具”

### 5. Tavily 搜索工具（结构化检索与引用）
- **文件**: `03_tavily_search_tool_demo.py`
- **内容**: 使用 `tavily-python` 实现文本检索，并以 `@tool` 封装为 `web_search_tavily` 集成到 Agent；返回结果包含文档片段与来源链接，适合 LLM 总结。
- **示例**:
  - 直接搜索模式（需 `.env` 配置 `TAVILY_API_KEY`，不依赖 OPENAI）：`python 03_tavily_search_tool_demo.py`
  - Agent 工具模式（需 `.env` 配置 `OPENAI_API_KEY` 与 `TAVILY_API_KEY`）：`python 03_tavily_search_tool_demo.py`
- **要点**:
  - 统一返回结构：`title/url/snippet`
  - 自动根据环境变量选择运行模式（有 `OPENAI_API_KEY` 则运行 Agent，无则直接检索）
  - 更适合“检索+引用链接”的总结型任务

## 进阶学习点

### 1. 工具验证与错误处理
- 输入参数验证
- 工具执行异常处理
- 用户友好的错误消息

### 2. 工具组合与依赖管理
- 多个工具的协同工作
- 工具间的数据传递
- 工具执行顺序控制

### 3. 外部服务集成
- 真实 API 集成（天气、翻译等）
- 第三方服务认证
- 数据格式转换

### 4. 性能优化与缓存
- 工具调用缓存机制
- 批量工具调用优化
- 异步工具执行

### 5. 安全与权限控制
- 工具访问权限管理
- 敏感操作确认机制
- 用户身份验证

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 配置环境变量（创建 .env 文件）：
```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_BASE_URL=https://api.openai.com/v1

# Tavily（推荐用于结构化检索与引用）
TAVILY_API_KEY=your_tavily_api_key_here

# 可选：DDGS 搜索无需上述变量，直接运行 --mode direct 即可
```

3. 运行示例：
```bash
python chatbot_tools_demo.py
python 02_ddgs_search_tool_demo.py --mode direct
python 02_ddgs_search_tool_demo.py --mode agent
python 03_tavily_search_tool_demo.py
```

## 学习目标

- 掌握 LangChain 工具定义的基本语法
- 理解工具调用 Agent 的工作原理
- 学会构建支持多工具调用的对话系统
- 了解工具验证、错误处理和性能优化的最佳实践

## 进阶思考

1. 如何设计工具使其能够处理更复杂的业务逻辑？
2. 在什么场景下需要工具间的依赖关系？如何管理这些依赖？
3. 如何确保工具调用的安全性和可靠性？
4. 如何优化工具调用的性能，特别是在高并发场景下？
5. 如何设计工具的可扩展性，支持动态添加和移除工具？