# 快速开始指南 (Quick Start Guide)

## 🚀 5分钟上手

### 方式一：使用 OpenAI（推荐）

```bash
# 1. 设置 API Key
export OPENAI_API_KEY="sk-your-api-key-here"

# 2. 运行演示
cd 09-langgraph-workflow
python 05_clarification_system.py --demo

# 3. 交互模式
python 05_clarification_system.py
```

### 方式二：使用本地 Ollama（免费）

```bash
# 1. 安装 Ollama（如果还没安装）
curl -fsSL https://ollama.com/install.sh | sh

# 2. 启动 Ollama 服务
ollama serve

# 3. 下载模型（在新终端）
ollama pull qwen2.5:latest

# 4. 运行演示
cd 09-langgraph-workflow
export USE_OLLAMA=true
python 05_clarification_system.py --demo

# 5. 交互模式
USE_OLLAMA=true python 05_clarification_system.py
```

## 📦 依赖安装

```bash
# 核心依赖（必需）
pip install langgraph langchain-core

# OpenAI 支持
pip install langchain-openai

# Ollama 支持
pip install langchain-ollama
```

## 💡 运行模式

### 演示模式（自动化）
```bash
python 05_clarification_system.py --demo
```
- 自动运行 3 个预设场景
- 自动模拟用户回答
- 展示完整的澄清流程

### 交互模式（真实对话）
```bash
python 05_clarification_system.py
```
- 输入真实的问题
- 系统生成澄清问题
- 您回答问题
- 多轮对话直到需求明确

## 🎯 预设演示场景

### 场景 1: 投诉类（极度模糊）
```
输入: "这个东西有问题"
期望: 系统识别为投诉，询问对象和问题现象
```

### 场景 2: 操作类（相对明确）
```
输入: "我需要在用户管理页面添加批量导出功能"
期望: 系统识别为操作，询问格式和字段
```

### 场景 3: 需求类（完全模糊）
```
输入: "能不能弄一下"
期望: 系统识别为需求，询问具体需求内容
```

## 🔍 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 无 |
| `USE_OLLAMA` | 是否使用本地 Ollama | `false` |
| `OLLAMA_HOST` | Ollama 服务地址 | `http://localhost:11434` |

## ⚙️ 配置选项

### 切换 LLM 模型

编辑 `05_clarification_system.py` 第 48-62 行：

```python
# 使用不同的 OpenAI 模型
llm = ChatOpenAI(
    model="gpt-4",  # 或 "gpt-3.5-turbo", "gpt-4-turbo"
    temperature=0
)

# 使用不同的 Ollama 模型
llm = ChatOllama(
    model="llama2",  # 或 "mistral", "qwen2.5:latest"
    temperature=0
)
```

### 调整澄清策略

编辑 `05_clarification_system.py` 第 416-432 行：

```python
# 修改完整度阈值（更严格或更宽松）
if semantic_info.completeness >= 0.7 and semantic_info.confidence >= 0.6:
    state.is_clear = True  # 调整这里的阈值

# 限制澄清轮次
if state.round_count >= 3:  # 调整最大轮次
    print("⚠️ 已达到最大澄清轮次")
```

## 📊 输出说明

### 语义分析输出
```
🔍 语义分析: '这个东西有问题'
   意图识别: 投诉              ← 6种类型之一
   实体提取: 无                ← 提取到的实体
   缺失实体: {'对象', '问题现象'}  ← 缺少的关键信息
   完整度: 0.00                ← 0.0-1.0
   置信度: 0.35                ← 0.0-1.0
   可执行: False               ← 是否足够明确
```

### 澄清问题输出
```
💭 生成澄清问题（第 1 轮）...
📝 已生成 4 个澄清问题
   1. 能否详细描述问题的具体现象？
   2. 请问是哪个功能或模块出现了问题？
   3. 问题是什么时候开始出现的？
   4. 能否提供更多背景信息？
```

### 最终结果输出
```
【语义分析结果】
意图类型：投诉
识别实体：{'对象': '登录功能', '时间': '今天', '平台': '移动端'}
语义完整度：0.80
识别置信度：0.75
是否可执行：是
```

## 🐛 常见问题

### Q1: OpenAI API 调用失败
```
Error: OpenAI API key not found
```
**解决**：确保设置了 `OPENAI_API_KEY` 环境变量

### Q2: Ollama 连接失败
```
Error: Could not connect to Ollama
```
**解决**：
1. 检查 Ollama 服务是否启动：`ollama serve`
2. 检查模型是否下载：`ollama list`
3. 如果没有，下载模型：`ollama pull qwen2.5:latest`

### Q3: JSON 解析错误
```
⚠️ 实体提取JSON解析失败
```
**解决**：这通常是 LLM 输出格式问题，代码会自动返回空实体。如果频繁出现，可以优化 prompt。

### Q4: 响应速度慢
**解决**：
- OpenAI：使用 `gpt-3.5-turbo` 而非 `gpt-4`
- Ollama：选择更小的模型如 `qwen2.5:7b`

## 📚 进阶使用

### 集成到您的项目

```python
from langgraph.graph import StateGraph
from typing import TypedDict

# 导入系统
from clarification_system import ClarificationState, build_clarification_graph

# 创建工作流
graph = build_clarification_graph()
app = graph.compile()

# 使用
initial_state = ClarificationState(
    user_input="您的用户输入",
    # ... 其他初始状态
)

# 运行
for event in app.stream(initial_state):
    # 处理事件
    pass
```

### 自定义意图类型

编辑 `IntentType` 枚举（第 23-30 行）：

```python
class IntentType(str, Enum):
    QUERY = "查询"
    OPERATION = "操作"
    COMPLAINT = "投诉"
    REQUIREMENT = "需求"
    HELP = "求助"
    YOUR_NEW_TYPE = "新类型"  # 添加新类型
    UNKNOWN = "未知"
```

同时更新意图识别的 system prompt（第 117-147 行）。

### 自定义实体类型

编辑实体提取的 system prompt（第 164-178 行）：

```python
需要提取的实体类型：
1. 对象 - 用户提到的功能、模块、系统名称
2. 时间 - 时间相关信息
3. 平台 - 平台或设备类型
4. 您的新实体类型 - 描述
...
```

## 🎓 推荐学习路径

1. **基础使用**：运行演示模式，观察系统行为
2. **交互测试**：使用交互模式，输入各种问题
3. **代码阅读**：理解工作流节点和路由逻辑
4. **Prompt 优化**：调整 system prompt 提升效果
5. **功能扩展**：添加新的意图类型和实体类型
6. **项目集成**：将系统集成到您的应用中

## 📖 文档导航

- **[TRANSFORMATION_SUMMARY.md](./TRANSFORMATION_SUMMARY.md)** - 完整的转型总结
- **[05_clarification_system_PURE_LLM.md](./05_clarification_system_PURE_LLM.md)** - 纯 LLM 版本详细文档
- **[05_clarification_system_README.md](./05_clarification_system_README.md)** - 功能说明
- **[QUICK_START.md](./QUICK_START.md)** - 本文档

## ✅ 检查清单

开始使用前，请确认：

- [ ] Python 3.8+ 已安装
- [ ] 依赖包已安装 (`langgraph`, `langchain-core`, `langchain-openai` 或 `langchain-ollama`)
- [ ] 已配置 OpenAI API Key 或已安装 Ollama
- [ ] 已下载必要的模型（如使用 Ollama）
- [ ] 已进入项目目录 `09-langgraph-workflow/`

## 🎉 开始体验

准备好了吗？运行这条命令开始：

```bash
# OpenAI 用户
export OPENAI_API_KEY="sk-..." && python 05_clarification_system.py --demo

# Ollama 用户
export USE_OLLAMA=true && python 05_clarification_system.py --demo
```

享受智能语义识别和澄清的乐趣吧！🚀

---

*需要帮助？查看 [05_clarification_system_PURE_LLM.md](./05_clarification_system_PURE_LLM.md) 获取更多信息。*
