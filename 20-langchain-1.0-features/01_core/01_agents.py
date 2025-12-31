"""
LangChain 1.0 - Agents (代理)

官方文档: https://docs.langchain.com/oss/python/langchain/agents

核心概念:
- create_agent() - LangChain 1.0 的统一 Agent 创建 API
- 基于 LangGraph 构建，提供更强大的功能
- 支持中间件 (Middleware) 系统
- 支持动态模型选择
- 支持工具错误处理
- 支持动态系统提示词
- 支持结构化输出
- 支持自定义状态管理

本文件包含 10 个完整示例:
1. 基础 Agent 创建
2. 动态模型选择 (Dynamic Model Selection)
3. 工具调用与错误处理 (Tool Calling & Error Handling)
4. Prompt Caching (提示词缓存)
5. 动态系统提示词 (Dynamic System Prompt)
6. 结构化输出 (Structured Output)
7. 自定义中间件 (Custom Middleware)
8. 自定义状态管理 (Custom State)
9. 完整对话流 (Full Conversation Flow)
10. 最佳实践总结

官方文档:
- https://docs.langchain.com/oss/python/langchain/agents
- https://python.langchain.com/docs/how_to/agent_structured_output/
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Any, List
from pydantic import BaseModel, Field

# 注意: LangChain 1.0 的 create_agent 可能还在开发中
# 本文件展示官方文档中的 API 设计和使用方式
# 实际运行时可能需要安装最新版本或使用 LangGraph

load_dotenv()


# ========== 辅助函数 ==========
def get_llm():
    """创建并配置语言模型实例 (DeepSeek)"""
    from langchain.chat_models import init_chat_model

    api_key = os.getenv("DEEPSEEK_KEY")
    model = os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise ValueError("DEEPSEEK_KEY 未设置")
    if not base_url:
        raise ValueError("DEEPSEEK_BASE_URL 未设置")

    kwargs = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": 512,
        "timeout": 120,
        "base_url": base_url
    }

    return init_chat_model(**kwargs)


# ========== 示例 1: 基础 Agent 创建 ==========
def demo_basic_agent():
    """
    基础 Agent 创建

    官方文档: https://docs.langchain.com/oss/python/langchain/agents

    核心概念:
    - create_agent() 是 LangChain 1.0 的统一 API
    - 基于 LangGraph 构建
    - 自动处理工具调用和对话循环
    """
    print("=" * 60)
    print("示例 1: 基础 Agent 创建")
    print("=" * 60)

    print("""
⚠️ 注意: LangChain 1.0 的 create_agent API 可能还在开发中

官方文档展示的代码:
```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
)

agent = create_agent(model, tools=tools)
```

核心特性:
- ✅ 统一的 API (不再需要 initialize_agent)
- ✅ 基于 LangGraph 构建 (更强大的状态管理)
- ✅ 自动工具调用循环
- ✅ 支持中间件系统
- ✅ 支持流式输出
- ✅ 支持异步调用

对比旧版 API:
旧版 (LangChain 0.x):
  from langchain.agents import initialize_agent, AgentType
  agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS)

新版 (LangChain 1.0):
  from langchain.agents import create_agent
  agent = create_agent(model, tools=tools)

💡 更简洁、更强大！
    """)


# ========== 示例 2: 动态模型选择 ==========
def demo_dynamic_model_selection():
    """
    动态模型选择 - 根据对话复杂度自动切换模型

    使用场景:
    - 简单对话使用快速模型 (gpt-4o-mini / deepseek-chat)
    - 复杂对话使用强大模型 (gpt-4o / deepseek-reasoner)
    - 节省成本的同时保证质量
    """
    print("\n\n" + "=" * 60)
    print("示例 2: 动态模型选择 (Dynamic Model Selection)")
    print("=" * 60)

    print("""
官方文档示例:
```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    \"\"\"Choose model based on conversation complexity.\"\"\"
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

工作原理:
1. @wrap_model_call 装饰器包装模型调用
2. 根据对话轮数判断复杂度
3. 动态选择合适的模型
4. 自动切换，用户无感知

使用场景:
- 📊 对话开始使用快速模型（节省成本）
- 🧠 对话复杂后切换到强大模型（保证质量）
- 💰 平均成本降低 40-60%

💡 这是 LangChain 1.0 Middleware 系统的核心应用之一！
    """)


# ========== 示例 3: 工具调用与错误处理 ==========
def demo_tool_calling_and_error_handling():
    """
    工具调用与错误处理

    使用场景:
    - 工具调用失败时优雅处理
    - 向模型返回友好的错误信息
    - 避免整个 Agent 崩溃
    """
    print("\n\n" + "=" * 60)
    print("示例 3: 工具调用与错误处理")
    print("=" * 60)

    print("""
官方文档示例:

1️⃣ 定义工具:
```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    \"\"\"Search for information.\"\"\"
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    \"\"\"Get weather information for a location.\"\"\"
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

2️⃣ 错误处理中间件:
```python
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

@wrap_tool_call
def handle_tool_errors(request, handler):
    \"\"\"Handle tool execution errors with custom messages.\"\"\"
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

工作流程:
1. Agent 调用工具
2. 工具执行可能失败 (网络错误、参数错误等)
3. @wrap_tool_call 捕获异常
4. 返回友���的错误消息给模型
5. 模型根据错误信息重新决策

优势:
- ✅ 不会因为单个工具失败而崩溃
- ✅ 模型能够理解错误并重试或调整策略
- ✅ 用户体验更好（不会看到原始异常）

💡 生产环境必备功能！
    """)


# ========== 示例 4: Prompt Caching (提示词缓存) ==========
def demo_prompt_caching():
    """
    Prompt Caching - 缓存大量上下文以节省成本

    使用场景:
    - 处理长文档 (如整本书、大量代码)
    - 重复使用相同的上下文
    - 大幅降低 API 成本 (最高节省 90%)
    """
    print("\n\n" + "=" * 60)
    print("示例 4: Prompt Caching (提示词缓存)")
    print("=" * 60)

    print("""
官方文档示例 (适用于 Anthropic Claude):
```python
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}  # ⭐ 缓存控制
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
)
```

工作原理:
1. 第一次调用: 将整本书内容发送给 API (高成本)
2. 后续调用: 使用缓存的书籍内容 (低成本)
3. cache_control: {"type": "ephemeral"} 标记需要缓存的内容

成本对比:
- 不使用缓存: 每次调用都计费整本书的 tokens
- 使用缓存: 第一次全额计费，后续 5 分钟内缓存命中只收取 10% 费用

适用场景:
- 📚 文档分析 (长篇文章、书籍、论文)
- 💻 代码库分析 (整个代码库的上下文)
- 🗂️ 知识库问答 (大量背景知识)
- 📊 数据分析 (大型数据表)

⚠️ 注意:
- Anthropic Claude 支持最好
- OpenAI GPT 部分支持 (需要使用特定 API)
- DeepSeek 暂不支持

💡 处理大量上下文时的最佳实践！
    """)


# ========== 示例 5: 动态系统提示词 ==========
def demo_dynamic_system_prompt():
    """
    动态系统提示词 - 根据用户角色生成不同的系统提示词

    使用场景:
    - 根据用户身份调整回复风格
    - 专家模式 vs 初学者模式
    - 多租户应用中的角色定制
    """
    print("\n\n" + "=" * 60)
    print("示例 5: 动态系统提示词 (Dynamic System Prompt)")
    print("=" * 60)

    print("""
官方文档示例:
```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    \"\"\"Generate system prompt based on user role.\"\"\"
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}  # ⭐ 传递上下文
)
```

工作原理:
1. 定义 Context TypedDict (类型安全)
2. @dynamic_prompt 装饰器根据 context 生成提示词
3. 运行时根据 user_role 动态调整
4. 不同角色得到不同风格的回复

示例对比:

用户角色: beginner
提示词: "Explain concepts simply and avoid jargon."
问题: "什么是机器学习？"
回答: "机器学习就像教计算机从经验中学习..."

用户角色: expert
提示词: "Provide detailed technical responses."
问题: "什么是机器学习？"
回答: "机器学习是一种基于统计学习理论的方法，通过训练数据构建模型..."

使用场景:
- 👨‍🎓 教育应用 (学生 vs 老师模式)
- 💼 企业应用 (管理者 vs 技术人员)
- 🌍 多语言应用 (根据地区调整)
- 🎯 个性化服务 (根据用户历史调整)

💡 提供个性化用户体验的关键技术！
    """)


# ========== 示例 6: 结构化输出 ==========
def demo_structured_output():
    """
    结构化输出 - 强制 Agent 返回特定格式的数据

    使用场景:
    - 信息提取 (从对话中提取结构化数据)
    - 表单填充
    - 数据验证
    """
    print("\n\n" + "=" * 60)
    print("示例 6: 结构化输出 (Structured Output)")
    print("=" * 60)

    print("""
官方文档示例:
```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)  # ⭐ 结构化输出
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

工作原理:
1. 定义 Pydantic 模型 (ContactInfo)
2. 使用 ToolStrategy 包装模型
3. Agent 自动将输出转换为结构化格式
4. 类型安全 + 数据验证

对比 with_structured_output:

bind_tools() + ToolStrategy:
- ✅ Agent 可以调用工具
- ✅ 最终输出是结构化的
- ✅ 适合复杂工作流

model.with_structured_output():
- ✅ 直接从模型提取
- ✅ 不需要 Agent
- ✅ 适合简单提取任务

使用场景:
- 📋 表单自动填充 (从对话中提取字段)
- 📧 邮件信息提取 (发件人、主题、时间)
- 📊 数据抓取 (从网页提取结构化数据)
- 🤖 聊天机器人 (提取用户意图和参数)

💡 构建可靠 Agent 系统的关键！
    """)


# ========== 示例 7: 自定义中间件 ==========
def demo_custom_middleware():
    """
    自定义中间件 - 扩展 Agent 的行为

    使用场景:
    - 日志记录
    - 性能监控
    - 自定义决策逻辑
    - 状态管理
    """
    print("\n\n" + "=" * 60)
    print("示例 7: 自定义中间件 (Custom Middleware)")
    print("=" * 60)

    print("""
官方文档示例:
```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 在调用模型前执行
        print(f"User preferences: {state.user_preferences}")
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 在模型返回后执行
        return None

    def before_tool(self, state: CustomState, runtime) -> dict[str, Any] | None:
        # 在调用工具前执行
        return None

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

中间件生命周期:
1. before_model: 模型调用前
2. after_model: 模型返回后
3. before_tool: 工具调用前
4. after_tool: 工具返回后

常见中间件类型:
- 📊 日志中间件 (记录所有调用)
- ⏱️ 性能中间件 (测量耗时)
- 🔒 权限中间件 (检查用户权限)
- 💾 缓存中间件 (缓存结果)
- 🎯 路由中间件 (动态选择工具/模型)

💡 LangChain 1.0 最强大的扩展机制！
    """)


# ========== 示例 8: 自定义状态管理 ==========
def demo_custom_state():
    """
    自定义状态管理 - 在 Agent 中管理额外的状态

    使用场景:
    - 用户偏好设置
    - 会话上下文
    - 多轮对话状态
    """
    print("\n\n" + "=" * 60)
    print("示例 8: 自定义状态管理 (Custom State)")
    print("=" * 60)

    print("""
官方文档示例:
```python
from langchain.agents import AgentState

class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState  # ⭐ 自定义状态
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

AgentState 继承:
- messages: List[BaseMessage]  # 默认字段
- user_preferences: dict        # 自定义字段

对比中间件 vs 状态管理:

Middleware (方法 7):
- ✅ 定义行为逻辑
- ✅ 拦截调用
- ✅ 修改请求/响应

State (方法 8):
- ✅ 定义数据结构
- ✅ 持久化状态
- ✅ 跨轮对话共享

使用场景:
- 🎯 用户画像 (偏好、历史行为)
- 💬 对话管理 (话题、情绪、进度)
- 🛒 购物车 (多轮选购流程)
- 📝 表单填写 (分步骤收集信息)

💡 构建有状态 Agent 的基础！
    """)


# ========== 示例 9: 完整对话流 ==========
def demo_full_conversation_flow():
    """
    完整对话流 - 展示 Agent 的完整工作流程

    演示:
    1. 用户输入
    2. Agent 决策 (调用工具 or 直接回复)
    3. 工具执行
    4. 生成最终回复
    """
    print("\n\n" + "=" * 60)
    print("示例 9: 完整对话流 (Full Conversation Flow)")
    print("=" * 60)

    print("""
完整工作流程:

1️⃣ 用户输入
   "今天北京天气怎么样？"
   ↓

2️⃣ Agent 分析
   - 需要天气信息
   - 决定调用 get_weather 工具
   ↓

3️⃣ 工具调用
   get_weather(location="北京")
   返回: "北京: 晴天, 25°C"
   ↓

4️⃣ Agent 生成回复
   "今天北京天气晴朗，气温25°C，适合出行。"
   ↓

5️⃣ 返回用户

实现代码:
```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    \"\"\"Get weather information.\"\"\"
    # 实际应该调用天气 API
    return f"{location}: 晴天, 25°C"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather]
)

# 运行对话
result = agent.invoke({
    "messages": [{"role": "user", "content": "今天北京天气怎么样？"}]
})

print(result["messages"][-1]["content"])
# 输出: "今天北京天气晴朗，气温25°C，适合出行。"
```

内部流程:
1. create_agent 自动构建 LangGraph
2. LangGraph 管理状态机:
   - 调用模型 → 检查是否需要工具
   - 调用工具 → 将结果返回给模型
   - 生成回复 → 结束

与旧版 API 对比:

旧版 (需要手动循环):
```python
while True:
    response = llm(messages)
    if not response.tool_calls:
        break
    for tool_call in response.tool_calls:
        result = execute_tool(tool_call)
        messages.append(result)
```

新版 (自动循环):
```python
result = agent.invoke({"messages": messages})
# 完成！
```

💡 create_agent 自动处理所有复杂逻辑！
    """)


# ========== 示例 10: 最佳实践 ==========
def demo_best_practices():
    """
    最佳实践总结
    """
    print("\n\n" + "=" * 60)
    print("示例 10: 最佳实践总结")
    print("=" * 60)

    print("""
🎯 LangChain 1.0 Agents 最佳实践

1️⃣ API 选择
   ✅ 使用 create_agent (LangChain 1.0)
   ❌ 避免 initialize_agent (已废弃)

2️⃣ 工具设计
   ✅ 工具功能单一、明确
   ✅ 提供清晰的 docstring (模型会读取)
   ✅ 参数使用类型注解
   ❌ 避免工具做太多事情

3️⃣ 错误处理
   ✅ 使用 @wrap_tool_call 处理工具错误
   ✅ 返回友好的错误消息给模型
   ❌ 不要让异常传播到用户

4️⃣ 性能优化
   ✅ 使用动态模型选择 (节省成本)
   ✅ 使用 Prompt Caching (处理大上下文)
   ✅ 使用流式输出 (改善用户体验)
   ❌ 避免在简单任务上使用强大模型

5️⃣ 状态管理
   ✅ 定义 CustomState 管理额外状态
   ✅ 使用中间件扩展行为
   ❌ 不要在工具中存储全局状态

6️⃣ 个性化
   ✅ 使用 @dynamic_prompt 根据用户调整
   ✅ 使用 context 传递用户信息
   ❌ 不要在提示词中硬编码用户信息

7️⃣ 结构化输出
   ✅ 使用 ToolStrategy 提取结构化数据
   ✅ 定义 Pydantic 模型保证类型安全
   ❌ 不要依赖字符串解析

8️⃣ 测试
   ✅ 单独测��每个工具
   ✅ 测试不同的对话路径
   ✅ 测试错误处理逻辑
   ❌ 不要只测试 happy path

9️⃣ 监控
   ✅ 记录所有工具调用
   ✅ 监控成本和性能
   ✅ 追踪失败率
   ❌ 不要在生产环境盲目运行

🔟 文档
   ✅ 为每个工具写清晰的文档
   ✅ 记录常见问题和解决方案
   ✅ 提供示例代码
   ❌ 不要假设用户知道如何使用

---

📚 相关资源:
- 官方文档: https://docs.langchain.com/oss/python/langchain/agents
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
- 工具创建: 参考 04_tools.py
- 模型使用: 参考 02_models.py
    """)


# ========== 主函数 ==========
def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LangChain 1.0 Agents 完整示例")
    print("=" * 60)
    print("\n官方文档: https://docs.langchain.com/oss/python/langchain/agents")
    print("\n本文件包含 10 个示例:")
    print("  1. 基础 Agent 创建")
    print("  2. 动态模型选择 ⭐")
    print("  3. 工具调用与错误处理 ⭐")
    print("  4. Prompt Caching (提示词缓存)")
    print("  5. 动态系统提示词 ⭐")
    print("  6. 结构化输出")
    print("  7. 自定义中间件")
    print("  8. 自定义状态管理")
    print("  9. 完整对话流")
    print(" 10. 最佳实践总结")
    print("=" * 60)

    print("\n⚠️ 注意:")
    print("  LangChain 1.0 的 create_agent API 可能还在开发中")
    print("  本文件展示官方文档中的 API 设计和概念")
    print("  实际运行需要安装最新版本的 LangChain")
    print("\n  参考 02_models.py 查看基于 bind_tools 的实际实现")
    print("=" * 60)

    # 运行所有示例
    demo_basic_agent()
    demo_dynamic_model_selection()
    demo_tool_calling_and_error_handling()
    demo_prompt_caching()
    demo_dynamic_system_prompt()
    demo_structured_output()
    demo_custom_middleware()
    demo_custom_state()
    demo_full_conversation_flow()
    demo_best_practices()

    print("\n\n" + "=" * 60)
    print("✅ 所有 Agent 示例演示完成")
    print("=" * 60)
    print("\n📚 相关文件:")
    print("  - 02_models.py: 模型和工具调用的实际实现")
    print("  - 04_tools.py: 工具的 4 种创建方法")
    print("  - 08_structured_output.py: 结构化输出详解")
    print("=" * 60)


if __name__ == "__main__":
    main()
