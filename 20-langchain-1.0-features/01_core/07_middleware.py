"""
LangChain 1.0 - Middleware 系统示例
演示 PIIMiddleware, SummarizationMiddleware 和其他内置 middleware
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    PIIMiddleware,
    TodoListMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware
)
from langchain.agents.middleware import AgentState, ModelRequest, ModelResponse, dynamic_prompt
from langchain.agents.middleware import before_model, after_model, wrap_model_call
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage
from langgraph.runtime import Runtime


# 加载环境变量
load_dotenv()

# ========== 辅助函数 ==========
def get_llm(model_name: str = None, **kwargs):
    """创建并配置语言模型实例"""
    api_key = os.getenv("DEEPSEEK_KEY")
    model = model_name or os.getenv("OPENAI_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    if not api_key:
        raise ValueError("DEEPSEEK_KEY 未设置")
    if not base_url:
        raise ValueError("DEEPSEEK_BASE_URL 未设置")

    params = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": 0.7,
        **kwargs
    }

    return init_chat_model(**params)
    
# 定义工具
@tool
def process_payment(card_number: str, amount: float) -> str:
    """处理支付（包含敏感信息）"""
    return f"支付 ${amount} 已处理，卡号: {card_number[-4:]}"


@tool
def send_email(email: str, message: str) -> str:
    """发送邮件"""
    return f"邮件已发送到 {email}"


@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    return f"用户 {user_id}: 张三, 手机: 13800138000, 身份证: 110101199001011234"


# ========== PIIMiddleware 检测和处理对话中的个人身份信息 ==========

def demo_pii_middleware():
    """演示 PIIMiddleware - 保护敏感信息"""
    print("=" * 60)
    print("PIIMiddleware 示例 - 保护敏感信息")
    print("=" * 60)
    agent_model = get_llm()
    # 创建带 PII 保护的 agent
    agent = create_agent(
        model=agent_model,
        tools=[process_payment, send_email, get_user_info],
        middleware=[
            # 信用卡号码 - 部分遮罩
            PIIMiddleware("credit_card", strategy="mask"),
            # 邮箱 - 完全删除
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            # IP地址 - 哈希处理
            PIIMiddleware("ip", strategy="hash"),
            # 手机号 - 阻止（抛出错误）
            # PIIMiddleware("phone_number", strategy="block"),
        ],
        system_prompt="You are a customer service assistant."
    )

    print("\n✅ Agent 已创建，包含以下 PII 保护:")
    print("  - 信用卡号: 部分遮罩 (****)")
    print("  - 邮箱: 完全删除")
    print("  - IP地址: 哈希处理")
    print("=" * 60)

    # 测试 PII 保护
    print("\n📝 测试: 处理包含敏感信息的请求")
    print("-" * 60)

    response = agent.invoke({
        "messages": [HumanMessage(
            content="请发送邮件到 user@example.com，内容包含我的信用卡 4532-1234-5678-9010"
        )]
    })
    print(f"Agent响应: {response['messages'][-1].content}")
    print("\n💡 注意: 敏感信息已被自动处理")


def demo_pii_custom_detector():
    """演示自定义 PII 检测器"""
    print("\n\n" + "=" * 60)
    print("自定义 PII 检测器示例")
    print("=" * 60)

    import re

    agent_model = get_llm()
    # 自定义检测API密钥
    agent = create_agent(
        model=agent_model,
        tools=[],
        middleware=[
            # 使用正则表达式检测 API Key
            PIIMiddleware(
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",
                strategy="block"  # 检测到就阻止
            ),
            # 使用编译的正则表达式检测中国身份证号
            PIIMiddleware(
                "id_card",
                detector=re.compile(r"\d{17}[\dXx]"),
                strategy="redact"
            ),
        ]
    )

    print("\n✅ 自定义 PII 检测器已配置:")
    print("  - API Key (sk-*): 阻止执行")
    print("  - 身份证号: 完全删除")

    print("""
支持的策略:
- 'block': 检测到敏感信息时抛出错误
- 'redact': 替换为 [REDACTED_TYPE]
- 'mask': 部分遮罩 (如 ****1234)
- 'hash': 哈希处理
    """)

# ========== SummarizationMiddleware 自动摘要长对话 ==========

def demo_summarization_middleware():
    """演示 SummarizationMiddleware - 自动摘要长对话"""
    print("\n\n" + "=" * 60)
    print("SummarizationMiddleware 示例 - 管理对话长度")
    print("=" * 60)

    @tool
    def weather_tool(city: str) -> str:
        """获取天气"""
        return f"{city}: 晴天 25°C"

    @tool
    def calculator_tool(expr: str) -> str:
        """计算器"""
        return f"结果: {eval(expr)}"

    agent_model = get_llm()
    # 创建带摘要功能的 agent
    agent = create_agent(
        model= agent_model,
        tools=[weather_tool, calculator_tool],
        middleware=[
            SummarizationMiddleware(
                model="openai:gpt-4o-mini",  # 用于摘要的模型
                max_tokens_before_summary=400,  # 超过400 tokens触发摘要
                messages_to_keep=5  # 保留最近5条消息
            ),
        ],
        checkpointer=MemorySaver()  # 需要 checkpointer 来保存状态
    )

    print("\n✅ SummarizationMiddleware 配置:")
    print("  - 触发阈值: 400 tokens")
    print("  - 保留消息: 最近 5 条")
    print("  - 摘要模型: gpt-4o-mini")
    print("=" * 60)

    print("\n💡 工作原理:")
    print("""
当对话历史超过指定阈值时:
1. 自动识别需要摘要的旧消息
2. 使用 LLM 生成简洁摘要
3. 保留最近的消息（完整内容）
4. 保持 AI/Tool 消息对的完整性
5. 继续对话，无缝切换
    """)


def demo_summarization_advanced():
    """演示 SummarizationMiddleware 的高级配置"""
    print("\n\n" + "=" * 60)
    print("SummarizationMiddleware 高级配置")
    print("=" * 60)

    @tool
    def dummy_tool() -> str:
        return "OK"

    # 配置1: 基于消息数量和tokens的组合条件
    print("\n📌 配置1: 组合触发条件")

    agent_model = get_llm()

    agent1 = create_agent(
        model=agent_model,
        tools=[dummy_tool],
        middleware=[
            SummarizationMiddleware(
                model=agent_model,
                trigger=[("tokens", 4000), ("messages", 10)],  # AND条件
                keep=("messages", 20),  # 保留最近20条
            ),
        ],
    )
    print("  触发条件: tokens >= 4000 AND messages >= 10")
    print("  保留: 最近 20 条消息")

    # 配置2: 使用分数阈值
    print("\n📌 配置2: 分数阈值触发")
    agent2 = create_agent(
        model=agent_model,
        tools=[dummy_tool],
        middleware=[
            SummarizationMiddleware(
                model=agent_model,
                trigger=("fraction", 0.8),  # 达到80%上限时触发
                keep=("fraction", 0.3),  # 保留30%的内容
            ),
        ],
    )
    print("  触发条件: 达到模型上下文的 80%")
    print("  保留: 30% 的内容")

    print("""
\n可用的触发和保留选项:
- ("tokens", N): 基于token数量
- ("messages", N): 基于消息数量
- ("fraction", 0.X): 基于模型上下文百分比
- 组合条件: [("tokens", N), ("messages", M)]
    """)

# ========== TodoListMiddleware 为复杂的、多步骤任务添加待办事项列表管理功能 ==========

def demo_todolist_middleware():
    """演示 TodoListMiddleware"""
    print("\n\n" + "=" * 60)
    print("TodoListMiddleware 示例 - 任务管理")
    print("=" * 60)

    @tool
    def create_task(title: str, priority: str) -> str:
        """创建任务"""
        return f"任务已创建: {title} (优先级: {priority})"
    agent_model = get_llm()
    agent = create_agent(
        model=agent_model,
        tools=[create_task],
        middleware=[TodoListMiddleware()],
        system_prompt="You are a task management assistant."
    )

    print("\n✅ TodoListMiddleware 已启用")
    print("💡 Agent 可以自动跟踪和管理任务列表")


def demo_combined_middleware():
    """演示组合多个 middleware"""
    print("\n\n" + "=" * 60)
    print("组合多个 Middleware - 生产级配置")
    print("=" * 60)

    @tool
    def process_order(customer_email: str, card: str, amount: float) -> str:
        """处理订单"""
        return f"订单已处理: ${amount}"

    agent_model = get_llm()
    # 生产级 agent 配置
    agent = create_agent(
        model=agent_model,
        tools=[process_order],
        middleware=[
            # 1. 保护敏感信息
            PIIMiddleware("email", strategy="mask"),
            PIIMiddleware("credit_card", strategy="mask"),

            # 2. 管理对话长度
            SummarizationMiddleware(
                model=agent_model,
                max_tokens_before_summary=500,
                messages_to_keep=10
            ),

            # 3. 任务管理
            TodoListMiddleware(),
        ],
        checkpointer=MemorySaver(),
        system_prompt="You are a secure customer service assistant."
    )

    print("\n✅ 生产级 Agent 配置完成")
    print("\n📦 包含的 Middleware:")
    print("  1. PIIMiddleware - 保护邮箱和信用卡")
    print("  2. SummarizationMiddleware - 管理对话长度")
    print("  3. TodoListMiddleware - 任务跟踪")
    print("\n💡 这是一个安全、高效、可扩展的配置")

# ========== HumanInTheLoopMiddleware 在工具调用执行之前，暂停代理执行，以进行人工批准、编辑或拒绝 ==========

def demo_HumanInTheLoop_middleware():

    agent_model = get_llm()
    agent = create_agent(
        model=agent_model,
        tools=[send_email],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # Require approval, editing, or rejection for sending emails
                    "send_email_tool": {
                        "allowed_decisions": ["approve", "edit", "reject"],
                    },
                    # Auto-approve reading emails
                    "read_email_tool": False,
                }
            ),
        ],
    )

# ========== ModelCallLimitMiddleware 限制模型调用次数，以防止无限循环或过高成本 ==========

# 可用装饰器
# 节点式（在特定执行点运行）
# @before_agent - 代理启动前（每次调用一次）
# @before_model - 每次模型调用前
# @after_model - 每次模型响应后
# @after_agent - 代理完成时（每次调用一次）
# 包装式（拦截和控制执行）
# @wrap_model_call - 每次模型调用前后
# @wrap_tool_call - 每次工具调用前后
# 便利装饰器:
# @dynamic_prompt - 生成动态系统提示（相当于修改提示的 @wrap_model_call）

def demo_ModelCallLimit_middleware():

    agent_model = get_llm()
    agent = create_agent(
        model=agent_model,
        tools=[...],
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=10,  # Max 10 calls per thread (across runs)
                run_limit=5,  # Max 5 calls per run (single invocation)
                exit_behavior="end",  # Or "error" to raise exception
            ),
        ],
    )

# ========== 自定义 ==========

def demo_middleware():
    # Node-style: logging before model calls
    @before_model
    def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    # Node-style: validation after model calls
    @after_model(can_jump_to=["end"])
    def validate_output(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if "BLOCKED" in last_message.content:
            return {
                "messages": [AIMessage("I cannot respond to that request.")],
                "jump_to": "end"
            }
        return None

    # Wrap-style: retry logic
    @wrap_model_call
    def retry_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(3):
            try:
                return handler(request)
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"Retry {attempt + 1}/3 after error: {e}")

# Wrap-style: dynamic prompts
    @dynamic_prompt
    def personalized_prompt(request: ModelRequest) -> str:
        user_id = request.runtime.context.get("user_id", "guest")
        return f"You are a helpful assistant for user {user_id}. Be concise and friendly."

    # Use decorators in agent
    agent = create_agent(
        model="gpt-4o",
        middleware=[log_before_model, validate_output, retry_model, personalized_prompt],
        tools=[...],
    )

if __name__ == "__main__":
    demo_pii_middleware()
    demo_pii_custom_detector()
    demo_summarization_middleware()
    demo_summarization_advanced()
    demo_todolist_middleware()
    demo_combined_middleware()
    demo_HumanInTheLoop_middleware()
    print("\n\n" + "=" * 60)
    print("✅ 所有 Middleware 示例演示完成")
    print("=" * 60)
    print("\n📚 参考资源:")
    print("  - Built-in middleware: https://docs.langchain.com/oss/python/langchain/middleware/built-in")
    print("  - Middleware reference: https://reference.langchain.com/python/langchain/middleware/")
