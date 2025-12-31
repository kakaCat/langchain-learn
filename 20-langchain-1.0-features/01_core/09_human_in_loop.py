"""
LangChain 1.0 - Human-in-the-Loop (人机协作) 示例
演示 HumanInTheLoopMiddleware 和暂停/恢复功能
"""

import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()


# 定义需要人工审核的工具
@tool
def transfer_money(from_account: str, to_account: str, amount: float) -> str:
    """转账操作（需要人工确认）"""
    return f"已转账 ${amount} 从 {from_account} 到 {to_account}"


@tool
def delete_user(user_id: str) -> str:
    """删除用户（危险操作，需要确认）"""
    return f"用户 {user_id} 已被删除"


@tool
def send_notification(user_id: str, message: str) -> str:
    """发送通知（安全操作）"""
    return f"通知已发送给用户 {user_id}"


@tool
def查询余额(account: str) -> str:
    """查询账户余额（只读操作）"""
    return f"账户 {account} 余额: $10,000"


def demo_basic_human_in_loop():
    """基础的人机协作示例"""
    print("=" * 60)
    print("Human-in-the-Loop 基础示例")
    print("=" * 60)

    # 创建带人工审核的 agent
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[transfer_money, delete_user, send_notification, query_balance],
        middleware=[
            HumanInTheLoopMiddleware(
                # 指定哪些工具需要人工确认
                interrupt_on={
                    "transfer_money": True,  # 转账需要确认
                    "delete_user": True,     # 删除用户需要确认
                }
            )
        ],
        checkpointer=MemorySaver(),  # 必需，用于保存状态
        system_prompt="You are a banking assistant."
    )

    print("\n✅ Agent 已创建，包含人工审核:")
    print("  - transfer_money: 需要确认")
    print("  - delete_user: 需要确认")
    print("  - send_notification: 自动执行")
    print("  - query_balance: 自动执行")
    print("=" * 60)

    # 配置会话
    config = {"configurable": {"thread_id": "demo_thread_1"}}

    # 示例 1: 安全操作（无需确认）
    print("\n📝 示例 1: 查询余额（无需确认）")
    print("-" * 60)
    response1 = agent.invoke({
        "messages": [HumanMessage(content="查询账户 ACC001 的余额")]
    }, config=config)
    print(f"结果: {response1['messages'][-1].content}")

    # 示例 2: 危险操作（需要确认）
    print("\n📝 示例 2: 转账操作（需要人工确认）")
    print("-" * 60)
    print("请求: 从 ACC001 转账 $5000 到 ACC002")

    try:
        response2 = agent.invoke({
            "messages": [HumanMessage(content="从 ACC001 转账 $5000 到 ACC002")]
        }, config=config)

        # 检查是否被中断
        if "__interrupt__" in response2:
            print("\n⏸️  操作已暂停，等待人工审核...")
            print(f"待确认的工具: {response2['__interrupt__']}")

            # 模拟人工审核
            print("\n👤 人工审核中...")
            print("  选项: [批准/拒绝]")

            # 假设批准
            print("  决定: 批准 ✅")

            # 恢复执行
            response2_continued = agent.invoke(None, config=config)
            print(f"\n✅ 操作已完成: {response2_continued['messages'][-1].content}")
        else:
            print(f"结果: {response2['messages'][-1].content}")

    except Exception as e:
        print(f"错误: {e}")


def demo_interrupt_workflow():
    """演示完整的中断和恢复工作流"""
    print("\n\n" + "=" * 60)
    print("中断和恢复工作流")
    print("=" * 60)

    @tool
    def approve_expense(employee: str, amount: float, reason: str) -> str:
        """批准费用报销"""
        return f"已批准 {employee} 的 ${amount} 费用报销（{reason}）"

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[approve_expense],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"approve_expense": True}
            )
        ],
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],  # 在执行工具前暂停
    )

    config = {"configurable": {"thread_id": "expense_thread"}}

    print("\n💡 工作流程:")
    print("  1. Agent 决定调用工具")
    print("  2. 系统暂停，等待人工确认")
    print("  3. 人工审核并决定")
    print("  4. 系统恢复执行")

    # 发起费用报销请求
    print("\n📝 请求: 批准张三的$500差旅费")
    print("-" * 60)

    response = agent.invoke({
        "messages": [HumanMessage("批准张三的$500差旅费用")]
    }, config=config)

    if "__interrupt__" in response:
        print("\n⏸️  系统暂停")
        print(f"   待审核: {response['__interrupt__']}")
        print("\n👤 经理审核:")
        print("   员工: 张三")
        print("   金额: $500")
        print("   原因: 差旅费")
        print("   决定: [批准] ✅")

        # 继续执行
        response = agent.invoke(None, config=config)
        print(f"\n✅ {response['messages'][-1].content}")


def demo_conditional_interrupts():
    """演示条件性中断"""
    print("\n\n" + "=" * 60)
    print("条件性中断 - 基于金额的审批流程")
    print("=" * 60)

    @tool
    def process_refund(order_id: str, amount: float) -> str:
        """处理退款"""
        return f"订单 {order_id} 的 ${amount} 退款已处理"

    class ConditionalApprovalMiddleware:
        """自定义条件审批 middleware"""
        def __init__(self, threshold: float = 1000.0):
            self.threshold = threshold

        def __call__(self, state, tool_name, tool_input):
            # 如果金额超过阈值，需要审批
            if tool_name == "process_refund":
                amount = tool_input.get("amount", 0)
                if amount >= self.threshold:
                    return {"interrupt": True, "reason": f"金额 ${amount} 超过阈值"}
            return {"interrupt": False}

    print(f"\n✅ 配置: 退款金额 >= $1000 需要审批")
    print("\n示例场景:")
    print("  - $500 退款: 自动处理 ✅")
    print("  - $1500 退款: 需要审批 ⏸️")


def demo_multi_step_approval():
    """演示多步审批流程"""
    print("\n\n" + "=" * 60)
    print("多步审批流程")
    print("=" * 60)

    print("""
真实场景: 大额采购审批

流程:
1. 员工提交采购申请 → 自动记录
2. 部门经理审批 → 需要人工确认 ⏸️
3. 财务审核 → 需要人工确认 ⏸️
4. 执行采购 → 自动执行
5. 通知相关方 → 自动执行

实现方式:
- 使用多个 interrupt_before/interrupt_after
- 配置不同角色的审批节点
- 每个节点可以独立暂停和恢复
    """)

    @tool
    def submit_purchase(item: str, amount: float) -> str:
        return f"采购申请已提交: {item} (${amount})"

    @tool
    def manager_approve(request_id: str) -> str:
        return f"经理已批准申请 {request_id}"

    @tool
    def finance_review(request_id: str) -> str:
        return f"财务审核通过 {request_id}"

    @tool
    def execute_purchase(request_id: str) -> str:
        return f"采购已执行 {request_id}"

    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=[submit_purchase, manager_approve, finance_review, execute_purchase],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "manager_approve": True,
                    "finance_review": True,
                }
            )
        ],
        checkpointer=MemorySaver(),
        system_prompt="You are a procurement assistant."
    )

    print("\n✅ 多步审批 Agent 已配置")


def demo_rejection_handling():
    """演示拒绝处理"""
    print("\n\n" + "=" * 60)
    print("拒绝处理")
    print("=" * 60)

    print("""
当人工审核拒绝操作时:

1. 捕获拒绝信号
2. 通知 Agent
3. Agent 可以:
   - 尝试其他方案
   - 请求更多信息
   - 终止流程

示例代码:

response = agent.invoke(request, config)

if "__interrupt__" in response:
    # 等待人工决策
    decision = get_human_decision()

    if decision == "reject":
        # 发送拒绝消息给 agent
        response = agent.invoke({
            "messages": [HumanMessage("请求被拒绝，请尝试其他方案")]
        }, config)
    else:
        # 继续执行
        response = agent.invoke(None, config)
    """)


def demo_best_practices():
    """人机协作最佳实践"""
    print("\n\n" + "=" * 60)
    print("Human-in-the-Loop 最佳实践")
    print("=" * 60)

    print("""
🎯 何时使用人机协作:

✅ 应该使用:
- 金融交易 (转账、支付)
- 数据删除 (用户、订单)
- 敏感操作 (权限变更)
- 大额支出审批
- 内容发布审核

❌ 不应该使用:
- 只读查询
- 低风险操作
- 频繁的小操作
- 完全自动化的流程

💡 实现技巧:

1. **明确定义需要审核的操作**
   interrupt_on={"critical_tool": True}

2. **提供足够的上下文信息**
   包含操作详情、影响范围、风险评估

3. **支持批量审批**
   对于重复的低风险操作

4. **设置超时机制**
   自动拒绝长时间未响应的请求

5. **记录审批历史**
   用于审计和分析

6. **分级审批**
   根据风险级别设置不同审批者

📊 性能考虑:

- 异步处理: 不阻塞其他操作
- 通知机制: 及时提醒审批者
- 超时处理: 避免无限等待
- 缓存决策: 相似请求快速响应

🔒 安全考虑:

- 身份验证: 确认审批者身份
- 权限检查: 验证审批权限
- 操作日志: 记录所有审批决策
- 双因素认证: 高风险操作
    """)


if __name__ == "__main__":
    demo_basic_human_in_loop()
    demo_interrupt_workflow()
    demo_conditional_interrupts()
    demo_multi_step_approval()
    demo_rejection_handling()
    demo_best_practices()

    print("\n\n" + "=" * 60)
    print("✅ Human-in-the-Loop 示例演示完成")
    print("=" * 60)
    print("\n📚 参考资源:")
    print("  - LangChain 1.0 Blog: https://blog.langchain.com/langchain-langgraph-1dot0/")
    print("  - Middleware Guide: https://docs.langchain.com/oss/python/langchain/middleware/")
