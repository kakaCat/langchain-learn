"""
LangChain 1.0 - 防护措施 (Guardrails) - 官方文档完整实现

严格按照 https://docs.langchain.com/oss/python/langchain/guardrails 实现

核心功能:
1. 内容过滤 (Content Filtering)
2. 输入验证 (Input Validation)
3. 输出验证 (Output Validation)
4. PII 保护 (已在 Middleware 中实现)
5. 速率限制 (Rate Limiting)
6. 成本控制 (Cost Control)
7. 自定义守卫规则

参考文档:
- https://python.langchain.com/docs/guides/safety/
- https://python.langchain.com/docs/integrations/llms/
"""

import os
import re
import time
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


# ========== 1. 内容过滤基础 ==========
class ContentFilter:
    """内容过滤器 - 检测和阻止不当内容"""

    def __init__(self):
        # 敏感词列表 (实际应用中应该从配置文件加载)
        self.blocked_keywords = [
            "暴力", "色情", "赌博", "毒品",
            "violence", "illegal", "hack"
        ]

        # 敏感模式 (正则表达式)
        self.blocked_patterns = [
            r'\b\d{16}\b',  # 信用卡号
            r'\b\d{3}-\d{2}-\d{4}\b',  # 社保号
        ]

    def check_input(self, text: str) -> tuple[bool, Optional[str]]:
        """
        检查输入内容

        Returns:
            (is_safe, reason) - 是否安全，如果不安全则返回原因
        """
        # 检查敏感词
        for keyword in self.blocked_keywords:
            if keyword in text.lower():
                return False, f"包含敏感词: {keyword}"

        # 检查敏感模式
        for pattern in self.blocked_patterns:
            if re.search(pattern, text):
                return False, f"包含敏感信息"

        return True, None

    def check_output(self, text: str) -> tuple[bool, Optional[str]]:
        """检查输出内容"""
        return self.check_input(text)  # 使用相同规则


def demo_content_filtering():
    """演示内容过滤"""
    print("=" * 70)
    print("1. 内容过滤 (Content Filtering)")
    print("=" * 70)

    filter = ContentFilter()
    llm = ChatOpenAI(model="gpt-4o-mini")

    test_inputs = [
        "帮我写一段关于Python的代码",  # 安全
        "如何制作暴力内容",  # 不安全
        "我的信用卡号是 1234567890123456",  # 敏感信息
    ]

    for user_input in test_inputs:
        print(f"\n用户输入: {user_input}")

        # 输入检查
        is_safe, reason = filter.check_input(user_input)

        if not is_safe:
            print(f"  ❌ 输入被拦截: {reason}")
            continue

        # 调用 LLM
        response = llm.invoke(user_input)

        # 输出检查
        is_safe, reason = filter.check_output(response.content)

        if not is_safe:
            print(f"  ❌ 输出被拦截: {reason}")
            continue

        print(f"  ✅ 响应: {response.content[:50]}...")


# ========== 2. 输入验证 ==========
class InputValidator:
    """输入验证器 - 验证输入格式和内容"""

    def __init__(self, max_length: int = 1000, min_length: int = 1):
        self.max_length = max_length
        self.min_length = min_length

    def validate(self, text: str) -> tuple[bool, Optional[str]]:
        """验证输入"""
        # 长度检查
        if len(text) < self.min_length:
            return False, f"输入太短 (最少{self.min_length}字符)"

        if len(text) > self.max_length:
            return False, f"输入太长 (最多{self.max_length}字符)"

        # 空白检查
        if not text.strip():
            return False, "输入不能为空"

        # 格式检查
        if text.count('\n') > 50:
            return False, "输入包含过多换行"

        return True, None


def demo_input_validation():
    """演示输入验证"""
    print("\n\n" + "=" * 70)
    print("2. 输入验证 (Input Validation)")
    print("=" * 70)

    validator = InputValidator(max_length=100, min_length=5)

    test_inputs = [
        "",  # 太短
        "你好",  # 太短
        "这是一个正常长度的输入问题",  # 正常
        "x" * 150,  # 太长
    ]

    for user_input in test_inputs:
        print(f"\n输入: {user_input[:30]}...")

        is_valid, reason = validator.validate(user_input)

        if not is_valid:
            print(f"  ❌ 验证失败: {reason}")
        else:
            print(f"  ✅ 验证通过")


# ========== 3. 结构化输出验证 ==========
def demo_output_validation():
    """演示输出验证 - 确保输出符合预期格式"""
    print("\n\n" + "=" * 70)
    print("3. 结构化输出验证")
    print("=" * 70)

    # 定义期望的输出格式
    class UserInfo(BaseModel):
        name: str = Field(description="用户姓名")
        age: int = Field(ge=0, le=150, description="年龄(0-150)")
        email: str = Field(pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(UserInfo)

    print("\n提取用户信息:")

    try:
        # 正常输入
        result = structured_llm.invoke("我叫张三，今年25岁，邮箱是zhangsan@example.com")
        print(f"  ✅ 提取成功:")
        print(f"     姓名: {result.name}")
        print(f"     年龄: {result.age}")
        print(f"     邮箱: {result.email}")

    except ValidationError as e:
        print(f"  ❌ 验证失败: {e}")


# ========== 4. 速率限制 ==========
class RateLimiter:
    """速率限制器 - 防止滥用"""

    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口(秒)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, List[float]] = {}

    def check_rate_limit(self, user_id: str) -> tuple[bool, Optional[str]]:
        """
        检查速率限制

        Returns:
            (is_allowed, reason)
        """
        current_time = time.time()

        # 获取用户的请求历史
        if user_id not in self.requests:
            self.requests[user_id] = []

        # 清理过期的请求记录
        self.requests[user_id] = [
            t for t in self.requests[user_id]
            if current_time - t < self.time_window
        ]

        # 检查是否超过限制
        if len(self.requests[user_id]) >= self.max_requests:
            return False, f"超过速率限制 ({self.max_requests}次/{self.time_window}秒)"

        # 记录本次请求
        self.requests[user_id].append(current_time)
        return True, None


def demo_rate_limiting():
    """演示速率限制"""
    print("\n\n" + "=" * 70)
    print("4. 速率限制 (Rate Limiting)")
    print("=" * 70)

    limiter = RateLimiter(max_requests=3, time_window=10)
    user_id = "user_123"

    print(f"\n速率限制: {limiter.max_requests}次/{limiter.time_window}秒")

    # 模拟多次请求
    for i in range(5):
        is_allowed, reason = limiter.check_rate_limit(user_id)

        if is_allowed:
            print(f"\n请求 {i+1}: ✅ 允许")
        else:
            print(f"\n请求 {i+1}: ❌ 拒绝 - {reason}")

        time.sleep(0.5)


# ========== 5. 成本控制 ==========
class CostController:
    """成本控制器 - 监控和限制API成本"""

    def __init__(self, max_tokens_per_user: int = 10000):
        self.max_tokens_per_user = max_tokens_per_user
        self.token_usage: Dict[str, int] = {}

        # 价格 (每1K tokens, 美元)
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

    def check_budget(self, user_id: str, estimated_tokens: int) -> tuple[bool, Optional[str]]:
        """检查用户预算"""
        current_usage = self.token_usage.get(user_id, 0)

        if current_usage + estimated_tokens > self.max_tokens_per_user:
            remaining = self.max_tokens_per_user - current_usage
            return False, f"Token 配额不足 (剩余: {remaining})"

        return True, None

    def record_usage(self, user_id: str, tokens: int):
        """记录 token 使用量"""
        if user_id not in self.token_usage:
            self.token_usage[user_id] = 0
        self.token_usage[user_id] += tokens

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算成本"""
        if model not in self.pricing:
            return 0.0

        pricing = self.pricing[model]
        cost = (input_tokens / 1000 * pricing["input"] +
                output_tokens / 1000 * pricing["output"])
        return cost


def demo_cost_control():
    """演示成本控制"""
    print("\n\n" + "=" * 70)
    print("5. 成本控制 (Cost Control)")
    print("=" * 70)

    controller = CostController(max_tokens_per_user=1000)
    user_id = "user_456"

    print(f"\n用户配额: {controller.max_tokens_per_user} tokens")

    # 模拟多次请求
    requests = [
        ("请求1", 300),
        ("请求2", 400),
        ("请求3", 500),  # 这个会超出配额
    ]

    for request_name, estimated_tokens in requests:
        print(f"\n{request_name} (预计 {estimated_tokens} tokens):")

        # 检查预算
        is_allowed, reason = controller.check_budget(user_id, estimated_tokens)

        if not is_allowed:
            print(f"  ❌ 拒绝: {reason}")
            continue

        print(f"  ✅ 允许")

        # 记录使用
        controller.record_usage(user_id, estimated_tokens)

        current_usage = controller.token_usage[user_id]
        print(f"  已使用: {current_usage}/{controller.max_tokens_per_user} tokens")

    # 计算成本
    print("\n\n成本计算示例:")
    cost = controller.calculate_cost("gpt-4o-mini", 1000, 500)
    print(f"  gpt-4o-mini (1000 input + 500 output): ${cost:.4f}")


# ========== 6. 自定义守卫规则 ==========
class CustomGuardrail:
    """自定义守卫规则"""

    def __init__(self):
        self.rules = []

    def add_rule(self, name: str, check_func, error_message: str):
        """添加守卫规则"""
        self.rules.append({
            "name": name,
            "check": check_func,
            "error": error_message
        })

    def validate(self, data: Any) -> tuple[bool, List[str]]:
        """执行所有规则检查"""
        errors = []

        for rule in self.rules:
            try:
                if not rule["check"](data):
                    errors.append(f"{rule['name']}: {rule['error']}")
            except Exception as e:
                errors.append(f"{rule['name']}: 检查失败 - {str(e)}")

        return len(errors) == 0, errors


def demo_custom_guardrails():
    """演示自定义守卫规则"""
    print("\n\n" + "=" * 70)
    print("6. 自定义守卫规则")
    print("=" * 70)

    guardrail = CustomGuardrail()

    # 添加规则1: 禁止询问个人隐私
    guardrail.add_rule(
        "no_privacy",
        lambda text: not any(word in text.lower() for word in ["密码", "银行", "账号"]),
        "不能询问个人隐私信息"
    )

    # 添加规则2: 必须是问题格式
    guardrail.add_rule(
        "is_question",
        lambda text: "?" in text or "？" in text or any(w in text for w in ["什么", "如何", "为什么"]),
        "输入必须是问题格式"
    )

    # 添加规则3: 长度限制
    guardrail.add_rule(
        "length_check",
        lambda text: 5 <= len(text) <= 200,
        "输入长度必须在 5-200 字符之间"
    )

    test_inputs = [
        "Python怎么学?",  # 通过
        "告诉我你的密码",  # 违反规则1
        "hello",  # 违反规则2
        "x" * 300,  # 违反规则3
    ]

    for user_input in test_inputs:
        print(f"\n输入: {user_input[:30]}...")

        is_valid, errors = guardrail.validate(user_input)

        if is_valid:
            print("  ✅ 所有规则通过")
        else:
            print("  ❌ 违反以下规则:")
            for error in errors:
                print(f"     - {error}")


# ========== 7. 综合守卫系统 ==========
class ComprehensiveGuardrail:
    """综合守卫系统 - 组合所有守卫措施"""

    def __init__(self):
        self.content_filter = ContentFilter()
        self.input_validator = InputValidator(max_length=500)
        self.rate_limiter = RateLimiter(max_requests=5, time_window=60)
        self.cost_controller = CostController(max_tokens_per_user=5000)

    def check_all(self, user_id: str, user_input: str) -> tuple[bool, List[str]]:
        """执行所有检查"""
        errors = []

        # 1. 速率限制
        is_allowed, reason = self.rate_limiter.check_rate_limit(user_id)
        if not is_allowed:
            errors.append(f"速率限制: {reason}")

        # 2. 成本控制
        estimated_tokens = len(user_input) * 2  # 粗略估计
        is_allowed, reason = self.cost_controller.check_budget(user_id, estimated_tokens)
        if not is_allowed:
            errors.append(f"成本控制: {reason}")

        # 3. 输入验证
        is_valid, reason = self.input_validator.validate(user_input)
        if not is_valid:
            errors.append(f"输入验证: {reason}")

        # 4. 内容过滤
        is_safe, reason = self.content_filter.check_input(user_input)
        if not is_safe:
            errors.append(f"内容过滤: {reason}")

        return len(errors) == 0, errors


def demo_comprehensive_guardrail():
    """演示综合守卫系统"""
    print("\n\n" + "=" * 70)
    print("7. 综合守卫系统")
    print("=" * 70)

    guardrail = ComprehensiveGuardrail()
    llm = ChatOpenAI(model="gpt-4o-mini")

    test_cases = [
        ("user_1", "Python如何处理异常?"),  # 正常
        ("user_2", "暴力内容"),  # 内容过滤拦截
        ("user_3", "x"),  # 输入验证拦截
    ]

    for user_id, user_input in test_cases:
        print(f"\n用户 {user_id}: {user_input}")

        # 执行所有检查
        is_allowed, errors = guardrail.check_all(user_id, user_input)

        if not is_allowed:
            print("  ❌ 请求被拒绝:")
            for error in errors:
                print(f"     - {error}")
            continue

        # 通过所有检查，调用 LLM
        print("  ✅ 通过所有检查")
        response = llm.invoke(user_input)
        print(f"  响应: {response.content[:50]}...")

        # 记录成本
        estimated_tokens = len(user_input) * 2
        guardrail.cost_controller.record_usage(user_id, estimated_tokens)


# ========== 8. 最佳实践 ==========
def best_practices():
    """防护措施最佳实践"""
    print("\n\n" + "=" * 70)
    print("8. 最佳实践")
    print("=" * 70)

    print("""
🛡️ 防护措施分层架构:

Layer 1: 输入层
  ├─ 速率限制 (防止滥用)
  ├─ 输入验证 (格式检查)
  └─ 内容过滤 (敏感词)

Layer 2: 处理层
  ├─ 成本控制 (预算管理)
  ├─ Token 限制
  └─ 超时控制

Layer 3: 输出层
  ├─ 输出验证 (格式检查)
  ├─ 内容过滤 (敏感信息)
  └─ 结构化验证

💡 实施建议:

1. **多层防护**
   ✅ 不要依赖单一防护措施
   ✅ 输入和输出都要检查
   ✅ 组合使用多种守卫

2. **优先级顺序**
   1️⃣ 速率限制 (最快,最便宜)
   2️⃣ 输入验证 (防止无效请求)
   3️⃣ 内容过滤 (安全检查)
   4️⃣ 成本控制 (防止超支)
   5️⃣ 调用 LLM
   6️⃣ 输出验证 (确保质量)

3. **性能优化**
   ✅ 快速失败 (尽早拦截)
   ✅ 缓存检查结果
   ✅ 异步处理

4. **用户体验**
   ✅ 清晰的错误信息
   ✅ 友好的提示
   ✅ 提供替代方案

⚠️ 常见陷阱:

❌ 过度限制 - 影响正常用户
❌ 检查太慢 - 影响性能
❌ 错误信息不清 - 用户困惑
❌ 忘记输出检查 - 泄露敏感信息

📊 监控指标:

- 拦截率: 被拦截请求 / 总请求
- 误拦率: 误拦截 / 总拦截
- 响应时间: 守卫检查耗时
- 成本节省: 拦截的潜在成本

🔗 参考资源:
- Safety Guide: https://python.langchain.com/docs/guides/safety/
- Content Filtering: https://platform.openai.com/docs/guides/moderation
- Rate Limiting: https://python.langchain.com/docs/guides/productionization/
    """)


def main():
    """运行所有示例"""
    demo_content_filtering()
    demo_input_validation()
    demo_output_validation()
    demo_rate_limiting()
    demo_cost_control()
    demo_custom_guardrails()
    demo_comprehensive_guardrail()
    best_practices()

    print("\n\n" + "=" * 70)
    print("✅ Guardrails 所有示例演示完成")
    print("=" * 70)
    print("\n核心要点:")
    print("  ✓ 多层防护 (输入-处理-输出)")
    print("  ✓ 速率限制 (防止滥用)")
    print("  ✓ 成本控制 (预算管理)")
    print("  ✓ 内容过滤 (安全保障)")
    print("  ✓ 自定义规则 (灵活扩展)")


if __name__ == "__main__":
    main()
