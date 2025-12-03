#!/usr/bin/env python3
"""
16 - Claude Code Style Code Analysis Demo

演示 Claude Code 的代码分析能力：
1. Bug 检测 - 发现潜在错误
2. 代码审查 - 质量评估
3. 重构建议 - 改进方案
4. 测试生成 - 自动化测试

这是 Claude Code 的核心差异化能力之一。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def load_environment() -> None:
    load_dotenv(
        dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False
    )


def get_llm(model: Optional[str] = None, temperature: float = 0.2) -> object:
    provider = os.getenv("LLM_PROVIDER", "").lower()
    use_ollama = provider in {"ollama", "local"} or not os.getenv("OPENAI_API_KEY")

    if use_ollama:
        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=2000,
        )


# ============================================================================
# 代码分析工具
# ============================================================================


class CodeAnalyzer:
    """Claude Code 风格的代码分析器"""

    def __init__(self):
        self.llm = get_llm()

    def find_bugs(self, code: str, language: str = "python") -> str:
        """检测代码中的潜在 bug"""
        prompt = f"""你是一个专业的代码审查专家。请分析以下 {language} 代码，找出潜在的 bug 和问题。

代码：
```{language}
{code}
```

请按以下格式输出：

## 🐛 潜在 Bug

1. **[严重程度] Bug 类型**
   - 位置：第 X 行
   - 问题：具体描述
   - 影响：可能导致的后果
   - 修复建议：如何修复

（如果没有发现 bug，说明"✅ 未发现明显 bug"）

## ⚠️ 代码异味

1. **异味类型**
   - 位置：第 X 行
   - 问题：具体描述
   - 建议：改进方案

## 💡 最佳实践建议

（给出 2-3 条改进建议）
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def review_quality(self, code: str, language: str = "python") -> str:
        """代码质量审查"""
        prompt = f"""你是一个代码质量专家。请评估以下 {language} 代码的质量。

代码：
```{language}
{code}
```

请按以下维度评分（1-10分）并给出建议：

## 📊 质量评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 可读性 | X/10 | ... |
| 可维护性 | X/10 | ... |
| 性能 | X/10 | ... |
| 安全性 | X/10 | ... |
| 测试覆盖 | X/10 | ... |

**总体评分**: X/10

## 🎯 关键问题

1. ...
2. ...

## ✅ 优点

1. ...
2. ...

## 📝 改进建议

1. ...
2. ...
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def suggest_refactor(self, code: str, language: str = "python") -> str:
        """重构建议"""
        prompt = f"""你是一个重构专家。请为以下 {language} 代码提供重构建议。

代码：
```{language}
{code}
```

请提供：

## 🔧 重构建议

### 1. [重构类型]

**当前代码问题**：
- ...

**重构方案**：
```{language}
# 重构后的代码
...
```

**收益**：
- ...

### 2. [另一个重构点]

...

## 📋 优先级排序

1. 高优先级：...
2. 中优先级：...
3. 低优先级：...
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def generate_tests(self, code: str, language: str = "python") -> str:
        """生成测试用例"""
        prompt = f"""你是一个测试专家。请为以下 {language} 代码生成测试用例。

代码：
```{language}
{code}
```

请生成：

## 🧪 测试用例

### 单元测试

```{language}
import pytest

def test_normal_case():
    \"\"\"测试正常情况\"\"\"
    # ...

def test_edge_case():
    \"\"\"测试边界情况\"\"\"
    # ...

def test_error_case():
    \"\"\"测试错误处理\"\"\"
    # ...
```

### 测试覆盖说明

- ✅ 正常路径
- ✅ 边界条件
- ✅ 错误处理
- ⚠️ 待补充：...

### 测试数据

```{language}
# 测试数据示例
test_data = [
    # (input, expected_output)
    (..., ...),
]
```
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


# ============================================================================
# 演示示例
# ============================================================================


def demo_bug_detection():
    """演示 Bug 检测"""
    print("\n" + "=" * 80)
    print("示例 1: Bug 检测")
    print("=" * 80)

    # 有问题的代码示例
    buggy_code = '''
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def process_user_input(user_data):
    # 未验证输入
    result = eval(user_data)  # 危险！
    return result

def get_user_by_id(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL 注入风险
    return database.execute(query)
'''

    analyzer = CodeAnalyzer()
    print("\n分析代码...")
    print("\n" + analyzer.find_bugs(buggy_code))


def demo_quality_review():
    """演示代码质量审查"""
    print("\n" + "=" * 80)
    print("示例 2: 代码质量审查")
    print("=" * 80)

    code = '''
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            if item < 100:
                if item % 2 == 0:
                    result.append(item * 2)
                else:
                    result.append(item * 3)
    return result
'''

    analyzer = CodeAnalyzer()
    print("\n评估代码质量...")
    print("\n" + analyzer.review_quality(code))


def demo_refactoring():
    """演示重构建议"""
    print("\n" + "=" * 80)
    print("示例 3: 重构建议")
    print("=" * 80)

    code = '''
class UserManager:
    def __init__(self):
        self.users = []

    def add_user(self, name, email, age, city, country):
        user = {
            'name': name,
            'email': email,
            'age': age,
            'city': city,
            'country': country
        }
        self.users.append(user)

    def get_user(self, email):
        for user in self.users:
            if user['email'] == email:
                return user
        return None
'''

    analyzer = CodeAnalyzer()
    print("\n生成重构建议...")
    print("\n" + analyzer.suggest_refactor(code))


def demo_test_generation():
    """演示测试生成"""
    print("\n" + "=" * 80)
    print("示例 4: 测试生成")
    print("=" * 80)

    code = '''
def validate_email(email: str) -> bool:
    """验证邮箱格式"""
    if not email or '@' not in email:
        return False

    parts = email.split('@')
    if len(parts) != 2:
        return False

    username, domain = parts
    if not username or not domain:
        return False

    if '.' not in domain:
        return False

    return True
'''

    analyzer = CodeAnalyzer()
    print("\n生成测试用例...")
    print("\n" + analyzer.generate_tests(code))


# ============================================================================
# 主入口
# ============================================================================


def main():
    """运行所有演示"""
    load_environment()

    print("\n" + "=" * 80)
    print("Claude Code Style - 代码分析演示")
    print("=" * 80)
    print("\n这个演示展示了 Claude Code 的代码分析能力：")
    print("  - Bug 检测：发现潜在错误")
    print("  - 质量审查：评估代码质量")
    print("  - 重构建议：提供改进方案")
    print("  - 测试生成：自动化测试用例\n")

    demos = [
        ("Bug 检测", demo_bug_detection),
        ("代码质量审查", demo_quality_review),
        ("重构建议", demo_refactoring),
        ("测试生成", demo_test_generation),
    ]

    print("选择要运行的示例：")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. 运行所有示例")
    print("  0. 退出")

    try:
        choice = input("\n请输入选择 (0-5): ").strip()

        if choice == "0":
            print("退出演示")
            return
        elif choice == str(len(demos) + 1):
            for name, demo_func in demos:
                demo_func()
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            demos[int(choice) - 1][1]()
        else:
            print("无效选择")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n执行出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
