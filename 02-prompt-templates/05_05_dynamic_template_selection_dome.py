#!/usr/bin/env python3
"""
动态模板选择示例
"""

from __future__ import annotations

from typing import Dict
from langchain_core.prompts import PromptTemplate


def main() -> None:
    """动态模板选择示例"""
    print("=== 动态模板选择示例 ===")

    # 定义不同场景的模板
    templates: Dict[str, PromptTemplate] = {
        "greeting": PromptTemplate.from_template("你好，{name}！欢迎来到{company}。"),
        "support": PromptTemplate.from_template("您好，{name}。请问有什么可以帮助您的？"),
        "sales": PromptTemplate.from_template("尊敬的{name}，感谢您对我们{product}的关注！"),
        "technical": PromptTemplate.from_template("你好{name}，关于{issue}的技术问题，我将为您解答。")
    }

    # 模拟用户场景
    scenarios = [
        {"type": "greeting", "name": "张三", "company": "AI科技公司"},
        {"type": "support", "name": "李四", "product": "智能助手"},
        {"type": "sales", "name": "王五", "product": "云计算服务"},
        {"type": "technical", "name": "赵六", "issue": "系统部署"}
    ]

    for scenario in scenarios:
        template_type = scenario["type"]
        if template_type in templates:
            template = templates[template_type]
            # 移除 type 字段，保留其他参数
            params = {k: v for k, v in scenario.items() if k != "type"}
            result = template.format(**params)
            print(f"场景 [{template_type}]: {result}")

    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()