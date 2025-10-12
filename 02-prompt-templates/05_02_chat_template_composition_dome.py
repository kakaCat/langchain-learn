#!/usr/bin/env python3
"""
聊天模板组合示例：广告文案生成
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


def main() -> None:
    """聊天模板组合示例：广告文案生成"""
    print("=== 聊天模板组合示例：广告文案生成 ===")

    # 创建系统消息模板
    system_template = PromptTemplate.from_template(
        "你是一个{role}，擅长{expertise}。"
    )

    # 创建用户消息模板
    user_template = PromptTemplate.from_template(
        "请为{product}创作{style}风格的广告文案。"
    )

    # 组合聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template.format(role="专业广告文案创作师", expertise="各种风格的广告文案创作")),
        ("human", user_template.format(product="智能手机", style="科幻"))
    ])

    print("聊天提示词：")
    print(chat_prompt.format())
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()