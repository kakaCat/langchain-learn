#!/usr/bin/env python3

from __future__ import annotations


from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def main() -> None:
    """聊天模板示例：广告文案生成"""
    print("=== 聊天模板示例：广告文案生成 ===")

    # 系统消息：定义角色和任务
    system_template = "你是一位专业的广告文案专家，擅长创作各种风格的广告文案。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # 用户消息：提供具体需求
    human_template = "请写一篇关于{product}的{style}风格的广告文案。"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 创建聊天提示词模板
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    
    # 格式化提示词
    final_messages = chat_prompt.format_prompt(
        product="智能手机", 
        style="科幻"
    ).to_messages()
    
    print("生成的聊天消息：")
    for message in final_messages:
        print(f"{message.type}: {message.content}")
    print()

    # 演示不同产品的广告文案生成
    products_styles = [
        ("智能手表", "科技"),
        ("电动汽车", "环保"),
        ("智能家居系统", "温馨")
    ]
    
    print("=== 不同产品的广告文案生成示例 ===")
    for product, style in products_styles:
        messages = chat_prompt.format_prompt(product=product, style=style).to_messages()
        print(f"\n产品：{product}，风格：{style}")
        print(f"用户消息：{messages[1].content}")
        print("-" * 50)

if __name__ == "__main__":
    main()