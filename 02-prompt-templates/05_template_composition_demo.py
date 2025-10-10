#!/usr/bin/env python3
"""
模板组合与嵌套示例：多个模板的组合使用、模板继承和扩展、动态模板选择
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# 从当前模块目录加载 .env，避免在仓库根运行时找不到配置
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)

def basic_template_composition() -> None:
    """基础模板组合示例"""
    print("=== 基础模板组合示例 ===")
    
    # 创建多个基础模板
    greeting_template = PromptTemplate.from_template("你好，{name}！")
    question_template = PromptTemplate.from_template("今天感觉怎么样？")
    followup_template = PromptTemplate.from_template("有什么我可以帮助你的吗？")
    
    # 组合模板
    combined_prompt = greeting_template.format(name="张三") + "\n" + \
                     question_template.format() + "\n" + \
                     followup_template.format()
    
    print("组合后的提示词：")
    print(combined_prompt)
    print("\n" + "="*50 + "\n")

def chat_template_composition() -> None:
    """聊天模板组合示例"""
    print("=== 聊天模板组合示例 ===")
    
    # 创建系统消息模板
    system_template = "你是一位专业的{role}。请用{style}风格回答问题。"
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    
    # 创建用户消息模板
    user_template = "请解释以下概念：{concept}"
    user_message = HumanMessagePromptTemplate.from_template(user_template)
    
    # 创建 AI 回复模板
    ai_template = "我已经理解了你的问题。让我用{style}风格来解释{concept}："
    ai_message = AIMessagePromptTemplate.from_template(ai_template)
    
    # 组合聊天模板
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message,
        user_message,
        ai_message
    ])
    
    # 格式化模板
    formatted_messages = chat_prompt.format_prompt(
        role="技术专家",
        style="通俗易懂",
        concept="人工智能"
    ).to_messages()
    
    print("组合后的聊天消息：")
    for i, message in enumerate(formatted_messages):
        print(f"{i+1}. {message.type}: {message.content}")
    print("\n" + "="*50 + "\n")

def template_inheritance() -> None:
    """模板继承示例"""
    print("=== 模板继承示例 ===")
    
    # 基础模板
    base_template = """
    请为{product}写一篇{style}风格的{content_type}。
    
    要求：
    - 语言生动有趣
    - 突出产品特点
    - 吸引目标用户
    """
    
    # 扩展模板：广告文案
    ad_template = base_template + """
    
    广告文案特点：
    - 包含行动号召
    - 强调优惠信息
    - 长度控制在200字以内
    """
    
    # 扩展模板：产品介绍
    intro_template = base_template + """
    
    产品介绍特点：
    - 详细说明功能
    - 包含使用场景
    - 长度控制在500字以内
    """
    
    base_prompt = PromptTemplate.from_template(base_template)
    ad_prompt = PromptTemplate.from_template(ad_template)
    intro_prompt = PromptTemplate.from_template(intro_template)
    
    print("基础模板：")
    print(base_prompt.format(product="智能手机", style="科技", content_type="内容"))
    print("\n广告模板：")
    print(ad_prompt.format(product="智能手机", style="科技", content_type="广告文案"))
    print("\n介绍模板：")
    print(intro_prompt.format(product="智能手机", style="科技", content_type="产品介绍"))
    print("\n" + "="*50 + "\n")

def dynamic_template_selection() -> None:
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

def multi_step_conversation() -> None:
    """多步骤对话流程示例"""
    print("=== 多步骤对话流程示例 ===")
    
    # 步骤1：问候和确认需求
    step1_template = """
    系统：你好！我是AI助手。请问您需要什么帮助？
    用户：{user_query}
    系统：我理解您需要{service_type}服务。让我为您提供详细信息。
    """
    
    # 步骤2：提供具体信息
    step2_template = """
    系统：关于{service_type}，我们提供以下服务：
    {service_details}
    
    您对哪项服务感兴趣？
    """
    
    # 步骤3：处理用户选择
    step3_template = """
    系统：您选择了{selected_service}。
    
    服务详情：
    {service_description}
    
    是否需要立即预约？
    """
    
    step1_prompt = PromptTemplate.from_template(step1_template)
    step2_prompt = PromptTemplate.from_template(step2_template)
    step3_prompt = PromptTemplate.from_template(step3_template)
    
    # 模拟对话流程
    user_query = "我想了解云计算服务"
    service_type = "云计算"
    service_details = "- 云服务器租赁\n- 云存储服务\n- 大数据分析\n- 人工智能平台"
    selected_service = "云服务器租赁"
    service_description = "提供弹性可扩展的云服务器，支持多种操作系统和配置选项。"
    
    print("对话流程：")
    print(step1_prompt.format(user_query=user_query, service_type=service_type))
    print(step2_prompt.format(service_type=service_type, service_details=service_details))
    print(step3_prompt.format(selected_service=selected_service, service_description=service_description))

def main() -> None:
    """主函数"""
    print("模板组合与嵌套示例演示\n")
    
    # 基础模板组合
    basic_template_composition()
    
    # 聊天模板组合
    chat_template_composition()
    
    # 模板继承
    template_inheritance()
    
    # 动态模板选择
    dynamic_template_selection()
    
    # 多步骤对话流程
    multi_step_conversation()

if __name__ == "__main__":
    main()