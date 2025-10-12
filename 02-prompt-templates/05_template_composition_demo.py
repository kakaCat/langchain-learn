#!/usr/bin/env python3
"""
模板组合与嵌套示例：多个模板的组合使用、模板继承和扩展、动态模板选择
"""

from __future__ import annotations


from typing import Dict

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


def basic_template_composition() -> None:
    """基础模板组合示例：广告文案生成"""
    print("=== 基础模板组合示例：广告文案生成 ===")

    # 创建基础模板
    base_template = PromptTemplate.from_template(
        """你是一个{role}。请为{product}创作{style}风格的广告文案。"""
    )

    # 创建具体任务模板
    task_template = PromptTemplate.from_template(
        """{base_prompt}

        创作要求：
        - {requirement1}
        - {requirement2}
        - {requirement3}
        """
    )

    # 组合模板
    role = "专业广告文案创作师"
    product = "智能手机"
    style = "科幻"
    base_prompt = base_template.format(role=role, product=product, style=style)

    requirements = {
        "requirement1": "突出产品的科技感和未来感",
        "requirement2": "语言富有想象力和感染力",
        "requirement3": "长度控制在50-100字之间"
    }

    final_prompt = task_template.format(
        base_prompt=base_prompt,
        **requirements
    )

    print("组合后的提示词：")
    print(final_prompt)
    print("\n" + "="*50 + "\n")

def chat_template_composition() -> None:
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
    """多步骤模板组合示例：广告文案创作流程"""
    print("=== 多步骤模板组合示例：广告文案创作流程 ===")

    # 步骤1：产品分析
    analysis_template = PromptTemplate.from_template(
        """请分析以下产品特点：
        产品：{product}
        目标风格：{style}

        分析要点：
        - 产品核心卖点
        - 目标用户群体
        - 风格适配要点
        """
    )

    # 步骤2：文案创作
    creation_template = PromptTemplate.from_template(
        """{analysis_result}

        请基于以上分析，创作符合要求的广告文案：
        """
    )

    # 步骤3：优化建议
    optimization_template = PromptTemplate.from_template(
        """{ad_copy}

        请提供优化建议：
        - 语言表达
        - 情感共鸣
        - 营销效果
        """
    )

    # 执行多步骤流程
    product = "智能手机"
    style = "科幻"

    print("步骤1：产品分析")
    analysis_prompt = analysis_template.format(
        product=product,
        style=style
    )
    print(analysis_prompt)
    print("\n" + "-"*30 + "\n")

    print("步骤2：文案创作")
    creation_prompt = creation_template.format(
        analysis_result="[分析结果占位]"
    )
    print(creation_prompt)
    print("\n" + "-"*30 + "\n")

    print("步骤3：优化建议")
    optimization_prompt = optimization_template.format(
        ad_copy="[广告文案占位]"
    )
    print(optimization_prompt)
    print("\n" + "="*50 + "\n")

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
