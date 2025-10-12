#!/usr/bin/env python3
"""
多步骤模板组合示例：广告文案创作流程
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

def maim() -> None:
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


if __name__ == "__main__":
    maim()