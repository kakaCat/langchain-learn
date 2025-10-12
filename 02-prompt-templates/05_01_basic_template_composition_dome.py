#!/usr/bin/env python3
"""
基础模板组合示例：广告文案生成
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


def main() -> None:
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


if __name__ == "__main__":
    main()