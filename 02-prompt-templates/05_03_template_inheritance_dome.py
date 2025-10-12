#!/usr/bin/env python3
"""
模板继承示例
"""

from __future__ import annotations


from langchain_core.prompts import PromptTemplate

def main() -> None:
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
    - 详细说明功能（例如：{main_features}）
    - 包含使用场景（例如：{use_cases}）
    - 目标用户：{target_audience}
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
    print(intro_prompt.format(product="智能手机", style="科技", content_type="产品介绍", main_features="AI相机、120Hz屏幕、5000mAh电池", use_cases="差旅、摄影、移动办公", target_audience="年轻职场人群"))
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()